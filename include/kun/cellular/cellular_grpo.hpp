#ifndef KUN_CELLULAR_GRPO_HPP_
#define KUN_CELLULAR_GRPO_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string>
#include <random>
#include <iostream>

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"

namespace kun {

/**
 * 通用单步转移结构 (Universal Step Transition for GRPO)
 * 记录观测张量、采样动作、前向效应器 logits 分布与旧策略概率
 */
struct GRPOStepTransition {
    std::vector<float> obs;
    int action{0};
    std::vector<float> logits;
    float action_prob{0.0f};
};

/**
 * 完整轨迹包 (Trajectory Rollout)
 * 记录单条推演轨迹、环境标量奖励、组内相对优势值与终局状态
 */
struct GRPOTrajectory {
    std::vector<GRPOStepTransition> steps;
    float raw_reward{0.0f};
    float advantage{0.0f};
    bool success{false};
    uint32_t group_id{0};
};

/**
 * GRPO 组结构 (Group of Rollouts)
 * 对标 DeepSeek-GRPO: 同一起始环境分布下采样的 G 条异构探索轨迹
 */
struct GRPOGroup {
    uint32_t group_id{0};
    std::vector<GRPOTrajectory> rollouts;
    float mean_reward{0.0f};
    float std_reward{0.0f};
    float mean_advantage{0.0f};
};

/**
 * GRPO 超参数配置 (Hyperparameters)
 */
struct GRPOConfig {
    size_t group_size{4};         // 每组采样轨迹数 G
    float clip_eps{0.2f};         // 剪裁阈值 epsilon
    float entropy_coef{0.02f};     // 熵正则系数 beta
    float kl_coef{0.0f};          // 参考策略 KL 散度约束系数
    float learning_rate{0.002f};  // Adam 学习率
    float grad_clip_norm{1.0f};   // 全局梯度范数上限
    float lyapunov_max_gain{0.95f}; // 李雅普诺夫流形投影最大环增益
};

/**
 * GRPO 训练指标跟踪 (Metrics)
 */
struct GRPOMetrics {
    float total_loss{0.0f};
    float policy_loss{0.0f};
    float entropy{0.0f};
    float mean_advantage{0.0f};
    float grad_norm{0.0f};
    float max_loop_gain{0.0f};
    size_t valid_trajectories{0};
    size_t total_steps{0};
};

/**
 * 核心数学原语：组内相对优势度标准化 (Group Relative Advantage Normalization)
 * 
 * 对标 DeepSeekMath / DeepSeek-R1 创新公式:
 *   A_i = (R_i - mean({R})) / (std({R}) + eps)
 * 无需独立 Critic / Value 神经网络，直接利用多路探索方差消除基线偏差，实现纯原生算力节约与高效收敛。
 */
inline void compute_group_relative_advantages(GRPOGroup& group, float eps = 1e-6f) {
    if (group.rollouts.empty()) return;
    const size_t G = group.rollouts.size();

    float sum_r = 0.0f;
    for (const auto& traj : group.rollouts) {
        sum_r += traj.raw_reward;
    }
    float mean_r = sum_r / static_cast<float>(G);

    float var_r = 0.0f;
    for (const auto& traj : group.rollouts) {
        float diff = traj.raw_reward - mean_r;
        var_r += diff * diff;
    }
    float std_r = std::sqrt(var_r / static_cast<float>(G) + eps);
    group.mean_reward = mean_r;
    group.std_reward = std_r;

    float sum_adv = 0.0f;
    for (auto& traj : group.rollouts) {
        if (std_r < 1e-5f) {
            traj.advantage = 0.0f;
        } else {
            traj.advantage = (traj.raw_reward - mean_r) / std_r;
        }
        sum_adv += traj.advantage;
    }
    group.mean_advantage = sum_adv / static_cast<float>(G);
}

/**
 * 神经形态硅基细胞 GRPO 优化器 (Cellular GRPO Optimizer)
 * 
 * 整合:
 * 1. 组级相对优势标准化 (Zero-Critic Group Relative Advantage)
 * 2. 剪裁比率替代目标与信息熵正则 (Clipped Surrogate Objective + Entropy Regularization)
 * 3. CSR 图 Kahn 拓扑逆序 BPTT 雅可比逆推 (Analytic CSR BPTT)
 * 4. Adam 优化与李雅普诺夫 BIBO 稳定流形硬约束 (Lyapunov Manifold Projection)
 */
class CellularGRPOTrainer {
public:
    CellularBPTTEngine bptt;
    GRPOConfig config;

    explicit CellularGRPOTrainer(const GRPOConfig& cfg = GRPOConfig(), size_t max_window = 64)
        : bptt(max_window), config(cfg) {
        bptt.grad_clip_norm = cfg.grad_clip_norm;
    }

    void init_optimizer(const CellularOrganism& org) {
        bptt.init_optimizer(org);
    }

    /**
     * 对单组探索轨迹执行 GRPO 前向录带与伴随梯度回传，累加至 accum_grads
     */
    void accumulate_group_gradients(
        CellularOrganism& org,
        const GRPOGroup& group,
        BPTTGradients& accum_grads,
        GRPOMetrics& metrics
    ) {
        const size_t num_syn = org.compiled_synapses_.size();
        const size_t num_cells = org.cells.size();
        if (accum_grads.grad_synapses.size() != num_syn) accum_grads.grad_synapses.resize(num_syn, 0.0f);
        if (accum_grads.grad_gains.size() != num_cells) accum_grads.grad_gains.resize(num_cells, 0.0f);

        BPTTGradients traj_grads;
        for (const auto& traj : group.rollouts) {
            if (traj.steps.empty()) continue;

            // 1. 前向推演并录带
            bptt.reset_tape();
            for (const auto& st : traj.steps) {
                double d_in[4] = {0.0, 0.0, 0.0, 0.0};
                for (size_t i = 0; i < std::min(size_t(4), st.obs.size()); ++i) {
                    d_in[i] = static_cast<double>(st.obs[i]);
                }
                org.forward(d_in, false);
                bptt.record_step(org);
            }

            // 2. 构造 GRPO 替代目标监督向量:
            // targets: [0]=chosen_action_idx, [1]=advantage, [2]=old_prob, [3]=clip_eps, [4]=entropy_coef
            std::vector<std::vector<float>> target_outputs;
            target_outputs.reserve(traj.steps.size());
            for (const auto& st : traj.steps) {
                target_outputs.push_back({
                    static_cast<float>(st.action),
                    traj.advantage,
                    st.action_prob,
                    config.clip_eps,
                    config.entropy_coef
                });
            }

            // 3. 执行 BPTT VJP 反向传播
            float loss = bptt.backward_with_loss(
                org, target_outputs, traj_grads, SubstrateLossType::GRPO_SURROGATE
            );

            metrics.total_loss += loss;
            metrics.mean_advantage += traj.advantage;
            metrics.valid_trajectories++;
            metrics.total_steps += traj.steps.size();

            // 4. 累加全图参数梯度
            for (size_t k = 0; k < num_syn; ++k) {
                accum_grads.grad_synapses[k] += traj_grads.grad_synapses[k];
            }
            for (size_t k = 0; k < num_cells; ++k) {
                accum_grads.grad_gains[k] += traj_grads.grad_gains[k];
            }
        }
    }

    /**
     * 归一化平均梯度并执行 Adam 优化步与李雅普诺夫流形硬投影
     */
    void step_optimizer(
        CellularOrganism& org,
        BPTTGradients& accum_grads,
        size_t num_trajectories,
        GRPOMetrics& metrics
    ) {
        if (num_trajectories == 0) return;

        float inv_n = 1.0f / static_cast<float>(num_trajectories);
        double grad_sq = 0.0;
        for (auto& g : accum_grads.grad_synapses) {
            g *= inv_n;
            grad_sq += g * g;
        }
        for (auto& g : accum_grads.grad_gains) {
            g *= inv_n;
            grad_sq += g * g;
        }
        metrics.grad_norm = static_cast<float>(std::sqrt(grad_sq));

        bptt.step_adam(org, accum_grads, config.learning_rate);
        metrics.max_loop_gain = bptt.apply_lyapunov_projection(org, config.lyapunov_max_gain);
    }
};

} // namespace kun

#endif /* KUN_CELLULAR_GRPO_HPP_ */
