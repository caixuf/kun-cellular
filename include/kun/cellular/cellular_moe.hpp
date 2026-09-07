#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 细粒度微柱神经稀疏门控路由机制 (Cellular MoE)
// (Fine-grained Mixture of Cortical Column Experts & Sparse Gating Router)
//
// 体系结构规范:
// 1. 纯数学拓扑门控原语: 输入 x 经由可微分线性路由投影 s_i = W_r x + b_i 计算专家激活 logits。
// 2. 严格零算力旁路 (Zero-Compute Bypass): 仅激活 Top-K 个微柱专家，其余 N-K 个专家处于零能耗休眠态，
//    达到严格的 O(K) 动态计算时延而非 O(N)。
// 3. Top-K Softmax 归一化融合: 在选取的 K 个专家间进行局部概率归一化，形成凸组合决策。
// 4. 负载均衡辅助损失 (Switch / DeepSeek-V3 路由损失):
//    L_balance = alpha * N * \sum_{i=0}^{N-1} f_i * P_i，
//    附带闭式解析梯度反传与参数更新，强力抑制专家坍缩 (Expert Collapse)。
// 5. 绝对铁律: 恪守 Rule 7 底座中立性，严禁任何具身业务特化词汇，纯粹数学与拓扑动力学。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <string>
#include <functional>

#include "kun/cellular/cellular_genome.hpp"

namespace kun {

/**
 * @brief 稀疏门控路由超参数配置
 */
struct MoERouterConfig {
    size_t num_experts{4};               // 专家微柱总数 N (N >= 1)
    size_t top_k{2};                     // 激活专家数 K (1 <= K <= N)
    size_t input_dim{4};                 // 感受态输入维度 D
    double load_balance_alpha{0.01};     // 负载均衡辅助损失权重系数 alpha
    double routing_noise_std{0.0};       // 路由探索噪声标准差 (Noisy Top-K)
    bool normalize_topk{true};           // 是否对 Top-K 权重执行局部 Softmax 归一化
};

/**
 * @brief 单次路由前向决策遥测结构
 */
struct RoutingDecision {
    std::vector<size_t> topk_indices;    // 选中的 Top-K 专家索引（降序排列）
    std::vector<double> topk_weights;    // 对应的归一化门控权重 (和为 1.0)
    std::vector<double> all_logits;      // 全网 N 个专家的原始门控 logits s_i
    std::vector<double> all_probs;       // 全局全量 Softmax 概率分布 P_i
};

/**
 * @brief 路由参数解析梯度容器
 */
struct MoERouterGradients {
    std::vector<double> d_weights;       // 路由权重梯度 dL/dW_r, 大小: num_experts * input_dim
    std::vector<double> d_biases;        // 路由偏置梯度 dL/db,   大小: num_experts
    size_t num_samples{0};               // 批次累积样本数

    void clear() {
        std::fill(d_weights.begin(), d_weights.end(), 0.0);
        std::fill(d_biases.begin(), d_biases.end(), 0.0);
        num_samples = 0;
    }
};

/**
 * @brief 稀疏门控长程动力学生命周期遥测指标
 */
struct MoETelemetry {
    uint64_t total_routings{0};                 // 累计路由决策次数
    std::vector<uint64_t> expert_dispatches;    // 各专家的累计指派被选中次数
    std::vector<double> expert_prob_sums;       // 各专家的累计全局概率和
    double current_balance_loss{0.0};           // 当前窗口/批次的负载均衡损失
    double active_compute_ratio{0.0};           // 算力激活比率 K / N
    uint64_t total_bypassed_evaluations{0};     // 累计旁路节约的专家前向计算次数
};

/**
 * @brief 纯数学微柱稀疏门控路由器 (Cellular MoE Router)
 */
class CellularRouter {
public:
    explicit CellularRouter(const MoERouterConfig& cfg = MoERouterConfig{}, uint32_t seed = 42)
        : config_(cfg), rng_(seed) {
        if (config_.num_experts == 0) config_.num_experts = 1;
        if (config_.top_k == 0) config_.top_k = 1;
        if (config_.top_k > config_.num_experts) config_.top_k = config_.num_experts;
        if (config_.input_dim == 0) config_.input_dim = 1;

        weights_.assign(config_.num_experts * config_.input_dim, 0.0);
        biases_.assign(config_.num_experts, 0.0);

        telemetry_.expert_dispatches.assign(config_.num_experts, 0);
        telemetry_.expert_prob_sums.assign(config_.num_experts, 0.0);
        telemetry_.active_compute_ratio = static_cast<double>(config_.top_k) / config_.num_experts;

        scratch_indices_.resize(config_.num_experts);
        cached_decision_.all_logits.resize(config_.num_experts);
        cached_decision_.all_probs.resize(config_.num_experts);
        cached_decision_.topk_indices.resize(config_.top_k);
        cached_decision_.topk_weights.resize(config_.top_k);

        init_weights(seed);
    }

    void init_weights(uint32_t seed = 42) {
        rng_.seed(seed);
        // 正交/He 尺度初始化: stddev = sqrt(2.0 / (input_dim + num_experts))
        double stddev = std::sqrt(2.0 / static_cast<double>(config_.input_dim + config_.num_experts));
        std::normal_distribution<double> dist(0.0, stddev);
        for (auto& w : weights_) {
            w = dist(rng_);
        }
        std::fill(biases_.begin(), biases_.end(), 0.0);
    }

    const MoERouterConfig& config() const { return config_; }
    const std::vector<double>& weights() const { return weights_; }
    std::vector<double>& weights() { return weights_; }
    const std::vector<double>& biases() const { return biases_; }
    std::vector<double>& biases() { return biases_; }
    const MoETelemetry& telemetry() const { return telemetry_; }
    const RoutingDecision& cached_decision() const { return cached_decision_; }

    void reset_telemetry() {
        telemetry_.total_routings = 0;
        std::fill(telemetry_.expert_dispatches.begin(), telemetry_.expert_dispatches.end(), 0);
        std::fill(telemetry_.expert_prob_sums.begin(), telemetry_.expert_prob_sums.end(), 0.0);
        telemetry_.current_balance_loss = 0.0;
        telemetry_.total_bypassed_evaluations = 0;
    }

    void reset_batch_history() {
        batch_probs_.clear();
        batch_topk_.clear();
        batch_inputs_.clear();
    }

    /**
     * @brief 稀疏门控路由前向推演 (零堆内存分配 hot path)
     * @param inputs 感受态输入向量
     * @param in_dim 输入维度
     * @param record_for_aux_loss 是否在批次缓冲中记录供辅助损失反传
     * @return 路由决策结构体常量引用 (包含 Top-K 索引与权重)
     */
    const RoutingDecision& route(const double* inputs, size_t in_dim, bool record_for_aux_loss = false) {
        const size_t N = config_.num_experts;
        const size_t K = config_.top_k;
        const size_t D = config_.input_dim;
        const size_t actual_dim = (inputs == nullptr) ? 0 : std::min(in_dim, D);

        auto& all_logits = cached_decision_.all_logits;
        auto& all_probs = cached_decision_.all_probs;
        auto& topk_indices = cached_decision_.topk_indices;
        auto& topk_weights = cached_decision_.topk_weights;

        // 1. 线性投影: s_i = W_r x + b_i
        for (size_t i = 0; i < N; ++i) {
            double logit = biases_[i];
            const size_t row_offset = i * D;
            if (inputs != nullptr) {
                for (size_t d = 0; d < actual_dim; ++d) {
                    double val = inputs[d];
                    if (std::isnan(val) || std::isinf(val)) val = 0.0;
                    logit += weights_[row_offset + d] * val;
                }
            }
            // 探索性路由噪声
            if (config_.routing_noise_std > 0.0) {
                std::normal_distribution<double> noise_dist(0.0, config_.routing_noise_std);
                logit += noise_dist(rng_);
            }
            if (std::isnan(logit)) logit = 0.0;
            all_logits[i] = logit;
        }

        // 2. 全局 Softmax 概率分布 P_i (用于无偏负载均衡度量)
        double max_logit = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < N; ++i) {
            if (!std::isnan(all_logits[i]) && all_logits[i] > max_logit) {
                max_logit = all_logits[i];
            }
        }
        if (!std::isfinite(max_logit)) max_logit = 0.0;

        double exp_sum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double exp_val = std::exp(all_logits[i] - max_logit);
            if (std::isnan(exp_val) || std::isinf(exp_val)) exp_val = 0.0;
            all_probs[i] = exp_val;
            exp_sum += exp_val;
        }
        if (exp_sum > 0.0 && !std::isnan(exp_sum)) {
            for (size_t i = 0; i < N; ++i) {
                all_probs[i] /= exp_sum;
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                all_probs[i] = 1.0 / static_cast<double>(N);
            }
        }

        // 3. 确定性 Top-K 专家选取 (严格降序排列，数值相同时按专家 ID 确定性仲裁，并保证严格弱序)
        std::iota(scratch_indices_.begin(), scratch_indices_.end(), 0);
        std::stable_sort(scratch_indices_.begin(), scratch_indices_.end(), [&](size_t a, size_t b) {
            double la = all_logits[a];
            double lb = all_logits[b];
            if (std::isnan(la) && !std::isnan(lb)) return false;
            if (!std::isnan(la) && std::isnan(lb)) return true;
            if (la != lb) {
                return la > lb;
            }
            return a < b; // 确定性打破平局，保证比特级确定性
        });

        for (size_t k = 0; k < K; ++k) {
            topk_indices[k] = scratch_indices_[k];
        }

        // 4. 局部 Top-K Softmax 权重归一化 (Top-K Gating Normalization)
        if (config_.normalize_topk) {
            double topk_max = all_logits[topk_indices[0]];
            if (!std::isfinite(topk_max)) topk_max = 0.0;
            double topk_exp_sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                double e = std::exp(all_logits[topk_indices[k]] - topk_max);
                if (std::isnan(e) || std::isinf(e)) e = 0.0;
                topk_weights[k] = e;
                topk_exp_sum += e;
            }
            if (topk_exp_sum > 0.0 && !std::isnan(topk_exp_sum)) {
                for (size_t k = 0; k < K; ++k) {
                    topk_weights[k] /= topk_exp_sum;
                }
            } else {
                for (size_t k = 0; k < K; ++k) {
                    topk_weights[k] = 1.0 / static_cast<double>(K);
                }
            }
        } else {
            for (size_t k = 0; k < K; ++k) {
                topk_weights[k] = all_probs[topk_indices[k]];
            }
        }

        // 5. 更新长程遥测指标
        telemetry_.total_routings++;
        telemetry_.total_bypassed_evaluations += (N - K);
        for (size_t k = 0; k < K; ++k) {
            telemetry_.expert_dispatches[topk_indices[k]]++;
        }
        for (size_t i = 0; i < N; ++i) {
            telemetry_.expert_prob_sums[i] += all_probs[i];
        }

        // 6. 批次历史入队 (仅在训练/损失反传模式下执行)
        if (record_for_aux_loss) {
            batch_probs_.push_back(all_probs);
            batch_topk_.push_back(topk_indices);
            std::vector<double> inp_copy(D, 0.0);
            if (inputs != nullptr) {
                for (size_t d = 0; d < actual_dim; ++d) {
                    double val = inputs[d];
                    if (std::isnan(val) || std::isinf(val)) val = 0.0;
                    inp_copy[d] = val;
                }
            }
            batch_inputs_.push_back(std::move(inp_copy));
        }

        return cached_decision_;
    }

    /**
     * @brief 计算 Switch / DeepSeek-V3 规范辅助负载均衡损失
     * L_balance = alpha * N * \sum_{i=0}^{N-1} f_i * P_i
     * 其中:
     *   P_i = (1/T) \sum_{t=1}^T p_i(t)   (全量 Softmax 概率批次期望)
     *   f_i = (1 / (T * K)) \sum_{t=1}^T \mathbb{I}(i \in TopK(t)) (实际指派率)
     */
    double compute_load_balancing_loss() const {
        const size_t T = batch_probs_.size();
        if (T == 0) return 0.0;

        const size_t N = config_.num_experts;
        const size_t K = config_.top_k;

        std::vector<double> P(N, 0.0);
        std::vector<double> f(N, 0.0);

        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < N; ++i) {
                P[i] += batch_probs_[t][i];
            }
            for (size_t idx : batch_topk_[t]) {
                f[idx] += 1.0;
            }
        }

        for (size_t i = 0; i < N; ++i) {
            P[i] /= static_cast<double>(T);
            f[i] /= static_cast<double>(T * K);
        }

        double sum_fp = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum_fp += f[i] * P[i];
        }

        return config_.load_balance_alpha * static_cast<double>(N) * sum_fp;
    }

    /**
     * @brief 计算负载均衡辅助损失关于路由参数 (W_r, b) 的闭式精确解析梯度
     * 
     * 理论推导:
     *   L = (alpha * N / T) \sum_{t=1}^T \sum_{i=0}^{N-1} f_i * p_i(t)   [f_i 作为非微常数算子阻断反传]
     *   dL / ds_j(t) = (alpha * N / T) * p_j(t) * (f_j - \bar{f}(t)), 
     *   其中 \bar{f}(t) = \sum_{i=0}^{N-1} f_i * p_i(t)
     *   dL / dW_{j, d} = \sum_{t=1}^T (dL / ds_j(t)) * x_d(t)
     *   dL / db_j      = \sum_{t=1}^T (dL / ds_j(t))
     */
    MoERouterGradients compute_auxiliary_loss_gradients() const {
        MoERouterGradients grads;
        const size_t N = config_.num_experts;
        const size_t D = config_.input_dim;
        const size_t K = config_.top_k;
        const size_t T = batch_probs_.size();

        grads.d_weights.assign(N * D, 0.0);
        grads.d_biases.assign(N, 0.0);
        grads.num_samples = T;

        if (T == 0) return grads;

        // 1. 统计全局指派率分布 f_i
        std::vector<double> f(N, 0.0);
        for (size_t t = 0; t < T; ++t) {
            for (size_t idx : batch_topk_[t]) {
                f[idx] += 1.0;
            }
        }
        for (size_t i = 0; i < N; ++i) {
            f[i] /= static_cast<double>(T * K);
        }

        const double scale = (config_.load_balance_alpha * static_cast<double>(N)) / static_cast<double>(T);

        // 2. 逐样本累积解析梯度
        for (size_t t = 0; t < T; ++t) {
            const auto& p_t = batch_probs_[t];
            const auto& x_t = batch_inputs_[t];

            double f_bar = 0.0;
            for (size_t i = 0; i < N; ++i) {
                f_bar += f[i] * p_t[i];
            }

            for (size_t j = 0; j < N; ++j) {
                double dL_ds = scale * p_t[j] * (f[j] - f_bar);
                grads.d_biases[j] += dL_ds;

                const size_t row_offset = j * D;
                for (size_t d = 0; d < D; ++d) {
                    grads.d_weights[row_offset + d] += dL_ds * x_t[d];
                }
            }
        }

        return grads;
    }

    /**
     * @brief 路由参数梯度下降更新步
     * @param grads 梯度向量
     * @param learning_rate 学习率 eta
     * @param weight_decay 权重衰减系数
     */
    void apply_gradients(const MoERouterGradients& grads, double learning_rate, double weight_decay = 0.0) {
        if (grads.num_samples == 0) return;
        if (grads.d_weights.size() != weights_.size() || grads.d_biases.size() != biases_.size()) return;

        const size_t total_weights = weights_.size();
        for (size_t i = 0; i < total_weights; ++i) {
            if (weight_decay > 0.0) {
                weights_[i] -= learning_rate * weight_decay * weights_[i];
            }
            weights_[i] -= learning_rate * grads.d_weights[i];
        }

        const size_t total_biases = biases_.size();
        for (size_t i = 0; i < total_biases; ++i) {
            biases_[i] -= learning_rate * grads.d_biases[i];
        }
    }

private:
    MoERouterConfig config_;
    std::vector<double> weights_;
    std::vector<double> biases_;
    MoETelemetry telemetry_;
    mutable std::mt19937 rng_;

    std::vector<std::vector<double>> batch_probs_;
    std::vector<std::vector<size_t>> batch_topk_;
    std::vector<std::vector<double>> batch_inputs_;

    std::vector<size_t> scratch_indices_;
    RoutingDecision cached_decision_;
};

/**
 * @brief 基于微柱神经基底的稀疏混合专家网络 (CellularOrganism MoE)
 * 
 * 核心特性:
 * - 聚合 N 个独立微柱生命体 (CellularOrganism)
 * - 门控路由器精确调度 Top-K 活跃专家
 * - 严格零算力旁路 (Zero-Compute Bypass): 未激活的 N-K 个微柱不消耗任何前向与循环计算
 */
class CellularOrganismMoE {
public:
    CellularOrganismMoE(
        const std::vector<CellularOrganism>& expert_organisms,
        size_t top_k = 2,
        size_t input_dim = 4,
        double load_balance_alpha = 0.01,
        uint32_t router_seed = 42)
        : experts_(expert_organisms),
          active_call_count_(0) {
        if (experts_.empty()) {
            CellularOrganism default_expert;
            default_expert.compile();
            experts_.push_back(default_expert);
        }
        if (top_k == 0) top_k = 1;
        if (top_k > experts_.size()) top_k = experts_.size();

        MoERouterConfig cfg;
        cfg.num_experts = experts_.size();
        cfg.top_k = top_k;
        cfg.input_dim = input_dim;
        cfg.load_balance_alpha = load_balance_alpha;
        router_ = CellularRouter(cfg, router_seed);

        for (auto& exp : experts_) {
            if (!exp.is_compiled()) {
                exp.compile();
            }
        }
    }

    size_t num_experts() const { return experts_.size(); }
    size_t top_k() const { return router_.config().top_k; }
    CellularRouter& router() { return router_; }
    const CellularRouter& router() const { return router_; }
    std::vector<CellularOrganism>& experts() { return experts_; }
    const std::vector<CellularOrganism>& experts() const { return experts_; }
    uint64_t active_call_count() const { return active_call_count_; }

    void reset_call_counter() { active_call_count_ = 0; }

    void reset_state(bool full_reset = false) {
        for (auto& exp : experts_) {
            exp.reset_state(full_reset);
        }
        router_.reset_batch_history();
    }

    /**
     * @brief 稀疏 MoE 4 维固定输入前向推演 (与 CellularOrganism::forward 接口完全对称)
     */
    CellularOrganism::ActionOutputs forward(
        const double inputs[4],
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        return forward_nd(inputs, 4, enable_hebbian, out_decision, record_for_aux_loss);
    }

    /**
     * @brief 稀疏 MoE 动态多维指针输入前向推演
     */
    CellularOrganism::ActionOutputs forward(
        const double* inputs,
        size_t in_dim,
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        return forward_nd(inputs, in_dim, enable_hebbian, out_decision, record_for_aux_loss);
    }

    /**
     * @brief 稀疏 MoE 核心前向推演 (具有严格的 Zero-Compute Bypass 与零堆分配)
     * 仅对被选中的 Top-K 专家执行实际的前向计算！
     */
    CellularOrganism::ActionOutputs forward_nd(
        const double* inputs,
        size_t in_dim = 4,
        bool enable_hebbian = false,
        RoutingDecision* out_decision = nullptr,
        bool record_for_aux_loss = false) {
        
        const auto& decision = router_.route(inputs, in_dim, record_for_aux_loss);
        if (out_decision) *out_decision = decision;

        CellularOrganism::ActionOutputs combined{};
        combined.thought_mode = "MOE_SPARSE";

        // === 严格零算力旁路 (Zero-Compute Bypass) ===
        for (size_t k = 0; k < decision.topk_indices.size(); ++k) {
            size_t idx = decision.topk_indices[k];
            if (idx >= experts_.size()) continue;
            double weight = decision.topk_weights[k];

            active_call_count_++;
            auto out = experts_[idx].forward_nd(inputs, in_dim, enable_hebbian);

            combined.positive_action += weight * out.positive_action;
            combined.negative_action += weight * out.negative_action;
            combined.defensive_reset += weight * out.defensive_reset;
            combined.predicted_sense_0 += weight * out.predicted_sense_0;
            combined.predicted_sense_1 += weight * out.predicted_sense_1;
            combined.prediction_error  += weight * out.prediction_error;
            combined.thought_energy    += weight * out.thought_energy;
            if (out.immune_lock) combined.immune_lock = true;
        }

        return combined;
    }

    /**
     * @brief 密集基准前向推演 (全量 N 个专家均激活计算，用于算力与时延消融对账，真·零 GC 零堆分配)
     */
    CellularOrganism::ActionOutputs forward_dense(
        const double* inputs,
        size_t in_dim = 4,
        bool enable_hebbian = false) {

        const size_t N = experts_.size();
        CellularOrganism::ActionOutputs combined{};
        combined.thought_mode = "DENSE_ALL_ACTIVE";
        if (N == 0) return combined;

        // 统一均权累加融合，杜绝堆分配
        double weight = 1.0 / static_cast<double>(N);
        for (size_t i = 0; i < N; ++i) {
            active_call_count_++;
            auto out = experts_[i].forward_nd(inputs, in_dim, enable_hebbian);
            combined.positive_action += weight * out.positive_action;
            combined.negative_action += weight * out.negative_action;
            combined.defensive_reset += weight * out.defensive_reset;
            combined.predicted_sense_0 += weight * out.predicted_sense_0;
            combined.predicted_sense_1 += weight * out.predicted_sense_1;
            combined.prediction_error  += weight * out.prediction_error;
            combined.thought_energy    += weight * out.thought_energy;
            if (out.immune_lock) combined.immune_lock = true;
        }

        return combined;
    }

private:
    std::vector<CellularOrganism> experts_;
    CellularRouter router_;
    uint64_t active_call_count_{0};
};

} // namespace kun
