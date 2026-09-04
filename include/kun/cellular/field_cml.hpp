#pragma once

// ============================================================================
// FieldCMLTask — 32 节点耦合映射晶格 (CML) 时空混沌场 · 宽契约旗舰任务
//
// 动力学: u_i(t+1) = (1-ε)·f(u_i) + ε/2·[f(u_{i-1}) + f(u_{i+1})],  f = logistic(4x(1-x))
//         时空混沌 + 环形扩散耦合, 真实高维场 (无解析简化)。
//
// 契约:
//   宽模式 (n_channels=32): obs = 全部 32 个节点状态 → 邻居携带因果信息,
//                           预测目标节点时理论上唯有宽输入占优 —— 宽契约价值的试金石
//   窄模式 (n_channels=2):  obs = 仅目标节点 (0, 16) 当前状态 —— 对照组
//   目标: positive_action ≈ 下一节点0状态, negative_action ≈ 下一节点16状态
//   适应度 = exp(-err/评分尺度), 评分尺度由训练器做课程退火 (处处有梯度)
//
// 用途: 证明底座宽契约 (forward_nd + SENSE_CHANNEL + ensure_receptors) 的
//       价值 —— 高维信息吞吐下, 规模与宽度才可能买到真实预测力。
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace kun {

class FieldCMLTask : public EvolvableTask {
public:
    FieldCMLTask(size_t n_nodes = 32, size_t n_channels = 32, double coupling = 0.15, double r = 3.7)
        : n_nodes_(n_nodes), n_channels_(n_channels), eps_(coupling), r_(r) {}

    const char* name() const override { return "FieldCML"; }
    size_t obs_dim() const override { return n_channels_; }
    size_t act_dim() const override { return 4; }
    void set_max_steps(int s) { max_steps_ = s; }
    void set_score_scale(double s) { score_scale_ = s; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::uniform_real_distribution<double> u(0.05, 0.95);
        u_.assign(n_nodes_, 0.0);
        for (auto& v : u_) v = u(rng_);
        for (int i = 0; i < 120; ++i) cml_step();   // 预热进入混沌吸引子
        steps_ = 0; q_sum_ = 0.0;
        fill_obs();
    }

    std::vector<float> current_observation() const override {
        std::vector<float> o(obs_.size());
        for (size_t i = 0; i < obs_.size(); ++i) o[i] = static_cast<float>(obs_[i]);
        return o;
    }

    StepResult step(int) override {
        CellularOrganism::ActionOutputs a;
        return step_continuous(a);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        const double t0 = u_[0], t1 = u_[node_b_];
        cml_step();                                  // 真实场推进 (有机体输出不影响场)

        double err = 0.5 * (std::fabs(acts.positive_action - t0) +
                            std::fabs(acts.negative_action - t1));
        if (!std::isfinite(err)) err = 1.0;
        const double q = std::exp(-err / score_scale_);
        q_sum_ += q;
        ++steps_;

        fill_obs();
        res.obs = current_observation();
        res.steps = steps_;
        res.done = false;
        res.success = false;
        res.reward = q;
        res.min_dist_to_goal = 1.0 - q;
        return res;
    }

    double current_fitness() const override {
        return steps_ > 0 ? q_sum_ / static_cast<double>(steps_) : 0.0;
    }

    // 持续性基线 (预测"下一步=当前"), 最终评分尺度下实测
    double persistence_quality(uint32_t seed, int steps = 200) {
        reset(seed);
        double e = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double a = u_[0], b = u_[node_b_];
            cml_step();
            e += 0.5 * (std::fabs(a - u_[0]) + std::fabs(b - u_[node_b_]));
        }
        return std::exp(-(e / steps) / 0.15);
    }

private:
    void cml_step() {
        std::vector<double> nx(n_nodes_);
        for (size_t i = 0; i < n_nodes_; ++i) {
            const size_t p = (i + n_nodes_ - 1) % n_nodes_;
            const size_t n = (i + 1) % n_nodes_;
            const double f  = r_ * u_[i] * (1.0 - u_[i]);
            const double fp = r_ * u_[p] * (1.0 - u_[p]);
            const double fn = r_ * u_[n] * (1.0 - u_[n]);
            nx[i] = (1.0 - eps_) * f + 0.5 * eps_ * (fp + fn);
        }
        u_.swap(nx);
    }
    void fill_obs() {
        if (n_channels_ >= n_nodes_) {
            obs_.resize(n_nodes_);
            for (size_t i = 0; i < n_nodes_; ++i) obs_[i] = u_[i];
        } else {
            obs_ = {u_[0], u_[node_b_]};             // 窄契约: 仅目标节点自身
        }
    }

    size_t n_nodes_;
    size_t n_channels_;
    double eps_;
    double r_;
    size_t node_b_ = 16;
    std::vector<double> u_, obs_;
    int steps_{0};
    int max_steps_{200};
    double score_scale_{0.15};
    double q_sum_{0.0};
    std::mt19937 rng_{1};
};

} // namespace kun
