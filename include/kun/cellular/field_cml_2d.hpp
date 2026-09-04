#pragma once

// ============================================================================
// FieldCML2DTask — N×N 耦合映射晶格 (CML) 二维时空混沌场 · 百万体素旗舰任务
//
// 动力学: u_{i,j}(t+1) = (1-ε)·f(u_{i,j}) + ε/4·[f(u_{i±1,j}) + f(u_{i,j±1})]
//         f = logistic(r·x·(1-x)), 环形边界, 时空混沌湍流场 (无解析简化)。
//
// 契约:
//   宽模式 (wide=true):  obs = 全场 N² 个体素状态 → 一受体 ≈ 一体素,
//                        细胞-体素同构对齐 (宪章第5条: 百万细胞用于高维物理空间体素)
//   窄模式 (wide=false): obs = 仅 2 个目标体素当前状态 —— 无宽度对照
//   目标: positive_action ≈ 目标A(0,0) 下一状态, negative_action ≈ 目标B(N/2,N/2) 下一状态
//   适应度 = exp(-err/评分尺度), 评分尺度由训练器课程退火 (处处有梯度)
//
// 用途: 百万细胞规模边际收益的诚实试金石 —— 受体网格本身是被动的, 场的因果
//       信息 (扩散耦合邻域) 能否被中间神经元网络真正利用, 实测说话。
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace kun {

class FieldCML2DTask : public EvolvableTask {
public:
    FieldCML2DTask(size_t n = 512, bool wide = true, double coupling = 0.15, double r = 3.7)
        : n_(n), wide_(wide), eps_(coupling), r_(r), u_(n * n, 0.0), nx_(n * n, 0.0) {
        target_a_ = 0;                       // 体素 (0, 0)
        target_b_ = (n / 2) * n + (n / 2);   // 体素 (n/2, n/2) — 场对角远端
    }

    const char* name() const override { return "FieldCML2D"; }
    size_t obs_dim() const override { return wide_ ? n_ * n_ : 2; }
    size_t act_dim() const override { return 4; }
    void set_max_steps(int s) { max_steps_ = s; }
    void set_score_scale(double s) { score_scale_ = s; }
    size_t lattice() const { return n_; }
    size_t voxels() const { return n_ * n_; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::uniform_real_distribution<double> u(0.05, 0.95);
        for (auto& v : u_) v = u(rng_);
        for (int i = 0; i < 120; ++i) cml2d_step();   // 预热进入混沌吸引子
        steps_ = 0; q_sum_ = 0.0;
    }

    std::vector<float> current_observation() const override {
        std::vector<float> o(obs_dim(), 0.0f);
        if (wide_) {
            for (size_t i = 0; i < u_.size(); ++i) o[i] = static_cast<float>(u_[i]);
        } else {
            o[0] = static_cast<float>(u_[target_a_]);
            o[1] = static_cast<float>(u_[target_b_]);
        }
        return o;
    }

    StepResult step(int) override {
        CellularOrganism::ActionOutputs a;
        return step_continuous(a);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        const double t0 = u_[target_a_], t1 = u_[target_b_];
        cml2d_step();                                // 真实场推进 (有机体输出不影响场)

        double err = 0.5 * (std::fabs(acts.positive_action - t0) +
                            std::fabs(acts.negative_action - t1));
        if (!std::isfinite(err)) err = 1.0;
        const double q = std::exp(-err / score_scale_);
        q_sum_ += q;
        ++steps_;

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

    // 持续性基线 (预测"下一步=当前"), 最终评分尺度 0.15 下实测
    double persistence_quality(uint32_t seed, int steps = 200) {
        reset(seed);
        double e = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double a = u_[target_a_], b = u_[target_b_];
            cml2d_step();
            e += 0.5 * (std::fabs(a - u_[target_a_]) + std::fabs(b - u_[target_b_]));
        }
        return std::exp(-(e / steps) / 0.15);
    }

private:
    void cml2d_step() {
        const size_t n = n_;
        for (size_t i = 0; i < n; ++i) {
            const size_t ip = ((i + n - 1) % n) * n;
            const size_t ic = i * n;
            const size_t in = ((i + 1) % n) * n;
            for (size_t j = 0; j < n; ++j) {
                const size_t jl = (j + n - 1) % n;
                const size_t jr = (j + 1) % n;
                const double f  = r_ * u_[ic + j] * (1.0 - u_[ic + j]);
                const double fu = r_ * u_[ip + j] * (1.0 - u_[ip + j]);
                const double fd = r_ * u_[in + j] * (1.0 - u_[in + j]);
                const double fl = r_ * u_[ic + jl] * (1.0 - u_[ic + jl]);
                const double fr = r_ * u_[ic + jr] * (1.0 - u_[ic + jr]);
                nx_[ic + j] = (1.0 - eps_) * f + 0.25 * eps_ * (fu + fd + fl + fr);
            }
        }
        u_.swap(nx_);
    }

    size_t n_;
    bool wide_;
    double eps_;
    double r_;
    size_t target_a_{0}, target_b_{0};
    std::vector<double> u_, nx_;
    int steps_{0};
    int max_steps_{200};
    double score_scale_{0.15};
    double q_sum_{0.0};
    std::mt19937 rng_{1};
};

} // namespace kun
