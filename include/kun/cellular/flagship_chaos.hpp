#pragma once

// ============================================================================
// FlagshipChaosTask — 旗舰百万细胞任务: Lorenz 混沌一步预测 (世界模型/储层)
//
// 为什么选它: 这是少数"规模真有用处"的诚实场景。控制任务 8~16 细胞就够
// (尺度真理已证), 但混沌时间序列的一步预测里, 储层容量直接决定对吸引子
// 动力学的覆盖能力 —— 百万细胞的边际收益可以被真实测量, 而不是吹出来的。
//
// 契约:
//   obs = (x/30, y/30, z/40, x_prev/30)   Lorenz 吸引子状态 (4通道)
//   目标: ActionOutputs.predicted_sense_0 ≈ 下一步 x/30
//         ActionOutputs.predicted_sense_1 ≈ 下一步 y/30
//   质量质量 = 1 - min(1, err/0.03), err = 0.5(|Δpx| + |Δpy|)
//   无物理死亡 (纯预测任务), 适应度 = 平均预测质量
//   持续性基线 (persistence: 预测"下一步=当前") 由训练器实测对照。
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace kun {

class FlagshipChaosTask : public EvolvableTask {
public:
    FlagshipChaosTask() = default;

    const char* name() const override { return "FlagshipChaos"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 4; }
    void set_max_steps(int s) { max_steps_ = s; }
    // 课程学习: 评分尺度退火 (粗→细), 保证演化早期处处有坡度
    void set_score_scale(double s) { score_scale_ = s; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::normal_distribution<double> n;
        x_ = n(rng_) * 5.0; y_ = n(rng_) * 7.0; z_ = 25.0 + n(rng_) * 5.0;
        // 预热: 落回吸引子流形
        for (int i = 0; i < 300; ++i) lorenz_step();
        steps_ = 0; q_sum_ = 0.0;
        prev_x_ = x_;
        fill_obs();
    }

    std::vector<float> current_observation() const override {
        return {static_cast<float>(o_[0]), static_cast<float>(o_[1]),
                static_cast<float>(o_[2]), static_cast<float>(o_[3])};
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        return step_continuous(acts);
    }

    // 推进 Lorenz 一环境步, 用有机体动作通道的前瞻预测对照真实下一步
    // (预测走标准效应器: positive_action ≈ 下步 x/30, negative_action ≈ 下步 y/30 ——
    //  保证任何形态的有机体都有预测输出通路, 演化梯度恒存在)
    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        const double nx = x_, ny = y_;          // 预测目标基于推进前状态缓存
        lorenz_step();                           // 真实动力学推进 (有机体输出不影响物理)

        double err = 0.5 * (std::fabs(acts.positive_action - nx / 30.0) +
                            std::fabs(acts.negative_action - ny / 30.0));
        if (!std::isfinite(err)) err = 1.0;
        const double q = std::exp(-err / score_scale_);  // 处处有梯度的指数评分
        q_sum_ += q;
        ++steps_;

        prev_x_ = nx;
        fill_obs();
        res.obs = current_observation();
        res.steps = steps_;
        res.done = false;                        // 纯预测任务无死亡
        res.success = (q_sum_ / steps_) > 0.6;
        res.reward = q;
        res.min_dist_to_goal = 1.0 - q;
        return res;
    }

    double current_fitness() const override {
        return steps_ > 0 ? q_sum_ / static_cast<double>(steps_) : 0.0;
    }

    // 持续性基线: 预测"下一步=当前", 按最终评分尺度 (0.06) 实测
    double persistence_quality(uint32_t seed, int steps = 300) {
        reset(seed);
        double e = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double cx = x_, cy = y_;
            lorenz_step();
            e += 0.5 * (std::fabs(cx - x_) + std::fabs(cy - y_)) / 30.0;
        }
        return std::exp(-(e / steps) / 0.06);
    }

    static constexpr double PRED_NORM = 0.05;   // 参考尺度 (持续性基线 err ≈ 0.025)

private:
    void lorenz_step() {
        for (int k = 0; k < 4; ++k) {            // 4 子步 Euler, dt=0.005
            const double dx = 10.0 * (y_ - x_);
            const double dy = x_ * (28.0 - z_) - y_;
            const double dz = x_ * y_ - (8.0 / 3.0) * z_;
            x_ += 0.005 * dx; y_ += 0.005 * dy; z_ += 0.005 * dz;
        }
    }
    void fill_obs() {
        o_[0] = x_ / 30.0; o_[1] = y_ / 30.0; o_[2] = z_ / 40.0; o_[3] = prev_x_ / 30.0;
    }

    double x_{0}, y_{0}, z_{0}, prev_x_{0};
    double o_[4]{0, 0, 0, 0};
    int steps_{0};
    int max_steps_{300};
    double score_scale_{0.12};
    double q_sum_{0.0};
    std::mt19937 rng_{1};
    std::normal_distribution<double> n_{0.0, 1.0};
};

} // namespace kun
