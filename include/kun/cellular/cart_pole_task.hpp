#pragma once

// ============================================================================
// CartPoleBalanceTask — 倒立摆平衡任务 (两轮平衡/悬停类邻居域的教科书形态)
//
// 目的: 验证 EvolvableTask 管线的横向可复制性 —— 新域只需实现本任务类,
//       复用 MorphogeneticEvolutionEngine / TaskEvaluator / 门禁 / 前端全链路。
//
// 物理观测量 (4 维, 恰好对齐底座 4 通道输入):
//   obs[0] = 小车位置 x      (归一化 by x_threshold)
//   obs[1] = 小车速度 x_dot  (归一化 by 3.0)
//   obs[2] = 摆角 theta      (归一化 by 0.35 rad)
//   obs[3] = 摆角速度        (归一化 by 3.0)
//
// 动作: 连续推力 force = clamp(pos - neg, [-1,1]) * force_mag (+ 可选噪声)
// 生存判定: |theta| > 12° 或 |x| > 2.4m 或 NaN → 回合终止
// success 定义 = 满步存活 (门禁语义: 生存率)
// OOD 变体: 更重的摆锤 / 更长的摆杆 / 推力噪声 (跨物理参数泛化)
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>

namespace kun {

class CartPoleBalanceTask : public EvolvableTask {
public:
    struct Params {
        double gravity{9.8};
        double masscart{1.0};
        double masspole{0.1};
        double length{0.5};      // 半摆长
        double force_mag{10.0};
        double tau{0.02};
        double theta_threshold{12.0 * M_PI / 180.0};
        double x_threshold{2.4};
        double force_noise{0.0}; // OOD 干扰: 推力高斯噪声幅度
    };

    CartPoleBalanceTask() : P() {}
    explicit CartPoleBalanceTask(const Params& p) : P(p) {}

    const char* name() const override { return "CartPoleBalance"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 4; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::uniform_real_distribution<double> ux(-0.10, 0.10);
        std::uniform_real_distribution<double> ut(-0.12, 0.12);
        x_ = ux(rng_);
        theta_ = ut(rng_);
        x_dot_ = 0.0;
        theta_dot_ = 0.0;
        steps_ = 0;
        sum_abs_theta_ = 0.0;
    }

    std::vector<float> current_observation() const override {
        // 通道 0/1 = 摆角/角速度 (默认 LOCKED 骨架的 2 个感受器必须直视摆杆)
        return {static_cast<float>(theta_ / 0.35),
                static_cast<float>(theta_dot_ / 3.0),
                static_cast<float>(x_ / P.x_threshold),
                static_cast<float>(x_dot_ / 3.0)};
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = (action == 0 || action == 2) ? 1.0 : 0.0;
        acts.negative_action = (action == 1 || action == 2) ? 1.0 : 0.0;
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        double force = acts.positive_action - acts.negative_action;
        if (!std::isfinite(force)) force = 0.0;
        force = std::max(-1.0, std::min(1.0, force)) * P.force_mag;
        if (P.force_noise > 0.0) force += P.force_noise * gauss_(rng_);

        const double cos_t = std::cos(theta_);
        const double sin_t = std::sin(theta_);
        const double total_mass = P.masscart + P.masspole;
        const double polemass_length = P.masspole * P.length;

        const double temp = (force + polemass_length * theta_dot_ * theta_dot_ * sin_t) / total_mass;
        const double theta_acc =
            (P.gravity * sin_t - cos_t * temp) /
            (P.length * (4.0 / 3.0 - P.masspole * cos_t * cos_t / total_mass));
        const double x_acc = temp - polemass_length * theta_acc * cos_t / total_mass;

        x_ += P.tau * x_dot_;
        x_dot_ += P.tau * x_acc;
        theta_ += P.tau * theta_dot_;
        theta_dot_ += P.tau * theta_acc;

        ++steps_;
        sum_abs_theta_ += std::fabs(theta_);

        const bool diverged = !std::isfinite(x_) || !std::isfinite(theta_);
        const bool fell = std::fabs(theta_) > P.theta_threshold;
        const bool ran_away = std::fabs(x_) > P.x_threshold;
        const bool done = diverged || fell || ran_away;

        res.obs = current_observation();
        res.steps = steps_;
        res.done = done;
        res.success = !done && steps_ >= max_steps_;  // 满步存活 = 成功
        res.reward = done ? 0.0 : 1.0;
        res.min_dist_to_goal = std::fabs(theta_);     // "距目标" = 距竖直的偏差
        return res;
    }

    double current_fitness() const override {
        const double survival = static_cast<double>(steps_) / static_cast<double>(max_steps_);
        const double upright = 1.0 - std::min(1.0, (steps_ > 0 ? sum_abs_theta_ / steps_ : 1.0) / P.theta_threshold);
        return survival * 0.8 + upright * 0.2;  // 生存为主, 直立质量为辅
    }

    void set_max_steps(int s) { max_steps_ = s; }
    int max_steps() const { return max_steps_; }

private:
    Params P;
    std::mt19937 rng_{42};
    std::normal_distribution<double> gauss_{0.0, 1.0};
    double x_{0.0}, x_dot_{0.0}, theta_{0.0}, theta_dot_{0.0};
    int steps_{0};
    int max_steps_{300};
    double sum_abs_theta_{0.0};
};

} // namespace kun
