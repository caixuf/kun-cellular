#pragma once

// ============================================================================
// SlingshotNavTask — 混沌三体引力弹弓导航任务 (organism 真前向驱动)
//
// 目的: 把「引力弹弓」演示接入底座真前向 —— 探测器推力由演化出的
//       CellularOrganism 连续输出驱动，在三天体牛顿引力场中抵达目标。
//
// 世界 (画布坐标 800x600):
//   3 个天体相互牛顿引力运动 (混沌三体)，探测器受天体引力 + 自身推力。
//   目标为固定坐标点，探测器进入目标半径即成功。
//
// 观测 (16 维): 3×天体相对位/速 (12) + 目标相对位 (2) + 探测器速度 (2)
// 动作 (3 维): positive_action->ax, negative_action->ay, defensive_reset->boost
// 成功: 探测器进入目标半径
// OOD 变体: 更强引力常数 / 更大天体质量 (跨物理参数泛化)
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <array>
#include <cmath>
#include <random>

namespace kun {

class SlingshotNavTask : public EvolvableTask {
public:
    struct Params {
        double G{1.5};
        double thrust{0.18};
        double target_r{50.0};
        double max_speed{12.0};
        double mass_scale{1.0}; // OOD: 天体质量缩放
        double width{800.0};
        double height{600.0};
    };

    static constexpr size_t kStars = 3;

    SlingshotNavTask() : P_() {}
    explicit SlingshotNavTask(const Params& p) : P_(p) {}

    const char* name() const override { return "SlingshotNav"; }
    size_t obs_dim() const override { return 16; }
    size_t act_dim() const override { return 3; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::uniform_real_distribution<double> jitter(-20.0, 20.0);
        // 三天体初始 (混沌但确定性)
        star_x_ = {300.0 + jitter(rng_), 470.0 + jitter(rng_), 560.0 + jitter(rng_)};
        star_y_ = {340.0 + jitter(rng_), 250.0 + jitter(rng_), 430.0 + jitter(rng_)};
        star_vx_ = {0.6, -0.5, 0.4};
        star_vy_ = {-0.3, 0.5, -0.4};
        star_m_ = {800.0 * P_.mass_scale, 600.0 * P_.mass_scale, 700.0 * P_.mass_scale};
        // 探测器: 从左下发射
        probe_x_ = 100.0 + jitter(rng_);
        probe_y_ = 520.0 + jitter(rng_);
        probe_vx_ = 1.0;
        probe_vy_ = -0.5;
        target_x_ = 600.0;
        target_y_ = 220.0;
        steps_ = 0;
        min_dist_ = dist_to_target();
        init_dist_ = min_dist_;
        reached_ = false;
    }

    std::vector<float> current_observation() const override {
        std::vector<float> obs(obs_dim(), 0.0f);
        for (size_t s = 0; s < kStars; ++s) {
            obs[s * 4 + 0] = static_cast<float>((star_x_[s] - probe_x_) / 400.0);
            obs[s * 4 + 1] = static_cast<float>((star_y_[s] - probe_y_) / 400.0);
            obs[s * 4 + 2] = static_cast<float>(star_vx_[s] / 10.0);
            obs[s * 4 + 3] = static_cast<float>(star_vy_[s] / 10.0);
        }
        obs[12] = static_cast<float>((target_x_ - probe_x_) / 400.0);
        obs[13] = static_cast<float>((target_y_ - probe_y_) / 400.0);
        obs[14] = static_cast<float>(probe_vx_ / 10.0);
        obs[15] = static_cast<float>(probe_vy_ / 10.0);
        return obs;
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = (action == 0) ? 1.0 : ((action == 1) ? -1.0 : 0.0);
        acts.negative_action = (action == 2) ? 1.0 : ((action == 3) ? -1.0 : 0.0);
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        double ax_in = clamp1(acts.positive_action);
        double ay_in = clamp1(acts.negative_action);
        double boost = 0.5 + 0.5 * clamp1(acts.defensive_reset);
        if (!std::isfinite(ax_in)) ax_in = 0.0;
        if (!std::isfinite(ay_in)) ay_in = 0.0;
        if (!std::isfinite(boost)) boost = 0.5;

        // 天体相互引力 (混沌三体)
        double asx[kStars] = {0, 0, 0};
        double asy[kStars] = {0, 0, 0};
        for (size_t i = 0; i < kStars; ++i) {
            for (size_t j = i + 1; j < kStars; ++j) {
                const double dx = star_x_[j] - star_x_[i];
                const double dy = star_y_[j] - star_y_[i];
                const double r2 = dx * dx + dy * dy + 100.0;
                const double inv = P_.G / (r2 * std::sqrt(r2));
                asx[i] += inv * star_m_[j] * dx; asy[i] += inv * star_m_[j] * dy;
                asx[j] -= inv * star_m_[i] * dx; asy[j] -= inv * star_m_[i] * dy;
            }
        }

        // 探测器引力 + 推力
        double apx = 0.0, apy = 0.0;
        for (size_t s = 0; s < kStars; ++s) {
            const double dx = star_x_[s] - probe_x_;
            const double dy = star_y_[s] - probe_y_;
            const double r2 = dx * dx + dy * dy + 100.0;
            const double inv = P_.G / (r2 * std::sqrt(r2));
            apx += inv * star_m_[s] * dx;
            apy += inv * star_m_[s] * dy;
        }
        apx += P_.thrust * boost * ax_in;
        apy += P_.thrust * boost * ay_in;

        // 积分
        for (size_t s = 0; s < kStars; ++s) {
            star_vx_[s] += asx[s]; star_vy_[s] += asy[s];
            star_x_[s] += star_vx_[s]; star_y_[s] += star_vy_[s];
        }
        probe_vx_ += apx; probe_vy_ += apy;
        probe_vx_ = std::max(-P_.max_speed, std::min(P_.max_speed, probe_vx_));
        probe_vy_ = std::max(-P_.max_speed, std::min(P_.max_speed, probe_vy_));
        probe_x_ += probe_vx_; probe_y_ += probe_vy_;

        ++steps_;
        const double d = dist_to_target();
        if (d < min_dist_) min_dist_ = d;
        const bool diverged = !std::isfinite(probe_x_) || !std::isfinite(probe_y_);
        if (d <= P_.target_r) reached_ = true;
        const bool out_of_bounds = probe_x_ < -200.0 || probe_x_ > P_.width + 200.0 ||
                                   probe_y_ < -200.0 || probe_y_ > P_.height + 200.0;
        const bool done = diverged || out_of_bounds || steps_ >= max_steps_;

        StepResult res;
        res.obs = current_observation();
        res.steps = steps_;
        res.done = done;
        res.success = reached_;
        res.reward = reached_ ? 1.0 : 0.0;
        res.min_dist_to_goal = min_dist_;
        return res;
    }

    double current_fitness() const override {
        const double base = std::max(1.0, init_dist_);
        const double progress = std::min(1.0, std::max(0.0, (base - min_dist_) / base));
        return progress + (reached_ ? 1.0 : 0.0);
    }

    void set_max_steps(int s) { max_steps_ = s; }
    int max_steps() const { return max_steps_; }
    size_t num_stars() const { return kStars; }
    double star_x(size_t i) const { return star_x_[i]; }
    double star_y(size_t i) const { return star_y_[i]; }
    double probe_x() const { return probe_x_; }
    double probe_y() const { return probe_y_; }
    double target_x() const { return target_x_; }
    double target_y() const { return target_y_; }
    double target_r() const { return P_.target_r; }
    bool reached() const { return reached_; }
    double min_dist() const { return min_dist_; }

private:
    static double clamp1(double v) { return std::max(-1.0, std::min(1.0, v)); }
    double dist_to_target() const {
        const double dx = target_x_ - probe_x_, dy = target_y_ - probe_y_;
        return std::sqrt(dx * dx + dy * dy);
    }

    Params P_;
    std::mt19937 rng_{42};
    std::array<double, kStars> star_x_{}, star_y_{}, star_vx_{}, star_vy_{}, star_m_{};
    double probe_x_{0}, probe_y_{0}, probe_vx_{0}, probe_vy_{0};
    double target_x_{700.0}, target_y_{120.0};
    int steps_{0};
    int max_steps_{400};
    double min_dist_{999.0};
    double init_dist_{999.0};
    bool reached_{false};
};

} // namespace kun
