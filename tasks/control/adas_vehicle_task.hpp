#pragma once

// ============================================================================
// adas_vehicle_task.hpp — 智能驾驶具身车辆控制仿真任务 (EvolvableTask 契约)
//
// 物理模型:
//   - 真实连续阿克曼运动学与车体横向航向动力学 (Bicycle Kinematics/Dynamics)
//   - 闭环公路赛道 (Grand Circuit, 720高精度离线预建采样点)
//   - 4通道受体感知输入:
//       obs[0] = 带符号横向误差 CTE (归一化 by 3.0m, 正=偏左需右打, 负=偏右需左打)
//       obs[1] = 航向偏差 dpsi (归一化 by 0.4 rad, 正=路偏左需左转)
//       obs[2] = 道路当前曲率 kappa (归一化 by 0.04 rad/m, 包含正负转向方向)
//       obs[3] = 横向偏差变化率 CTE rate (阻尼微分, 归一化 by 2.0 m/s)
//   - 动作效应器:
//       pos = 左转向 (ACT_PRIMARY_POSITIVE)
//       neg = 右转向 (ACT_PRIMARY_NEGATIVE)
//       net_steer = (pos - neg) * delta_max
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <algorithm>

namespace kun {

class ADASVehicleTask : public EvolvableTask {
public:
    struct TrackPoint {
        float x{0.0f};
        float y{0.0f};
        float theta{0.0f};
        float curv{0.0f};
    };

    struct Params {
        double L{18.0};            // 轴距 (m, 适配全景 Grand Circuit 300m 尺度)
        double dt{0.04};           // 仿真步长 40ms (25Hz 连续控制)
        double road_half_w{23.0};  // 路面半宽 (m)
        double delta_max{0.55};    // 最大前轮转角 (rad, 约 31.5度)
        double steer_rate_lim{0.06};// 单步转向角增量限幅 (rad/step)
        double steer_filter{0.38}; // 执行器一阶响应系数
        double base_speed{4.8};    // 基础车速 (m/s)
        double friction{1.0};      // 路面附着系数 (OOD 时降至 0.65)
        double gust_disturb{0.0};  // 侧风扰动加速度 (m/s^2)
        double track_scale{1.0};   // 赛道尺度缩放 (OOD 时 1.15)
        int max_steps{1000};       // 单回合最大步数
    };

    ADASVehicleTask() : P() {
        build_track();
    }

    explicit ADASVehicleTask(const Params& p) : P(p) {
        build_track();
    }

    const char* name() const override { return "ADASVehicleTrackFollowing"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 2; }
    void set_max_steps(int s) { P.max_steps = s; }
    int max_steps() const { return P.max_steps; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        std::normal_distribution<double> pos_noise(0.0, 0.15);
        std::normal_distribution<double> th_noise(0.0, 0.02);

        x_ = track_[0].x + pos_noise(rng_);
        y_ = track_[0].y + pos_noise(rng_);
        theta_ = track_[0].theta + th_noise(rng_);
        v_ = P.base_speed;
        delta_ = 0.0;
        prev_delta_ = 0.0;

        track_idx_ = 0;
        steps_ = 0;
        total_abs_cte_ = 0.0;
        max_cte_ = 0.0;
        jerk_sum_ = 0.0;
        done_ = false;
        success_ = false;

        // 初始化计算初始跟踪误差
        update_tracking_errors();
        prev_signed_cte_ = current_signed_cte_;
    }

    std::vector<float> current_observation() const override {
        float obs0 = static_cast<float>(std::clamp(current_signed_cte_ / 3.0, -1.5, 1.5));
        float obs1 = static_cast<float>(std::clamp(current_heading_err_ / 0.4, -1.5, 1.5));
        float obs2 = static_cast<float>(std::clamp(current_curv_ * 25.0, -1.5, 1.5));
        float obs3 = static_cast<float>(std::clamp(-current_cte_rate_ / 2.0, -1.5, 1.5));
        return {obs0, obs1, obs2, obs3};
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = (action == 0) ? 1.0 : 0.0;
        acts.negative_action = (action == 1) ? 1.0 : 0.0;
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;
        if (done_) {
            res.obs = current_observation();
            res.steps = steps_;
            res.done = true;
            res.success = success_;
            return res;
        }

        // 解析连续转向指令 (positive = 左转, negative = 右转)
        double steer_cmd = acts.positive_action - acts.negative_action;
        if (!std::isfinite(steer_cmd)) steer_cmd = 0.0;
        steer_cmd = std::clamp(steer_cmd, -1.0, 1.0);

        // 执行器一阶响应滤波与速率限幅
        double target_delta = steer_cmd * P.delta_max;
        double delta_diff = (target_delta - delta_) * P.steer_filter;
        double delta_change = std::clamp(delta_diff, -P.steer_rate_lim, P.steer_rate_lim);
        delta_ += delta_change;

        // 记录控制平滑度
        jerk_sum_ += std::abs(delta_change);
        prev_delta_ = delta_;

        // 弯道自适应控速
        double abs_curv = std::abs(current_curv_);
        double target_v = std::clamp(P.base_speed - abs_curv * 65.0, 2.8, 5.5);
        v_ += (target_v - v_) * 0.15;

        // 阿克曼运动学积分
        double beta = std::atan(0.5 * std::tan(delta_));
        double step_dist = v_ * P.dt * 25.0; // 虚拟时间比例因子

        // 叠加环境侧向扰动 (OOD 摩擦系数折减与侧风冲击)
        double lateral_gust = P.gust_disturb * P.dt;
        double eff_cos = std::cos(theta_ + beta);
        double eff_sin = std::sin(theta_ + beta);

        x_ += step_dist * eff_cos;
        y_ += step_dist * eff_sin + lateral_gust;
        theta_ += (step_dist / P.L) * std::cos(beta) * std::tan(delta_) * P.friction;

        // 更新赛道跟踪状态与误差
        update_tracking_errors();

        total_abs_cte_ += current_abs_cte_;
        max_cte_ = std::max(max_cte_, current_abs_cte_);
        ++steps_;

        // 出界检测 (超出道路边缘 92% 判定为驶出路界坠毁)
        bool out_of_bounds = (current_abs_cte_ > P.road_half_w * 0.92);
        if (out_of_bounds || steps_ >= P.max_steps) {
            done_ = true;
            double mean_cte = steps_ > 0 ? (total_abs_cte_ / steps_) : 999.0;
            success_ = (!out_of_bounds && steps_ >= P.max_steps && mean_cte < 1.0);
        }

        // 奖励函数: 稠密跟踪精度奖励
        double r_track = std::max(0.0, 1.0 - current_abs_cte_ / 2.0);
        double r_smooth = std::max(0.0, 1.0 - std::abs(delta_change) / P.steer_rate_lim);
        res.reward = out_of_bounds ? -50.0 : (1.0 + 4.0 * r_track + 0.5 * r_smooth);

        res.obs = current_observation();
        res.steps = steps_;
        res.done = done_;
        res.success = success_;
        res.min_dist_to_goal = current_abs_cte_;

        return res;
    }

    double current_fitness() const override {
        if (steps_ == 0) return 0.0;
        double mean_cte = total_abs_cte_ / static_cast<double>(steps_);
        double mean_jerk = jerk_sum_ / static_cast<double>(steps_);
        double surv = static_cast<double>(steps_) / static_cast<double>(P.max_steps);

        // 强选择压力: 误差越小适应度急剧呈反比例上升
        double fit = (surv * 200.0) + (1000.0 / (0.15 + mean_cte)) - (mean_jerk * 40.0) - (max_cte_ * 10.0);
        if (success_) fit += 2000.0;
        return fit;
    }

    double mean_abs_cte() const {
        return steps_ > 0 ? (total_abs_cte_ / static_cast<double>(steps_)) : 999.0;
    }

    double max_cte() const { return max_cte_; }
    double mean_jerk() const { return steps_ > 0 ? (jerk_sum_ / static_cast<double>(steps_)) : 0.0; }
    double vehicle_x() const { return x_; }
    double vehicle_y() const { return y_; }
    double vehicle_theta() const { return theta_; }
    double current_signed_cte() const { return current_signed_cte_; }
    double current_heading_err() const { return current_heading_err_; }
    double current_curv() const { return current_curv_; }

    // 取得理想的 Stanley + 前馈解析转向目标角 (供 BPTT/Ridge 进行时序反传与闭式解读出)
    float get_ideal_steer_target() const {
        double k_e = 0.95;
        double lat_term = -std::atan(k_e * current_signed_cte_ / (v_ + 0.1));
        double ff_term = current_curv_ * P.L * 0.75;
        double target = current_heading_err_ + lat_term + ff_term;
        return static_cast<float>(std::clamp(target / P.delta_max, -1.0, 1.0));
    }

private:
    void build_track() {
        const int num_pts = 720;
        track_.resize(num_pts);
        const double cx = 400.0;
        const double cy = 300.0;
        const double s_factor = P.track_scale;

        std::vector<double> xs(num_pts), ys(num_pts), thetas(num_pts), curvs(num_pts);
        for (int i = 0; i < num_pts; ++i) {
            double t = (static_cast<double>(i) / num_pts) * (2.0 * M_PI);
            xs[i] = (cx + std::cos(t) * 280.0 + std::sin(t * 2.0) * 80.0) * s_factor;
            ys[i] = (cy + std::sin(t) * 200.0 + std::cos(t * 2.0) * 35.0) * s_factor;
            double dx = (-std::sin(t) * 280.0 + std::cos(t * 2.0) * 160.0) * s_factor;
            double dy = ( std::cos(t) * 200.0 - std::sin(t * 2.0) * 70.0) * s_factor;
            thetas[i] = std::atan2(dy, dx);
        }

        for (int i = 0; i < num_pts; ++i) {
            int next_i = (i + 1) % num_pts;
            double dtheta = thetas[next_i] - thetas[i];
            while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
            while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
            double ds = std::hypot(xs[next_i] - xs[i], ys[next_i] - ys[i]);
            curvs[i] = dtheta / std::max(ds, 1e-4); // signed curvature

            track_[i].x = static_cast<float>(xs[i]);
            track_[i].y = static_cast<float>(ys[i]);
            track_[i].theta = static_cast<float>(thetas[i]);
            track_[i].curv = static_cast<float>(curvs[i]);
        }
    }

    void update_tracking_errors() const {
        const int num_pts = static_cast<int>(track_.size());
        double best_d2 = 1e12;
        int best_idx = track_idx_;

        for (int offset = -3; offset <= 14; ++offset) {
            int cand = (track_idx_ + offset + num_pts) % num_pts;
            double dx = x_ - track_[cand].x;
            double dy = y_ - track_[cand].y;
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = cand;
            }
        }
        track_idx_ = best_idx;

        const auto& pt = track_[best_idx];
        double dx = x_ - pt.x;
        double dy = y_ - pt.y;

        // signed_cte: 正 = 车在切线左侧, 负 = 车在切线右侧
        current_signed_cte_ = std::cos(pt.theta) * dy - std::sin(pt.theta) * dx;
        current_abs_cte_ = std::abs(current_signed_cte_);

        // 航向偏差: 正 = 道路切线偏左
        double dpsi = pt.theta - theta_;
        while (dpsi > M_PI) dpsi -= 2.0 * M_PI;
        while (dpsi < -M_PI) dpsi += 2.0 * M_PI;
        current_heading_err_ = dpsi;

        // 横向偏差变化率
        current_cte_rate_ = (current_signed_cte_ - prev_signed_cte_) / P.dt;
        prev_signed_cte_ = current_signed_cte_;

        // 当前点曲率
        current_curv_ = pt.curv;
    }

    Params P;
    std::vector<TrackPoint> track_;
    mutable int track_idx_{0};

    double x_{0.0};
    double y_{0.0};
    double theta_{0.0};
    double v_{4.8};
    double delta_{0.0};
    double prev_delta_{0.0};
    mutable double prev_signed_cte_{0.0};

    mutable double current_signed_cte_{0.0};
    mutable double current_abs_cte_{0.0};
    mutable double current_heading_err_{0.0};
    mutable double current_cte_rate_{0.0};
    mutable double current_curv_{0.0};

    int steps_{0};
    double total_abs_cte_{0.0};
    double max_cte_{0.0};
    double jerk_sum_{0.0};
    bool done_{false};
    bool success_{false};

    std::mt19937 rng_{1};
};

} // namespace kun
