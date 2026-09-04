#pragma once

/**
 * ============================================================================
 * KunCellular ADAS 具身场景外围适配器 (1M 级: 本能硬实时动力学阻尼)
 * 场景: 100km/h 极速爆胎横摆抑制、防滑扭矩反向代偿与微秒级连续阻尼闭环
 * ============================================================================
 */

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <cstdio>

namespace kun::adas {

struct VehicleState {
    double vx{27.78};     // 纵向速度 (100 km/h = 27.78 m/s)
    double vy{0.0};       // 侧向速度 (m/s)
    double r{0.0};        // 横摆角速度 yaw rate (rad/s)
    double psi{0.0};      // 航向角 yaw (rad)
    double x{0.0};        // 惯性系 X 坐标 (m)
    double y{0.0};        // 惯性系 Y 坐标 / 侧向位移 (m)
    double delta{0.0};    // 前轮转角 (rad)
    
    // 4 轮角速度 (FL, FR, RL, RR) (rad/s)
    std::array<double, 4> omega{27.78 / 0.32, 27.78 / 0.32, 27.78 / 0.32, 27.78 / 0.32};
    std::array<double, 4> slip{0.0, 0.0, 0.0, 0.0};
    
    bool blowout_active{false};
    double blowout_time{0.05}; // 50ms 时爆胎
};

struct VehicleParams {
    double m{1650.0};          // 整备质量 (kg)
    double Iz{2600.0};         // 横摆转动惯量 (kg*m^2)
    double lf{1.25};           // 质心到前轴距离 (m)
    double lr{1.55};           // 质心到后轴距离 (m)
    double track{1.62};        // 轮距 (m)
    double tire_radius{0.32};  // 车轮有效半径 (m)
    double Cf{95000.0};        // 前轮侧偏刚度 (N/rad)
    double Cr{90000.0};        // 后轮侧偏刚度 (N/rad)
};

class BlowoutDynamicsSimulator {
public:
    explicit BlowoutDynamicsSimulator(VehicleParams params = VehicleParams())
        : p_(params) {
        reset();
    }

    void reset() {
        state_ = VehicleState();
        sim_time_ = 0.0;
        peak_yaw_rate_ = 0.0;
        peak_lat_dev_ = 0.0;
        settled_time_ = -1.0;
    }

    // 16 维受体感知张量 (IMU 6 轴 + 轮速 + 侧偏角 + 载荷)
    std::vector<float> get_observation() const {
        std::vector<float> obs(16, 0.0f);
        // 0: yaw_rate (r) 归一化 ([-1, 1] -> [-1 rad/s, 1 rad/s])
        obs[0] = static_cast<float>(std::clamp(state_.r / 1.0, -1.0, 1.0));
        // 1: 侧向加速度 ay = vy_dot + vx * r
        double ay = state_.vx * state_.r;
        obs[1] = static_cast<float>(std::clamp(ay / 10.0, -1.0, 1.0));
        // 2: 质心侧偏角 beta = vy / vx
        double beta = (std::abs(state_.vx) > 0.1) ? (state_.vy / state_.vx) : 0.0;
        obs[2] = static_cast<float>(std::clamp(beta / 0.2, -1.0, 1.0));
        // 3: 纵向车速差 (vx - 27.78)
        obs[3] = static_cast<float>(std::clamp((state_.vx - 27.78) / 10.0, -1.0, 1.0));
        // 4~7: 4 轮滑移率
        for (int i = 0; i < 4; ++i) {
            obs[4 + i] = static_cast<float>(std::clamp(state_.slip[i] / 0.5, -1.0, 1.0));
        }
        // 8~11: 4 轮角速度相对偏差
        double base_omega = state_.vx / p_.tire_radius;
        for (int i = 0; i < 4; ++i) {
            obs[8 + i] = static_cast<float>(std::clamp((state_.omega[i] - base_omega) / 10.0, -1.0, 1.0));
        }
        // 12: 侧向位移 y
        obs[12] = static_cast<float>(std::clamp(state_.y / 3.0, -1.0, 1.0));
        // 13: 航向角 psi
        obs[13] = static_cast<float>(std::clamp(state_.psi / 0.5, -1.0, 1.0));
        // 14: 爆胎突发传感器脉冲 (高频突变检测)
        obs[14] = state_.blowout_active ? 1.0f : 0.0f;
        // 15: 备用微分触觉通道
        obs[15] = static_cast<float>(std::clamp(ay * state_.r, -1.0, 1.0));
        return obs;
    }

    /**
     * 动力学积分单步
     * @param control_outputs: 8 维效应动作 (FL/FR/RL/RR 差动扭矩, delta补偿, 主动阻尼)
     * @param dt: 仿真步长 (s), 例如 0.0005s = 0.5ms (2000Hz 硬实时控制)
     */
    void step(const std::vector<float>& control_outputs, double dt = 0.0005) {
        sim_time_ += dt;

        // 爆胎触发: t >= 0.05s 左前轮爆胎
        if (sim_time_ >= state_.blowout_time && !state_.blowout_active) {
            state_.blowout_active = true;
        }

        // 解析效应控制
        // T_diff: 左右差动补偿力矩 (N*m)
        double torque_fl = (control_outputs.size() > 0) ? control_outputs[0] * 2500.0 : 0.0;
        double torque_fr = (control_outputs.size() > 1) ? control_outputs[1] * 2500.0 : 0.0;
        double torque_rl = (control_outputs.size() > 2) ? control_outputs[2] * 2500.0 : 0.0;
        double torque_rr = (control_outputs.size() > 3) ? control_outputs[3] * 2500.0 : 0.0;
        double steer_comp = (control_outputs.size() > 4) ? control_outputs[4] * 0.05 : 0.0; // 线控转向补偿 (rad)

        // 差动横摆控制力矩 M_z_control
        double M_z_control = (torque_fr + torque_rr - torque_fl - torque_rl) * (p_.track / 2.0) / p_.tire_radius;

        // 爆胎外加瞬态扰动
        double M_z_disturb = 0.0;
        double F_drag_disturb = 0.0;
        double cur_Cf = p_.Cf;
        if (state_.blowout_active) {
            // 左前轮爆胎: 剧烈增加左侧滚阻，且侧偏刚度崩解 80%
            M_z_disturb = -14500.0 * std::exp(-(sim_time_ - state_.blowout_time) * 2.0); // 瞬发 14.5k N*m 负向横摆力矩
            F_drag_disturb = 6500.0;
            cur_Cf = p_.Cf * 0.25;
        }

        // 轮胎侧偏角与侧偏力
        double beta = (std::abs(state_.vx) > 0.5) ? (state_.vy / state_.vx) : 0.0;
        double alpha_f = beta + p_.lf * state_.r / state_.vx - (state_.delta + steer_comp);
        double alpha_r = beta - p_.lr * state_.r / state_.vx;

        // Pacejka 魔术非线性饱和
        double Fyf = -cur_Cf * std::sin(1.65 * std::atan(0.8 * alpha_f));
        double Fyr = -p_.Cr * std::sin(1.65 * std::atan(0.8 * alpha_r));

        // 动力学微分方程:
        // m * (vy_dot + vx * r) = Fyf + Fyr
        // Iz * r_dot = lf * Fyf - lr * Fyr + M_z_disturb + M_z_control
        double vy_dot = (Fyf + Fyr) / p_.m - state_.vx * state_.r;
        double r_dot = (p_.lf * Fyf - p_.lr * Fyr + M_z_disturb + M_z_control) / p_.Iz;
        double vx_dot = -F_drag_disturb / p_.m;

        // 状态更新 (欧拉-辛积分)
        state_.vy += vy_dot * dt;
        state_.r += r_dot * dt;
        state_.vx = std::max(5.0, state_.vx + vx_dot * dt);

        state_.psi += state_.r * dt;
        state_.x += (state_.vx * std::cos(state_.psi) - state_.vy * std::sin(state_.psi)) * dt;
        state_.y += (state_.vx * std::sin(state_.psi) + state_.vy * std::cos(state_.psi)) * dt;

        // 记录极值
        peak_yaw_rate_ = std::max(peak_yaw_rate_, std::abs(state_.r));
        peak_lat_dev_ = std::max(peak_lat_dev_, std::abs(state_.y));

        // 评估横摆收敛时间 (从爆胎起，横摆角速度收敛到 +/- 0.03 rad/s 以内)
        if (state_.blowout_active && std::abs(state_.r) < 0.03 && sim_time_ > state_.blowout_time + 0.015) {
            if (settled_time_ < 0.0) {
                settled_time_ = (sim_time_ - state_.blowout_time) * 1000.0; // ms
            }
        }
    }

    const VehicleState& state() const { return state_; }
    double sim_time() const { return sim_time_; }
    double peak_yaw_rate() const { return peak_yaw_rate_; }
    double peak_lat_dev() const { return peak_lat_dev_; }
    double settled_time_ms() const { return settled_time_ > 0 ? settled_time_ : (sim_time_ - state_.blowout_time) * 1000.0; }

private:
    VehicleParams p_;
    VehicleState state_;
    double sim_time_{0.0};
    double peak_yaw_rate_{0.0};
    double peak_lat_dev_{0.0};
    double settled_time_{-1.0};
};

} // namespace kun::adas
