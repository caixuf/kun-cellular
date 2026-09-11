#pragma once

// ============================================================================
// LocomotionGaitTask — 多足步态形态发生任务 (organism 真前向驱动)
//
// 目的: 把「软体肌腱/多足步态」演示接入底座真前向 —— 肌肉目标长度由演化出的
//       CellularOrganism 连续输出驱动, 而非脚本化正弦。
//
// 物理: 4 质点 (躯干前/后 + 左/右足) + 4 阻尼线性肌肉弹簧; 重力 + 地面库仑摩擦。
//       muscle_k.target_len = rest_k * (1 + amp * a_k)，a_k 由 organism 输出并 clamp[-1,1]。
//
// 观测 (10 维): 躯干位置/速度 + 双足离地高度 + 4 肌肉长度比
// 动作 (4 维): positive_action / negative_action / defensive_reset / predicted_sense_0
// 成功: 前进位移 >= target_dist (默认 120px) 且未坍塌
// OOD 变体: 更强重力 / 更低摩擦 (跨物理参数泛化)
// ============================================================================

#include "kun/cellular/evolvable_task.hpp"
#include <cmath>
#include <random>

namespace kun {

class LocomotionGaitTask : public EvolvableTask {
public:
    struct Params {
        double gravity{0.30};
        double damping{0.96};
        double ground_friction{0.25};
        double spring_k{0.18};
        double amp{0.25};
        double ground_y{380.0};
        double x_base{80.0};
        double target_dist{80.0};
        double fall_y{358.0}; // 躯干点低于此高度视为坍塌
    };

    LocomotionGaitTask() : P_() { init_body(); }
    explicit LocomotionGaitTask(const Params& p) : P_(p) { init_body(); }

    const char* name() const override { return "LocomotionGait"; }
    size_t obs_dim() const override { return 10; }
    size_t act_dim() const override { return 4; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        init_body();
    }

    std::vector<float> current_observation() const override {
        const size_t nc = std::min(kNumNodes, kNumMuscles + 1);
        std::vector<float> obs(obs_dim(), 0.0f);
        obs[0] = static_cast<float>(nx_[0] / 200.0);
        obs[1] = static_cast<float>(ny_[0] / 400.0);
        obs[2] = static_cast<float>(nvx_[0] / 10.0);
        obs[3] = static_cast<float>(nvy_[0] / 10.0);
        obs[4] = static_cast<float>((ny_[2] - P_.ground_y) / 50.0);
        obs[5] = static_cast<float>((ny_[3] - P_.ground_y) / 50.0);
        for (size_t m = 0; m < kNumMuscles && 6 + m < obs.size(); ++m) {
            const double len = muscle_length(m);
            obs[6 + m] = static_cast<float>(len / muscles_[m].rest - 1.0);
        }
        (void)nc;
        return obs;
    }

    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = (action == 0) ? 1.0 : ((action == 1) ? -1.0 : 0.0);
        acts.negative_action = (action == 2) ? 1.0 : ((action == 3) ? -1.0 : 0.0);
        return step_continuous(acts);
    }

    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        double a[kNumMuscles];
        a[0] = clamp1(acts.positive_action);
        a[1] = clamp1(acts.negative_action);
        a[2] = clamp1(acts.defensive_reset);
        a[3] = clamp1(acts.predicted_sense_0);
        for (size_t m = 0; m < kNumMuscles; ++m) {
            if (!std::isfinite(a[m])) a[m] = 0.0;
        }

        // 1. 肌肉弹簧力 (以速度冲量近似, 与既有演示物理一致)
        for (size_t m = 0; m < kNumMuscles; ++m) {
            const int i = muscles_[m].n1, j = muscles_[m].n2;
            double dx = nx_[j] - nx_[i], dy = ny_[j] - ny_[i];
            double d = std::sqrt(dx * dx + dy * dy) + 1e-6;
            const double target = muscles_[m].rest * (1.0 + P_.amp * a[m]);
            const double f = (d - target) * P_.spring_k;
            const double ux = dx / d, uy = dy / d;
            nvx_[i] += ux * f; nvy_[i] += uy * f;
            nvx_[j] -= ux * f; nvy_[j] -= uy * f;
        }

        // 2. 重力 + 积分 + 阻尼 + 地面接触
        double min_top_y = 1e9;
        for (size_t n = 0; n < kNumNodes; ++n) {
            nvy_[n] += P_.gravity;
            nx_[n] += nvx_[n];
            ny_[n] += nvy_[n];
            nvx_[n] *= P_.damping;
            nvy_[n] *= P_.damping;
            if (ny_[n] >= P_.ground_y) {
                ny_[n] = P_.ground_y;
                if (nvy_[n] > 0.0) nvy_[n] = 0.0;
                nvx_[n] *= P_.ground_friction;
            }
            if (n < 2) min_top_y = std::min(min_top_y, ny_[n]);
        }

        ++steps_;
        const double disp = nx_[0] - P_.x_base;
        if (disp > best_disp_) best_disp_ = disp;
        const bool diverged = !std::isfinite(nx_[0]) || !std::isfinite(ny_[0]);
        const bool fell = min_top_y > P_.fall_y;
        const bool done = diverged || fell || steps_ >= max_steps_;

        StepResult res;
        res.obs = current_observation();
        res.steps = steps_;
        res.done = done;
        res.success = (best_disp_ >= P_.target_dist) && !diverged;
        res.reward = fell ? 0.0 : 1.0;
        res.min_dist_to_goal = std::max(0.0, P_.target_dist - best_disp_);
        return res;
    }

    double current_fitness() const override {
        const double progress = std::min(1.0, std::max(0.0, (best_disp_) / P_.target_dist));
        return progress;
    }

    void set_max_steps(int s) { max_steps_ = s; }
    int max_steps() const { return max_steps_; }
    size_t num_nodes() const { return kNumNodes; }
    size_t num_muscles() const { return kNumMuscles; }
    double node_x(size_t i) const { return nx_[i]; }
    double node_y(size_t i) const { return ny_[i]; }
    int muscle_n1(size_t m) const { return muscles_[m].n1; }
    int muscle_n2(size_t m) const { return muscles_[m].n2; }
    double muscle_rest(size_t m) const { return muscles_[m].rest; }
    double best_disp() const { return best_disp_; }

    static constexpr size_t kNumNodes = 4;
    static constexpr size_t kNumMuscles = 4;

private:
    struct Muscle { int n1, n2; double rest; };

    static double clamp1(double v) { return std::max(-1.0, std::min(1.0, v)); }

    void init_body() {
        // 躯干前(0)/后(1) + 左足(2)/右足(3)
        nx_[0] = P_.x_base + 40.0; ny_[0] = 300.0;
        nx_[1] = P_.x_base;        ny_[1] = 300.0;
        nx_[2] = P_.x_base;        ny_[2] = 350.0;
        nx_[3] = P_.x_base + 40.0; ny_[3] = 350.0;
        for (size_t n = 0; n < kNumNodes; ++n) { nvx_[n] = 0.0; nvy_[n] = 0.0; }
        muscles_[0] = {0, 1, 40.0}; // 躯干
        muscles_[1] = {1, 2, 50.0}; // 后-左足
        muscles_[2] = {0, 3, 50.0}; // 前-右足
        muscles_[3] = {1, 3, 64.0}; // 对角
        steps_ = 0;
        best_disp_ = 0.0;
    }

    double muscle_length(size_t m) const {
        const int i = muscles_[m].n1, j = muscles_[m].n2;
        const double dx = nx_[j] - nx_[i], dy = ny_[j] - ny_[i];
        return std::sqrt(dx * dx + dy * dy);
    }

    Params P_;
    std::mt19937 rng_{42};
    double nx_[kNumNodes]{}, ny_[kNumNodes]{}, nvx_[kNumNodes]{}, nvy_[kNumNodes]{};
    Muscle muscles_[kNumMuscles]{};
    int steps_{0};
    int max_steps_{200};
    double best_disp_{0.0};
};

} // namespace kun
