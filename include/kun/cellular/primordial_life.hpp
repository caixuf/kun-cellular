#pragma once

// ============================================================================
// PrimordialLife — 无目标原始进化生命体 (Objective-Free Primordial viability)
//
// 设计原则 (呼应宪章"智能分层"与门禁2):
// 1. 零外部目标: 没有地图、没有收益曲线、没有业务契约。环境只有一个物理事实——
//    熵涨落场 (Entropy Flux Field): 四通道连续随机能量注入，且强度随时间指数爬升。
// 2. 唯一选择压力 = 生存 (Viability): 有机体的输出电位若发散超越活力界限
//    (Vital Limit) 或产生 NaN，即判定代谢崩溃死亡。适应度 = 存活步数比例。
// 3. 因此演化能发现的唯一"智慧"是内稳态 (Homeostasis): 连续物理阻尼、
//    负反馈门控、耗散结构。这正是 SDSCC 底座 19ns 无抖动确定性的生物学根源。
// 4. 复用 EvolvableTask 标准契约 (obs→forward→step_continuous→fitness)，
//    零修改接入 MorphogeneticEvolutionEngine 的变异/选择/凋亡算子。
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace kun {

class PrimordialViabilityTask : public EvolvableTask {
public:
    explicit PrimordialViabilityTask(double base_pressure = 0.05,
                                     double pressure_growth = 1.004,
                                     double vital_limit = 8.0)
        : base_pressure_(base_pressure),
          pressure_growth_(pressure_growth),
          vital_limit_(vital_limit) {}

    const char* name() const override { return "PrimordialViability"; }
    size_t obs_dim() const override { return 4; }
    size_t act_dim() const override { return 4; }

    void set_max_steps(int steps) { max_steps_ = steps; }

    // 每代抬升基础熵压 (红皇后: 环境越来越狂暴，幸存者必须越来越稳)
    void set_base_pressure(double p) { base_pressure_ = p; }
    double base_pressure() const { return base_pressure_; }

    void reset(uint32_t episode_seed) override {
        rng_.seed(episode_seed);
        flux_ = {0.0f, 0.0f, 0.0f, 0.0f};
        pressure_ = base_pressure_;
        steps_ = 0;
        died_ = false;
        energy_sum_ = 0.0;
        peak_energy_ = 0.0;
    }

    std::vector<float> current_observation() const override {
        return {flux_[0], flux_[1], flux_[2], flux_[3]};
    }

    // 离散动作接口合规实现 (本任务纯连续，仅作通道转发)
    StepResult step(int action) override {
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = static_cast<double>(action) - 1.5;
        return step_continuous(acts);
    }

    // 核心一步: 有机体输出电位 → 活力判定 → 熵场演化
    StepResult step_continuous(const CellularOrganism::ActionOutputs& acts) override {
        StepResult res;

        // --- 1. 活力判定 (唯一死亡法则) ---
        const double energy = std::max({std::fabs(acts.positive_action),
                                        std::fabs(acts.negative_action),
                                        std::fabs(acts.defensive_reset)});
        const bool nan_collapse = !std::isfinite(acts.positive_action) ||
                                  !std::isfinite(acts.negative_action) ||
                                  !std::isfinite(acts.defensive_reset);
        if (nan_collapse || energy > vital_limit_) {
            died_ = true;
            res.done = true;
            res.reward = 0.0;
        } else {
            ++steps_;
            res.reward = 1.0;  // 存活本身就是全部回报
            energy_sum_ += energy;
            peak_energy_ = std::max(peak_energy_, energy);
        }
        res.steps = steps_;
        res.success = false;  // 无目标世界没有"达成"，只有"仍在"

        // --- 2. 熵场演化: OU 过程 + 指数爬升幅度 ---
        pressure_ *= pressure_growth_;
        for (size_t i = 0; i < 4; ++i) {
            const double drift = -ou_theta_ * flux_[i];
            const double shock = sigma_scale_ * pressure_ * gauss_(rng_);
            flux_[i] = static_cast<float>(flux_[i] + drift + shock);
        }
        res.obs.assign(flux_.begin(), flux_.end());
        return res;
    }

    // 适应度 = 存活比例 ∈ [0,1]。无任何外部目标加成。
    double current_fitness() const override {
        return static_cast<double>(steps_) / static_cast<double>(max_steps_);
    }

    // --- 诊断 (不进入适应度) ---
    bool last_died() const { return died_; }
    double last_mean_energy() const {
        return steps_ > 0 ? energy_sum_ / static_cast<double>(steps_) : 0.0;
    }
    double last_peak_energy() const { return peak_energy_; }
    int max_steps() const { return max_steps_; }

private:
    std::mt19937 rng_{777};
    std::normal_distribution<double> gauss_{0.0, 1.0};
    std::array<float, 4> flux_{};

    double base_pressure_{0.05};   // 代际红皇后基础熵压
    double pressure_{0.05};        // 当前回合瞬时熵压
    double pressure_growth_{1.004};// 回合内每步熵压乘数 (1.004^400 ≈ 4.9x)
    double vital_limit_{8.0};      // 活力界限: 输出电位发散阈值
    double ou_theta_{0.05};        // OU 均值回归速率
    double sigma_scale_{0.6};      // 熵冲击尺度

    int steps_{0};
    int max_steps_{400};
    bool died_{false};
    double energy_sum_{0.0};
    double peak_energy_{0.0};
};

} // namespace kun
