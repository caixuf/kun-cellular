#pragma once

/**
 * ============================================================================
 * KunCellular ADAS 具身场景外围适配器 (100M+ 级: 全息 4D 时空连续体素与因果反事实推演)
 * 场景: 3D 几何立体空间体素、遮挡盲区反事实分支生成、长程多智能体时空博弈 (5s~8s)
 * ============================================================================
 */

#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <algorithm>

namespace kun::adas {

struct Voxel3DPoint {
    float x, y, z;
    float occupancy;
    float velocity_x;
    float velocity_y;
};

struct CounterfactualBranch {
    std::string hypothesis_name;
    float intrusion_probability; // 盲区突发侵入先验概率
    float target_speed;          // 侵入者假设速度 (m/s)
    float time_to_collision;     // 反事实碰撞时间 TTC (s)
    float risk_severity;         // 碰撞势能严重度 [0, 1]
};

class Holographic4DWorldModelHabitat {
public:
    static constexpr int VOXEL_RECEPTORS = 1024; // 1024 维 3D 空间立体受体通道 (32x16x2)
    static constexpr int EFFECTOR_DIM = 128;     // 128 维全息因果决策与博弈效应张量

    Holographic4DWorldModelHabitat(uint32_t seed = 2026) {
        reset(seed);
    }

    void reset(uint32_t seed = 2026) {
        rng_.seed(seed);
        sim_time_ = 0.0f;
        branches_.clear();

        // 构造三类经典反事实推演分支:
        // 分支 0: 标称无风险分支 (大货车旁盲区无行人)
        branches_.push_back({"Nominal_Clear", 0.65f, 0.0f, 99.0f, 0.0f});
        // 分支 1: 极端鬼探头分支 (大货车盲区 1.8s 后冲出加速儿童)
        branches_.push_back({"Blindspot_Pedestrian_Intrusion", 0.25f, 4.5f, 2.1f, 0.95f});
        // 分支 2: 对向盲区车辆左转强切分支
        branches_.push_back({"Aggressive_Unprotected_Turn", 0.10f, 11.0f, 3.2f, 0.80f});
    }

    /**
     * 生成 1024 维 3D 空间连续体素观测张量
     * 空间拓扑: X 轴 [-32m, 32m], Y 轴 [0m, 64m], Z 轴 [-1m, 3m] (分层体素)
     */
    std::vector<float> generate_3d_voxel_observation() const {
        std::vector<float> obs(VOXEL_RECEPTORS, 0.0f);
        
        // 模拟大货车在左前方产生的立体几何遮挡体 (长 12m, 宽 2.8m, 高 3.6m)
        // 以及道路右侧绿化隔离带立体体素
        for (int i = 0; i < VOXEL_RECEPTORS; ++i) {
            int z_idx = (i / (32 * 16)) % 2;
            int y_idx = (i / 32) % 16;
            int x_idx = i % 32;

            float x = (x_idx - 16.0f) * 2.0f;
            float y = y_idx * 4.0f;
            float z = z_idx * 2.0f; (void)z;

            // 大车实体占据
            if (x >= -6.0f && x <= -3.0f && y >= 15.0f && y <= 27.0f) {
                obs[i] = 0.90f + 0.08f * std::sin(sim_time_ * 5.0f + x);
            }
            // 大车右侧视线盲区体素 (低可见度，高反事实不确定性)
            else if (x >= -2.5f && x <= 0.0f && y >= 18.0f && y <= 24.0f) {
                obs[i] = -0.75f; // 负值表征高信息熵未探测盲区
            }
            // 道路开阔区
            else {
                obs[i] = 0.05f * std::cos(x * 0.2f + y * 0.1f);
            }
        }
        return obs;
    }

    /**
     * 评估 128 维全息反事实输出与长程博弈对账
     */
    struct CounterfactualEvaluation {
        float blindspot_risk_awareness;     // 盲区风险觉醒度 [0, 1]
        float prediction_horizon_seconds;   // 稳定推演时空地平线 (s)
        float counterfactual_entropy;       // 多分支反事实熵
        float longitudinal_decel_intent;    // 提前防御性制动势能 (m/s^2)
        bool safety_margin_preserved;       // 是否提前建立安全博弈裕度
    };

    CounterfactualEvaluation evaluate_effectors(const float* effectors, size_t dim) const {
        CounterfactualEvaluation eval{};
        if (dim < EFFECTOR_DIM || !effectors) return eval;

        // effectors[0..31]: 盲区因果分支能量响应
        float blindspot_energy = 0.0f;
        for (int i = 0; i < 32; ++i) {
            blindspot_energy += std::abs(effectors[i]);
        }
        eval.blindspot_risk_awareness = std::clamp(blindspot_energy / 16.0f, 0.0f, 1.0f);

        // effectors[32..63]: 长程推演时间地平线 (预测衰减特征)
        float horizon_energy = 0.0f;
        for (int i = 32; i < 64; ++i) {
            horizon_energy += std::max(0.0f, effectors[i]);
        }
        eval.prediction_horizon_seconds = 5.0f + std::clamp(horizon_energy * 0.1f, 0.0f, 3.5f); // 5.0s ~ 8.5s

        // effectors[64..95]: 反事实多模态熵
        float var_sum = 0.0f;
        for (int i = 64; i < 96; ++i) {
            var_sum += effectors[i] * effectors[i];
        }
        eval.counterfactual_entropy = std::sqrt(var_sum / 32.0f);

        // effectors[96..127]: 宏观博弈与提前减速梯度
        eval.longitudinal_decel_intent = std::clamp(effectors[96] * 4.5f, 0.0f, 6.0f);
        eval.safety_margin_preserved = (eval.blindspot_risk_awareness > 0.35f && eval.longitudinal_decel_intent > 1.2f);

        return eval;
    }

    void step(float dt = 0.02f) {
        sim_time_ += dt;
    }

    float sim_time() const { return sim_time_; }
    const std::vector<CounterfactualBranch>& branches() const { return branches_; }

private:
    std::mt19937 rng_;
    float sim_time_{0.0f};
    std::vector<CounterfactualBranch> branches_;
};

} // namespace kun::adas
