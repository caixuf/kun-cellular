#pragma once

/**
 * ============================================================================
 * Software-Defined Silicon Cellular Computer (SDSCC)
 * 256 通道 3D 湍流物理场具身生境 (3D Continuous Turbulence Habitat)
 * ============================================================================
 * 
 * 体系结构突破 (M3 规范):
 * 1. 256 通道超宽空间感知: 8x8x4 三维流变网格体素
 * 2. 连续 3D 偏微分方程 (PDE): 反应扩散 + 柯尔莫哥洛夫局部湍流孤立波
 * 3. 物理守恒与能量阻尼: 生命体必须自发形成图灵时空阻尼场，抑制相空间湍流爆炸
 * 4. 彻底展现百万/千万硅基细胞在超高维连续物理世界的涌现统治力
 */

#include "kun/cellular/evolvable_task.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace kun {

class Field3DTurbulenceTask : public EvolvableTask {
public:
    static constexpr int DIM_X = 8;
    static constexpr int DIM_Y = 8;
    static constexpr int DIM_Z = 4;
    static constexpr size_t NUM_VOXELS = DIM_X * DIM_Y * DIM_Z; // 256 通道

    Field3DTurbulenceTask(double diffusion = 0.20, double r = 3.65)
        : diffusion_(diffusion), r_(r) {
        field_.assign(NUM_VOXELS, 0.0f);
        next_field_.assign(NUM_VOXELS, 0.0f);
    }

    const char* name() const override { return "Field3DTurbulence"; }
    size_t obs_dim() const override { return NUM_VOXELS; } // 256 维空间感知
    size_t act_dim() const override { return 4; }          // 4 维主控效应器

    void reset(uint32_t seed) override {
        rng_.seed(seed);
        std::uniform_real_distribution<float> dist(-0.8f, 0.8f);
        for (size_t i = 0; i < NUM_VOXELS; ++i) {
            field_[i] = dist(rng_);
        }
        steps_ = 0;
        q_sum_ = 0.0;
    }

    std::vector<float> current_observation() const override {
        return field_;
    }

    inline size_t idx(int x, int y, int z) const {
        return static_cast<size_t>((z * DIM_Y + y) * DIM_X + x);
    }

    StepResult step(int) override {
        CellularOrganism::ActionOutputs a;
        return step_continuous(a);
    }

    /**
     * @brief 单步连续 3D 偏微分方程演化 + 硅基生命体空间阻尼干涉
     */
    StepResult step_continuous(const CellularOrganism::ActionOutputs& actions) override {
        StepResult res;
        ++steps_;

        // 1. 三维拉普拉斯扩散 (7 点有限差分算子) + 混沌非线性映射
        for (int z = 0; z < DIM_Z; ++z) {
            int zm = (z > 0) ? (z - 1) : (DIM_Z - 1);
            int zp = (z < DIM_Z - 1) ? (z + 1) : 0;
            for (int y = 0; y < DIM_Y; ++y) {
                int ym = (y > 0) ? (y - 1) : (DIM_Y - 1);
                int yp = (y < DIM_Y - 1) ? (y + 1) : 0;
                for (int x = 0; x < DIM_X; ++x) {
                    int xm = (x > 0) ? (x - 1) : (DIM_X - 1);
                    int xp = (x < DIM_X - 1) ? (x + 1) : 0;

                    float center = field_[idx(x, y, z)];
                    float neighbors = field_[idx(xm, y, z)] + field_[idx(xp, y, z)] +
                                      field_[idx(x, ym, z)] + field_[idx(x, yp, z)] +
                                      field_[idx(x, y, zm)] + field_[idx(x, y, zp)];

                    float laplacian = (neighbors - 6.0f * center) / 6.0f;
                    float local_dyn = 1.0f - static_cast<float>(r_) * center * center;
                    float val = (1.0f - static_cast<float>(diffusion_)) * local_dyn + 
                                static_cast<float>(diffusion_) * laplacian;

                    next_field_[idx(x, y, z)] = std::clamp(val, -2.0f, 2.0f);
                }
            }
        }

        // 2. 注入生命体的主动空间阻尼效应
        float damp_pos = static_cast<float>(actions.positive_action);
        float damp_neg = static_cast<float>(actions.negative_action);

        // 分布式注入流场核心特征区
        for (size_t a = 0; a < 16; ++a) {
            next_field_[a] -= damp_pos * 0.25f;
            next_field_[NUM_VOXELS - 1 - a] += damp_neg * 0.25f;
        }

        field_ = next_field_;

        // 3. 计算场内总李雅普诺夫涡动能 E = sum(u^2)
        float field_energy = 0.0f;
        for (float v : field_) field_energy += v * v;
        field_energy /= static_cast<float>(NUM_VOXELS);

        // 目标: 抑制高能湍流暴发，将场稳定在低能平滑态
        double score = std::exp(-static_cast<double>(field_energy) / 0.50);
        q_sum_ += score;

        res.obs = field_;
        res.reward = score;
        res.done = (steps_ >= max_steps_);
        res.success = (field_energy < 0.35f);
        res.steps = steps_;
        return res;
    }

    double current_fitness() const override {
        return steps_ > 0 ? q_sum_ / static_cast<double>(steps_) : 0.0;
    }

    void set_max_steps(int ms) { max_steps_ = ms; }

private:
    double diffusion_{0.20};
    double r_{3.65};
    int max_steps_{80};
    int steps_{0};
    double q_sum_{0.0};
    std::vector<float> field_;
    std::vector<float> next_field_;
    std::mt19937 rng_;
};

} // namespace kun
