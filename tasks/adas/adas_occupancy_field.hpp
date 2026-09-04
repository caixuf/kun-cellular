#pragma once

/**
 * ============================================================================
 * KunCellular ADAS 具身场景外围适配器 (10M 级: 连续动态时空占用网格与流场推演)
 * 场景: 360° 环视连续动态占据场、多动态障碍物轨迹流变、遮挡盲区概率弥散
 * ============================================================================
 */

#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <algorithm>

namespace kun::adas {

struct DynamicObstacle {
    float x;      // 相对本车 X (m)
    float y;      // 相对本车 Y (m)
    float vx;     // 相对速度 X (m/s)
    float vy;     // 相对速度 Y (m/s)
    float length; // 几何长 (m)
    float width;  // 几何宽 (m)
    float heading;// 航向角 (rad)
};

class DynamicOccupancyHabitat {
public:
    static constexpr int BEAMS = 256;         // 256 环视感知受体波束
    static constexpr int SECTORS = 64;        // 64 关键扇区效应评估
    static constexpr float RANGE_MAX = 64.0f; // 探测半径 64 米 (128m x 128m 视野)

    DynamicOccupancyHabitat(uint32_t seed = 42) {
        reset(seed);
    }

    void reset(uint32_t seed = 42) {
        rng_.seed(seed);
        sim_time_ = 0.0f;
        obstacles_.clear();

        // 构造典型城市场景中 6 个高动态交互目标
        // 1. 对向左转抢行车 (高碰撞风险)
        obstacles_.push_back({28.0f, 12.0f, -14.0f, -4.0f, 4.8f, 2.0f, 3.4f});
        // 2. 侧前方同向切入加塞车
        obstacles_.push_back({18.0f, -4.5f, 4.0f, 1.8f, 4.5f, 1.9f, 0.2f});
        // 3. 盲区突然窜出的外卖非机动车 (极速横穿)
        obstacles_.push_back({8.0f, 14.0f, 0.5f, -8.5f, 1.8f, 0.8f, -1.5f});
        // 4. 正前方同速巡航前车
        obstacles_.push_back({35.0f, 0.0f, 0.0f, 0.0f, 4.9f, 2.1f, 0.0f});
        // 5. 左侧对向直行快速货车 (产生大面积视线遮挡)
        obstacles_.push_back({45.0f, 7.0f, -22.0f, 0.0f, 9.0f, 2.5f, 3.14f});
        // 6. 右后方高速超车车辆
        obstacles_.push_back({-25.0f, -7.0f, 12.0f, 0.5f, 4.6f, 2.0f, 0.05f});
    }

    /**
     * 生成 256 维环视感知受体输入 (角度与距离相关之连续反射占有强度)
     */
    std::vector<float> generate_observation() const {
        std::vector<float> obs(BEAMS, 0.0f);
        const float d_angle = 2.0f * M_PI / BEAMS;

        for (int b = 0; b < BEAMS; ++b) {
            float angle = -M_PI + b * d_angle;
            float ray_dir_x = std::cos(angle);
            float ray_dir_y = std::sin(angle);

            float min_dist = RANGE_MAX;
            float dop_vel = 0.0f;

            for (const auto& obs_obj : obstacles_) {
                // 射线与障碍物包围盒最近交点距离估算
                float dx = obs_obj.x;
                float dy = obs_obj.y;
                float dist_obj = std::sqrt(dx * dx + dy * dy);
                float angle_obj = std::atan2(dy, dx);
                float angle_diff = std::remainder(angle - angle_obj, 2.0f * M_PI);

                float obj_angular_span = std::atan2(obs_obj.length * 0.5f, dist_obj);
                if (std::abs(angle_diff) < obj_angular_span && dist_obj < min_dist) {
                    min_dist = dist_obj;
                    // 多普勒径向相对速度
                    dop_vel = (obs_obj.vx * ray_dir_x + obs_obj.vy * ray_dir_y) / 25.0f;
                }
            }

            if (min_dist < RANGE_MAX) {
                // 距离反比占有强度 + 多普勒调制
                float occ_intensity = (1.0f - min_dist / RANGE_MAX);
                obs[b] = std::clamp(occ_intensity + dop_vel * 0.2f, -1.0f, 1.0f);
            }
        }
        return obs;
    }

    /**
     * 场景演化步进与真值地平线前瞻计算
     * @param dt: 秒 (如 0.02s = 50Hz)
     */
    void step(float dt = 0.02f) {
        sim_time_ += dt;
        for (auto& o : obstacles_) {
            o.x += o.vx * dt;
            o.y += o.vy * dt;
        }
    }

    /**
     * 计算真实世界在 future_seconds 后的 64 扇区危险度基准真值 (Ground Truth)
     */
    std::vector<float> compute_future_ground_truth(float future_seconds) const {
        std::vector<float> gt(SECTORS, 0.0f);
        const float d_sec = 2.0f * M_PI / SECTORS;

        for (const auto& o : obstacles_) {
            float future_x = o.x + o.vx * future_seconds;
            float future_y = o.y + o.vy * future_seconds;
            float future_dist = std::sqrt(future_x * future_x + future_y * future_y);

            if (future_dist < RANGE_MAX) {
                float future_angle = std::atan2(future_y, future_x);
                int sec_idx = static_cast<int>((future_angle + M_PI) / d_sec) % SECTORS;
                if (sec_idx < 0) sec_idx += SECTORS;

                float threat = (1.0f - future_dist / RANGE_MAX);
                gt[sec_idx] = std::max(gt[sec_idx], threat);
            }
        }
        return gt;
    }

    float sim_time() const { return sim_time_; }

private:
    std::mt19937 rng_;
    float sim_time_{0.0f};
    std::vector<DynamicObstacle> obstacles_;
};

} // namespace kun::adas
