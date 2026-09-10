#pragma once
// ============================================================================
// population/adversarial_course.hpp — L1 系统层: 对抗性压力课程 (捕食者调度)
// 概念吸收自底座 AdversarialStressProfile 四级分级。L1 只做"何时/多强"的调度,
// 压力算子本体由任务层注册 (L2 注入可调用对象, L1 不定义任何业务冲击公式)。
// ============================================================================
#include <cmath>
#include <cstdint>
#include <random>

namespace kun {
namespace population {

enum class PressureLevel : uint32_t { OFF = 0, LOW = 1, MEDIUM = 2, EXTREME = 3 };

struct PressurePoint {
    double trigger_prob{0.0};   // 每次评估前压力事件触发概率
    double intensity{0.0};      // 事件强度系数 (无量纲; 语义由任务层定义)
};

inline PressurePoint make_pressure_point(PressureLevel level) {
    switch (level) {
        case PressureLevel::LOW:    return PressurePoint{0.02, 1.0};
        case PressureLevel::MEDIUM: return PressurePoint{0.05, 2.0};
        case PressureLevel::EXTREME:return PressurePoint{0.10, 4.0};
        case PressureLevel::OFF:
        default:                    return PressurePoint{0.0, 0.0};
    }
}

class AdversarialCourse {
public:
    void set_level(PressureLevel level) {
        level_ = level;
        point_ = make_pressure_point(level);
    }

    PressureLevel level() const { return level_; }
    const PressurePoint& point() const { return point_; }

    // 是否在本代触发压力事件 (确定性: 随机源显式传入)
    bool fire(std::mt19937& rng) const {
        if (level_ == PressureLevel::OFF) return false;
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return u(rng) < point_.trigger_prob;
    }

private:
    PressureLevel level_{PressureLevel::OFF};
    PressurePoint point_{};
};

}  // namespace population
}  // namespace kun
