#pragma once
// ============================================================================
// population/population_eval.hpp — L1 系统层: 评估委托接口
// 任务层 (L2) 实现: 输入个体, 输出逐日 returns 序列与标量适应度。
// L1 只消费该接口, 不关心领域 (市场/牌局/轨迹皆是合法实例)。
// ============================================================================
#include <functional>
#include <vector>

namespace kun {
namespace population {

struct EvaluationResult {
    std::vector<double> daily_returns;  // 时间对齐的逐期收益序列 (池评估原语输入)
    double fitness = 0.0;               // 岛内代际选择用的标量适应度
};

template <typename Individual>
using PopulationEval = std::function<EvaluationResult(const Individual&)>;

}  // namespace population
}  // namespace kun
