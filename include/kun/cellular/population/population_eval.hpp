#pragma once
// ============================================================================
// population/population_eval.hpp — L1 系统层: 评估委托契约
//
// 任务层 (L2) 实现两选一:
//   (a) 逐个体: Individual& -> double            (IndividualEvaluator)
//   (b) 批量:   BatchView<Individual> -> void     (BatchEvaluator)
// L1 只消费该契约, 不关心领域 (市场/牌局/轨迹皆是合法实例)。
//
// 设计说明:
//   - 采用 C++20 concept + 编译期分派 (if constexpr), 不采用 std::function:
//     类型擦除阻止内联, 且逐个体返回 vector (EvaluationResult) = 每次评估一次堆分配,
//     这正是旧 PopulationEval seam 至今无消费者的原因 (仅作源兼容保留)。
//   - BatchView 用指针 span: 待评估槽位非连续 (精英与后代交错), 避免 gather/拷贝。
//   - 确定性纪律: 批量评估委托必须对固定输入顺序确定性; L1 恒以固定升序槽位呈现。
// ============================================================================
#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace kun {
namespace population {

// ── legacy 源兼容 (非热路径; 保留供既有 using/文档引用) ────────────────────────
struct EvaluationResult {
    std::vector<double> daily_returns;  // 时间对齐的逐期收益序列 (池评估原语输入)
    double fitness = 0.0;               // 岛内代际选择用的标量适应度
};

template <typename Individual>
using PopulationEval = std::function<EvaluationResult(const Individual&)>;

// ── 批量评估视图 (系统层 seam) ────────────────────────────────────────────────
// individuals: 本批待评估个体 (可写: 评估可能更新个体内部运行时状态, 如微柱前向)
// fitness_out: 与 individuals 等长; 任务逐位写入适应度
// batch_seq:   单调递增批号 (遥测/调试; 不参与语义)
template <typename Individual>
struct BatchView {
    std::span<Individual*> individuals;
    std::span<double> fitness_out;
    uint64_t batch_seq = 0;
};

// ── 概念: 逐个体评估委托 ──────────────────────────────────────────────────────
template <typename Eval, typename Individual>
concept IndividualEvaluator = requires(Eval& e, Individual& v) {
    { e(v) } -> std::convertible_to<double>;
};

// ── 概念: 批量评估委托 ────────────────────────────────────────────────────────
template <typename Eval, typename Individual>
concept BatchEvaluator = requires(Eval& e, BatchView<Individual> b) {
    { e(b) };
};

// ── 适配器: 把逐个体委托包成批量委托 (等价性测试 / 旧代码平滑迁移) ─────────────
template <typename Individual, typename F>
    requires IndividualEvaluator<F, Individual>
auto make_batch_adapter(F f) {
    return [f = std::move(f)](BatchView<Individual> b) mutable {
        for (size_t j = 0; j < b.individuals.size(); ++j) {
            b.fitness_out[j] = static_cast<double>(f(*b.individuals[j]));
        }
    };
}

}  // namespace population
}  // namespace kun
