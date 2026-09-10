#pragma once
// ============================================================================
// population/individual_traits.hpp — L1 系统层: 个体遗传算子概念与特化
//
// 系统层纪律 (docs/population_ecology_v1_design.md):
//   - 仅消费 L0 底座公开 API, 零修改 L0
//   - 领域无关: 无任何业务名词
//   - 确定性: 全部随机源显式传入 (std::mt19937), 同种子位级可复现
//   - 训练离线使用, 运行时前向链路不感知本层
// ============================================================================
#include "kun/cellular/cortical_column.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace kun {
namespace population {

// ── 变异算子概念: 个体必须提供 mutate(rate, sigma, rng) ──────────────────────
// CorticalMacroArray 已满足 (cortical_column.hpp:239)。

// ── 杂交算子 traits: 默认无杂交 (克隆父 A); 由 L1 为具体底座类型提供特化 ──────
template <typename Individual>
struct CrossoverTrait {
    // 默认: 有性重组不存在时退化为克隆父 A (语义: 单亲繁殖)
    static Individual cross(const Individual& a, const Individual& /*b*/, std::mt19937& /*rng*/) {
        return a;
    }
};

// ── CorticalMacroArray 列级有性杂交特化 ──────────────────────────────────────
// 语义: 子代逐列取父 A 或父 B (50/50), 每列保持完整合法的编译微柱 (不打碎列内突触);
//       长程轴突按子代列拓扑重新布线; 列内寄存器状态归零 (运行时状态不遗传)。
template <>
struct CrossoverTrait<CorticalMacroArray> {
    static CorticalMacroArray cross(const CorticalMacroArray& a, const CorticalMacroArray& b,
                                    std::mt19937& rng) {
        // 架构兼容性校验: 两亲本必须是同构阵列 (列数/每列细胞数一致)
        const auto& ac = a.columns();
        const auto& bc = b.columns();
        if (ac.size() != bc.size() || ac.empty()) {
            return a;  // 不同构: 退化为克隆父 A (诚实降级, 不产生非法子代)
        }
        const size_t cells_per_col = ac[0].genome.num_cells;
        for (size_t c = 0; c < ac.size(); ++c) {
            if (bc[c].genome.num_cells != cells_per_col) return a;
        }

        CorticalMacroArray child = a;  // 结构拷贝 (含全部列与轴突)
        auto& cc = child.columns();
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        for (size_t c = 0; c < cc.size(); ++c) {
            if (coin(rng) < 0.5f) {
                cc[c] = bc[c];                 // 整列重组 (列内拓扑完整)
                cc[c].column_id = static_cast<uint32_t>(c);
                cc[c].reset();                 // 寄存器运行时状态归零
            }
        }

        // 长程轴突按子代重连: 轴突密度从父 A 推断 (每柱轴突数 = 总数 / 列数)
        const uint32_t n_cols = static_cast<uint32_t>(cc.size());
        uint32_t axons_per_col = n_cols > 0
            ? static_cast<uint32_t>(child.macro_axons().size() / n_cols) : 0;
        if (axons_per_col == 0) axons_per_col = 1;
        child.wire_small_world_axons(axons_per_col, static_cast<uint32_t>(rng()));
        return child;
    }
};

// ── 基因组等价判定 (字段级; 基因组含 vector 成员, 禁止 memcmp) ────────────────
inline bool genome_equal(const CompactSoAGenome& x, const CompactSoAGenome& y) {
    if (x.num_cells != y.num_cells || x.num_synapses != y.num_synapses) return false;
    if (x.op_types != y.op_types) return false;
    if (x.gains != y.gains || x.biases != y.biases) return false;
    if (x.inc_off != y.inc_off || x.inc_from != y.inc_from || x.inc_weight != y.inc_weight) return false;
    return true;
}

// ── 列级有性杂交自由函数 (CrossoverTrait<CorticalMacroArray> 的具名入口) ─────
inline CorticalMacroArray column_crossover(const CorticalMacroArray& a, const CorticalMacroArray& b,
                                           std::mt19937& rng) {
    return CrossoverTrait<CorticalMacroArray>::cross(a, b, rng);
}

// ── 冒烟校验: 全零输入前向, 输出必须有限 (杂交/迁移/变异后的合法性闸门) ────────
inline bool forward_smoke_finite(CorticalMacroArray& arr) {
    const size_t n_cols = arr.columns().size();
    if (n_cols == 0) return false;
    const uint32_t in_dim = static_cast<uint32_t>(arr.columns()[0].local_inputs.size());
    const uint32_t out_dim = static_cast<uint32_t>(arr.columns()[0].local_outputs.size());
    std::vector<float> in_buf(n_cols * in_dim, 0.0f);
    std::vector<float> out_buf(n_cols * out_dim, 0.0f);
    std::vector<const float*> in_ptrs(n_cols);
    for (size_t c = 0; c < n_cols; ++c) in_ptrs[c] = in_buf.data() + c * in_dim;
    arr.forward_multi_channel(in_ptrs.data(), out_buf.data());
    for (float v : out_buf) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

}  // namespace population
}  // namespace kun
