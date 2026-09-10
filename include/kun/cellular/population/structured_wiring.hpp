#pragma once
// ============================================================================
// population/structured_wiring.hpp — L1 系统层: 结构化发育接线 (task-agnostic)
//
// 问题: 旧接线 develop_to_scale(随机链) + wire_global_bridge(随机桥) 造出的细胞
//       到效应器无有向路径 ⇒ compile() 活性集(反向可达闭包)不含它们 ⇒
//       prune_apoptosis 按定义删除 ⇒ 百万细胞自坍缩。
//
// 方案: 用「层单调 + 环形 2D lattice 局部 + 每个非终端细胞出边必达效应器」的
//       前馈有向 DAG 替换随机接线。机制定理: 层号严格递增 ⇒ DAG; 每层细胞出边到
//       下一层(环形 lattice 无死点) ⇒ 归纳可达 V_L ⇒ V_L→效应器 ⇒ 活性集=全体细胞。
//
// 纪律: 仅消费 L0 公开 API (Cell/Synapse/CellularOrganism::cells/synapses/compile);
//       领域无关(无体素/车辆/K线等业务名词); 显式 rng; 同 seed 逐字段可复现;
//       不依赖 tasks/。L0 零修改。
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace kun {
namespace population {

// ── 列格规格 (与 CML 环形边界一致; vox_per_column_axis=b, lattice=G, n=G*b) ────
struct ColumnLatticeSpec {
    uint32_t lattice_w{64};            // G (环形, 需 == lattice_h)
    uint32_t lattice_h{64};            // G
    uint32_t vox_per_column_axis{4};   // b
    uint32_t layers{4};                // L (列内皮层深度)
    uint32_t cells_per_layer{4};       // k
    uint32_t lateral_radius{1};        // r (列间 Chebyshev 半径)
    float    w_receptor{0.15f};
    float    w_lateral{0.10f};
    float    w_readout{0.005f};        // 全 V_L→效应器 保命小权重
    uint32_t seed{1};
};

// ── 目标锚点 (由 L2 提供: 目标体素所属列 → 效应器下标); L1 不含目标语义 ────────
struct ReadoutAnchor {
    uint32_t col_x{0};
    uint32_t col_y{0};
    uint32_t effector_index{0};   // 0=positive, 1=negative, 2=defensive, 3=immune
    double   weight{0.5};
};

// 机制不变量: compile() 认定的活细胞数 (== 全体细胞即零坍缩)
inline size_t active_cell_count(const CellularOrganism& org) {
    return org.execution_order_.size();
}

inline uint32_t wrap_index(uint32_t a, int d, uint32_t g) {
    int v = static_cast<int>(a) + d;
    v %= static_cast<int>(g);
    if (v < 0) v += static_cast<int>(g);
    return static_cast<uint32_t>(v);
}

// ── 构建结构化接线 (原地重建 org 的 cells/synapses; 末尾 compile()) ────────────
inline void build_columnar_structured_wiring(CellularOrganism& org,
                                             const ColumnLatticeSpec& spec,
                                             const std::vector<ReadoutAnchor>& anchors) {
    const uint32_t G = spec.lattice_w;
    const uint32_t b = spec.vox_per_column_axis;
    const uint32_t L = std::max<uint32_t>(spec.layers, 1);
    const uint32_t k = std::max<uint32_t>(spec.cells_per_layer, 1);
    const uint32_t r = spec.lateral_radius;
    const size_t n = static_cast<size_t>(G) * b;      // 场边长 (环形)
    const size_t n_vox = n * n;

    org.cells.clear();
    org.synapses.clear();
    org.is_compiled_ = false;

    uint32_t next_id = 0;

    // (a) 效应器 (4): 0=positive 1=negative 2=defensive 3=immune
    const CellType eff_types[4] = {
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE,
        CellType::ACT_DEFENSIVE_RESET, CellType::ACT_IMMUNE_BLOCK};
    std::vector<uint32_t> eff_ids(4);
    for (uint32_t e = 0; e < 4; ++e) {
        Cell c;
        c.id = next_id++;
        c.type = eff_types[e];
        c.param2 = static_cast<double>(e);
        c.z = -2.0f;   // z 标签: -2=效应器 (不参与动力学; 供结构断言)
        org.cells.push_back(c);
        eff_ids[e] = c.id;
    }

    // (b) 受体: 每体素一个 SENSE_CHANNEL (读 inputs[channel], param2=体素通道)
    std::vector<uint32_t> rec_ids(n_vox);
    for (size_t v = 0; v < n_vox; ++v) {
        Cell c;
        c.id = next_id++;
        c.type = CellType::SENSE_CHANNEL;
        c.param1 = 1.0;
        c.param2 = static_cast<double>(v);
        c.z = -1.0f;   // z 标签: -1=受体
        org.cells.push_back(c);
        rec_ids[v] = c.id;
    }

    // (c) 内部层 V1..VL: G×G 列 × L 层 × k 细胞
    auto cell_index = [&](uint32_t cx, uint32_t cy, uint32_t l, uint32_t kk) -> size_t {
        return (static_cast<size_t>(cy) * G + cx) * L * k +
               (static_cast<size_t>(l - 1) * k + kk);
    };
    const size_t n_internal = static_cast<size_t>(G) * G * L * k;
    std::vector<uint32_t> int_ids(n_internal);
    // 确定性原语配比 (含非线性项, 供演化搜出 logistic 类回归)
    const CellType palette[] = {
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_SUM, CellType::OP_MULTIPLY,
        CellType::OP_QUADRATIC, CellType::OP_ABS, CellType::GATE_THRESHOLD, CellType::OP_INTEGRAL};
    const uint32_t palette_sz = static_cast<uint32_t>(sizeof(palette) / sizeof(palette[0]));
    for (uint32_t cy = 0; cy < G; ++cy) {
        for (uint32_t cx = 0; cx < G; ++cx) {
            for (uint32_t l = 1; l <= L; ++l) {
                for (uint32_t kk = 0; kk < k; ++kk) {
                    Cell c;
                    c.id = next_id++;
                    c.type = palette[(cx + cy * 3 + l * 5 + kk * 7) % palette_sz];
                    c.param1 = 0.1;
                    c.x = static_cast<float>(cx * b);
                    c.y = static_cast<float>(cy * b);
                    c.z = static_cast<float>(l);  // z 标签: 层号 (1..L)
                    org.cells.push_back(c);
                    int_ids[cell_index(cx, cy, l, kk)] = c.id;
                }
            }
        }
    }
    org.next_generated_cell_id_ = next_id;

    auto add_syn = [&](uint32_t from, uint32_t to, double w, uint8_t port = 0) {
        Synapse s;
        s.from_cell_id = from;
        s.to_cell_id = to;
        s.to_port = port;
        s.weight = w;
        s.initial_weight = w;
        s.is_active = true;
        org.synapses.push_back(s);
    };

    // (1) 受体 → 本列 V1 全部 k 细胞
    for (uint32_t cy = 0; cy < G; ++cy) {
        for (uint32_t cx = 0; cx < G; ++cx) {
            for (uint32_t bi = 0; bi < b; ++bi) {
                for (uint32_t bj = 0; bj < b; ++bj) {
                    const size_t v = (static_cast<size_t>(cy) * b + bi) * n + (cx * b + bj);
                    for (uint32_t kk = 0; kk < k; ++kk)
                        add_syn(rec_ids[v], int_ids[cell_index(cx, cy, 1, kk)], spec.w_receptor);
                }
            }
        }
    }

    // (2) V_l → V_{l+1}: 同列 + Manhattan 菱形半径 r 的环形邻列, 同通道 kk
    //     (5 邻居/cell @r=1; 局部卷积式, 显著轻于全 Chebyshev 扇出)
    for (uint32_t cy = 0; cy < G; ++cy) {
        for (uint32_t cx = 0; cx < G; ++cx) {
            for (uint32_t l = 1; l < L; ++l) {
                for (uint32_t kk = 0; kk < k; ++kk) {
                    const uint32_t from = int_ids[cell_index(cx, cy, l, kk)];
                    for (int dy = -static_cast<int>(r); dy <= static_cast<int>(r); ++dy) {
                        const int rem = static_cast<int>(r) - std::abs(dy);
                        for (int dx = -rem; dx <= rem; ++dx) {
                            const uint32_t tx = wrap_index(cx, dx, G);
                            const uint32_t ty = wrap_index(cy, dy, G);
                            add_syn(from, int_ids[cell_index(tx, ty, l + 1, kk)], spec.w_lateral);
                        }
                    }
                }
            }
        }
    }

    // (3) V_L → 全部效应器 (保命小权重)
    for (uint32_t cy = 0; cy < G; ++cy) {
        for (uint32_t cx = 0; cx < G; ++cx) {
            for (uint32_t kk = 0; kk < k; ++kk) {
                const uint32_t from = int_ids[cell_index(cx, cy, L, kk)];
                for (uint32_t e = 0; e < 4; ++e) add_syn(from, eff_ids[e], spec.w_readout);
            }
        }
    }

    // (3b) 目标锚点: 命中列 → 对应效应器 (强权重; 归纳偏置, 可被 S⁻ᵃ 消融)
    for (const auto& a : anchors) {
        if (a.col_x >= G || a.col_y >= G || a.effector_index >= 4) continue;
        for (uint32_t kk = 0; kk < k; ++kk) {
            const uint32_t from = int_ids[cell_index(a.col_x, a.col_y, L, kk)];
            add_syn(from, eff_ids[a.effector_index], a.weight);
        }
    }

    org.compile();  // 活性集应 == 全体细胞 (机制定理)
}

// ── 消融对照: 打乱内部突触的目标端点 (同细胞数/同突触数, 仅拓扑随机化) ──────────
// 用于 R′ 对照臂: 隔离「空间局部性」与「数量」对结果的影响。
inline void randomize_internal_synapse_targets(CellularOrganism& org, uint32_t seed) {
    if (org.cells.empty()) return;
    std::mt19937 rng(seed);
    std::vector<uint32_t> ids;
    ids.reserve(org.cells.size());
    for (const auto& c : org.cells) ids.push_back(c.id);
    std::uniform_int_distribution<size_t> pick(0, ids.size() - 1);
    for (auto& s : org.synapses) {
        s.to_cell_id = ids[pick(rng)];
    }
    org.is_compiled_ = false;
}

}  // namespace population
}  // namespace kun
