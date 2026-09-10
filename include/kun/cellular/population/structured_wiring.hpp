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
//       领域无关(禁止具身业务专有名词); 显式 rng; 同 seed 逐字段可复现;
//       不依赖 tasks/。L0 零修改。
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_map>
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
    float    internal_param1{0.1f};    // 内部细胞初值 (EMA/积分等增益; 消融可抬到 1.0)
    bool     add_interlayer_feedback{false}; // V_{l+1}→V_l 同列反馈 (compile 标 is_recurrent)
    float    w_feedback{0.10f};        // 反馈边权重
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
                    c.param1 = static_cast<double>(spec.internal_param1);
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

    // (4) 可选层间反馈: 同列 V_{l+1} → V_l (打破严格 DAG; compile 标 is_recurrent)
    //     单变量消融用: 保持幅度默认, 只加记忆通道 —— 对照「R′ 意外递归」假说
    if (spec.add_interlayer_feedback && L >= 2) {
        for (uint32_t cy = 0; cy < G; ++cy) {
            for (uint32_t cx = 0; cx < G; ++cx) {
                for (uint32_t l = 1; l < L; ++l) {
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        add_syn(int_ids[cell_index(cx, cy, l + 1, kk)],
                                int_ids[cell_index(cx, cy, l, kk)],
                                spec.w_feedback);
                    }
                }
            }
        }
    }

    org.compile();  // 活性集应 == 全体细胞 (机制定理; 反馈边不破坏反向可达)
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
    org.compile();
}

// R′ 边类消融: 按源/汇角色选择性地把 to_cell_id 改写到全体细胞池 (与 R′ 同分布)。
struct EdgeClassRandomizeFlags {
    bool receptor_out{false};  // 源为受体 (z≈-1)
    bool effector_in{false};   // 汇为效应器 (z≈-2) —— 按改写前的汇判定
    bool internal{false};      // 源与汇皆为层细胞 (z≥1)
};

inline size_t randomize_targets_by_edge_class(CellularOrganism& org,
                                             EdgeClassRandomizeFlags flags,
                                             uint32_t seed) {
    if (org.cells.empty()) return 0;
    std::unordered_map<uint32_t, const Cell*> by_id;
    by_id.reserve(org.cells.size() * 2);
    for (const auto& c : org.cells) by_id[c.id] = &c;

    std::vector<uint32_t> all_ids;
    all_ids.reserve(org.cells.size());
    for (const auto& c : org.cells) all_ids.push_back(c.id);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, all_ids.size() - 1);
    size_t rewritten = 0;
    for (auto& s : org.synapses) {
        const Cell* src = by_id[s.from_cell_id];
        const Cell* dst = by_id[s.to_cell_id];
        if (!src || !dst) continue;
        const bool is_recv = (src->z < 0.0f && src->z > -1.5f);  // z=-1 受体
        const bool is_eff_in = (dst->z < -1.5f);                 // z=-2 效应器
        const bool is_int = (src->z >= 1.0f && dst->z >= 1.0f);
        bool hit = false;
        if (flags.receptor_out && is_recv) hit = true;
        if (flags.effector_in && is_eff_in) hit = true;
        if (flags.internal && is_int) hit = true;
        if (!hit) continue;
        s.to_cell_id = all_ids[pick(rng)];
        ++rewritten;
    }
    org.is_compiled_ = false;
    return rewritten;
}

// 路线 A: 仅随机化「非 IO 骨架」边的目标 —— 保受体出边与效应器入边不动。
// 返回被改写的边数。compile() 由调用方或 ensure_active_closure 负责。
inline size_t randomize_internal_preserving_io(CellularOrganism& org, uint32_t seed) {
    if (org.cells.empty()) return 0;
    std::unordered_map<uint32_t, const Cell*> by_id;
    by_id.reserve(org.cells.size() * 2);
    for (const auto& c : org.cells) by_id[c.id] = &c;

    std::vector<uint32_t> internal_ids;
    internal_ids.reserve(org.cells.size());
    for (const auto& c : org.cells) {
        if (c.z >= 1.0f) internal_ids.push_back(c.id);  // 层细胞 (非受体/效应器)
    }
    if (internal_ids.empty()) return 0;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, internal_ids.size() - 1);
    size_t rewritten = 0;
    for (auto& s : org.synapses) {
        const Cell* src = by_id[s.from_cell_id];
        const Cell* dst = by_id[s.to_cell_id];
        if (!src || !dst) continue;
        if (src->z < 0.0f) continue;   // 受体出边: 保
        if (dst->z < -1.5f) continue;  // 效应器入边 (z=-2): 保
        // 内部→内部 (含反馈边): 目标改写到随机内部细胞
        s.to_cell_id = internal_ids[pick(rng)];
        ++rewritten;
    }
    org.is_compiled_ = false;
    return rewritten;
}

// 活性闭包修复: 对 compile 后不在 execution_order_ 的细胞, 加一条到效应器 0 的保命边。
// 返回新增边数。要求细胞 z 标签仍有效 (结构化构建产物)。
inline size_t ensure_active_closure(CellularOrganism& org, float w_lifeline = 0.05f) {
    if (org.cells.empty()) return 0;
    org.compile();
    if (org.execution_order_.size() == org.cells.size()) return 0;

    std::vector<char> active(org.cells.size(), 0);
    for (size_t idx : org.execution_order_) {
        if (idx < active.size()) active[idx] = 1;
    }

    uint32_t eff0 = std::numeric_limits<uint32_t>::max();
    for (const auto& c : org.cells) {
        if (c.z < -1.5f) { eff0 = c.id; break; }
    }
    if (eff0 == std::numeric_limits<uint32_t>::max()) return 0;

    size_t added = 0;
    for (size_t i = 0; i < org.cells.size(); ++i) {
        if (active[i]) continue;
        Synapse s;
        s.from_cell_id = org.cells[i].id;
        s.to_cell_id = eff0;
        s.to_port = 0;
        s.weight = static_cast<double>(w_lifeline);
        s.initial_weight = s.weight;
        s.is_active = true;
        org.synapses.push_back(s);
        ++added;
    }
    if (added > 0) {
        org.is_compiled_ = false;
        org.compile();
    }
    return added;
}

// 编译后递归突触计数 (消融诊断: R′ / recur / rrac 臂应 >0, 纯 DAG 为 0)
inline size_t recurrent_synapse_count(const CellularOrganism& org) {
    size_t n = 0;
    for (const auto& cs : org.compiled_synapses_) {
        if (cs.is_recurrent) ++n;
    }
    return n;
}

// ── 受体直路: 跳过恒等 SENSE_CHANNEL, 观测经 param1*w 直接打进下游端口 ──
// 无塑性时动作输出应与 forward_nd 对齐; 膜电位/glow 不进适应度, 不更新。
struct ReceptorBypass {
    bool ok{false};
    std::vector<char> is_receptor;
    std::vector<size_t> internal_order;
    std::vector<uint32_t> from_idx;
    std::vector<uint32_t> to_idx;
    std::vector<uint8_t> to_port;
    std::vector<uint32_t> channel;
    std::vector<uint32_t> syn_idx;
    uint32_t n_receptors{0};
};

inline ReceptorBypass build_receptor_bypass(const CellularOrganism& org) {
    ReceptorBypass b;
    if (!org.is_compiled() || org.cells.empty()) return b;
    b.is_receptor.assign(org.cells.size(), 0);
    for (size_t i = 0; i < org.cells.size(); ++i) {
        if (org.cells[i].type == CellType::SENSE_CHANNEL) {
            b.is_receptor[i] = 1;
            ++b.n_receptors;
        }
    }
    if (b.n_receptors == 0) return b;
    b.internal_order.reserve(org.execution_order_.size());
    for (size_t idx : org.execution_order_) {
        if (idx < b.is_receptor.size() && !b.is_receptor[idx])
            b.internal_order.push_back(idx);
    }
    for (uint32_t si = 0; si < static_cast<uint32_t>(org.compiled_synapses_.size()); ++si) {
        const auto& syn = org.compiled_synapses_[si];
        if (syn.from_idx >= b.is_receptor.size() || !b.is_receptor[syn.from_idx]) continue;
        if (syn.is_recurrent) continue;
        const Cell& src = org.cells[syn.from_idx];
        const uint32_t ch = (src.param2 >= 0.0) ? static_cast<uint32_t>(src.param2) : 0u;
        b.from_idx.push_back(static_cast<uint32_t>(syn.from_idx));
        b.to_idx.push_back(static_cast<uint32_t>(syn.to_idx));
        b.to_port.push_back(syn.to_port);
        b.channel.push_back(ch);
        b.syn_idx.push_back(si);
    }
    b.ok = !b.internal_order.empty();
    return b;
}

inline CellularOrganism::ActionOutputs forward_nd_skip_receptors(
    CellularOrganism& org, const ReceptorBypass& bypass,
    const double* inputs, size_t in_dim)
{
    if (!bypass.ok) return org.forward_nd(inputs, in_dim, false);
    if (!org.is_compiled_) org.compile();
    if (inputs == nullptr) in_dim = 0;

    double* __restrict port_ptr = org.flat_port_inputs_.data();
    std::memset(port_ptr, 0, org.flat_port_inputs_.size() * sizeof(double));
    Cell* __restrict cells_ptr = org.cells.data();
    auto* __restrict syn_ptr = org.compiled_synapses_.data();
    const size_t num_synapses = org.compiled_synapses_.size();

    for (size_t i = 0; i < num_synapses; ++i) {
        const auto& syn = syn_ptr[i];
        if (syn.is_recurrent) {
            port_ptr[syn.to_idx * 2 + syn.to_port] +=
                cells_ptr[syn.from_idx].prev_output_val * syn.weight;
        }
    }

    const uint32_t n_inj = static_cast<uint32_t>(bypass.from_idx.size());
    for (uint32_t i = 0; i < n_inj; ++i) {
        const uint32_t ch = bypass.channel[i];
        const double src = (ch < in_dim) ? inputs[ch] : 0.0;
        const uint32_t fi = bypass.from_idx[i];
        const auto& syn = syn_ptr[bypass.syn_idx[i]];
        const double rec_out = src * cells_ptr[fi].param1;
        cells_ptr[fi].output_val = rec_out;
        port_ptr[bypass.to_idx[i] * 2 + bypass.to_port[i]] += rec_out * syn.weight;
    }

    for (size_t idx : bypass.internal_order) {
        auto& c = cells_ptr[idx];
        dispatch_cell_forward(c, port_ptr[idx * 2], port_ptr[idx * 2 + 1], in_dim, inputs);
        if (std::abs(c.output_val) > 1e-6) {
            c.activation_count++;
        }
        for (size_t k = org.out_start_[idx]; k < org.out_start_[idx + 1]; ++k) {
            const auto& syn = syn_ptr[org.out_edges_[k]];
            if (!syn.is_recurrent) {
                port_ptr[syn.to_idx * 2 + syn.to_port] += c.output_val * syn.weight;
            }
        }
    }

    for (size_t i = 0; i < org.cells.size(); ++i) {
        cells_ptr[i].prev_output_val = cells_ptr[i].output_val;
    }

    CellularOrganism::ActionOutputs actions{};
    for (const auto& ac : org.compiled_actions_) {
        const double val = cells_ptr[ac.cell_idx].output_val;
        if (ac.type == CellType::ACT_PRIMARY_POSITIVE) actions.positive_action = val;
        else if (ac.type == CellType::ACT_PRIMARY_NEGATIVE) actions.negative_action = val;
        else if (ac.type == CellType::ACT_DEFENSIVE_RESET) actions.defensive_reset = val;
        else if (ac.type == CellType::ACT_IMMUNE_BLOCK && val > 0.5) actions.immune_lock = true;
    }
    return actions;
}

}  // namespace population
}  // namespace kun
