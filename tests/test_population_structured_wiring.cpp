// ============================================================================
// test_population_structured_wiring.cpp — L1 结构化发育接线不变量测试
//
// 断言: H1 零坍缩 (活性集==全体细胞) / H2 DAG+同种子逐字段确定性 /
//       空间局部性 (内部细胞入边仅来自本列受体或 ≤r 邻列上一层) / 前向有限。
// ============================================================================
#include "kun/cellular/population/structured_wiring.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

using kun::Cell;
using kun::CellularOrganism;
using kun::population::ColumnLatticeSpec;
using kun::population::ReadoutAnchor;
using kun::population::active_cell_count;
using kun::population::build_columnar_structured_wiring;

namespace {

ColumnLatticeSpec spec_of(uint32_t G, uint32_t b, uint32_t L = 4, uint32_t k = 4,
                          uint32_t r = 1, uint32_t seed = 7) {
    ColumnLatticeSpec s;
    s.lattice_w = G;
    s.lattice_h = G;
    s.vox_per_column_axis = b;
    s.layers = L;
    s.cells_per_layer = k;
    s.lateral_radius = r;
    s.seed = seed;
    return s;
}

std::vector<ReadoutAnchor> default_anchors(uint32_t G, uint32_t b) {
    // A=(0,0) → 列(0,0)→positive; B=(n/2,n/2) → 列(G/2,G/2)→negative
    return {{0, 0, 0, 0.5}, {G / 2, G / 2, 1, 0.5}};
}

CellularOrganism build(const ColumnLatticeSpec& s) {
    CellularOrganism org = CellularOrganism::create_seed_organism();
    build_columnar_structured_wiring(org, s, default_anchors(s.lattice_w, s.vox_per_column_axis));
    return org;
}

void test_H1_zero_collapse_small() {
    auto s = spec_of(8, 2);  // n=16
    auto org = build(s);
    const size_t n_vox = (size_t)s.lattice_w * s.vox_per_column_axis;
    const size_t expect = 4 + n_vox * n_vox + (size_t)s.lattice_w * s.lattice_w * s.layers * s.cells_per_layer;
    assert(org.cells.size() == expect);
    assert(active_cell_count(org) == org.cells.size());  // active/cells == 1.0 ≥ 0.99
    std::printf("  ✓ H1 零坍缩 (n=16): 全体 %zu 细胞均在活性集\n", org.cells.size());
}

void test_H1_zero_collapse_scale() {
    auto s = spec_of(64, 2);  // n=128 → 81920 细胞
    auto org = build(s);
    const double ratio = (double)active_cell_count(org) / (double)org.cells.size();
    assert(ratio >= 0.99);
    std::printf("  ✓ H1 零坍缩 (n=128): %zu 细胞, 活性比 %.4f\n", org.cells.size(), ratio);
}

void test_H2_dag_and_determinism() {
    auto s = spec_of(8, 2);
    auto a = build(s);
    // DAG: 执行序覆盖全体细胞 (无环保护补齐路径不会触发)
    assert(a.execution_order_.size() == a.cells.size());

    auto b = build(s);  // 同 spec+seed 再次构建
    assert(b.cells.size() == a.cells.size());
    assert(b.synapses.size() == a.synapses.size());
    for (size_t i = 0; i < a.cells.size(); ++i) {
        assert(a.cells[i].id == b.cells[i].id);
        assert(a.cells[i].type == b.cells[i].type);
        assert(a.cells[i].param1 == b.cells[i].param1);
        assert(a.cells[i].param2 == b.cells[i].param2);
    }
    for (size_t i = 0; i < a.synapses.size(); ++i) {
        assert(a.synapses[i].from_cell_id == b.synapses[i].from_cell_id);
        assert(a.synapses[i].to_cell_id == b.synapses[i].to_cell_id);
        assert(a.synapses[i].weight == b.synapses[i].weight);
    }
    std::printf("  ✓ H2 DAG (%zu 序) + 同种子逐字段一致 (%zu 突触)\n",
                a.execution_order_.size(), a.synapses.size());
}

void test_locality() {
    const uint32_t G = 8, b = 2, L = 4, k = 4, r = 1;
    auto s = spec_of(G, b, L, k, r);
    auto org = build(s);
    const size_t n = (size_t)G * b;

    std::unordered_map<uint32_t, const Cell*> by_id;
    for (const auto& c : org.cells) by_id[c.id] = &c;

    auto is_internal = [](const Cell* c) { return c->z >= 1.0f; };
    auto col_of = [&](const Cell* c, int& cx, int& cy) {
        cx = static_cast<int>(c->x / (float)b);
        cy = static_cast<int>(c->y / (float)b);
    };

    for (const auto& sy : org.synapses) {
        const Cell* t = by_id[sy.to_cell_id];
        const Cell* src = by_id[sy.from_cell_id];
        assert(t && src);
        if (!is_internal(t)) continue;  // 目标是受体/效应器: 跳过
        int tcx, tcy;
        col_of(t, tcx, tcy);
        if (src->z == -1.0f) {  // 受体 → 仅本列 V1
            assert(t->z == 1.0f);
            const size_t v = static_cast<size_t>(src->param2);
            const int rcx = static_cast<int>((v % n) / b);
            const int rcy = static_cast<int>((v / n) / b);
            assert(rcx == tcx && rcy == tcy);
        } else {
            assert(is_internal(src));
            assert(static_cast<int>(src->z) + 1 == static_cast<int>(t->z));  // 层单调
            int scx, scy;
            col_of(src, scx, scy);
            int dx = std::abs(scx - tcx); dx = std::min(dx, static_cast<int>(G) - dx);
            int dy = std::abs(scy - tcy); dy = std::min(dy, static_cast<int>(G) - dy);
            assert(dx <= static_cast<int>(r) && dy <= static_cast<int>(r));  // Chebyshev 局部
        }
    }
    std::printf("  ✓ 空间局部性: 内部入边仅本列受体 / ≤r 邻列上一层\n");
}

void test_forward_finite() {
    auto s = spec_of(8, 2);  // n=16
    auto org = build(s);
    const size_t n = (size_t)s.lattice_w * s.vox_per_column_axis;
    std::vector<double> in(n * n, 0.0);
    for (size_t i = 0; i < in.size(); ++i) in[i] = std::sin(static_cast<double>(i) * 0.07) * 0.5 + 0.5;
    auto out = org.forward_nd(in.data(), in.size(), /*hebbian=*/false);
    assert(std::isfinite(out.positive_action));
    assert(std::isfinite(out.negative_action));
    std::printf("  ✓ 前向有限 (n=16, %zu 输入)\n", in.size());
}

void test_ampfix_defaults_and_feedback_H1() {
    // 幅度校准: param1 / w_readout 可配置且仍零坍缩
    auto s = spec_of(8, 2);
    s.internal_param1 = 1.0f;
    s.w_readout = s.w_receptor;
    auto org_amp = build(s);
    assert(active_cell_count(org_amp) == org_amp.cells.size());
    for (const auto& c : org_amp.cells) {
        if (c.z >= 1.0f) assert(c.param1 == 1.0);
    }
    assert(kun::population::recurrent_synapse_count(org_amp) == 0);
    std::printf("  ✓ ampfix: param1=1.0, 递归边=0, 活性比=1\n");

    // 层间反馈: 引入 is_recurrent 且仍零坍缩
    auto sf = spec_of(8, 2);
    sf.add_interlayer_feedback = true;
    auto org_f = build(sf);
    assert(active_cell_count(org_f) == org_f.cells.size());
    assert(kun::population::recurrent_synapse_count(org_f) > 0);
    std::printf("  ✓ recur: 递归边=%zu, 活性比=1\n",
                kun::population::recurrent_synapse_count(org_f));
}

void test_rrac_preserving_io_anticollapse() {
    auto s = spec_of(8, 2);
    s.add_interlayer_feedback = true;
    auto org = build(s);
    const size_t syn_before = org.synapses.size();
    const size_t rec_before = kun::population::recurrent_synapse_count(org);
    assert(rec_before > 0);

    // 记录 IO 边指纹
    std::vector<std::pair<uint32_t, uint32_t>> io_edges;
    std::unordered_map<uint32_t, const Cell*> by_id;
    for (const auto& c : org.cells) by_id[c.id] = &c;
    for (const auto& sy : org.synapses) {
        const Cell* src = by_id[sy.from_cell_id];
        const Cell* dst = by_id[sy.to_cell_id];
        if (src->z < 0.0f || dst->z < -1.5f)
            io_edges.push_back({sy.from_cell_id, sy.to_cell_id});
    }

    const size_t rewritten = kun::population::randomize_internal_preserving_io(org, 42);
    assert(rewritten > 0);
    assert(org.synapses.size() == syn_before);  // 只改目标, 不增删

    by_id.clear();
    for (const auto& c : org.cells) by_id[c.id] = &c;
    size_t io_i = 0;
    for (const auto& sy : org.synapses) {
        const Cell* src = by_id[sy.from_cell_id];
        const Cell* dst = by_id[sy.to_cell_id];
        if (src->z < 0.0f || dst->z < -1.5f) {
            assert(io_edges[io_i].first == sy.from_cell_id);
            assert(io_edges[io_i].second == sy.to_cell_id);
            ++io_i;
        }
    }
    assert(io_i == io_edges.size());

    const size_t repaired = kun::population::ensure_active_closure(org);
    assert(active_cell_count(org) == org.cells.size());
    assert(kun::population::recurrent_synapse_count(org) > 0);
    std::printf("  ✓ RRAC: 改写 %zu 边 | 修复 %zu | 递归边 %zu | 活性比=1 | IO 骨架保留\n",
                rewritten, repaired, kun::population::recurrent_synapse_count(org));
}

void test_edge_class_randomize_counts() {
    auto org = build(spec_of(8, 2));
    std::unordered_map<uint32_t, const Cell*> by_id;
    for (const auto& c : org.cells) by_id[c.id] = &c;
    size_t n_recv = 0, n_eff = 0, n_int = 0;
    for (const auto& sy : org.synapses) {
        const Cell* src = by_id[sy.from_cell_id];
        const Cell* dst = by_id[sy.to_cell_id];
        if (src->z < 0.0f && src->z > -1.5f) ++n_recv;
        if (dst->z < -1.5f) ++n_eff;
        if (src->z >= 1.0f && dst->z >= 1.0f) ++n_int;
    }
    kun::population::EdgeClassRandomizeFlags fr{true, false, false};
    assert(kun::population::randomize_targets_by_edge_class(org, fr, 1) == n_recv);
    org = build(spec_of(8, 2));
    kun::population::EdgeClassRandomizeFlags fe{false, true, false};
    assert(kun::population::randomize_targets_by_edge_class(org, fe, 2) == n_eff);
    org = build(spec_of(8, 2));
    kun::population::EdgeClassRandomizeFlags fi{false, false, true};
    assert(kun::population::randomize_targets_by_edge_class(org, fi, 3) == n_int);
    kun::population::ensure_active_closure(org);
    assert(active_cell_count(org) == org.cells.size());
    std::printf("  ✓ 边类消融计数: recv=%zu eff=%zu int=%zu\n", n_recv, n_eff, n_int);

    auto org_io = build(spec_of(8, 2));
    auto [rw, rp] = kun::population::apply_io_mix_developmental(org_io, 99);
    assert(rw == n_recv + n_eff);
    assert(active_cell_count(org_io) == org_io.cells.size());
    std::printf("  ✓ io_mix 配方: 改写 %zu | 修复 %zu | 活性比=1\n", rw, rp);
}

void test_receptor_bypass_matches_forward() {
    auto org = build(spec_of(8, 2));
    auto bp = kun::population::build_receptor_bypass(org);
    assert(bp.ok);
    assert(bp.n_receptors > 0);
    const size_t n = 16 * 16;
    std::vector<double> in(n, 0.0);
    for (size_t i = 0; i < n; ++i) in[i] = std::sin(static_cast<double>(i) * 0.11) * 0.4 + 0.5;
    auto a = org;
    auto b = org;
    a.reset_state(true);
    b.reset_state(true);
    auto oa = a.forward_nd(in.data(), in.size(), false);
    auto ob = kun::population::forward_nd_skip_receptors(b, bp, in.data(), in.size());
    assert(std::fabs(oa.positive_action - ob.positive_action) < 1e-9);
    assert(std::fabs(oa.negative_action - ob.negative_action) < 1e-9);
    oa = a.forward_nd(in.data(), in.size(), false);
    ob = kun::population::forward_nd_skip_receptors(b, bp, in.data(), in.size());
    assert(std::fabs(oa.positive_action - ob.positive_action) < 1e-9);
    std::printf("  ✓ 受体直路: 动作输出与 forward_nd 对齐 (受体 %u / 注入 %zu)\n",
                bp.n_receptors, bp.from_idx.size());
}

}  // namespace

int main() {
    std::printf("==================================================================\n");
    std::printf("  L1 系统层: 结构化发育接线不变量测试\n");
    std::printf("==================================================================\n");
    test_H1_zero_collapse_small();
    test_H2_dag_and_determinism();
    test_locality();
    test_forward_finite();
    test_ampfix_defaults_and_feedback_H1();
    test_rrac_preserving_io_anticollapse();
    test_edge_class_randomize_counts();
    test_receptor_bypass_matches_forward();
    test_H1_zero_collapse_scale();
    std::printf("[PASS] test_population_structured_wiring all assertions passed!\n");
    return 0;
}
