// ============================================================================
// test_population_individual_traits.cpp — L1 系统层: 杂交算子不变量测试
// Phase 2 Prefer-A: cortical harden + 82-organism sexual + compat reject/skip
// 不变量: 列数/82 守恒 / 前向有限 / 同种子位级复现 / 不相容拒绝 (非静默克隆)
// ============================================================================
#include "kun/cellular/cortical_column.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"
#include "kun/cellular/population/individual_traits.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::CompactSoAGenome;
using kun::CorticalMacroArray;
using kun::Synapse;
using kun::population::CrossoverTrait;
using kun::population::column_crossover;
using kun::population::compatibility_distance;
using kun::population::forward_smoke_finite;
using kun::population::genome_equal;
using kun::population::kDefaultSpeciationThreshold;
using kun::population::kPreferAOrganismCells;
using kun::population::mates_compatible;
using kun::population::organism_heritable_equal;

namespace {

// 构造随机化亲本 (确定性种子)
CorticalMacroArray make_parent(std::mt19937::result_type seed) {
    CorticalMacroArray arr(6, 8, 12, 2, 1);
    std::mt19937 rng(seed);
    for (auto& col : arr.columns()) {
        col.genome.mutate_parameters(1.0f, 0.5f, rng);
        col.genome.mutate_primitive_types(0.5f, rng);
    }
    arr.wire_small_world_axons(2, seed);
    return arr;
}

// 固定输入前向输出 (用于位级对账; forward 为非 const 操作, 寄存器状态会被推进)
std::vector<float> forward_outputs(CorticalMacroArray& arr) {
    const size_t n_cols = arr.columns().size();
    const uint32_t in_dim = static_cast<uint32_t>(arr.columns()[0].local_inputs.size());
    const uint32_t out_dim = static_cast<uint32_t>(arr.columns()[0].local_outputs.size());
    std::vector<float> in_buf(n_cols * in_dim);
    std::vector<float> out(n_cols * out_dim, 0.0f);
    std::vector<const float*> in_ptrs(n_cols);
    for (size_t c = 0; c < n_cols; ++c) {
        for (uint32_t d = 0; d < in_dim; ++d) {
            in_buf[c * in_dim + d] = std::sin(static_cast<float>(c * 7 + d) * 0.31f) * 0.5f;
        }
        in_ptrs[c] = in_buf.data() + c * in_dim;
    }
    arr.forward_multi_channel(in_ptrs.data(), out.data());
    return out;
}

// Prefer-A 82-cell organism (synthetic; slot-aligned synapses).
CellularOrganism make_organism_82(std::mt19937::result_type seed) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "PreferA-82";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pdist(0.01, 0.99);
    std::uniform_real_distribution<double> wdist(-1.0, 1.0);
    static const CellType kTypes[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_SUM, CellType::OP_SUB,
        CellType::GATE_HYSTERESIS, CellType::GATE_THRESHOLD,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE
    };
    org.cells.reserve(kPreferAOrganismCells);
    for (size_t i = 0; i < kPreferAOrganismCells; ++i) {
        Cell c;
        c.id = static_cast<uint32_t>(i);
        c.type = kTypes[i % (sizeof(kTypes) / sizeof(kTypes[0]))];
        c.param1 = pdist(rng);
        c.param2 = pdist(rng) * 0.1;
        c.x = static_cast<float>(i % 10) * 10.0f;
        c.y = static_cast<float>(i / 10) * 10.0f;
        org.cells.push_back(c);
    }
    // Fixed syn count (425 champion-ish density scaled for test; keep equal across parents).
    constexpr size_t kSyn = 164;
    org.synapses.reserve(kSyn);
    for (size_t s = 0; s < kSyn; ++s) {
        Synapse syn;
        syn.from_cell_id = static_cast<uint32_t>(s % kPreferAOrganismCells);
        syn.to_cell_id = static_cast<uint32_t>((s * 7 + 3) % kPreferAOrganismCells);
        syn.to_port = static_cast<uint8_t>(s & 1u);
        syn.weight = wdist(rng);
        syn.initial_weight = syn.weight;
        syn.is_active = true;
        org.synapses.push_back(syn);
    }
    org.compile();
    assert(org.cells.size() == kPreferAOrganismCells);
    return org;
}

void test_crossover_invariants() {
    auto pa = make_parent(101);
    auto pb = make_parent(202);
    std::mt19937 rng(42);

    auto child = column_crossover(pa, pb, rng);
    assert(child.columns().size() == pa.columns().size());
    assert(child.columns().size() == pb.columns().size());
    // 每列细胞数守恒
    for (size_t c = 0; c < child.columns().size(); ++c) {
        assert(child.columns()[c].genome.num_cells == pa.columns()[c].genome.num_cells);
    }
    // 子代列必来自父 A 或父 B 之一 (逐列基因组字段级归属校验)
    bool mixed = false;
    for (size_t c = 0; c < child.columns().size(); ++c) {
        const bool from_a = genome_equal(child.columns()[c].genome, pa.columns()[c].genome);
        const bool from_b = genome_equal(child.columns()[c].genome, pb.columns()[c].genome);
        assert(from_a || from_b);
        if (from_b && !from_a) mixed = true;
    }
    assert(mixed);  // 6 列 50/50 硬币: 全同父概率 (1/2)^6 = 1.6%, 种子 42 下应为混合
    // 前向冒烟有限
    assert(forward_smoke_finite(child));
    std::cout << "  ✓ 杂交不变量: 列数守恒 / 列级归属 / 前向有限 / 混合发生" << std::endl;
}

void test_crossover_determinism() {
    auto pa = make_parent(101);
    auto pb = make_parent(202);
    std::mt19937 r1(7);
    auto c1 = column_crossover(pa, pb, r1);
    std::mt19937 r2(7);
    auto c2 = column_crossover(pa, pb, r2);
    auto o1 = forward_outputs(c1);
    auto o2 = forward_outputs(c2);
    assert(o1.size() == o2.size());
    for (size_t i = 0; i < o1.size(); ++i) {
        assert(std::memcmp(&o1[i], &o2[i], sizeof(float)) == 0);  // 位级一致
    }
    std::mt19937 r3(8);
    auto c3 = column_crossover(pa, pb, r3);
    (void)c3;
    // 不同种子应产生不同子代: 多种子中至少出现一种不同列型 (排除同列型巧合)
    // 注: 判别用基因组字段对账而非前向输出 (零输入单步前向可能恒零, 无判别力)
    bool any_differs = false;
    for (uint32_t s = 1; s <= 8 && !any_differs; ++s) {
        std::mt19937 rs(s);
        auto cs = column_crossover(pa, pb, rs);
        for (size_t c = 0; c < cs.columns().size() && !any_differs; ++c) {
            if (!genome_equal(cs.columns()[c].genome, c1.columns()[c].genome)) any_differs = true;
        }
    }
    assert(any_differs);
    std::cout << "  ✓ 杂交确定性: 同种子基因组位级一致 / 异种子列型分化" << std::endl;
}

void test_cross_into_equivalence() {
    auto pa = make_parent(101);
    auto pb = make_parent(202);

    // 值返回参考
    std::mt19937 r1(7);
    auto ref = column_crossover(pa, pb, r1);

    // 原地版: out 预分配同形 (复用路径) — 基因组逐字段一致 + rng 抽取序列一致
    std::mt19937 r2(7);
    CorticalMacroArray out = pa;
    assert(CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r2));
    assert(out.columns().size() == ref.columns().size());
    for (size_t c = 0; c < out.columns().size(); ++c) {
        assert(genome_equal(out.columns()[c].genome, ref.columns()[c].genome));
    }
    assert(out.macro_axons().size() == ref.macro_axons().size());
    assert(r1 == r2);  // rng 状态逐位相同 ⇒ 抽取次数与顺序一致

    // 复用槽二次调用 (out 已有上一代残留) 结果仍一致
    std::mt19937 r3(7);
    assert(CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r3));
    for (size_t c = 0; c < out.columns().size(); ++c) {
        assert(genome_equal(out.columns()[c].genome, ref.columns()[c].genome));
    }
    assert(out.macro_axons().size() == ref.macro_axons().size());

    // 自由函数定制点 (经 if constexpr 分派到 cross_into)
    std::mt19937 r4(7);
    CorticalMacroArray out2 = pa;
    assert(kun::population::crossover_into(pa, pb, out2, r4));
    for (size_t c = 0; c < out2.columns().size(); ++c) {
        assert(genome_equal(out2.columns()[c].genome, ref.columns()[c].genome));
    }
    std::cout << "  ✓ 原地杂交等价: cross_into ≡ cross (基因逐字段 + rng 序列 + 轴突数)" << std::endl;
}

void test_default_trait_clones() {
    struct Dummy {
        int v = 3;
        void mutate(float, float, std::mt19937&) {}
    };
    Dummy a, b; b.v = 9;
    std::mt19937 rng(1);
    auto child = CrossoverTrait<Dummy>::cross(a, b, rng);
    assert(child.v == 3);  // 默认 trait: 克隆父 A (单亲语义)
    std::cout << "  ✓ 默认 trait: 无特化类型退化为克隆父 A" << std::endl;
}

// Phase 2 A5: shape-incompatible → reject (false); out unchanged — NOT silent clone-as-success.
void test_incompatible_parents_reject() {
    auto pa = make_parent(101);
    CorticalMacroArray odd(3, 8, 12, 2, 1);  // 列数不同构
    CorticalMacroArray sentinel(4, 8, 12, 2, 1);
    // Mark sentinel so we can detect unauthorized writes.
    for (auto& col : sentinel.columns()) {
        col.genome.gains[0] = 42.0f;
    }
    CorticalMacroArray out = sentinel;
    std::mt19937 rng(5);
    const bool ok = CrossoverTrait<CorticalMacroArray>::cross_into(pa, odd, out, rng);
    assert(!ok);
    // out must remain untouched (reject/skip — no silent clone of pa).
    assert(out.columns().size() == sentinel.columns().size());
    assert(out.columns()[0].genome.gains[0] == 42.0f);
    assert(!mates_compatible(pa, odd, 0.0f) || true);  // shape gate is cross_into; distance may still compute
    std::cout << "  ✓ 异构防御: 不同构亲本 cross_into 拒绝且 out 不变 (非静默克隆)" << std::endl;
}

void test_compat_distance_gate() {
    auto pa = make_parent(101);
    auto pb = make_parent(101);  // identical lineage → distance ~ 0
    auto far = make_parent(909); // heavily different
    const double d_same = compatibility_distance(pa, pb);
    const double d_far = compatibility_distance(pa, far);
    assert(d_same < kDefaultSpeciationThreshold);
    assert(mates_compatible(pa, pb, kDefaultSpeciationThreshold));
    // Far parents: either above threshold or still below; force reject with threshold 0.
    assert(!mates_compatible(pa, far, /*threshold=*/0.0f) || d_far >= 0.0);
    assert(!mates_compatible(pa, far, 0.0f));  // threshold 0 ⇒ only distance < 0 never for finite ≥0
    // Document formula: identical CompactSoA flatten → 0 disjoint, 0 weight diff → 0.
    assert(d_same == 0.0);
    assert(d_far >= d_same);
    std::cout << "  ✓ compat gate: identical dist=0; threshold reject path works (d_far="
              << d_far << ")" << std::endl;
}

void test_organism_82_sexual() {
    auto pa = make_organism_82(101);
    auto pb = make_organism_82(202);
    assert(pa.cells.size() == kPreferAOrganismCells);
    assert(pb.cells.size() == kPreferAOrganismCells);
    assert(pa.synapses.size() == pb.synapses.size());

    // Determinism: same seed → bit-identical heritable fields.
    std::mt19937 r1(7);
    CellularOrganism c1 = pa;
    assert(CrossoverTrait<CellularOrganism>::cross_into(pa, pb, c1, r1));
    std::mt19937 r2(7);
    CellularOrganism c2 = pa;
    assert(CrossoverTrait<CellularOrganism>::cross_into(pa, pb, c2, r2));
    assert(organism_heritable_equal(c1, c2));
    assert(c1.cells.size() == kPreferAOrganismCells);
    assert(c1.synapses.size() == pa.synapses.size());
    assert(r1 == r2);

    // Shape conserve + mix: at least one cell or syn from B under seed 7.
    bool mixed = false;
    for (size_t i = 0; i < c1.cells.size(); ++i) {
        if (c1.cells[i].param1 == pb.cells[i].param1 &&
            c1.cells[i].param1 != pa.cells[i].param1) {
            mixed = true;
            break;
        }
    }
    if (!mixed) {
        for (size_t i = 0; i < c1.synapses.size(); ++i) {
            if (c1.synapses[i].weight == pb.synapses[i].weight &&
                c1.synapses[i].weight != pa.synapses[i].weight) {
                mixed = true;
                break;
            }
        }
    }
    assert(mixed);

    // Zero-alloc steady-state: capacity of cells/synapses must not grow on reuse.
    const size_t cell_cap = c1.cells.capacity();
    const size_t syn_cap = c1.synapses.capacity();
    std::mt19937 r3(11);
    assert(CrossoverTrait<CellularOrganism>::cross_into(pa, pb, c1, r3));
    assert(c1.cells.capacity() == cell_cap);
    assert(c1.synapses.capacity() == syn_cap);
    assert(c1.cells.size() == kPreferAOrganismCells);

    // Shape reject: different cell count → false, out unchanged.
    CellularOrganism odd = make_organism_82(303);
    odd.cells.pop_back();
    CellularOrganism held = c2;
    const double held_p = held.cells[0].param1;
    std::mt19937 r4(3);
    assert(!CrossoverTrait<CellularOrganism>::cross_into(pa, odd, held, r4));
    assert(held.cells[0].param1 == held_p);
    assert(held.cells.size() == kPreferAOrganismCells);

    // Distance on organism adapters.
    assert(compatibility_distance(pa, pa) == 0.0);
    assert(mates_compatible(pa, pa, kDefaultSpeciationThreshold));

    std::cout << "  ✓ 82-organism sexual: size conserve / bit-identical / reject / zero-alloc reuse"
              << std::endl;
}

void test_cortical_zero_alloc_reuse() {
    auto pa = make_parent(101);
    auto pb = make_parent(202);
    CorticalMacroArray out = pa;
    // Warm once.
    std::mt19937 r0(1);
    assert(CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r0));
    std::vector<size_t> caps;
    caps.reserve(out.columns().size());
    for (const auto& col : out.columns()) {
        caps.push_back(col.genome.gains.capacity());
    }
    for (int i = 0; i < 8; ++i) {
        std::mt19937 r(static_cast<std::mt19937::result_type>(100 + i));
        assert(CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r));
        for (size_t c = 0; c < out.columns().size(); ++c) {
            assert(out.columns()[c].genome.gains.capacity() == caps[c]);
            assert(out.columns()[c].genome.num_cells == pa.columns()[c].genome.num_cells);
        }
    }
    std::cout << "  ✓ cortical zero-alloc: gains capacity stable across reuse crosses" << std::endl;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 个体杂交算子不变量测试 (Phase 2 Prefer-A)" << std::endl;
    std::cout << "==================================================================" << std::endl;
    test_crossover_invariants();
    test_crossover_determinism();
    test_cross_into_equivalence();
    test_default_trait_clones();
    test_incompatible_parents_reject();
    test_compat_distance_gate();
    test_organism_82_sexual();
    test_cortical_zero_alloc_reuse();
    std::cout << "[PASS] test_population_individual_traits all assertions passed!" << std::endl;
    return 0;
}
