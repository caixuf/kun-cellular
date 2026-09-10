// ============================================================================
// test_population_individual_traits.cpp — L1 系统层: 杂交算子不变量测试
// 不变量: 列数守恒 / 前向有限 / 同种子位级复现 / 默认 trait 克隆语义
// ============================================================================
#include "kun/cellular/cortical_column.hpp"
#include "kun/cellular/population/individual_traits.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using kun::CompactSoAGenome;
using kun::CorticalMacroArray;
using kun::population::CrossoverTrait;
using kun::population::column_crossover;
using kun::population::forward_smoke_finite;
using kun::population::genome_equal;

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
    CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r2);
    assert(out.columns().size() == ref.columns().size());
    for (size_t c = 0; c < out.columns().size(); ++c) {
        assert(genome_equal(out.columns()[c].genome, ref.columns()[c].genome));
    }
    assert(out.macro_axons().size() == ref.macro_axons().size());
    assert(r1 == r2);  // rng 状态逐位相同 ⇒ 抽取次数与顺序一致

    // 复用槽二次调用 (out 已有上一代残留) 结果仍一致
    std::mt19937 r3(7);
    CrossoverTrait<CorticalMacroArray>::cross_into(pa, pb, out, r3);
    for (size_t c = 0; c < out.columns().size(); ++c) {
        assert(genome_equal(out.columns()[c].genome, ref.columns()[c].genome));
    }
    assert(out.macro_axons().size() == ref.macro_axons().size());

    // 自由函数定制点 (经 if constexpr 分派到 cross_into)
    std::mt19937 r4(7);
    CorticalMacroArray out2 = pa;
    kun::population::crossover_into(pa, pb, out2, r4);
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

void test_incompatible_parents_degrade() {
    auto pa = make_parent(101);
    CorticalMacroArray odd(3, 8, 12, 2, 1);  // 列数不同构
    std::mt19937 rng(5);
    auto child = column_crossover(pa, odd, rng);
    assert(child.columns().size() == pa.columns().size());  // 诚实降级为克隆父 A
    assert(forward_smoke_finite(child));
    std::cout << "  ✓ 异构防御: 不同构亲本退化为克隆父 A, 不产生非法子代" << std::endl;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 个体杂交算子不变量测试" << std::endl;
    std::cout << "==================================================================" << std::endl;
    test_crossover_invariants();
    test_crossover_determinism();
    test_cross_into_equivalence();
    test_default_trait_clones();
    test_incompatible_parents_degrade();
    std::cout << "[PASS] test_population_individual_traits all assertions passed!" << std::endl;
    return 0;
}
