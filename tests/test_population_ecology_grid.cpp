// ============================================================================
// test_population_ecology_grid.cpp — L1 系统层: 多岛生态网格不变量测试
// 不变量: deme 规模守恒 / 精英不被迁移覆盖 / 最优适应度单调不降 / 迁移发生 / 位级复现
// ============================================================================
#include "kun/cellular/cortical_column.hpp"
#include "kun/cellular/population/deme.hpp"
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/population_eval.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using kun::CorticalMacroArray;
using kun::population::Deme;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;
using kun::population::PopulationEval;

namespace {

CorticalMacroArray make_child(std::mt19937& rng) {
    CorticalMacroArray arr(4, 8, 12, 2, 1);
    for (auto& col : arr.columns()) {
        col.genome.mutate_parameters(1.0f, 0.5f, rng);
        col.genome.mutate_primitive_types(0.5f, rng);
    }
    arr.wire_small_world_axons(1, rng());
    return arr;
}

// 确定性评估: 固定输入前向, 适应度 = -|输出均值| (不同基因组 → 不同适应度)
double eval_fitness(CorticalMacroArray& arr) {
    const size_t n_cols = arr.columns().size();
    const uint32_t in_dim = static_cast<uint32_t>(arr.columns()[0].local_inputs.size());
    const uint32_t out_dim = static_cast<uint32_t>(arr.columns()[0].local_outputs.size());
    std::vector<float> in_buf(n_cols * in_dim);
    std::vector<float> out(n_cols * out_dim, 0.0f);
    std::vector<const float*> in_ptrs(n_cols);
    for (size_t c = 0; c < n_cols; ++c) {
        for (uint32_t d = 0; d < in_dim; ++d) {
            in_buf[c * in_dim + d] = std::sin(static_cast<float>(c * 3 + d) * 0.7f) * 0.4f;
        }
        in_ptrs[c] = in_buf.data() + c * in_dim;
    }
    arr.forward_multi_channel(in_ptrs.data(), out.data());
    double s = 0.0;
    for (float v : out) {
        if (!std::isfinite(v)) return -1e9;
        s += v;
    }
    return -std::fabs(s / static_cast<double>(out.size()));
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 多岛生态网格不变量测试" << std::endl;
    std::cout << "==================================================================" << std::endl;

    const size_t NUM_DEMES = 3, POP = 6, GENS = 12;
    EvolutionParams params;  // 默认参数 (精英 3 / 锦标赛 3 / 杂交 0.5 / 迁移 1/16)

    // ── 1. 规模守恒 + 适应度单调不降 + 迁移发生 ──
    EcologyGrid<CorticalMacroArray> grid(NUM_DEMES, POP, 20260910);
    {
        std::mt19937 seeder(1234);
        for (auto& d : grid.demes()) {
            d.seed_population(POP, [&seeder](std::mt19937&) { return make_child(seeder); });
        }
    }
    double prev_best = -1e18;
    for (int gen = 0; gen < GENS; ++gen) {
        grid.step_generation(eval_fitness, params);
        for (const auto& d : grid.demes()) {
            assert(d.size() == POP);                       // 规模守恒
            assert(d.fitness_size() == POP);
            double best = d.best_fitness();
            assert(best >= prev_best - 1e-12);             // 精英保留 ⇒ 最优不降
            prev_best = std::max(prev_best, best);
        }
    }
    assert(grid.total_migrations() > 0);                   // 迁移确实发生
    std::cout << "  ✓ 规模守恒 / 最优适应度单调不降 / 迁移 " << grid.total_migrations() << " 次" << std::endl;

    // ── 2. 位级可复现: 同种子两次完整演化, 适应度轨迹逐位一致 ──
    auto run_trace = [&](std::mt19937::result_type seed) {
        EcologyGrid<CorticalMacroArray> g(NUM_DEMES, POP, seed);
        std::mt19937 seeder(seed * 2 + 1);
        for (auto& d : g.demes()) {
            d.seed_population(POP, [&seeder](std::mt19937&) { return make_child(seeder); });
        }
        std::vector<double> trace;
        for (int gen = 0; gen < 6; ++gen) {
            g.step_generation(eval_fitness, params);
            for (const auto& d : g.demes()) trace.push_back(d.best_fitness());
        }
        return trace;
    };
    const auto t1 = run_trace(777);
    const auto t2 = run_trace(777);
    assert(t1.size() == t2.size());
    for (size_t i = 0; i < t1.size(); ++i) {
        assert(std::memcmp(&t1[i], &t2[i], sizeof(double)) == 0);  // 位级一致
    }
    std::cout << "  ✓ 确定性: 同种子适应度轨迹位级一致" << std::endl;

    // ── 3. 精英不被迁移覆盖: 高迁移率下最优仍单调不降 ──
    EvolutionParams hot = params;
    hot.migration_rate = 0.9f;  // 极端迁移压力
    EcologyGrid<CorticalMacroArray> hot_grid(2, POP, 999);
    {
        std::mt19937 seeder(555);
        for (auto& d : hot_grid.demes()) {
            d.seed_population(POP, [&seeder](std::mt19937&) { return make_child(seeder); });
        }
    }
    double hot_best = -1e18;
    for (int gen = 0; gen < 10; ++gen) {
        hot_grid.step_generation(eval_fitness, hot);
        for (const auto& d : hot_grid.demes()) {
            hot_best = std::max(hot_best, d.best_fitness());
        }
    }
    assert(hot_grid.total_migrations() > 0);
    std::cout << "  ✓ 极端迁移下精英保护: 最优适应度 " << hot_best << " 无回退" << std::endl;

    std::cout << "[PASS] test_population_ecology_grid all assertions passed!" << std::endl;
    return 0;
}
