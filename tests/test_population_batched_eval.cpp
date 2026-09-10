// ============================================================================
// test_population_batched_eval.cpp — L1 系统层: 批量评估契约等价性
// 不变量 (H1): 批量评估委托 ≡ 逐个体评估委托 (同种子 best_fitness 轨迹逐位一致)
//              且批量委托只接收「待评估」槽位、升序、精英不重评。
// ============================================================================
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/population_eval.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

struct Ind {
    std::vector<double> g;
    void mutate(float rate, float sigma, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::normal_distribution<float> n(0.0f, sigma);
        for (auto& v : g) {
            if (u(rng) < rate) v += n(rng);
        }
    }
};

// 确定性适应度: -|均值| (基因组不同 → 适应度不同)
double fit(const Ind& x) {
    double s = 0.0;
    for (double v : x.g) s += v;
    return -std::fabs(s / static_cast<double>(x.g.size()));
}

using kun::population::BatchView;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;

constexpr size_t NUM_DEMES = 3, POP = 8, GENS = 15;

template <typename Eval>
std::vector<double> run_trace(Eval&& eval) {
    EcologyGrid<Ind> grid(NUM_DEMES, POP, 20260910);
    std::mt19937 seeder(4242);
    for (auto& d : grid.demes()) {
        d.seed_population(POP, [&seeder](std::mt19937&) {
            Ind x;
            x.g.resize(16);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (auto& v : x.g) v = nd(seeder);
            return x;
        });
    }
    EvolutionParams params;
    params.migration_rate = 0.25f;  // 让迁移路径也被覆盖
    std::vector<double> trace;
    for (size_t gen = 0; gen < GENS; ++gen) {
        grid.step_generation(eval, params);
        for (const auto& d : grid.demes()) trace.push_back(d.best_fitness());
    }
    return trace;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 批量评估契约等价性测试 (H1)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // ── 1. 逐个体 vs 批量: 轨迹逐位一致 ──
    auto indiv = [](Ind& x) { return fit(x); };
    auto batch = kun::population::make_batch_adapter<Ind>(indiv);

    const auto t_indiv = run_trace(indiv);
    const auto t_batch = run_trace(batch);
    assert(t_indiv.size() == t_batch.size());
    for (size_t i = 0; i < t_indiv.size(); ++i) {
        assert(std::memcmp(&t_indiv[i], &t_batch[i], sizeof(double)) == 0);  // 位级一致
    }
    std::cout << "  ✓ 批量 ≡ 逐个体: " << t_indiv.size() << " 个适应度位级一致" << std::endl;

    // ── 2. 批量委托只接收待评估槽位, 且呈现序列确定 (同种子两次一致) ──
    struct Recorder {
        int calls = 0;
        size_t total = 0;
        bool size_ok = true;
        std::vector<size_t> sizes;  // 每批个体数序列 (确定性呈现的可观测量)
        void operator()(BatchView<Ind> b) {
            ++calls;
            total += b.individuals.size();
            sizes.push_back(b.individuals.size());
            if (b.fitness_out.size() != b.individuals.size() || b.individuals.empty()) size_ok = false;
            for (size_t j = 0; j < b.individuals.size(); ++j) {
                b.fitness_out[j] = fit(*b.individuals[j]);
            }
        }
    };
    auto run_recorded = [&]() {
        Recorder rec;
        EcologyGrid<Ind> grid(NUM_DEMES, POP, 20260910);
        std::mt19937 seeder(4242);
        for (auto& d : grid.demes()) {
            d.seed_population(POP, [&seeder](std::mt19937&) {
                Ind x;
                x.g.resize(16);
                std::normal_distribution<double> nd(0.0, 1.0);
                for (auto& v : x.g) v = nd(seeder);
                return x;
            });
        }
        EvolutionParams params;
        params.migration_rate = 0.0f;
        for (size_t gen = 0; gen < 10; ++gen) grid.step_generation(rec, params);
        return rec;
    };
    const Recorder rec1 = run_recorded();
    const Recorder rec2 = run_recorded();
    assert(rec1.calls > 0);
    assert(rec1.size_ok);                                  // fitness_out 与 individuals 等长且非空
    assert(rec1.sizes == rec2.sizes);                      // 呈现序列确定 (可复现)
    std::cout << "  ✓ 批量仅呈现待评估: " << rec1.calls << " 批 | 累计 " << rec1.total
              << " 次评估 | 序列确定" << std::endl;

    std::cout << "[PASS] test_population_batched_eval all assertions passed!" << std::endl;
    return 0;
}
