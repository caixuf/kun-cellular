// ============================================================================
// test_population_parallel_determinism.cpp — L1 系统层: 并行调度确定性
// 不变量 (H2): 同种子下, 并行/串行、不同 OMP 线程数的演化轨迹逐位一致。
//   并行只作用于「不相交槽位的评估」; migrate 严格串行; 回写定序 ⇒ 调度无关。
// ============================================================================
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/population_eval.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

double fit(const Ind& x) {
    double s = 0.0;
    for (double v : x.g) s += v * v;
    double acc = s;
    for (int i = 0; i < 32; ++i) acc += std::sin(acc * 1e-9 + i * 1e-6);  // 一点真实工作
    return -acc;
}

using kun::population::EcologyGrid;
using kun::population::EvolutionParams;
using kun::population::ExecutionConfig;

std::vector<double> run_trace(bool parallel, float migration, std::mt19937::result_type seed) {
    EcologyGrid<Ind> grid(4, 12, seed, ExecutionConfig{parallel});
    std::mt19937 seeder(seed * 3 + 7);
    for (auto& d : grid.demes()) {
        d.seed_population(12, [&seeder](std::mt19937&) {
            Ind x;
            x.g.resize(24);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (auto& v : x.g) v = nd(seeder);
            return x;
        });
    }
    EvolutionParams params;
    params.migration_rate = migration;
    std::vector<double> trace;
    for (int gen = 0; gen < 20; ++gen) {
        grid.step_generation(fit, params);
        for (const auto& d : grid.demes()) trace.push_back(d.best_fitness());
    }
    return trace;
}

void assert_bitequal(const std::vector<double>& a, const std::vector<double>& b, const char* what) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        assert(std::memcmp(&a[i], &b[i], sizeof(double)) == 0);
    }
    std::cout << "  ✓ " << what << ": " << a.size() << " 个值位级一致" << std::endl;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 并行调度确定性测试 (H2)" << std::endl;
    std::cout << "==================================================================" << std::endl;

    // ── 1. 并行 vs 串行 (含迁移) ──
    const auto t_par = run_trace(true, 0.3f, 2026);
    const auto t_ser = run_trace(false, 0.3f, 2026);
    assert_bitequal(t_par, t_ser, "并行 vs 串行 (migration=0.3)");

#ifdef _OPENMP
    // ── 2. 不同线程数 (并行) ──
    omp_set_num_threads(1);
    const auto t1 = run_trace(true, 0.3f, 2026);
    omp_set_num_threads(8);
    const auto t8 = run_trace(true, 0.3f, 2026);
    assert_bitequal(t1, t8, "OMP 1 线程 vs 8 线程");
    std::cout << "  (OpenMP 启用, 编译期 _OPENMP=" << _OPENMP << ")" << std::endl;
#else
    std::cout << "  (未启用 OpenMP: 仅验证串行等价)" << std::endl;
#endif

    // ── 3. 无迁移时同样一致 ──
    const auto t_par0 = run_trace(true, 0.0f, 31);
    const auto t_ser0 = run_trace(false, 0.0f, 31);
    assert_bitequal(t_par0, t_ser0, "并行 vs 串行 (migration=0)");

    std::cout << "[PASS] test_population_parallel_determinism all assertions passed!" << std::endl;
    return 0;
}
