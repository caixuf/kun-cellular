// ============================================================================
// test_population_buffer_reuse.cpp — L1 系统层: 双缓冲复用不变量
// 不变量: 多代后规模守恒 / best_fitness 单调不降 / 同种子位级可复现 /
//         每代后无残留待评估哨兵 (迁移关闭时)。
// ============================================================================
#include "kun/cellular/population/ecology_grid.hpp"

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
double fit(Ind& x) {
    double s = 0.0;
    for (double v : x.g) s += v;
    return -std::fabs(s);
}

using kun::population::EcologyGrid;
using kun::population::EvolutionParams;

std::vector<double> run(std::mt19937::result_type seed) {
    EcologyGrid<Ind> grid(3, 16, seed);
    std::mt19937 seeder(seed + 1);
    for (auto& d : grid.demes()) {
        d.seed_population(16, [&seeder](std::mt19937&) {
            Ind x;
            x.g.resize(20);
            std::normal_distribution<double> nd(0.0, 1.0);
            for (auto& v : x.g) v = nd(seeder);
            return x;
        });
    }
    EvolutionParams params;
    params.migration_rate = 0.0f;  // 关闭迁移 ⇒ 每代末不应有残留哨兵
    std::vector<double> trace;
    std::vector<double> prev(grid.demes().size(), -1e18);  // per-deme 前一最优
    for (int gen = 0; gen < 40; ++gen) {
        grid.step_generation(fit, params);
        for (size_t di = 0; di < grid.demes().size(); ++di) {
            const auto& d = grid.demes()[di];
            assert(d.size() == 16);           // 规模守恒
            assert(d.fitness_size() == 16);
            const double best = d.best_fitness();
            assert(best >= prev[di] - 1e-12);  // 精英携带 ⇒ 本 deme 单调不降
            prev[di] = best;
            // 迁移关闭时, 每代结束所有成员都应有真实适应度 (无 -1e18 残留)
            for (double fv : d.fitness()) assert(fv > -1e17);
            trace.push_back(best);
        }
    }
    return trace;
}
}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  L1 系统层: 双缓冲复用不变量测试" << std::endl;
    std::cout << "==================================================================" << std::endl;

    const auto t1 = run(2026);
    const auto t2 = run(2026);
    assert(t1.size() == t2.size());
    for (size_t i = 0; i < t1.size(); ++i) {
        assert(std::memcmp(&t1[i], &t2[i], sizeof(double)) == 0);  // 位级可复现
    }
    std::cout << "  ✓ 40 代规模守恒 / 单调不降 / 无残留哨兵 / 同种子位级一致" << std::endl;

    std::cout << "[PASS] test_population_buffer_reuse all assertions passed!" << std::endl;
    return 0;
}
