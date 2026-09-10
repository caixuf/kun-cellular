// ============================================================================
// bench_population_evolution.cpp — 演化速度基准 (L1 系统层, 只读消费者)
//
// 目的: 在改动 L1 之前, 如实记录「种群演化外层循环」的吞吐与阶段拆分。
// 不注册为 ctest (报告工具, 非门禁)。不修改任何 L1 行为。
//
// 度量:
//   [throughput] grid.step_generation 的 generations/sec (端到端)
//   [phase]      逐阶段串行归因: evaluate / evolve / migrate 的时间占比
//   [scaling]    可选: 不同 OMP_NUM_THREADS 下的吞吐 (由外部 env 控制)
//
// 用法:
//   ./bench_population_evolution --demes 8 --pop 16 --gens 200 --work 64
//   ./bench_population_evolution --demes 8 --pop 64 --gens 200 --work 64 --migration 0.0625
//   OMP_NUM_THREADS=4 ./bench_population_evolution ...
// ============================================================================
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/population_eval.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// ── 合成个体: 可配基因组大小 + 变异成本, 使引擎开销与任务数据路解耦 ─────────────
struct SynthIndividual {
    std::vector<double> g;

    void mutate(float rate, float sigma, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::normal_distribution<float> n(0.0f, sigma);
        for (auto& v : g) {
            if (u(rng) < rate) v += n(rng);
        }
    }
};

// 列/元素级均匀杂交 (便宜), 用于让演化阶段的杂交开销可观测
namespace kun {
namespace population {
template <>
struct CrossoverTrait<SynthIndividual> {
    static SynthIndividual cross(const SynthIndividual& a, const SynthIndividual& b, std::mt19937& rng) {
        SynthIndividual c = a;
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        for (size_t i = 0; i < c.g.size(); ++i) {
            if (coin(rng) < 0.5f) c.g[i] = b.g[i];
        }
        return c;
    }
};
}  // namespace population
}  // namespace kun

// ── 评估 functor: gene 求和 + 可配 work 自旋, 使 eval 成本可控且确定 ───────────
struct WorkEval {
    int work;
    double operator()(SynthIndividual& x) const {
        double s = 0.0;
        for (double v : x.g) s += v * v;
        double acc = s;
        for (int i = 0; i < work; ++i) acc += std::sin(acc * 1e-9 + i * 1e-6);
        return acc;
    }
};

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    using namespace kun::population;
    setvbuf(stdout, nullptr, _IONBF, 0);

    // ── 参数解析 ──
    size_t demes = 8, pop = 16, gens = 200, genome = 32;
    int work = 64;
    float migration = 0.0625f;
    size_t elite = 3;
    uint32_t seed = 42;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--demes")        demes = (size_t)std::atoll(next());
        else if (a == "--pop")     pop = (size_t)std::atoll(next());
        else if (a == "--gens")    gens = (size_t)std::atoll(next());
        else if (a == "--work")    work = std::atoi(next());
        else if (a == "--genome")  genome = (size_t)std::atoll(next());
        else if (a == "--migration") migration = (float)std::atof(next());
        else if (a == "--elite")   elite = (size_t)std::atoll(next());
        else if (a == "--seed")    seed = (uint32_t)std::atoll(next());
    }

    EvolutionParams params;
    params.migration_rate = migration;
    params.elite_keep = elite;

    std::printf("==================================================================\n");
    std::printf("  L1 演化基准: %zu deme × %zu 个体 × %zu 代 | work=%d genome=%zu\n",
                demes, pop, gens, work, genome);
    std::printf("  migration=%.4f elite=%zu seed=%u\n", migration, elite, seed);
    std::printf("==================================================================\n");

    WorkEval eval{work};
    size_t genome_sz = genome;

    auto factory = [genome_sz](std::mt19937& rng) {
        SynthIndividual x;
        x.g.resize(genome_sz);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : x.g) v = nd(rng);
        return x;
    };

    // ── 1. 端到端吞吐 (step_generation) ──
    {
        EcologyGrid<SynthIndividual> grid(demes, pop, seed);
        for (auto& d : grid.demes()) d.seed_population(pop, factory);
        auto t0 = Clock::now();
        for (size_t g = 0; g < gens; ++g) grid.step_generation(eval, params);
        double total_ms = ms_since(t0);
        double gens_per_sec = (double)gens / (total_ms / 1000.0);
        double evals = (double)demes * (double)pop * (double)gens;  // 名义评估量
        std::printf("[throughput] 总 %.1f ms | %.2f 代/s | 名义 %.1f 万 评估/s | 迁移 %llu 次\n",
                    total_ms, gens_per_sec, evals / (total_ms / 1000.0) / 1e4,
                    (unsigned long long)grid.total_migrations());
    }

    // ── 2. 阶段归因 (串行调用公开子方法, 复刻当前 step_generation 语义) ──
    //    当前语义: 每 deme: evaluate → evolve → evaluate; 然后 migrate
    {
        EcologyGrid<SynthIndividual> grid(demes, pop, seed);
        for (auto& d : grid.demes()) d.seed_population(pop, factory);
        double eval_ms = 0, evolve_ms = 0, migrate_ms = 0;
        for (size_t g = 0; g < gens; ++g) {
            for (auto& d : grid.demes()) {
                auto t0 = Clock::now(); d.evaluate(eval);      eval_ms += ms_since(t0);
                auto t1 = Clock::now(); d.evolve_generation(params); evolve_ms += ms_since(t1);
                auto t2 = Clock::now(); d.evaluate(eval);      eval_ms += ms_since(t2);
            }
            auto t3 = Clock::now(); grid.migrate(params);      migrate_ms += ms_since(t3);
        }
        double sum = eval_ms + evolve_ms + migrate_ms;
        std::printf("[phase]      evaluate %.1f ms (%.1f%%) | evolve %.1f ms (%.1f%%) | migrate %.1f ms (%.1f%%)\n",
                    eval_ms, 100.0 * eval_ms / sum,
                    evolve_ms, 100.0 * evolve_ms / sum,
                    migrate_ms, 100.0 * migrate_ms / sum);
    }

    std::printf("------------------------------------------------------------------\n");
    std::printf("注: 合成个体仅测「引擎外层开销」; 真实任务 (量化生态) 的 eval 成本远高,\n");
    std::printf("    请另跑 ./build/train_quant_population_ecology 读取其 evo_sec 作为任务口径基线。\n");
    return 0;
}
