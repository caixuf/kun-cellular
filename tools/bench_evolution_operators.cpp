// ============================================================================
// bench_evolution_operators.cpp — Deme::evolve_generation 真成本基准 (H1 直接口径)
//
// 目的: 在 L1 个体 = CorticalMacroArray 的真实演化路径上, 如实度量:
//   [steady] 稳态每代 evolve_generation 的 wall time (排除首代槽位分配)
//   [alloc ] 稳态每代堆分配次数 (全局 operator new 计数) — 判据: 稳态 = 0
//   [warmup] 首代 (槽位一次性构造) 的分配次数, 供对照
//
// 报告工具, 非 ctest 门禁。用法:
//   ./bench_evolution_operators --cols 1024 --cells-per-col 1024 --syns-per-col 8192 --pop 8 --gens 6
// ============================================================================
#include "kun/cellular/cortical_column.hpp"
#include "kun/cellular/population/deme.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <string>

// ── 全局分配计数 (仅本报告工具; 统计稳态 evolve 路径的堆分配次数) ──────────────
namespace { std::atomic<long long> g_new_count{0}; }
void* operator new(std::size_t n) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](std::size_t n) {
    g_new_count.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using kun::CorticalMacroArray;
using kun::population::Deme;
using kun::population::EvolutionParams;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint32_t cols = 1024, cpc = 1024, spc = 8192, in_d = 4, out_d = 2;
    size_t pop = 8, gens = 6, warmup = 2;
    uint32_t seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--cols") cols = (uint32_t)std::atoll(next());
        else if (a == "--cells-per-col") cpc = (uint32_t)std::atoll(next());
        else if (a == "--syns-per-col") spc = (uint32_t)std::atoll(next());
        else if (a == "--pop") pop = (size_t)std::atoll(next());
        else if (a == "--gens") gens = (size_t)std::atoll(next());
        else if (a == "--warmup") warmup = (size_t)std::atoll(next());
        else if (a == "--seed") seed = (uint32_t)std::atoll(next());
    }

    std::printf("=====================================================================\n");
    std::printf("  Deme::evolve_generation 真成本 | %u列 × %u细胞 × %u突触/列 = %llu 细胞 | pop=%zu\n",
                cols, cpc, spc, (unsigned long long)cols * cpc, pop);
    std::printf("=====================================================================\n");

    EvolutionParams params;  // 默认: mut 0.12/0.25, elite 3, tour 3, cross 0.5

    Deme<CorticalMacroArray> deme(0, pop, seed);
    deme.seed_population(pop, [&](std::mt19937&) {
        return CorticalMacroArray(cols, cpc, spc, in_d, out_d);
    });
    // 廉价评估 (列数), 不深拷贝 —— 让 evolve_generation 成为被测项
    auto eval = [](CorticalMacroArray& a) -> double {
        return static_cast<double>(a.columns().size());
    };
    deme.evaluate(eval);

    // ── 首代预热 (槽位一次性构造) ──
    long long warm_allocs = 0;
    {
        g_new_count.store(0);
        for (size_t g = 0; g < warmup; ++g) {
            deme.evolve_generation(params);
            deme.evaluate(eval);
        }
        warm_allocs = g_new_count.load();
    }

    // ── 稳态计时 + 分配计数 ──
    g_new_count.store(0);
    auto t0 = Clock::now();
    for (size_t g = 0; g < gens; ++g) {
        deme.evolve_generation(params);
        deme.evaluate(eval);
    }
    double total_ms = ms_since(t0);
    long long steady_allocs = g_new_count.load();

    std::printf("[steady]  %.2f ms/代 (稳态 %zu 代) | 稳态堆分配 %lld 次 (%.2f 次/代)\n",
                total_ms / (double)gens, gens, steady_allocs, (double)steady_allocs / (double)gens);
    std::printf("[warmup]  首 %zu 代 (槽位一次性构造) 堆分配 %lld 次\n", warmup, warm_allocs);
    std::printf("[hint  ]  稳态分配 = 0 即达 H1「offspring 缓冲复用」判据\n");
    return 0;
}
