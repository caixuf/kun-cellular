// ============================================================================
// train_flagship_wide.cpp — 宽契约旗舰对照实验 v2 (底座手术验收)
//
// v2 方法学修正 (相对 v1):
//   [修正1] 塑性隔离: 每 episode reset_state(true), 杜绝 hebbian 跨种子漂移污染评估
//   [修正2] 软混沌地板: CML ε=0.15, r=3.7 (持续性基线升入可学习梯度带)
//   [修正3] 规模上直接演化: 预训练 2048 → 精英前三发育至 1M → POP3×4代
//           直接在 1M 规模带选择压力演化 (破坏性变异被淘汰, 不再仅靠适应)
//
// 对照: 窄 2通道 vs 宽 32通道, 各自带 2048 与 1M 两个规模的未见种子评估。
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_flagship_wide.cpp -o bin/train_flagship_wide
// ============================================================================

#include "kun/cellular/field_cml.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>

using namespace kun;

namespace {

struct VariantResult {
    std::string name;
    double persist = 0, train_best = 0, unseen_2048 = 0, unseen_1m = 0;
    size_t cells = 0;
    double sec_1m = 0;
};

VariantResult run_variant(const char* name, size_t n_channels, uint32_t seed) {
    VariantResult r;
    r.name = name;
    const size_t POP = 12, GENS = 200, PRETRAIN = 2048;
    const int MS = 200;
    const std::vector<uint32_t> train_seeds = {11, 12};
    const std::vector<uint32_t> eval_seeds = {21, 22};

    auto env = std::make_unique<FieldCMLTask>(32, n_channels, 0.15, 3.7);
    env->set_max_steps(MS);
    r.persist = env->persistence_quality(21, MS);

    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    cfg.max_cells_limit = 0;             // 无上限：细胞规模由动态代谢自然调节
    cfg.max_synapses_limit = 8000000;
    MorphogeneticEvolutionEngine engine(POP, seed, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& org : engine.population()) {
        org.develop_to_scale(PRETRAIN);
        org.ensure_receptors(n_channels, seed);
        org.wire_global_bridge(1024, seed + 1);
    }

    auto curriculum = [&](size_t gen) {
        if (gen <= 100) env->set_score_scale(0.60);
        else if (gen <= 150) env->set_score_scale(0.30);
        else env->set_score_scale(0.15);
    };

    double best = -1e9;
    double eval_ms_total = 0.0, evolve_ms_total = 0.0;
    auto tick = std::chrono::high_resolution_clock::now;
    for (size_t gen = 1; gen <= GENS; ++gen) {
        curriculum(gen);
        auto e0 = tick();
        auto fits = engine.evaluate_population_parallel(*env, train_seeds, MS, true);
        auto e1 = tick();
        eval_ms_total += std::chrono::duration<double, std::milli>(e1 - e0).count();
        auto& popv = engine.population();
        for (size_t i = 0; i < popv.size(); ++i) popv[i].fitness_score = fits[i];
        if (gen < GENS) {
            engine.evolve_generation();
            evolve_ms_total += std::chrono::duration<double, std::milli>(tick() - e1).count();
        }
        if (fits[std::max_element(fits.begin(), fits.end()) - fits.begin()] > best)
            best = *std::max_element(fits.begin(), fits.end());
    }
    r.train_best = best;
    std::printf("  [%s] 预训练 %zu 代: 评估并行 %.1fs (%.0fms/代) | 演化 %.1fs\n",
                name, GENS, eval_ms_total / 1000.0, eval_ms_total / GENS, evolve_ms_total / 1000.0);

    // 2048 冠军未见种子评估
    auto& popv = engine.population();
    size_t best_i = 0;
    for (size_t i = 1; i < popv.size(); ++i)
        if (popv[i].fitness_score > popv[best_i].fitness_score) best_i = i;
    env->set_score_scale(0.15);
    r.unseen_2048 = env->evaluate_organism(popv[best_i], eval_seeds, MS, false).mean_fitness;

    // ---- 精英前三 → 发育至 1M → 规模上直接演化 (带选择压力) ----
    std::vector<size_t> order(popv.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return popv[a].fitness_score > popv[b].fitness_score;
    });

    MorphogeneticEvolutionEngine engine_1m(3, seed + 9, cfg);
    {
        auto& dst = engine_1m.population();
        for (size_t i = 0; i < 3; ++i) {
            CellularOrganism elite = popv[order[i % order.size()]];
            elite.develop_to_scale(1048576);
            elite.ensure_receptors(n_channels, seed + 3 + static_cast<uint32_t>(i));
            elite.wire_global_bridge(8192, seed + 5 + static_cast<uint32_t>(i));
            dst[i] = std::move(elite);
        }
    }

    const size_t GENS_1M = 4;
    double best_1m = -1e9;
    size_t best_1m_i = 0;
    double eval_1m_ms = 0.0, evolve_1m_ms = 0.0;
    for (size_t gen = 1; gen <= GENS_1M; ++gen) {
        auto e0 = tick();
        auto fits = engine_1m.evaluate_population_parallel(*env, train_seeds, MS, true, 3);
        auto e1 = tick();
        eval_1m_ms += std::chrono::duration<double, std::milli>(e1 - e0).count();
        auto& p1m = engine_1m.population();
        for (size_t i = 0; i < p1m.size(); ++i) p1m[i].fitness_score = fits[i];
        if (fits[std::max_element(fits.begin(), fits.end()) - fits.begin()] > best_1m) {
            best_1m = *std::max_element(fits.begin(), fits.end());
            best_1m_i = std::max_element(fits.begin(), fits.end()) - fits.begin();
        }
        if (gen < GENS_1M) {
            engine_1m.evolve_generation();
            evolve_1m_ms += std::chrono::duration<double, std::milli>(tick() - e1).count();
        }
    }
    std::printf("  [%s] 1M 直接演化 %zu 代: 评估(3并行) %.1fs (%.0fms/代) | 演化算子 %.1fs (%.0fms/代)\n",
                name, GENS_1M, eval_1m_ms / 1000.0, eval_1m_ms / GENS_1M,
                evolve_1m_ms / 1000.0, evolve_1m_ms / GENS_1M);
    r.sec_1m = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    r.cells = engine_1m.population()[best_1m_i].cells.size();
    r.unseen_1m = env->evaluate_organism(engine_1m.population()[best_1m_i], eval_seeds, MS, false).mean_fitness;
    return r;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=====================================================================\n");
    std::printf("  宽契约旗舰对照实验 v2 — 塑性隔离 + 软混沌地板 + 1M 规模直接演化\n");
    std::printf("=====================================================================\n");

    const uint32_t SEED = 20260903;
    auto narrow = run_variant("窄 2通道", 2, SEED);
    auto wide   = run_variant("宽 32通道", 32, SEED + 100);

    std::printf("---------------------------------------------------------------------\n");
    std::printf("  %-10s | 持续性 %.3f | 预训练最佳 %.3f | 2048未见 %.3f | 1M直接演化未见 %.3f | %zu 细胞 | 1M阶段 %.0fs\n",
                narrow.name.c_str(), narrow.persist, narrow.train_best,
                narrow.unseen_2048, narrow.unseen_1m, narrow.cells, narrow.sec_1m);
    std::printf("  %-10s | 持续性 %.3f | 预训练最佳 %.3f | 2048未见 %.3f | 1M直接演化未见 %.3f | %zu 细胞 | 1M阶段 %.0fs\n",
                wide.name.c_str(), wide.persist, wide.train_best,
                wide.unseen_2048, wide.unseen_1m, wide.cells, wide.sec_1m);
    std::printf("---------------------------------------------------------------------\n");
    const double gain = wide.unseen_1m - narrow.unseen_1m;
    std::printf("  宽契约信息增益 (1M 规模, 宽-窄): %+.4f %s\n", gain,
                gain > 0.02 ? "→ 宽输入买到真实预测力 (场耦合信息被利用)"
                            : "→ 宽输入暂无增益 (信息未被利用, 需更多代数/更大种群)");
    const double s2048 = wide.unseen_2048 - narrow.unseen_2048;
    std::printf("  宽契约信息增益 (2048 规模):     %+.4f\n", s2048);
    std::printf("=====================================================================\n");
    return 0;
}
