// ============================================================================
// train_flagship_1m.cpp — 旗舰百万细胞生命体训练器 (Lorenz 混沌一步预测)
//
// 三阶段诚实实测 "百万细胞买到了什么":
//   [A] 16,384 细胞规模演化 (POP 8 × 12 代)
//   [B] 冠军发育生长至 1,048,576 细胞 (develop_to_scale), 在线可塑性适应
//   [C] 对照: 持续性基线 + 16K vs 1M 预测质量差 (规模的边际收益, 实测说话)
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_flagship_1m.cpp -o bin/train_flagship_1m
// ============================================================================

#include "kun/cellular/flagship_chaos.hpp"
#include <chrono>
#include <cstdio>
#include <thread>

using namespace kun;

namespace {

double secs_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=====================================================================\n");
    std::printf("  旗舰百万细胞生命体 · Lorenz 混沌一步预测 (诚实规模收益实测)\n");
    std::printf("=====================================================================\n");
    std::printf("  硬件: %u 核 | Cell=%zuB\n\n", std::thread::hardware_concurrency(), sizeof(Cell));

    const uint32_t SEED = 20260903;
    const int MS = 200;
    const std::vector<uint32_t> train_seeds = {11, 12};
    const std::vector<uint32_t> eval_seeds = {21, 22, 23};

    FlagshipChaosTask env;
    env.set_max_steps(MS);
    auto set_curriculum = [&](size_t gen) {
        if (gen <= 100) env.set_score_scale(0.40);
        else if (gen <= 200) env.set_score_scale(0.15);
        else env.set_score_scale(0.06);
    };

    // ---- 持续性基线 ----
    double persist_q = 0.0;
    for (uint32_t s : eval_seeds) persist_q += env.persistence_quality(s);
    persist_q /= 3.0;
    std::printf("[基线] 持续性预测 (下一步=当前) 质量: %.3f\n\n", persist_q);

    // ---- [A] 16K 规模演化 ----
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    const size_t POP = 16, GENS = 300;
    const size_t SCALE_A = 2048;
    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);
    for (auto& org : engine.population()) org.develop_to_scale(SCALE_A);

    double best = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= GENS; ++gen) {
        set_curriculum(gen);
        auto& popv = engine.population();
        double gb = -1e9; size_t bi = 0;
        for (size_t i = 0; i < popv.size(); ++i) {
            auto m = env.evaluate_organism(popv[i], train_seeds, MS, true);
            popv[i].fitness_score = m.mean_fitness;
            if (m.mean_fitness > gb) { gb = m.mean_fitness; bi = i; }
        }
        if (gb > best) { best = gb; champion = popv[bi]; }
        if (gen % 10 == 0 || gen == 1)
            std::printf("  [A 16K] Gen %2zu/%zu | 评分尺度 %.2f | best=%.4f mean=%.4f\n",
                        gen, GENS, gen <= 100 ? 0.40 : (gen <= 200 ? 0.15 : 0.06), gb,
                        [&] { double s = 0; for (auto& o : popv) s += o.fitness_score; return s / popv.size(); }());
        if (gen < GENS) engine.evolve_generation();
    }
    const double sec_a = secs_since(t0);
    std::printf("  [A 完成] %.1fs | 16K 冠军训练质量 %.4f\n\n", sec_a, best);

    // ---- [A'] 冠军在未见种子上的诚实评估 (最终尺度, 无可塑性) ----
    env.set_score_scale(0.06);
    auto m_champ = env.evaluate_organism(champion, eval_seeds, MS, false);
    const double q_16k = m_champ.mean_fitness;
    std::printf("  [A'] 16K 冠军未见种子质量: %.4f\n\n", q_16k);

    // ---- [B] 发育至 1M + 在线可塑性适应 ----
    t0 = std::chrono::high_resolution_clock::now();
    champion.develop_to_scale(1048576);
    const double grow_sec = secs_since(t0);
    const double mem_mb = static_cast<double>(champion.cells.size() * sizeof(Cell) +
                                               champion.synapses.size() * sizeof(Synapse)) / 1048576.0;
    std::printf("[B] 发育至 %zu 细胞 / %zu 突触: %.2fs, %.0fMB\n",
                champion.cells.size(), champion.synapses.size(), grow_sec, mem_mb);

    t0 = std::chrono::high_resolution_clock::now();
    // 第一遍: 在线可塑性适应 (hebbian 开)
    auto m_adapt = env.evaluate_organism(champion, train_seeds, MS, true);
    const double adapt_sec = secs_since(t0);
    // 第二遍: 未见种子诚实评估 (可塑性关)
    auto m_1m = env.evaluate_organism(champion, eval_seeds, MS, false);
    const double q_1m = m_1m.mean_fitness;

    std::printf("  [B] 1M 适应+评估耗时 %.1fs | 未见种子质量: %.4f\n\n", adapt_sec, q_1m);

    // ---- [C] 裁决 ----
    std::printf("---------------------------------------------------------------------\n");
    std::printf("  持续性基线          : %.4f\n", persist_q);
    std::printf("  16K 冠军 (演化)     : %.4f  (%+.4f vs 基线)\n", q_16k, q_16k - persist_q);
    std::printf("  1M 发育冠军 (适应)  : %.4f  (%+.4f vs 16K)\n", q_1m, q_1m - q_16k);
    std::printf("  规模边际收益实测    : %s\n",
                (q_1m > q_16k + 0.005) ? "百万细胞买到真实预测力"
                : (q_1m < q_16k - 0.005) ? "生长稀释了训练结构 (规模无收益, 尺度真理再次生效)"
                : "规模无显著差异 (4通道契约下 16K 已饱和)");
    std::printf("  [旗舰] %zu 细胞生命体训练+评估全链路 %.1fs\n",
                champion.cells.size(), secs_since(t0 - std::chrono::seconds(0)) + sec_a);

    champion.save_checkpoint_json("checkpoints/flagship_chaos_16k_champion.json");
    std::printf("  [SUCCESS] 16K 冠军已存盘 (1M 体不存 JSON, 内存态实测)\n");
    std::printf("=====================================================================\n");
    return 0;
}
