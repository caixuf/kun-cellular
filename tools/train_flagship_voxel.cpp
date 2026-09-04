// ============================================================================
// train_flagship_voxel.cpp — 百万体素旗舰对照实验 (宪章第5条验收)
//
// 任务: FieldCML2D — N×N 耦合映射晶格二维时空混沌场 (默认 512×512 = 262,144 体素)
//
// 三方对照 (同一柔性混沌地板 ε=0.15, r=3.7, 同一课程退火):
//   [窄]  2 通道契约, 2048 细胞 —— 无规模无宽度地板
//   [宽S1] 全场 N² 体素契约, 65,536 中间神经元 + N² 受体 ≈ 330K 细胞
//          (一受体 ≈ 一体素, 细胞-体素同构)
//   [宽1M] 精英前三发育至 786,432 中间神经元 + N² 受体 ≈ 1.05M 细胞,
//          规模上直接演化 4 代 (带选择压力)
//
// 诚实判据: 未见种子最终评分尺度下的预测质量。规模边际收益 = 1M − S1。
//
// 编译: g++ -O3 -march=native -std=c++20 -I include tools/train_flagship_voxel.cpp -o bin/train_flagship_voxel
// 运行: ./bin/train_flagship_voxel [晶格N] [窄代数] [宽S1代数]
//       默认 N=512, 200/120 代; 冒烟: ./bin/train_flagship_voxel 64 40 24
// ============================================================================

#include "kun/cellular/field_cml_2d.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace kun;

namespace {

double secs_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

EvolutionConstraintConfig voxel_cfg() {
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    cfg.max_cells_limit = 0;              // 无上限 (规模对照实验的生命线)
    cfg.max_synapses_limit = 16000000;
    // 规模对照的控制变量: 代谢能耗衰减 3 个数量级, 让预测质量单独说话
    // (否则 330K/1M 细胞的固定维持成本会淹没选择信号)
    cfg.basal_metabolic_cost = 0.000002;
    cfg.synaptic_metabolic_cost = 0.0000005;
    return cfg;
}

struct VariantResult {
    std::string name;
    double persist = 0, train_best = 0, unseen = 0;
    size_t cells = 0;
    double sec = 0;
};

// ── [窄] 2 通道契约 · 2048 细胞 · 常规种群演化 ──
VariantResult run_narrow(size_t n, size_t gens, uint32_t seed) {
    VariantResult r;
    r.name = "窄 2通道";
    const size_t POP = 12;
    const int MS = 200;
    const std::vector<uint32_t> train_seeds = {11, 12};
    const std::vector<uint32_t> eval_seeds = {21, 22, 23};

    auto t0 = std::chrono::high_resolution_clock::now();
    auto env = std::make_unique<FieldCML2DTask>(n, /*wide=*/false);
    env->set_max_steps(MS);
    r.persist = env->persistence_quality(21, MS);

    auto cfg = voxel_cfg();
    MorphogeneticEvolutionEngine engine(POP, seed, cfg);
    for (auto& org : engine.population()) {
        org.develop_to_scale(2048);
        org.ensure_receptors(2, seed);
        org.wire_global_bridge(256, seed + 1);
    }

    auto curriculum = [&](size_t gen) {
        if (gen <= gens * 0.5) env->set_score_scale(0.60);
        else if (gen <= gens * 0.75) env->set_score_scale(0.30);
        else env->set_score_scale(0.15);
    };

    double best = -1e9;
    for (size_t gen = 1; gen <= gens; ++gen) {
        curriculum(gen);
        auto fits = engine.evaluate_population_parallel(*env, train_seeds, MS, true);
        auto& popv = engine.population();
        for (size_t i = 0; i < popv.size(); ++i) popv[i].fitness_score = fits[i];
        best = std::max(best, *std::max_element(fits.begin(), fits.end()));
        if (gen < gens) engine.evolve_generation();
    }
    r.train_best = best;

    auto& popv = engine.population();
    size_t best_i = 0;
    for (size_t i = 1; i < popv.size(); ++i)
        if (popv[i].fitness_score > popv[best_i].fitness_score) best_i = i;
    env->set_score_scale(0.15);
    r.unseen = env->evaluate_organism(popv[best_i], eval_seeds, MS, false).mean_fitness;
    r.cells = popv[best_i].cells.size();
    r.sec = secs_since(t0);
    return r;
}

// ── [宽] 全场 N² 体素契约 · 65K 中间神经元 → 1M 发育 · 规模上直接演化 ──
VariantResult run_wide(size_t n, size_t gens_a, uint32_t seed) {
    VariantResult r;
    r.name = "宽全场体素";
    const size_t POP = 12, INTER_NEURON_S1 = 65536, INTER_NEURON_1M = 786432;
    const int MS = 200;
    const size_t CH = n * n;   // 全场体素契约宽度
    const std::vector<uint32_t> train_seeds = {11, 12};
    const std::vector<uint32_t> eval_seeds = {21, 22, 23};

    auto t0 = std::chrono::high_resolution_clock::now();
    auto env = std::make_unique<FieldCML2DTask>(n, /*wide=*/true);
    env->set_max_steps(MS);
    r.persist = env->persistence_quality(21, MS);

    auto cfg = voxel_cfg();
    MorphogeneticEvolutionEngine engine(POP, seed, cfg);
    for (auto& org : engine.population()) {
        org.develop_to_scale(INTER_NEURON_S1);
        org.ensure_receptors(CH, seed);
        org.wire_global_bridge(8192, seed + 1);
    }

    auto curriculum = [&](size_t gen) {
        if (gen <= gens_a * 0.5) env->set_score_scale(0.60);
        else if (gen <= gens_a * 0.75) env->set_score_scale(0.30);
        else env->set_score_scale(0.15);
    };

    // ---- S1 阶段: ~330K 细胞规模带种群演化 ----
    double best = -1e9;
    double eval_ms = 0.0, evolve_ms = 0.0;
    auto tick = std::chrono::high_resolution_clock::now;
    for (size_t gen = 1; gen <= gens_a; ++gen) {
        curriculum(gen);
        auto e0 = tick();
        auto fits = engine.evaluate_population_parallel(*env, train_seeds, MS, true);
        auto e1 = tick();
        eval_ms += std::chrono::duration<double, std::milli>(e1 - e0).count();
        auto& popv = engine.population();
        for (size_t i = 0; i < popv.size(); ++i) popv[i].fitness_score = fits[i];
        best = std::max(best, *std::max_element(fits.begin(), fits.end()));
        if (gen < gens_a) {
            engine.evolve_generation();
            evolve_ms += std::chrono::duration<double, std::milli>(tick() - e1).count();
        }
        if (gen % 10 == 0 || gen <= 3 || gen == gens_a) {
            std::printf("  [宽S1] 代 %zu/%zu | 最佳 %.4f | 评估 %.0fs/代 | 演化 %.0fs/代\n",
                        gen, gens_a, best, eval_ms / gen / 1000.0, evolve_ms / gen / 1000.0);
        }
    }
    r.train_best = best;

    auto& popv = engine.population();
    std::vector<size_t> order(popv.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return popv[a].fitness_score > popv[b].fitness_score;
    });
    env->set_score_scale(0.15);
    size_t best_i = order[0];
    r.cells = popv[best_i].cells.size();
    const double unseen_s1 = env->evaluate_organism(popv[best_i], eval_seeds, MS, false).mean_fitness;
    std::printf("  [宽S1] 冠军 %zu 细胞 | 未见种子预测质量 %.4f | 阶段耗时 %.0fs\n",
                r.cells, unseen_s1, secs_since(t0));

    // ---- 1M 阶段: 精英前三发育至 786,432 中间神经元 (≈1.05M 细胞), 规模上直接演化 ----
    MorphogeneticEvolutionEngine engine_1m(3, seed + 9, cfg);
    {
        auto& dst = engine_1m.population();
        for (size_t i = 0; i < 3; ++i) {
            CellularOrganism elite = popv[order[i % order.size()]];
            elite.develop_to_scale(INTER_NEURON_1M);
            elite.ensure_receptors(CH, seed + 3 + static_cast<uint32_t>(i));
            elite.wire_global_bridge(8192, seed + 5 + static_cast<uint32_t>(i));
            dst[i] = std::move(elite);
        }
    }

    const size_t GENS_1M = 4;
    double best_1m = -1e9;
    size_t best_1m_i = 0;
    for (size_t gen = 1; gen <= GENS_1M; ++gen) {
        auto e0 = tick();
        auto fits = engine_1m.evaluate_population_parallel(*env, train_seeds, MS, true, 3);
        auto e1 = tick();
        auto& p1m = engine_1m.population();
        for (size_t i = 0; i < p1m.size(); ++i) p1m[i].fitness_score = fits[i];
        if (fits[std::max_element(fits.begin(), fits.end()) - fits.begin()] > best_1m) {
            best_1m = *std::max_element(fits.begin(), fits.end());
            best_1m_i = std::max_element(fits.begin(), fits.end()) - fits.begin();
        }
        if (gen < GENS_1M) engine_1m.evolve_generation();
        std::printf("  [宽1M] 代 %zu/%zu | 最佳 %.4f | 评估 %.0fs/代\n",
                    gen, GENS_1M, best_1m,
                    std::chrono::duration<double, std::milli>(e1 - e0).count() / 1000.0);
    }

    r.cells = engine_1m.population()[best_1m_i].cells.size();
    r.unseen = env->evaluate_organism(engine_1m.population()[best_1m_i], eval_seeds, MS, false).mean_fitness;
    r.sec = secs_since(t0);
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 512;
    size_t gens_narrow = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 200;
    size_t gens_wide = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 120;

    std::printf("=====================================================================\n");
    std::printf("  百万体素旗舰对照实验 — %zux%zu CML 时空混沌场 (%zu 体素契约)\n", n, n, n * n);
    std::printf("=====================================================================\n");
    std::printf("  硬件: %u 核\n\n", std::thread::hardware_concurrency());

    const uint32_t SEED = 20260903;
    auto narrow = run_narrow(n, gens_narrow, SEED);
    std::printf("\n");
    auto wide = run_wide(n, gens_wide, SEED + 100);

    std::printf("---------------------------------------------------------------------\n");
    std::printf("  %-12s | 持续性 %.4f | 训练最佳 %.4f | 未见 %.4f | %zu 细胞 | %.0fs\n",
                narrow.name.c_str(), narrow.persist, narrow.train_best,
                narrow.unseen, narrow.cells, narrow.sec);
    std::printf("  %-12s | 持续性 %.4f | 训练最佳 %.4f | 未见 %.4f | %zu 细胞 | %.0fs\n",
                wide.name.c_str(), wide.persist, wide.train_best,
                wide.unseen, wide.cells, wide.sec);
    std::printf("---------------------------------------------------------------------\n");
    const double gw = wide.unseen - narrow.unseen;
    std::printf("  宽契约信息增益 (宽-窄): %+.4f %s\n", gw,
                gw > 0.02 ? "→ 场的邻域因果信息被中间神经元利用"
                          : "→ 宽输入暂无增益 (受体网格仍是被动的, 需更多代数/选择压力)");
    std::printf("=====================================================================\n");
    return 0;
}
