// ============================================================================
// train_slingshot_nav.cpp — 混沌三体引力弹弓导航生命体训练器 (organism 真前向)
//
// 流程: 随机基线 -> 形态发生演化 -> Train/Holdout-ID/Holdout-OOD 三隔离门禁
// OOD = 更强引力常数(6.0) + 更大天体质量(1.3x) —— 跨物理参数泛化
// ============================================================================

#include "tasks/robotics/slingshot_nav.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>

using namespace kun;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("==========================================================\n");
    std::printf("  混沌三体引力弹弓导航生命体 · organism 真前向演化\n");
    std::printf("==========================================================\n");

    const size_t POP = 48;
    const size_t GENS = 220;
    const int MAX_STEPS = 400;
    const uint32_t SEED = 20260911;

    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_28;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;

    SlingshotNavTask train_env, id_env;
    SlingshotNavTask::Params ood_p;
    ood_p.G = 1.6;
    ood_p.mass_scale = 1.0;
    SlingshotNavTask ood_env(ood_p);
    train_env.set_max_steps(MAX_STEPS);
    id_env.set_max_steps(MAX_STEPS);
    ood_env.set_max_steps(MAX_STEPS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = "SlingshotNav";
    split.max_steps_per_episode = MAX_STEPS;

    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);

    double base_fit = 0.0;
    {
        auto& pop = engine.population();
        auto m = train_env.evaluate_organism(pop[0], split.train_seeds, MAX_STEPS, false);
        base_fit = m.mean_fitness;
        std::printf("[基线] 未演化祖细胞: 适应度 %.3f (SR %.1f%%)\n\n",
                    m.mean_fitness, m.success_rate * 100.0);
    }

    double best_fit = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= GENS; ++gen) {
        auto& pop = engine.population();
        double gen_best = -1e9, gen_sum = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < pop.size(); ++i) {
            auto m = train_env.evaluate_organism(pop[i], split.train_seeds, MAX_STEPS, false);
            pop[i].fitness_score = m.mean_fitness;
            gen_sum += m.mean_fitness;
            if (m.mean_fitness > gen_best) { gen_best = m.mean_fitness; best_idx = i; }
        }
        if (gen_best > best_fit) { best_fit = gen_best; champion = pop[best_idx]; }
        if (gen % 10 == 0 || gen == 1) {
            std::printf("  Gen %3zu/%zu | best=%.3f mean=%.3f | %zu 细胞 %zu 突触\n",
                        gen, GENS, gen_best, gen_sum / static_cast<double>(POP),
                        pop[best_idx].cells.size(), pop[best_idx].synapses.size());
        }
        if (gen < GENS) engine.evolve_generation();
    }
    const double sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    OOSReport report = TaskEvaluator::evaluate_task_split(
        train_env, id_env, ood_env, champion, split, 0.70);
    if (report.train_metrics.success_rate <= 0.0) {
        report.passes_m1_gate = false;
        report.verdict = "FAIL: 训练成功率 0 (门禁不虚报)";
    }

    std::printf("----------------------------------------------------------\n");
    std::printf("  训练耗时 %.1fs | 随机基线 %.3f -> 冠军 %.3f\n", sec, base_fit, best_fit);
    std::printf("  %s\n", report.verdict.c_str());
    std::printf("  冠军: %zu 细胞 %zu 突触 | WL=%s\n",
                champion.cells.size(), champion.synapses.size(),
                TaskEvaluator::compute_topology_hash(champion).c_str());

    for (auto& s : champion.synapses) s.initial_weight = s.weight;
    champion.save_checkpoint_bin("checkpoints/slingshot_nav_champion.bin");
    std::ofstream rf("checkpoints/slingshot_nav_report.json");
    rf << report.to_json();
    rf.close();
    std::printf("  [SAVED] champion + report\n");
    std::printf("==========================================================\n");
    return report.passes_m1_gate ? 0 : 1;
}
