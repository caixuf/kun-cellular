// ============================================================================
// train_cart_pole.cpp — 倒立摆平衡生命体训练器 (管线横向复刻第一证)
//
// 流程: 随机基线 → 形态发生演化 → Train/Holdout-ID/Holdout-OOD 三隔离门禁
// OOD = 更重摆锤(0.2kg) + 更长摆杆(0.7m) + 推力噪声(2N) —— 跨物理参数泛化
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_cart_pole.cpp -o bin/train_cart_pole
// ============================================================================

#include "kun/cellular/cart_pole_task.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>

using namespace kun;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("==========================================================\n");
    std::printf("  倒立摆平衡生命体 · 管线横向复刻验证 (C++20 Native)\n");
    std::printf("==========================================================\n");

    const size_t POP = 48;
    const size_t GENS = 150;
    const int MAX_STEPS = 300;
    const uint32_t SEED = 20260903;

    // 骨架解锁: 演化必须能长出新的感受器/效应器 (LOCKED 时摆杆信号进不来)
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;

    // 三隔离环境: 训练 / 同分布留出 / 跨物理参数 OOD
    CartPoleBalanceTask train_env, id_env;
    CartPoleBalanceTask::Params ood_p;
    ood_p.masspole = 0.2; ood_p.length = 0.7; ood_p.force_noise = 2.0;
    CartPoleBalanceTask ood_env(ood_p);
    train_env.set_max_steps(MAX_STEPS); id_env.set_max_steps(MAX_STEPS);
    ood_env.set_max_steps(MAX_STEPS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = "CartPoleBalance";
    split.max_steps_per_episode = MAX_STEPS;

    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);

    // ---- 随机基线 (未演化祖细胞) ----
    double base_sr = 0.0;
    {
        auto& pop = engine.population();
        auto m = train_env.evaluate_organism(pop[0], split.train_seeds, MAX_STEPS, false);
        base_sr = m.success_rate;
        std::printf("[基线] 未演化祖细胞: 生存率 %.1f%% (适应度 %.3f)\n\n",
                    m.success_rate * 100.0, m.mean_fitness);
    }

    // ---- 代际演化 ----
    double best_fit = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= GENS; ++gen) {
        auto& pop = engine.population();
        double gen_best = -1e9, gen_sum = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < pop.size(); ++i) {
            auto m = train_env.evaluate_organism(pop[i], split.train_seeds, MAX_STEPS, true);
            pop[i].fitness_score = m.mean_fitness;
            gen_sum += m.mean_fitness;
            if (m.mean_fitness > gen_best) { gen_best = m.mean_fitness; best_idx = i; }
        }
        const bool improved = gen_best > best_fit;
        if (improved) { best_fit = gen_best; champion = pop[best_idx]; }

        if (gen % 5 == 0 || gen == 1 || improved) {
            std::printf("  Gen %2zu/%zu | best=%.3f mean=%.3f | %zu 细胞 %zu 突触\n",
                        gen, GENS, gen_best, gen_sum / static_cast<double>(POP),
                        pop[best_idx].cells.size(), pop[best_idx].synapses.size());
        }
        if (gen < GENS) engine.evolve_generation();
    }
    const double train_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ---- 三隔离门禁终审 (TaskEvaluator 规范路径) ----
    OOSReport report = TaskEvaluator::evaluate_task_split(
        train_env, id_env, ood_env, champion, split, 0.70);

    // 诚实门禁: 生存型任务训练 SR=0 时, 距离回退分支会虚报 PASS —— 强制改判
    if (report.train_metrics.success_rate <= 0.0) {
        report.passes_m1_gate = false;
        report.verdict = "FAIL: 训练生存率为 0 (距离回退分支对生存型任务无效, 不得虚报通过)";
    }

    std::printf("----------------------------------------------------------\n");
    std::printf("  训练耗时 %.1fs | 随机基线生存率 %.1f%% → 冠军 %.1f%%\n",
                train_sec, base_sr * 100.0, report.train_metrics.success_rate * 100.0);
    std::printf("  %s\n", report.verdict.c_str());
    std::printf("  冠军: %zu 细胞 %zu 突触 | WL=%s\n",
                champion.cells.size(), champion.synapses.size(),
                TaskEvaluator::compute_topology_hash(champion).c_str());

    champion.save_checkpoint_json("checkpoints/cartpole_balance_champion.json");
    std::ofstream rf("checkpoints/cartpole_balance_report.json");
    rf << report.to_json();
    rf.close();
    std::printf("  [SUCCESS] 冠军与门禁报告已存盘\n");
    std::printf("==========================================================\n");
    return 0;
}
