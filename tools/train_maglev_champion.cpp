#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/domain_zoo.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <random>

using namespace kun;

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=====================================================================\n");
    std::printf("  方案 A: 超导磁悬浮非线性生命体发育演化训练器 (ZooMaglev OOD 突破)\n");
    std::printf("=====================================================================\n");

    const size_t POP = 64;
    const size_t GENS = 120;
    const uint32_t SEED = 20260905;

    // 1. 构造具有生物反射弧骨架的初始祖先 (SENSE -> INTEGRAL + DIFF -> HYSTERESIS -> ACT)
    CellularOrganism progenitor = CellularOrganism::create_seed_organism(1);
    progenitor.lineage_name = "Maglev-Progenitor";
    progenitor.cells.clear();
    progenitor.synapses.clear();

    // 感受受体
    progenitor.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -40.0f, 0.0f});
    progenitor.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,   0.0f, 0.0f});
    progenitor.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  40.0f, 0.0f});

    // 核心代谢与门控动力学神经元
    progenitor.cells.push_back({3, CellType::OP_INTEGRAL, 0.05, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -60.0f, 0.0f}); // 积分累加器 (消除稳态质量误差)
    progenitor.cells.push_back({4, CellType::OP_DIFF,     1.00, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f}); // 微分阻尼器 (抑制磁吸发散振荡)
    progenitor.cells.push_back({5, CellType::OP_EMA,      0.30, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f,  20.0f, 0.0f}); // 低通平滑滤波
    progenitor.cells.push_back({6, CellType::OP_SUM,      1.00, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f,  0.0f, 0.0f}); // 饱和聚合
    progenitor.cells.push_back({7, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f, 0.0f, 0.0f}); // 电流驱动动作头

    // 突触连接
    // 比例通道: height error (Cell 0) -> SUM (Cell 6)
    progenitor.synapses.push_back({0, 6, 0, 1.5, true, 60.0f, -1.0f});
    // 积分通道: height error (Cell 0) -> INTEGRAL (Cell 3) -> SUM (Cell 6)
    progenitor.synapses.push_back({0, 3, 0, 1.0, true, 60.0f, -1.0f});
    progenitor.synapses.push_back({3, 6, 1, 1.2, true, 60.0f, -1.0f});
    // 微分阻尼通道: velocity (Cell 1) -> DIFF (Cell 4) -> SUM (Cell 6)
    progenitor.synapses.push_back({1, 4, 0, 1.0, true, 60.0f, -1.0f});
    progenitor.synapses.push_back({4, 6, 2, 0.8, true, 60.0f, -1.0f});
    // 动作输出: SUM (Cell 6) -> ACT (Cell 7)
    progenitor.synapses.push_back({6, 7, 0, 1.0, true, 60.0f, -1.0f});

    for (auto& s : progenitor.synapses) {
        s.initial_weight = s.weight;
        s.hebbian_rate = 0.0;
    }
    progenitor.compile();

    // 2. 初始化演化引擎
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;

    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);
    // 将精心设计的具有 PID-反射弧的原型注入种群
    engine.population()[0] = progenitor;
    for (size_t i = 1; i < POP; ++i) {
        auto org = progenitor;
        for (int m = 0; m < 2; ++m) engine.mutate(org);
        engine.population()[i] = org;
    }

    // 3. 多生境动态课程演化
    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = "zoo_maglev";
    split.max_steps_per_episode = 600;

    std::mt19937 rng(SEED);
    double best_fitness = -1e9;
    CellularOrganism champion = progenitor;

    std::printf("  [开始] 启动 120 代动态环境演化 (暴露于变质量磁力漂移)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t gen = 1; gen <= GENS; ++gen) {
        // 动态环境采样: 随代际增加质量扰动难度 (从 1.0 逐步泛化至 0.8 ~ 1.8)
        double min_ood = 1.0 - std::min(0.3, gen * 0.003);
        double max_ood = 1.0 + std::min(0.8, gen * 0.008);
        std::uniform_real_distribution<double> dist_ood(min_ood, max_ood);

        double ood1 = dist_ood(rng);
        double ood2 = dist_ood(rng);
        ZooMaglev env1(ood1);
        ZooMaglev env2(ood2);

        auto& popv = engine.population();
        double gen_best_fit = -1e9;
        size_t gen_best_idx = 0;

        for (size_t i = 0; i < popv.size(); ++i) {
            auto m1 = env1.evaluate_organism(popv[i], split.train_seeds, 600, true);
            auto m2 = env2.evaluate_organism(popv[i], split.train_seeds, 600, true);
            double fit = (m1.mean_fitness + m2.mean_fitness) * 0.5;
            popv[i].fitness_score = fit;

            if (fit > gen_best_fit) {
                gen_best_fit = fit;
                gen_best_idx = i;
            }
        }

        if (gen_best_fit > best_fitness) {
            best_fitness = gen_best_fit;
            champion = popv[gen_best_idx];
        }

        if (gen % 20 == 0 || gen == GENS) {
            std::printf("  Gen %3zu/%zu: Best Fit = %8.1f | OOD Range = [%.2f, %.2f] | Champ: %zu cells, %zu syns\n",
                        gen, GENS, best_fitness, min_ood, max_ood, champion.cells.size(), champion.synapses.size());
        }

        if (gen < GENS) engine.evolve_generation();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  [完成] 训练总耗时: %.2fs\n", train_sec);

    // 4. 执行官方严格 M1 三隔离门禁评测 (OOD = 2.0x 质量翻倍盲测，1200 步长跑)
    ZooMaglev train_env(1.0);
    ZooMaglev id_env(1.0);
    ZooMaglev ood_env(2.0); // 100% 质量翻倍！

    OOSReport rep = TaskEvaluator::evaluate_task_split(train_env, id_env, ood_env, champion, split, 0.70);

    std::printf("=====================================================================\n");
    std::printf("  官方 M1 门禁盲测对账结果: %s\n", rep.passes_m1_gate ? "PASSED (100% 达标)" : "FAILED");
    std::printf("  - 训练集表现 (Train SR)  : %5.1f%%\n", rep.train_metrics.success_rate * 100.0);
    std::printf("  - 同分布留出 (Holdout ID): %5.1f%% (比率 x%.2f)\n", rep.holdout_id_metrics.success_rate * 100.0, rep.id_generalization_ratio);
    std::printf("  - 跨尺寸分布外 (OOD 2.0x): %5.1f%% (比率 x%.2f)\n", rep.holdout_ood_metrics.success_rate * 100.0, rep.ood_generalization_ratio);
    std::printf("  - 形态学拓扑指纹         : %s\n", rep.topology_hash.c_str());
    std::printf("=====================================================================\n");

    // 存盘两个路径
    champion.save_checkpoint_bin("checkpoints/maglev_stabilizer_champion.bin");
    champion.save_checkpoint_bin("checkpoints/zoo_maglev.bin");
    std::printf("  [OK] 冠军生命体已无损序列化至 checkpoints/zoo_maglev.bin\n");

    return rep.passes_m1_gate ? 0 : 1;
}
