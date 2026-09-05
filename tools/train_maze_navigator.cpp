#include "tasks/robotics/maze_navigator.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace kun;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  SDSCC 空间迷宫导航真泛化演化训练器 (C++20 Native Curriculum)\n";
    std::cout << "=========================================================\n";

    const int POPULATION_SIZE = 36;
    const int GENERATIONS = 35;
    const uint32_t SEED = 20260903;
    const int MAP_SIZE = 11;
    const int MAX_STEPS = 250;

    MazeTask train_task(MAP_SIZE, MAP_SIZE, 42, MAX_STEPS, 0.15f);
    MazeTask val_task(MAP_SIZE, MAP_SIZE, 99, MAX_STEPS, 0.15f);

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, SeedInitMode::CONTRACT_PROGENITOR);

    // 注入反应式廊道居中与信标趋向祖先细胞回路 (避免从零纯随机陷入局部极小)
    {
        OrganismBlueprint bp;
        bp.lineage_name = "ReactiveMazeNavigator";
        bp.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -40.0f, 0.0f});
        bp.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -20.0f, 0.0f});
        bp.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  20.0f, 0.0f});
        bp.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  40.0f, 0.0f});
        bp.cells.push_back({4, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});
        bp.cells.push_back({5, CellType::OP_SUM, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 40.0f, 0.0f, 0.0f});
        bp.cells.push_back({6, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, -20.0f, 0.0f});
        bp.cells.push_back({7, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f,  20.0f, 0.0f});

        bp.synapses.push_back({2, 4, 0, 1.0, true, 50.0f, -1.0f});
        bp.synapses.push_back({1, 4, 1, 1.0, true, 50.0f, -1.0f});
        bp.synapses.push_back({4, 5, 0, 6.0, true, 50.0f, -1.0f});
        bp.synapses.push_back({3, 5, 1, 0.5, true, 50.0f, -1.0f});
        bp.synapses.push_back({5, 7, 0, 1.0, true, 50.0f, -1.0f});
        bp.synapses.push_back({0, 6, 0, 1.5, true, 50.0f, -1.0f});

        for (auto& s : bp.synapses) {
            s.initial_weight = s.weight;
            s.hebbian_rate = 0.0;
        }
        engine.population()[0] = CellularOrganism::create_from_blueprint(bp, 1);
        engine.population()[0].compile();
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_val_fitness = -1e9;
    CellularOrganism global_champion = engine.population()[0];

    std::mt19937 seed_gen(SEED);

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        // 动态随机训练种子池 (每代重抽 8 个新种子，强制泛化，彻底阻断固定地图死记硬背)
        std::vector<uint32_t> train_seeds(8);
        for (auto& s : train_seeds) s = seed_gen() % 100000 + 1000;

        std::vector<uint32_t> val_seeds = {2001, 2002, 2003, 2004};

        auto& pop = engine.population();
        double gen_best_train = -1e9;
        size_t best_idx = 0;

        for (size_t i = 0; i < pop.size(); ++i) {
            auto& org = pop[i];
            auto metrics = train_task.evaluate_organism(org, train_seeds, MAX_STEPS, true);
            org.fitness_score = metrics.mean_fitness;
            if (metrics.mean_fitness > gen_best_train) {
                gen_best_train = metrics.mean_fitness;
                best_idx = i;
            }
        }

        // 验证集留出评估
        auto val_metrics = val_task.evaluate_organism(pop[best_idx], val_seeds, MAX_STEPS, false);
        if (gen % 5 == 0 || gen == 1 || gen == GENERATIONS) {
            std::cout << "  Gen " << gen << "/" << GENERATIONS 
                      << " | 最佳训练适应度: " << gen_best_train
                      << " | 留出验证适应度: " << val_metrics.mean_fitness
                      << " | 验证通关率: " << (val_metrics.success_rate * 100.0) << "%"
                      << " | 细胞数: " << pop[best_idx].cells.size()
                      << " | 突触数: " << pop[best_idx].synapses.size() << "\n";
        }

        if (val_metrics.mean_fitness > best_val_fitness || gen == 1) {
            best_val_fitness = val_metrics.mean_fitness;
            global_champion = pop[best_idx];
        }

        if (gen < GENERATIONS) {
            engine.evolve_generation();
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "---------------------------------------------------------\n";
    std::cout << "  演化收敛完成! 耗时: " << elapsed_sec << " 秒\n";
    std::cout << "  最终冠军个体: " << global_champion.cells.size() << " 细胞, " 
              << global_champion.synapses.size() << " 突触, WL 拓扑哈希: " 
              << TaskEvaluator::compute_topology_hash(global_champion) << "\n";

    // 门禁 3: 100 组全新独立随机种子严格 OOD 盲测
    int ood_passed = 0;
    const int OOD_TOTAL = 100;
    for (int s = 0; s < OOD_TOTAL; ++s) {
        uint32_t unseen_seed = 50000 + s * 17;
        MazeTask ood_eval(MAP_SIZE, MAP_SIZE, unseen_seed, MAX_STEPS, 0.15f);
        auto m = ood_eval.evaluate_organism(global_champion, {unseen_seed}, MAX_STEPS, false);
        if (m.success_rate >= 0.99) ood_passed++;
    }
    double ood_sr = static_cast<double>(ood_passed) / OOD_TOTAL;

    std::cout << "🛡️ 门禁 3 (100 独立随机种子 OOD 盲测): 成功率 = " 
              << (ood_sr * 100.0) << "% (" << ood_passed << "/" << OOD_TOTAL << ")\n";

    std::string out_path = "checkpoints/maze_navigation_champion.bin";
    bool saved = global_champion.save_checkpoint_bin(out_path);
    if (saved) {
        std::cout << "📦 [SUCCESS] 真泛化迷宫生命体已成功存盘至: " << out_path << "\n";
    } else {
        std::cerr << "❌ [ERROR] 保存失败: " << out_path << "\n";
        return 1;
    }

    std::cout << "=========================================================\n";
    return 0;
}
