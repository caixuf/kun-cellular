#include "tasks/robotics/maze_navigator.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace kun;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  SDSCC 空间迷宫导航拓扑形态发生演化训练器 (C++20 Native) \n";
    std::cout << "=========================================================\n";

    const int POPULATION_SIZE = 36;
    const int GENERATIONS = 35;
    const uint32_t SEED = 20260903;
    const int MAP_SIZE = 15; // 升级迷宫复杂度: 15x15，含环路拓扑与欺骗性死胡同
    const int MAX_STEPS = 160;

    std::vector<uint32_t> train_seeds = {101, 102, 103, 104, 105, 106, 107, 108};
    std::vector<uint32_t> val_seeds   = {201, 202, 203, 204};

    MazeTask train_task(MAP_SIZE, MAP_SIZE, 42, MAX_STEPS, 0.15f);
    MazeTask val_task(MAP_SIZE, MAP_SIZE, 99, MAX_STEPS, 0.15f);

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, SeedInitMode::CONTRACT_PROGENITOR);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_val_fitness = -1e9;
    CellularOrganism global_champion;

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
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
        std::cout << "  Gen " << gen << "/" << GENERATIONS 
                  << " | 最佳训练适应度: " << gen_best_train
                  << " | 留出验证适应度: " << val_metrics.mean_fitness
                  << " | 验证通关率: " << (val_metrics.success_rate * 100.0) << "%"
                  << " | 细胞数: " << pop[best_idx].cells.size()
                  << " | 突触数: " << pop[best_idx].synapses.size() << "\n";

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

    // OOD 跨尺度大迷宫 (21x21 Braided) 留出盲测
    MazeTask ood_task(21, 21, 301, MAX_STEPS * 2, 0.20f);
    std::vector<uint32_t> ood_seeds = {301, 302, 303};
    auto ood_metrics = ood_task.evaluate_organism(global_champion, ood_seeds, MAX_STEPS * 2, false);
    std::cout << "  OOD 跨尺度大迷宫 (21x21 Braided) 盲测适应度: " << ood_metrics.mean_fitness
              << " | 通关成功率: " << (ood_metrics.success_rate * 100.0) << "%\n";

    std::string out_path = "checkpoints/maze_navigation_champion.bin";
    bool saved = global_champion.save_checkpoint_bin(out_path);
    if (saved) {
        std::cout << "  [SUCCESS] 真实迷宫生命体已成功存盘至: " << out_path << "\n";
    } else {
        std::cerr << "  [ERROR] 保存失败: " << out_path << "\n";
        return 1;
    }

    std::cout << "=========================================================\n";
    return 0;
}
