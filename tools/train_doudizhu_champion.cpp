#include "kun/cellular/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace kun;

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  SDSCC 斗地主非完全信息离散博弈演化训练器 (C++20 Native) \n";
    std::cout << "=========================================================\n";

    const int POPULATION_SIZE = 24;
    const int GENERATIONS = 25;
    const uint32_t SEED = 20260903;
    const int MAX_ROUNDS = 40;

    std::vector<uint32_t> train_seeds = {301, 302, 303, 304, 305};
    std::vector<uint32_t> val_seeds   = {401, 402, 403};

    DouDiZhuCardGameTask train_task(MAX_ROUNDS, 42);
    DouDiZhuCardGameTask val_task(MAX_ROUNDS, 99);

    MorphogeneticEvolutionEngine engine(POPULATION_SIZE, SEED, SeedInitMode::HANDCRAFTED_PROGENITOR);

    auto start_time = std::chrono::high_resolution_clock::now();
    double best_val_fitness = -1e9;
    CellularOrganism global_champion;

    for (int gen = 1; gen <= GENERATIONS; ++gen) {
        auto& pop = engine.population();
        double gen_best_train = -1e9;
        size_t best_idx = 0;

        for (size_t i = 0; i < pop.size(); ++i) {
            auto& org = pop[i];
            auto metrics = train_task.evaluate_organism(org, train_seeds, MAX_ROUNDS, true);
            org.fitness_score = metrics.mean_fitness;
            if (metrics.mean_fitness > gen_best_train) {
                gen_best_train = metrics.mean_fitness;
                best_idx = i;
            }
        }

        // 验证集留出评估
        auto val_metrics = val_task.evaluate_organism(pop[best_idx], val_seeds, MAX_ROUNDS, false);
        std::cout << "  Gen " << gen << "/" << GENERATIONS 
                  << " | 训练胜率与进度适应度: " << gen_best_train
                  << " | 留出测试适应度: " << val_metrics.mean_fitness
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
    std::cout << "  最终冠军博弈脑: " << global_champion.cells.size() << " 细胞, " 
              << global_champion.synapses.size() << " 突触, WL 拓扑哈希: " 
              << TaskEvaluator::compute_topology_hash(global_champion) << "\n";

    std::string out_path = "checkpoints/doudizhu_game_champion.json";
    bool saved = global_champion.save_checkpoint_json(out_path);
    if (saved) {
        std::cout << "  [SUCCESS] 真实斗地主生命体已成功存盘至: " << out_path << "\n";
    } else {
        std::cerr << "  [ERROR] 保存失败: " << out_path << "\n";
        return 1;
    }

    std::cout << "=========================================================\n";
    return 0;
}
