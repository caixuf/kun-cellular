#include <iostream>
#include <vector>
#include <cmath>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "kun/cellular/tripartite_learning.hpp"

using namespace kun;

void test_ridge_readout_solver() {
    std::cout << "[Tripartite Test 1] 验证闭式解岭回归求解器 (Ridge Closed-Form Readout)...\n";

    CellularOrganism org;
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -50.0f, 0.0f, 0.0f});
    org.cells.push_back({1, CellType::OP_INTEGRAL, 0.2, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});
    org.cells.push_back({2, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 50.0f, 0.0f, 0.0f});

    org.synapses.push_back({0, 1, 0, 0.8, true, 50.0f, -1.0f});
    org.synapses.push_back({1, 2, 0, 0.1, true, 50.0f, -1.0f}); // 待定权重
    org.compile();

    size_t T = 20;
    std::vector<std::vector<float>> reservoir_tape(T, std::vector<float>(3, 0.0f));
    std::vector<std::vector<float>> target_tape(T, std::vector<float>(1, 0.0f));

    // 假设真实目标权重为 1.75
    for (size_t t = 0; t < T; ++t) {
        float in_val = std::sin(static_cast<float>(t) * 0.4f);
        double inp[4] = {in_val, 0.0, 0.0, 0.0};
        org.forward(inp);
        for (size_t c = 0; c < org.cells.size(); ++c) {
            reservoir_tape[t][c] = static_cast<float>(org.cells[c].output_val);
        }
        // 目标: 1.75 * cell[1]
        target_tape[t][0] = 1.75f * reservoir_tape[t][1];
    }

    // 求解读出突触 (lambda 从 1e-4 收紧至 1e-6，理论收缩偏差 < 0.1%)
    std::vector<size_t> eff_idx = {2};
    bool ok = RidgeReadoutSolver::solve_organism_readouts(org, reservoir_tape, target_tape, eff_idx, 1e-6f);
    assert(ok && "Ridge solver should succeed");

    // 验证求解出的权重非常接近 1.75
    double solved_w = org.synapses[1].weight;
    std::cout << "  ↳ 岭回归求解读出权重: 目标=1.75, 求解值=" << solved_w << "\n";
    assert(std::abs(solved_w - 1.75) < 0.05);

    // 验证前向误差极小
    float max_err = 0.0f;
    for (size_t t = 0; t < T; ++t) {
        float pred = static_cast<float>(reservoir_tape[t][1] * solved_w);
        float err = std::abs(pred - target_tape[t][0]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  ↳ 读出层最大绝对拟合残差: " << max_err << "\n";
    assert(max_err < 0.02f);
}

void test_tripartite_learning_stack_loop() {
    std::cout << "[Tripartite Test 2] 验证三权分立学习机闭环 (NEAT 结构 + BPTT 梯度 + Ridge 读出 + PBT 调度)...\n";

    TripartiteLearningStack stack(16, 123);
    assert(stack.population.size() == 16);

    // 模拟简易环境评测器
    auto task_eval = [](TripartiteIndividual& ind) {
        double acc = 0.0;
        for (int step = 0; step < 10; ++step) {
            double inps[4] = {static_cast<double>(step) * 0.1, 0.5, 0.0, 0.0};
            auto acts = ind.organism.forward(inps);
            acc += acts.positive_action - acts.negative_action;
        }
        ind.task_fitness = std::max(0.0, acc);
        ind.novelty_score = static_cast<double>(ind.organism.synapses.size()) * 0.05;
    };

    auto get_traj = [](TripartiteIndividual& /*ind*/) {
        TrainingTrajectoryBatch batch;
        for (int t = 0; t < 12; ++t) {
            batch.inputs.push_back({static_cast<float>(t) * 0.1f, 0.5f, 0.0f, 0.0f});
            batch.targets.push_back({0.8f, -0.8f});
        }
        return batch;
    };

    double initial_top_fit = 0.0;
    for (size_t gen = 0; gen < 5; ++gen) {
        stack.step_generation(task_eval, get_traj);
        double top_fit = stack.population[0].composite_fitness;
        if (gen == 0) initial_top_fit = top_fit;

        std::cout << "  ↳ 代数 Gen " << stack.generation 
                  << " | 冠军适应度: " << top_fit 
                  << " | 最大环路增益: " << stack.population[0].max_loop_gain
                  << " | 细胞数: " << stack.population[0].organism.cells.size()
                  << " | 突触数: " << stack.population[0].organism.synapses.size()
                  << "\n";

        // 验证每代最优个体的李雅普诺夫稳定认证
        assert(stack.population[0].max_loop_gain <= 0.95f + 1e-5f);
    }

    std::cout << "  ↳ 初始冠军适应度: " << initial_top_fit 
              << " -> 5 代后最优适应度: " << stack.population[0].composite_fitness << "\n";
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << " SDSCC 三权分立学习栈 (NEAT结构 + BPTT参数 + Ridge读出) 单元测试\n";
    std::cout << "===================================================================\n";

    test_ridge_readout_solver();
    test_tripartite_learning_stack_loop();

    std::cout << "\n🎉 全部三权分立学习栈 (NEAT + BPTT + Ridge + PBT) 闭环测试通过!\n";
    return 0;
}
