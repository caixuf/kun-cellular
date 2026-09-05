// ============================================================================
// train_cart_pole_tripartite.cpp — 三权分立学习栈 (NEAT + BPTT + Ridge + Lyapunov)
// 倒立摆极速闭环训练器与形式化安全认证
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/tripartite_learning.hpp"
#include "tasks/control/cart_pole_task.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>

using namespace kun;

// 构造倒立摆任务专用全感知微柱拓扑 (4 感知通道 + 2 动态积分/平滑记忆核 + 2 动作效应器)
static CellularOrganism make_cartpole_seed(std::mt19937& rng) {
    CellularOrganism org;
    std::uniform_real_distribution<double> w_dist(-0.2, 0.2);

    // 感觉受体 (4 通道: theta, theta_dot, x, x_dot)
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -30.0f, 0.0f});
    org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -10.0f, 0.0f});
    org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  10.0f, 0.0f});
    org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  30.0f, 0.0f});

    // 内部稳态积分核与阻尼滤波核
    org.cells.push_back({4, CellType::OP_INTEGRAL, 0.05, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f});
    org.cells.push_back({5, CellType::OP_EMA, 0.30, 1.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 20.0f, 0.0f});

    // 动作效应器 (+推力通道 6, -推力通道 7)
    org.cells.push_back({6, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f, -30.0f, 0.0f});
    org.cells.push_back({7, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 100.0f,  30.0f, 0.0f});

    // 突触连接: 受体 -> 效应器
    org.synapses.push_back({0, 6, 0, 0.5 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({0, 7, 0, -0.5 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({1, 6, 0, 0.3 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({1, 7, 0, -0.3 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({2, 6, 0, 0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({2, 7, 0, -0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({3, 6, 0, 0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({3, 7, 0, -0.1 + w_dist(rng), true, 50.0f, -1.0f});

    // 突触连接: 受体 -> 内部记忆核
    org.synapses.push_back({0, 4, 0, 0.2 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({1, 5, 0, 0.2 + w_dist(rng), true, 50.0f, -1.0f});

    // 内部记忆核 -> 效应器
    org.synapses.push_back({4, 6, 0, 0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({4, 7, 0, -0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({5, 6, 0, 0.1 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({5, 7, 0, -0.1 + w_dist(rng), true, 50.0f, -1.0f});

    org.compile();
    return org;
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  SDSCC 三权分立学习栈 · 倒立摆平衡生命体极速训练器 (C++20 Native)\n";
    std::cout << "  NEAT 结构探索 × BPTT 参数微调 × Ridge 读出闭式解 × Lyapunov 流形投影\n";
    std::cout << "======================================================================\n\n";

    const size_t POP_SIZE = 24;
    const uint32_t TRAIN_SEED = 20260905;
    const int MAX_STEPS = 500;

    // 1. 初始化标准训练环境与 OOD 跨参数严苛环境 (重摆锤 0.2kg, 长摆杆 0.7m, 强噪声 2.0N)
    CartPoleBalanceTask train_env;
    train_env.set_max_steps(MAX_STEPS);

    CartPoleBalanceTask::Params ood_p;
    ood_p.masspole = 0.20;
    ood_p.length = 0.70;
    ood_p.force_noise = 2.0;
    CartPoleBalanceTask ood_env(ood_p);
    ood_env.set_max_steps(MAX_STEPS);

    // 2. 初始化三权分立学习栈并用专用拓扑赋予个体
    TripartiteLearningStack stack(POP_SIZE, TRAIN_SEED);
    std::mt19937 seed_rng(TRAIN_SEED);
    for (auto& ind : stack.population) {
        ind.organism = make_cartpole_seed(seed_rng);
        ind.pbt_params.learning_rate = 0.02f;
        ind.pbt_params.mutation_rate = 0.05f;
        ind.pbt_params.bptt_steps = 64;
    }

    // 评估函数：单体在物理环境中运行并收集轨迹与适应度
    auto evaluate_individual = [&](TripartiteIndividual& ind, CartPoleBalanceTask& env, uint32_t seed) {
        env.reset(seed);
        ind.organism.reset_state(false);

        double total_reward = 0.0;
        int survived_steps = 0;

        for (int step = 0; step < MAX_STEPS; ++step) {
            auto obs = env.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = ind.organism.forward(inps);

            auto res = env.step_continuous(acts);
            total_reward += res.reward;
            survived_steps = step + 1;
            if (res.done) break;
        }

        return std::make_pair(survived_steps, total_reward);
    };

    // 3. 初始未学习基线评测
    {
        auto [steps_id, rew_id] = evaluate_individual(stack.population[0], train_env, 101);
        auto [steps_ood, rew_ood] = evaluate_individual(stack.population[0], ood_env, 202);
        std::cout << "[基线] 初始胚胎: ID 存活 " << steps_id << "/" << MAX_STEPS
                  << " 步 | OOD 存活 " << steps_ood << "/" << MAX_STEPS << " 步\n\n";
    }

    // 4. 定义三权分立任务评测与 BPTT 轨迹提取器
    auto task_eval = [&](TripartiteIndividual& ind) {
        double fit_sum = 0.0;
        // 评测 3 个不同扰动初始种子的平均存活率与姿态稳定性
        for (uint32_t s : {101, 103, 107}) {
            auto [steps, rew] = evaluate_individual(ind, train_env, s);
            fit_sum += static_cast<double>(steps) + rew * 0.1;
        }
        ind.task_fitness = fit_sum / 3.0;
        ind.novelty_score = static_cast<double>(ind.organism.synapses.size()) * 0.02;
    };

    auto get_traj = [&](TripartiteIndividual& ind) {
        TrainingTrajectoryBatch batch;
        train_env.reset(101);
        ind.organism.reset_state(false);

        for (int step = 0; step < 64; ++step) {
            auto obs = train_env.current_observation();
            batch.inputs.push_back({obs[0], obs[1], obs[2], obs[3]});

            // 物理自稳反向监督导引: 摆杆倒向右侧 (theta > 0) 时小车必须右推以使支点跟进质心
            // F* = + kp*theta + kd*theta_dot + kx*x + kv*x_dot
            float theta = obs[0] * 0.35f;
            float theta_dot = obs[1] * 3.0f;
            float x = obs[2] * 2.4f;
            float x_dot = obs[3] * 3.0f;

            float target_f = 16.0f * theta + 3.5f * theta_dot + 1.2f * x + 1.0f * x_dot;
            target_f = std::clamp(target_f, -1.0f, 1.0f);

            float pos_t = std::max(0.0f, target_f);
            float neg_t = std::max(0.0f, -target_f);
            batch.targets.push_back({pos_t, neg_t});

            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = ind.organism.forward(inps);
            auto res = train_env.step_continuous(acts);
            if (res.done) break;
        }
        return batch;
    };

    // 5. 开始代际联合训练闭环
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << " 代数  | 适应度      | ID 存活率   | OOD 存活率  | 环路增益 | 细胞/突触\n";
    std::cout << "----------------------------------------------------------------------\n";

    CellularOrganism champion = stack.population[0].organism;
    double best_ood_rate = 0.0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t gen = 0; gen < 15; ++gen) {
        stack.step_generation(task_eval, get_traj);
        auto& top = stack.population[0];

        // 独立跨物理种子盲测 (ID 5 种子, OOD 5 种子)
        int id_passed = 0;
        for (uint32_t s = 301; s <= 305; ++s) {
            auto [st, rw] = evaluate_individual(top, train_env, s);
            if (st >= MAX_STEPS) id_passed++;
        }
        double id_rate = (id_passed / 5.0) * 100.0;

        int ood_passed = 0;
        for (uint32_t s = 401; s <= 405; ++s) {
            auto [st, rw] = evaluate_individual(top, ood_env, s);
            if (st >= MAX_STEPS) ood_passed++;
        }
        double ood_rate = (ood_passed / 5.0) * 100.0;

        std::cout << " Gen " << std::left << std::setw(2) << stack.generation << " | "
                  << std::fixed << std::setprecision(2) << std::setw(11) << top.composite_fitness << " | "
                  << std::setw(9) << (std::to_string(static_cast<int>(id_rate)) + "%") << " | "
                  << std::setw(9) << (std::to_string(static_cast<int>(ood_rate)) + "%") << " | "
                  << std::setw(8) << top.max_loop_gain << " | "
                  << top.organism.cells.size() << " / " << top.organism.synapses.size()
                  << std::endl;

        if (ood_rate > best_ood_rate || (ood_rate == 100.0 && id_rate == 100.0)) {
            best_ood_rate = ood_rate;
            champion = top.organism;
        }

        if (id_rate >= 100.0 && ood_rate >= 100.0) {
            std::cout << "\n🎉 达成双 100% 完美鲁棒控制，提前收敛！\n";
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_s = std::chrono::duration<double>(t_end - t_start).count();

    // 6. 落盘 Champion 检查点
    const std::string ckpt_path = "checkpoints/cartpole_tripartite_champion.bin";
    champion.save_checkpoint_bin(ckpt_path);
    std::cout << "\n[产物] 最优检查点已保存至: " << ckpt_path << " (耗时: " << duration_s << "s)\n";

    // 7. 调用形式化认证管线 (kun_certify) 验证 50,000 步零跌落硬安全证书
    const std::string cert_path = "checkpoints/cartpole_tripartite_champion.bin.cert.json";
    std::string certify_cmd = "./build/kun_certify --organism " + ckpt_path + 
                              " --regression-steps 50000 --out " + cert_path;
    std::cout << "[认证] 正在调用形式化认证管线执行 50,000 步硬实时回归...\n";
    int ret = std::system(certify_cmd.c_str());

    if (ret == 0) {
        std::cout << "✅ 形式化证书生成成功: " << cert_path << "\n";
    } else {
        std::cout << "❌ 形式化认证未通过，请检查 bounds！\n";
    }

    return 0;
}
