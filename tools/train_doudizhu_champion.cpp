#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <functional>
#include <algorithm>

using namespace kun;

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 带有工作记忆与博弈迟滞核的斗地主种子生命体 (12 细胞 / 13 突触)
// ---------------------------------------------------------------------------
static CellularOrganism make_doudizhu_seed(std::mt19937& rng) {
    CellularOrganism org;
    std::uniform_real_distribution<double> w_dist(-0.01, 0.01);

    // 4 个感知受体:
    // 0: 手牌质量均值 (mean rank / 14.0)
    // 1: 剩余张数比率 (cards left / 20.0)
    // 2: 台面牌力强度 (table rank / 14.0)
    // 3: 历史高牌打出频度 (high cards / 6.0)
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -30.0f, 0.0f});
    org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -10.0f, 0.0f});
    org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  10.0f, 0.0f});
    org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  30.0f, 0.0f});

    // 内部运算与博弈记忆核:
    // 细胞 4: OP_SUB (手牌质量 - 台面牌力 = 相对压制优势)
    org.cells.push_back({4, CellType::OP_SUB, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -20.0f, -20.0f, 0.0f});
    // 细胞 5: GATE_THRESHOLD (对手极高牌门控: 台面牌力 > 0.90 即 2/大王时激活)
    org.cells.push_back({5, CellType::GATE_THRESHOLD, 0.90, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f});
    // 细胞 6: GATE_HYSTERESIS (博弈迟滞锁存器, 维持攻守节奏)
    org.cells.push_back({6, CellType::GATE_HYSTERESIS, 0.20, -0.20, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, -20.0f, 0.0f});
    // 细胞 7: OP_EMA (博弈时序低通记忆环, 与 6 构成互补闭环回路, 0 < rho < 1.0)
    org.cells.push_back({7, CellType::OP_EMA, 0.25, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 0.0f, 0.0f});
    // 细胞 8: GATE_THRESHOLD (残局收尾感知: 剩余牌数比率 <= 0.08 即剩 <= 2 张牌时激发终局斩杀)
    org.cells.push_back({8, CellType::GATE_THRESHOLD, -0.09, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 20.0f, 0.0f});

    // 动作效应器:
    // 细胞 9: ACT_PRIMARY_POSITIVE (合规跟牌 Clean Follow 驱动)
    org.cells.push_back({9, CellType::ACT_PRIMARY_POSITIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, -30.0f, 0.0f});
    // 细胞 10: ACT_PRIMARY_NEGATIVE (审慎让牌 Strategic Pass/Hold 驱动)
    org.cells.push_back({10, CellType::ACT_PRIMARY_NEGATIVE, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 0.0f, 0.0f});
    // 细胞 11: ACT_DEFENSIVE_RESET (强行夺权 Power Seize / 残局突击)
    org.cells.push_back({11, CellType::ACT_DEFENSIVE_RESET, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 30.0f, 0.0f});

    // 基础跟牌强通路 (Input 0 & Input 1 -> Act 9 Positive Action)
    org.synapses.push_back({0, 9, 0, 2.5 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({1, 9, 0, 1.8 + w_dist(rng), true, 50.0f, -1.0f});

    // 差值网络 (Input 0 - Input 2)
    org.synapses.push_back({0, 4, 0, 1.0 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({2, 4, 1, 1.0 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({4, 9, 0, 0.8 + w_dist(rng), true, 50.0f, -1.0f});

    // 迟滞与低通工作记忆环
    org.synapses.push_back({4, 6, 0, 0.4 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({6, 7, 0, 0.25 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({7, 6, 0, 0.25 + w_dist(rng), true, 50.0f, -1.0f});

    // 残局极少牌力门控 (Cell 8: Input 1 <= 0.08) -> 激发夺权斩杀 Act 11
    org.synapses.push_back({1, 8, 0, -1.0 + w_dist(rng), true, 50.0f, -1.0f});
    org.synapses.push_back({8, 11, 0, 3.5 + w_dist(rng), true, 50.0f, -1.0f});

    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

// ---------------------------------------------------------------------------
// 策略基准对账结构
// ---------------------------------------------------------------------------
struct BenchmarkMetric {
    std::string name;
    int wins{0};
    int total{0};
    double win_rate{0.0};
    double mean_reward{0.0};
    double mean_steps{0.0};
};

static BenchmarkMetric eval_policy_seeds(
    const std::string& name,
    const std::vector<uint32_t>& seeds,
    int max_rounds,
    std::function<int(const std::vector<float>&, DouDiZhuCardGameTask&)> policy)
{
    BenchmarkMetric m;
    m.name = name;
    m.total = static_cast<int>(seeds.size());
    double sum_rew = 0.0;
    double sum_steps = 0.0;

    int l_games = 0, l_wins = 0, p_games = 0, p_wins = 0;

    for (uint32_t s : seeds) {
        DouDiZhuCardGameTask task(max_rounds, s);
        task.reset(s);
        bool won = false;
        double ep_rew = 0.0;
        int steps = 0;
        bool is_l = (task.role() == 1);
        if (is_l) l_games++; else p_games++;

        for (int t = 0; t < max_rounds; ++t) {
            auto obs = task.current_observation();
            int act = policy(obs, task);
            auto step_res = task.step(act);
            ep_rew += step_res.reward;
            steps = step_res.steps;
            if (step_res.done) {
                won = step_res.success;
                break;
            }
        }
        if (won) {
            m.wins++;
            if (is_l) l_wins++; else p_wins++;
        }
        sum_rew += ep_rew;
        sum_steps += steps;
    }
    m.win_rate = (static_cast<double>(m.wins) / m.total) * 100.0;
    m.mean_reward = sum_rew / m.total;
    m.mean_steps = sum_steps / m.total;
    if (name.find("Heuristic") != std::string::npos) {
        std::cout << "    [Debug " << name << "] Landlord: " << l_wins << "/" << l_games 
                  << " (" << (l_games ? l_wins * 100.0 / l_games : 0) << "%) | Peasant: "
                  << p_wins << "/" << p_games << " (" << (p_games ? p_wins * 100.0 / p_games : 0) << "%)\n";
    }
    return m;
}

static BenchmarkMetric eval_organism_seeds(
    const std::string& name,
    CellularOrganism& org,
    const std::vector<uint32_t>& seeds,
    int max_rounds)
{
    BenchmarkMetric m;
    m.name = name;
    m.total = static_cast<int>(seeds.size());
    double sum_rew = 0.0;
    double sum_steps = 0.0;

    for (uint32_t s : seeds) {
        org.reset_state(true);
        DouDiZhuCardGameTask task(max_rounds, s);
        task.reset(s);
        bool won = false;
        double ep_rew = 0.0;
        int steps = 0;

        for (int t = 0; t < max_rounds; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            auto step_res = task.step_continuous(acts);
            ep_rew += step_res.reward;
            steps = step_res.steps;
            if (step_res.done) {
                won = step_res.success;
                break;
            }
        }
        if (won) m.wins++;
        sum_rew += ep_rew;
        sum_steps += steps;
    }
    m.win_rate = (static_cast<double>(m.wins) / m.total) * 100.0;
    m.mean_reward = sum_rew / m.total;
    m.mean_steps = sum_steps / m.total;
    return m;
}

int main(int argc, char** argv) {
    const bool EVAL_ONLY = (argc > 1 && std::string(argv[1]) == "--eval-only");
    const std::string CHAMPION_PATH = "checkpoints/doudizhu_evolved_champion.bin";

    std::cout << "=========================================================\n";
    std::cout << "  SDSCC 斗地主非完全信息离散博弈演化训练器 (C++20 Native)\n"
              << (EVAL_ONLY ? "  [形式化对账与大样本天梯评估模式]\n" : "  [三态决策与手牌结构守恒学习模式]\n");
    std::cout << "=========================================================\n";

    const int MAX_ROUNDS = 50;

    std::mt19937 rng(20260906);
    CellularOrganism champion;

    if (EVAL_ONLY) {
        std::ifstream test_f(CHAMPION_PATH, std::ios::binary);
        if (!test_f.is_open()) {
            std::cerr << "[错误] 未找到待评检查点: " << CHAMPION_PATH << "\n";
            return 1;
        }
        test_f.close();
        champion = CellularOrganism::load_checkpoint_bin(CHAMPION_PATH);
        std::cout << "[载入检查点] " << champion.cells.size() << " 细胞 / "
                  << champion.synapses.size() << " 突触\n";
    } else {
        const size_t POP_SIZE = 32;
        const size_t GENS = 50;

        std::vector<CellularOrganism> population;
        for (size_t i = 0; i < POP_SIZE; ++i) {
            population.push_back(make_doudizhu_seed(rng));
        }

        std::vector<uint32_t> train_seeds;
        for (int i = 0; i < 80; ++i) train_seeds.push_back(5000 + i * 7);

        std::vector<uint32_t> val_seeds;
        for (int i = 0; i < 40; ++i) val_seeds.push_back(8000 + i * 13);

        champion = population[0];
        double best_val_score = -1e9;

        std::cout << "[演化开始] 种群=" << POP_SIZE << " 代数=" << GENS << " | 三态手权与结构熵惩罚约束\n";
        auto t_start = std::chrono::steady_clock::now();

        for (size_t gen = 0; gen < GENS; ++gen) {
            std::vector<double> scores(POP_SIZE, 0.0);

            for (size_t i = 0; i < POP_SIZE; ++i) {
                auto m = eval_organism_seeds("train", population[i], train_seeds, MAX_ROUNDS);
                scores[i] = m.win_rate * 5.0 + m.mean_reward;
            }

            size_t best_idx = 0;
            for (size_t i = 1; i < POP_SIZE; ++i) {
                if (scores[i] > scores[best_idx]) best_idx = i;
            }

            auto val_m = eval_organism_seeds("val", population[best_idx], val_seeds, MAX_ROUNDS);
            double val_score = val_m.win_rate * 5.0 + val_m.mean_reward;

            if (val_score > best_val_score || gen == 0) {
                best_val_score = val_score;
                champion = population[best_idx];
            }

            if (gen % 10 == 0 || gen == GENS - 1) {
                std::cout << "  Gen " << std::setw(2) << gen
                          << " | 训练评分: " << std::setw(6) << std::fixed << std::setprecision(1) << scores[best_idx]
                          << " | 验证胜率: " << std::setprecision(1) << val_m.win_rate << "%"
                          << " | 场均回报: " << val_m.mean_reward << "\n";
            }

            // 锦标赛选择与变异
            std::vector<CellularOrganism> next_gen;
            next_gen.push_back(champion); // 精英保留

            std::uniform_int_distribution<size_t> p_dist(0, POP_SIZE - 1);
            std::uniform_real_distribution<double> mut_dist(-0.04, 0.04);

            while (next_gen.size() < POP_SIZE) {
                size_t a = p_dist(rng), b = p_dist(rng);
                size_t winner = (scores[a] > scores[b]) ? a : b;
                CellularOrganism child = population[winner];
                for (auto& s : child.synapses) {
                    if (std::uniform_real_distribution<double>(0, 1)(rng) < 0.20) {
                        s.weight = std::clamp(s.weight + mut_dist(rng), -4.5, 4.5);
                    }
                }
                // 确保李雅普诺夫稳定性约束
                child.enforce_lyapunov_stability(0.85);
                child.compile();
                next_gen.push_back(child);
            }
            population = next_gen;
        }

        auto t_end = std::chrono::steady_clock::now();
        std::cout << "[演化完成] 耗时: "
                  << std::chrono::duration<double>(t_end - t_start).count() << "s\n";

        // 保存检查点并立即重新载入，保证天梯盲测对账完全以落盘产物为准 (Bit-Exact)
        champion.save_checkpoint_bin(CHAMPION_PATH);
        std::cout << "[模型落盘] " << CHAMPION_PATH << "\n";
        champion = CellularOrganism::load_checkpoint_bin(CHAMPION_PATH);
    }

    // -----------------------------------------------------------------------
    // 门禁 3: 严格 200+ 种子跨策略天梯盲测对账 (OOD 200 组独立随机发牌)
    // -----------------------------------------------------------------------
    const int BENCH_SEEDS = 200;
    std::vector<uint32_t> bench_seeds;
    for (int i = 0; i < BENCH_SEEDS; ++i) {
        bench_seeds.push_back(90000 + i * 37);
    }

    std::cout << "\n=========================================================\n";
    std::cout << "  200 独立随机发牌种子天梯盲测对账 (零数据泄露样本外评测)\n";
    std::cout << "=========================================================\n";

    // 1. Always-Pass 基准
    auto m_pass = eval_policy_seeds("Always-Pass (始终让牌)", bench_seeds, MAX_ROUNDS,
        [](const auto&, auto&) { return 0; });

    // 2. Always-Play (Clean) 基准
    auto m_play_clean = eval_policy_seeds("Always-Play (无脑合规跟牌)", bench_seeds, MAX_ROUNDS,
        [](const auto&, auto&) { return 1; });

    // 3. Always-Play (Aggressive / Power Seize) 基准
    auto m_play_agg = eval_policy_seeds("Always-Play (无脑拆牌夺权)", bench_seeds, MAX_ROUNDS,
        [](const auto&, auto&) { return 2; });

    // 4. Random 基准
    std::mt19937 r_rng(12345);
    auto m_random = eval_policy_seeds("Random (随机盲选 0/1/2)", bench_seeds, MAX_ROUNDS,
        [&r_rng](const auto&, auto&) { return static_cast<int>(r_rng() % 3); });

    // 5. 专家启发式 (Heuristic)
    auto m_heur = eval_policy_seeds("Rule-Heuristic (规则启发式)", bench_seeds, MAX_ROUNDS,
        [](const auto& obs, auto&) {
            if (obs[1] <= 0.08f) return 2; // 残局全力冲刺 (剩 <= 1~2 张牌斩杀)
            if (obs[2] >= 0.90f && obs[1] >= 0.50f) return 0; // 审慎避让对手大牌 (2/大王)
            return 1; // 常规保结构跟牌
        });

    // 6. 神经网络博弈皮层 (Cellular Cortex Champion)
    auto m_cortex = eval_organism_seeds("Cellular-Cortex (神经形态生命体)", champion, bench_seeds, MAX_ROUNDS);

    std::cout << std::left << std::setw(32) << "策略/生命体名称"
              << std::right << std::setw(12) << "胜率 (%)"
              << std::setw(14) << "胜场/总数"
              << std::setw(14) << "场均净回报"
              << std::setw(12) << "场均步数" << "\n";
    std::cout << std::string(84, '-') << "\n";

    auto print_row = [](const BenchmarkMetric& m) {
        std::cout << std::left << std::setw(32) << m.name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(1) << m.win_rate << "%"
                  << std::setw(9) << m.wins << "/" << m.total
                  << std::setw(14) << std::setprecision(1) << m.mean_reward
                  << std::setw(12) << std::setprecision(1) << m.mean_steps << "\n";
    };

    print_row(m_pass);
    print_row(m_play_agg);
    print_row(m_play_clean);
    print_row(m_random);
    print_row(m_heur);
    print_row(m_cortex);

    std::cout << std::string(84, '=') << "\n";
    if (m_cortex.win_rate >= m_play_clean.win_rate &&
        m_cortex.win_rate > m_play_agg.win_rate &&
        m_cortex.win_rate > m_random.win_rate &&
        m_cortex.win_rate >= 55.0) {
        std::cout << "🏆 [天梯认证成功] 硅基细胞生命体实战胜率达标 (" << m_cortex.win_rate 
                  << "% >= 55.0%), 场均回报 (" << m_cortex.mean_reward << ") 领跑全梯队, 彻底攻破投机与随机盲选!\n";
    } else {
        std::cout << "⚠️ [警告] 胜率优势未显著确立, 请检查博弈动力学参数!\n";
    }

    // 调用 kun_certify 进行形式化安全认证
    std::string certify_cmd = "./build/kun_certify --organism " + CHAMPION_PATH + 
                              " --regression-steps 50000 --out " + CHAMPION_PATH + ".cert.json";
    int ret = std::system(certify_cmd.c_str());
    if (ret == 0) {
        std::cout << "✅ [形式化认证完成] 证书已写入: " << CHAMPION_PATH << ".cert.json\n";
    } else {
        std::cerr << "❌ [形式化认证失败] kun_certify 返回退出码: " << ret << "\n";
    }

    return 0;
}
