#include "tasks/transfer/cross_domain_tasks.hpp"
#include "tasks/transfer/doudizhu_eval_harness.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/cellular_grpo.hpp"
#include "kun/cellular/cellular_relaxation.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include <sstream>

using namespace kun;

/**
 * 动作与效应器通道映射契约:
 *   Action 0: 审慎让牌 (PASS / HOLD)       -> Effector Channel 1 (ACT_PRIMARY_NEGATIVE)
 *   Action 1: 合规跟牌 (FOLLOW / CLEAN)    -> Effector Channel 0 (ACT_PRIMARY_POSITIVE)
 *   Action 2: 强行夺权 (SEIZE / SPRINT)    -> Effector Channel 2 (ACT_DEFENSIVE_RESET)
 */
inline int action_to_channel(int act) {
    switch (act) {
        case 0: return 1; // PASS -> Channel 1
        case 1: return 0; // FOLLOW -> Channel 0
        case 2: return 2; // SEIZE -> Channel 2
        default: return 0;
    }
}

// 快速预评估 (验证训练过程中的胜率与筹码)
static double quick_eval(CellularOrganism& org, int num_episodes, uint32_t seed_base, size_t k_steps, double& out_chips) {
    int wins = 0;
    double total_chips = 0.0;
    RelaxationConfig cfg;
    cfg.max_steps = k_steps;
    cfg.bibo_bound = 50.0;

    for (int i = 0; i < num_episodes; ++i) {
        org.reset_state(true);
        DouDiZhuCardGameTask task(40, seed_base + i * 13 + 7, 5.0);
        while (true) {
            auto obs = task.current_observation();
            std::vector<double> inps(obs.begin(), obs.end());
            auto acts = org.forward_with_relaxation(inps.data(), inps.size(), cfg, nullptr);

            auto res = task.step_continuous(acts);
            if (res.done) {
                if (res.success) wins++;
                total_chips += (res.success ? 200.0 : -200.0);
                break;
            }
        }
    }
    out_chips = total_chips / static_cast<double>(num_episodes);
    return static_cast<double>(wins) / num_episodes * 100.0;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "=================================================================================\n";
    std::cout << " 🧠 硅基细胞计算机 (SDSCC) 斗地主 64-Cell 循环皮层 GRPO + Test-Time Compute 联合训练器\n";
    std::cout << " 理论架构: 32维全息感知 + 64细胞循环微柱 (rho≈0.60 吸引子反馈回路)\n";
    std::cout << "           + DeepSeek-GRPO (Zero-Critic Group Relative Policy Optimization)\n";
    std::cout << "           + BPTT 逆序穿透 K=4 内部松弛思考步 (Co-Trained Test-Time Compute)\n";
    std::cout << "           + 李雅普诺夫 BIBO 稳定流形硬投影 (rho <= 0.95)\n";
    std::cout << "=================================================================================\n\n";

    // 1. 实例化 64 细胞时序循环皮层结构
    CellularOrganism org = build_doudizhu_64cell_recurrent_cortex();
    std::cout << "✅ 成功构建 64 细胞循环皮层微柱架构:\n";
    std::cout << "   - 感知输入维度: 32 维全息特征 (15手牌 + 7高牌记忆 + 6台面态势 + 4角色信息)\n";
    std::cout << "   - 细胞总数: " << org.cells.size() << " (包含 12 个内部循环吸引子核细胞 48..59)\n";
    std::cout << "   - 编译突触总数: " << org.compiled_synapses_.size() << "\n";
    size_t rec_count = 0;
    for (const auto& s : org.compiled_synapses_) if (s.is_recurrent) rec_count++;
    std::cout << "   - 循环反馈突触数: " << rec_count << " (谱半径 rho ≈ 0.60)\n";
    std::cout << "   - Kahn 拓扑执行链节点: " << org.execution_order_.size() << "\n\n";

    // 2. 初始基线盲测 (100 episodes)
    std::cout << "[Step 1] 评测初始未训练模型基线 (100 episodes)...\n";
    double init_chips_k1 = 0.0, init_chips_k4 = 0.0;
    double init_wr_k1 = quick_eval(org, 100, 10001, 1, init_chips_k1);
    double init_wr_k4 = quick_eval(org, 100, 10001, 4, init_chips_k4);
    std::cout << "   ↳ 初始 K=1 (反射) 胜率: " << std::fixed << std::setprecision(1) << init_wr_k1 
              << "% | 筹码: " << init_chips_k1 << " 豆/局\n";
    std::cout << "   ↳ 初始 K=4 (松弛) 胜率: " << std::fixed << std::setprecision(1) << init_wr_k4 
              << "% | 筹码: " << init_chips_k4 << " 豆/局\n\n";

    // 3. 配置 GRPO 优化器与 BPTT 引擎
    const size_t K_RELAX = 4;
    GRPOConfig grpo_cfg;
    grpo_cfg.group_size = 4;        // 每组 4 路异构推演 (同发牌下多路探索)
    grpo_cfg.clip_eps = 0.2f;       // PPO/GRPO 剪裁比率
    grpo_cfg.entropy_coef = 0.012f; // 熵正则系数
    grpo_cfg.learning_rate = 0.010f;// Adam 学习率
    grpo_cfg.grad_clip_norm = 1.0f;
    grpo_cfg.lyapunov_max_gain = 0.95f;

    CellularBPTTEngine bptt_engine(256);
    bptt_engine.init_optimizer(org);
    bptt_engine.grad_clip_norm = grpo_cfg.grad_clip_norm;

    const int TOTAL_EPOCHS = 500;
    const int GROUPS_PER_EPOCH = 16;
    const int GROUP_SIZE = grpo_cfg.group_size; // 4
    const int TOTAL_GAMES = TOTAL_EPOCHS * GROUPS_PER_EPOCH * GROUP_SIZE; // 12,800 局

    std::cout << "[Step 2] 启动大规模 GRPO + Test-Time Compute 联合优化 (总计 " << TOTAL_GAMES << " 局实战):\n";
    std::cout << "   - 轮数: " << TOTAL_EPOCHS << " | 每轮组数: " << GROUPS_PER_EPOCH << " | 每组采样: " << GROUP_SIZE << " 局\n";
    std::cout << "   - 思考松弛步数: K=" << K_RELAX << " (BPTT 全程跨步展开求导)\n";
    std::cout << "   - 剪裁门限: " << grpo_cfg.clip_eps << " | 熵系数: " << grpo_cfg.entropy_coef << "\n";
    std::cout << "   - 初始学习率: " << grpo_cfg.learning_rate << " | 李雅普诺夫边界: rho <= " << grpo_cfg.lyapunov_max_gain << "\n\n";

    std::cout << "---------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8)  << "Epoch"
              << std::setw(14) << "Total Loss"
              << std::setw(14) << "Policy Loss"
              << std::setw(12) << "Entropy"
              << std::setw(18) << "Group Mean Adv"
              << std::setw(14) << "Win Rate"
              << std::setw(16) << "Chip Profit"
              << "Grad Norm\n";
    std::cout << "---------------------------------------------------------------------------------------------------\n";

    std::mt19937 train_rng(20260907);
    BPTTGradients total_grads;

    RelaxationConfig rollout_relax_cfg;
    rollout_relax_cfg.max_steps = K_RELAX;
    rollout_relax_cfg.bibo_bound = 50.0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= TOTAL_EPOCHS; ++epoch) {
        float epoch_total_loss = 0.0f;
        float epoch_pol_loss = 0.0f;
        float epoch_entropy = 0.0f;
        float epoch_abs_adv = 0.0f;
        double epoch_chips = 0.0;
        int epoch_wins = 0;
        int total_rollouts = 0;
        size_t total_steps_count = 0;

        total_grads.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
        total_grads.grad_gains.assign(org.cells.size(), 0.0f);

        float progress = static_cast<float>(epoch - 1) / static_cast<float>(TOTAL_EPOCHS);
        float current_lr = grpo_cfg.learning_rate * (0.2f + 0.8f * 0.5f * (1.0f + std::cos(progress * 3.14159265f)));

        for (int g = 0; g < GROUPS_PER_EPOCH; ++g) {
            uint32_t deal_seed = train_rng();
            GRPOGroup group;
            group.group_id = g;

            for (int r = 0; r < GROUP_SIZE; ++r) {
                org.reset_state(true);
                DouDiZhuCardGameTask task(40, deal_seed, 5.0);
                GRPOTrajectory traj;
                float step_reward_sum = 0.0f;

                while (true) {
                    auto obs = task.current_observation();
                    std::vector<double> inps(obs.begin(), obs.end());

                    RelaxationTelemetry telem;
                    auto acts = org.forward_with_relaxation(inps.data(), inps.size(), rollout_relax_cfg, &telem);

                    float act0 = static_cast<float>(acts.negative_action); // PASS -> Ch 1
                    float act1 = static_cast<float>(acts.positive_action); // FOLLOW -> Ch 0
                    float act2 = static_cast<float>(acts.defensive_reset);  // SEIZE -> Ch 2

                    // 合法动作掩蔽 (Legal Action Masking): 自由出牌权下规则禁止过牌 (Action 0)
                    bool free_lead = (task.table_trick().type == DouDiZhuCardGameTask::TRICK_NONE);
                    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;

                    if (free_lead) {
                        float max_l = std::max(act1, act2);
                        float exp1 = std::exp(act1 - max_l);
                        float exp2 = std::exp(act2 - max_l);
                        float sum_exp = exp1 + exp2;
                        p0 = 0.0f;
                        p1 = exp1 / sum_exp;
                        p2 = exp2 / sum_exp;
                    } else {
                        float max_l = std::max({act0, act1, act2});
                        float exp0 = std::exp(act0 - max_l);
                        float exp1 = std::exp(act1 - max_l);
                        float exp2 = std::exp(act2 - max_l);
                        float sum_exp = exp0 + exp1 + exp2;
                        p0 = exp0 / sum_exp;
                        p1 = exp1 / sum_exp;
                        p2 = exp2 / sum_exp;
                    }

                    std::discrete_distribution<int> act_dist({p0, p1, p2});
                    int chosen_action = act_dist(train_rng);

                    GRPOStepTransition st;
                    st.obs = obs;
                    st.action = chosen_action;
                    st.logits = {act1, act0, act2};
                    st.action_prob = (chosen_action == 0 ? p0 : (chosen_action == 1 ? p1 : p2));
                    traj.steps.push_back(st);

                    float ent = 0.0f;
                    if (p0 > 1e-7f) ent -= p0 * std::log(p0);
                    if (p1 > 1e-7f) ent -= p1 * std::log(p1);
                    if (p2 > 1e-7f) ent -= p2 * std::log(p2);
                    epoch_entropy += ent;
                    total_steps_count++;

                    auto res = task.step(chosen_action);
                    step_reward_sum += static_cast<float>(res.reward);

                    if (res.done) {
                        traj.success = res.success;
                        if (res.success) epoch_wins++;

                        float chips = res.success ? 200.0f : -200.0f;
                        epoch_chips += chips;
                        traj.raw_reward = (chips / 100.0f) + step_reward_sum * 0.05f;
                        break;
                    }
                }

                group.rollouts.push_back(traj);
            }

            compute_group_relative_advantages(group);

            for (const auto& traj : group.rollouts) {
                if (traj.steps.empty()) continue;

                epoch_abs_adv += std::abs(traj.advantage);
                epoch_pol_loss += (-traj.advantage);

                org.reset_state(true);
                bptt_engine.reset_tape();

                std::vector<std::vector<float>> target_outputs;

                for (const auto& st : traj.steps) {
                    std::vector<double> d_in(st.obs.begin(), st.obs.end());

                    for (size_t k = 0; k < K_RELAX; ++k) {
                        org.forward_nd(d_in.data(), d_in.size(), false);
                        bptt_engine.record_step(org);

                        if (k == K_RELAX - 1) {
                            int eff_ch = action_to_channel(st.action);
                            target_outputs.push_back({
                                static_cast<float>(eff_ch),
                                traj.advantage,
                                st.action_prob,
                                grpo_cfg.clip_eps,
                                grpo_cfg.entropy_coef
                            });
                        } else {
                            target_outputs.push_back({
                                0.0f, 0.0f, 1.0f, grpo_cfg.clip_eps, 0.0f
                            });
                        }
                    }
                }

                BPTTGradients step_grads;
                float loss = bptt_engine.backward_with_loss(
                    org, target_outputs, step_grads, SubstrateLossType::GRPO_SURROGATE
                );

                epoch_total_loss += loss;
                total_rollouts++;

                for (size_t k = 0; k < step_grads.grad_synapses.size(); ++k) {
                    total_grads.grad_synapses[k] += step_grads.grad_synapses[k];
                }
                for (size_t k = 0; k < step_grads.grad_gains.size(); ++k) {
                    total_grads.grad_gains[k] += step_grads.grad_gains[k];
                }
            }
        }

        if (total_rollouts > 0) {
            float inv_n = 1.0f / static_cast<float>(total_rollouts);
            double grad_sq = 0.0;
            for (auto& g : total_grads.grad_synapses) {
                g *= inv_n;
                grad_sq += g * g;
            }
            for (auto& g : total_grads.grad_gains) {
                g *= inv_n;
                grad_sq += g * g;
            }
            float grad_norm = static_cast<float>(std::sqrt(grad_sq));

            bptt_engine.step_adam(org, total_grads, current_lr);
            bptt_engine.apply_lyapunov_projection(org, grpo_cfg.lyapunov_max_gain);

            float avg_loss = epoch_total_loss / total_rollouts;
            float avg_abs_adv = epoch_abs_adv / total_rollouts;
            float avg_ent = total_steps_count > 0 ? (epoch_entropy / total_steps_count) : 0.0f;
            float avg_pol = epoch_pol_loss / total_rollouts;
            double avg_chips = epoch_chips / total_rollouts;
            double win_rate = static_cast<double>(epoch_wins) / total_rollouts * 100.0;

            if (epoch % 50 == 0 || epoch == 1 || epoch == TOTAL_EPOCHS) {
                std::cout << std::left << std::setw(8)  << (std::to_string(epoch) + "/" + std::to_string(TOTAL_EPOCHS))
                          << std::setw(14) << std::fixed << std::setprecision(4) << avg_loss
                          << std::setw(14) << std::fixed << std::setprecision(4) << avg_pol
                          << std::setw(12) << std::fixed << std::setprecision(4) << avg_ent
                          << std::setw(18) << std::fixed << std::setprecision(3) << avg_abs_adv
                          << std::setw(14) << (std::to_string(static_cast<int>(win_rate)) + "%")
                          << std::setw(16) << (std::string(avg_chips >= 0 ? "+" : "") + std::to_string(static_cast<int>(avg_chips)) + " 豆")
                          << std::fixed << std::setprecision(4) << grad_norm << "\n";
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "---------------------------------------------------------------------------------------------------\n";
    std::cout << "⏱️ 训练完成! 总耗时: " << std::fixed << std::setprecision(2) << train_sec << " 秒 (" 
              << static_cast<int>(TOTAL_GAMES / train_sec) << " 局/秒)\n\n";

    // 4. 保存模型检查点
    const std::string ckpt_path = "checkpoints/doudizhu_64cell_grpo_cotrained.bin";
    try {
        org.save_checkpoint_bin(ckpt_path);
        std::cout << "💾 [检查点持久化] 成功导出联合微调后的 64 细胞循环皮层检查点: " << ckpt_path << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "⚠️ 导出检查点失败: " << e.what() << "\n";
    }

    // 5. 5,000 局 3-Baseline 严格实证检验
    std::cout << "=========================================================================================\n";
    std::cout << " 🔬 启动 5,000 局 3-Baseline 严格配对实证盲测检验 (McNemar 检定 + Wilson 95% 置信区间)\n";
    std::cout << "=========================================================================================\n";

    const int NUM_EVAL = 5000;
    const uint32_t EVAL_SEED = 20260907;
    DouDiZhuThreeBaselineHarness harness(NUM_EVAL, EVAL_SEED, 5.0, 40);

    const auto& b0 = harness.baseline_0();
    const auto& b1 = harness.baseline_1();

    std::cout << "\n--------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| " << std::left << std::setw(23) << "架构评测通道"
              << "| " << std::setw(12) << "胜场/总数"
              << "| " << std::setw(24) << "胜率 (Wilson 95% CI)"
              << "| " << std::setw(16) << "场均筹码净利"
              << "| " << std::setw(15) << "McNemar vs B1"
              << "| " << std::setw(15) << "McNemar vs K1"
              << "| " << std::setw(14) << "单步时延 (ns)"
              << "|\n";
    std::cout << "|------------------------|-------------|-------------------------|-----------------|----------------|----------------|--------------|\n";

    std::stringstream b0_cnt, b0_ch, b1_cnt, b1_ch;
    b0_cnt << b0.wins << "/" << b0.total_episodes;
    b0_ch << (b0.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b0.avg_chips << " 豆/局";
    std::cout << "| " << std::left << std::setw(23) << "Baseline 0 (随机无门)"
              << "| " << std::setw(12) << b0_cnt.str()
              << "| " << std::setw(24) << b0.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << b0_ch.str()
              << "| " << std::setw(15) << "对照 (Control)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns" << "|\n";

    b1_cnt << b1.wins << "/" << b1.total_episodes;
    b1_ch << (b1.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b1.avg_chips << " 豆/局";
    std::cout << "| " << std::left << std::setw(23) << "Baseline 1 (随机门禁)"
              << "| " << std::setw(12) << b1_cnt.str()
              << "| " << std::setw(24) << b1.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << b1_ch.str()
              << "| " << std::setw(15) << "基准 (Gate>=5)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns" << "|\n";
    std::cout << "|------------------------|-------------|-------------------------|-----------------|----------------|----------------|--------------|\n";

    int k_eval_steps[] = {1, 2, 4, 8};
    ThreeBaselineReport r_k1;

    for (int k : k_eval_steps) {
        RelaxationConfig cfg;
        cfg.max_steps = static_cast<size_t>(k);
        cfg.bibo_bound = 50.0;
        cfg.early_stop = false;

        std::string label = "Target 64-Cell (K=" + std::to_string(k) + ")";
        DouDiZhuThreeBaselineHarness::RelaxationMetrics metrics;
        auto rep = harness.evaluate_organism_relaxation(label, org, cfg, &metrics);
        if (k == 1) r_k1 = rep;

        McNemarResult vs_k1 = mcnemar_test(rep.target.win_records, r_k1.target.win_records, true);

        std::stringstream cnt_ss, ch_ss, vs_b1_ss, vs_k1_ss, lat_ss;
        cnt_ss << rep.target.wins << "/" << rep.target.total_episodes;
        ch_ss << (rep.target.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << rep.target.avg_chips << " 豆/局";

        if (rep.mcnemar_target_vs_b1.p_value < 1e-4) {
            vs_b1_ss << "p=" << std::scientific << std::setprecision(1) << rep.mcnemar_target_vs_b1.p_value;
        } else {
            vs_b1_ss << "p=" << std::fixed << std::setprecision(4) << rep.mcnemar_target_vs_b1.p_value;
        }

        if (k == 1) {
            vs_k1_ss << "基准 (K=1)";
        } else {
            if (vs_k1.p_value < 1e-4) {
                vs_k1_ss << "p=" << std::scientific << std::setprecision(1) << vs_k1.p_value;
            } else {
                vs_k1_ss << "p=" << std::fixed << std::setprecision(4) << vs_k1.p_value;
            }
        }
        lat_ss << std::fixed << std::setprecision(1) << metrics.avg_latency_ns << " ns";

        std::cout << "| " << std::left << std::setw(23) << label
                  << "| " << std::setw(12) << cnt_ss.str()
                  << "| " << std::setw(24) << rep.target.win_rate_ci.to_string(1, true)
                  << "| " << std::setw(16) << ch_ss.str()
                  << "| " << std::setw(15) << vs_b1_ss.str()
                  << "| " << std::setw(15) << vs_k1_ss.str()
                  << "| " << std::setw(14) << lat_ss.str()
                  << "|\n";
    }
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n\n";

    return 0;
}
