#include "tasks/transfer/cross_domain_tasks.hpp"
#include "tasks/transfer/doudizhu_eval_harness.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/cellular_grpo.hpp"
#include "kun/cellular/cellular_relaxation.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>
#include <sstream>

using namespace kun;

inline int action_to_channel(int act) {
    switch (act) {
        case 0: return 1; // PASS -> Effector Ch 1 (ACT_PRIMARY_NEGATIVE)
        case 1: return 0; // FOLLOW -> Effector Ch 0 (ACT_PRIMARY_POSITIVE)
        case 2: return 2; // SEIZE -> Effector Ch 2 (ACT_DEFENSIVE_RESET)
        default: return 0;
    }
}

// 快速预评估
static double quick_eval_moe(
    CellularOrganism& bid_org,
    CellularOrganism& landlord_org,
    CellularOrganism& peasant_org,
    int num_episodes,
    uint32_t seed_base,
    double threshold,
    double& out_chips,
    int& out_landlord_wins,
    int& out_landlord_games,
    int& out_peasant_wins,
    int& out_peasant_games)
{
    int wins = 0;
    double total_chips = 0.0;
    out_landlord_wins = 0;
    out_landlord_games = 0;
    out_peasant_wins = 0;
    out_peasant_games = 0;

    auto bid_eval = [&](const std::vector<float>& hand_obs, double opp_max_score) -> bool {
        auto b = bid_org;
        b.reset_state(true);
        std::vector<double> inps(hand_obs.begin(), hand_obs.end());
        auto acts = b.forward_nd(inps.data(), inps.size(), false);
        double net = acts.positive_action - acts.negative_action;
        double bid_score = 12.0 + 10.0 * net;
        return (bid_score >= threshold && bid_score >= opp_max_score);
    };

    for (int i = 0; i < num_episodes; ++i) {
        uint32_t s = seed_base + static_cast<uint32_t>(i) * 19 + 7;
        DouDiZhuCardGameTask task(40, s, threshold, bid_eval);

        auto l_org = landlord_org;
        auto p_org = peasant_org;
        l_org.reset_state(true);
        p_org.reset_state(true);

        bool is_landlord = (task.role() == 1);
        if (is_landlord) out_landlord_games++;
        else out_peasant_games++;

        while (true) {
            auto obs = task.current_observation();
            std::vector<double> inps(obs.begin(), obs.end());

            CellularOrganism::ActionOutputs acts;
            if (task.role() == 1) {
                acts = l_org.forward_nd(inps.data(), inps.size(), false);
            } else {
                acts = p_org.forward_nd(inps.data(), inps.size(), false);
            }

            auto res = task.step_continuous(acts);
            if (res.done) {
                if (res.success) {
                    wins++;
                    if (is_landlord) out_landlord_wins++;
                    else out_peasant_wins++;
                }
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

    std::cout << "====================================================================================================\n";
    std::cout << " 🧠 硅基细胞计算机 (SDSCC) 斗地主端到端神经叫牌 + MoE 双角色专精微柱 GRPO 联合训练器\n";
    std::cout << " 架构规范:\n";
    std::cout << "   1. 神经叫牌中枢 (Neural Bidding Column): 32维手牌特征 -> 64细胞动力学图谱 -> 自主控盘门限决断\n";
    std::cout << "   2. 地主进攻专精微柱 (Expert 0: Landlord Assault Column): 强化领打与冲刺控制\n";
    std::cout << "   3. 农民协作专精微柱 (Expert 1: Peasant Cooperation Column): 强化盟友让牌护送与顶家死卡地主\n";
    std::cout << "   4. 角色解耦路由 (Router): 彻底消除地主与农民梯度冲突 (Zero Role Conflict)\n";
    std::cout << "   5. 校准专家基准线: 对齐真实专家门禁 (Bidding Gate = 17.5 / 16.0)\n";
    std::cout << "====================================================================================================\n\n";

    const double CALIBRATED_THRESHOLD = 17.5;
    // 修复 GRPO 叫牌塌缩 (net 塌至 -2.24, 地主场率 0%): 门控解耦 + 可选冻结
    // DZ_MODEL_BID_GATE: 最终评测时模型侧叫牌门限 (任务/对手门控仍为 CALIBRATED_THRESHOLD)
    // DZ_FREEZE_BID=1: 冻结叫牌皮层于健康先验 (阻止 GRPO 把它推向永不叫地主的退化吸引子)
    const double MODEL_BID_GATE = []() {
        const char* e = std::getenv("DZ_MODEL_BID_GATE");
        return e ? std::atof(e) : 17.5;
    }();
    const bool FREEZE_BID = []() {
        const char* e = std::getenv("DZ_FREEZE_BID");
        return e ? (std::atoi(e) != 0) : true;   // 默认冻结 (先验 18.8%@17.5 已证健康)
    }();
    std::cout << "   - 模型叫牌门控: " << MODEL_BID_GATE << " | 冻结叫牌皮层: " << (FREEZE_BID ? "是" : "否") << "\n";
    const std::string base_ckpt = "checkpoints/doudizhu_64cell_grpo_cotrained.bin";

    // 1. 实例化三大皮层微柱
    CellularOrganism bid_org = build_doudizhu_bidding_cortex();
    CellularOrganism landlord_org = build_doudizhu_landlord_cortex(base_ckpt);
    CellularOrganism peasant_org = build_doudizhu_peasant_cortex(base_ckpt);

    std::cout << "✅ 成功构建神经叫牌与 MoE 双角色微柱结构:\n";
    std::cout << "   - 叫牌微柱细胞数: " << bid_org.cells.size() << " | 突触数: " << bid_org.compiled_synapses_.size() << "\n";
    std::cout << "   - 地主微柱细胞数: " << landlord_org.cells.size() << " | 突触数: " << landlord_org.compiled_synapses_.size() << "\n";
    std::cout << "   - 农民微柱细胞数: " << peasant_org.cells.size() << " | 突触数: " << peasant_org.compiled_synapses_.size() << "\n\n";

    // 2. 初始状态快速评估
    std::cout << "[Step 1] 评测初始未训练微柱集群基线 (300 episodes, Gate=" << CALIBRATED_THRESHOLD << ")...\n";
    double init_chips = 0.0;
    int l_wins = 0, l_games = 0, p_wins = 0, p_games = 0;
    double init_wr = quick_eval_moe(bid_org, landlord_org, peasant_org, 300, 10001, CALIBRATED_THRESHOLD,
                                   init_chips, l_wins, l_games, p_wins, p_games);
    std::cout << "   ↳ 初始全局胜率: " << std::fixed << std::setprecision(1) << init_wr << "% | 筹码: " << init_chips << " 豆/局\n";
    std::cout << "   ↳ 地主角色: " << l_wins << "/" << l_games << " (" << (l_games > 0 ? (double)l_wins/l_games*100.0 : 0.0) << "%)\n";
    std::cout << "   ↳ 农民角色: " << p_wins << "/" << p_games << " (" << (p_games > 0 ? (double)p_wins/p_games*100.0 : 0.0) << "%)\n\n";

    // 3. 配置 GRPO 优化器与 BPTT 引擎
    GRPOConfig grpo_cfg;
    grpo_cfg.group_size = 4;
    grpo_cfg.clip_eps = 0.2f;
    grpo_cfg.entropy_coef = 0.012f;
    grpo_cfg.learning_rate = 0.008f;
    grpo_cfg.grad_clip_norm = 1.0f;
    grpo_cfg.lyapunov_max_gain = 0.95f;

    CellularBPTTEngine bptt_bid(64);
    bptt_bid.init_optimizer(bid_org);
    bptt_bid.grad_clip_norm = grpo_cfg.grad_clip_norm;

    CellularBPTTEngine bptt_landlord(256);
    bptt_landlord.init_optimizer(landlord_org);
    bptt_landlord.grad_clip_norm = grpo_cfg.grad_clip_norm;

    CellularBPTTEngine bptt_peasant(256);
    bptt_peasant.init_optimizer(peasant_org);
    bptt_peasant.grad_clip_norm = grpo_cfg.grad_clip_norm;

    const int TOTAL_EPOCHS = 300;
    const int GROUPS_PER_EPOCH = 16;
    const int GROUP_SIZE = grpo_cfg.group_size; // 4
    const int TOTAL_GAMES = TOTAL_EPOCHS * GROUPS_PER_EPOCH * GROUP_SIZE; // 19,200 局

    std::cout << "[Step 2] 启动神经叫牌 + MoE 双角色专精联合 GRPO 优化 (总计 " << TOTAL_GAMES << " 局实战):\n";
    std::cout << "   - 训练轮数: " << TOTAL_EPOCHS << " | 每轮组数: " << GROUPS_PER_EPOCH << " | 每组采样: " << GROUP_SIZE << " 局\n";
    std::cout << "   - 剪裁门限: " << grpo_cfg.clip_eps << " | 初始学习率: " << grpo_cfg.learning_rate << "\n\n";

    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(8)  << "Epoch"
              << std::setw(14) << "Total Loss"
              << std::setw(12) << "Entropy"
              << std::setw(16) << "Mean Adv"
              << std::setw(14) << "Win Rate"
              << std::setw(16) << "Chip Profit"
              << std::setw(16) << "Landlord WR"
              << "Peasant WR\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";

    std::mt19937 train_rng(20260907);

    struct RolloutRecord {
        bool bid_decision{false};
        float bid_prob{0.5f};
        std::vector<float> hand_obs;
        int role{0}; // 1: landlord, 0: peasant
        GRPOTrajectory traj;
    };

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= TOTAL_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_entropy = 0.0f;
        float epoch_adv = 0.0f;
        double epoch_chips = 0.0;
        int epoch_wins = 0;
        int epoch_landlord_wins = 0, epoch_landlord_games = 0;
        int epoch_peasant_wins = 0, epoch_peasant_games = 0;

        BPTTGradients grads_bid;
        grads_bid.grad_synapses.assign(bid_org.compiled_synapses_.size(), 0.0f);
        grads_bid.grad_gains.assign(bid_org.cells.size(), 0.0f);

        BPTTGradients grads_l;
        grads_l.grad_synapses.assign(landlord_org.compiled_synapses_.size(), 0.0f);
        grads_l.grad_gains.assign(landlord_org.cells.size(), 0.0f);

        BPTTGradients grads_p;
        grads_p.grad_synapses.assign(peasant_org.compiled_synapses_.size(), 0.0f);
        grads_p.grad_gains.assign(peasant_org.cells.size(), 0.0f);

        int count_bid = 0, count_l = 0, count_p = 0;

        float progress = static_cast<float>(epoch - 1) / static_cast<float>(TOTAL_EPOCHS);
        float current_lr = grpo_cfg.learning_rate * (0.30f + 0.70f * 0.5f * (1.0f + std::cos(progress * 3.14159265f)));

        for (int g = 0; g < GROUPS_PER_EPOCH; ++g) {
            uint32_t deal_seed = train_rng();
            GRPOGroup group;
            group.group_id = g;

            std::vector<RolloutRecord> rollouts;
            rollouts.reserve(GROUP_SIZE);

            for (int r = 0; r < GROUP_SIZE; ++r) {
                RolloutRecord rr;
                rr.hand_obs.resize(32, 0.0f);

                auto bid_eval_train = [&](const std::vector<float>& hand_obs, double opp_max_score) -> bool {
                    rr.hand_obs = hand_obs;
                    std::vector<double> d_in(hand_obs.begin(), hand_obs.end());
                    auto b_eval = bid_org;
                    b_eval.reset_state(true);
                    auto acts = b_eval.forward_nd(d_in.data(), d_in.size(), false);

                    double net = acts.positive_action - acts.negative_action;
                    double bid_score = 12.0 + 10.0 * net;

                    // 探索采样
                    double diff = (bid_score - std::max(CALIBRATED_THRESHOLD, opp_max_score)) / 1.2;
                    float p_bid = std::clamp(static_cast<float>(1.0 / (1.0 + std::exp(-diff))), 0.01f, 0.99f);

                    std::bernoulli_distribution b_dist(p_bid);
                    bool decision = b_dist(train_rng);

                    rr.bid_decision = decision;
                    rr.bid_prob = decision ? p_bid : (1.0f - p_bid);
                    return decision;
                };

                DouDiZhuCardGameTask task(40, deal_seed, CALIBRATED_THRESHOLD, bid_eval_train);
                rr.role = task.role();
                if (rr.role == 1) epoch_landlord_games++;
                else epoch_peasant_games++;

                float step_reward_sum = 0.0f;
                CellularOrganism active_expert = (rr.role == 1) ? landlord_org : peasant_org;
                active_expert.reset_state(true);

                while (true) {
                    auto obs = task.current_observation();
                    std::vector<double> inps(obs.begin(), obs.end());

                    auto acts = active_expert.forward_nd(inps.data(), inps.size(), false);

                    float act0 = static_cast<float>(acts.negative_action); // PASS
                    float act1 = static_cast<float>(acts.positive_action); // FOLLOW
                    float act2 = static_cast<float>(acts.defensive_reset);  // SEIZE

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
                    int chosen = act_dist(train_rng);

                    GRPOStepTransition st;
                    st.obs = obs;
                    st.action = chosen;
                    st.logits = {act1, act0, act2};
                    st.action_prob = (chosen == 0 ? p0 : (chosen == 1 ? p1 : p2));
                    rr.traj.steps.push_back(st);

                    float ent = 0.0f;
                    if (p0 > 1e-7f) ent -= p0 * std::log(p0);
                    if (p1 > 1e-7f) ent -= p1 * std::log(p1);
                    if (p2 > 1e-7f) ent -= p2 * std::log(p2);
                    epoch_entropy += ent;

                    auto res = task.step(chosen);
                    step_reward_sum += static_cast<float>(res.reward);

                    if (res.done) {
                        rr.traj.success = res.success;
                        if (res.success) {
                            epoch_wins++;
                            if (rr.role == 1) epoch_landlord_wins++;
                            else epoch_peasant_wins++;
                        }

                        // 真实斗地主经济杠杆: 地主输赢翻倍 (输承担双倍惩罚 -400)
                        float chips = 0.0f;
                        if (rr.role == 1) {
                            chips = res.success ? 400.0f : -400.0f;
                        } else {
                            chips = res.success ? 200.0f : -200.0f;
                        }
                        epoch_chips += (res.success ? 200.0f : -200.0f);
                        rr.traj.raw_reward = (chips / 100.0f) + step_reward_sum * 0.05f;
                        break;
                    }
                }

                group.rollouts.push_back(rr.traj);
                rollouts.push_back(std::move(rr));
            }

            compute_group_relative_advantages(group);

            for (size_t idx = 0; idx < rollouts.size(); ++idx) {
                const auto& r_rec = rollouts[idx];
                float adv = group.rollouts[idx].advantage;
                epoch_adv += std::abs(adv);

                // --- 1. 神经叫牌微柱 BPTT 求导与更新 ---
                if (!r_rec.hand_obs.empty()) {
                    bid_org.reset_state(true);
                    bptt_bid.reset_tape();

                    std::vector<double> d_in(r_rec.hand_obs.begin(), r_rec.hand_obs.end());
                    bid_org.forward_nd(d_in.data(), d_in.size(), false);
                    bptt_bid.record_step(bid_org);

                    // 叫牌动作映射: 叫牌对应通道 0 (ACT_PRIMARY_POSITIVE), 放弃对应通道 1 (ACT_PRIMARY_NEGATIVE)
                    int bid_ch = r_rec.bid_decision ? 0 : 1;
                    std::vector<std::vector<float>> bid_target = {{
                        static_cast<float>(bid_ch),
                        adv,
                        r_rec.bid_prob,
                        grpo_cfg.clip_eps,
                        grpo_cfg.entropy_coef
                    }};

                    BPTTGradients bid_step_grads;
                    float b_loss = bptt_bid.backward_with_loss(
                        bid_org, bid_target, bid_step_grads, SubstrateLossType::GRPO_SURROGATE
                    );
                    epoch_loss += b_loss;
                    count_bid++;

                    for (size_t k = 0; k < bid_step_grads.grad_synapses.size(); ++k) {
                        grads_bid.grad_synapses[k] += bid_step_grads.grad_synapses[k];
                    }
                    for (size_t k = 0; k < bid_step_grads.grad_gains.size(); ++k) {
                        grads_bid.grad_gains[k] += bid_step_grads.grad_gains[k];
                    }
                }

                // --- 2. 对应角色专精微柱 BPTT 求导与更新 (严格解耦) ---
                if (!r_rec.traj.steps.empty()) {
                    bool is_l = (r_rec.role == 1);
                    CellularOrganism& active_org = is_l ? landlord_org : peasant_org;
                    CellularBPTTEngine& active_bptt = is_l ? bptt_landlord : bptt_peasant;
                    BPTTGradients& active_grads = is_l ? grads_l : grads_p;
                    int& active_count = is_l ? count_l : count_p;

                    active_org.reset_state(true);
                    active_bptt.reset_tape();

                    std::vector<std::vector<float>> target_outputs;
                    for (const auto& st : r_rec.traj.steps) {
                        std::vector<double> d_in(st.obs.begin(), st.obs.end());
                        active_org.forward_nd(d_in.data(), d_in.size(), false);
                        active_bptt.record_step(active_org);

                        int eff_ch = action_to_channel(st.action);
                        target_outputs.push_back({
                            static_cast<float>(eff_ch),
                            adv,
                            st.action_prob,
                            grpo_cfg.clip_eps,
                            grpo_cfg.entropy_coef
                        });
                    }

                    BPTTGradients step_grads;
                    float p_loss = active_bptt.backward_with_loss(
                        active_org, target_outputs, step_grads, SubstrateLossType::GRPO_SURROGATE
                    );
                    epoch_loss += p_loss;
                    active_count++;

                    for (size_t k = 0; k < step_grads.grad_synapses.size(); ++k) {
                        active_grads.grad_synapses[k] += step_grads.grad_synapses[k];
                    }
                    for (size_t k = 0; k < step_grads.grad_gains.size(); ++k) {
                        active_grads.grad_gains[k] += step_grads.grad_gains[k];
                    }
                }
            }
        }

        // 应用 Adam 梯度更新
        auto apply_adam = [&](CellularOrganism& o, CellularBPTTEngine& eng, BPTTGradients& g, int cnt) {
            if (cnt > 0) {
                float inv = 1.0f / static_cast<float>(cnt);
                for (auto& v : g.grad_synapses) v *= inv;
                for (auto& v : g.grad_gains) v *= inv;
                eng.step_adam(o, g, current_lr);
                for (auto& syn : o.synapses) {
                    syn.weight = std::clamp(syn.weight, -10.0, 10.0);
                }
                o.compile();
            }
        };

        if (!FREEZE_BID) apply_adam(bid_org, bptt_bid, grads_bid, count_bid);
        else { /* 冻结: 保留健康叫牌先验, 探索概率不塌缩 */ }
        apply_adam(landlord_org, bptt_landlord, grads_l, count_l);
        apply_adam(peasant_org, bptt_peasant, grads_p, count_p);

        // 打印训练进度
        if (epoch % 25 == 0 || epoch == 1 || epoch == TOTAL_EPOCHS) {
            int total_games = GROUPS_PER_EPOCH * GROUP_SIZE;
            double wr = static_cast<double>(epoch_wins) / total_games * 100.0;
            double avg_c = epoch_chips / total_games;
            double l_wr = epoch_landlord_games > 0 ? (double)epoch_landlord_wins / epoch_landlord_games * 100.0 : 0.0;
            double p_wr = epoch_peasant_games > 0 ? (double)epoch_peasant_wins / epoch_peasant_games * 100.0 : 0.0;

            std::stringstream ss_l, ss_p;
            ss_l << std::fixed << std::setprecision(1) << l_wr << "% (" << epoch_landlord_wins << "/" << epoch_landlord_games << ")";
            ss_p << std::fixed << std::setprecision(1) << p_wr << "% (" << epoch_peasant_wins << "/" << epoch_peasant_games << ")";

            std::cout << std::left << std::setw(8)  << epoch
                      << std::setw(14) << std::fixed << std::setprecision(4) << epoch_loss / (GROUPS_PER_EPOCH * GROUP_SIZE)
                      << std::setw(12) << std::fixed << std::setprecision(3) << epoch_entropy / std::max(1, count_l + count_p)
                      << std::setw(16) << std::fixed << std::setprecision(4) << epoch_adv / (GROUPS_PER_EPOCH * GROUP_SIZE)
                      << std::setw(14) << (std::to_string((int)wr) + "%")
                      << std::setw(16) << (std::string(avg_c >= 0 ? "+" : "") + std::to_string((int)avg_c) + " 豆/局")
                      << std::setw(16) << ss_l.str()
                      << ss_p.str() << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double train_sec = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n✅ 大规模联合优化收敛完成！总耗时: " << std::fixed << std::setprecision(2) << train_sec << " 秒\n\n";

    // 4. 保存微柱检查点
    bid_org.save_checkpoint_bin("checkpoints/doudizhu_moe_bidding.bin");
    landlord_org.save_checkpoint_bin("checkpoints/doudizhu_moe_landlord.bin");
    peasant_org.save_checkpoint_bin("checkpoints/doudizhu_moe_peasant.bin");
    std::cout << "💾 成功导出三微柱独立检查点至 checkpoints/\n\n";

    // 5. 启动严格 5,000 局 3-Baseline 标准实证对账盲测 (门禁 = CALIBRATED_THRESHOLD 专家基准线)
    std::cout << "[Step 3] 启动 5,000 局 3-Baseline 标准实证对账盲测 (对齐专家门禁 threshold=" << CALIBRATED_THRESHOLD << ")...\n";
    DouDiZhuThreeBaselineHarness harness(5000, 20260907, CALIBRATED_THRESHOLD, 40);

    auto final_bid_eval = [&](const std::vector<float>& hand_obs, double opp_max_score) -> bool {
        auto b = bid_org;
        b.reset_state(true);
        std::vector<double> inps(hand_obs.begin(), hand_obs.end());
        auto acts = b.forward_nd(inps.data(), inps.size(), false);
        double net = acts.positive_action - acts.negative_action;
        double bid_score = 12.0 + 10.0 * net;
        return (bid_score >= MODEL_BID_GATE && bid_score >= opp_max_score);
    };

    double avg_lat = 0.0;
    uint64_t decisions = 0;
    auto report = harness.evaluate_neural_moe_organism(
        "SDSCC Neural Bidding + Dual-Role MoE",
        final_bid_eval,
        landlord_org,
        peasant_org,
        &avg_lat,
        &decisions
    );

    report.print_table(std::cout);

    std::cout << "⚡ 性能与效率评估:\n";
    std::cout << "   - 总决策次数: " << decisions << " 次\n";
    std::cout << "   - 单步推理时延: " << std::fixed << std::setprecision(2) << avg_lat << " ns\n\n";

    // 验证细分角色战绩
    double eval_chips = 0.0;
    int eval_lw = 0, eval_lg = 0, eval_pw = 0, eval_pg = 0;
    double eval_wr = quick_eval_moe(bid_org, landlord_org, peasant_org, 5000, 20260907, CALIBRATED_THRESHOLD,
                                   eval_chips, eval_lw, eval_lg, eval_pw, eval_pg);

    std::cout << "📊 [分角色实证对账明细]:\n";
    std::cout << "   - 地主进攻场次: " << eval_lg << " 局 | 胜场: " << eval_lw 
              << " | 胜率: " << std::fixed << std::setprecision(2) << (eval_lg > 0 ? (double)eval_lw / eval_lg * 100.0 : 0.0) << "%\n";
    std::cout << "   - 农民协作场次: " << eval_pg << " 局 | 胜场: " << eval_pw 
              << " | 胜率: " << std::fixed << std::setprecision(2) << (eval_pg > 0 ? (double)eval_pw / eval_pg * 100.0 : 0.0) << "%\n";
    std::cout << "   - 全局总体胜率: " << std::fixed << std::setprecision(2) << eval_wr << "% | 场均筹码净利: " 
              << (eval_chips >= 0 ? "+" : "") << eval_chips << " 豆/局\n\n";

    return 0;
}
