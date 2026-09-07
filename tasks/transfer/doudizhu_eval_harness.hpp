#pragma once

// ============================================================================
// 软件定义硅基细胞计算机 (SDSCC) - 斗地主三基线实证对账评测架构
// (DouDiZhu 3-Baseline Empirical Evaluation Harness)
//
// 治理原则 (实证评测治理方案):
// 1. 彻底根除叙事过载 (Narrative Overclaiming) 与统计噪声 (Statistical Noise):
//    - 评测规模强制提升至 5,000+ 配对随机发牌种子 (Paired-Seed Episodes)；
//    - 彻底分离起手牌门禁势能收益与出牌博弈策略真实增益。
// 2. 强制三基线对账账本 (Mandatory 3-Baseline Comparison):
//    - Baseline 0: 均匀随机合法出牌 (Uniform Random Legal Move, 无叫牌门禁 threshold=-100)；
//    - Baseline 1: 均匀随机合法出牌 + 叫牌门禁 (Bidding Gate: Hand Strength >= 5.0)；
//    - Target: 待评测生命体/微柱模型 (在完全相同的 Bidding Gate >= 5.0 与相同发牌种子下评测)。
// 3. 统计推断规范:
//    - 胜率采用 Wilson 95% 置信区间 [lower, upper]；
//    - 统计显著性采用严格麦克尼马尔配对检验 (McNemar's Test vs Baseline 1)，判定 p < 0.01；
//    - 筹码净利采用配对样本差值置信区间推断。
// ============================================================================

#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/statistical_evaluation.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_relaxation.hpp"
#include "kun/cellular/cellular_moe.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <chrono>
#include <random>
#include <algorithm>

namespace kun {

/**
 * @brief 在斗地主当前局面下均匀抽取合规合法动作
 * - 自由出牌权 (TRICK_NONE): 规则禁止让牌 (0)，合规合法动作为 {1: 跟牌, 2: 抢牌}
 * - 对手出牌 (TRICK != NONE): 合法尝试动作为 {0: 让牌, 1: 跟牌, 2: 强行抢牌}
 */
inline int sample_uniform_random_legal_action(const DouDiZhuCardGameTask& task, std::mt19937& rng) {
    if (task.table_trick().type == DouDiZhuCardGameTask::TRICK_NONE) {
        std::uniform_int_distribution<int> dist(1, 2);
        return dist(rng);
    } else {
        std::uniform_int_distribution<int> dist(0, 2);
        return dist(rng);
    }
}

/**
 * @brief 单策略在大样本天梯下的实证表现
 */
struct BaselinePerformance {
    std::string name;
    int total_episodes{0};
    int wins{0};
    double win_rate{0.0};
    ConfidenceInterval win_rate_ci; // Wilson 95% CI
    double avg_chips{0.0};
    double total_chips{0.0};
    double avg_steps{0.0};
    std::vector<int> win_records;    // 1=胜, 0=负 (按 episode 配对索引对齐)
    std::vector<double> chip_records; // 筹码记录 (+200 / -200)
};

/**
 * @brief 三基线综合实证检验报告
 */
struct ThreeBaselineReport {
    int total_episodes{0};
    uint32_t seed_base{0};
    double bidding_gate_threshold{16.0};
    BaselinePerformance baseline_0; // 随机合法出牌 (无门禁)
    BaselinePerformance baseline_1; // 随机合法出牌 + 门禁 (Strength >= 5)
    BaselinePerformance target;     // 待评生命体
    McNemarResult mcnemar_target_vs_b1;
    PairedDifferenceResult chips_target_vs_b1;
    McNemarResult mcnemar_b1_vs_b0;
    PairedDifferenceResult chips_b1_vs_b0;

    void print_table(std::ostream& os = std::cout) const {
        os << "\n====================================================================================================================\n";
        os << " 📊 斗地主 3-Baseline 标准实证对账账本 (" << total_episodes << " 局配对随机发牌种子盲测)\n";
        os << "====================================================================================================================\n";
        os << "| " << std::left << std::setw(34) << "策略 / 生命体名称"
           << "| " << std::setw(12) << "胜场/总数"
           << "| " << std::setw(24) << "胜率 (Wilson 95% CI)"
           << "| " << std::setw(16) << "场均筹码净利"
           << "| " << std::setw(14) << "McNemar vs B1"
           << "| " << std::setw(16) << "显著性 (p<0.01)"
           << "|\n";
        os << "|-----------------------------------|-------------|-------------------------|-----------------|---------------|-----------------|\n";

        auto print_row = [&](const BaselinePerformance& b, const std::string& mcnemar_str, const std::string& sig_str) {
            std::stringstream count_ss, chip_ss;
            count_ss << b.wins << "/" << b.total_episodes;
            chip_ss << (b.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b.avg_chips << " 豆/局";

            os << "| " << std::left << std::setw(34) << b.name
               << "| " << std::setw(12) << count_ss.str()
               << "| " << std::setw(24) << b.win_rate_ci.to_string(1, true)
               << "| " << std::setw(16) << chip_ss.str()
               << "| " << std::setw(14) << mcnemar_str
               << "| " << std::setw(16) << sig_str
               << "|\n";
        };

        print_row(baseline_0, "对照 (Control 0)", "-");
        print_row(baseline_1, "基准 (Control 1)", "-");

        std::stringstream p_str;
        if (mcnemar_target_vs_b1.p_value < 1e-4) {
            p_str << "p=" << std::scientific << std::setprecision(1) << mcnemar_target_vs_b1.p_value;
        } else {
            p_str << "p=" << std::fixed << std::setprecision(4) << mcnemar_target_vs_b1.p_value;
        }
        std::string sig_label = "-";
        if (mcnemar_target_vs_b1.significant_01) {
            sig_label = (mcnemar_target_vs_b1.b > mcnemar_target_vs_b1.c) ? "✅ 显著优势 (p<0.01)" : "🔻 显著劣势 (p<0.01)";
        } else if (mcnemar_target_vs_b1.significant_05) {
            sig_label = (mcnemar_target_vs_b1.b > mcnemar_target_vs_b1.c) ? "弱优势 (p<0.05)" : "弱劣势 (p<0.05)";
        } else {
            sig_label = "❌ 不显著 (p>=0.05)";
        }
        print_row(target, p_str.str(), sig_label);

        os << "====================================================================================================================\n\n";

        // 打印统计解耦分析
        double gate_bonus = baseline_1.win_rate - baseline_0.win_rate;
        double policy_gain = target.win_rate - baseline_1.win_rate;

        os << "🔬 [实证统计治理剖析]:\n";
        os << "   1. 叫牌门禁贡献: 从 Baseline 0 (" << std::fixed << std::setprecision(1) << baseline_0.win_rate 
           << "%) 到 Baseline 1 (" << baseline_1.win_rate << "%), 纯门禁贡献 +" << gate_bonus << "%\n";
        os << "   2. 策略实战净增益: 待评模型相对 Baseline 1 净增益 " << (policy_gain >= 0 ? "+" : "") 
           << policy_gain << "% (95% CI: [" << target.win_rate_ci.lower * 100.0 << "%, " << target.win_rate_ci.upper * 100.0 << "%])\n";
        os << "   3. 配对卡方检定: " << mcnemar_target_vs_b1.to_string(2) << "\n";
        os << "   4. 最终实证结论: " << (mcnemar_target_vs_b1.significant_01 
               ? (mcnemar_target_vs_b1.b > mcnemar_target_vs_b1.c
                   ? "生命体策略在控制起手牌势能后，仍取得严格统计显著优势 (p < 0.01)！"
                   : "生命体策略在控制起手牌势能后，显著劣于基准 (p < 0.01)！")
               : "差异未达 p < 0.01 显著性门限，无法拒绝虚无假设（无统计显著优势）！") << "\n\n";
    }
};

/**
 * @brief 斗地主三基线大样本配对实证评测线
 */
class DouDiZhuThreeBaselineHarness {
public:
    explicit DouDiZhuThreeBaselineHarness(
        int num_episodes = 5000,
        uint32_t seed_base = 20260907,
        double bidding_gate_threshold = 16.0,
        int max_rounds = 40)
        : num_episodes_(num_episodes),
          seed_base_(seed_base),
          bidding_gate_threshold_(bidding_gate_threshold),
          max_rounds_(max_rounds) {
        seeds_.reserve(num_episodes_);
        for (int i = 0; i < num_episodes_; ++i) {
            seeds_.push_back(seed_base_ + static_cast<uint32_t>(i) * 17 + 1);
        }
        precompute_baselines();
    }

    int total_episodes() const { return num_episodes_; }
    uint32_t seed_base() const { return seed_base_; }
    double bidding_gate_threshold() const { return bidding_gate_threshold_; }
    const BaselinePerformance& baseline_0() const { return b0_; }
    const BaselinePerformance& baseline_1() const { return b1_; }
    const std::vector<uint32_t>& seeds() const { return seeds_; }

    /**
     * @brief 评测任意自定义策略函数
     */
    ThreeBaselineReport evaluate_policy(
        const std::string& target_name,
        std::function<int(const std::vector<float>& obs, DouDiZhuCardGameTask& task)> policy_fn,
        uint64_t* out_decision_calls = nullptr) const
    {
        BaselinePerformance target_perf;
        target_perf.name = target_name;
        target_perf.total_episodes = num_episodes_;
        target_perf.win_records.resize(num_episodes_);
        target_perf.chip_records.resize(num_episodes_);

        int wins = 0;
        double total_chips = 0.0;
        double total_steps = 0.0;
        uint64_t decision_calls = 0;

        for (int i = 0; i < num_episodes_; ++i) {
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, bidding_gate_threshold_);

            while (true) {
                auto obs = task.current_observation();
                decision_calls++;
                int act = policy_fn(obs, task);
                auto res = task.step(act);
                if (res.done) {
                    if (res.success) wins++;
                    double chips = res.success ? 200.0 : -200.0;
                    total_chips += chips;
                    total_steps += res.steps;
                    target_perf.win_records[i] = res.success ? 1 : 0;
                    target_perf.chip_records[i] = chips;
                    break;
                }
            }
        }

        if (out_decision_calls) {
            *out_decision_calls = decision_calls;
        }

        finalize_performance(target_perf, wins, total_chips, total_steps);
        return compile_report(target_perf);
    }

    /**
     * @brief 评测标准前向推理生命体 (CellularOrganism Forward)
     */
    ThreeBaselineReport evaluate_organism(
        const std::string& target_name,
        CellularOrganism base_org,
        double* out_avg_latency_ns = nullptr,
        uint64_t* out_decision_calls = nullptr) const
    {
        BaselinePerformance target_perf;
        target_perf.name = target_name;
        target_perf.total_episodes = num_episodes_;
        target_perf.win_records.resize(num_episodes_);
        target_perf.chip_records.resize(num_episodes_);

        int wins = 0;
        double total_chips = 0.0;
        double total_steps = 0.0;
        double total_latency_ns = 0.0;
        uint64_t decision_calls = 0;

        for (int i = 0; i < num_episodes_; ++i) {
            auto org = base_org;
            org.reset_state(true);
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, bidding_gate_threshold_);

            while (true) {
                auto obs = task.current_observation();
                std::vector<double> inps(obs.begin(), obs.end());

                auto t0 = std::chrono::high_resolution_clock::now();
                auto acts = org.forward_nd(inps.data(), inps.size(), false);
                auto t1 = std::chrono::high_resolution_clock::now();
                total_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
                decision_calls++;

                auto res = task.step_continuous(acts);
                if (res.done) {
                    if (res.success) wins++;
                    double chips = res.success ? 200.0 : -200.0;
                    total_chips += chips;
                    total_steps += res.steps;
                    target_perf.win_records[i] = res.success ? 1 : 0;
                    target_perf.chip_records[i] = chips;
                    break;
                }
            }
        }

        if (out_avg_latency_ns) {
            *out_avg_latency_ns = decision_calls > 0 ? (total_latency_ns / decision_calls) : 0.0;
        }
        if (out_decision_calls) {
            *out_decision_calls = decision_calls;
        }

        finalize_performance(target_perf, wins, total_chips, total_steps);
        return compile_report(target_perf);
    }

    /**
     * @brief 评测具备测试时动态松弛 (K-step Relaxation) 的生命体
     */
    struct RelaxationMetrics {
        double avg_latency_ns{0.0};
        double avg_settling_energy{0.0};
        int endgame_decisions{0};
        int endgame_errors{0};
        double endgame_error_rate{0.0};
    };

    ThreeBaselineReport evaluate_organism_relaxation(
        const std::string& target_name,
        CellularOrganism base_org,
        const RelaxationConfig& cfg,
        RelaxationMetrics* out_metrics = nullptr) const
    {
        BaselinePerformance target_perf;
        target_perf.name = target_name;
        target_perf.total_episodes = num_episodes_;
        target_perf.win_records.resize(num_episodes_);
        target_perf.chip_records.resize(num_episodes_);

        int wins = 0;
        double total_chips = 0.0;
        double total_steps = 0.0;
        double total_latency_ns = 0.0;
        double total_energy = 0.0;
        uint64_t decision_calls = 0;
        int endgame_decisions = 0;
        int endgame_errors = 0;

        for (int i = 0; i < num_episodes_; ++i) {
            auto org = base_org;
            org.reset_state(true);
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, bidding_gate_threshold_);

            while (true) {
                auto obs = task.current_observation();
                std::vector<double> inps(obs.begin(), obs.end());

                RelaxationTelemetry telem;
                auto t0 = std::chrono::high_resolution_clock::now();
                auto acts = org.forward_with_relaxation(inps.data(), inps.size(), cfg, &telem);
                auto t1 = std::chrono::high_resolution_clock::now();

                total_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
                total_energy += telem.final_energy;
                decision_calls++;

                float act0 = static_cast<float>(acts.negative_action); // PASS
                float act1 = static_cast<float>(acts.positive_action); // FOLLOW
                float act2 = static_cast<float>(acts.defensive_reset);  // SEIZE

                int act = 1;
                if (task.table_trick().type == DouDiZhuCardGameTask::TRICK_NONE) {
                    act = (act2 > act1) ? 2 : 1;
                } else {
                    if (act2 > act1 && act2 > act0) act = 2;
                    else if (act0 > act1) act = 0;
                    else act = 1;
                }

                // 残局决策评测 (手牌 <= 5 张)
                int cards_p0 = task.cards_left(0);
                if (cards_p0 <= 5) {
                    endgame_decisions++;
                    bool is_landlord = (task.role() == 1);
                    int teammate = is_landlord ? -1 : (task.landlord() == 1 ? 2 : 1);
                    int opp_min = is_landlord ? std::min(task.cards_left(1), task.cards_left(2)) : task.cards_left(task.landlord());

                    if (task.table_trick().type == DouDiZhuCardGameTask::TRICK_NONE && act == 0) {
                        endgame_errors++;
                    } else if (!is_landlord && task.table_trick().owner == teammate && task.table_trick().rank >= 8 && act != 0) {
                        endgame_errors++;
                    } else if (opp_min <= 2 && task.table_trick().owner != 0 && task.table_trick().owner != teammate && act == 0) {
                        endgame_errors++;
                    }
                }

                auto res = task.step(act);
                if (res.done) {
                    if (res.success) wins++;
                    double chips = res.success ? 200.0 : -200.0;
                    total_chips += chips;
                    total_steps += res.steps;
                    target_perf.win_records[i] = res.success ? 1 : 0;
                    target_perf.chip_records[i] = chips;
                    break;
                }
            }
        }

        if (out_metrics) {
            out_metrics->avg_latency_ns = decision_calls > 0 ? (total_latency_ns / decision_calls) : 0.0;
            out_metrics->avg_settling_energy = decision_calls > 0 ? (total_energy / decision_calls) : 0.0;
            out_metrics->endgame_decisions = endgame_decisions;
            out_metrics->endgame_errors = endgame_errors;
            out_metrics->endgame_error_rate = endgame_decisions > 0 ? (static_cast<double>(endgame_errors) / endgame_decisions * 100.0) : 0.0;
        }

        finalize_performance(target_perf, wins, total_chips, total_steps);
        return compile_report(target_perf);
    }

    /**
     * @brief 评测具备端到端神经叫牌 + MoE 双角色专精微柱的生命体
     */
    ThreeBaselineReport evaluate_neural_moe_organism(
        const std::string& target_name,
        DouDiZhuCardGameTask::NeuralBidEvaluator bid_eval,
        CellularOrganism landlord_org,
        CellularOrganism peasant_org,
        double* out_avg_latency_ns = nullptr,
        uint64_t* out_decision_calls = nullptr) const
    {
        BaselinePerformance target_perf;
        target_perf.name = target_name;
        target_perf.total_episodes = num_episodes_;
        target_perf.win_records.resize(num_episodes_);
        target_perf.chip_records.resize(num_episodes_);

        int wins = 0;
        double total_chips = 0.0;
        double total_steps = 0.0;
        double total_latency_ns = 0.0;
        uint64_t decision_calls = 0;

        for (int i = 0; i < num_episodes_; ++i) {
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, bidding_gate_threshold_, bid_eval);

            auto l_org = landlord_org;
            auto p_org = peasant_org;
            l_org.reset_state(true);
            p_org.reset_state(true);

            while (true) {
                auto obs = task.current_observation();
                std::vector<double> inps(obs.begin(), obs.end());

                auto t0 = std::chrono::high_resolution_clock::now();
                CellularOrganism::ActionOutputs acts;
                if (task.role() == 1) {
                    acts = l_org.forward_nd(inps.data(), inps.size(), false);
                } else {
                    acts = p_org.forward_nd(inps.data(), inps.size(), false);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                total_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
                decision_calls++;

                auto res = task.step_continuous(acts);
                if (res.done) {
                    if (res.success) wins++;
                    double chips = res.success ? 200.0 : -200.0;
                    total_chips += chips;
                    total_steps += res.steps;
                    target_perf.win_records[i] = res.success ? 1 : 0;
                    target_perf.chip_records[i] = chips;
                    break;
                }
            }
        }

        if (out_avg_latency_ns) {
            *out_avg_latency_ns = decision_calls > 0 ? (total_latency_ns / decision_calls) : 0.0;
        }
        if (out_decision_calls) {
            *out_decision_calls = decision_calls;
        }

        finalize_performance(target_perf, wins, total_chips, total_steps);
        return compile_report(target_perf);
    }

private:
    int num_episodes_{5000};
    uint32_t seed_base_{20260907};
    double bidding_gate_threshold_{16.0};
    int max_rounds_{40};
    std::vector<uint32_t> seeds_;
    BaselinePerformance b0_;
    BaselinePerformance b1_;

    void precompute_baselines() {
        // 1. Baseline 0: 均匀随机合法动作 (无叫牌门禁, threshold=-100)
        b0_.name = "Baseline 0 (随机合法·无叫牌门禁)";
        b0_.total_episodes = num_episodes_;
        b0_.win_records.resize(num_episodes_);
        b0_.chip_records.resize(num_episodes_);

        int wins0 = 0;
        double chips0 = 0.0;
        double steps0 = 0.0;

        for (int i = 0; i < num_episodes_; ++i) {
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, -100.0);
            std::mt19937 rng(s ^ 0xB0B0B0B0);

            while (true) {
                int act = sample_uniform_random_legal_action(task, rng);
                auto res = task.step(act);
                if (res.done) {
                    if (res.success) wins0++;
                    double c = res.success ? 200.0 : -200.0;
                    chips0 += c;
                    steps0 += res.steps;
                    b0_.win_records[i] = res.success ? 1 : 0;
                    b0_.chip_records[i] = c;
                    break;
                }
            }
        }
        finalize_performance(b0_, wins0, chips0, steps0);

        // 2. Baseline 1: 均匀随机合法动作 + 叫牌门禁 (Strength >= bidding_gate_threshold_)
        std::stringstream b1_name;
        b1_name << "Baseline 1 (随机合法·门禁>=" << std::fixed << std::setprecision(1) << bidding_gate_threshold_ << ")";
        b1_.name = b1_name.str();
        b1_.total_episodes = num_episodes_;
        b1_.win_records.resize(num_episodes_);
        b1_.chip_records.resize(num_episodes_);

        int wins1 = 0;
        double chips1 = 0.0;
        double steps1 = 0.0;

        for (int i = 0; i < num_episodes_; ++i) {
            uint32_t s = seeds_[i];
            DouDiZhuCardGameTask task(max_rounds_, s, bidding_gate_threshold_);
            std::mt19937 rng(s ^ 0xB1B1B1B1);

            while (true) {
                int act = sample_uniform_random_legal_action(task, rng);
                auto res = task.step(act);
                if (res.done) {
                    if (res.success) wins1++;
                    double c = res.success ? 200.0 : -200.0;
                    chips1 += c;
                    steps1 += res.steps;
                    b1_.win_records[i] = res.success ? 1 : 0;
                    b1_.chip_records[i] = c;
                    break;
                }
            }
        }
        finalize_performance(b1_, wins1, chips1, steps1);
    }

    static void finalize_performance(BaselinePerformance& perf, int wins, double chips, double steps) {
        perf.wins = wins;
        perf.win_rate = (static_cast<double>(wins) / perf.total_episodes) * 100.0;
        perf.win_rate_ci = wilson_score_interval(wins, perf.total_episodes, 0.95);
        perf.total_chips = chips;
        perf.avg_chips = chips / perf.total_episodes;
        perf.avg_steps = steps / perf.total_episodes;
    }

    ThreeBaselineReport compile_report(const BaselinePerformance& target) const {
        ThreeBaselineReport r;
        r.total_episodes = num_episodes_;
        r.seed_base = seed_base_;
        r.bidding_gate_threshold = bidding_gate_threshold_;
        r.baseline_0 = b0_;
        r.baseline_1 = b1_;
        r.target = target;
        r.mcnemar_target_vs_b1 = mcnemar_test(target.win_records, b1_.win_records, true);
        r.chips_target_vs_b1 = analyze_paired_differences(target.chip_records, b1_.chip_records, 0.95);
        r.mcnemar_b1_vs_b0 = mcnemar_test(b1_.win_records, b0_.win_records, true);
        r.chips_b1_vs_b0 = analyze_paired_differences(b1_.chip_records, b0_.chip_records, 0.95);
        return r;
    }
};

} // namespace kun
