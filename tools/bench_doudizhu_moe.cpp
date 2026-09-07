#include "tasks/transfer/cross_domain_tasks.hpp"
#include "tasks/transfer/doudizhu_eval_harness.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_moe.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <sstream>

using namespace kun;

// 辅助函数: 创建具有特定功能倾向的微柱神经生命体 (Cortical Micro-column)
static CellularOrganism build_expert_column(int specialization_type) {
    CellularOrganism org;

    // 0..3: 感受器 (4 维物理观测)
    // 0: 手牌质量均值 (hand strength)
    // 1: 剩余张数比率 (cards left ratio: cards / 20.0)
    // 2: 台面牌力威胁 (table trick intensity: rank / 14.0)
    // 3: 历史高牌打出度 (high cards played intensity: count / 6.0)
    org.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -30.0f, 0.0f});
    org.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f, -10.0f, 0.0f});
    org.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  10.0f, 0.0f});
    org.cells.push_back({3, CellType::SENSE_RAW_INPUT_3, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -60.0f,  30.0f, 0.0f});

    // 4..7: 内部运算中枢
    org.cells.push_back({4, CellType::OP_SUB, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -20.0f, -20.0f, 0.0f});
    org.cells.push_back({5, CellType::OP_EMA, 0.40, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 0.0f, 0.0f});
    org.cells.push_back({6, CellType::OP_SUM, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 20.0f, 0.0f, 0.0f});
    org.cells.push_back({7, CellType::GATE_THRESHOLD, -0.26, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, 20.0f, 0.0f});

    // 8..10: 动作效应器
    org.cells.push_back({8, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, -30.0f, 0.0f}); // Follow (1)
    org.cells.push_back({9, CellType::ACT_PRIMARY_NEGATIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 0.0f, 0.0f});  // Pass (0)
    org.cells.push_back({10, CellType::ACT_DEFENSIVE_RESET, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f, 30.0f, 0.0f});  // Seize (2)

    // 基础反射链路
    org.synapses.push_back({0, 4, 0, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({2, 4, 1, 1.0, true, 50.0f, -1.0f});
    org.synapses.push_back({1, 7, 0, -1.0, true, 50.0f, -1.0f});

    switch (specialization_type) {
        case 0: // 专家 0: 开局进攻与起手压制
            org.synapses.push_back({0, 8, 0, 2.4, true, 50.0f, -1.0f});
            org.synapses.push_back({1, 8, 0, 1.5, true, 50.0f, -1.0f});
            org.synapses.push_back({4, 6, 0, 1.2, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 10, 0, 1.8, true, 50.0f, -1.0f});
            org.synapses.push_back({0, 9, 0, -2.0, true, 50.0f, -1.0f});
            break;

        case 1: // 专家 1: 中盘战术让牌与盟友协同
            org.synapses.push_back({2, 9, 0, 2.5, true, 50.0f, -1.0f});
            org.synapses.push_back({0, 8, 0, 0.8, true, 50.0f, -1.0f});
            org.synapses.push_back({4, 9, 0, -1.5, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 10, 0, 0.4, true, 50.0f, -1.0f});
            break;

        case 2: // 专家 2: 动态记忆与全场节奏调控
            org.synapses.push_back({6, 5, 0, 0.7, true, 50.0f, -1.0f});
            org.synapses.push_back({5, 6, 1, 0.7, true, 50.0f, -1.0f});
            org.synapses.push_back({3, 6, 0, 1.4, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 8, 0, 1.3, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 9, 0, -1.0, true, 50.0f, -1.0f});
            break;

        case 3: // 专家 3: 残局斩杀与冲刺收割
            org.synapses.push_back({7, 6, 0, 2.5, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 10, 0, 2.8, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 8, 0, 1.6, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 9, 0, -3.0, true, 50.0f, -1.0f});
            break;

        default: // 通用平衡微柱
            org.synapses.push_back({0, 8, 0, 1.8, true, 50.0f, -1.0f});
            org.synapses.push_back({1, 8, 0, 0.8, true, 50.0f, -1.0f});
            org.synapses.push_back({2, 9, 0, 1.7, true, 50.0f, -1.0f});
            org.synapses.push_back({4, 6, 0, 0.6, true, 50.0f, -1.0f});
            org.synapses.push_back({7, 6, 0, 0.8, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 10, 0, 1.4, true, 50.0f, -1.0f});
            org.synapses.push_back({6, 9, 0, -1.1, true, 50.0f, -1.0f});
            break;
    }

    for (auto& s : org.synapses) s.initial_weight = s.weight;
    org.compile();
    return org;
}

// 专家在不同游戏阶段的先验指派统计
struct PhaseDispatchStats {
    uint64_t bidding_calls[8]{0}; // 叫牌/起手阶段 (cards >= 17)
    uint64_t early_calls[8]{0};   // 开局阶段 (13 <= cards <= 16)
    uint64_t midgame_calls[8]{0}; // 中盘阶段 (6 <= cards <= 12)
    uint64_t endgame_calls[8]{0}; // 残局阶段 (cards <= 5)
    uint64_t total_decisions{0};
};

struct MoEBenchEntry {
    std::string name;
    int num_experts{1};
    int top_k{1};
    bool is_sparse{true};
    ThreeBaselineReport report;
    double avg_latency_ns{0.0};
    uint64_t total_expert_evaluations{0};
    double active_compute_ratio{1.0};
    PhaseDispatchStats phase_stats;
    McNemarResult mcnemar_vs_single;
};

// 评测单一微柱基线
static MoEBenchEntry eval_single_column_entry(const DouDiZhuThreeBaselineHarness& harness) {
    MoEBenchEntry entry;
    entry.name = "单微柱基线 (Single Column)";
    entry.num_experts = 1;
    entry.top_k = 1;
    entry.is_sparse = false;
    entry.active_compute_ratio = 1.0;

    CellularOrganism org = build_expert_column(4);
    double lat = 0.0;
    uint64_t decisions = 0;
    entry.report = harness.evaluate_organism(entry.name, org, &lat, &decisions);
    entry.avg_latency_ns = lat;
    entry.total_expert_evaluations = decisions; // 单微柱每次决策仅推演 1 次专家
    return entry;
}

// 评测 Dense 密集全量多微柱融合
static MoEBenchEntry eval_dense_entry(const DouDiZhuThreeBaselineHarness& harness, size_t N) {
    MoEBenchEntry entry;
    entry.name = "Dense 全量多微柱 (N=" + std::to_string(N) + " 全激活)";
    entry.num_experts = static_cast<int>(N);
    entry.top_k = static_cast<int>(N);
    entry.is_sparse = false;
    entry.active_compute_ratio = 1.0;

    std::vector<CellularOrganism> experts;
    for (size_t i = 0; i < N; ++i) {
        experts.push_back(build_expert_column(static_cast<int>(i % 4)));
    }
    CellularOrganismMoE moe(experts, N, 4, 0.01, harness.seed_base());

    double total_latency_ns = 0.0;
    uint64_t decision_calls = 0;

    auto rep = harness.evaluate_policy(entry.name, [&](const std::vector<float>& obs, DouDiZhuCardGameTask&) {
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto t0 = std::chrono::high_resolution_clock::now();
        auto acts = moe.forward_dense(inps, 4, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
        decision_calls++;

        if (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) return 2;
        if (acts.negative_action > acts.positive_action) return 0;
        return 1;
    });

    entry.report = rep;
    entry.avg_latency_ns = decision_calls > 0 ? (total_latency_ns / decision_calls) : 0.0;
    entry.total_expert_evaluations = moe.active_call_count();
    return entry;
}

// 评测 Sparse 细粒度 MoE 稀疏路由 [Hand-crafted Prior Benchmark for Structural Latency & Bypass]
static MoEBenchEntry eval_sparse_entry(
    const DouDiZhuThreeBaselineHarness& harness,
    size_t N,
    size_t K)
{
    MoEBenchEntry entry;
    entry.name = "Sparse MoE [人工先验] (N=" + std::to_string(N) + ", Top-" + std::to_string(K) + ")";
    entry.num_experts = static_cast<int>(N);
    entry.top_k = static_cast<int>(K);
    entry.is_sparse = true;
    entry.active_compute_ratio = static_cast<double>(K) / static_cast<double>(N);

    std::vector<CellularOrganism> experts;
    for (size_t i = 0; i < N; ++i) {
        experts.push_back(build_expert_column(static_cast<int>(i % 4)));
    }
    CellularOrganismMoE moe(experts, K, 4, 0.02, 2026);

    // 显式硬编码人工规则先验 (Handcrafted Prior Rules)
    // 目的: 评测硬件级 Top-K 零算力旁路与时延收缩物理特性
    auto& rw = moe.router().weights();
    auto& rb = moe.router().biases();
    // 专家 0 (开局进攻): 偏重 cards_left_ratio 与 hand_strength
    rw[0 * 4 + 0] = 1.2; rw[0 * 4 + 1] = 2.5; rw[0 * 4 + 2] = -0.5; rw[0 * 4 + 3] = -0.2; rb[0] = 0.5;
    // 专家 1 (中盘防御): 偏重 table_trick
    rw[1 * 4 + 0] = -0.5; rw[1 * 4 + 1] = 0.2; rw[1 * 4 + 2] = 2.8; rw[1 * 4 + 3] = 0.4; rb[1] = 0.8;
    // 专家 2 (节奏调控): 偏重 high_cards_played
    rw[2 * 4 + 0] = 0.6; rw[2 * 4 + 1] = 0.1; rw[2 * 4 + 2] = 0.2; rw[2 * 4 + 3] = 2.2; rb[2] = 0.3;
    // 专家 3 (残局冲刺): 负相关 cards_left_ratio (cards <= 5)
    rw[3 * 4 + 0] = 1.0; rw[3 * 4 + 1] = -3.8; rw[3 * 4 + 2] = -0.4; rw[3 * 4 + 3] = 0.6; rb[3] = 1.2;

    if (N == 8) {
        rw[4 * 4 + 0] = 1.5; rw[4 * 4 + 1] = 2.0; rw[4 * 4 + 2] = 0.2; rw[4 * 4 + 3] = -0.4; rb[4] = 0.4;
        rw[5 * 4 + 0] = -0.8; rw[5 * 4 + 1] = 0.3; rw[5 * 4 + 2] = 3.0; rw[5 * 4 + 3] = 0.5; rb[5] = 0.7;
        rw[6 * 4 + 0] = 0.8; rw[6 * 4 + 1] = -0.2; rw[6 * 4 + 2] = 0.5; rw[6 * 4 + 3] = 2.0; rb[6] = 0.2;
        rw[7 * 4 + 0] = 1.2; rw[7 * 4 + 1] = -4.2; rw[7 * 4 + 2] = -0.2; rw[7 * 4 + 3] = 0.8; rb[7] = 1.4;
    }

    double total_latency_ns = 0.0;
    uint64_t decision_calls = 0;

    auto rep = harness.evaluate_policy(entry.name, [&](const std::vector<float>& obs, DouDiZhuCardGameTask& task) {
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto t0 = std::chrono::high_resolution_clock::now();
        auto acts = moe.forward(inps, 4, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        const auto& dec = moe.router().cached_decision();

        total_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
        decision_calls++;

        int cards = task.cards_left(0);
        entry.phase_stats.total_decisions++;
        for (size_t exp_idx : dec.topk_indices) {
            if (exp_idx < 8) {
                if (cards >= 17) entry.phase_stats.bidding_calls[exp_idx]++;
                else if (cards >= 13) entry.phase_stats.early_calls[exp_idx]++;
                else if (cards >= 6) entry.phase_stats.midgame_calls[exp_idx]++;
                else entry.phase_stats.endgame_calls[exp_idx]++;
            }
        }

        if (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) return 2;
        if (acts.negative_action > acts.positive_action) return 0;
        return 1;
    });

    entry.report = rep;
    entry.avg_latency_ns = decision_calls > 0 ? (total_latency_ns / decision_calls) : 0.0;
    entry.total_expert_evaluations = moe.active_call_count();
    return entry;
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "=========================================================================================\n";
    std::cout << " 🧠 硅基细胞计算机 (SDSCC) 微柱稀疏混合专家 (Cellular MoE) 5,000 局对账基准\n";
    std::cout << " 标定说明: [Hand-crafted Prior Benchmark for Structural Latency & Bypass]\n";
    std::cout << " 科学声明: 本基准路由门控为显式人工先验规则 (Handcrafted Prior)，非自发涌现分工 (Non-Emergent)。\n";
    std::cout << " 评测核心: 验证 Top-K 硬件零算力旁路加速比，并在 5,000 局 3-Baseline 账本下客观对账。\n";
    std::cout << "=========================================================================================\n\n";

    const int NUM_EPISODES = 5000;
    const uint32_t SEED_BASE = 20260907;
    const double BIDDING_GATE = 5.0;

    std::cout << "⏳ 正在初始化 3-Baseline 评测管线 (预先计算 5,000 局配对发牌 Baseline 0 与 Baseline 1)...\n";
    DouDiZhuThreeBaselineHarness harness(NUM_EPISODES, SEED_BASE, BIDDING_GATE, 40);
    std::cout << "✅ 3-Baseline 预计算完成!\n";
    std::cout << "   - Baseline 0 (随机无门禁) 胜率: " << std::fixed << std::setprecision(1) << harness.baseline_0().win_rate << "%\n";
    std::cout << "   - Baseline 1 (随机门禁>=5) 胜率: " << std::fixed << std::setprecision(1) << harness.baseline_1().win_rate << "%\n\n";

    std::vector<MoEBenchEntry> results;

    std::cout << "⏳ [1/4] 正在评测单微柱基线 (Single Column, 5,000 局)..." << std::flush;
    results.push_back(eval_single_column_entry(harness));
    std::cout << " 完成!\n";

    std::cout << "⏳ [2/4] 正在评测 Dense 全量多微柱 (N=4 全激活, 5,000 局)..." << std::flush;
    results.push_back(eval_dense_entry(harness, 4));
    std::cout << " 完成!\n";

    std::cout << "⏳ [3/4] 正在评测 Sparse MoE (N=4, Top-2 稀疏门控, 5,000 局)..." << std::flush;
    results.push_back(eval_sparse_entry(harness, 4, 2));
    std::cout << " 完成!\n";

    std::cout << "⏳ [4/4] 正在评测 Sparse MoE (N=8, Top-2 极稀疏门控, 5,000 局)..." << std::flush;
    results.push_back(eval_sparse_entry(harness, 8, 2));
    std::cout << " 完成!\n";

    // 计算相对单微柱基线的配对 McNemar 检定
    const auto& single_records = results[0].report.target.win_records;
    for (size_t i = 1; i < results.size(); ++i) {
        results[i].mcnemar_vs_single = mcnemar_test(
            results[i].report.target.win_records,
            single_records,
            true
        );
    }

    // 1. 打印综合 3-Baseline 性能对账表
    std::cout << "\n====================================================================================================================\n";
    std::cout << " 📊 SDSCC 细粒度微柱 MoE 结构性能对账账本 (5,000 局实证盲测)\n";
    std::cout << "====================================================================================================================\n";
    std::cout << "| " << std::left << std::setw(32) << "架构配置"
              << "| " << std::setw(12) << "胜场/总数"
              << "| " << std::setw(24) << "胜率 (Wilson 95% CI)"
              << "| " << std::setw(15) << "场均筹码"
              << "| " << std::setw(15) << "McNemar vs B1"
              << "| " << std::setw(15) << "McNemar vs 单柱"
              << "| " << std::setw(14) << "单步时延 (ns)"
              << "| " << std::setw(12) << "激活算力比"
              << "|\n";
    std::cout << "|---------------------------------|-------------|-------------------------|----------------|----------------|----------------|--------------|------------|\n";

    const auto& b0 = harness.baseline_0();
    const auto& b1 = harness.baseline_1();
    std::stringstream b0_cnt, b0_ch, b1_cnt, b1_ch;
    b0_cnt << b0.wins << "/" << b0.total_episodes;
    b0_ch << (b0.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b0.avg_chips << " 豆";
    std::cout << "| " << std::left << std::setw(32) << "Baseline 0 (随机无门禁)"
              << "| " << std::setw(12) << b0_cnt.str()
              << "| " << std::setw(24) << b0.win_rate_ci.to_string(1, true)
              << "| " << std::setw(15) << b0_ch.str()
              << "| " << std::setw(15) << "对照 (Control)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns"
              << "| " << std::setw(12) << "-"
              << "|\n";

    b1_cnt << b1.wins << "/" << b1.total_episodes;
    b1_ch << (b1.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b1.avg_chips << " 豆";
    std::cout << "| " << std::left << std::setw(32) << "Baseline 1 (随机门禁>=5)"
              << "| " << std::setw(12) << b1_cnt.str()
              << "| " << std::setw(24) << b1.win_rate_ci.to_string(1, true)
              << "| " << std::setw(15) << b1_ch.str()
              << "| " << std::setw(15) << "基准 (Gate>=5)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns"
              << "| " << std::setw(12) << "-"
              << "|\n";
    std::cout << "|---------------------------------|-------------|-------------------------|----------------|----------------|----------------|--------------|------------|\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        const auto& tgt = r.report.target;
        std::stringstream cnt_ss, ch_ss, vs_b1_ss, vs_single_ss, lat_ss, ratio_ss;
        cnt_ss << tgt.wins << "/" << tgt.total_episodes;
        ch_ss << (tgt.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << tgt.avg_chips << " 豆";

        if (r.report.mcnemar_target_vs_b1.p_value < 1e-4) {
            vs_b1_ss << "p=" << std::scientific << std::setprecision(1) << r.report.mcnemar_target_vs_b1.p_value;
        } else {
            vs_b1_ss << "p=" << std::fixed << std::setprecision(4) << r.report.mcnemar_target_vs_b1.p_value;
        }

        if (i == 0) {
            vs_single_ss << "基准 (Single)";
        } else {
            if (r.mcnemar_vs_single.p_value < 1e-4) {
                vs_single_ss << "p=" << std::scientific << std::setprecision(1) << r.mcnemar_vs_single.p_value;
            } else {
                vs_single_ss << "p=" << std::fixed << std::setprecision(4) << r.mcnemar_vs_single.p_value;
            }
        }
        lat_ss << std::fixed << std::setprecision(1) << r.avg_latency_ns << " ns";
        ratio_ss << std::fixed << std::setprecision(1) << (r.active_compute_ratio * 100.0) << "%";

        std::cout << "| " << std::left << std::setw(32) << r.name
                  << "| " << std::setw(12) << cnt_ss.str()
                  << "| " << std::setw(24) << tgt.win_rate_ci.to_string(1, true)
                  << "| " << std::setw(15) << ch_ss.str()
                  << "| " << std::setw(15) << vs_b1_ss.str()
                  << "| " << std::setw(15) << vs_single_ss.str()
                  << "| " << std::setw(14) << lat_ss.str()
                  << "| " << std::setw(12) << ratio_ss.str()
                  << "|\n";
    }
    std::cout << "====================================================================================================================\n\n";

    // 2. 打印人工先验门控调度矩阵 (Explicit Handcrafted Routing Matrix)
    const auto& moe4 = results[2];
    std::cout << "🎯 [Sparse MoE N=4 人工先验路由分发分布 (Handcrafted Prior Routing Distribution)]:\n";
    std::cout << "   (注意: 如下分工比例源于人工先验偏置，绝非自发涌现)\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";
    std::cout << "| " << std::left << std::setw(20) << "对局阶段 (Phase)"
              << "| " << std::setw(16) << "微柱0 (先验进攻)"
              << "| " << std::setw(16) << "微柱1 (先验协同)"
              << "| " << std::setw(16) << "微柱2 (先验调控)"
              << "| " << std::setw(16) << "微柱3 (先验斩杀)"
              << "|\n";
    std::cout << "|---------------------|-----------------|-----------------|-----------------|-----------------|\n";

    auto print_phase_row_4 = [&](const std::string& name, const uint64_t calls[8]) {
        uint64_t sum = calls[0] + calls[1] + calls[2] + calls[3];
        std::stringstream c0, c1, c2, c3;
        if (sum > 0) {
            c0 << calls[0] << " (" << std::fixed << std::setprecision(1) << (100.0 * calls[0] / sum) << "%)";
            c1 << calls[1] << " (" << std::fixed << std::setprecision(1) << (100.0 * calls[1] / sum) << "%)";
            c2 << calls[2] << " (" << std::fixed << std::setprecision(1) << (100.0 * calls[2] / sum) << "%)";
            c3 << calls[3] << " (" << std::fixed << std::setprecision(1) << (100.0 * calls[3] / sum) << "%)";
        } else {
            c0 << "0 (0.0%)"; c1 << "0 (0.0%)"; c2 << "0 (0.0%)"; c3 << "0 (0.0%)";
        }
        std::cout << "| " << std::left << std::setw(20) << name
                  << "| " << std::setw(16) << c0.str()
                  << "| " << std::setw(16) << c1.str()
                  << "| " << std::setw(16) << c2.str()
                  << "| " << std::setw(16) << c3.str()
                  << "|\n";
    };

    print_phase_row_4("叫牌/起手 (C >= 17)", moe4.phase_stats.bidding_calls);
    print_phase_row_4("开局压制 (13<=C<=16)", moe4.phase_stats.early_calls);
    print_phase_row_4("中盘协同 (6<=C<=12)", moe4.phase_stats.midgame_calls);
    print_phase_row_4("残局斩杀 (C <= 5)", moe4.phase_stats.endgame_calls);
    std::cout << "-----------------------------------------------------------------------------------------\n\n";

    // 3. 核心物理与实证结论对比
    double dense_lat = results[1].avg_latency_ns;
    double sparse_lat = results[2].avg_latency_ns;
    double speedup = (sparse_lat > 0) ? (dense_lat / sparse_lat) : 1.0;

    std::cout << "📌 [科学实证客观结论]:\n";
    std::cout << "   1. 结构零算力旁路验证: Sparse MoE (N=4, Top-2) 仅激活 2/4 = 50.0% 微柱专家，\n"
              << "      推演总次数为 " << results[2].total_expert_evaluations << " 次 (Dense 全激活为 "
              << results[1].total_expert_evaluations << " 次)，严格验证 O(K) 零算力旁路 (Zero-Compute Bypass) 计算缩减特性!\n";
    std::cout << "   2. 推理加速比验证: Dense 全量多微柱耗时 " << dense_lat << " ns -> Sparse 稀疏门控耗时 "
              << sparse_lat << " ns (实测加速比: " << std::fixed << std::setprecision(2) << speedup << "x)!\n";
    std::cout << "   3. 胜率与统计显著性定性: Sparse MoE 胜率 " << results[2].report.target.win_rate << "% vs 单微柱 " 
              << results[0].report.target.win_rate << "%\n"
              << "      McNemar vs 单微柱检定: " << results[2].mcnemar_vs_single.to_string(4) << "\n"
              << "      显著性状态: " << (results[2].mcnemar_vs_single.significant_01 ? "✅ 统计显著 (p<0.01)" : "❌ 未达统计显著 (p>=0.01)") << "\n";
    std::cout << "   4. 叙事纠偏总结: 明确本测试的核心贡献为硬件级非冯稀疏路由与旁路时延优化，\n"
              << "      严禁在未经过端到端梯度演化训练时使用'自主涌现认知分工'等误导性话术。\n\n";

    return 0;
}
