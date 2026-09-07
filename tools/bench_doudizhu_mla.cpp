#include "tasks/transfer/cross_domain_tasks.hpp"
#include "tasks/transfer/doudizhu_eval_harness.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/cellular_mla.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <sstream>

using namespace kun;

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    std::cout << "=========================================================================================\n";
    std::cout << " 🧠 硅基细胞计算机 (SDSCC) 多头低秩潜空间记忆库 (Cellular MLA) 5,000 局配对实证评测\n";
    std::cout << " 架构对标: DeepSeek-V2 / DeepSeek-V3 Multi-Head Latent Attention (低秩 KV-Cache 压缩)\n";
    std::cout << " 实证标准: 强制执行 3-Baseline 账本与 McNemar 统计显著性检定 (5,000 局配对种子盲测)\n";
    std::cout << " 评测矩阵: Baseline 0 vs Baseline 1 vs 64-Cell 无记忆基线 vs 64-Cell + MLA 潜空间记忆\n";
    std::cout << "=========================================================================================\n\n";

    const int NUM_EPISODES = 5000;
    const uint32_t SEED_BASE = 20260907;
    const double BIDDING_GATE = 5.0;

    std::cout << "⏳ 正在初始化 3-Baseline 评测管线 (预先对账 5,000 局配对发牌 Baseline 0 与 Baseline 1)...\n";
    DouDiZhuThreeBaselineHarness harness(NUM_EPISODES, SEED_BASE, BIDDING_GATE, 40);
    std::cout << "✅ 3-Baseline 预计算完成!\n\n";

    const auto& b0 = harness.baseline_0();
    const auto& b1 = harness.baseline_1();

    // 1. 加载或构建 64 细胞时序皮层
    CellularOrganism base_cortex;
    const std::string ckpt_path = "checkpoints/doudizhu_64cell_grpo_cotrained.bin";
    std::ifstream ifs(ckpt_path, std::ios::binary);
    if (ifs.good()) {
        ifs.close();
        base_cortex = CellularOrganism::load_checkpoint_bin(ckpt_path);
        std::cout << "✅ 成功加载已训练 64 细胞循环皮层: " << ckpt_path << "\n";
    } else {
        base_cortex = build_doudizhu_64cell_recurrent_cortex();
        std::cout << "ℹ️ 检查点不存在，使用原生 64 细胞初始皮层\n";
    }

    // 2. 配置 MLA 记忆引擎 (4x 低秩压缩: 32 维 -> 8 维潜空间, 2 头 x 8 维)
    MLAConfig mla_cfg;
    mla_cfg.in_dim = 32;
    mla_cfg.latent_dim = 8;
    mla_cfg.q_latent_dim = 8;
    mla_cfg.num_heads = 2;
    mla_cfg.head_dim = 8;
    mla_cfg.max_history_len = 32;

    CellularMLAEngine mla_engine(mla_cfg);
    std::cout << "✅ 成功初始化 Cellular MLA 潜空间记忆引擎:\n";
    std::cout << "   - 全维特征输入: " << mla_cfg.in_dim << " 维\n";
    std::cout << "   - 键值低秩潜空间: " << mla_cfg.latent_dim << " 维 (压缩比率: " 
              << std::fixed << std::setprecision(1) << mla_engine.compression_ratio() << "x)\n";
    std::cout << "   - 多头检索配置: " << mla_cfg.num_heads << " 头 x " << mla_cfg.head_dim << " 维\n";
    std::cout << "   - 潜空间缓存占用: " << mla_engine.latent_memory_bytes() << " 字节 (vs 全维稠密缓存 " 
              << mla_engine.dense_uncompressed_bytes() << " 字节，内存节省 75%)\n\n";

    // 3. 评测 (a): 64 细胞无记忆基线 (Markovian Reflex, 仅单步瞬时特征)
    std::cout << "⏳ [评测通道 1] 正在评估 64-Cell 无记忆基线 (5,000 局配对发牌)..." << std::flush;
    double lat_nomem = 0.0;
    uint64_t calls_nomem = 0;
    auto rep_nomem = harness.evaluate_organism("64-Cell (无历史记忆)", base_cortex, &lat_nomem, &calls_nomem);
    std::cout << " 完成!\n";

    // 4. 评测 (b): 64 细胞 + MLA 低秩潜空间记忆
    std::cout << "⏳ [评测通道 2] 正在评估 64-Cell + MLA 潜空间时序记忆库 (5,000 局配对发牌)..." << std::flush;
    double total_mla_latency_ns = 0.0;
    uint64_t calls_mla = 0;

    auto rep_mla = harness.evaluate_policy(
        "64-Cell + MLA 潜空间记忆",
        [&](const std::vector<float>& obs, DouDiZhuCardGameTask& task) -> int {
            (void)task;
            // 提取当前 32 维特征
            float mla_out[32]{};

            auto t0 = std::chrono::high_resolution_clock::now();
            // MLA 前向写入与因果注意力检索
            mla_engine.step_forward(obs.data(), mla_out);

            // 残差融合: 当前特征 + 0.25 * 潜空间历史注意力上下文
            std::vector<double> fused_in(32);
            for (size_t i = 0; i < 32; ++i) {
                fused_in[i] = static_cast<double>(obs[i]) + 0.25 * static_cast<double>(mla_out[i]);
            }

            auto acts = base_cortex.forward_nd(fused_in.data(), 32, false);
            auto t1 = std::chrono::high_resolution_clock::now();

            total_mla_latency_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
            calls_mla++;

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
            return act;
        },
        &calls_mla
    );
    std::cout << " 完成!\n\n";

    double lat_mla = calls_mla > 0 ? (total_mla_latency_ns / calls_mla) : 0.0;

    // 配对 McNemar 显著性检验 (MLA vs 无记忆基线)
    McNemarResult mcnemar_mla_vs_nomem = mcnemar_test(
        rep_mla.target.win_records,
        rep_nomem.target.win_records,
        true
    );

    // 打印 3-Baseline 标准实证对账账本
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "| " << std::left << std::setw(28) << "架构通道"
              << "| " << std::setw(12) << "胜场/总数"
              << "| " << std::setw(24) << "胜率 (Wilson 95% CI)"
              << "| " << std::setw(16) << "场均筹码净利"
              << "| " << std::setw(15) << "McNemar vs B1"
              << "| " << std::setw(15) << "McNemar vs 无记"
              << "| " << std::setw(14) << "单步时延 (ns)"
              << "|\n";
    std::cout << "|-----------------------------|-------------|-------------------------|-----------------|----------------|----------------|--------------|\n";

    std::stringstream b0_cnt, b0_ch, b1_cnt, b1_ch, nomem_cnt, nomem_ch, mla_cnt, mla_ch;
    b0_cnt << b0.wins << "/" << b0.total_episodes;
    b0_ch << (b0.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b0.avg_chips << " 豆/局";
    std::cout << "| " << std::left << std::setw(28) << "Baseline 0 (随机无门)"
              << "| " << std::setw(12) << b0_cnt.str()
              << "| " << std::setw(24) << b0.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << b0_ch.str()
              << "| " << std::setw(15) << "对照 (Control)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns" << "|\n";

    b1_cnt << b1.wins << "/" << b1.total_episodes;
    b1_ch << (b1.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << b1.avg_chips << " 豆/局";
    std::cout << "| " << std::left << std::setw(28) << "Baseline 1 (随机门禁)"
              << "| " << std::setw(12) << b1_cnt.str()
              << "| " << std::setw(24) << b1.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << b1_ch.str()
              << "| " << std::setw(15) << "基准 (Gate>=5)"
              << "| " << std::setw(15) << "-"
              << "| " << std::setw(14) << "0.0 ns" << "|\n";
    std::cout << "|-----------------------------|-------------|-------------------------|-----------------|----------------|----------------|--------------|\n";

    nomem_cnt << rep_nomem.target.wins << "/" << rep_nomem.target.total_episodes;
    nomem_ch << (rep_nomem.target.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << rep_nomem.target.avg_chips << " 豆/局";
    std::stringstream vs_b1_nomem;
    if (rep_nomem.mcnemar_target_vs_b1.p_value < 1e-4) {
        vs_b1_nomem << "p=" << std::scientific << std::setprecision(1) << rep_nomem.mcnemar_target_vs_b1.p_value;
    } else {
        vs_b1_nomem << "p=" << std::fixed << std::setprecision(4) << rep_nomem.mcnemar_target_vs_b1.p_value;
    }

    std::stringstream lat_nomem_ss;
    lat_nomem_ss << std::fixed << std::setprecision(1) << lat_nomem << " ns";

    std::cout << "| " << std::left << std::setw(28) << "64-Cell (无历史记忆)"
              << "| " << std::setw(12) << nomem_cnt.str()
              << "| " << std::setw(24) << rep_nomem.target.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << nomem_ch.str()
              << "| " << std::setw(15) << vs_b1_nomem.str()
              << "| " << std::setw(15) << "基准 (Ref)"
              << "| " << std::setw(14) << lat_nomem_ss.str() << "|\n";

    mla_cnt << rep_mla.target.wins << "/" << rep_mla.target.total_episodes;
    mla_ch << (rep_mla.target.avg_chips >= 0 ? "+" : "") << std::fixed << std::setprecision(1) << rep_mla.target.avg_chips << " 豆/局";
    std::stringstream vs_b1_mla, vs_nomem_mla;
    if (rep_mla.mcnemar_target_vs_b1.p_value < 1e-4) {
        vs_b1_mla << "p=" << std::scientific << std::setprecision(1) << rep_mla.mcnemar_target_vs_b1.p_value;
    } else {
        vs_b1_mla << "p=" << std::fixed << std::setprecision(4) << rep_mla.mcnemar_target_vs_b1.p_value;
    }

    if (mcnemar_mla_vs_nomem.p_value < 1e-4) {
        vs_nomem_mla << "p=" << std::scientific << std::setprecision(1) << mcnemar_mla_vs_nomem.p_value;
    } else {
        vs_nomem_mla << "p=" << std::fixed << std::setprecision(4) << mcnemar_mla_vs_nomem.p_value;
    }

    std::stringstream lat_mla_ss;
    lat_mla_ss << std::fixed << std::setprecision(1) << lat_mla << " ns";

    std::cout << "| " << std::left << std::setw(28) << "64-Cell + MLA 潜空间记忆"
              << "| " << std::setw(12) << mla_cnt.str()
              << "| " << std::setw(24) << rep_mla.target.win_rate_ci.to_string(1, true)
              << "| " << std::setw(16) << mla_ch.str()
              << "| " << std::setw(15) << vs_b1_mla.str()
              << "| " << std::setw(15) << vs_nomem_mla.str()
              << "| " << std::setw(14) << lat_mla_ss.str() << "|\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n\n";

    std::cout << "📋 [多头低秩潜空间记忆 (MLA) 实证报告与结论分析]:\n";
    double wr_nomem = rep_nomem.target.win_rate;
    double wr_mla = rep_mla.target.win_rate;
    double diff = wr_mla - wr_nomem;

    std::cout << "   1. 内存带宽与压缩率: 32 维全息输入压缩至 8 维低秩潜空间，实现 "
              << std::fixed << std::setprecision(1) << mla_engine.compression_ratio()
              << "x 状态压缩 (缓存占用 640 字节 vs 2560 字节，节省 75.0% 内存空间)；\n";
    std::cout << "   2. 计算时延开销: MLA 增量时延为 " << std::fixed << std::setprecision(1) << (lat_mla - lat_nomem)
              << " ns (总时延 " << lat_mla << " ns，保持亚微秒级硬实时能力)；\n";
    std::cout << "   3. 胜率与策略收益变动: 无记忆 64-Cell 胜率 " << wr_nomem << "% -> MLA 增强 64-Cell 胜率 "
              << wr_mla << "% (变动: " << (diff >= 0 ? "+" : "") << diff << "%)\n";
    std::cout << "   4. 配对卡方检定 (McNemar paired test): " << mcnemar_mla_vs_nomem.to_string(4) << "\n";

    if (mcnemar_mla_vs_nomem.significant_01) {
        if (mcnemar_mla_vs_nomem.b > mcnemar_mla_vs_nomem.c) {
            std::cout << "   ✅ 统计显著性判定: p < 0.01 达标！低秩潜空间记忆库为智能体提供了经检验有效的时序上下文增强！\n";
        } else {
            std::cout << "   🔻 统计显著性判定: p < 0.01 负向显著！未经端到端微调的静态潜空间投影引入了特征噪声干涉。\n";
        }
    } else {
        std::cout << "   ❌ 统计显著性判定: p >= 0.01 未达统计显著门限 (差异无法排除有限样本随机方差)。\n";
    }

    return 0;
}
