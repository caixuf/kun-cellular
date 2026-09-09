// P9-D: 分歧分析 — 冠军选点 vs 启发式教师选点, 逐决策配对
// 目的: 定位 1.3pp 差距的信息源 (57.0% → 58.3%)
//   - 分歧率、分歧决策分布 (自由出牌 vs 跟牌)
//   - 按本局分歧次数分桶的胜率 (0 / 1 / ≥2)
//   - 一致局胜率 vs 分歧局胜率; 分歧处双方选择各自的后续方向
// 教师选点: probe 副本上 teacher_play_capture() (与 gen_dataset_cand 同法, 不推进真实账本)
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

#include "kun/cellular/statistical_evaluation.hpp"

using namespace kun;

namespace {

double wilson_lower(int wins, int n) {
    if (n == 0) return 0.0;
    const double z = 1.96, p = (double)wins / n;
    const double d = 1 + z * z / n;
    const double center = p + z * z / (2 * n);
    const double rad = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return (center - rad) / d;
}

// 教师 label 映射 (与 train_doudizhu_bc_distill.cpp:650-658 完全同法)
int teacher_label(DouDiZhuCardGameTask& probe, const std::vector<DouDiZhuCardGameTask::CandPlay>& cands) {
    int lr = probe.teacher_play_capture();
    auto t_trick = probe.table_trick();
    for (size_t i = 0; i < cands.size(); ++i) {
        if (lr < 0) { if (cands[i].type == 0) return (int)i; }
        else if (cands[i].type == (int)t_trick.type && cands[i].rank == t_trick.rank) return (int)i;
    }
    return -1;   // 教师动作不在候选集 (协议与数据生成一致: 跳过该决策)
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int games = argc > 1 ? std::atoi(argv[1]) : 2000;
    const char* model = argc > 2 ? argv[2] : "checkpoints/doudizhu_cand_scorer.bin";

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    printf("[P9-D] 冠军冷迁移: cells=%zu\n", org.cells.size());
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 9000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(9), core::GraphRevision(1), cfg);
    if (!assembled.ok()) { printf("[错误] 装配: %s\n", assembled.diagnostic.c_str()); return 1; }
    core::RuntimeState& rt = assembled.phenotype->runtime();
    core::CompiledExecutor& ex = assembled.phenotype->executor();
    size_t head_index = 0;
    for (size_t i = 0; i < rt.cell_states().size(); ++i)
        if (rt.cell_states()[i].type == CellType::ACT_CHANNEL) { head_index = i; break; }

    long total_decisions = 0, disagree_decisions = 0, teacher_skip = 0;
    long d_free = 0, d_follow = 0;                       // 分歧决策: 自由领出 vs 跟牌
    long d_champ_pass = 0, d_teacher_pass = 0;           // 分歧处: 冠军过牌 / 教师过牌
    // 分桶: 本局分歧次数 → 胜率
    int b0_n = 0, b0_w = 0, b1_n = 0, b1_w = 0, b2_n = 0, b2_w = 0;
    long free_decisions = 0, follow_decisions = 0;       // 全体决策中两类占比
    std::vector<uint8_t> champ_won;                      // 逐种子配对用

    for (int g = 0; g < games; ++g) {
        kun::DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        bool done = false;
        bool game_won = false;
        int game_disagrees = 0;
        while (!done) {
            auto cands = task.enumerate_candidates(0);
            // 冠军选点 (与 p9_runner 逐位一致)
            size_t pick = 0;
            {
                auto o = task.current_observation();
                std::vector<float> obs44(o.begin(), o.end());
                for (float v : task.seat_context()) obs44.push_back(v);
                double best = -1e308;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto cf = task.candidate_features(cands[i]);
                    std::vector<double> in(56);
                    for (int d = 0; d < 44; ++d) in[d] = obs44[d];
                    for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
                    rt.reset_episode();
                    auto r = ex.step(rt, in);
                    if (!r.ok()) { printf("[错误] 前向失败\n"); return 1; }
                    const double sc = rt.cell_states()[head_index].output_val;
                    if (!std::isfinite(sc)) { printf("[错误] 非有限分数\n"); return 1; }
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            // 教师选点 (probe 副本, 不推进真实账本)
            kun::DouDiZhuCardGameTask probe = task;
            int tl = teacher_label(probe, cands);
            bool free_lead = (task.table_trick().type == 0);   // TRICK_NONE → 自由领出
            total_decisions++;
            if (free_lead) free_decisions++; else follow_decisions++;
            if (tl < 0) { teacher_skip++; }
            else if ((size_t)tl != pick) {
                game_disagrees++;
                disagree_decisions++;
                if (free_lead) d_free++; else d_follow++;
                if (cands[pick].type == 0) d_champ_pass++;
                if (cands[tl].type == 0) d_teacher_pass++;
            }
            // 执行冠军动作 (与 p9_runner 协议一致)
            int pre = task.cards_left(0);
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            game_won = res.success;
        }
        if (game_disagrees == 0) { b0_n++; if (game_won) b0_w++; }
        else if (game_disagrees == 1) { b1_n++; if (game_won) b1_w++; }
        else { b2_n++; if (game_won) b2_w++; }
        champ_won.push_back(game_won ? 1 : 0);
    }

    printf("[分歧率] 决策级 %ld/%ld = %.2f%% (教师不可映射跳过 %ld) | 决策构成: 自由 %ld 跟牌 %ld\n",
           disagree_decisions, total_decisions, 100.0 * disagree_decisions / std::max(1L, total_decisions),
           teacher_skip, free_decisions, follow_decisions);
    printf("[分歧构成] 自由领出 %ld | 跟牌 %ld | 冠军选过牌 %ld | 教师选过牌 %ld\n",
           d_free, d_follow, d_champ_pass, d_teacher_pass);
    printf("[分桶胜率] 0分歧 %d/%d=%.1f%% | 1分歧 %d/%d=%.1f%% | ≥2分歧 %d/%d=%.1f%%\n",
           b0_w, b0_n, b0_n ? 100.0 * b0_w / b0_n : 0.0,
           b1_w, b1_n, b1_n ? 100.0 * b1_w / b1_n : 0.0,
           b2_w, b2_n, b2_n ? 100.0 * b2_w / b2_n : 0.0);
    const int agree_n = b0_n, agree_w = b0_w;
    const int dis_n = b1_n + b2_n, dis_w = b1_w + b2_w;
    printf("[对照] 一致局胜率 %.1f%% (%d/%d) vs 含分歧局胜率 %.1f%% (%d/%d) | Wilson(分歧局下界) %.1f%%\n",
           agree_n ? 100.0 * agree_w / agree_n : 0.0, agree_w, agree_n,
           dis_n ? 100.0 * dis_w / dis_n : 0.0, dis_w, dis_n, wilson_lower(dis_w, dis_n));

    // ── 同种子配对: 教师驱动重放 (协议 = eval_cand use_teacher 路径) ──
    long t_wins = 0, m_b = 0, m_c = 0, m_a = 0, m_d = 0;
    for (int g = 0; g < games; ++g) {
        kun::DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        bool done = false;
        task.teacher_play_capture();
        auto res = task.settle_turn();
        done = res.done;
        while (!done) {
            task.teacher_play_capture();
            res = task.settle_turn();
            done = res.done;
        }
        const bool tw = res.success;
        if (tw) t_wins++;
        // 2x2: a=双胜 b=学生胜师负 c=学生负师胜 d=双负
        if (champ_won[g] && tw) m_a++;
        else if (champ_won[g] && !tw) m_b++;
        else if (!champ_won[g] && tw) m_c++;
        else m_d++;
    }
    auto mres = kun::mcnemar_test((uint64_t)m_b, (uint64_t)m_c, (uint64_t)m_a, (uint64_t)m_d);
    printf("[同种子配对] 学生 %.1f%% (%d/%d) vs 教师 %.1f%% (%ld/%d) | b=%ld c=%ld a=%ld d=%ld | McNemar chi2=%.3f p=%.4f %s\n",
           100.0 * (m_a + m_b) / games, m_a + m_b, games,
           100.0 * t_wins / games, t_wins, games,
           m_b, m_c, m_a, m_d, mres.chi2, mres.p_value,
           mres.significant_05 ? "(p<0.05 显著)" : "(不显著)");
    return 0;
}
