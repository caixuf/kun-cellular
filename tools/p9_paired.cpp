// P9-PAIRED: 同种子双策略配对评测 — M3 超越判据的统计载体
// 输出: 两模型各自胜率 + 2x2 列联 + McNemar 配对检验 (同组种子, 方差远小于独立比较)
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/statistical_evaluation.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

using namespace kun;

namespace {

struct Pheno {
    kun::migration::ColdAssemblyConfig cfg;
    std::unique_ptr<core::Phenotype> phenotype;
    core::RuntimeState* rt = nullptr;
    core::CompiledExecutor* ex = nullptr;
    size_t score_idx = 0;
};

bool load_pheno(const char* path, uint64_t organism_id, Pheno& out) {
    CellularOrganism org;
    auto b = CellularOrganism::load_checkpoint_bin(path);
    if (!b.cells.empty()) org = std::move(b);
    else org = CellularOrganism::load_checkpoint_json(path);
    if (org.cells.empty()) { printf("[错误] 加载失败: %s\n", path); return false; }
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = organism_id;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(9), core::GraphRevision(1), cfg);
    if (!assembled.ok()) { printf("[错误] 装配 %s: %s\n", path, assembled.diagnostic.c_str()); return false; }
    out.phenotype = std::move(assembled.phenotype);
    out.rt = &out.phenotype->runtime();
    out.ex = &out.phenotype->executor();
    const auto& plan_cells = out.rt->plan()->cells();
    for (size_t i = 0; i < plan_cells.size(); ++i) {
        if (plan_cells[i].type != CellType::ACT_CHANNEL) continue;
        const auto& pv = out.rt->parameters()[plan_cells[i].parameter_indices[1]].value;
        const auto* ch = std::get_if<core::ChannelIndex>(&pv);
        if (ch && ch->value == 0) { out.score_idx = i; break; }
    }
    return true;
}

std::vector<uint8_t> run_family(Pheno& p, int games, uint32_t seed_base) {
    std::vector<uint8_t> won;
    won.reserve(games);
    core::RuntimeState& rt = *p.rt;
    core::CompiledExecutor& ex = *p.ex;
    for (int g = 0; g < games; ++g) {
        kun::DouDiZhuCardGameTask task(40, seed_base + (uint32_t)(g * 97), 17.5);
        bool done = false;
        {
            kun::DouDiZhuCardGameTask probe = task;
            int lr0 = probe.teacher_play_capture();
            (void)lr0;
        }
        bool game_won = false;
        while (!done) {
            auto cands = task.enumerate_candidates(0);
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
                    if (!r.ok()) { printf("[错误] 前向失败\n"); exit(1); }
                    const double sc = rt.cell_states()[p.score_idx].output_val;
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            game_won = res.success;
        }
        won.push_back(game_won ? 1 : 0);
    }
    return won;
}

}  // namespace

// 用法: p9_paired <games> <modelA> <modelB> [seed_base] [teacher]
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        printf("用法: p9_paired <games> <modelA> <modelB> [seed_base=3100000] [teacher]\n");
        printf("  teacher: modelB 以教师启发式驱动 (配对判定 学生vs教师)\n");
        return 1;
    }
    const int games = std::atoi(argv[1]);
    const uint32_t seed_base = argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 3100000u;
    const bool b_is_teacher = argc > 5 && !std::strcmp(argv[5], "teacher");

    Pheno pa;
    if (!load_pheno(argv[2], 9100, pa)) return 1;
    auto wa = run_family(pa, games, seed_base);
    int wins_a = 0;
    for (uint8_t w : wa) wins_a += w;

    int wins_b = 0;
    uint64_t m_b = 0, m_c = 0, m_a = 0, m_d = 0;
    if (b_is_teacher) {
        for (int g = 0; g < games; ++g) {
            kun::DouDiZhuCardGameTask task(40, seed_base + (uint32_t)(g * 97), 17.5);
            bool done = false;
            task.teacher_play_capture();
            auto res = task.settle_turn();
            done = res.done;
            while (!done) {
                task.teacher_play_capture();
                res = task.settle_turn();
                done = res.done;
            }
            const uint8_t wb = res.success ? 1 : 0;
            wins_b += wb;
            if (wa[g] && wb) m_a++;
            else if (wa[g]) m_b++;
            else if (wb) m_c++;
            else m_d++;
        }
    } else {
        Pheno pb;
        if (!load_pheno(argv[3], 9200, pb)) return 1;
        auto wb = run_family(pb, games, seed_base);
        for (int g = 0; g < games; ++g) {
            wins_b += wb[g];
            if (wa[g] && wb[g]) m_a++;
            else if (wa[g]) m_b++;
            else if (wb[g]) m_c++;
            else m_d++;
        }
    }
    auto mres = kun::mcnemar_test(m_b, m_c, m_a, m_d);
    printf("[A] %s: %.1f%% (%d/%d) Wilson 下界 %.1f%%\n", argv[2],
           100.0 * wins_a / games, wins_a, games,
           100.0 * (wins_a ? [] (int w, int n) {
               const double z = 1.96, p = (double)w / n, d = 1 + z*z/n;
               return (p + z*z/(2*n) - z*std::sqrt(p*(1-p)/n + z*z/(4*n*n))) / d; }(wins_a, games) : 0.0));
    printf("[B] %s%s: %.1f%% (%d/%d)\n", argv[3], b_is_teacher ? " (teacher)" : "",
           100.0 * wins_b / games, wins_b, games);
    printf("[2x2] A&B胜=%llu A胜B负(b)=%llu A负B胜(c)=%llu 双负=%llu | McNemar chi2=%.3f p=%.4f %s\n",
           (unsigned long long)m_a, (unsigned long long)m_b, (unsigned long long)m_c,
           (unsigned long long)m_d, mres.chi2, mres.p_value,
           mres.significant_05 ? "→ 显著(p<0.05)" : "→ 不显著");
    return 0;
}
