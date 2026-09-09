// P9: 新核心原生牌局 runner — 结构改进→牌力转化的载体
// 协议与 legacy eval_cand 完全一致 (种子 3100000+g*97, 叫牌 17.5, 40 轮,
// 角色分层, Wilson95), 前向内核 = 新核心 RuntimeState + CompiledExecutor。
// 每决策 reset (与 57.0% 基线协议等价, 保证配对可比); 在线轨迹版后续开关。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>

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

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int games = argc > 1 ? std::atoi(argv[1]) : 500;
    const char* model = argc > 2 ? argv[2] : "checkpoints/doudizhu_cand_scorer.bin";

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    printf("[P9] 冠军冷迁移: cells=%zu\n", org.cells.size());
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

    uint32_t lcg = 424243u;
    auto rnd01 = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (double)(lcg >> 8) / 16777216.0;
    };
    int wins = 0, ll_w = 0, ll_n = 0, fm_w = 0, fm_n = 0;
    long live_steps = 0, live_pass = 0;
    for (int g = 0; g < games; ++g) {
        kun::DouDiZhuCardGameTask task(40, (uint32_t)(3100000 + g * 97), 17.5);
        bool done = false;
        bool agent_is_landlord = false;
        {
            kun::DouDiZhuCardGameTask probe = task;
            int lr0 = probe.teacher_play_capture();
            (void)lr0;
            agent_is_landlord = (probe.role() == 1);
        }
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
                    for (int d = 0; d < 44 && d < (int)obs44.size(); ++d) in[d] = obs44[d];
                    for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
                    rt.reset_episode();
                    auto r = ex.step(rt, in);
                    if (!r.ok()) { printf("[错误] 前向失败\n"); return 1; }
                    const double sc = rt.cell_states()[head_index].output_val;
                    if (!std::isfinite(sc)) { printf("[错误] 非有限分数\n"); return 1; }
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            int pre = task.cards_left(0);
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            if (res.success) wins++;
            live_steps++;
            if (task.cards_left(0) == pre) live_pass++;
            if (done) {
                if (agent_is_landlord) { ll_n++; if (res.success) ll_w++; }
                else { fm_n++; if (res.success) fm_w++; }
            }
        }
    }
    printf("[角色分层] 地主 %d/%d = %.1f%% | 农民 %d/%d = %.1f%%\n",
           ll_w, ll_n, ll_n ? 100.0 * ll_w / ll_n : 0.0,
           fm_w, fm_n, fm_n ? 100.0 * fm_w / fm_n : 0.0);
    printf("[P9实战/新核心] %d 局: 胜率 %.1f%% (%d/%d) Wilson95下界 %.1f%% | 步数 %ld 过牌 %.1f%%\n",
           games, 100.0 * wins / games, wins, games, 100.0 * wilson_lower(wins, games),
           live_steps, 100.0 * live_pass / std::max(1L, live_steps));
    return 0;
}
