// P9-ONLINE: 在线轨迹学习 (M1) — 优势加权回归 (REINFORCE-as-regression) 双头更新
// 协议: docs/superpowers/plans/2026-09-09-online-rl-protocol.md (预注册, 不可改)
//   - 线协议每决策 reset (p9_runner:85) → 一局 = 独立 (状态,候选) 评估 + 局末团队回报
//   - score 头 (channel 0): 被选中行 target=±A (A=自适应归一化回报), 其余 0
//   - value 头 (channel 1, 冠军中零初始化未消费): 全行 target=won?1:0 (MC 回报回归)
//   - 每局一次 backward + step_adam; 每 N 局 holdout 配对评测 + 回退/漂移门
//   - holdout 种子族 9100000+g*97 与训练族 3100000+g*97 完全不相交
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/statistical_evaluation.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <string>
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

std::vector<core::ParameterBinding> full_bindings(const core::RuntimeState& rt) {
    std::vector<core::ParameterBinding> out;
    const auto& entries = rt.parameters();
    for (size_t i = 0; i < entries.size(); ++i) out.push_back(entries[i].binding);
    return out;
}

// 突触发射序重建 (与 import_execution_snapshot 完全同法): recurrent 先行,
// 再按 execution_order × out_csr 展开非递归边 → edge_id = 序号+1
std::vector<size_t> rebuild_emission_order(const CellularOrganism& org) {
    std::vector<size_t> order;
    std::vector<uint8_t> emitted(org.compiled_synapses_.size(), 0);
    for (size_t i = 0; i < org.compiled_synapses_.size(); ++i)
        if (org.compiled_synapses_[i].is_recurrent) { order.push_back(i); emitted[i] = 1; }
    for (size_t cell_index : org.execution_order_)
        for (size_t c = org.out_start_[cell_index]; c < org.out_start_[cell_index + 1]; ++c) {
            const size_t idx = org.out_edges_[c];
            if (!org.compiled_synapses_[idx].is_recurrent) {
                if (emitted[idx] != 0) return {};   // 重复引用 → 拒绝
                order.push_back(idx);
                emitted[idx] = 1;
            }
        }
    if (order.size() != org.compiled_synapses_.size()) return {};
    return order;
}

// 训练后形态导出: runtime 活参数 → legacy 组识体 (raw+compiled 同步) → checkpoint bin
// 前置: 启动期已验证 runtime 参数与 org_master 位级一致 (export_export_trust)
bool export_form(CellularOrganism& org_master, const core::RuntimeState& rt,
                 const std::vector<size_t>& emission_order, const std::string& path) {
    for (const auto& p : rt.parameters()) {
        const auto& b = p.binding;
        if (b.kind == core::ParameterBindingKind::EdgeWeight) {
            const auto* cv = std::get_if<core::ContinuousValue>(&p.value);
            if (!cv || b.edge.value == 0 || b.edge.value > emission_order.size()) return false;
            const size_t ci = emission_order[b.edge.value - 1];
            if (ci >= org_master.compiled_synapses_.size()) return false;
            org_master.compiled_synapses_[ci].weight = cv->value;
            if (ci < org_master.synapses.size()) org_master.synapses[ci].weight = cv->value;
        } else if (b.kind == core::ParameterBindingKind::CellParameter &&
                   b.slot == core::ParameterSlot::Param1) {
            const auto* cv = std::get_if<core::ContinuousValue>(&p.value);
            if (!cv) continue;   // 非连续参数 (如 ChannelIndex) 不可训练, 跳过
            for (auto& c : org_master.cells)
                if (c.id == b.cell.value) { c.param1 = cv->value; break; }
        }
    }
    return org_master.save_checkpoint_bin(path);
}

struct EvalResult {
    int wins = 0;
    long pass = 0, steps = 0;
    std::vector<uint8_t> won;
};

EvalResult eval_family(core::RuntimeState& rt, core::CompiledExecutor& ex,
                       size_t score_idx, int games, uint32_t seed_base) {
    EvalResult out;
    out.won.reserve(games);
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
                    const double sc = rt.cell_states()[score_idx].output_val;
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            int pre = task.cards_left(0);
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            game_won = res.success;
            out.steps++;
            if (task.cards_left(0) == pre) out.pass++;
        }
        out.wins += game_won ? 1 : 0;
        out.won.push_back(game_won ? 1 : 0);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model = "checkpoints/doudizhu_cand_scorer.bin";
    int games = 20000, eval_every = 200, holdout_n = 200;
    double lr = 2e-4;
    bool value_baseline = false;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { printf("[错误] --%s 需要参数\n", what); exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--games")) games = std::atoi(need("games"));
        else if (!std::strcmp(argv[i], "--eval-every")) eval_every = std::atoi(need("eval-every"));
        else if (!std::strcmp(argv[i], "--holdout")) holdout_n = std::atoi(need("holdout"));
        else if (!std::strcmp(argv[i], "--lr")) lr = std::atof(need("lr"));
        else if (!std::strcmp(argv[i], "--baseline")) value_baseline = std::atoi(need("baseline")) != 0;
        else if (!std::strcmp(argv[i], "--model")) model = need("model");
    }
    const uint32_t TRAIN_BASE = 3100000u, HOLD_BASE = 9100000u;

    CellularOrganism org_master;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org_master = std::move(b);
        else org_master = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org_master.cells.empty());

    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 9100;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    auto assembled = kun::migration::assemble_phenotype(
        org_master, core::GraphIdentity(9), core::GraphRevision(1), cfg);
    if (!assembled.ok()) { printf("[错误] 装配: %s\n", assembled.diagnostic.c_str()); return 1; }
    core::RuntimeState& rt = assembled.phenotype->runtime();
    core::CompiledExecutor& ex = assembled.phenotype->executor();

    // 双头定位: score=ACT_CHANNEL channel 0, value=channel 1 (经 plan 参数 ChannelIndex)
    size_t score_idx = (size_t)-1, value_idx = (size_t)-1;
    {
        const auto& plan_cells = rt.plan()->cells();
        for (size_t i = 0; i < plan_cells.size(); ++i) {
            if (plan_cells[i].type != CellType::ACT_CHANNEL) continue;
            const auto& pv = rt.parameters()[plan_cells[i].parameter_indices[1]].value;
            const auto* ch = std::get_if<core::ChannelIndex>(&pv);
            if (!ch) continue;
            if (ch->value == 0) score_idx = i;
            else if (ch->value == 1) value_idx = i;
        }
    }
    if (score_idx == (size_t)-1) { printf("[错误] 找不到分数头\n"); return 1; }
    const bool has_value_head = (value_idx != (size_t)-1);
    if (value_baseline && !has_value_head) { printf("[错误] --baseline 需要价值头\n"); return 1; }
    printf("[P9-ONLINE] 冷迁移: cells=%zu score_idx=%zu value_idx=%zu lr=%.2e baseline=%d\n",
           rt.cell_states().size(), score_idx, value_idx, lr, (int)value_baseline);

    // export 信任链: 发射序重建 + runtime 参数与 legacy 位级一致验证
    auto emission = rebuild_emission_order(org_master);
    if (emission.empty()) { printf("[错误] 发射序重建失败\n"); return 1; }
    {
        long mismatches = 0;
        for (const auto& p : rt.parameters()) {
            const auto& b = p.binding;
            if (b.kind == core::ParameterBindingKind::EdgeWeight) {
                const auto* cv = std::get_if<core::ContinuousValue>(&p.value);
                if (!cv || b.edge.value == 0 || b.edge.value > emission.size()) { mismatches = -1; break; }
                if (std::abs(cv->value - org_master.compiled_synapses_[emission[b.edge.value - 1]].weight) > 1e-9)
                    mismatches++;
                if (emission[b.edge.value - 1] >= org_master.synapses.size() ||
                    std::abs(cv->value - org_master.synapses[emission[b.edge.value - 1]].weight) > 1e-9)
                    mismatches = -1;
            } else if (b.kind == core::ParameterBindingKind::CellParameter &&
                       b.slot == core::ParameterSlot::Param1) {
                const auto* cv = std::get_if<core::ContinuousValue>(&p.value);
                if (!cv) continue;
                bool found = false;
                for (const auto& c : org_master.cells)
                    if (c.id == b.cell.value) { found = true; if (std::abs(cv->value - c.param1) > 1e-9) mismatches++; break; }
                if (!found) mismatches = -1;
            }
        }
        if (mismatches != 0) { printf("[错误] export 信任链失败 (mismatch=%ld)\n", mismatches); return 1; }
        printf("[信任链] runtime↔legacy 参数位级一致 (%zu 边) → export 可用\n", emission.size());
    }

    auto window = core::LearningWindow::open(9100, rt, full_bindings(rt));
    CoreCellularBPTTEngine engine(1024);
    engine.init_optimizer(rt);

    // t=0 holdout 基线 (配对参照 + 漂移参照)
    EvalResult t0 = eval_family(rt, ex, score_idx, holdout_n, HOLD_BASE);
    printf("[t0 基线/holdout] %d 局: %.1f%% (%d/%d) Wilson 下界 %.1f%% | 过牌 %.1f%%\n",
           holdout_n, 100.0 * t0.wins / holdout_n, t0.wins, holdout_n,
           100.0 * wilson_lower(t0.wins, holdout_n),
           100.0 * t0.pass / std::max(1L, t0.steps));

    std::deque<double> outcome_window;   // A 自适应归一化滑窗 (近 256 局)
    int best_wins = -1;
    std::string best_path;
    std::vector<std::string> eval_log;
    bool aborted = false;
    std::string abort_reason;

    long game_records_total = 0;
    for (int g = 0; g < games && !aborted; ++g) {
        kun::DouDiZhuCardGameTask task(40, TRAIN_BASE + (uint32_t)(g * 97), 17.5);
        engine.reset_tape();
        rt.reset_episode();
        std::vector<uint8_t> row_chosen;        // 与录带行严格对齐
        bool done = false;
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
                    auto rec = engine.record_step(rt, ex, in);
                    if (!rec.ok()) { printf("[错误] 录带: %s\n", rec.error->reason.c_str()); return 1; }
                    const double sc = rt.cell_states()[score_idx].output_val;
                    if (!std::isfinite(sc)) { printf("[错误] 非有限分数\n"); return 1; }
                    if (sc > best) { best = sc; pick = i; }
                }
            }
            row_chosen.resize(row_chosen.size() + cands.size(), 0);
            row_chosen[row_chosen.size() - cands.size() + pick] = 1;
            auto res = task.play_candidate(cands[pick]);
            done = res.done;
            game_won = res.success;
        }
        // 局末: 优势自适应归一化 + 双头 targets + 一次 backward/step
        const double outcome = game_won ? 1.0 : -1.0;
        outcome_window.push_back(outcome);
        if ((int)outcome_window.size() > 256) outcome_window.pop_front();
        double mean = 0.0;
        for (double v : outcome_window) mean += v;
        mean /= outcome_window.size();
        double var = 0.0;
        for (double v : outcome_window) var += (v - mean) * (v - mean);
        var /= outcome_window.size();
        const double A = (outcome - mean) / std::max(std::sqrt(var), 0.25);

        std::vector<std::vector<double>> targets;
        targets.reserve(row_chosen.size());
        if (value_baseline) {
            // 逐行优势基线: A_t = A - V(s_t) (V 取自该行价值头输出, tape 逆序可回溯)
            size_t row = 0;
            for (uint8_t chosen : row_chosen) {
                const double v_pred = engine.tape[row].state_post[value_idx].output_val;
                targets.push_back({chosen ? (A - v_pred) : 0.0, game_won ? 1.0 : 0.0});
                ++row;
            }
        } else {
            for (uint8_t chosen : row_chosen)
                targets.push_back({chosen ? A : 0.0, game_won ? 1.0 : 0.0});
        }
        game_records_total += (long)targets.size();
        CoreBPTTGradients grads;
        auto back = engine.backward(rt, targets, grads, &window);
        if (!back.ok()) { printf("[错误] backward: %s\n", back.error->reason.c_str()); return 1; }
        auto upd = engine.step_adam(rt, window, grads, lr);
        if (!upd.ok()) { printf("[错误] adam: %s\n", upd.error->reason.c_str()); return 1; }

        if ((g + 1) % eval_every == 0) {
            EvalResult ev = eval_family(rt, ex, score_idx, holdout_n, HOLD_BASE);
            const double rate = 100.0 * ev.wins / holdout_n;
            const double pass_rate = 100.0 * ev.pass / std::max(1L, ev.steps);
            // 配对 McNemar vs t0 基线 (同种子)
            uint64_t b = 0, c = 0, a = 0, d = 0;
            for (int k = 0; k < holdout_n; ++k) {
                if (ev.won[k] && t0.won[k]) a++;
                else if (ev.won[k]) b++;
                else if (t0.won[k]) c++;
                else d++;
            }
            auto mres = kun::mcnemar_test(b, c, a, d);
            printf("[eval @%d] %.1f%% (%d/%d) Wilson 下界 %.1f%% | Δt0=%+d | 过牌 %.1f%% | McNemar b=%llu c=%llu p=%.3f | loss=%.4f\n",
                   g + 1, rate, ev.wins, holdout_n, 100.0 * wilson_lower(ev.wins, holdout_n),
                   ev.wins - t0.wins, pass_rate,
                   (unsigned long long)b, (unsigned long long)c, mres.p_value, grads.loss);
            {
                std::ostringstream os;
                os << "{\"games\":" << (g + 1) << ",\"wins\":" << ev.wins
                   << ",\"pass_rate\":" << pass_rate << ",\"mcnemar_p\":" << mres.p_value
                   << ",\"loss\":" << grads.loss << "}";
                eval_log.push_back(os.str());
            }
            // 预注册门禁 (勘误版, docs/superpowers/plans/2026-09-09-online-rl-protocol.md §三)
            // 回退门: 同种子配对 t0−8pp 或绝对灾难线 45% (holdout_n≥500)
            if (100.0 * ev.wins / holdout_n < 100.0 * t0.wins / holdout_n - 8.0 ||
                rate < 45.0) {
                aborted = true;
                abort_reason = "回退门: holdout 点估计 < t0−8pp 或 < 45%";
                break;
            }
            if (pass_rate > 51.0 || pass_rate < 45.0) {
                aborted = true; abort_reason = "漂移哨兵: 过牌率偏离 48%±3%"; break;
            }
            if (ev.wins > best_wins) {
                best_wins = ev.wins;
                best_path = "checkpoints/pool/gen" + std::to_string(g + 1) + ".bin";
                if (!export_form(org_master, rt, emission, best_path)) {
                    printf("[错误] 冻结导出失败: %s\n", best_path.c_str()); return 1;
                }
                printf("[冻结] %s (%d/%d = %.1f%%)\n", best_path.c_str(), ev.wins, holdout_n, rate);
            }
        }
    }

    if (aborted)
        printf("[中止] %s → 回滚到最近冻结形态: %s\n", abort_reason.c_str(),
               best_path.empty() ? "(无 — 冠军原形保留)" : best_path.c_str());

    // 终局导出 + 冠军对账
    const std::string final_path = "checkpoints/pool/p9_online_final.bin";
    if (!export_form(org_master, rt, emission, final_path)) {
        printf("[错误] 终局导出失败\n"); return 1;
    }
    printf("[终局] %s | 训练 %d 局 / 录步 %ld | 最佳冻结: %s (%d/%d)\n",
           final_path.c_str(), aborted ? -1 : games, game_records_total,
           best_path.c_str(), best_wins, holdout_n);
    {
        std::ostringstream os;
        os << "md5sum " << final_path << " " << model;
        std::system(os.str().c_str());
    }
    // JSON 归档
    {
        std::ostringstream name;
        name << "runs/p9_online_" << time(nullptr) << ".json";
        std::system("mkdir -p runs");
        std::ofstream f(name.str());
        f << "{\"model\":\"" << model << "\",\"lr\":" << lr
          << ",\"games\":" << games << ",\"eval_every\":" << eval_every
          << ",\"holdout\":" << holdout_n << ",\"baseline_mode\":" << (value_baseline ? 1 : 0)
          << ",\"t0_wins\":" << t0.wins << ",\"aborted\":" << (aborted ? 1 : 0)
          << ",\"abort_reason\":\"" << abort_reason << "\""
          << ",\"best\":\"" << best_path << "\",\"best_wins\":" << best_wins
          << ",\"final\":\"" << final_path << "\""
          << ",\"evals\":[";
        for (size_t i = 0; i < eval_log.size(); ++i)
            f << (i ? "," : "") << eval_log[i];
        f << "]}\n";
        printf("[归档] %s\n", name.str().c_str());
    }
    return aborted ? 2 : 0;
}
