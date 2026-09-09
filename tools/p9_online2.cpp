// P9-ONLINE-V2: 在线轨迹学习 — 优势加权 CE (advantage-weighted listwise CE)
// v1 教训 (runs/p9_online_*.json, honest negative): 选点回归 chosen→±A/其余→0
//   缺局内对比信号, 200 局即单调退化 (53.2%→45.6%@1000, McNemar p<0.001), 回退门触发。
// v2 配方 = 冠军 BC 配方的 RL 化 (train_bc_cand listwise): delta_i = A·(p_i − 1[i=chosen])
//   BC 是 A=+1 特例; A 自适应归一化 (近 256 局 outcome 滑窗); 价值头默认冻结 (vw=0)。
// 门禁与判据: docs/superpowers/plans/2026-09-09-online-rl-protocol.md (预注册)
#include "kun/cellular/cellular_genome.hpp"
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

size_t find_head_by_channel(const CellularOrganism& org, double ch) {
    for (size_t i = 0; i < org.cells.size(); ++i)
        if (org.cells[i].type == CellType::ACT_CHANNEL && org.cells[i].param2 == ch) return i;
    return (size_t)-1;
}

// legacy 44 维打分 (与修复后 eval_cand 完全同法)
void build_scorer_input(const std::vector<float>& obs, const std::vector<float>& cf, std::vector<double>& in) {
    in.assign(56, 0.0);
    for (int d = 0; d < 44; ++d) in[d] = obs[d];
    for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
}

double score_candidate(CellularOrganism& org, size_t head, const std::vector<float>& obs, const std::vector<float>& cf) {
    std::vector<double> in;
    build_scorer_input(obs, cf, in);
    org.reset_state(false);
    org.forward_nd(in.data(), in.size(), false);
    return org.cells[head].output_val;
}

// 优势加权 CE 损失: dL/dout[0] = A·(p_i − 1[i=chosen]) (由全局 delta 注入)
static std::vector<float> g_adv_delta{0.0f, 0.0f};   // [0]=score 头, [1]=价值头 (vw)
static const SubstrateLossFn kAdvLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    g.loss_val = targets.empty() ? 0.0f : targets[0];
    g.dL_dout.assign(preds.size(), 0.0f);
    if (!g_adv_delta.empty()) g.dL_dout[0] = g_adv_delta[0];
    if (g.dL_dout.size() > 1) g.dL_dout[1] = g_adv_delta[1];
    return g;
};

struct EvalResult {
    int wins = 0;
    long pass = 0, steps = 0;
    std::vector<uint8_t> won;
};

EvalResult eval_family(CellularOrganism& org, size_t head, int games, uint32_t seed_base) {
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
                    const double sc = score_candidate(org, head, obs44, cf);
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
    int games = 20000, eval_every = 200, holdout_n = 500;
    double lr = 2e-4, value_w = 0.0;
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { printf("[错误] --%s 需要参数\n", what); exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--games")) games = std::atoi(need("games"));
        else if (!std::strcmp(argv[i], "--eval-every")) eval_every = std::atoi(need("eval-every"));
        else if (!std::strcmp(argv[i], "--holdout")) holdout_n = std::atoi(need("holdout"));
        else if (!std::strcmp(argv[i], "--lr")) lr = std::atof(need("lr"));
        else if (!std::strcmp(argv[i], "--value-w")) value_w = std::atof(need("value-w"));
        else if (!std::strcmp(argv[i], "--model")) model = need("model");
    }
    const uint32_t TRAIN_BASE = 3100000u, HOLD_BASE = 9100000u;

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    const size_t score_head = find_head_by_channel(org, 0.0);
    const size_t value_head = find_head_by_channel(org, 1.0);
    if (score_head == (size_t)-1) { printf("[错误] 找不到分数头\n"); return 1; }
    printf("[P9-ONLINE-V2] cells=%zu score=%zu value=%zu lr=%.2e value_w=%.2e\n",
           org.cells.size(), score_head, value_head, lr, value_w);

    CellularBPTTEngine bptt(64);
    bptt.init_optimizer(org);

    EvalResult t0 = eval_family(org, score_head, holdout_n, HOLD_BASE);
    printf("[t0 基线/holdout] %d 局: %.1f%% (%d/%d) Wilson 下界 %.1f%% | 过牌 %.1f%%\n",
           holdout_n, 100.0 * t0.wins / holdout_n, t0.wins, holdout_n,
           100.0 * wilson_lower(t0.wins, holdout_n), 100.0 * t0.pass / std::max(1L, t0.steps));

    std::deque<double> outcome_window;
    int best_wins = -1;
    std::string best_path;
    std::vector<std::string> eval_log;
    bool aborted = false;
    std::string abort_reason;

    for (int g = 0; g < games && !aborted; ++g) {
        kun::DouDiZhuCardGameTask task(40, TRAIN_BASE + (uint32_t)(g * 97), 17.5);
        bool done = false;
        bool game_won = false;
        // 决策快照: 因果顺序 (动作前快照 obs/候选特征) — DAgger 纪律
        struct Snap { std::vector<float> obs; std::vector<std::vector<float>> feats; size_t chosen; };
        std::vector<Snap> snaps;
        while (!done) {
            auto cands = task.enumerate_candidates(0);
            auto o = task.current_observation();
            std::vector<float> obs44(o.begin(), o.end());
            for (float v : task.seat_context()) obs44.push_back(v);
            Snap s;
            s.obs = obs44;
            s.feats.reserve(cands.size());
            double best = -1e308;
            s.chosen = 0;
            for (size_t i = 0; i < cands.size(); ++i) {
                auto cf = task.candidate_features(cands[i]);
                s.feats.push_back(cf);
                const double sc = score_candidate(org, score_head, obs44, cf);
                if (sc > best) { best = sc; s.chosen = i; }
            }
            snaps.push_back(std::move(s));
            auto res = task.play_candidate(cands[s.chosen]);
            done = res.done;
            game_won = res.success;
        }
        // 局末: A 自适应归一化 + 优势加权 CE 一次更新
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

        BPTTGradients sum, one;
        sum.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
        sum.grad_gains.assign(org.cells.size(), 0.0f);
        for (const auto& s : snaps) {
            // softmax (log-sum-exp 稳定)
            std::vector<double> scores(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i)
                scores[i] = score_candidate(org, score_head, s.obs, s.feats[i]);
            const double mx = *std::max_element(scores.begin(), scores.end());
            double Z = 0.0;
            for (double v : scores) Z += std::exp(v - mx);
            std::vector<double> p(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i) p[i] = std::exp(scores[i] - mx) / Z;
            const double ce = -std::log(std::max(p[s.chosen], 1e-9));
            // 逐候选带反传 (权重未变, 前向确定性)
            for (size_t i = 0; i < s.feats.size(); ++i) {
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[i], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_adv_delta[0] = (float)(A * (p[i] - ((int)i == (int)s.chosen ? 1.0 : 0.0)));
                g_adv_delta[1] = 0.0f;   // v2 默认价值头冻结 (--value-w 预留)
                std::vector<std::vector<float>> tgts = {{(float)ce}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        if (value_w != 0.0 && value_head != (size_t)-1) {
            // 价值头 MC 回报回归 (经独立损益注入, 默认关)
            g_adv_delta[0] = 0.0f;
            for (const auto& s : snaps) {
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[s.chosen], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                const float v_pred = org.cells[value_head].output_val;
                g_adv_delta[1] = value_w * (v_pred - (game_won ? 1.0f : 0.0f));
                std::vector<std::vector<float>> tgts = {{0.0f}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        bptt.step_adam(org, sum, (float)lr);

        if ((g + 1) % eval_every == 0) {
            EvalResult ev = eval_family(org, score_head, holdout_n, HOLD_BASE);
            const double rate = 100.0 * ev.wins / holdout_n;
            const double pass_rate = 100.0 * ev.pass / std::max(1L, ev.steps);
            uint64_t b = 0, c = 0, a = 0, d = 0;
            for (int k = 0; k < holdout_n; ++k) {
                if (ev.won[k] && t0.won[k]) a++;
                else if (ev.won[k]) b++;
                else if (t0.won[k]) c++;
                else d++;
            }
            auto mres = kun::mcnemar_test(b, c, a, d);
            printf("[eval @%d] %.1f%% (%d/%d) Wilson 下界 %.1f%% | Δt0=%+d | 过牌 %.1f%% | McNemar b=%llu c=%llu p=%.3f\n",
                   g + 1, rate, ev.wins, holdout_n, 100.0 * wilson_lower(ev.wins, holdout_n),
                   ev.wins - t0.wins, pass_rate,
                   (unsigned long long)b, (unsigned long long)c, mres.p_value);
            {
                std::ostringstream os;
                os << "{\"games\":" << (g + 1) << ",\"wins\":" << ev.wins
                   << ",\"pass_rate\":" << pass_rate << ",\"mcnemar_p\":" << mres.p_value << "}";
                eval_log.push_back(os.str());
            }
            // 预注册门禁 (勘误版): 配对 t0−8pp 或绝对灾难 45%; 漂移哨兵 48%±3%
            if (rate < 100.0 * t0.wins / holdout_n - 8.0 || rate < 45.0) {
                aborted = true;
                abort_reason = "回退门: holdout 点估计 < t0−8pp 或 < 45%";
                break;
            }
            if (pass_rate > 51.0 || pass_rate < 45.0) {
                aborted = true; abort_reason = "漂移哨兵: 过牌率偏离 48%±3%"; break;
            }
            if (ev.wins > best_wins) {
                best_wins = ev.wins;
                best_path = "checkpoints/pool/v2_gen" + std::to_string(g + 1) + ".bin";
                if (!org.save_checkpoint_bin(best_path)) {
                    printf("[错误] 冻结导出失败: %s\n", best_path.c_str()); return 1;
                }
                printf("[冻结] %s (%d/%d = %.1f%%)\n", best_path.c_str(), ev.wins, holdout_n, rate);
            }
        }
    }

    if (aborted)
        printf("[中止] %s → 回滚到最近冻结形态: %s\n", abort_reason.c_str(),
               best_path.empty() ? "(无 — 冠军原形保留)" : best_path.c_str());
    const std::string final_path = "checkpoints/pool/p9_online_v2_final.bin";
    if (!org.save_checkpoint_bin(final_path)) { printf("[错误] 终局导出失败\n"); return 1; }
    printf("[终局] %s | 最佳冻结: %s (%d/%d)\n", final_path.c_str(), best_path.c_str(), best_wins, holdout_n);
    {
        std::ostringstream os;
        os << "md5sum " << final_path << " " << model;
        std::system(os.str().c_str());
    }
    {
        std::ostringstream name;
        name << "runs/p9_online_v2_" << time(nullptr) << ".json";
        std::system("mkdir -p runs");
        std::ofstream f(name.str());
        f << "{\"model\":\"" << model << "\",\"lr\":" << lr << ",\"value_w\":" << value_w
          << ",\"games\":" << games << ",\"eval_every\":" << eval_every
          << ",\"holdout\":" << holdout_n
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
