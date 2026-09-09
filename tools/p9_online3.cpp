// P9-ONLINE-3 (M2): 自博弈对手池 + 课程 — 优势加权 CE (v2 配方不变)
// 协议: docs/superpowers/plans/2026-09-09-online-rl-protocol.md §二b (预注册)
//   - 起点 = 冠军 3c0f1546 (干净归因)
//   - 池: 每 500 局冻结当前形态 → checkpoints/pool/m2_gen<N>.bin
//   - 课程: r = min(0.5, 0.05·⌊g/1000⌋); 每局座位 1/2 独立掷币: r 概率池快照驱动,
//     否则教师启发式; 池空全教师
//   - seat 0 观测走 current_observation+seat_context (与评测逐位一致)
//   - 评测/门禁/判据同 M1 (holdout 9100000 族, McNemar 主判据)
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

void build_scorer_input(const std::vector<float>& obs, const std::vector<float>& cf, std::vector<double>& in) {
    in.assign(56, 0.0);
    for (int d = 0; d < 44; ++d) in[d] = obs[d];
    for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
}

double score_org(CellularOrganism& org, size_t head, const std::vector<float>& obs, const std::vector<float>& cf) {
    std::vector<double> in;
    build_scorer_input(obs, cf, in);
    org.reset_state(false);
    org.forward_nd(in.data(), in.size(), false);
    return org.cells[head].output_val;
}

static std::vector<float> g_adv_delta{0.0f, 0.0f};
static const SubstrateLossFn kAdvLoss = [](const std::vector<float>& preds, const std::vector<float>& targets) -> SubstrateLossGrad {
    SubstrateLossGrad g;
    g.loss_val = targets.empty() ? 0.0f : targets[0];
    g.dL_dout.assign(preds.size(), 0.0f);
    if (!g_adv_delta.empty()) g.dL_dout[0] = g_adv_delta[0];
    if (g.dL_dout.size() > 1) g.dL_dout[1] = g_adv_delta[1];
    return g;
};

// 模型选候选 (池对手): 座位参数化观测 + 特征
size_t model_pick(CellularOrganism& org, size_t head, const DouDiZhuCardGameTask& task, int p,
                  const std::vector<DouDiZhuCardGameTask::CandPlay>& cands) {
    auto obs = task.observation44_for(p);
    size_t pick = 0;
    double best = -1e308;
    for (size_t i = 0; i < cands.size(); ++i) {
        auto cf = task.candidate_features_for(p, cands[i]);
        const double sc = score_org(org, head, obs, cf);
        if (sc > best) { best = sc; pick = i; }
    }
    return pick;
}

struct EvalResult {
    int wins = 0;
    long pass = 0, steps = 0;
    std::vector<uint8_t> won;
};

// 评测协议 (与 v2 eval_family 完全一致: seat 0 模型, 1/2 教师)
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
                    const double sc = score_org(org, head, obs44, cf);
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
    int games = 20000, eval_every = 200, holdout_n = 500, pool_every = 500;
    double lr = 2e-4, r_max = 0.5, r_start = -1.0, value_w = 0.0;
    int r_period = 1000;
    const char* pool_list = "";
    const char* final_out = "checkpoints/pool/p9_online_m2_final.bin";
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { printf("[错误] --%s 需要参数\n", what); exit(1); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--games")) games = std::atoi(need("games"));
        else if (!std::strcmp(argv[i], "--eval-every")) eval_every = std::atoi(need("eval-every"));
        else if (!std::strcmp(argv[i], "--holdout")) holdout_n = std::atoi(need("holdout"));
        else if (!std::strcmp(argv[i], "--pool-every")) pool_every = std::atoi(need("pool-every"));
        else if (!std::strcmp(argv[i], "--lr")) lr = std::atof(need("lr"));
        else if (!std::strcmp(argv[i], "--r-max")) r_max = std::atof(need("r-max"));
        else if (!std::strcmp(argv[i], "--r-start")) r_start = std::atof(need("r-start"));
        else if (!std::strcmp(argv[i], "--r-period")) r_period = std::atoi(need("r-period"));
        else if (!std::strcmp(argv[i], "--value-w")) value_w = std::atof(need("value-w"));
        else if (!std::strcmp(argv[i], "--pool-list")) pool_list = need("pool-list");
        else if (!std::strcmp(argv[i], "--final-out")) final_out = need("final-out");
        else if (!std::strcmp(argv[i], "--model")) model = need("model");
    }
    std::string final_path = final_out;
    const uint32_t TRAIN_BASE = 3100000u, HOLD_BASE = 9100000u;

    CellularOrganism org;
    {
        auto b = CellularOrganism::load_checkpoint_bin(model);
        if (!b.cells.empty()) org = std::move(b);
        else org = CellularOrganism::load_checkpoint_json(model);
    }
    assert(!org.cells.empty());
    const size_t score_head = find_head_by_channel(org, 0.0);
    if (score_head == (size_t)-1) { printf("[错误] 找不到分数头\n"); return 1; }
    const size_t value_head = find_head_by_channel(org, 1.0);
    if (value_w > 0.0 && value_head == (size_t)-1) { printf("[错误] --value-w 需要价值头\n"); return 1; }
    printf("[M2] 起点 cells=%zu lr=%.2e r_max=%.2f\n", org.cells.size(), lr, r_max);

    CellularBPTTEngine bptt(64);
    bptt.init_optimizer(org);

    EvalResult t0 = eval_family(org, score_head, holdout_n, HOLD_BASE);
    printf("[t0 基线/holdout] %d 局: %.1f%% (%d/%d) Wilson 下界 %.1f%% | 过牌 %.1f%%\n",
           holdout_n, 100.0 * t0.wins / holdout_n, t0.wins, holdout_n,
           100.0 * wilson_lower(t0.wins, holdout_n), 100.0 * t0.pass / std::max(1L, t0.steps));

    // 对手池 (M2c: --pool-list 预载快照, 保持课程连续性)
    std::vector<CellularOrganism> pool;
    {
        std::string list(pool_list);
        size_t pos = 0;
        while (!list.empty()) {
            const size_t comma = list.find(',');
            const std::string item = (comma == std::string::npos) ? list : list.substr(0, comma);
            if (comma == std::string::npos) list.clear(); else list.erase(0, comma + 1);
            if (item.empty()) continue;
            CellularOrganism loaded;
            auto bl = CellularOrganism::load_checkpoint_bin(item.c_str());
            if (!bl.cells.empty()) loaded = std::move(bl);
            else loaded = CellularOrganism::load_checkpoint_json(item.c_str());
            if (loaded.cells.empty()) { printf("[错误] 池加载失败: %s\n", item.c_str()); return 1; }
            pool.push_back(std::move(loaded));
            printf("[池预载] %s (池=%zu)\n", item.c_str(), pool.size());
        }
    }
    uint32_t lcg = 20260909u;
    auto rnd01 = [&]() {
        lcg = lcg * 1664525u + 1013904223u;
        return (double)(lcg >> 8) / 16777216.0;
    };

    std::deque<double> outcome_window;
    int best_wins = -1;
    std::string best_path;
    std::vector<std::string> eval_log;
    bool aborted = false;
    std::string abort_reason;
    long selfplay_decisions = 0, teacher_decisions = 0, student_decisions = 0;

    for (int g = 0; g < games && !aborted; ++g) {
        const double r = (r_start >= 0.0)
            ? std::min(r_max, r_start)
            : std::min(r_max, 0.05 * (double)(g / r_period));
        kun::DouDiZhuCardGameTask task(40, TRAIN_BASE + (uint32_t)(g * 97), 17.5);
        // 座位课程: 1/2 独立掷币 (r 概率池对手; 池空全教师)
        int model_seats = 0;
        if (!pool.empty()) {
            if (rnd01() < r) model_seats |= 1;
            if (rnd01() < r) model_seats |= 2;
        }
        size_t opp_head = (size_t)-1;
        CellularOrganism* opp = nullptr;
        if (model_seats) {
            opp = &pool[(size_t)(rnd01() * (double)pool.size()) % pool.size()];
            opp_head = find_head_by_channel(*opp, 0.0);
        }

        // 自博弈驱动: 全座位外部驱动 (seat 0 = 学生, 模型座位 = 池对手, 其余教师启发式)
        struct Snap { std::vector<float> obs; std::vector<std::vector<float>> feats; size_t chosen; };
        std::vector<Snap> snaps;
        bool done = false;
        bool game_won = false;
        int guard = 0;
        while (!done && guard++ < 400) {
            const int p = task.current_turn();
            if (p == 0) {
                auto cands = task.enumerate_candidates(0);
                auto o = task.current_observation();
                std::vector<float> obs44(o.begin(), o.end());
                for (float v : task.seat_context()) obs44.push_back(v);
                Snap s;
                s.obs = obs44;
                double best = -1e308;
                s.chosen = 0;
                for (size_t i = 0; i < cands.size(); ++i) {
                    auto cf = task.candidate_features(cands[i]);
                    s.feats.push_back(cf);
                    const double sc = score_org(org, score_head, obs44, cf);
                    if (sc > best) { best = sc; s.chosen = i; }
                }
                snaps.push_back(std::move(s));
                task.play_candidate_at(0, cands[s.chosen]);
                if (task.cards_left(0) <= 0) {
                    done = true;
                    game_won = (task.landlord() == 0);   // 座 0 出完: 地主则胜, 农民则友方胜
                }
                student_decisions++;
            } else {
                auto cands = task.enumerate_candidates(p);
                if (cands.empty()) { printf("[错误] 空候选 @座%d\n", p); return 1; }
                if (opp && (model_seats & (1 << p))) {
                    const size_t pick = model_pick(*opp, opp_head, task, p, cands);
                    task.play_candidate_at(p, cands[pick]);
                    selfplay_decisions++;
                } else {
                    task.play_opponent_turn(p);   // 教师启发式执行器 (与评测协议同一执行器)
                    teacher_decisions++;
                }
                if (task.cards_left(p) <= 0) {
                    done = true;
                    game_won = (p != task.landlord());   // p≠0 出完且为农民 → 学生友方胜
                }
            }
        }
        if (!done) game_won = false;   // 轮转守卫兜底 (语义同 max_rounds 判负)

        // 优势加权 CE 一次更新 (M3: --value-w > 0 时启用价值基线 A_t = A − V(s_t))
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
        const double value_target = game_won ? 1.0 : 0.0;

        BPTTGradients sum, one;
        sum.grad_synapses.assign(org.compiled_synapses_.size(), 0.0f);
        sum.grad_gains.assign(org.cells.size(), 0.0f);
        // 价值基线 (M3, 预注册): V_t 取当前权重的选中行输出, 优势 A_t = A − V_t
        std::vector<double> snap_v(snaps.size(), 0.0);
        if (value_w > 0.0) {
            for (size_t si = 0; si < snaps.size(); ++si) {
                const auto& s = snaps[si];
                snap_v[si] = score_org(org, value_head, s.obs, s.feats[s.chosen]);
                // 价值头 MC 回报回归: delta_v = 2·value_w·(V − won)
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[s.chosen], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_adv_delta[0] = 0.0f;
                g_adv_delta[1] = (float)(2.0 * value_w * (snap_v[si] - value_target));
                std::vector<std::vector<float>> tgts = {{0.0f}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        size_t snap_idx = 0;
        for (const auto& s : snaps) {
            const double A_t = A - snap_v[snap_idx];
            ++snap_idx;
            std::vector<double> scores(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i)
                scores[i] = score_org(org, score_head, s.obs, s.feats[i]);
            const double mx = *std::max_element(scores.begin(), scores.end());
            double Z = 0.0;
            for (double v : scores) Z += std::exp(v - mx);
            std::vector<double> p(s.feats.size());
            for (size_t i = 0; i < s.feats.size(); ++i) p[i] = std::exp(scores[i] - mx) / Z;
            const double ce = -std::log(std::max(p[s.chosen], 1e-9));
            for (size_t i = 0; i < s.feats.size(); ++i) {
                std::vector<double> in;
                build_scorer_input(s.obs, s.feats[i], in);
                org.reset_state(false);
                bptt.reset_tape();
                org.forward_nd(in.data(), in.size(), false);
                bptt.record_step(org);
                g_adv_delta[0] = (float)(A_t * (p[i] - ((int)i == (int)s.chosen ? 1.0 : 0.0)));
                g_adv_delta[1] = 0.0f;
                std::vector<std::vector<float>> tgts = {{(float)ce}};
                bptt.backward_with_loss(org, tgts, one, SubstrateLossType::MSE, kAdvLoss);
                for (size_t j = 0; j < sum.grad_synapses.size(); ++j) sum.grad_synapses[j] += one.grad_synapses[j];
                for (size_t j = 0; j < sum.grad_gains.size(); ++j) sum.grad_gains[j] += one.grad_gains[j];
            }
        }
        bptt.step_adam(org, sum, (float)lr);

        // 池冻结 (每 pool_every 局)
        if ((g + 1) % pool_every == 0) {
            const std::string path = "checkpoints/pool/m2_gen" + std::to_string(g + 1) + ".bin";
            if (!org.save_checkpoint_bin(path)) { printf("[错误] 池冻结失败\n"); return 1; }
            CellularOrganism loaded;
            auto bl = CellularOrganism::load_checkpoint_bin(path.c_str());
            if (!bl.cells.empty()) loaded = std::move(bl);
            else loaded = CellularOrganism::load_checkpoint_json(path.c_str());
            pool.push_back(std::move(loaded));
            printf("[池] %s (池=%zu, r=%.2f)\n", path.c_str(), pool.size(), r);
        }

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
            // 哨兵修正案四 (M2c 复跑, 预注册): 漂移 = 相对自身 t0 的变化, 非绝对水平
            if (rate < 100.0 * t0.wins / holdout_n - 8.0 || rate < 45.0) {
                aborted = true; abort_reason = "回退门: holdout 点估计 < t0−8pp 或 < 45%"; break;
            }
            if (std::abs(pass_rate - 100.0 * t0.pass / std::max(1L, t0.steps)) > 5.0) {
                aborted = true; abort_reason = "漂移哨兵: 过牌率相对自身 t0 漂移 > 5pp"; break;
            }
            if (ev.wins > best_wins) {
                best_wins = ev.wins;
                best_path = "checkpoints/pool/m2_best" + std::to_string(g + 1) + ".bin";
                if (!org.save_checkpoint_bin(best_path)) {
                    printf("[错误] 冻结导出失败\n"); return 1;
                }
                printf("[冻结] %s (%d/%d = %.1f%%)\n", best_path.c_str(), ev.wins, holdout_n, rate);
            }
        }
    }

    if (aborted)
        printf("[中止] %s → 回滚: %s\n", abort_reason.c_str(),
               best_path.empty() ? "(无 — 冠军原形保留)" : best_path.c_str());
    if (!org.save_checkpoint_bin(final_path)) { printf("[错误] 终局导出失败\n"); return 1; }
    printf("[终局] %s | 最佳: %s (%d/%d) | 自博弈 %ld / 教师 %ld / 学生 %ld\n",
           final_path.c_str(), best_path.c_str(), best_wins, holdout_n,
           selfplay_decisions, teacher_decisions, student_decisions);
    {
        std::ostringstream os;
        os << "md5sum " << final_path;
        std::system(os.str().c_str());
    }
    {
        std::ostringstream name;
        name << "runs/p9_online_m2_" << time(nullptr) << ".json";
        std::system("mkdir -p runs");
        std::ofstream f(name.str());
        f << "{\"model\":\"" << model << "\",\"lr\":" << lr << ",\"r_max\":" << r_max
          << ",\"games\":" << games << ",\"eval_every\":" << eval_every
          << ",\"holdout\":" << holdout_n
          << ",\"t0_wins\":" << t0.wins << ",\"aborted\":" << (aborted ? 1 : 0)
          << ",\"abort_reason\":\"" << abort_reason << "\""
          << ",\"best\":\"" << best_path << "\",\"best_wins\":" << best_wins
          << ",\"final\":\"" << final_path << "\""
          << ",\"selfplay_decisions\":" << selfplay_decisions
          << ",\"teacher_decisions\":" << teacher_decisions
          << ",\"student_decisions\":" << student_decisions
          << ",\"evals\":[";
        for (size_t i = 0; i < eval_log.size(); ++i)
            f << (i ? "," : "") << eval_log[i];
        f << "]}\n";
        printf("[归档] %s\n", name.str().c_str());
    }
    return aborted ? 2 : 0;
}
