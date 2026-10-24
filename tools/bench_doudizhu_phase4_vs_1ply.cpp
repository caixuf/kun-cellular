// Phase 4 Prefer-A bench: frozen 1-ply cand-scorer vs thin 2-ply maximin (≤10ms).
// Reports: illegal_rate, win/McNemar paired, wall-clock p50/p95, timeout fallback count.
//
// Usage:
//   bench_doudizhu_phase4_vs_1ply [N=200] [seed_base=20261025] [ckpt=checkpoints/doudizhu_cand_scorer.bin] [budget_ms=10]
//
// Policy A = 1-ply argmax (frozen scorer)
// Policy B = 2-ply maximin under budget (timeout → 1-ply)
// Seat-0 agent under test; opponents via task settle heuristic (same for A/B).
#include "tasks/transfer/doudizhu_search_2ply.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace kun;
using namespace kun::doudizhu_search;

namespace {

struct GameOutcome {
    bool won{false};
    bool illegal{false};
    int steps{0};
    int timeouts{0};
    int decisions{0};
    std::vector<double> decision_ms;
};

bool cand_in_legal(const DouDiZhuCardGameTask::CandPlay& play,
                   const std::vector<DouDiZhuCardGameTask::CandPlay>& legal) {
    for (const auto& c : legal) {
        if (c.type == play.type && c.rank == play.rank && c.count == play.count) return true;
    }
    return false;
}

enum class PolicyKind { Ply1, Ply2 };

GameOutcome run_game(ScorerOrg& scorer, uint32_t seed, PolicyKind kind,
                     std::chrono::milliseconds budget) {
    GameOutcome go;
    DouDiZhuCardGameTask task(40, seed, 17.5);
    bool done = false;
    bool won = false;
    int guard = 0;
    const int kMaxSteps = 400;
    while (!done && guard++ < kMaxSteps) {
        auto legal = task.enumerate_candidates(0);
        if (legal.empty()) {
            go.illegal = true;
            break;
        }
        ChooseResult ch;
        if (kind == PolicyKind::Ply1) {
            ch = choose_1ply(task, scorer, 0);
        } else {
            ch = choose_2ply_maximin(task, scorer, 0, budget);
        }
        go.decisions++;
        const double ms = std::chrono::duration<double, std::milli>(ch.elapsed).count();
        go.decision_ms.push_back(ms);
        if (ch.timed_out || ch.used_1ply_fallback) go.timeouts++;

        if (!cand_in_legal(ch.play, legal)) {
            go.illegal = true;
            break;
        }
        auto res = task.play_candidate(ch.play);
        done = res.done;
        won = res.success;
        go.steps = res.steps;
    }
    if (!done && guard >= kMaxSteps) {
        // Treat as loss / truncate; not illegal.
        won = false;
    }
    go.won = won;
    return go;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p * static_cast<double>(v.size() - 1);
    const size_t lo = static_cast<size_t>(idx);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int N = (argc > 1) ? std::atoi(argv[1]) : 200;
    const uint32_t seed_base = (argc > 2) ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 20261025u;
    const char* ckpt = (argc > 3) ? argv[3] : "checkpoints/doudizhu_cand_scorer.bin";
    const int budget_ms = (argc > 4) ? std::atoi(argv[4]) : 10;

    std::printf("=== Phase4 Prefer-A bench: 1-ply vs 2-ply maximin ===\n");
    std::printf("N=%d seed_base=%u ckpt=%s budget_ms=%d\n", N, seed_base, ckpt, budget_ms);

    ScorerOrg scorer;
    if (!scorer.load(ckpt)) {
        std::fprintf(stderr, "FAIL: cannot load scorer %s\n", ckpt);
        return 2;
    }
    std::printf("scorer cells=%zu score_head=%zu\n", scorer.org.cells.size(), scorer.score_head);

    const auto budget = std::chrono::milliseconds(budget_ms);
    std::vector<uint8_t> wins_a(N, 0), wins_b(N, 0);
    int illegal_a = 0, illegal_b = 0;
    int timeouts_b = 0, decisions_b = 0;
    std::vector<double> all_ms_a, all_ms_b;

    for (int i = 0; i < N; ++i) {
        const uint32_t seed = seed_base + static_cast<uint32_t>(i) * 97u;
        auto ga = run_game(scorer, seed, PolicyKind::Ply1, budget);
        auto gb = run_game(scorer, seed, PolicyKind::Ply2, budget);
        wins_a[i] = ga.won ? 1 : 0;
        wins_b[i] = gb.won ? 1 : 0;
        if (ga.illegal) illegal_a++;
        if (gb.illegal) illegal_b++;
        timeouts_b += gb.timeouts;
        decisions_b += gb.decisions;
        all_ms_a.insert(all_ms_a.end(), ga.decision_ms.begin(), ga.decision_ms.end());
        all_ms_b.insert(all_ms_b.end(), gb.decision_ms.begin(), gb.decision_ms.end());
        if ((i + 1) % 25 == 0 || i + 1 == N) {
            std::printf("  progress %d/%d  A_wins=%d B_wins=%d illegal_B=%d timeouts_B=%d\n",
                        i + 1, N,
                        static_cast<int>(std::accumulate(wins_a.begin(), wins_a.begin() + i + 1, 0)),
                        static_cast<int>(std::accumulate(wins_b.begin(), wins_b.begin() + i + 1, 0)),
                        illegal_b, timeouts_b);
        }
    }

    const int wa = std::accumulate(wins_a.begin(), wins_a.end(), 0);
    const int wb = std::accumulate(wins_b.begin(), wins_b.end(), 0);
    const double rate_a = 100.0 * wa / N;
    const double rate_b = 100.0 * wb / N;
    const double illegal_rate_b = 100.0 * illegal_b / std::max(1, decisions_b);
    // Also report per-game illegal fraction for gate spirit
    const double illegal_game_rate_b = 100.0 * illegal_b / N;

    int b01 = 0, b10 = 0, both = 0, neither = 0;
    for (int i = 0; i < N; ++i) {
        if (wins_a[i] && wins_b[i]) both++;
        else if (!wins_a[i] && !wins_b[i]) neither++;
        else if (!wins_a[i] && wins_b[i]) b01++;  // B win, A loss
        else b10++;  // A win, B loss
    }
    // API: b=Awin_Bloss, c=Aloss_Bwin
    McNemarResult mc = mcnemar_test(static_cast<uint64_t>(b10), static_cast<uint64_t>(b01),
                                   static_cast<uint64_t>(both), static_cast<uint64_t>(neither), true);

    const double p50_a = percentile(all_ms_a, 0.50);
    const double p95_a = percentile(all_ms_a, 0.95);
    const double p50_b = percentile(all_ms_b, 0.50);
    const double p95_b = percentile(all_ms_b, 0.95);

    std::printf("\n--- Results ---\n");
    std::printf("Policy A (1-ply): wins=%d/%d (%.2f%%) illegal_games=%d\n", wa, N, rate_a, illegal_a);
    std::printf("Policy B (2-ply): wins=%d/%d (%.2f%%) illegal_games=%d illegal_decisions=%d/%d (%.4f%%)\n",
                wb, N, rate_b, illegal_b, illegal_b, decisions_b, illegal_rate_b);
    std::printf("McNemar B-vs-A: b=Bwin_Aloss=%d c=Awin_Bloss=%d both=%d neither=%d  %s\n",
                b01, b10, both, neither, mc.to_string(4).c_str());
    std::printf("Wall-clock ms: A p50=%.3f p95=%.3f | B p50=%.3f p95=%.3f\n",
                p50_a, p95_a, p50_b, p95_b);
    std::printf("Timeouts (B fallback decisions): %d / %d (%.2f%%)\n",
                timeouts_b, decisions_b,
                100.0 * timeouts_b / std::max(1, decisions_b));

    const bool flat = (wb == wa) || (std::abs(rate_b - rate_a) < 0.5 && !mc.significant_05);
    if (flat) {
        std::printf("DELIVER_OR_REASON: flat/not-worse vs 1-ply — do NOT claim search win.\n");
    } else if (wb > wa && mc.p_value < 0.05) {
        std::printf("SIGNAL: 2-ply ahead of 1-ply on this fixed-seed ladder (see McNemar).\n");
    } else if (wb < wa) {
        std::printf("DELIVER_OR_REASON: 2-ply behind 1-ply on this ladder — search not a win.\n");
    }

    // JSON artifact
    const char* out_path = "artifacts/phase4-2ply/bench_result.json";
    {
        std::system("mkdir -p artifacts/phase4-2ply");
        std::ofstream ofs(out_path);
        if (ofs) {
            ofs << "{\n";
            ofs << "  \"seed_base\": " << seed_base << ",\n";
            ofs << "  \"N\": " << N << ",\n";
            ofs << "  \"budget_ms\": " << budget_ms << ",\n";
            ofs << "  \"ckpt\": \"" << ckpt << "\",\n";
            ofs << "  \"shape\": \"thin_2ply_maximin\",\n";
            ofs << "  \"sign_convention\": \"leaf=role_aware_pm_opp_best_1ply\",\n";
            ofs << "  \"policy_a_wins\": " << wa << ",\n";
            ofs << "  \"policy_b_wins\": " << wb << ",\n";
            ofs << "  \"policy_a_win_rate_pct\": " << rate_a << ",\n";
            ofs << "  \"policy_b_win_rate_pct\": " << rate_b << ",\n";
            ofs << "  \"illegal_a_games\": " << illegal_a << ",\n";
            ofs << "  \"illegal_b_games\": " << illegal_b << ",\n";
            ofs << "  \"illegal_b_decisions\": " << illegal_b << ",\n";
            ofs << "  \"decisions_b\": " << decisions_b << ",\n";
            ofs << "  \"illegal_rate_b_pct\": " << illegal_rate_b << ",\n";
            ofs << "  \"illegal_game_rate_b_pct\": " << illegal_game_rate_b << ",\n";
            ofs << "  \"mcnemar_b\": " << b01 << ",\n";
            ofs << "  \"mcnemar_c\": " << b10 << ",\n";
            ofs << "  \"mcnemar_both\": " << both << ",\n";
            ofs << "  \"mcnemar_neither\": " << neither << ",\n";
            ofs << "  \"mcnemar_p\": " << mc.p_value << ",\n";
            ofs << "  \"mcnemar_significant\": " << (mc.significant_05 ? "true" : "false") << ",\n";
            ofs << "  \"p50_ms_a\": " << p50_a << ",\n";
            ofs << "  \"p95_ms_a\": " << p95_a << ",\n";
            ofs << "  \"p50_ms_b\": " << p50_b << ",\n";
            ofs << "  \"p95_ms_b\": " << p95_b << ",\n";
            ofs << "  \"timeouts_b\": " << timeouts_b << ",\n";
            ofs << "  \"flat_or_reason\": " << (flat || wb <= wa ? "true" : "false") << "\n";
            ofs << "}\n";
            std::printf("Wrote %s\n", out_path);
        }
    }

    // Gate: illegal ≤ 0.1%
    if (illegal_rate_b > 0.1 || illegal_game_rate_b > 0.1) {
        std::fprintf(stderr, "FAIL A4: illegal rate too high\n");
        return 1;
    }
    std::printf("PASS illegal gate (A4)\n");
    return 0;
}
