#pragma once
// ============================================================================
// Phase 4 Prefer-A — thin 2-ply maximin over legal CandPlay (task-layer only).
//
// Shape (documented Prefer-A, NOT deep PUCT / DouZero):
//   1. Root expand: DouDiZhuCardGameTask::enumerate_candidates(seat)
//   2. Score each root cand with frozen 1-ply cand-scorer (56-d → score head)
//   3. For each root cand: copy state → play_candidate_at → opponent 1-ply argmax
//      mirror → leaf = ±opponent_best_score  (role-aware sign; see below)
//   4. Choose root cand maximizing leaf (maximin)
//   5. Hard wall-clock budget (default 10ms). On timeout / budget exceed →
//      IMMEDIATE deterministic 1-ply argmax. Never return illegal play.
//
// Sign convention (leaf) — role-aware Prefer-A:
//   After apply(R), opp replies 1-ply argmax with score S.
//   leaf(R) = -S if opp is adversary (landlord↔peasant); +S if teammate.
//   Immediate win for root seat  → +1e9
//   Immediate loss (hostile empty) / teammate empty → ±1e9 as coded
//   If apply(R) returns turn to root without an opp reply → leaf = max 1-ply
//   score for root at the resulting position.
//
// Layer: tasks/transfer + tools only. NEVER include/kun/cellular/**.
// ============================================================================

#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace kun {
namespace doudizhu_search {

struct ScorerOrg {
    CellularOrganism org;
    size_t score_head{0};
    bool loaded{false};

    bool load(const std::string& path) {
        auto b = CellularOrganism::load_checkpoint_bin(path);
        if (!b.cells.empty()) {
            org = std::move(b);
        } else {
            org = CellularOrganism::load_checkpoint_json(path);
        }
        if (org.cells.empty()) {
            loaded = false;
            return false;
        }
        locate_heads();
        loaded = true;
        return true;
    }

    void locate_heads() {
        score_head = 0;
        for (size_t i = 0; i < org.cells.size(); ++i) {
            const auto& c = org.cells[i];
            if (c.type == CellType::ACT_CHANNEL && c.param2 == 0.0) {
                score_head = i;
                return;
            }
        }
        // Fallback: last ACT_CHANNEL
        for (size_t i = 0; i < org.cells.size(); ++i) {
            if (org.cells[i].type == CellType::ACT_CHANNEL) score_head = i;
        }
    }

    // Score one CandPlay for seat p using observation44_for (44-d) + 12 cand feats.
    // Same protocol as tools/doudizhu_c_runtime.cpp score_all_candidates (seat-0 path
    // is bit-compatible via observation44_for(0) ≡ current_observation+seat_context).
    double score_candidate(const DouDiZhuCardGameTask& task, int seat,
                           const DouDiZhuCardGameTask::CandPlay& c) {
        auto obs44 = task.observation44_for(seat);
        auto cf = task.candidate_features_for(seat, c);
        std::vector<double> in(56, 0.0);
        const int nobs = std::min(44, static_cast<int>(obs44.size()));
        for (int d = 0; d < nobs; ++d) in[d] = obs44[d];
        for (int d = 0; d < 12 && d < static_cast<int>(cf.size()); ++d) in[44 + d] = cf[d];
        org.reset_state(false);
        org.forward_nd(in.data(), in.size(), false);
        return org.cells[score_head].output_val;
    }

    // Deterministic 1-ply argmax. On empty legal set returns 0 (caller must guard).
    size_t argmax_1ply(const DouDiZhuCardGameTask& task, int seat,
                       const std::vector<DouDiZhuCardGameTask::CandPlay>& cands,
                       double* out_best = nullptr) {
        size_t pick = 0;
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < cands.size(); ++i) {
            const double sc = score_candidate(task, seat, cands[i]);
            if (sc > best) {
                best = sc;
                pick = i;
            }
        }
        if (out_best) *out_best = best;
        return pick;
    }
};

struct ChooseResult {
    DouDiZhuCardGameTask::CandPlay play{0, -1, 0};
    bool timed_out{false};           // true → returned 1-ply fallback
    bool used_1ply_fallback{false};  // same as timed_out for Prefer-A
    bool completed_2ply{false};
    int root_cands{0};
    int root_expanded{0};
    double best_leaf{-std::numeric_limits<double>::infinity()};
    double best_1ply_score{-std::numeric_limits<double>::infinity()};
    std::chrono::nanoseconds elapsed{0};
};

inline bool budget_exceeded(std::chrono::steady_clock::time_point deadline) {
    return std::chrono::steady_clock::now() >= deadline;
}

// Thin 2-ply maximin choose under hard wall-clock budget.
// `seat` is the to-move root seat (Prefer-A bench uses 0).
// On ANY timeout mid-expand → immediate 1-ply (not partial maximin) for determinism.
inline ChooseResult choose_2ply_maximin(
    const DouDiZhuCardGameTask& task,
    ScorerOrg& scorer,
    int seat = 0,
    std::chrono::milliseconds budget = std::chrono::milliseconds(10)) {
    ChooseResult out;
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + budget;

    auto cands = task.enumerate_candidates(seat);
    out.root_cands = static_cast<int>(cands.size());
    if (cands.empty()) {
        // Must not happen with legal oracle; emit pass sentinel.
        out.play = DouDiZhuCardGameTask::CandPlay{0, -1, 0};
        out.timed_out = true;
        out.used_1ply_fallback = true;
        out.elapsed = std::chrono::steady_clock::now() - t0;
        return out;
    }

    // Always compute 1-ply first (fallback + determinism anchor).
    const size_t ply1 = scorer.argmax_1ply(task, seat, cands, &out.best_1ply_score);
    out.play = cands[ply1];

    if (budget_exceeded(deadline)) {
        out.timed_out = true;
        out.used_1ply_fallback = true;
        out.elapsed = std::chrono::steady_clock::now() - t0;
        return out;
    }

    double best_leaf = -std::numeric_limits<double>::infinity();
    size_t best_idx = ply1;
    bool aborted = false;

    for (size_t i = 0; i < cands.size(); ++i) {
        if (budget_exceeded(deadline)) {
            aborted = true;
            break;
        }

        // Copy full task state; apply root cand without settle/heuristic rotation.
        DouDiZhuCardGameTask child = task;
        child.play_candidate_at(seat, cands[i]);
        out.root_expanded++;

        double leaf = 0.0;
        if (child.cards_left(seat) <= 0) {
            leaf = 1e9;  // root empties → win
        } else {
            const int opp = child.current_turn();
            if (opp == seat) {
                // Lead returned to root (e.g. trick cleared) — leaf = our best 1-ply.
                auto next = child.enumerate_candidates(seat);
                if (next.empty()) {
                    leaf = 0.0;
                } else {
                    double best = 0.0;
                    scorer.argmax_1ply(child, seat, next, &best);
                    leaf = best;
                }
            } else {
                auto opp_cands = child.enumerate_candidates(opp);
                if (opp_cands.empty()) {
                    leaf = 0.0;
                } else {
                    double opp_best = 0.0;
                    const size_t oi = scorer.argmax_1ply(child, opp, opp_cands, &opp_best);
                    child.play_candidate_at(opp, opp_cands[oi]);
                    if (child.cards_left(opp) <= 0) {
                        // Opp emptied. Landlord emptying vs peasant seat → loss;
                        // teammate emptying (both peasants) → win for seat.
                        const bool opp_is_landlord = (opp == child.landlord());
                        const bool seat_is_landlord = (seat == child.landlord());
                        if (seat_is_landlord) {
                            leaf = -1e9;  // any other seat emptying beats landlord
                        } else if (opp_is_landlord) {
                            leaf = -1e9;  // landlord emptied → peasants lose
                        } else {
                            leaf = 1e9;   // teammate emptied → peasants win
                        }
                    } else {
                        // Prefer-A sign convention (role-aware):
                        //   adversary (landlord↔peasant): leaf = -opp_best
                        //   teammate (peasant↔peasant):   leaf = +opp_best
                        // Documented: pure adversarial maximin hurt peasant co-op on N=200.
                        const bool seat_ll = (seat == child.landlord());
                        const bool opp_ll = (opp == child.landlord());
                        const bool adversarial = seat_ll || opp_ll;
                        leaf = adversarial ? -opp_best : opp_best;
                    }
                }
            }
        }

        if (leaf > best_leaf) {
            best_leaf = leaf;
            best_idx = i;
        }
    }

    out.best_leaf = best_leaf;
    out.elapsed = std::chrono::steady_clock::now() - t0;

    if (aborted || budget_exceeded(deadline)) {
        // Prefer-A: timeout → IMMEDIATE 1-ply (not partial maximin).
        out.play = cands[ply1];
        out.timed_out = true;
        out.used_1ply_fallback = true;
        out.completed_2ply = false;
    } else {
        out.play = cands[best_idx];
        out.timed_out = false;
        out.used_1ply_fallback = false;
        out.completed_2ply = true;
    }
    return out;
}

// Convenience: frozen 1-ply only (Policy A).
inline ChooseResult choose_1ply(
    const DouDiZhuCardGameTask& task,
    ScorerOrg& scorer,
    int seat = 0) {
    ChooseResult out;
    const auto t0 = std::chrono::steady_clock::now();
    auto cands = task.enumerate_candidates(seat);
    out.root_cands = static_cast<int>(cands.size());
    if (cands.empty()) {
        out.play = DouDiZhuCardGameTask::CandPlay{0, -1, 0};
        out.used_1ply_fallback = true;
        out.elapsed = std::chrono::steady_clock::now() - t0;
        return out;
    }
    const size_t pick = scorer.argmax_1ply(task, seat, cands, &out.best_1ply_score);
    out.play = cands[pick];
    out.completed_2ply = false;
    out.elapsed = std::chrono::steady_clock::now() - t0;
    return out;
}

}  // namespace doudizhu_search
}  // namespace kun
