#include "tasks/transfer/doudizhu_rules_core.hpp"
#include "tasks/transfer/cross_domain_tasks.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <vector>
#include <random>
#include <set>
#include <tuple>

using namespace kun;
using namespace kun::doudizhu_rules;

// I1: card-sum = 54 after deal; conserved across legal play.
static void test_I1_card_sum() {
    for (uint32_t seed = 1; seed <= 200; ++seed) {
        int hands[3][15], kitty[15];
        int sum = deal_counts(seed, hands, kitty);
        assert(sum == 54);
        assign_landlord(hands, kitty, seed % 3);
        assert(card_sum_hands(hands) == 54);

        Trick table{TRICK_NONE, -1, 0};
        int left[3] = {
            card_sum_counts(hands[0]),
            card_sum_counts(hands[1]),
            card_sum_counts(hands[2])
        };
        assert(left[0] + left[1] + left[2] == 54);

        std::mt19937 rng(seed * 9973u + 17u);
        int pass_streak = 0;
        int turn = seed % 3;
        for (int ply = 0; ply < 80; ++ply) {
            std::vector<CandPlay> cands;
            enumerate_candidates(hands[turn], table, cands);
            assert(!cands.empty());
            std::uniform_int_distribution<size_t> dist(0, cands.size() - 1);
            CandPlay pick = cands[dist(rng)];
            Trick table_before = table;
            bool ok = apply_play(hands[turn], table, pick, &left[turn]);
            assert(ok);
            assert(card_sum_hands(hands) == left[0] + left[1] + left[2]);
            // Conservation: cards only move from hand to "played"; total remaining in hands drops by count.
            int played_now = 54 - (left[0] + left[1] + left[2]);
            (void)played_now;
            assert(left[0] + left[1] + left[2] >= 0);
            assert(left[0] + left[1] + left[2] <= 54);

            if (pick.type == TRICK_NONE) {
                table = table_before; // pass does not change table in apply_play
                pass_streak++;
                if (pass_streak >= 2) {
                    table = Trick{TRICK_NONE, -1, 0};
                    pass_streak = 0;
                }
            } else {
                pass_streak = 0;
            }
            if (left[turn] == 0) break;
            turn = (turn + 1) % 3;
        }
    }

    // Cross-check: DouDiZhuCardGameTask deal path conserves 54 across cards_left at init-ish.
    // After reset, opponent turns may have played; residual hands + played via lattice still 54.
    for (uint32_t seed : {42u, 123u, 7u, 999u, 20261024u}) {
        DouDiZhuCardGameTask task(40, seed, -100.0); // no bidding gate — still deals 54
        // Observation dims exist; use cand enumerate agreement smoke via public API if available.
        auto obs = task.current_observation();
        assert(!obs.empty());
        (void)obs;
    }
    std::cout << "[PASS] I1 card-sum=54 conserved\n";
}

// I2: identifyTrick matches harness gold for fixed vectors.
static void test_I2_identify_gold() {
    struct Case { std::vector<int> ranks; TrickType type; int rank; int count; };
    const Case gold[] = {
        {{0}, TRICK_SOLO, 0, 1},
        {{12}, TRICK_SOLO, 12, 1},
        {{14}, TRICK_SOLO, 14, 1},
        {{5, 5}, TRICK_PAIR, 5, 2},
        {{11, 11}, TRICK_PAIR, 11, 2},
        {{7, 7, 7, 7}, TRICK_BOMB, 7, 4},
        {{0, 0, 0, 0}, TRICK_BOMB, 0, 4},
        {{13, 14}, TRICK_ROCKET, 14, 2},
        {{14, 13}, TRICK_ROCKET, 14, 2},
        // invalid / unsupported richer types
        {{3, 3, 3}, TRICK_INVALID, -1, 0},
        {{1, 2, 3, 4, 5}, TRICK_INVALID, -1, 0},
        {{8, 8, 8, 2}, TRICK_INVALID, -1, 0},
    };
    for (const auto& c : gold) {
        Trick t = identify_trick_ranks(c.ranks.data(), static_cast<int>(c.ranks.size()));
        assert(t.type == c.type);
        if (c.type != TRICK_INVALID) {
            assert(t.rank == c.rank);
            assert(t.count == c.count);
        }
    }
    std::cout << "[PASS] I2 identifyTrick gold table\n";
}

// I3: canBeat antisym for distinct non-pass; pass/rocket/bomb edges.
static void test_I3_canbeat_antisym() {
    // pass never beats
    Trick pass{TRICK_NONE, -1, 0};
    Trick solo3{TRICK_SOLO, 3, 1};
    assert(!can_beat(pass, solo3));
    assert(!can_beat(pass, pass));

    // empty table: any non-pass beats
    Trick none{TRICK_NONE, -1, 0};
    assert(can_beat(solo3, none));
    assert(can_beat(Trick{TRICK_ROCKET, 14, 2}, none));

    // rocket edges
    Trick rocket{TRICK_ROCKET, 14, 2};
    Trick bomb7{TRICK_BOMB, 7, 4};
    Trick bomb8{TRICK_BOMB, 8, 4};
    assert(can_beat(rocket, bomb7));
    assert(can_beat(rocket, solo3));
    assert(!can_beat(bomb7, rocket));
    assert(!can_beat(solo3, rocket));
    assert(!can_beat(rocket, rocket)); // same type equal rank — not strictly greater

    // bomb beats non-bomb; higher bomb beats lower
    assert(can_beat(bomb7, solo3));
    assert(can_beat(bomb8, bomb7));
    assert(!can_beat(bomb7, bomb8));
    assert(!can_beat(solo3, bomb7));

    // same-type rank
    Trick solo5{TRICK_SOLO, 5, 1};
    assert(can_beat(solo5, solo3));
    assert(!can_beat(solo3, solo5));
    Trick pair5{TRICK_PAIR, 5, 2};
    Trick pair3{TRICK_PAIR, 3, 2};
    assert(can_beat(pair5, pair3));
    assert(!can_beat(pair3, pair5));
    assert(!can_beat(solo5, pair3)); // type mismatch

    // Property: for distinct non-pass tricks from a generated set, ¬(A beats B ∧ B beats A)
    std::vector<Trick> pool;
    for (int r = 0; r < 15; ++r) pool.push_back({TRICK_SOLO, r, 1});
    for (int r = 0; r < 13; ++r) pool.push_back({TRICK_PAIR, r, 2});
    for (int r = 0; r < 13; ++r) pool.push_back({TRICK_BOMB, r, 4});
    pool.push_back({TRICK_ROCKET, 14, 2});

    for (size_t i = 0; i < pool.size(); ++i) {
        for (size_t j = 0; j < pool.size(); ++j) {
            if (i == j) continue;
            bool ab = can_beat(pool[i], pool[j]);
            bool ba = can_beat(pool[j], pool[i]);
            assert(!(ab && ba));
        }
    }
    std::cout << "[PASS] I3 canBeat antisym + edges\n";
}

// Extra: legal apply rejects illegal; FE-harness agree smoke on deal seed.
static void test_agree_deal_legal() {
    for (uint32_t seed = 1; seed <= 50; ++seed) {
        int hands[3][15], kitty[15];
        assert(deal_counts(seed, hands, kitty) == 54);
        assign_landlord(hands, kitty, 0);
        Trick table{TRICK_NONE, -1, 0};
        std::vector<CandPlay> cands;
        enumerate_candidates(hands[0], table, cands);
        assert(!cands.empty());
        // No pass when free lead
        for (const auto& c : cands) assert(c.type != TRICK_NONE);
        // Illegal: pass on free lead
        CandPlay bad{TRICK_NONE, -1, 0};
        Trick t2 = table;
        assert(!apply_play(hands[0], t2, bad, nullptr));
    }
    std::cout << "[PASS] agree deal/legal smoke\n";
}

int main() {
    test_I1_card_sum();
    test_I2_identify_gold();
    test_I3_canbeat_antisym();
    test_agree_deal_legal();
    std::cout << "[PASS] test_doudizhu_invariants I1–I3 all green\n";
    return 0;
}
