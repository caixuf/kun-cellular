#pragma once
// DouDiZhu rules core — single source aligned with DouDiZhuCardGameTask CandPlay / trick semantics.
// Trick surface (harness / task): NONE(pass), SOLO, PAIR, BOMB, ROCKET.
// Used by: native I1–I3 tests, emscripten WASM exports, FE glue (no JS rules oracle).
//
// =============================================================================
// CONTRACT v1 — eval harness frozen ops (Phase 1 Correctness lock)
// =============================================================================
// Breaking change to any op below requires Prefer bounce + rails amend — no silent reshape.
// Tests: tests/test_doudizhu_invariants.cpp (I1–I3 + contract freeze). CI red on drift.
//
// | Op      | Symbol(s)                                              | Meaning / freeze                                      |
// |---------|--------------------------------------------------------|-------------------------------------------------------|
// | deal    | deal_counts(seed, hands, kitty); assign_landlord(...)  | Seed → 3×17 + kitty3; card-sum == 54 conserved        |
// | play    | apply_play(hand, table, cand, cards_left*)             | Apply cand; reject illegal; update table / counts     |
// | legal   | enumerate_candidates / enumerate_candidates_n          | Legal CandPlay set for seat hand + table trick        |
// | score   | terminal_score(agent_won)                              | Episode chips: +200 win / −200 loss (eval harness)    |
//
// TrickType numeric values are part of CONTRACT v1 (WASM / FE bit-agree):
//   NONE=0, SOLO=1, PAIR=2, BOMB=3, ROCKET=4, INVALID=-1
// Subset SOLO–PAIR–BOMB–ROCKET vs fuller Chinese DDZ is NON-BLOCK (Prefer #1).
// =============================================================================
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <random>

namespace kun {
namespace doudizhu_rules {

// CONTRACT v1 semver marker — bump only with Prefer bounce.
inline constexpr int kContractMajor = 1;
inline constexpr int kContractMinor = 0;
inline constexpr const char* kContractId = "doudizhu-rules-CONTRACT-v1";

enum TrickType : int32_t {
    TRICK_NONE = 0,
    TRICK_SOLO = 1,
    TRICK_PAIR = 2,
    TRICK_BOMB = 3,
    TRICK_ROCKET = 4,
    TRICK_INVALID = -1
};

struct Trick {
    TrickType type{TRICK_NONE};
    int32_t rank{-1};
    int32_t count{0};
};

struct CandPlay {
    int32_t type;
    int32_t rank;
    int32_t count;
};

inline int card_sum_hands(const int hands[3][15]) {
    int s = 0;
    for (int p = 0; p < 3; ++p)
        for (int r = 0; r < 15; ++r)
            s += hands[p][r];
    return s;
}

inline int card_sum_counts(const int counts[15]) {
    int s = 0;
    for (int r = 0; r < 15; ++r) s += counts[r];
    return s;
}

inline Trick identify_trick(const int counts[15], int n) {
    if (n <= 0) return Trick{TRICK_NONE, -1, 0};
    if (n == 2 && counts[13] == 1 && counts[14] == 1)
        return Trick{TRICK_ROCKET, 14, 2};
    int max_c = 0, max_r = -1;
    for (int r = 0; r < 15; ++r) {
        if (counts[r] > max_c) { max_c = counts[r]; max_r = r; }
    }
    if (n == 4 && max_c == 4 && max_r >= 0 && max_r < 13)
        return Trick{TRICK_BOMB, max_r, 4};
    if (n == 1 && max_c == 1)
        return Trick{TRICK_SOLO, max_r, 1};
    if (n == 2 && max_c == 2 && max_r < 13)
        return Trick{TRICK_PAIR, max_r, 2};
    return Trick{TRICK_INVALID, -1, 0};
}

inline Trick identify_trick_ranks(const int* ranks, int n) {
    int counts[15] = {0};
    for (int i = 0; i < n; ++i) {
        int r = ranks[i];
        if (r < 0 || r > 14) return Trick{TRICK_INVALID, -1, 0};
        counts[r]++;
    }
    return identify_trick(counts, n);
}

inline bool can_beat(const Trick& neu, const Trick& prev) {
    if (neu.type == TRICK_INVALID || neu.type == TRICK_NONE) return false;
    if (prev.type == TRICK_NONE || prev.type == TRICK_INVALID) return true;
    // Rocket beats any non-rocket; rocket vs rocket is not a strict beat (single deck).
    if (neu.type == TRICK_ROCKET) return prev.type != TRICK_ROCKET;
    if (prev.type == TRICK_ROCKET) return false;
    if (neu.type == TRICK_BOMB) {
        if (prev.type == TRICK_BOMB) return neu.rank > prev.rank;
        return true;
    }
    if (prev.type == TRICK_BOMB) return false;
    if (neu.type != prev.type) return false;
    if (neu.count != prev.count) return false;
    return neu.rank > prev.rank;
}

// CONTRACT v1 · deal
inline int deal_counts(uint32_t seed, int hands[3][15], int kitty[15]) {
    std::memset(hands, 0, sizeof(int) * 3 * 15);
    std::memset(kitty, 0, sizeof(int) * 15);
    std::vector<int> deck;
    deck.reserve(54);
    for (int r = 0; r < 13; ++r)
        for (int k = 0; k < 4; ++k) deck.push_back(r);
    deck.push_back(13);
    deck.push_back(14);
    std::mt19937 rng(seed);
    std::shuffle(deck.begin(), deck.end(), rng);
    for (int i = 0; i < 17; ++i) hands[0][deck[i]]++;
    for (int i = 17; i < 34; ++i) hands[1][deck[i]]++;
    for (int i = 34; i < 51; ++i) hands[2][deck[i]]++;
    for (int i = 51; i < 54; ++i) kitty[deck[i]]++;
    return card_sum_hands(hands) + card_sum_counts(kitty);
}

inline void assign_landlord(int hands[3][15], const int kitty[15], int landlord) {
    for (int r = 0; r < 15; ++r) hands[landlord][r] += kitty[r];
}

// CONTRACT v1 · legal
inline void enumerate_candidates(const int hand[15], const Trick& table, std::vector<CandPlay>& out) {
    out.clear();
    if (table.type == TRICK_NONE) {
        for (int r = 0; r < 15; ++r) {
            if (hand[r] >= 1) out.push_back({TRICK_SOLO, r, 1});
            if (r < 13 && hand[r] >= 2) out.push_back({TRICK_PAIR, r, 2});
            if (r < 13 && hand[r] >= 4) out.push_back({TRICK_BOMB, r, 4});
        }
        if (hand[13] >= 1 && hand[14] >= 1) out.push_back({TRICK_ROCKET, 14, 2});
    } else {
        if (table.type == TRICK_SOLO) {
            for (int r = table.rank + 1; r < 15; ++r)
                if (hand[r] >= 1) out.push_back({TRICK_SOLO, r, 1});
        } else if (table.type == TRICK_PAIR) {
            for (int r = table.rank + 1; r < 13; ++r)
                if (hand[r] >= 2) out.push_back({TRICK_PAIR, r, 2});
        } else if (table.type == TRICK_BOMB) {
            for (int r = table.rank + 1; r < 13; ++r)
                if (hand[r] == 4) out.push_back({TRICK_BOMB, r, 4});
        }
        if (table.type != TRICK_BOMB && table.type != TRICK_ROCKET) {
            for (int r = 0; r < 13; ++r)
                if (hand[r] == 4) out.push_back({TRICK_BOMB, r, 4});
        }
        if (table.type != TRICK_ROCKET && hand[13] >= 1 && hand[14] >= 1)
            out.push_back({TRICK_ROCKET, 14, 2});
        out.push_back({TRICK_NONE, -1, 0});
    }
}

// Zero-alloc variant: write up to max_n cands into out[]; returns total legal count.
inline int enumerate_candidates_n(const int hand[15], const Trick& table, CandPlay* out, int max_n) {
    std::vector<CandPlay> tmp;
    enumerate_candidates(hand, table, tmp);
    int n = static_cast<int>(tmp.size());
    int w = n < max_n ? n : max_n;
    for (int i = 0; i < w; ++i) out[i] = tmp[i];
    return n;
}

// CONTRACT v1 · play
inline bool apply_play(int hand[15], Trick& table, const CandPlay& c, int* cards_left) {
    std::vector<CandPlay> legal;
    enumerate_candidates(hand, table, legal);
    bool ok = false;
    for (const auto& L : legal) {
        if (L.type == c.type && L.rank == c.rank && L.count == c.count) { ok = true; break; }
    }
    if (!ok) return false;
    if (c.type == TRICK_NONE) return true;
    if (c.type == TRICK_ROCKET) {
        hand[13]--; hand[14]--;
        if (cards_left) *cards_left -= 2;
        table = Trick{TRICK_ROCKET, 14, 2};
        return true;
    }
    hand[c.rank] -= c.count;
    if (cards_left) *cards_left -= c.count;
    table = Trick{static_cast<TrickType>(c.type), c.rank, c.count};
    return true;
}

// CONTRACT v1 · score — episode/terminal chips used by doudizhu_eval_harness.
// Frozen: win = +200, loss = −200. Prefer bounce required to reshape.
inline double terminal_score(bool agent_won) {
    return agent_won ? 200.0 : -200.0;
}

}  // namespace doudizhu_rules
}  // namespace kun
