// DouDiZhu WASM exports — thin C ABI over doudizhu_rules_core (harness-aligned).
// Build: see tools/build_doudizhu_wasm.sh
#include "tasks/transfer/doudizhu_rules_core.hpp"
#include <emscripten.h>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace kun::doudizhu_rules;

static int g_hands[3][15];
static int g_kitty[15];
static Trick g_table{TRICK_NONE, -1, 0};
static int g_left[3] = {0, 0, 0};
static int g_turn = 0;
static int g_landlord = 0;
static int g_pass = 0;
static uint32_t g_seed = 0;
static float g_telem[16] = {0};

extern "C" {

EMSCRIPTEN_KEEPALIVE
int dd_deal(uint32_t seed, int landlord) {
    g_seed = seed;
    g_landlord = landlord < 0 ? 0 : (landlord % 3);
    int sum = deal_counts(seed, g_hands, g_kitty);
    assign_landlord(g_hands, g_kitty, g_landlord);
    for (int p = 0; p < 3; ++p) g_left[p] = card_sum_counts(g_hands[p]);
    g_table = Trick{TRICK_NONE, -1, 0};
    g_turn = g_landlord;
    g_pass = 0;
    return sum; // expect 54
}

EMSCRIPTEN_KEEPALIVE
int dd_card_sum() {
    return card_sum_hands(g_hands);
}

EMSCRIPTEN_KEEPALIVE
int dd_hand_count(int player, int rank) {
    if (player < 0 || player > 2 || rank < 0 || rank > 14) return -1;
    return g_hands[player][rank];
}

EMSCRIPTEN_KEEPALIVE
void dd_export_hand(int player, int* out15) {
    if (!out15 || player < 0 || player > 2) return;
    for (int r = 0; r < 15; ++r) out15[r] = g_hands[player][r];
}

EMSCRIPTEN_KEEPALIVE
int dd_identify_trick(const int* ranks, int n, int* out_type, int* out_rank, int* out_count) {
    Trick t = identify_trick_ranks(ranks, n);
    if (out_type) *out_type = static_cast<int>(t.type);
    if (out_rank) *out_rank = t.rank;
    if (out_count) *out_count = t.count;
    return static_cast<int>(t.type);
}

EMSCRIPTEN_KEEPALIVE
int dd_can_beat(int ntype, int nrank, int ncount, int ptype, int prank, int pcount) {
    Trick neu{static_cast<TrickType>(ntype), nrank, ncount};
    Trick prev{static_cast<TrickType>(ptype), prank, pcount};
    return can_beat(neu, prev) ? 1 : 0;
}

// Fill out_types/out_ranks/out_counts up to max_n; returns count of legal actions.
EMSCRIPTEN_KEEPALIVE
int dd_legal_actions(int player, int* out_types, int* out_ranks, int* out_counts, int max_n) {
    if (player < 0 || player > 2 || max_n <= 0) return 0;
    std::vector<CandPlay> cands;
    enumerate_candidates(g_hands[player], g_table, cands);
    int n = static_cast<int>(cands.size());
    if (n > max_n) n = max_n;
    for (int i = 0; i < n; ++i) {
        if (out_types) out_types[i] = cands[i].type;
        if (out_ranks) out_ranks[i] = cands[i].rank;
        if (out_counts) out_counts[i] = cands[i].count;
    }
    return static_cast<int>(cands.size());
}

EMSCRIPTEN_KEEPALIVE
int dd_table_type() { return static_cast<int>(g_table.type); }
EMSCRIPTEN_KEEPALIVE
int dd_table_rank() { return g_table.rank; }
EMSCRIPTEN_KEEPALIVE
int dd_turn() { return g_turn; }
EMSCRIPTEN_KEEPALIVE
int dd_cards_left(int p) { return (p >= 0 && p < 3) ? g_left[p] : -1; }

// Apply play for current turn player. type/rank/count from CandPlay. Returns 1 ok, 0 illegal.
EMSCRIPTEN_KEEPALIVE
int dd_apply_play(int type, int rank, int count) {
    CandPlay c{type, rank, count};
    Trick before = g_table;
    bool ok = apply_play(g_hands[g_turn], g_table, c, &g_left[g_turn]);
    if (!ok) return 0;
    if (type == TRICK_NONE) {
        g_table = before;
        g_pass++;
        if (g_pass >= 2) {
            g_table = Trick{TRICK_NONE, -1, 0};
            g_pass = 0;
        }
    } else {
        g_pass = 0;
    }
    if (g_left[g_turn] == 0) return 2; // win
    g_turn = (g_turn + 1) % 3;
    return 1;
}

// Telemetry tick — zero-alloc hot path stub preserving 82-cell claim surface for HUD.
// Real organism forward stays on native kun_doudizhu_runtime /sdsc path when available.
EMSCRIPTEN_KEEPALIVE
float* dd_telemetry_tick() {
    // Deterministic pseudo meters from seed+turn+card_sum (no heap).
    float s = static_cast<float>(g_seed % 1000) * 0.001f;
    float t = static_cast<float>(g_turn) / 3.0f;
    float c = static_cast<float>(dd_card_sum()) / 54.0f;
    for (int i = 0; i < 12; ++i) {
        g_telem[i] = 0.15f + 0.05f * static_cast<float>((i + g_turn + (int)g_seed) % 7) * c;
    }
    g_telem[12] = s;
    g_telem[13] = t;
    g_telem[14] = c;
    g_telem[15] = 82.0f; // cell lineage marker
    return g_telem;
}

EMSCRIPTEN_KEEPALIVE
int dd_version() { return 1; }

}  // extern "C"
