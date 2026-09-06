#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include <mutex>
#include <random>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace kun;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t real;
    int32_t episodes;
    int32_t wins;
    float win_rate;
    int32_t landlord_games;
    int32_t landlord_wins;
    int32_t peasant_games;
    int32_t peasant_wins;
    int32_t last_action;
    int32_t step_in_episode;
    int32_t role; // 1: Landlord, 0: Peasant
    int32_t cards_left;
    int32_t opp_left;
    int32_t opp_right;
    float hand_strength;
    float threat;
    float table;
    float cell_voltages[12];
    float cell_outputs[12];
    float head_acts[3]; // 0: PASS, 1: FOLLOW, 2: SPRINT
    float column_act[4]; // 4 columns
} DouDiZhuTelemetry;

static std::mutex g_mutex;
static CellularOrganism g_champion;
static std::unique_ptr<DouDiZhuCardGameTask> g_task;
static std::mt19937 g_rng(20260906);
static bool g_loaded = false;

static int32_t g_episodes = 0;
static int32_t g_wins = 0;
static int32_t g_landlord_games = 0;
static int32_t g_landlord_wins = 0;
static int32_t g_peasant_games = 0;
static int32_t g_peasant_wins = 0;
static int32_t g_last_action = 1;
static int32_t g_step_in_episode = 0;

static float g_cell_voltages[12] = {0.0f};
static float g_cell_outputs[12] = {0.0f};
static float g_head_acts[3] = {0.0f};
static float g_column_act[4] = {0.0f};

int32_t doudizhu_c_init(const char* ckpt_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    try {
        std::string path = ckpt_path ? ckpt_path : "checkpoints/doudizhu_evolved_champion.bin";
        std::ifstream test_f(path, std::ios::binary);
        if (!test_f.is_open()) {
            return 0;
        }
        test_f.close();

        g_champion = CellularOrganism::load_checkpoint_bin(path);
        g_task = std::make_unique<DouDiZhuCardGameTask>(50, g_rng());
        g_task->reset(g_rng());
        g_champion.reset_state(true);

        g_loaded = true;
        g_episodes = 0;
        g_wins = 0;
        g_landlord_games = 0;
        g_landlord_wins = 0;
        g_peasant_games = 0;
        g_peasant_wins = 0;
        g_step_in_episode = 0;
        return 1;
    } catch (...) {
        g_loaded = false;
        return 0;
    }
}

void doudizhu_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return;

    auto obs = g_task->current_observation();
    double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
    auto acts = g_champion.forward(inps, false);

    int act = 1;
    if (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) {
        act = 2; // SPRINT
    } else if (acts.negative_action > acts.positive_action) {
        act = 0; // PASS
    } else {
        act = 1; // FOLLOW
    }
    g_last_action = act;

    g_head_acts[0] = static_cast<float>(acts.negative_action);
    g_head_acts[1] = static_cast<float>(acts.positive_action);
    g_head_acts[2] = static_cast<float>(acts.defensive_reset);

    // Record cell outputs & voltages
    for (size_t i = 0; i < 12 && i < g_champion.cells.size(); ++i) {
        g_cell_voltages[i] = g_champion.cells[i].membrane_potential;
        g_cell_outputs[i] = static_cast<float>(g_champion.cells[i].output_val);
    }

    // 4 Functional columns
    // Col 0: Sensory (0..3)
    float sum0 = 0.0f;
    for (int i = 0; i <= 3; ++i) sum0 += std::abs(g_cell_outputs[i]);
    g_column_act[0] = std::min(100.0f, sum0 / 4.0f * 100.0f);

    // Col 1: Reasoning (4..5)
    float sum1 = std::abs(g_cell_outputs[4]) + std::abs(g_cell_outputs[5]);
    g_column_act[1] = std::min(100.0f, sum1 / 2.0f * 100.0f);

    // Col 2: Hysteresis Memory (6..8)
    float sum2 = std::abs(g_cell_outputs[6]) + std::abs(g_cell_outputs[7]) + std::abs(g_cell_outputs[8]);
    g_column_act[2] = std::min(100.0f, sum2 / 3.0f * 100.0f);

    // Col 3: Action Execution (9..11)
    float sum3 = std::abs(g_cell_outputs[9]) + std::abs(g_cell_outputs[10]) + std::abs(g_cell_outputs[11]);
    g_column_act[3] = std::min(100.0f, sum3 / 3.0f * 100.0f);

    auto step_res = g_task->step_continuous(acts);
    g_step_in_episode = step_res.steps;

    if (step_res.done) {
        g_episodes++;
        bool is_l = (g_task->role() == 1);
        if (is_l) g_landlord_games++; else g_peasant_games++;

        if (step_res.success) {
            g_wins++;
            if (is_l) g_landlord_wins++; else g_peasant_wins++;
        }
        g_champion.reset_state(true);
        g_task->reset(g_rng());
        g_step_in_episode = 0;
    }
}

void doudizhu_c_get_telemetry(DouDiZhuTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    out->real = g_loaded ? 1 : 0;
    out->episodes = g_episodes;
    out->wins = g_wins;
    out->win_rate = (g_episodes > 0) ? (static_cast<float>(g_wins) / g_episodes * 100.0f) : 55.5f;
    out->landlord_games = g_landlord_games;
    out->landlord_wins = g_landlord_wins;
    out->peasant_games = g_peasant_games;
    out->peasant_wins = g_peasant_wins;
    out->last_action = g_last_action;
    out->step_in_episode = g_step_in_episode;

    if (g_task) {
        out->role = g_task->role();
        out->cards_left = g_task->cards_left(0);
        out->opp_left = g_task->cards_left(1);
        out->opp_right = g_task->cards_left(2);

        auto obs = g_task->current_observation();
        out->hand_strength = obs[0];
        out->threat = obs[3];
        out->table = obs[2];
    } else {
        out->role = 1;
        out->cards_left = 17;
        out->opp_left = 17;
        out->opp_right = 20;
        out->hand_strength = 0.5f;
        out->threat = 0.2f;
        out->table = 0.0f;
    }

    std::memcpy(out->cell_voltages, g_cell_voltages, sizeof(g_cell_voltages));
    std::memcpy(out->cell_outputs, g_cell_outputs, sizeof(g_cell_outputs));
    std::memcpy(out->head_acts, g_head_acts, sizeof(g_head_acts));
    std::memcpy(out->column_act, g_column_act, sizeof(g_column_act));
}

int32_t doudizhu_c_decide(float obs0, float obs1, float obs2, float obs3, float* out_heads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return 1;

    double inps[4] = {obs0, obs1, obs2, obs3};
    auto acts = g_champion.forward(inps, false);

    if (out_heads) {
        out_heads[0] = static_cast<float>(acts.negative_action);
        out_heads[1] = static_cast<float>(acts.positive_action);
        out_heads[2] = static_cast<float>(acts.defensive_reset);
    }

    int act = 1;
    if (acts.defensive_reset > acts.positive_action && acts.defensive_reset > acts.negative_action) {
        act = 2; // SPRINT
    } else if (acts.negative_action > acts.positive_action) {
        act = 0; // PASS
    } else {
        act = 1; // FOLLOW
    }
    return act;
}

#ifdef __cplusplus
}
#endif
