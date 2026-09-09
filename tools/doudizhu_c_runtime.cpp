// P9 前端遥测 runtime v2: 当前斗地主生命体 (82 细胞 / 425 突触候选打分世系)
// (v1 旧 12 细胞三动作架构已删除 — 单一真相源, 见 docs/STATUS_BOARD.md)
// C ABI 与 v1 完全兼容 (Python/前端零改动), 语义重映射:
//   - 冠军: checkpoints/doudizhu_cand_scorer.bin (U0.2 冠军 57.0%, bin/JSON 双兼容加载)
//   - 决策协议: 候选打分 (44 obs + 4 座次 + 12 候选特征 → 56 维 → argmax)
//     与 tools/p9_runner.cpp 评测协议逐位一致
//   - cell_outputs[12]: 6 感知 + 4 特征 + 打分头 + 价值头 (采样展示)
//   - column_act[4]: 感知柱 / 特征柱 / 价值头 / 打分头 (mean|out| 归一)
//   - head_acts[3]: 过牌分 / 最优出牌分 / 次优出牌分
#include "tasks/transfer/cross_domain_tasks.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include <mutex>
#include <random>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

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
    float head_acts[3]; // 0: 过牌分, 1: 最优出牌分, 2: 次优出牌分
    float column_act[4]; // 感知 / 特征 / 价值头 / 打分头
} DouDiZhuTelemetry;

static std::mutex g_mutex;
static CellularOrganism g_champion;
static std::unique_ptr<DouDiZhuCardGameTask> g_task;
static std::mt19937 g_rng(20260909);
static bool g_loaded = false;
static size_t g_score_head = 0;
static size_t g_score_idx = 0, g_value_idx = 0;
static size_t g_sample_idx[12] = {0};
static size_t g_feat_idx[40] = {0};
static size_t g_n_feat = 0;

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

static bool load_organism(const char* path, CellularOrganism& out) {
    auto b = CellularOrganism::load_checkpoint_bin(path);
    if (!b.cells.empty()) { out = std::move(b); return true; }
    out = CellularOrganism::load_checkpoint_json(path);
    return !out.cells.empty();
}

static void locate_heads_and_samples() {
    g_score_idx = g_value_idx = 0;
    g_n_feat = 0;
    size_t n_sense = 0, n_sample = 0;
    for (size_t i = 0; i < g_champion.cells.size(); ++i) {
        const auto& c = g_champion.cells[i];
        if (c.type == CellType::SENSE_CHANNEL) {
            if (n_sense < 6) g_sample_idx[n_sample++] = i;
            ++n_sense;
        } else if (c.type == CellType::OP_SUM || c.type == CellType::OP_ABS) {
            if (g_n_feat < 40) g_feat_idx[g_n_feat++] = i;
            if (n_sample < 10) g_sample_idx[n_sample++] = i;
        } else if (c.type == CellType::ACT_CHANNEL) {
            if (c.param2 == 0.0) {
                g_score_idx = i;
                if (n_sample < 11) g_sample_idx[n_sample++] = i;
            } else {
                g_value_idx = i;
                if (n_sample < 12) g_sample_idx[n_sample++] = i;
            }
        }
    }
    while (n_sample < 12) g_sample_idx[n_sample++] = 0;
    g_score_head = g_score_idx;
}

// 候选打分 (56 维输入 → 分数头), 返回 {pick, best, second, pass}
static void score_all_candidates(std::vector<DouDiZhuCardGameTask::CandPlay>& cands,
                                 size_t& pick, size_t& second,
                                 double& best, double& second_best, double& pass_score) {
    pick = second = 0;
    best = second_best = -1e308;
    pass_score = 0.0;
    auto o = g_task->current_observation();
    std::vector<float> obs44(o.begin(), o.end());
    for (float v : g_task->seat_context()) obs44.push_back(v);
    for (size_t i = 0; i < cands.size(); ++i) {
        auto cf = g_task->candidate_features(cands[i]);
        std::vector<double> in(56);
        for (int d = 0; d < 44; ++d) in[d] = obs44[d];
        for (int d = 0; d < 12; ++d) in[44 + d] = cf[d];
        g_champion.reset_state(false);
        g_champion.forward_nd(in.data(), in.size(), false);
        const double sc = g_champion.cells[g_score_head].output_val;
        if (sc > best) { second_best = best; second = pick; best = sc; pick = i; }
        else if (sc > second_best) { second_best = sc; second = i; }
        if (cands[i].type == 0) pass_score = sc;
    }
}

int32_t doudizhu_c_init(const char* ckpt_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string path = (ckpt_path && ckpt_path[0]) ? ckpt_path : "checkpoints/doudizhu_cand_scorer.bin";
    std::ifstream test_f(path, std::ios::binary);
    if (!test_f.is_open()) return 0;
    test_f.close();
    if (!load_organism(path.c_str(), g_champion)) return 0;
    locate_heads_and_samples();

    g_task = std::make_unique<DouDiZhuCardGameTask>(40, g_rng(), 17.5);
    g_champion.reset_state(false);
    g_loaded = true;
    g_episodes = g_wins = g_landlord_games = g_landlord_wins = 0;
    g_peasant_games = g_peasant_wins = 0;
    g_step_in_episode = 0;
    return 1;
}

void doudizhu_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return;

    // ── 候选打分决策 (一个 step = 一次决策, 与 p9_runner 逐位一致) ──
    auto cands = g_task->enumerate_candidates(0);
    size_t pick = 0, second = 0;
    double best = 0, second_best = 0, pass_score = 0;
    score_all_candidates(cands, pick, second, best, second_best, pass_score);
    g_head_acts[0] = (float)pass_score;
    g_head_acts[1] = (float)best;
    g_head_acts[2] = (float)second_best;
    g_last_action = (cands[pick].type == 0) ? 0 : 1;

    // 细胞采样遥测
    for (int i = 0; i < 12; ++i) {
        const size_t idx = g_sample_idx[i];
        g_cell_outputs[i] = (float)g_champion.cells[idx].output_val;
        g_cell_voltages[i] = (float)g_champion.cells[idx].membrane_potential;
    }

    // 4 功能柱: 感知 / 特征 / 价值头 / 打分头
    double s0 = 0;
    for (size_t k = 0; k < 6; ++k) s0 += std::abs(g_champion.cells[g_sample_idx[k]].output_val);
    g_column_act[0] = (float)std::min(100.0, s0 / 6.0 * 100.0);
    double s1 = 0;
    for (size_t k = 0; k < g_n_feat; ++k) s1 += std::abs(g_champion.cells[g_feat_idx[k]].output_val);
    g_column_act[1] = (float)std::min(100.0, s1 / std::max<size_t>(1, g_n_feat) * 100.0);
    g_column_act[2] = (float)std::min(100.0, std::abs(g_champion.cells[g_value_idx].output_val) * 100.0);
    g_column_act[3] = (float)std::min(100.0, std::abs(g_champion.cells[g_score_idx].output_val) * 20.0);

    // 执行最优候选
    auto res = g_task->play_candidate(cands[pick]);
    g_step_in_episode = res.steps;
    if (res.done) {
        g_episodes++;
        const bool is_l = (g_task->role() == 1);
        if (is_l) g_landlord_games++; else g_peasant_games++;
        if (res.success) {
            g_wins++;
            if (is_l) g_landlord_wins++; else g_peasant_wins++;
        }
        g_champion.reset_state(false);
        g_task = std::make_unique<DouDiZhuCardGameTask>(40, g_rng(), 17.5);
        g_step_in_episode = 0;
    }
}

void doudizhu_c_get_telemetry(DouDiZhuTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    out->real = g_loaded ? 1 : 0;
    out->episodes = g_episodes;
    out->wins = g_wins;
    out->win_rate = (g_episodes > 0) ? (static_cast<float>(g_wins) / g_episodes * 100.0f) : 57.0f;
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
        auto o = g_task->current_observation();
        out->hand_strength = o[29];   // 己方余牌率
        out->threat = o[31];          // 地主余牌率 (威胁度)
        out->table = o[22];           // 台面牌型
    } else {
        out->role = 1;
        out->cards_left = 17;
        out->opp_left = 17;
        out->opp_right = 17;
        out->hand_strength = 0.85f;
        out->threat = 0.85f;
        out->table = 0.0f;
    }
    std::memcpy(out->cell_voltages, g_cell_voltages, sizeof(g_cell_voltages));
    std::memcpy(out->cell_outputs, g_cell_outputs, sizeof(g_cell_outputs));
    std::memcpy(out->head_acts, g_head_acts, sizeof(g_head_acts));
    std::memcpy(out->column_act, g_column_act, sizeof(g_column_act));
}

int32_t doudizhu_c_decide(float, float, float, float, float* out_heads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return 1;
    auto cands = g_task->enumerate_candidates(0);
    size_t pick = 0, second = 0;
    double best = 0, second_best = 0, pass_score = 0;
    score_all_candidates(cands, pick, second, best, second_best, pass_score);
    if (out_heads) {
        out_heads[0] = (float)pass_score;
        out_heads[1] = (float)best;
        out_heads[2] = (float)second_best;
    }
    return (cands[pick].type == 0) ? 0 : 1;
}

#ifdef __cplusplus
}
#endif
