// 多足步态专业页原生运行时
//   - 冠军: checkpoints/locomotion_gait_champion.bin (organism 真前向驱动肌肉)
//   - LocomotionGaitTask + CellularOrganism::forward_nd
#include "tasks/robotics/locomotion_gait.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <type_traits>

using namespace kun;

#ifdef __cplusplus
extern "C" {
#endif

#define LOCO_MAX_NODES 8
#define LOCO_MAX_MUSCLES 8
#define LOCO_HIST 64

typedef struct { float x, y; } LocoNode;
typedef struct { int32_t n1, n2; float rest; } LocoMuscle;

typedef struct {
    int32_t real;
    int32_t generation;
    int32_t step_count;
    int32_t max_steps;
    float best_distance;
    int32_t history_len;
    float history_dist[LOCO_HIST];
    int32_t n_nodes;
    LocoNode nodes[LOCO_MAX_NODES];
    int32_t n_muscles;
    LocoMuscle muscles[LOCO_MAX_MUSCLES];
} LocomotionTelemetry;

static_assert(std::is_trivially_copyable<LocomotionTelemetry>::value,
              "LocomotionTelemetry must be POD for ctypes mirroring");

static std::mutex g_mutex;
static CellularOrganism g_champion;
static std::unique_ptr<LocomotionGaitTask> g_task;
static std::mt19937 g_rng(20260911);
static bool g_loaded = false;
static int32_t g_episodes = 0;
static int32_t g_step_count = 0;
static int32_t g_max_steps = 200;
static float g_best_distance = 0.0f;
static float g_history[LOCO_HIST] = {0.0f};
static int32_t g_history_len = 0;

static bool load_organism(const char* path, CellularOrganism& out) {
    auto b = CellularOrganism::load_checkpoint_bin(path);
    if (!b.cells.empty()) { out = std::move(b); return true; }
    out = CellularOrganism::load_checkpoint_json(path);
    return !out.cells.empty();
}

static void start_episode_locked(uint32_t seed) {
    g_task = std::make_unique<LocomotionGaitTask>();
    g_task->set_max_steps(g_max_steps);
    g_task->reset(seed);
    g_champion.reset_state(false);
    g_step_count = 0;
}

int32_t locomotion_c_init(const char* ckpt_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string path = (ckpt_path && ckpt_path[0])
        ? ckpt_path
        : "checkpoints/locomotion_gait_champion.bin";
    std::ifstream test_f(path, std::ios::binary);
    if (!test_f.is_open()) return 0;
    test_f.close();
    if (!load_organism(path.c_str(), g_champion)) return 0;
    g_champion.compile();
    g_episodes = 0;
    g_best_distance = 0.0f;
    g_history_len = 0;
    std::memset(g_history, 0, sizeof(g_history));
    start_episode_locked(g_rng());
    g_loaded = true;
    return 1;
}

void locomotion_c_reset(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return;
    if (seed == 0) seed = g_rng();
    g_best_distance = 0.0f;
    start_episode_locked(seed);
}

// 返回 1 表示本步结束了一轮 (已自动开启新一轮)
int32_t locomotion_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return 0;

    auto obs = g_task->current_observation();
    std::vector<double> inps(obs.begin(), obs.end());
    auto acts = g_champion.forward_nd(inps.data(), inps.size(), false);
    auto res = g_task->step_continuous(acts);
    g_step_count++;

    if (!res.done) return 0;

    g_episodes++;
    const float disp = static_cast<float>(g_task->best_disp());
    if (disp > g_best_distance) g_best_distance = disp;
    if (g_history_len < LOCO_HIST) {
        g_history[g_history_len++] = static_cast<float>(disp);
    } else {
        std::memmove(g_history, g_history + 1, sizeof(float) * (LOCO_HIST - 1));
        g_history[LOCO_HIST - 1] = static_cast<float>(disp);
    }
    start_episode_locked(g_rng());
    return 1;
}

void locomotion_c_get_telemetry(LocomotionTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    std::memset(out, 0, sizeof(LocomotionTelemetry));
    out->real = g_loaded ? 1 : 0;
    out->generation = g_episodes;
    out->step_count = g_step_count;
    out->max_steps = g_max_steps;
    out->best_distance = g_best_distance;
    out->history_len = g_history_len;
    std::memcpy(out->history_dist, g_history, sizeof(g_history));

    if (!g_task) return;
    const int nn = static_cast<int>(std::min(g_task->num_nodes(), size_t{LOCO_MAX_NODES}));
    for (int i = 0; i < nn; ++i) {
        out->nodes[i].x = static_cast<float>(g_task->node_x(static_cast<size_t>(i)));
        out->nodes[i].y = static_cast<float>(g_task->node_y(static_cast<size_t>(i)));
    }
    out->n_nodes = nn;
    const int nm = static_cast<int>(std::min(g_task->num_muscles(), size_t{LOCO_MAX_MUSCLES}));
    for (int m = 0; m < nm; ++m) {
        out->muscles[m].n1 = g_task->muscle_n1(static_cast<size_t>(m));
        out->muscles[m].n2 = g_task->muscle_n2(static_cast<size_t>(m));
        out->muscles[m].rest = static_cast<float>(g_task->muscle_rest(static_cast<size_t>(m)));
    }
    out->n_muscles = nm;
}

#ifdef __cplusplus
}
#endif
