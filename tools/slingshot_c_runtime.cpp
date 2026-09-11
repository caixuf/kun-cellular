// 引力弹弓专业页原生运行时
//   - 冠军: checkpoints/slingshot_nav_champion.bin (organism 真前向驱动推力)
//   - SlingshotNavTask (3 体牛顿引力 + 探测器) + CellularOrganism::forward_nd
#include "tasks/robotics/slingshot_nav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

#define SL_MAX_STARS 8
#define SL_MAX_PROBES 8
#define SL_MAX_TRAIL 64
#define SL_HIST 64

typedef struct { float x, y; int32_t r; } SlTarget;
typedef struct { float x, y; char color[16]; } SlStar;
typedef struct {
    int32_t id;
    float x, y;
    int32_t alive;
    int32_t reached;
    int32_t trail_len;
    float trail[SL_MAX_TRAIL][2];
} SlProbe;

typedef struct {
    int32_t real;
    int32_t generation;
    int32_t step_count;
    int32_t max_steps;
    float success_rate; // 0..100
    int32_t history_len;
    float history_success[SL_HIST];
    SlTarget target;
    int32_t n_stars;
    SlStar stars[SL_MAX_STARS];
    int32_t n_probes;
    SlProbe probes[SL_MAX_PROBES];
} SlingshotTelemetry;

static_assert(std::is_trivially_copyable<SlingshotTelemetry>::value,
              "SlingshotTelemetry must be POD for ctypes mirroring");

static std::mutex g_mutex;
static CellularOrganism g_champion;
static std::unique_ptr<SlingshotNavTask> g_task;
static std::mt19937 g_rng(20260911);
static bool g_loaded = false;
static int32_t g_episodes = 0;
static int32_t g_wins = 0;
static int32_t g_step_count = 0;
static int32_t g_max_steps = 400;
static float g_history[SL_HIST] = {0.0f};
static int32_t g_history_len = 0;
static int32_t g_trail_len = 0;
static float g_trail[SL_MAX_TRAIL][2];

static const char* kStarColors[3] = {"#f59e0b", "#38bdf8", "#a855f7"};

static bool load_organism(const char* path, CellularOrganism& out) {
    auto b = CellularOrganism::load_checkpoint_bin(path);
    if (!b.cells.empty()) { out = std::move(b); return true; }
    out = CellularOrganism::load_checkpoint_json(path);
    return !out.cells.empty();
}

static void push_trail_locked() {
    if (!g_task) return;
    const float px = static_cast<float>(g_task->probe_x());
    const float py = static_cast<float>(g_task->probe_y());
    if (g_trail_len < SL_MAX_TRAIL) {
        g_trail[g_trail_len][0] = px;
        g_trail[g_trail_len][1] = py;
        g_trail_len++;
    } else {
        std::memmove(g_trail, g_trail + 1, sizeof(float) * 2 * (SL_MAX_TRAIL - 1));
        g_trail[SL_MAX_TRAIL - 1][0] = px;
        g_trail[SL_MAX_TRAIL - 1][1] = py;
    }
}

static void start_episode_locked(uint32_t seed) {
    g_task = std::make_unique<SlingshotNavTask>();
    g_task->set_max_steps(g_max_steps);
    g_task->reset(seed);
    g_champion.reset_state(false);
    g_step_count = 0;
    g_trail_len = 0;
    push_trail_locked();
}

int32_t slingshot_c_init(const char* ckpt_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string path = (ckpt_path && ckpt_path[0])
        ? ckpt_path
        : "checkpoints/slingshot_nav_champion.bin";
    std::ifstream test_f(path, std::ios::binary);
    if (!test_f.is_open()) return 0;
    test_f.close();
    if (!load_organism(path.c_str(), g_champion)) return 0;
    g_champion.compile();
    g_episodes = 0;
    g_wins = 0;
    g_history_len = 0;
    std::memset(g_history, 0, sizeof(g_history));
    start_episode_locked(g_rng());
    g_loaded = true;
    return 1;
}

void slingshot_c_reset(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return;
    if (seed == 0) seed = g_rng();
    start_episode_locked(seed);
}

int32_t slingshot_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return 0;

    auto obs = g_task->current_observation();
    std::vector<double> inps(obs.begin(), obs.end());
    auto acts = g_champion.forward_nd(inps.data(), inps.size(), false);
    auto res = g_task->step_continuous(acts);
    g_step_count++;
    if (g_step_count % 2 == 0) push_trail_locked();

    if (!res.done) return 0;

    g_episodes++;
    if (res.success) g_wins++;
    const float rate = (g_episodes > 0)
        ? 100.0f * static_cast<float>(g_wins) / static_cast<float>(g_episodes)
        : 0.0f;
    if (g_history_len < SL_HIST) {
        g_history[g_history_len++] = rate;
    } else {
        std::memmove(g_history, g_history + 1, sizeof(float) * (SL_HIST - 1));
        g_history[SL_HIST - 1] = rate;
    }
    start_episode_locked(g_rng());
    return 1;
}

void slingshot_c_get_telemetry(SlingshotTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    std::memset(out, 0, sizeof(SlingshotTelemetry));
    out->real = g_loaded ? 1 : 0;
    out->generation = g_episodes;
    out->step_count = g_step_count;
    out->max_steps = g_max_steps;
    out->success_rate = (g_episodes > 0)
        ? 100.0f * static_cast<float>(g_wins) / static_cast<float>(g_episodes)
        : 0.0f;
    out->history_len = g_history_len;
    std::memcpy(out->history_success, g_history, sizeof(g_history));

    if (!g_task) return;

    out->target.x = static_cast<float>(g_task->target_x());
    out->target.y = static_cast<float>(g_task->target_y());
    out->target.r = static_cast<int32_t>(g_task->target_r());

    const int ns = static_cast<int>(std::min(g_task->num_stars(), size_t{SL_MAX_STARS}));
    for (int i = 0; i < ns; ++i) {
        out->stars[i].x = static_cast<float>(g_task->star_x(static_cast<size_t>(i)));
        out->stars[i].y = static_cast<float>(g_task->star_y(static_cast<size_t>(i)));
        std::snprintf(out->stars[i].color, sizeof(out->stars[i].color), "%s",
                      kStarColors[i % 3]);
    }
    out->n_stars = ns;

    out->probes[0].id = 0;
    out->probes[0].x = static_cast<float>(g_task->probe_x());
    out->probes[0].y = static_cast<float>(g_task->probe_y());
    out->probes[0].alive = 1;
    out->probes[0].reached = g_task->reached() ? 1 : 0;
    out->probes[0].trail_len = g_trail_len;
    std::memcpy(out->probes[0].trail, g_trail, sizeof(g_trail));
    out->n_probes = 1;
}

#ifdef __cplusplus
}
#endif
