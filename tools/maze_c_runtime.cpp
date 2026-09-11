// 迷宫专业页原生运行时：与 bench_easy_task_regression / L3 锁档协议对齐
//   - 冠军: checkpoints/maze_navigation_champion.bin
//   - MazeTask(11×11) + set_use_geodesic_bearing(true) + CellularOrganism::forward
//   - 禁止随机基因组演化沙盒冒充战役成绩
#include "tasks/robotics/maze_navigator.hpp"
#include "kun/cellular/cellular_genome.hpp"
#include <mutex>
#include <random>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>

using namespace kun;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t real;
    int32_t episodes;
    int32_t wins;
    float success_rate; // 0..1 滚动通关率
    int32_t width;
    int32_t height;
    int32_t step_count;
    int32_t max_steps;
    int32_t done;
    int32_t success;
    float agent_x;
    float agent_y;
    float agent_theta;
    float ray_front;
    float ray_left;
    float ray_right;
    float bearing;
    float start_x;
    float start_y;
    float goal_x;
    float goal_y;
    int32_t n_cells;
    int32_t n_synapses;
    int32_t grid[121]; // 11×11
    float cell_outs[16];
} MazeTelemetry;

static std::mutex g_mutex;
static CellularOrganism g_champion;
static std::unique_ptr<MazeTask> g_task;
static std::mt19937 g_rng(20260911);
static bool g_loaded = false;
static int32_t g_episodes = 0;
static int32_t g_wins = 0;
static int32_t g_max_steps = 250;
static constexpr int kMap = 11;
static constexpr float kBraid = 0.15f;

static bool load_organism(const char* path, CellularOrganism& out) {
    auto b = CellularOrganism::load_checkpoint_bin(path);
    if (!b.cells.empty()) {
        out = std::move(b);
        return true;
    }
    out = CellularOrganism::load_checkpoint_json(path);
    return !out.cells.empty();
}

static void start_episode_locked(uint32_t seed) {
    g_task = std::make_unique<MazeTask>(kMap, kMap, seed, g_max_steps, kBraid);
    g_task->set_use_geodesic_bearing(true);
    g_task->reset(seed);
    g_champion.reset_state(false);
}

int32_t maze_c_init(const char* ckpt_path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string path = (ckpt_path && ckpt_path[0])
        ? ckpt_path
        : "checkpoints/maze_navigation_champion.bin";
    std::ifstream test_f(path, std::ios::binary);
    if (!test_f.is_open()) return 0;
    test_f.close();
    if (!load_organism(path.c_str(), g_champion)) return 0;
    g_champion.compile();
    g_episodes = 0;
    g_wins = 0;
    start_episode_locked(g_rng());
    g_loaded = true;
    return 1;
}

void maze_c_reset(uint32_t seed) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded) return;
    if (seed == 0) seed = g_rng();
    start_episode_locked(seed);
}

// 返回 1 表示本步结束了一局（已自动开新局）
int32_t maze_c_step() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_loaded || !g_task) return 0;

    auto obs = g_task->current_observation();
    double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
    auto acts = g_champion.forward(inps, false);
    auto res = g_task->step_continuous(acts);

    if (!res.done) return 0;

    g_episodes++;
    if (res.success) g_wins++;
    start_episode_locked(g_rng());
    return 1;
}

void maze_c_get_telemetry(MazeTelemetry* out) {
    if (!out) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    std::memset(out, 0, sizeof(MazeTelemetry));
    out->real = g_loaded ? 1 : 0;
    out->episodes = g_episodes;
    out->wins = g_wins;
    out->success_rate = (g_episodes > 0)
        ? (static_cast<float>(g_wins) / static_cast<float>(g_episodes))
        : 0.0f;
    out->width = kMap;
    out->height = kMap;
    out->max_steps = g_max_steps;
    out->n_cells = static_cast<int32_t>(g_champion.cells.size());
    out->n_synapses = static_cast<int32_t>(g_champion.synapses.size());

    if (!g_task) return;

    const auto& maze = g_task->get_maze();
    const auto& agent = g_task->get_agent();
    const auto& grid = maze.get_grid();
    out->step_count = agent.steps;
    out->agent_x = agent.x;
    out->agent_y = agent.y;
    out->agent_theta = agent.theta;
    out->ray_front = agent.ray_dists[0];
    out->ray_left = agent.ray_dists[1];
    out->ray_right = agent.ray_dists[2];
    out->bearing = agent.goal_bearing;
    out->start_x = maze.get_start_x();
    out->start_y = maze.get_start_y();
    out->goal_x = maze.get_goal_x();
    out->goal_y = maze.get_goal_y();
    out->done = agent.reached_goal ? 1 : 0;
    out->success = agent.reached_goal ? 1 : 0;

    const int n = kMap * kMap;
    for (int i = 0; i < n && i < static_cast<int>(grid.size()); ++i) {
        out->grid[i] = static_cast<int32_t>(grid[static_cast<size_t>(i)]);
    }

    const size_t nc = std::min(g_champion.cells.size(), size_t{16});
    for (size_t i = 0; i < nc; ++i) {
        out->cell_outs[i] = static_cast<float>(g_champion.cells[i].output_val);
    }
}

#ifdef __cplusplus
}
#endif
