// 迷宫 Gate 5：同一锁档、同一未见种子，两次独立回放轨迹必须位级重合。
// 任务层 only；不改 include/kun/cellular/。不宣称 Gate 6。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace kun;

static const char* find_maze_bin() {
    static const char* cands[] = {
        "checkpoints/maze_navigation_champion.bin",
        "../checkpoints/maze_navigation_champion.bin",
    };
    for (const char* p : cands) {
        CellularOrganism probe = CellularOrganism::load_checkpoint_bin(p);
        if (!probe.cells.empty()) return p;
    }
    return nullptr;
}

struct Pose {
    float x{0}, y{0}, theta{0};
    double pos{0}, neg{0};
};

static std::vector<Pose> replay_once(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    MazeTask task(11, 11, seed, max_steps, 0.15f);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    org.reset_state(false);

    std::vector<Pose> trace;
    trace.reserve(static_cast<size_t>(max_steps));
    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = task.step_continuous(acts);
        const auto& ag = task.get_agent();
        trace.push_back({ag.x, ag.y, ag.theta, acts.positive_action, acts.negative_action});
        if (res.done) break;
    }
    return trace;
}

int main() {
    const char* path = find_maze_bin();
    assert(path && "missing checkpoints/maze_navigation_champion.bin");

    const uint32_t seed = 50000u + 17u;
    const int max_steps = 250;
    auto a = replay_once(path, seed, max_steps);
    auto b = replay_once(path, seed, max_steps);
    assert(a.size() == b.size());
    assert(!a.empty());

    double max_dx = 0, max_dy = 0, max_dth = 0, max_dact = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_dx = std::max(max_dx, static_cast<double>(std::fabs(a[i].x - b[i].x)));
        max_dy = std::max(max_dy, static_cast<double>(std::fabs(a[i].y - b[i].y)));
        max_dth = std::max(max_dth, static_cast<double>(std::fabs(a[i].theta - b[i].theta)));
        max_dact = std::max(max_dact, std::fabs(a[i].pos - b[i].pos));
        max_dact = std::max(max_dact, std::fabs(a[i].neg - b[i].neg));
    }
    std::cout << "GATE5_MAZE steps=" << a.size()
              << " max_dx=" << max_dx
              << " max_dy=" << max_dy
              << " max_dth=" << max_dth
              << " max_dact=" << max_dact << "\n";
    assert(max_dx < 1e-6);
    assert(max_dy < 1e-6);
    assert(max_dth < 1e-6);
    assert(max_dact < 1e-6);
    std::cout << "PASS maze Gate 5 offline replay\n";
    return 0;
}
