// 全屋覆盖 Gate 5：同一锁档、同一未见种子，两次独立回放轨迹必须位级重合。
// 任务层 only；不改 include/kun/cellular/。不注入动态障碍（注入依赖当时朝向，属 G6 另议）。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/robotics/household_coverage.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace kun;

static const char* find_household_bin() {
    static const char* cands[] = {
        "checkpoints/household_coverage_champion.bin",
        "../checkpoints/household_coverage_champion.bin",
    };
    for (const char* p : cands) {
        CellularOrganism probe = CellularOrganism::load_checkpoint_bin(p);
        if (!probe.cells.empty()) return p;
    }
    return nullptr;
}

struct Frame {
    int x{0}, y{0}, heading{0};
    double cov{0}, bat{0};
    double pos{0}, neg{0}, def{0};
};

static std::vector<Frame> replay_once(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    HouseholdCoverageTask task(24, 16, seed, max_steps);
    task.reset(seed);
    org.reset_state(false);

    std::vector<Frame> trace;
    trace.reserve(static_cast<size_t>(max_steps));
    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = task.step_continuous(acts);
        const auto& e = task.env();
        trace.push_back({
            e.robot_x(), e.robot_y(), static_cast<int>(e.heading()),
            e.coverage_ratio(), e.battery_ratio(),
            acts.positive_action, acts.negative_action, acts.defensive_reset
        });
        if (res.done) break;
    }
    return trace;
}

int main() {
    const char* path = find_household_bin();
    assert(path && "missing checkpoints/household_coverage_champion.bin");

    const uint32_t seed = 1000u;  // 与 tripartite --eval-only ID 种子族一致
    const int max_steps = 1200;
    auto a = replay_once(path, seed, max_steps);
    auto b = replay_once(path, seed, max_steps);
    assert(a.size() == b.size());
    assert(!a.empty());

    double max_dxy = 0, max_dcov = 0, max_dbat = 0, max_dact = 0;
    int max_dhead = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_dxy = std::max(max_dxy, static_cast<double>(std::abs(a[i].x - b[i].x)));
        max_dxy = std::max(max_dxy, static_cast<double>(std::abs(a[i].y - b[i].y)));
        max_dhead = std::max(max_dhead, std::abs(a[i].heading - b[i].heading));
        max_dcov = std::max(max_dcov, std::fabs(a[i].cov - b[i].cov));
        max_dbat = std::max(max_dbat, std::fabs(a[i].bat - b[i].bat));
        max_dact = std::max(max_dact, std::fabs(a[i].pos - b[i].pos));
        max_dact = std::max(max_dact, std::fabs(a[i].neg - b[i].neg));
        max_dact = std::max(max_dact, std::fabs(a[i].def - b[i].def));
    }
    std::cout << "GATE5_HOUSEHOLD steps=" << a.size()
              << " max_dxy=" << max_dxy
              << " max_dhead=" << max_dhead
              << " max_dcov=" << max_dcov
              << " max_dbat=" << max_dbat
              << " max_dact=" << max_dact << std::endl;
    assert(max_dxy < 1e-6);
    assert(max_dhead == 0);
    assert(max_dcov < 1e-12);
    assert(max_dbat < 1e-12);
    assert(max_dact < 1e-6);
    std::cout << "PASS household Gate 5 offline replay" << std::endl;
    return 0;
}
