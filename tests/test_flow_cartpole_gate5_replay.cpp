// CartPole Gate 5：同一锁档、同一未见种子，两次独立回放轨迹必须位级重合。
// 任务层 only；不改 include/kun/cellular/。只用 ID（force_noise=0）；OOD 高斯噪声不可回放。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace kun;

static const char* find_cartpole_bin() {
    static const char* cands[] = {
        "checkpoints/cartpole_balance_champion.bin",
        "../checkpoints/cartpole_balance_champion.bin",
    };
    for (const char* p : cands) {
        CellularOrganism probe = CellularOrganism::load_checkpoint_bin(p);
        if (!probe.cells.empty()) return p;
    }
    return nullptr;
}

struct Frame {
    float o0{0}, o1{0}, o2{0}, o3{0};
    double pos{0}, neg{0};
};

static std::vector<Frame> replay_once(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    CartPoleBalanceTask env;
    env.set_max_steps(max_steps);
    env.reset(seed);
    org.reset_state(false);

    std::vector<Frame> trace;
    trace.reserve(static_cast<size_t>(max_steps));
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = env.step_continuous(acts);
        trace.push_back({obs[0], obs[1], obs[2], obs[3], acts.positive_action, acts.negative_action});
        if (res.done) break;
    }
    return trace;
}

int main() {
    const char* path = find_cartpole_bin();
    assert(path && "missing checkpoints/cartpole_balance_champion.bin");

    const uint32_t seed = 9000u + 17u;  // 与 bench_easy_task_regression ID 种子族一致
    const int max_steps = 300;
    auto a = replay_once(path, seed, max_steps);
    auto b = replay_once(path, seed, max_steps);
    assert(a.size() == b.size());
    assert(!a.empty());

    double max_dobs = 0, max_dact = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_dobs = std::max(max_dobs, static_cast<double>(std::fabs(a[i].o0 - b[i].o0)));
        max_dobs = std::max(max_dobs, static_cast<double>(std::fabs(a[i].o1 - b[i].o1)));
        max_dobs = std::max(max_dobs, static_cast<double>(std::fabs(a[i].o2 - b[i].o2)));
        max_dobs = std::max(max_dobs, static_cast<double>(std::fabs(a[i].o3 - b[i].o3)));
        max_dact = std::max(max_dact, std::fabs(a[i].pos - b[i].pos));
        max_dact = std::max(max_dact, std::fabs(a[i].neg - b[i].neg));
    }
    std::cout << "GATE5_CARTPOLE steps=" << a.size()
              << " max_dobs=" << max_dobs
              << " max_dact=" << max_dact << std::endl;
    assert(a.size() == static_cast<size_t>(max_steps) && "champion fell before full replay");
    assert(max_dobs < 1e-6);
    assert(max_dact < 1e-6);
    std::cout << "PASS cartpole Gate 5 offline replay" << std::endl;
    return 0;
}
