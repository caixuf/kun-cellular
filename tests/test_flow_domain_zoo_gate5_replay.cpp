// DomainZoo Gate 5：12 域各自锁档、同一 ID 留出种子，两次独立回放必须位级重合。
// 任务层 only；不改底座。ood=1.0。振动/锅炉等过程噪声由 reset(seed) 决定，同动作应对齐。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/domain_zoo.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace kun;

static const char* kIds[] = {
    "zoo_cartpole", "zoo_ballbeam", "zoo_maglev", "zoo_rocket_hover",
    "zoo_cruise", "zoo_thermal", "zoo_water_tank", "zoo_dc_motor",
    "zoo_vibration", "zoo_servo", "zoo_boiler", "zoo_bicycle",
};

static std::unique_ptr<ZooTask> make_zoo(const char* id) {
    if (std::strcmp(id, "zoo_cartpole") == 0) return std::make_unique<ZooCartPole>(1.0);
    if (std::strcmp(id, "zoo_ballbeam") == 0) return std::make_unique<ZooBallBeam>(1.0);
    if (std::strcmp(id, "zoo_maglev") == 0) return std::make_unique<ZooMaglev>(1.0);
    if (std::strcmp(id, "zoo_rocket_hover") == 0) return std::make_unique<ZooRocketHover>(1.0);
    if (std::strcmp(id, "zoo_cruise") == 0) return std::make_unique<ZooCruise>(1.0);
    if (std::strcmp(id, "zoo_thermal") == 0) return std::make_unique<ZooThermal>(1.0);
    if (std::strcmp(id, "zoo_water_tank") == 0) return std::make_unique<ZooWaterTank>(1.0);
    if (std::strcmp(id, "zoo_dc_motor") == 0) return std::make_unique<ZooDCMotor>(1.0);
    if (std::strcmp(id, "zoo_vibration") == 0) return std::make_unique<ZooVibration>(1.0);
    if (std::strcmp(id, "zoo_servo") == 0) return std::make_unique<ZooServo>(1.0);
    if (std::strcmp(id, "zoo_boiler") == 0) return std::make_unique<ZooBoiler>(1.0);
    if (std::strcmp(id, "zoo_bicycle") == 0) return std::make_unique<ZooBicycle>(1.0);
    return nullptr;
}

static std::string find_bin(const char* id) {
    const std::string a = std::string("checkpoints/") + id + ".bin";
    const std::string b = std::string("../checkpoints/") + id + ".bin";
    for (const auto& p : {a, b}) {
        CellularOrganism probe = CellularOrganism::load_checkpoint_bin(p);
        if (!probe.cells.empty()) return p;
    }
    return {};
}

struct Frame {
    float o0{0}, o1{0}, o2{0}, o3{0};
    double pos{0}, neg{0};
};

static std::vector<Frame> replay_once(const std::string& path, const char* id,
                                      uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    auto env = make_zoo(id);
    assert(env);
    env->set_max_steps(max_steps);
    env->reset(seed);
    org.reset_state(true);  // 与 ZooTask::evaluate_organism 塑性隔离一致
    std::vector<Frame> trace;
    trace.reserve(static_cast<size_t>(max_steps));
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env->current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = env->step_continuous(acts);
        trace.push_back({obs[0], obs[1], obs[2], obs[3], acts.positive_action, acts.negative_action});
        if (res.done) break;
    }
    return trace;
}

int main() {
    const uint32_t seed = 201u;  // holdout ID 族首种子
    int n_ok = 0;
    for (const char* id : kIds) {
        const std::string path = find_bin(id);
        assert(!path.empty() && "missing zoo_*.bin");
        const int max_steps = (std::strcmp(id, "zoo_maglev") == 0) ? 600 : 300;
        auto a = replay_once(path, id, seed, max_steps);
        auto b = replay_once(path, id, seed, max_steps);
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
        std::cout << "  " << id << " steps=" << a.size()
                  << " max_dobs=" << max_dobs
                  << " max_dact=" << max_dact << std::endl;
        assert(max_dobs < 1e-6);
        assert(max_dact < 1e-6);
        n_ok++;
    }
    std::cout << "GATE5_ZOO domains=" << n_ok << "/12 seed=" << seed << std::endl;
    assert(n_ok == 12);
    std::cout << "PASS domain zoo Gate 5 offline replay" << std::endl;
    return 0;
}
