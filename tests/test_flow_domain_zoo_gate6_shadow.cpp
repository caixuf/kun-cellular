// DomainZoo Gate 6：物理符号 PD 探针 + 冠军推力抖动影子。任务层 only；不改底座。
// 生存率只记录：fresh-load 下 zoo_cartpole 远低于 domain_zoo_report.json 的 id_sr，
// 不得用 JSON 0.9 当本测试地板。G6 硬阈 = 12 域 mean|ΔF| < 0.80。
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

static bool plus_plant(const char* id) {
    return std::strcmp(id, "zoo_cartpole") == 0 || std::strcmp(id, "zoo_bicycle") == 0;
}

static CellularOrganism::ActionOutputs pd_expert(const char* id, const std::vector<float>& obs) {
    const double raw = 1.5 * obs[0] + 0.40 * obs[1];
    const double u = std::clamp(plus_plant(id) ? raw : -raw, -1.0, 1.0);
    CellularOrganism::ActionOutputs a;
    if (u >= 0.0) {
        a.positive_action = u;
        a.negative_action = 0.0;
    } else {
        a.positive_action = 0.0;
        a.negative_action = -u;
    }
    return a;
}

struct Episode {
    bool success{false};
    int steps{0};
    double mean_df{0};
};

static Episode run_loop(ZooTask& env, int max_steps, bool expert,
                        CellularOrganism* org, const char* id) {
    Episode ep;
    double prev = 0, dsum = 0;
    int n = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        CellularOrganism::ActionOutputs acts;
        if (expert) {
            acts = pd_expert(id, obs);
        } else {
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            acts = org->forward(inps, false);
        }
        const double u = acts.positive_action - acts.negative_action;
        if (t > 0) {
            dsum += std::fabs(u - prev);
            n++;
        }
        prev = u;
        auto res = env.step_continuous(acts);
        if (res.done) {
            ep.success = false;
            ep.steps = res.steps;
            ep.mean_df = n ? dsum / n : 0;
            return ep;
        }
    }
    ep.success = true;
    ep.steps = max_steps;
    ep.mean_df = n ? dsum / n : 0;
    return ep;
}

int main() {
    const int N = 10;
    int domains_champ_ok = 0, domains_jitter_ok = 0;
    int domains_expert_ok = 0;
    int cartpole_champ = -1, ballbeam_exp = -1;
    double rocket_j = 0, thermal_j = 0, servo_j = 0, bicycle_j = 0;
    for (const char* id : kIds) {
        const std::string path = find_bin(id);
        assert(!path.empty());
        CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
        assert(!org.cells.empty());
        org.compile();
        const int max_steps = (std::strcmp(id, "zoo_maglev") == 0) ? 600 : 300;
        int exp_ok = 0, champ_ok = 0;
        double champ_j = 0;
        for (int i = 0; i < N; ++i) {
            const uint32_t seed = 201u + static_cast<uint32_t>(i);
            auto env_e = make_zoo(id);
            env_e->set_max_steps(max_steps);
            env_e->reset(seed);
            auto e = run_loop(*env_e, max_steps, true, nullptr, id);
            auto env_c = make_zoo(id);
            env_c->set_max_steps(max_steps);
            env_c->reset(seed);
            org.reset_state(true);
            auto c = run_loop(*env_c, max_steps, false, &org, id);
            if (e.success) exp_ok++;
            if (c.success) champ_ok++;
            champ_j += c.mean_df;
        }
        champ_j /= N;
        const bool c_floor = champ_ok >= 7;
        const bool j_ok = champ_j < 0.80;
        if (exp_ok >= 7) domains_expert_ok++;
        if (c_floor) domains_champ_ok++;
        if (j_ok) domains_jitter_ok++;
        if (std::strcmp(id, "zoo_cartpole") == 0) cartpole_champ = champ_ok;
        if (std::strcmp(id, "zoo_ballbeam") == 0) ballbeam_exp = exp_ok;
        if (std::strcmp(id, "zoo_rocket_hover") == 0) rocket_j = champ_j;
        if (std::strcmp(id, "zoo_thermal") == 0) thermal_j = champ_j;
        if (std::strcmp(id, "zoo_servo") == 0) servo_j = champ_j;
        if (std::strcmp(id, "zoo_bicycle") == 0) bicycle_j = champ_j;
        std::cout << "  " << id
                  << " expert=" << exp_ok << "/" << N
                  << " champ=" << champ_ok << "/" << N
                  << " mean_|dF|=" << champ_j
                  << (j_ok ? " jitter_ok" : " JITTER_FAIL") << std::endl;
    }
    std::cout << "GATE6_ZOO VERDICT=FAIL jitter=" << domains_jitter_ok << "/12"
              << " champ_ge7=" << domains_champ_ok << "/12"
              << " pd_probe_ge7=" << domains_expert_ok << "/12" << std::endl;
    // 负例锁档：不得把本测试改成 12/12 丝滑 PASS。ctest 绿 = 审计复现，不是 G6 过关。
    assert(domains_jitter_ok == 8);
    assert(domains_champ_ok == 11);
    assert(domains_expert_ok == 11);
    assert(cartpole_champ == 1);
    assert(ballbeam_exp == 0);
    assert(rocket_j > 0.80 && thermal_j > 0.80 && servo_j > 0.80 && bicycle_j > 0.80);
    std::cout << "AUDIT domain zoo Gate 6 (FAIL vs 12/12 smoothness; numbers locked)\n";
    return 0;
}
