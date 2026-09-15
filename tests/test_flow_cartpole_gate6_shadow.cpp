// CartPole Gate 6：PD 专家与冠军双轨影子。任务层 only；不改底座。
// 只用 ID（force_noise=0）。冠军与专家不必轨迹重合。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
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

// 观测解码与 CartPoleBalanceTask::current_observation 互逆。
static CellularOrganism::ActionOutputs pd_expert(const std::vector<float>& obs) {
    const double theta = static_cast<double>(obs[0]) * 0.35;
    const double theta_dot = static_cast<double>(obs[1]) * 3.0;
    const double x = static_cast<double>(obs[2]) * 2.4;
    const double x_dot = static_cast<double>(obs[3]) * 3.0;
    // 正 theta 向 +x 倾倒；正推力把小车推向 +x 以回正。
    const double u = std::clamp(
        12.0 * theta + 2.5 * theta_dot + 1.2 * x + 1.0 * x_dot, -1.0, 1.0);
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
    double mean_dforce{0};
};

static Episode run_expert(uint32_t seed, int max_steps) {
    CartPoleBalanceTask env;
    env.set_max_steps(max_steps);
    env.reset(seed);
    Episode ep;
    double prev_u = 0, dsum = 0;
    int n = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        auto acts = pd_expert(obs);
        const double u = acts.positive_action - acts.negative_action;
        if (t > 0) {
            dsum += std::fabs(u - prev_u);
            n++;
        }
        prev_u = u;
        auto res = env.step_continuous(acts);
        if (res.done) {
            ep.success = false;
            ep.steps = res.steps;
            ep.mean_dforce = n ? dsum / n : 0;
            return ep;
        }
    }
    ep.success = true;
    ep.steps = max_steps;
    ep.mean_dforce = n ? dsum / n : 0;
    return ep;
}

static Episode run_champion(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    CartPoleBalanceTask env;
    env.set_max_steps(max_steps);
    env.reset(seed);
    org.reset_state(false);
    Episode ep;
    double prev_u = 0, dsum = 0;
    int n = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        const double u = acts.positive_action - acts.negative_action;
        if (t > 0) {
            dsum += std::fabs(u - prev_u);
            n++;
        }
        prev_u = u;
        auto res = env.step_continuous(acts);
        if (res.done) {
            ep.success = false;
            ep.steps = res.steps;
            ep.mean_dforce = n ? dsum / n : 0;
            return ep;
        }
    }
    ep.success = true;
    ep.steps = max_steps;
    ep.mean_dforce = n ? dsum / n : 0;
    return ep;
}

int main() {
    const char* path = find_cartpole_bin();
    assert(path && "missing cartpole_balance_champion.bin");
    const int N = 20;
    const int max_steps = 300;
    int exp_ok = 0, champ_ok = 0;
    double champ_jitter = 0;
    for (int i = 0; i < N; ++i) {
        const uint32_t seed = 9000u + static_cast<uint32_t>(i) * 17u;
        auto e = run_expert(seed, max_steps);
        auto c = run_champion(path, seed, max_steps);
        if (e.success) exp_ok++;
        if (c.success) champ_ok++;
        champ_jitter += c.mean_dforce;
        std::cout << "  seed=" << seed
                  << " expert=" << (e.success ? "OK" : "FAIL") << "/" << e.steps
                  << " champ=" << (c.success ? "OK" : "FAIL") << "/" << c.steps
                  << " dF=" << c.mean_dforce << std::endl;
    }
    champ_jitter /= N;
    std::cout << "GATE6_CARTPOLE expert=" << exp_ok << "/" << N
              << " champ=" << champ_ok << "/" << N
              << " mean_|dF|=" << champ_jitter << std::endl;
    assert(exp_ok >= 16 && "PD expert cannot balance; environment/probe unhealthy");
    assert(champ_ok >= 16 && "champion shadow success below floor");
    assert(champ_ok + 2 >= exp_ok && "champion lags PD expert by more than 2/20");
    assert(champ_jitter < 0.80 && "champion force jitter too high vs shadow protocol");
    std::cout << "PASS cartpole Gate 6 shadow vs PD expert" << std::endl;
    return 0;
}
