// 全屋覆盖 Gate 6：BFS 最近污渍教师（环境可解）与冠军影子。任务层 only；不改底座。
// 教师握全图，冠军 4 维局部；禁止成功计数对撞。ID 24×16，不注入动态障碍。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/robotics/household_coverage.hpp"

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

struct Episode {
    bool success{false};
    double coverage{0};
    bool docked{false};
    int collisions{0};
    int steps{0};
    double mean_dneg{0};
};

static Episode run_expert(uint32_t seed, int max_steps) {
    auto report = HouseholdCoverageEvaluator::run_baseline(24, 16, seed, max_steps);
    Episode ep;
    ep.coverage = report.coverage_ratio;
    ep.docked = report.returned_to_dock;
    ep.collisions = report.collisions;
    ep.success = (report.coverage_ratio >= 0.70 && report.returned_to_dock);
    return ep;
}

static Episode run_champion(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    HouseholdCoverageTask task(24, 16, seed, max_steps);
    task.reset(seed);
    org.reset_state(false);
    Episode ep;
    double prev_neg = 0, dsum = 0;
    int n = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        if (t > 0) {
            dsum += std::fabs(acts.negative_action - prev_neg);
            n++;
        }
        prev_neg = acts.negative_action;
        auto res = task.step_continuous(acts);
        if (res.done) {
            ep.steps = res.steps;
            break;
        }
        ep.steps = t + 1;
    }
    const auto& e = task.env();
    ep.coverage = e.coverage_ratio();
    ep.docked = e.at_dock();
    ep.collisions = static_cast<int>(e.collision_count());
    ep.success = (ep.coverage >= 0.70 && ep.docked);
    ep.mean_dneg = n ? dsum / n : 0;
    return ep;
}

int main() {
    const char* path = find_household_bin();
    assert(path && "missing household_coverage_champion.bin");
    const int N = 20;
    const int max_steps = 1200;
    int exp_ok = 0, champ_ok = 0;
    double champ_jitter = 0, champ_cov = 0, exp_cov = 0;
    int champ_col = 0;
    for (int i = 0; i < N; ++i) {
        const uint32_t seed = 1000u + static_cast<uint32_t>(i);
        auto e = run_expert(seed, max_steps);
        auto c = run_champion(path, seed, max_steps);
        if (e.success) exp_ok++;
        if (c.success) champ_ok++;
        champ_jitter += c.mean_dneg;
        champ_cov += c.coverage;
        exp_cov += e.coverage;
        champ_col += c.collisions;
        std::cout << "  seed=" << seed
                  << " expert=" << (e.success ? "OK" : "FAIL")
                  << "/" << e.coverage
                  << " dock=" << e.docked
                  << " champ=" << (c.success ? "OK" : "FAIL")
                  << "/" << c.coverage
                  << " dock=" << c.docked
                  << " col=" << c.collisions
                  << " dneg=" << c.mean_dneg << std::endl;
    }
    champ_jitter /= N;
    champ_cov /= N;
    exp_cov /= N;
    std::cout << "GATE6_HOUSEHOLD expert=" << exp_ok << "/" << N
              << " champ=" << champ_ok << "/" << N
              << " mean_cov e/c=" << exp_cov << "/" << champ_cov
              << " champ_col=" << champ_col
              << " mean_|dneg|=" << champ_jitter << std::endl;
    // BFS 教师握全图；冠军只有 4 维局部观测。禁止用成功计数对撞（迷宫同信息测地才比 2/20）。
    assert(exp_ok >= 16 && "BFS coverage teacher cannot solve ID rooms; environment/probe unhealthy");
    assert(champ_ok >= 16 && "champion shadow success below 0.70 coverage + dock floor");
    assert(champ_cov >= 0.70 && "champion mean coverage below 0.70");
    assert(champ_col == 0 && "champion collisions on ID shadow set");
    assert(champ_jitter < 0.80 && "champion turn jitter too high vs shadow protocol");
    std::cout << "PASS household Gate 6 shadow vs BFS nearest-dirt teacher" << std::endl;
    return 0;
}
