// ============================================================================
// bench_easy_task_regression.cpp — T4 易证三任务冷评烟测
// CartPole balance / 迷宫导航 / 流体冠军拓扑冒烟（流体动力学见 C11 压测）
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace kun;

static bool episode_cartpole(CellularOrganism& org, CartPoleBalanceTask& env,
                             uint32_t seed, int max_steps, bool wipe_weights) {
    env.reset(seed);
    org.reset_state(wipe_weights);
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = env.step_continuous(acts);
        if (res.done) return false;
    }
    return true;
}

static int bench_cartpole(const char* path, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    if (org.cells.empty()) {
        std::printf("[C1 FAIL] 无法加载 %s\n", path);
        return 0;
    }
    org.compile();

    CartPoleBalanceTask id_env;
    id_env.set_max_steps(max_steps);

    CartPoleBalanceTask::Params ood_p;
    ood_p.masspole = 0.2;
    ood_p.length = 0.7;
    ood_p.force_noise = 2.0;
    CartPoleBalanceTask ood_env(ood_p);
    ood_env.set_max_steps(max_steps);

    // 20 ID + 20 OOD（对齐预注册门禁；种子与训练报告独立）
    int id_ok = 0, ood_ok = 0;
    for (int i = 0; i < 20; ++i) {
        if (episode_cartpole(org, id_env, 9000u + static_cast<uint32_t>(i) * 17u, max_steps, false))
            ++id_ok;
        if (episode_cartpole(org, ood_env, 9100u + static_cast<uint32_t>(i) * 19u, max_steps, false))
            ++ood_ok;
    }

    const double id_sr = id_ok / 20.0;
    const double ood_sr = ood_ok / 20.0;
    const bool pass = (id_sr >= 0.95) && (ood_sr >= 0.90);

    std::printf("[C1 CartPole] %s | cells=%zu syns=%zu | steps=%d | ID %d/20=%.0f%% | OOD %d/20=%.0f%% | %s\n",
                path, org.cells.size(), org.synapses.size(), max_steps,
                id_ok, id_sr * 100.0, ood_ok, ood_sr * 100.0,
                pass ? "PASS" : "FAIL");
    return pass ? 1 : 0;
}

static bool episode_maze(CellularOrganism& org, int map_size, uint32_t seed,
                         int max_steps, float braid, bool wipe_weights) {
    MazeTask task(map_size, map_size, seed, max_steps, braid);
    org.reset_state(wipe_weights);
    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = task.step_continuous(acts);
        if (res.done) return res.success;
    }
    return false;
}

static int bench_maze(const char* path) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    if (org.cells.empty()) {
        std::printf("[M1 FAIL] 无法加载 %s\n", path);
        return 0;
    }
    org.compile();

    const int MAP = 11;
    const int MAX_STEPS = 250;
    const float BRAID = 0.15f;
    const int N = 100;

    int ok_false = 0, ok_true = 0;
    for (int s = 0; s < N; ++s) {
        uint32_t unseen = 50000u + static_cast<uint32_t>(s) * 17u;
        if (episode_maze(org, MAP, unseen, MAX_STEPS, BRAID, false)) ++ok_false;
        if (episode_maze(org, MAP, unseen, MAX_STEPS, BRAID, true)) ++ok_true;
    }

    const double sr_f = ok_false / static_cast<double>(N);
    const double sr_t = ok_true / static_cast<double>(N);
    // 主口径：表型冷评 reset_state(false)；≥0.95 软锁
    const bool pass = (sr_f >= 0.95);

    std::printf("[M1 Maze] %s | cells=%zu syns=%zu | phenotype(false) %d/%d=%.0f%% | genome(true) %d/%d=%.0f%% | %s\n",
                path, org.cells.size(), org.synapses.size(),
                ok_false, N, sr_f * 100.0, ok_true, N, sr_t * 100.0,
                pass ? "PASS" : "FAIL");
    return pass ? 1 : 0;
}

static int bench_fluid_topology(const char* path) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    if (org.cells.empty()) {
        std::printf("[F1 FAIL] 无法加载 %s\n", path);
        return 0;
    }
    org.compile();
    org.reset_state(false);

    // 拓扑/前向冒烟：6 维扰动输入 → 有限动作（动力学闭环由 C11 压测承担）
    double inps[6] = {0.1, -0.05, 0.02, 0.5, 0.1, 0.0};
    auto acts = org.forward_nd(inps, 6, false);
    const bool finite =
        std::isfinite(acts.positive_action) && std::isfinite(acts.negative_action);
    const bool topo_ok = (org.cells.size() >= 8) && (org.synapses.size() >= 8) && finite;

    std::printf("[F1 Fluid topo] %s | cells=%zu syns=%zu | act=(%.4f,%.4f) | %s\n",
                path, org.cells.size(), org.synapses.size(),
                acts.positive_action, acts.negative_action,
                topo_ok ? "PASS" : "FAIL");
    return topo_ok ? 1 : 0;
}

int main(int argc, char** argv) {
    // 主锁档：tripartite（balance 历史 bin 的 initial_weight 全 0，冷评失效）
    const char* cartpole = "checkpoints/cartpole_tripartite_champion.bin";
    const char* maze = "checkpoints/maze_navigation_champion.bin";
    const char* fluid = "checkpoints/fluid_damper_champion.bin";
    if (argc > 1) cartpole = argv[1];
    if (argc > 2) maze = argv[2];
    if (argc > 3) fluid = argv[3];

    std::printf("======================================================================\n");
    std::printf("  T4 易证三任务冷评回归 (CartPole / Maze / Fluid topology)\n");
    std::printf("======================================================================\n");

    int passed = 0;
    passed += bench_cartpole(cartpole, 500);
    // 诊断旁路：旧 README 指向的 balance 冠军（期望 FAIL，不计入门禁）
    (void)bench_cartpole("checkpoints/cartpole_balance_champion.bin", 300);
    passed += bench_maze(maze);
    passed += bench_fluid_topology(fluid);

    std::printf("----------------------------------------------------------------------\n");
    std::printf("  合计 %d/3 子项通过（流体全动力学请另跑 ./build/test_multiphase_fluid_stress）\n",
                passed);
    std::printf("======================================================================\n");
    return (passed == 3) ? 0 : 1;
}
