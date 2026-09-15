// 迷宫 Gate 6：BFS 测地贪婪专家与冠军双轨影子。任务层 only；不改底座。
// 协议：环境可解（专家通关）+ 冠军不弱于专家太多 + 冠军转向不狂抖。
// 冠军与专家不必轨迹重合。obs[3] 比例控制实测 0/20，改用与任务层同构的 BFS 势场。
#include "kun/cellular/cellular_genome.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
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

// 与 MazeTask::rebuild_geodesic_field_ / train_maze_tripartite DistField 同构：终点反推最短格距。
struct DistField {
    std::vector<int> dist;
    int w = 0, h = 0;
    void build(const MazeEnvironment& maze, int width, int height) {
        w = width;
        h = height;
        dist.assign(static_cast<size_t>(w * h), -1);
        const int gx = w - 2, gy = h - 2;
        if (maze.is_wall(static_cast<float>(gx) + 0.5f, static_cast<float>(gy) + 0.5f)) return;
        dist[static_cast<size_t>(gy * w + gx)] = 0;
        std::vector<std::pair<int, int>> q{{gx, gy}};
        size_t head = 0;
        const int dx4[4] = {1, -1, 0, 0};
        const int dy4[4] = {0, 0, 1, -1};
        while (head < q.size()) {
            auto [cx, cy] = q[head++];
            for (int d = 0; d < 4; ++d) {
                int nx = cx + dx4[d], ny = cy + dy4[d];
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                if (dist[static_cast<size_t>(ny * w + nx)] != -1) continue;
                if (maze.is_wall(static_cast<float>(nx) + 0.5f, static_cast<float>(ny) + 0.5f)) continue;
                dist[static_cast<size_t>(ny * w + nx)] = dist[static_cast<size_t>(cy * w + cx)] + 1;
                q.push_back({nx, ny});
            }
        }
    }
};

// 测地贪婪：朝距离场下降邻格转向；大航向误差时停车转向，避免走廊里顶墙。
static CellularOrganism::ActionOutputs geodesic_expert(
    const DistField& df, const MazeTask& task, const std::vector<float>& obs) {
    const auto& ag = task.get_agent();
    int cx = static_cast<int>(std::floor(ag.x));
    int cy = static_cast<int>(std::floor(ag.y));
    const int dx4[4] = {1, -1, 0, 0};
    const int dy4[4] = {0, 0, 1, -1};
    int best_dist = 1 << 30, bx = cx, by = cy;
    for (int d = 0; d < 4; ++d) {
        int nx = cx + dx4[d], ny = cy + dy4[d];
        if (nx < 0 || ny < 0 || nx >= df.w || ny >= df.h) continue;
        int dv = df.dist[static_cast<size_t>(ny * df.w + nx)];
        if (dv < 0) continue;
        if (dv < best_dist) {
            best_dist = dv;
            bx = nx;
            by = ny;
        }
    }
    double desired = std::atan2(static_cast<double>(by) + 0.5 - ag.y,
                                static_cast<double>(bx) + 0.5 - ag.x);
    double diff = desired - static_cast<double>(ag.theta);
    while (diff > M_PI) diff -= 2 * M_PI;
    while (diff < -M_PI) diff += 2 * M_PI;

    CellularOrganism::ActionOutputs a;
    a.negative_action = std::clamp(2.5 * diff, -1.0, 1.0);
    const float front = obs[0];
    if (std::fabs(diff) > 0.55) {
        a.positive_action = 0.08;
    } else if (front < 0.22f) {
        a.positive_action = 0.12;
    } else {
        a.positive_action = (std::fabs(diff) < 0.35) ? 1.0 : 0.50;
    }
    return a;
}

struct Episode {
    bool success{false};
    int steps{0};
    int collisions{0};
    double mean_dneg{0};
};

static Episode run_expert(uint32_t seed, int max_steps) {
    MazeTask task(11, 11, seed, max_steps, 0.15f);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    DistField df;
    df.build(task.get_maze(), task.get_width(), task.get_height());
    Episode ep;
    double prev_neg = 0, dsum = 0;
    int n = 0;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        auto acts = geodesic_expert(df, task, obs);
        if (t > 0) {
            dsum += std::fabs(acts.negative_action - prev_neg);
            n++;
        }
        prev_neg = acts.negative_action;
        auto res = task.step_continuous(acts);
        if (res.done) {
            ep.success = res.success;
            ep.steps = res.steps;
            ep.collisions = task.get_agent().collision_count;
            ep.mean_dneg = n ? dsum / n : 0;
            return ep;
        }
    }
    ep.steps = max_steps;
    ep.collisions = task.get_agent().collision_count;
    ep.mean_dneg = n ? dsum / n : 0;
    return ep;
}

static Episode run_champion(const char* path, uint32_t seed, int max_steps) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    assert(!org.cells.empty());
    org.compile();
    MazeTask task(11, 11, seed, max_steps, 0.15f);
    task.set_use_geodesic_bearing(true);
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
            ep.success = res.success;
            ep.steps = res.steps;
            ep.collisions = task.get_agent().collision_count;
            ep.mean_dneg = n ? dsum / n : 0;
            return ep;
        }
    }
    ep.steps = max_steps;
    ep.collisions = task.get_agent().collision_count;
    ep.mean_dneg = n ? dsum / n : 0;
    return ep;
}

int main() {
    const char* path = find_maze_bin();
    assert(path && "missing maze_navigation_champion.bin");
    const int N = 20;
    const int max_steps = 250;
    int exp_ok = 0, champ_ok = 0;
    double champ_jitter = 0;
    for (int i = 0; i < N; ++i) {
        const uint32_t seed = 50000u + static_cast<uint32_t>(i) * 17u;
        auto e = run_expert(seed, max_steps);
        auto c = run_champion(path, seed, max_steps);
        if (e.success) exp_ok++;
        if (c.success) champ_ok++;
        champ_jitter += c.mean_dneg;
        std::cout << "  seed=" << seed
                  << " expert=" << (e.success ? "OK" : "FAIL") << "/" << e.steps
                  << " champ=" << (c.success ? "OK" : "FAIL") << "/" << c.steps
                  << " dneg=" << c.mean_dneg << std::endl;
    }
    champ_jitter /= N;
    std::cout << "GATE6_MAZE expert=" << exp_ok << "/" << N
              << " champ=" << champ_ok << "/" << N
              << " mean_|dneg|=" << champ_jitter << std::endl;
    assert(exp_ok >= 16 && "geodesic expert cannot solve the maze; environment/probe unhealthy");
    assert(champ_ok >= 16 && "champion shadow success below expert floor");
    assert(champ_ok + 2 >= exp_ok && "champion lags geodesic expert by more than 2/20");
    assert(champ_jitter < 0.80 && "champion turn jitter too high vs shadow protocol");
    std::cout << "PASS maze Gate 6 shadow vs geodesic expert" << std::endl;
    return 0;
}
