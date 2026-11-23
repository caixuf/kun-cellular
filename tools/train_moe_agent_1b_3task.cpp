// ============================================================================
// train_moe_agent_1b_3task.cpp — S1 三任务：共享 1B 储层 + 每任务独立读出/路由
//
// 复用 S4 的 500 × 2M = 1B cells / 3B syns block bins（30 GB，/tmp 本地）。
// 每个任务：
//   - 冠军采集 (obs, action)
//   - 任务专属 4D 网格路由（500 桶）
//   - 每块 ridge 读出（64 block features + 4 raw obs = 68 维）
//   - top-5 = 10M active = 1%
//
// 判据（S1 三任务）：每个任务 agent SR >= champion SR - 5pp。
// 输出：runs/moe_agent_1b_3task_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_binary_runtime.h"
#include "kun/cellular/sdsc_compact_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"
#include "tasks/control/domain_zoo.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace kun;

namespace {

constexpr uint32_t kBlocks = 500;
constexpr uint32_t kBlockCells = 2'000'000;
constexpr uint32_t kBlockSyns = 6'000'000;
constexpr uint32_t kIn = 64;
constexpr uint32_t kFeat = 64;
constexpr uint32_t kRaw = 4;
constexpr uint32_t kX = kFeat + kRaw;
constexpr int kCacheMax = 24;
constexpr int kTasks = 3;
constexpr int kTopK = 5;  // 5 × 2M = 10M active = 1%

enum class Task { Maze, CartPole, Maglev };

struct Sample {
    std::array<float, kRaw> obs{};
    double a0{0.0};
    double a1{0.0};
    int block{-1};
};

struct BlockMeta {
    std::array<double, kRaw> centroid{};
    int samples{0};
    bool ready{false};
    std::vector<double> mean, std, w;  // (kX+1) x 2
};

static std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
    return buf;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else o.push_back(c);
    }
    return o;
}

static CompactSoAGenome make_block(uint32_t cells, uint32_t syns, uint32_t seed) {
    auto g = CompactSoAGenome::create_empty(cells, syns, kIn, kFeat);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> gain(0.5, 1.5);
    std::normal_distribution<double> wdist(0.0, 0.25);
    const uint8_t ops[] = {
        SDSC_OP_DAMPER, SDSC_OP_INTEGRATE, SDSC_OP_DIFF, SDSC_OP_SUM,
        SDSC_OP_MULTIPLY, SDSC_OP_CLIP, SDSC_OP_ABS,
    };
    for (uint32_t i = 0; i < cells; ++i) {
        g.op_types[i] = (i < kIn) ? SDSC_OP_PASSTHRU : ops[rng() % (sizeof(ops))];
        g.gains[i] = static_cast<float>(gain(rng));
        g.biases[i] = 0.0f;
    }
    for (uint32_t i = cells - kFeat; i < cells; ++i) {
        g.op_types[i] = SDSC_OP_SUM;
        g.gains[i] = 1.0f;
    }
    uint32_t cursor = 0;
    g.inc_off[0] = 0;
    for (uint32_t i = 0; i < kIn; ++i) g.inc_off[i + 1] = 0;
    for (uint32_t i = kIn; i < cells; ++i) {
        g.inc_off[i] = cursor;
        const uint32_t remaining_cells = cells - i;
        const uint32_t remaining_edges = syns - cursor;
        uint32_t edges = remaining_cells ? remaining_edges / remaining_cells : 0;
        if (edges == 0) edges = 1;
        if (cursor + edges > syns) edges = syns - cursor;
        for (uint32_t e = 0; e < edges; ++e) {
            uint32_t src = static_cast<uint32_t>(rng() % i);
            g.inc_from[cursor] = src;
            g.inc_weight[cursor] = static_cast<float>((src < kIn) ? (0.8 + 0.4 * gain(rng))
                                                                  : wdist(rng));
            ++cursor;
        }
        g.inc_off[i + 1] = cursor;
    }
    g.inc_off[cells] = cursor;
    for (uint32_t k = cursor; k < syns; ++k) {
        g.inc_from[k] = 0;
        g.inc_weight[k] = 0.0f;
    }
    if (cursor != syns) g.inc_off[cells] = syns;
    for (uint32_t k = 0; k < kIn; ++k) g.in_cell_ids[k] = k;
    for (uint32_t k = 0; k < kFeat; ++k) g.out_cell_ids[k] = cells - kFeat + k;
    return g;
}

static void reset_block(SDSCBinaryGraph* g) {
    const size_t n = g->header.num_cells;
    std::memset(g->states, 0, n * sizeof(float));
    std::memset(g->aux_states, 0, n * sizeof(float));
    std::memset(g->outputs, 0, n * sizeof(float));
    std::memset(g->inputs_accum, 0, n * sizeof(float));
}

static void block_forward(SDSCBinaryGraph* g, const std::array<float, kRaw>& obs, float* out) {
    float in[kIn];
    for (uint32_t i = 0; i < kIn; ++i) in[i] = obs[i % kRaw] * ((i & 1u) ? -1.0f : 1.0f);
    in[kIn - 1] = 1.0f;
    sdsc_binary_forward(g, in, out);
}

static int grid_block(Task task, const std::array<float, kRaw>& obs) {
    auto bin = [](float v, float lo, float hi, int n) {
        int k = static_cast<int>((v - lo) / (hi - lo) * n);
        return std::max(0, std::min(n - 1, k));
    };
    uint32_t idx = 0;
    if (task == Task::Maze) {
        idx = static_cast<uint32_t>(((bin(obs[0], 0, 1, 9) * 9 + bin(obs[1], 0, 1, 9)) * 9 +
                                     bin(obs[2], 0, 1, 9)) * 7 + bin(obs[3], -1, 1, 7));
    } else if (task == Task::CartPole) {
        idx = static_cast<uint32_t>(((bin(obs[0], -1.5f, 1.5f, 8) * 8 + bin(obs[1], -1.5f, 1.5f, 8)) * 8 +
                                     bin(obs[2], -1.5f, 1.5f, 8)) * 8 + bin(obs[3], -1.5f, 1.5f, 8));
    } else {
        idx = static_cast<uint32_t>(((bin(obs[0], -2, 2, 8) * 8 + bin(obs[1], -2, 2, 8)) * 4 +
                                     bin(obs[2], -1, 1, 4)) * 4 + bin(obs[3], -1, 1, 4));
    }
    return static_cast<int>(idx % kBlocks);
}

static bool solve_linear(std::vector<double>& A, std::vector<double>& B, int n, int m) {
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int r = col + 1; r < n; ++r) {
            if (std::abs(A[r * n + col]) > std::abs(A[pivot * n + col])) pivot = r;
        }
        if (std::abs(A[pivot * n + col]) < 1e-12) return false;
        if (pivot != col) {
            for (int k = 0; k < n; ++k) std::swap(A[col * n + k], A[pivot * n + k]);
            for (int k = 0; k < m; ++k) std::swap(B[col * m + k], B[pivot * m + k]);
        }
        const double d = A[col * n + col];
        for (int k = 0; k < n; ++k) A[col * n + k] /= d;
        for (int k = 0; k < m; ++k) B[col * m + k] /= d;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = A[r * n + col];
            if (std::abs(f) < 1e-18) continue;
            for (int k = 0; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            for (int k = 0; k < m; ++k) B[r * m + k] -= f * B[col * m + k];
        }
    }
    return true;
}

static void train_readout(BlockMeta& m, const std::vector<Sample*>& ss,
                          const std::vector<std::array<float, kX>>& feats, double lambda) {
    const int F = kX;
    const int N = F + 1;
    std::vector<double> A(N * N, 0.0), B(N * 2, 0.0);
    m.mean.assign(F, 0.0);
    m.std.assign(F, 0.0);
    std::vector<double> sq(F, 0.0);
    for (const auto& f : feats) {
        for (int i = 0; i < F; ++i) { m.mean[i] += f[i]; sq[i] += f[i] * f[i]; }
    }
    const double inv = feats.empty() ? 0.0 : 1.0 / feats.size();
    for (int i = 0; i < F; ++i) {
        m.mean[i] *= inv;
        const double var = std::max(0.0, sq[i] * inv - m.mean[i] * m.mean[i]);
        m.std[i] = std::sqrt(var) + 1e-6;
    }
    for (size_t k = 0; k < feats.size(); ++k) {
        double x[N];
        for (int i = 0; i < F; ++i) x[i] = (feats[k][i] - m.mean[i]) / m.std[i];
        x[F] = 1.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) A[i * N + j] += x[i] * x[j];
            B[i * 2 + 0] += x[i] * ss[k]->a0;
            B[i * 2 + 1] += x[i] * ss[k]->a1;
        }
    }
    for (int i = 0; i < N; ++i) A[i * N + i] += lambda;
    solve_linear(A, B, N, 2);
    m.w = B;
    m.samples = static_cast<int>(ss.size());
    m.ready = m.samples >= 8;
}

static void readout(const BlockMeta& m, const float* feat, const std::array<float, kRaw>& obs,
                    double* a0, double* a1) {
    if (!m.ready) { *a0 = 0; *a1 = 0; return; }
    double x[kX];
    for (uint32_t i = 0; i < kFeat; ++i) x[i] = feat[i];
    for (uint32_t i = 0; i < kRaw; ++i) x[kFeat + i] = obs[i];
    for (uint32_t i = 0; i < kX; ++i) x[i] = (x[i] - m.mean[i]) / m.std[i];
    double s0 = m.w[kX * 2 + 0], s1 = m.w[kX * 2 + 1];
    for (uint32_t i = 0; i < kX; ++i) {
        s0 += m.w[i * 2 + 0] * x[i];
        s1 += m.w[i * 2 + 1] * x[i];
    }
    *a0 = s0;
    *a1 = s1;
}

struct Cache {
    std::unordered_map<int, std::pair<SDSCBinaryGraph*, uint64_t>> map;
    uint64_t tick{0};
    std::string dir;

    SDSCBinaryGraph* get(int c) {
        auto it = map.find(c);
        if (it != map.end()) { it->second.second = ++tick; return it->second.first; }
        if (map.size() >= kCacheMax) {
            auto victim = map.begin();
            for (auto i = map.begin(); i != map.end(); ++i) {
                if (i->second.second < victim->second.second) victim = i;
            }
            sdsc_binary_free(victim->second.first);
            map.erase(victim);
        }
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%05d.bin", dir.c_str(), c);
        SDSCBinaryGraph* g = sdsc_binary_load(path);
        if (!g) return nullptr;
        map.emplace(c, std::make_pair(g, ++tick));
        return g;
    }

    void reset_all() {
        for (auto& kv : map) reset_block(kv.second.first);
    }
};

static std::string task_name(Task t) {
    switch (t) {
        case Task::Maze: return "maze";
        case Task::CartPole: return "cartpole";
        case Task::Maglev: return "maglev";
    }
    return "?";
}

static double collect_task(Task task, const std::string& champ_path, int episodes,
                           std::vector<Sample>& out) {
    CellularOrganism champ = CellularOrganism::load_checkpoint_bin(champ_path);
    if (champ.cells.empty()) return 0.0;
    champ.compile();
    int ok = 0;
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = 90000u + static_cast<uint32_t>(e) * 17u;
        std::unique_ptr<EvolvableTask> env;
        int max_steps = 250;
        if (task == Task::Maze) {
            auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
            t->set_use_geodesic_bearing(true);
            t->reset(seed);
            env.reset(t);
            max_steps = 250;
        } else if (task == Task::CartPole) {
            auto* t = new CartPoleBalanceTask();
            t->set_max_steps(300);
            t->reset(seed);
            env.reset(t);
            max_steps = 300;
        } else {
            auto* t = new ZooMaglev();
            t->set_max_steps(200);
            t->reset(seed);
            env.reset(t);
            max_steps = 200;
        }
        champ.reset_state(false);
        bool success = false;
        bool done = false;
        for (int t = 0; t < max_steps; ++t) {
            auto obs_v = env->current_observation();
            std::array<float, kRaw> obs{};
            for (uint32_t i = 0; i < kRaw; ++i) obs[i] = obs_v[i];
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = champ.forward(inps, false);
            Sample s;
            s.obs = obs;
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            s.block = grid_block(task, obs);
            out.push_back(s);
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && task != Task::Maze) success = true;
        if (success) ++ok;
    }
    return double(ok) / episodes;
}

static double eval_champion(Task task, const std::string& champ_path, int episodes) {
    CellularOrganism champ = CellularOrganism::load_checkpoint_bin(champ_path);
    if (champ.cells.empty()) return 0.0;
    champ.compile();
    int ok = 0;
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
        std::unique_ptr<EvolvableTask> env;
        int max_steps = 250;
        if (task == Task::Maze) {
            auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
            t->set_use_geodesic_bearing(true);
            t->reset(seed);
            env.reset(t);
            max_steps = 250;
        } else if (task == Task::CartPole) {
            auto* t = new CartPoleBalanceTask();
            t->set_max_steps(300);
            t->reset(seed);
            env.reset(t);
            max_steps = 300;
        } else {
            auto* t = new ZooMaglev();
            t->set_max_steps(200);
            t->reset(seed);
            env.reset(t);
            max_steps = 200;
        }
        champ.reset_state(false);
        bool success = false;
        bool done = false;
        for (int t = 0; t < max_steps; ++t) {
            auto obs_v = env->current_observation();
            double inps[4] = {obs_v[0], obs_v[1], obs_v[2], obs_v[3]};
            auto acts = champ.forward(inps, false);
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && task != Task::Maze) success = true;
        if (success) ++ok;
    }
    return double(ok) / episodes;
}

static int eval_agent(Task task, std::vector<BlockMeta>& metas, Cache& cache,
                      int episodes) {
    int ok = 0;
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
        std::unique_ptr<EvolvableTask> env;
        int max_steps = 250;
        if (task == Task::Maze) {
            auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
            t->set_use_geodesic_bearing(true);
            t->reset(seed);
            env.reset(t);
            max_steps = 250;
        } else if (task == Task::CartPole) {
            auto* t = new CartPoleBalanceTask();
            t->set_max_steps(300);
            t->reset(seed);
            env.reset(t);
            max_steps = 300;
        } else {
            auto* t = new ZooMaglev();
            t->set_max_steps(200);
            t->reset(seed);
            env.reset(t);
            max_steps = 200;
        }
        cache.reset_all();
        bool success = false;
        bool done = false;
        for (int t = 0; t < max_steps; ++t) {
            auto obs_v = env->current_observation();
            std::array<float, kRaw> obs{};
            for (uint32_t i = 0; i < kRaw; ++i) obs[i] = obs_v[i];
            std::vector<std::pair<double, int>> order;
            for (uint32_t c = 0; c < kBlocks; ++c) {
                if (!metas[c].ready) continue;
                double d = 0;
                for (uint32_t i = 0; i < kRaw; ++i) {
                    const double diff = obs[i] - metas[c].centroid[i];
                    d += diff * diff;
                }
                order.emplace_back(d, static_cast<int>(c));
            }
            std::sort(order.begin(), order.end());
            double a0 = 0, a1 = 0;
            int used = 0;
            for (int k = 0; k < kTopK && k < static_cast<int>(order.size()); ++k) {
                const int c = order[k].second;
                SDSCBinaryGraph* g = cache.get(c);
                if (!g) continue;
                float feat[kFeat];
                block_forward(g, obs, feat);
                double x0, x1;
                readout(metas[c], feat, obs, &x0, &x1);
                a0 += x0;
                a1 += x1;
                ++used;
            }
            if (used) { a0 /= used; a1 /= used; }
            CellularOrganism::ActionOutputs acts;
            acts.positive_action = a0;
            acts.negative_action = a1;
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && task != Task::Maze) success = true;
        if (success) ++ok;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "runs/moe_agent_1b_3task_" + today() + ".json";
    std::string block_dir = "/tmp/opencode/moe_blocks_1b_v2";
    int collect_episodes = 300;
    int eval_episodes = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--out") out = next();
        else if (a == "--block-dir") block_dir = next();
        else if (a == "--collect-episodes") collect_episodes = std::max(1, std::stoi(next()));
        else if (a == "--eval-episodes") eval_episodes = std::max(1, std::stoi(next()));
        else if (a == "--help") {
            std::cout << "usage: train_moe_agent_1b_3task [--out path] [--block-dir path] "
                         "[--collect-episodes N] [--eval-episodes N]\n";
            return 0;
        }
    }
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";
    std::filesystem::create_directories(block_dir);

    const Task tasks[kTasks] = {Task::Maze, Task::CartPole, Task::Maglev};
    const char* champs[kTasks] = {
        "checkpoints/maze_navigation_champion.bin",
        "checkpoints/cartpole_balance_champion.bin",
        "checkpoints/zoo_maglev.bin",
    };

    std::cout << "S1 3-task 1B MoE | blocks=" << kBlocks << " x " << kBlockCells
              << " = " << (uint64_t)kBlocks * kBlockCells << " cells\n";

    std::vector<Sample> samples[kTasks];
    std::vector<std::vector<Sample*>> per_block[kTasks];
    std::vector<BlockMeta> metas[kTasks];
    for (int ti = 0; ti < kTasks; ++ti) {
        const double csr = collect_task(tasks[ti], champs[ti], collect_episodes, samples[ti]);
        std::cout << "  " << task_name(tasks[ti]) << " samples=" << samples[ti].size()
                  << " champ_sr=" << csr << "\n";
        per_block[ti].resize(kBlocks);
        metas[ti].resize(kBlocks);
        for (auto& s : samples[ti]) per_block[ti][s.block].push_back(&s);
        for (uint32_t c = 0; c < kBlocks; ++c) {
            if (per_block[ti][c].empty()) continue;
            for (uint32_t i = 0; i < kRaw; ++i) {
                double sum = 0;
                for (auto* s : per_block[ti][c]) sum += s->obs[i];
                metas[ti][c].centroid[i] = sum / per_block[ti][c].size();
            }
        }
    }

    Cache cache;
    cache.dir = block_dir;
    for (uint32_t c = 0; c < kBlocks; ++c) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%05u.bin", block_dir.c_str(), c);
        if (!std::filesystem::exists(path)) {
            std::cerr << "missing block " << path << "\n";
            return 2;
        }
        SDSCBinaryGraph* g = sdsc_binary_load(path);
        if (!g) { std::cerr << "cannot load " << path << "\n"; return 2; }
        for (int ti = 0; ti < kTasks; ++ti) {
            std::vector<std::array<float, kX>> feats;
            feats.reserve(per_block[ti][c].size());
            for (auto* s : per_block[ti][c]) {
                reset_block(g);
                std::array<float, kFeat> f{};
                block_forward(g, s->obs, f.data());
                std::array<float, kX> x{};
                for (uint32_t i = 0; i < kFeat; ++i) x[i] = f[i];
                for (uint32_t i = 0; i < kRaw; ++i) x[kFeat + i] = s->obs[i];
                feats.push_back(x);
            }
            train_readout(metas[ti][c], per_block[ti][c], feats, 1e-3);
        }
        sdsc_binary_free(g);
        if ((c + 1) % 100 == 0) std::cout << "  trained " << (c + 1) << "/" << kBlocks << "\n";
    }

    std::ofstream os(out);
    if (!os) { std::cerr << "cannot write " << out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"S1 3-task shared 1B reservoir + per-task readouts\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"blocks\":" << kBlocks << ",\"block_cells\":" << kBlockCells
       << ",\"total_cells\":" << (uint64_t)kBlocks * kBlockCells << ",\n";
    os << "  \"topk\":" << kTopK << ",\"active_cells\":" << kTopK * kBlockCells << ",\n";
    os << "  \"tasks\":[";

    bool pass_all = true;
    for (int ti = 0; ti < kTasks; ++ti) {
        const double champ_sr = eval_champion(tasks[ti], champs[ti], eval_episodes);
        const int agent_ok = eval_agent(tasks[ti], metas[ti], cache, eval_episodes);
        const double agent_sr = eval_episodes ? double(agent_ok) / eval_episodes : 0.0;
        const bool ok = agent_sr >= champ_sr - 0.05;
        pass_all = pass_all && ok;
        if (ti) os << ",";
        os << "\n    {\"task\":\"" << task_name(tasks[ti])
           << "\",\"champion_sr\":" << champ_sr
           << ",\"agent_sr\":" << agent_sr
           << ",\"delta\":" << (agent_sr - champ_sr)
           << ",\"pass\":" << (ok ? "true" : "false") << "}";
        std::cout << "  " << task_name(tasks[ti]) << ": champion=" << champ_sr
                  << " agent=" << agent_sr << " delta=" << (agent_sr - champ_sr)
                  << " " << (ok ? "PASS" : "FAIL") << "\n";
    }
    os << "\n  ],\n  \"gate\":{\"S1_3TASK\":\"" << (pass_all ? "PASS" : "FAIL") << "\"}\n}\n";
    os.close();
    std::cout << "GATE: S1_3TASK " << (pass_all ? "PASS" : "FAIL") << "\n";
    std::cout << "ARTIFACT: " << out << "\n";
    return 0;
}
