// ============================================================================
// train_from_teacher.cpp — 可学习架构：新任务 = task adapter + teacher + 启动训练
//
// 契约：
//   1) TaskSpec：环境工厂 + 成功语义 + teacher checkpoint；
//   2) Teacher：任意 CellularOrganism checkpoint（obs -> pos/neg action）；
//   3) Pipeline：采 teacher 轨迹 -> k-means 路由 -> 每块 ridge 读出 ->
//      每任务 combiner -> held-out 评测；
//   4) 新任务只需在 registry 加一条 TaskSpec，不改底座、不改训练管线。
//
// 默认 100 blocks × 100k = 10M 总参数；top-1 = 1% active；combiner top-5。
// 输出：runs/distill_<task>_<date>.json
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

constexpr uint32_t kIn = 64;
constexpr uint32_t kFeat = 64;
constexpr uint32_t kRaw = 4;

struct TaskSpec {
    std::string name;
    std::string teacher;
    int max_steps;
    std::unique_ptr<EvolvableTask> (*make)(uint32_t seed);
};

static std::unique_ptr<EvolvableTask> make_maze(uint32_t seed) {
    auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
    t->set_use_geodesic_bearing(true);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_cartpole(uint32_t seed) {
    auto* t = new CartPoleBalanceTask();
    t->set_max_steps(300);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_maglev(uint32_t seed) {
    auto* t = new ZooMaglev();
    t->set_max_steps(200);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_rocket(uint32_t seed) {
    auto* t = new ZooRocketHover();
    t->set_max_steps(200);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_bicycle(uint32_t seed) {
    auto* t = new ZooBicycle();
    t->set_max_steps(250);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_thermal(uint32_t seed) {
    auto* t = new ZooThermal();
    t->set_max_steps(200);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_cruise(uint32_t seed) {
    auto* t = new ZooCruise();
    t->set_max_steps(200);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}
static std::unique_ptr<EvolvableTask> make_water(uint32_t seed) {
    auto* t = new ZooWaterTank();
    t->set_max_steps(200);
    t->reset(seed);
    return std::unique_ptr<EvolvableTask>(t);
}

static std::vector<TaskSpec> registry() {
    return {
        {"maze", "checkpoints/maze_navigation_champion.bin", 250, make_maze},
        {"cartpole", "checkpoints/cartpole_balance_champion.bin", 300, make_cartpole},
        {"maglev", "checkpoints/zoo_maglev.bin", 200, make_maglev},
        {"rocket", "checkpoints/zoo_rocket_hover.bin", 200, make_rocket},
        {"bicycle", "checkpoints/zoo_bicycle.bin", 250, make_bicycle},
        {"thermal", "checkpoints/zoo_thermal.bin", 200, make_thermal},
        {"cruise", "checkpoints/zoo_cruise.bin", 200, make_cruise},
        {"water", "checkpoints/zoo_water_tank.bin", 200, make_water},
    };
}

struct Sample {
    std::array<float, kRaw> obs{};
    double a0{0.0}, a1{0.0};
    int cluster{-1};
};

struct BlockMeta {
    std::vector<double> mean, std, w;
    int samples{0};
    bool ready{false};
};

struct Combiner {
    int dim{0};
    std::vector<double> mean, std, w;
    void init(int d) { dim = d; mean.assign(d, 0); std.assign(d, 0); w.assign((d + 1) * 2, 0); }
    double out(const double* x, int j) const {
        double s = w[dim * 2 + j];
        for (int i = 0; i < dim; ++i) s += w[i * 2 + j] * (x[i] - mean[i]) / (std[i] + 1e-6);
        return s;
    }
};

static std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
    return buf;
}

static std::string resolve(const std::string& p) {
    if (std::filesystem::exists(p)) return p;
    for (const char* pre : {"../", "../../", "../../../"}) {
        std::string q = std::string(pre) + p;
        if (std::filesystem::exists(q)) return q;
    }
    return p;
}

static CompactSoAGenome make_block(uint32_t cells, uint32_t syns, uint32_t seed) {
    auto g = CompactSoAGenome::create_empty(cells, syns, kIn, kFeat);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> gain(0.5, 1.5);
    std::normal_distribution<double> wd(0.0, 0.25);
    const uint8_t ops[] = {
        SDSC_OP_DAMPER, SDSC_OP_INTEGRATE, SDSC_OP_DIFF, SDSC_OP_SUM,
        SDSC_OP_MULTIPLY, SDSC_OP_CLIP, SDSC_OP_ABS,
    };
    for (uint32_t i = 0; i < cells; ++i) {
        g.op_types[i] = (i < kIn) ? SDSC_OP_PASSTHRU : ops[rng() % sizeof(ops)];
        g.gains[i] = static_cast<float>(gain(rng));
        g.biases[i] = 0.0f;
    }
    for (uint32_t i = cells - kFeat; i < cells; ++i) {
        g.op_types[i] = SDSC_OP_SUM;
        g.gains[i] = 1.0f;
    }
    uint32_t cur = 0;
    g.inc_off[0] = 0;
    for (uint32_t i = 0; i < kIn; ++i) g.inc_off[i + 1] = 0;
    for (uint32_t i = kIn; i < cells; ++i) {
        g.inc_off[i] = cur;
        const uint32_t rc = cells - i;
        const uint32_t re = syns - cur;
        uint32_t edges = rc ? re / rc : 0;
        if (edges == 0) edges = 1;
        if (cur + edges > syns) edges = syns - cur;
        for (uint32_t e = 0; e < edges; ++e) {
            uint32_t src = static_cast<uint32_t>(rng() % i);
            g.inc_from[cur] = src;
            g.inc_weight[cur] = static_cast<float>((src < kIn) ? (0.8 + 0.4 * gain(rng)) : wd(rng));
            ++cur;
        }
        g.inc_off[i + 1] = cur;
    }
    g.inc_off[cells] = cur;
    for (uint32_t k = cur; k < syns; ++k) { g.inc_from[k] = 0; g.inc_weight[k] = 0.0f; }
    if (cur != syns) g.inc_off[cells] = syns;
    for (uint32_t k = 0; k < kIn; ++k) g.in_cell_ids[k] = k;
    for (uint32_t k = 0; k < kFeat; ++k) g.out_cell_ids[k] = cells - kFeat + k;
    return g;
}

static void reset_block(SDSCBinaryGraph* g) {
    const size_t n = g->header.num_cells;
    std::memset(g->states, 0, n * 4);
    std::memset(g->aux_states, 0, n * 4);
    std::memset(g->outputs, 0, n * 4);
    std::memset(g->inputs_accum, 0, n * 4);
}

static void block_forward(SDSCBinaryGraph* g, const std::array<float, kRaw>& obs, float* out) {
    float in[kIn];
    for (uint32_t i = 0; i < kIn; ++i) in[i] = obs[i % kRaw] * ((i & 1u) ? -1.0f : 1.0f);
    in[kIn - 1] = 1.0f;
    sdsc_binary_forward(g, in, out);
}

static bool solve(std::vector<double>& A, std::vector<double>& B, int n, int m) {
    for (int col = 0; col < n; ++col) {
        int p = col;
        for (int r = col + 1; r < n; ++r) if (std::abs(A[r * n + col]) > std::abs(A[p * n + col])) p = r;
        if (std::abs(A[p * n + col]) < 1e-12) return false;
        if (p != col) {
            for (int k = 0; k < n; ++k) std::swap(A[col * n + k], A[p * n + k]);
            for (int k = 0; k < m; ++k) std::swap(B[col * m + k], B[p * m + k]);
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

static int nearest(const std::vector<std::array<double, kRaw>>& cent,
                   const std::array<float, kRaw>& obs) {
    int best = 0;
    double bd = 1e18;
    for (size_t c = 0; c < cent.size(); ++c) {
        double d = 0;
        for (uint32_t i = 0; i < kRaw; ++i) {
            const double diff = obs[i] - cent[c][i];
            d += diff * diff;
        }
        if (d < bd) { bd = d; best = static_cast<int>(c); }
    }
    return best;
}

static void train_readout(BlockMeta& m, const std::vector<Sample*>& ss,
                          const std::vector<std::array<float, kFeat>>& fs, double lambda) {
    const int F = kFeat;
    const int N = F + 1;
    std::vector<double> A(N * N, 0.0), B(N * 2, 0.0), sq(F, 0.0);
    m.mean.assign(F, 0.0); m.std.assign(F, 0.0);
    for (const auto& f : fs) for (int i = 0; i < F; ++i) { m.mean[i] += f[i]; sq[i] += f[i] * f[i]; }
    const double inv = fs.empty() ? 0.0 : 1.0 / fs.size();
    for (int i = 0; i < F; ++i) {
        m.mean[i] *= inv;
        m.std[i] = std::sqrt(std::max(0.0, sq[i] * inv - m.mean[i] * m.mean[i])) + 1e-6;
    }
    for (size_t k = 0; k < fs.size(); ++k) {
        double x[N];
        for (int i = 0; i < F; ++i) x[i] = (fs[k][i] - m.mean[i]) / m.std[i];
        x[F] = 1.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) A[i * N + j] += x[i] * x[j];
            B[i * 2 + 0] += x[i] * ss[k]->a0;
            B[i * 2 + 1] += x[i] * ss[k]->a1;
        }
    }
    for (int i = 0; i < N; ++i) A[i * N + i] += lambda;
    solve(A, B, N, 2);
    m.w = B; m.samples = static_cast<int>(ss.size()); m.ready = m.samples >= 8;
}

static void block_readout(const BlockMeta& m, const float* f, double* a0, double* a1) {
    if (!m.ready) { *a0 = 0; *a1 = 0; return; }
    double x[kFeat];
    for (uint32_t i = 0; i < kFeat; ++i) x[i] = (f[i] - m.mean[i]) / m.std[i];
    double s0 = m.w[kFeat * 2 + 0], s1 = m.w[kFeat * 2 + 1];
    for (uint32_t i = 0; i < kFeat; ++i) { s0 += m.w[i * 2 + 0] * x[i]; s1 += m.w[i * 2 + 1] * x[i]; }
    *a0 = s0; *a1 = s1;
}

struct BlockCache {
    std::unordered_map<int, SDSCBinaryGraph*> map;
    std::string dir;
    ~BlockCache() { for (auto& kv : map) sdsc_binary_free(kv.second); }
    SDSCBinaryGraph* get(int c) {
        auto it = map.find(c);
        if (it != map.end()) return it->second;
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%04d.bin", dir.c_str(), c);
        SDSCBinaryGraph* g = sdsc_binary_load(path);
        if (g) map[c] = g;
        return g;
    }
    void reset() { for (auto& kv : map) reset_block(kv.second); }
};

static std::vector<int> top_blocks(const std::vector<std::array<double, kRaw>>& cent,
                                   const std::vector<BlockMeta>& metas,
                                   const std::array<float, kRaw>& obs, int k) {
    std::vector<std::pair<double, int>> order;
    for (size_t c = 0; c < cent.size(); ++c) {
        if (!metas[c].ready) continue;
        double d = 0;
        for (uint32_t i = 0; i < kRaw; ++i) { const double diff = obs[i] - cent[c][i]; d += diff * diff; }
        order.emplace_back(d, static_cast<int>(c));
    }
    std::sort(order.begin(), order.end());
    std::vector<int> ids;
    for (int i = 0; i < k && i < static_cast<int>(order.size()); ++i) ids.push_back(order[i].second);
    return ids;
}

static void features(BlockCache& cache, const std::vector<BlockMeta>& metas,
                     const std::vector<int>& ids, const std::array<float, kRaw>& obs, double* x) {
    for (size_t k = 0; k < ids.size(); ++k) {
        SDSCBinaryGraph* g = cache.get(ids[k]);
        double a0 = 0, a1 = 0;
        if (g) {
            float f[kFeat];
            block_forward(g, obs, f);
            block_readout(metas[ids[k]], f, &a0, &a1);
        }
        x[2 * k + 0] = a0;
        x[2 * k + 1] = a1;
    }
    for (uint32_t i = 0; i < kRaw; ++i) x[2 * ids.size() + i] = obs[i];
}

static bool episode(TaskSpec& spec, BlockCache& cache,
                    const std::vector<std::array<double, kRaw>>& cent,
                    const std::vector<BlockMeta>& metas, const Combiner& comb,
                    int topk, uint32_t seed) {
    auto env = spec.make(seed);
    cache.reset();
    for (int t = 0; t < spec.max_steps; ++t) {
        auto o = env->current_observation();
        std::array<float, kRaw> obs{};
        for (uint32_t i = 0; i < kRaw; ++i) obs[i] = o[i];
        auto ids = top_blocks(cent, metas, obs, topk);
        std::vector<double> x(2 * ids.size() + kRaw, 0.0);
        features(cache, metas, ids, obs, x.data());
        CellularOrganism::ActionOutputs acts;
        acts.positive_action = comb.out(x.data(), 0);
        acts.negative_action = comb.out(x.data(), 1);
        auto res = env->step_continuous(acts);
        if (res.done) return res.success;
    }
    return spec.name != "maze";
}

static double eval_teacher(TaskSpec& spec, uint32_t seed0, int episodes) {
    CellularOrganism t = CellularOrganism::load_checkpoint_bin(resolve(spec.teacher));
    if (t.cells.empty()) return 0.0;
    t.compile();
    int ok = 0;
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = seed0 + static_cast<uint32_t>(e) * 17u;
        auto env = spec.make(seed);
        t.reset_state(false);
        bool success = false, done = false;
        for (int k = 0; k < spec.max_steps; ++k) {
            auto o = env->current_observation();
            double in[4] = {o[0], o[1], o[2], o[3]};
            auto acts = t.forward(in, false);
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && spec.name != "maze") success = true;
        if (success) ++ok;
    }
    return double(ok) / episodes;
}

}  // namespace

int main(int argc, char** argv) {
    std::string task = "rocket";
    std::string out;
    std::string block_dir = "/tmp/opencode/distill_blocks";
    int blocks = 100;
    int cells = 100000;
    int syns = 300000;
    int teacher_episodes = 100;
    int eval_episodes = 100;
    int topk = 1;
    int combiner_topk = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--task") task = next();
        else if (a == "--out") out = next();
        else if (a == "--block-dir") block_dir = next();
        else if (a == "--blocks") blocks = std::stoi(next());
        else if (a == "--cells") cells = std::stoi(next());
        else if (a == "--syns") syns = std::stoi(next());
        else if (a == "--teacher-episodes") teacher_episodes = std::stoi(next());
        else if (a == "--eval-episodes") eval_episodes = std::stoi(next());
        else if (a == "--topk") topk = std::stoi(next());
        else if (a == "--combiner-topk") combiner_topk = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: train_from_teacher --task <name> [--blocks 100] [--cells 100000] "
                         "[--syns 300000] [--teacher-episodes 100] [--eval-episodes 100] "
                         "[--topk 1] [--combiner-topk 5] [--out path]\n";
            return 0;
        }
    }
    auto reg = registry();
    TaskSpec* spec = nullptr;
    for (auto& s : reg) if (s.name == task) spec = &s;
    if (!spec) { std::cerr << "unknown task " << task << "\n"; return 2; }
    if (out.empty()) out = "runs/distill_" + task + "_" + today() + ".json";
    std::filesystem::create_directories(block_dir);

    std::cout << "Learnable distillation | task=" << task
              << " teacher=" << spec->teacher << " blocks=" << blocks
              << " x " << cells << " = " << (int64_t)blocks * cells << " cells\n";

    // 1) teacher trajectories
    CellularOrganism teacher = CellularOrganism::load_checkpoint_bin(resolve(spec->teacher));
    if (teacher.cells.empty()) { std::cerr << "cannot load teacher\n"; return 2; }
    teacher.compile();
    std::vector<Sample> samples;
    int teacher_ok = 0;
    for (int e = 0; e < teacher_episodes; ++e) {
        const uint32_t seed = 90000u + static_cast<uint32_t>(e) * 17u;
        auto env = spec->make(seed);
        teacher.reset_state(false);
        bool success = false, done = false;
        for (int k = 0; k < spec->max_steps; ++k) {
            auto o = env->current_observation();
            double in[4] = {o[0], o[1], o[2], o[3]};
            auto acts = teacher.forward(in, false);
            Sample s;
            for (uint32_t i = 0; i < kRaw; ++i) s.obs[i] = o[i];
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            samples.push_back(s);
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && spec->name != "maze") success = true;
        if (success) ++teacher_ok;
    }
    const double teacher_sr = double(teacher_ok) / teacher_episodes;
    std::cout << "  teacher samples=" << samples.size() << " sr=" << teacher_sr << "\n";

    // 2) k-means router
    std::vector<std::array<double, kRaw>> cent(blocks);
    std::mt19937 rng(20261124);
    for (int c = 0; c < blocks; ++c) {
        const auto& s = samples[rng() % samples.size()];
        for (uint32_t i = 0; i < kRaw; ++i) cent[c][i] = s.obs[i];
    }
    for (int it = 0; it < 15; ++it) {
        for (auto& s : samples) s.cluster = nearest(cent, s.obs);
        std::vector<std::array<double, kRaw>> sum(blocks);
        std::vector<int> cnt(blocks, 0);
        for (auto& s : samples) {
            for (uint32_t i = 0; i < kRaw; ++i) sum[s.cluster][i] += s.obs[i];
            ++cnt[s.cluster];
        }
        for (int c = 0; c < blocks; ++c) if (cnt[c]) for (uint32_t i = 0; i < kRaw; ++i) cent[c][i] = sum[c][i] / cnt[c];
    }
    for (auto& s : samples) s.cluster = nearest(cent, s.obs);

    // 3) blocks + per-block readouts
    std::vector<std::vector<Sample*>> per(blocks);
    for (auto& s : samples) per[s.cluster].push_back(&s);
    std::vector<BlockMeta> metas(blocks);
    for (int c = 0; c < blocks; ++c) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%04d.bin", block_dir.c_str(), c);
        if (!std::filesystem::exists(path)) {
            CompactSoAGenome g = make_block(cells, syns, 20261124u + c * 7919u);
            g.save_binary(path);
        }
        SDSCBinaryGraph* g = sdsc_binary_load(path);
        if (!g) { std::cerr << "cannot load " << path << "\n"; return 2; }
        std::vector<std::array<float, kFeat>> fs;
        for (auto* s : per[c]) {
            reset_block(g);
            std::array<float, kFeat> f{};
            block_forward(g, s->obs, f.data());
            fs.push_back(f);
        }
        train_readout(metas[c], per[c], fs, 1e-3);
        sdsc_binary_free(g);
    }
    int ready = 0;
    for (auto& m : metas) if (m.ready) ++ready;

    // 4) combiner
    BlockCache cache;
    cache.dir = block_dir;
    Combiner comb;
    const int F = 2 * combiner_topk + kRaw;
    comb.init(F);
    cache.reset();
    std::vector<std::vector<double>> feats;
    for (auto& s : samples) {
        auto ids = top_blocks(cent, metas, s.obs, combiner_topk);
        std::vector<double> x(F, 0.0);
        features(cache, metas, ids, s.obs, x.data());
        feats.push_back(std::move(x));
    }
    std::vector<double> sq(F, 0.0);
    for (auto& x : feats) for (int i = 0; i < F; ++i) { comb.mean[i] += x[i]; sq[i] += x[i] * x[i]; }
    const double inv = feats.empty() ? 0.0 : 1.0 / feats.size();
    for (int i = 0; i < F; ++i) {
        comb.mean[i] *= inv;
        comb.std[i] = std::sqrt(std::max(0.0, sq[i] * inv - comb.mean[i] * comb.mean[i])) + 1e-6;
    }
    {
        const int N = F + 1;
        std::vector<double> A(N * N, 0.0), B(N * 2, 0.0);
        for (size_t k = 0; k < feats.size(); ++k) {
            double x[N];
            for (int i = 0; i < F; ++i) x[i] = (feats[k][i] - comb.mean[i]) / comb.std[i];
            x[F] = 1.0;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) A[i * N + j] += x[i] * x[j];
                B[i * 2 + 0] += x[i] * samples[k].a0;
                B[i * 2 + 1] += x[i] * samples[k].a1;
            }
        }
        for (int i = 0; i < N; ++i) A[i * N + i] += 1e-3;
        solve(A, B, N, 2);
        comb.w = B;
    }

    // 5) held-out eval
    int ok = 0;
    for (int e = 0; e < eval_episodes; ++e) {
        const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
        if (episode(*spec, cache, cent, metas, comb, topk, seed)) ++ok;
    }
    const double agent_sr = double(ok) / eval_episodes;
    const double champ_sr = eval_teacher(*spec, 93000u, eval_episodes);
    const bool pass = agent_sr >= champ_sr - 0.05;

    std::ofstream os(out);
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"learnable teacher distillation\",\n";
    os << "  \"task\":\"" << task << "\",\"teacher\":\"" << spec->teacher << "\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"blocks\":" << blocks << ",\"cells\":" << (int64_t)blocks * cells
       << ",\"active_cells\":" << (int64_t)topk * cells
       << ",\"active_pct\":" << (100.0 * topk / blocks) << ",\n";
    os << "  \"teacher_sr\":" << teacher_sr << ",\"champion_sr\":" << champ_sr
       << ",\"agent_sr\":" << agent_sr << ",\"delta\":" << (agent_sr - champ_sr) << ",\n";
    os << "  \"gate\":{\"DISTILL\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n}\n";
    os.close();

    std::cout << "  ready blocks=" << ready << "/" << blocks
              << "  teacher_sr=" << teacher_sr << " champion_sr=" << champ_sr
              << " agent_sr=" << agent_sr << "\n";
    std::cout << "GATE: DISTILL " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "ARTIFACT: " << out << "\n";
    return pass ? 0 : 1;
}
