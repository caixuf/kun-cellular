// ============================================================================
// train_moe_agent.cpp — S2 条件计算：10M 总参数 / 1% active 的 top-k 分块 MoE
//
// 架构：
//   100 个 block，每块 100k cells / 300k syns（合计 10M / 30M）；
//   每块是一个随机稀疏 DAG 储层（64 in / 64 features）；
//   路由器：k-means(100) on 4 维 maze 观测；
//   每步只跑 top-k 个 block（top-1 = 1% active）；
//   每块有自己的 ridge 读出（64 -> 2），用冠军蒸馏训练。
//
// 门禁（预注册）：
//   top-1 (1% active) Maze SR >= 0.90 * full-10M SR（full=95% → >=85.5%）
//   并给出 k=1/2/5/10 的 active 扩展曲线。
//
// 输出：runs/moe_agent_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_compact_genome.hpp"
#include "kun/cellular/sdsc_binary_runtime.h"
#include "tasks/robotics/maze_navigator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace kun;

namespace {

constexpr uint32_t kBlocks = 100;
constexpr uint32_t kBlockCells = 100'000;
constexpr uint32_t kBlockSyns = 300'000;
constexpr uint32_t kIn = 64;
constexpr uint32_t kFeat = 64;
constexpr uint32_t kRaw = 4;

struct Sample {
    std::array<float, kRaw> obs{};
    double a0{0.0};
    double a1{0.0};
    int cluster{-1};
    std::array<float, kFeat> feat{};
};

struct Block {
    SDSCBinaryGraph* g{nullptr};
    std::vector<double> mean, std, w;  // (kFeat+1) x 2
    int samples{0};
    bool ready{false};
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

// 与 train_scaled_agent 相同的 block 生成器（100k 规模）。
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
            g.inc_weight[cursor] = static_cast<float>((src < kIn) ? (0.8 + 0.4 * gain(rng)) : wdist(rng));
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
    for (uint32_t i = 0; i < kIn; ++i) {
        in[i] = obs[i % kRaw] * ((i & 1u) ? -1.0f : 1.0f);
    }
    in[kIn - 1] = 1.0f;
    sdsc_binary_forward(g, in, out);
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

static void train_block_readout(Block& b, const std::vector<Sample*>& ss, double lambda) {
    const int F = kFeat;
    const int N = F + 1;
    std::vector<double> A(N * N, 0.0), B(N * 2, 0.0);
    b.mean.assign(F, 0.0);
    b.std.assign(F, 0.0);
    std::vector<double> sq(F, 0.0);
    for (const auto* s : ss) {
        for (int i = 0; i < F; ++i) { b.mean[i] += s->feat[i]; sq[i] += s->feat[i] * s->feat[i]; }
    }
    const double inv = ss.empty() ? 0.0 : 1.0 / ss.size();
    for (int i = 0; i < F; ++i) {
        b.mean[i] *= inv;
        const double var = std::max(0.0, sq[i] * inv - b.mean[i] * b.mean[i]);
        b.std[i] = std::sqrt(var) + 1e-6;
    }
    for (const auto* s : ss) {
        double x[N];
        for (int i = 0; i < F; ++i) x[i] = (s->feat[i] - b.mean[i]) / b.std[i];
        x[F] = 1.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) A[i * N + j] += x[i] * x[j];
            B[i * 2 + 0] += x[i] * s->a0;
            B[i * 2 + 1] += x[i] * s->a1;
        }
    }
    for (int i = 0; i < N; ++i) A[i * N + i] += lambda;
    solve_linear(A, B, N, 2);
    b.w = B;
    b.samples = static_cast<int>(ss.size());
    b.ready = b.samples >= 8;
}

static void readout(const Block& b, const float* feat, double* a0, double* a1) {
    if (!b.ready) { *a0 = 0; *a1 = 0; return; }
    double x[kFeat];
    for (uint32_t i = 0; i < kFeat; ++i) x[i] = (feat[i] - b.mean[i]) / b.std[i];
    double s0 = b.w[kFeat * 2 + 0], s1 = b.w[kFeat * 2 + 1];
    for (uint32_t i = 0; i < kFeat; ++i) {
        s0 += b.w[i * 2 + 0] * x[i];
        s1 += b.w[i * 2 + 1] * x[i];
    }
    *a0 = s0;
    *a1 = s1;
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

static void kmeans(std::vector<Sample>& samples,
                   std::vector<std::array<double, kRaw>>& cent, int iters) {
    std::mt19937 rng(20261123);
    cent.clear();
    for (uint32_t c = 0; c < kBlocks; ++c) {
        const auto& s = samples[rng() % samples.size()];
        std::array<double, kRaw> m{};
        for (uint32_t i = 0; i < kRaw; ++i) m[i] = s.obs[i];
        cent.push_back(m);
    }
    for (int it = 0; it < iters; ++it) {
        for (auto& s : samples) s.cluster = nearest(cent, s.obs);
        std::vector<std::array<double, kRaw>> sum(kBlocks);
        std::vector<int> cnt(kBlocks, 0);
        for (auto& s : samples) {
            for (uint32_t i = 0; i < kRaw; ++i) sum[s.cluster][i] += s.obs[i];
            cnt[s.cluster]++;
        }
        for (uint32_t c = 0; c < kBlocks; ++c) {
            if (cnt[c] == 0) continue;
            for (uint32_t i = 0; i < kRaw; ++i) cent[c][i] = sum[c][i] / cnt[c];
        }
    }
    for (auto& s : samples) s.cluster = nearest(cent, s.obs);
}

static bool run_maze_episode(const std::vector<Block>& blocks,
                             const std::vector<std::array<double, kRaw>>& cent,
                             int topk, uint32_t seed, int* steps_out) {
    MazeTask task(11, 11, seed, 250, 0.15f);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    for (const auto& b : blocks) reset_block(b.g);

    for (int t = 0; t < 250; ++t) {
        auto obs_v = task.current_observation();
        std::array<float, kRaw> obs{};
        for (uint32_t i = 0; i < kRaw; ++i) obs[i] = obs_v[i];
        const int c0 = nearest(cent, obs);

        std::vector<std::pair<double, int>> order;
        for (uint32_t c = 0; c < kBlocks; ++c) {
            if (!blocks[c].ready) continue;
            double d = 0;
            for (uint32_t i = 0; i < kRaw; ++i) {
                const double diff = obs[i] - cent[c][i];
                d += diff * diff;
            }
            order.emplace_back(d, static_cast<int>(c));
        }
        std::sort(order.begin(), order.end());
        (void)c0;

        double a0 = 0, a1 = 0;
        int used = 0;
        for (int k = 0; k < topk && k < static_cast<int>(order.size()); ++k) {
            const int c = order[k].second;
            if (!blocks[c].ready) continue;
            float feat[kFeat];
            block_forward(blocks[c].g, obs, feat);
            double x0, x1;
            readout(blocks[c], feat, &x0, &x1);
            a0 += x0;
            a1 += x1;
            ++used;
        }
        if (used > 0) { a0 /= used; a1 /= used; }

        CellularOrganism::ActionOutputs acts;
        acts.positive_action = a0;
        acts.negative_action = a1;
        auto res = task.step_continuous(acts);
        if (res.done) {
            if (steps_out) *steps_out = res.steps;
            return res.success;
        }
    }
    if (steps_out) *steps_out = 250;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "runs/moe_agent_" + today() + ".json";
    std::string block_dir = "/tmp/opencode/moe_blocks";
    int collect_episodes = 100;
    int eval_episodes = 20;
    std::vector<int> ks{1, 2, 5, 10};
    double full_sr = 0.95;  // 10M full reference (20 episodes)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--out") out = next();
        else if (a == "--block-dir") block_dir = next();
        else if (a == "--collect-episodes") collect_episodes = std::max(1, std::stoi(next()));
        else if (a == "--eval-episodes") eval_episodes = std::max(1, std::stoi(next()));
        else if (a == "--full-sr") full_sr = std::stod(next());
        else if (a == "--help") {
            std::cout << "usage: train_moe_agent [--out path] [--collect-episodes N] "
                         "[--eval-episodes N] [--full-sr 0.95]\n";
            return 0;
        }
    }
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    std::filesystem::create_directories(block_dir);
    std::cout << "S2 MoE | blocks=" << kBlocks << " x " << kBlockCells
              << " cells = " << (uint64_t)kBlocks * kBlockCells << " cells\n";

    // 1) 生成/加载 100 个 block
    std::vector<Block> blocks(kBlocks);
    for (uint32_t c = 0; c < kBlocks; ++c) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%03u.bin", block_dir.c_str(), c);
        if (!std::filesystem::exists(path)) {
            CompactSoAGenome g = make_block(kBlockCells, kBlockSyns, 20261123u + c * 7919u);
            g.save_binary(path);
        }
        blocks[c].g = sdsc_binary_load(path);
        if (!blocks[c].g) { std::cerr << "cannot load " << path << "\n"; return 2; }
    }
    std::cout << "  blocks loaded; total " << (uint64_t)kBlocks * kBlockCells
              << " cells / " << (uint64_t)kBlocks * kBlockSyns << " syns\n";

    // 2) 冠军采集 + k-means 路由
    CellularOrganism champ = CellularOrganism::load_checkpoint_bin(
        "checkpoints/maze_navigation_champion.bin");
    champ.compile();
    std::vector<Sample> samples;
    int champ_ok = 0;
    for (int e = 0; e < collect_episodes; ++e) {
        const uint32_t seed = 90000u + static_cast<uint32_t>(e) * 17u;
        MazeTask task(11, 11, seed, 250, 0.15f);
        task.set_use_geodesic_bearing(true);
        task.reset(seed);
        champ.reset_state(false);
        for (int t = 0; t < 250; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = champ.forward(inps, false);
            Sample s;
            for (uint32_t i = 0; i < kRaw; ++i) s.obs[i] = obs[i];
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            samples.push_back(s);
            auto res = task.step_continuous(acts);
            if (res.done) { if (res.success) ++champ_ok; break; }
        }
    }
    std::cout << "  collected " << samples.size() << " champion samples, champ_sr="
              << double(champ_ok) / collect_episodes << "\n";

    std::vector<std::array<double, kRaw>> cent;
    kmeans(samples, cent, 20);

    // 3) 每块特征 + 读出
    std::vector<std::vector<Sample*>> per_block(kBlocks);
    for (auto& s : samples) per_block[s.cluster].push_back(&s);

    for (uint32_t c = 0; c < kBlocks; ++c) {
        // 采集该块特征（按样本顺序）
        for (auto* s : per_block[c]) {
            float feat[kFeat];
            reset_block(blocks[c].g);
            block_forward(blocks[c].g, s->obs, feat);
            for (uint32_t i = 0; i < kFeat; ++i) s->feat[i] = feat[i];
        }
        train_block_readout(blocks[c], per_block[c], 1e-3);
    }
    int ready_blocks = 0;
    for (const auto& b : blocks) if (b.ready) ++ready_blocks;
    std::cout << "  ready blocks: " << ready_blocks << "/" << kBlocks << "\n";

    // 4) 评测 top-k
    std::vector<std::pair<int, double>> curve;
    for (int k : ks) {
        int ok = 0;
        for (int e = 0; e < eval_episodes; ++e) {
            const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
            int steps = 0;
            if (run_maze_episode(blocks, cent, k, seed, &steps)) ++ok;
        }
        const double sr = double(ok) / eval_episodes;
        curve.emplace_back(k, sr);
        std::cout << "  top-" << k << " active=" << (100.0 * k / kBlocks)
                  << "% SR=" << sr << "\n";
    }
    const double top1 = curve.empty() ? 0.0 : curve[0].second;
    const double threshold = 0.90 * full_sr;
    const bool pass = top1 >= threshold;

    std::ofstream os(out);
    if (!os) { std::cerr << "cannot write " << out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"S2 conditional MoE agent (10M total / 1% active)\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"blocks\":" << kBlocks << ",\"block_cells\":" << kBlockCells
       << ",\"block_syns\":" << kBlockSyns
       << ",\"total_cells\":" << (uint64_t)kBlocks * kBlockCells
       << ",\"total_syns\":" << (uint64_t)kBlocks * kBlockSyns << ",\n";
    os << "  \"samples\":" << samples.size() << ",\"champion_sr\":"
       << (double(champ_ok) / collect_episodes) << ",\n";
    os << "  \"ready_blocks\":" << ready_blocks << ",\n";
    os << "  \"curve\":[";
    for (size_t i = 0; i < curve.size(); ++i) {
        if (i) os << ",";
        os << "{\"top_k\":" << curve[i].first
           << ",\"active_pct\":" << (100.0 * curve[i].first / kBlocks)
           << ",\"sr\":" << curve[i].second << "}";
    }
    os << "],\n  \"gate\":{\"S2\":\"" << (pass ? "PASS" : "FAIL")
       << "\",\"top1_sr\":" << top1
       << ",\"full_sr\":" << full_sr
       << ",\"threshold\":" << threshold << "}\n}\n";
    os.close();

    std::cout << "GATE: S2 " << (pass ? "PASS" : "FAIL")
              << " (top1=" << top1 << " threshold=" << threshold
              << " full=" << full_sr << ")\n";
    std::cout << "ARTIFACT: " << out << "\n";
    return 0;
}
