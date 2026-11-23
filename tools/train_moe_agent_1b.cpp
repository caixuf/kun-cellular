// ============================================================================
// train_moe_agent_1b.cpp — S4 1B 流式条件 MoE 探针
//
// 目标：1,000,000,000 cells / 3,000,000,000 syns，分 5000 个 200k 块；
//       top-50 active = 10M cells = 1%；Maze SR >= 0.90 × 95% = 85.5%。
//
// 关键：不把 1B 全部载入内存。
//   - 5000 个 block bin 落盘（约 33 GB）；
//   - 逐块加载 -> 采集特征 -> 训练 138 参数读出 -> 立即释放；
//   - 评测时只按需加载 top-k 块（LRU cache，最多 200 块 ≈ 1.3 GB）。
//
// 路由：4D 网格（ray×3 + bearing）→ 5000 桶；评测按桶质心取 top-k。
// 输出：runs/moe_agent_1b_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_binary_runtime.h"
#include "kun/cellular/sdsc_compact_genome.hpp"
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
#include <list>
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
constexpr uint32_t kCells = kBlocks * kBlockCells;
constexpr uint32_t kSyns = kBlocks * kBlockSyns;
constexpr uint32_t kIn = 64;
constexpr uint32_t kFeat = 64;
constexpr uint32_t kRaw = 4;
constexpr int kCacheMax = 24;

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
    std::vector<double> mean, std, w;  // (kFeat+1) x 2
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
    for (uint32_t i = 0; i < kIn; ++i) {
        in[i] = obs[i % kRaw] * ((i & 1u) ? -1.0f : 1.0f);
    }
    in[kIn - 1] = 1.0f;
    sdsc_binary_forward(g, in, out);
}

static int grid_block(const std::array<float, kRaw>& obs) {
    auto bin = [](float v, float lo, float hi, int n) {
        int k = static_cast<int>((v - lo) / (hi - lo) * n);
        return std::max(0, std::min(n - 1, k));
    };
    const int a = bin(obs[0], 0.0f, 1.0f, 9);
    const int b = bin(obs[1], 0.0f, 1.0f, 9);
    const int c = bin(obs[2], 0.0f, 1.0f, 9);
    const int d = bin(obs[3], -1.0f, 1.0f, 7);
    const uint32_t idx = static_cast<uint32_t>(((a * 9 + b) * 9 + c) * 7 + d);
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
                          const std::vector<std::array<float, kFeat>>& feats, double lambda) {
    const int F = kFeat;
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

static void readout(const BlockMeta& m, const float* feat, double* a0, double* a1) {
    if (!m.ready) { *a0 = 0; *a1 = 0; return; }
    double x[kFeat];
    for (uint32_t i = 0; i < kFeat; ++i) x[i] = (feat[i] - m.mean[i]) / m.std[i];
    double s0 = m.w[kFeat * 2 + 0], s1 = m.w[kFeat * 2 + 1];
    for (uint32_t i = 0; i < kFeat; ++i) {
        s0 += m.w[i * 2 + 0] * x[i];
        s1 += m.w[i * 2 + 1] * x[i];
    }
    *a0 = s0;
    *a1 = s1;
}

struct Cache {
    std::unordered_map<int, std::pair<SDSCBinaryGraph*, uint64_t>> map;
    uint64_t tick{0};

    SDSCBinaryGraph* get(int c, const std::string& dir) {
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

static bool run_maze_episode(std::vector<BlockMeta>& metas, Cache& cache,
                             const std::string& dir, int topk, uint32_t seed) {
    MazeTask task(11, 11, seed, 250, 0.15f);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    cache.reset_all();

    std::vector<std::pair<double, int>> order;
    order.reserve(kBlocks);
    for (int t = 0; t < 250; ++t) {
        auto obs_v = task.current_observation();
        std::array<float, kRaw> obs{};
        for (uint32_t i = 0; i < kRaw; ++i) obs[i] = obs_v[i];

        order.clear();
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
        for (int k = 0; k < topk && k < static_cast<int>(order.size()); ++k) {
            const int c = order[k].second;
            SDSCBinaryGraph* g = cache.get(c, dir);
            if (!g) continue;
            float feat[kFeat];
            block_forward(g, obs, feat);
            double x0, x1;
            readout(metas[c], feat, &x0, &x1);
            a0 += x0;
            a1 += x1;
            ++used;
        }
        if (used) { a0 /= used; a1 /= used; }

        CellularOrganism::ActionOutputs acts;
        acts.positive_action = a0;
        acts.negative_action = a1;
        auto res = task.step_continuous(acts);
        if (res.done) return res.success;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "runs/moe_agent_1b_" + today() + ".json";
    std::string block_dir = "/tmp/opencode/moe_blocks_1b_v2";
    int collect_episodes = 1000;
    int eval_episodes = 20;
    int topk = 5;
    double full_sr = 0.95;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--out") out = next();
        else if (a == "--block-dir") block_dir = next();
        else if (a == "--collect-episodes") collect_episodes = std::max(1, std::stoi(next()));
        else if (a == "--eval-episodes") eval_episodes = std::max(1, std::stoi(next()));
        else if (a == "--topk") topk = std::max(1, std::stoi(next()));
        else if (a == "--full-sr") full_sr = std::stod(next());
        else if (a == "--help") {
            std::cout << "usage: train_moe_agent_1b [--out path] [--block-dir path] "
                         "[--collect-episodes N] [--eval-episodes N] [--topk 50] [--full-sr 0.95]\n";
            return 0;
        }
    }
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";
    std::filesystem::create_directories(block_dir);

    std::cout << "S4 1B streaming MoE | blocks=" << kBlocks << " x " << kBlockCells
              << " = " << kCells << " cells / " << kSyns << " syns\n";

    // 1) champion samples + grid routing
    CellularOrganism champ = CellularOrganism::load_checkpoint_bin(
        "checkpoints/maze_navigation_champion.bin");
    champ.compile();
    std::vector<Sample> samples;
    samples.reserve(collect_episodes * 250);
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
            s.block = grid_block(s.obs);
            samples.push_back(s);
            auto res = task.step_continuous(acts);
            if (res.done) { if (res.success) ++champ_ok; break; }
        }
    }
    std::cout << "  collected " << samples.size() << " samples, champ_sr="
              << double(champ_ok) / collect_episodes << "\n";

    std::vector<BlockMeta> metas(kBlocks);
    std::vector<std::vector<Sample*>> per_block(kBlocks);
    for (auto& s : samples) per_block[s.block].push_back(&s);
    for (uint32_t c = 0; c < kBlocks; ++c) {
        if (!per_block[c].empty()) {
            for (uint32_t i = 0; i < kRaw; ++i) {
                double sum = 0;
                for (auto* s : per_block[c]) sum += s->obs[i];
                metas[c].centroid[i] = sum / per_block[c].size();
            }
        }
    }

    // 2) 逐块：生成/加载 -> 特征 -> 读出 -> 释放
    int ready = 0;
    for (uint32_t c = 0; c < kBlocks; ++c) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/block_%05u.bin", block_dir.c_str(), c);
        if (!std::filesystem::exists(path)) {
            CompactSoAGenome g = make_block(kBlockCells, kBlockSyns, 20261123u + c * 7919u);
            g.save_binary(path);
        }
        SDSCBinaryGraph* g = sdsc_binary_load(path);
        if (!g) { std::cerr << "cannot load " << path << "\n"; return 2; }

        std::vector<std::array<float, kFeat>> feats;
        feats.reserve(per_block[c].size());
        for (auto* s : per_block[c]) {
            reset_block(g);
            std::array<float, kFeat> f{};
            block_forward(g, s->obs, f.data());
            feats.push_back(f);
        }
        train_readout(metas[c], per_block[c], feats, 1e-3);
        if (metas[c].ready) ++ready;
        sdsc_binary_free(g);
        if ((c + 1) % 500 == 0) {
            std::cout << "  trained " << (c + 1) << "/" << kBlocks << " blocks\n";
        }
    }
    std::cout << "  ready blocks: " << ready << "/" << kBlocks << "\n";

    // 3) 评测 top-k（按需加载）
    Cache cache;
    int ok = 0;
    for (int e = 0; e < eval_episodes; ++e) {
        const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
        if (run_maze_episode(metas, cache, block_dir, topk, seed)) ++ok;
    }
    const double sr = double(ok) / eval_episodes;
    const double threshold = 0.90 * full_sr;
    const bool pass = sr >= threshold;

    std::ofstream os(out);
    if (!os) { std::cerr << "cannot write " << out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"S4 1B streaming conditional MoE\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"blocks\":" << kBlocks << ",\"block_cells\":" << kBlockCells
       << ",\"block_syns\":" << kBlockSyns
       << ",\"total_cells\":" << kCells << ",\"total_syns\":" << kSyns << ",\n";
    os << "  \"samples\":" << samples.size() << ",\"champion_sr\":"
       << double(champ_ok) / collect_episodes << ",\"ready_blocks\":" << ready << ",\n";
    os << "  \"topk\":" << topk << ",\"active_cells\":" << (uint64_t)topk * kBlockCells
       << ",\"active_pct\":" << (100.0 * topk / kBlocks) << ",\n";
    os << "  \"gate\":{\"S4\":\"" << (pass ? "PASS" : "FAIL")
       << "\",\"sr\":" << sr << ",\"full_sr\":" << full_sr
       << ",\"threshold\":" << threshold << "}\n}\n";
    os.close();

    std::cout << "GATE: S4 " << (pass ? "PASS" : "FAIL")
              << " (top-" << topk << " active=" << (100.0 * topk / kBlocks)
              << "% SR=" << sr << " threshold=" << threshold << ")\n";
    std::cout << "ARTIFACT: " << out << "\n";
    return 0;
}
