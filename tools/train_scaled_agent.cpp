// ============================================================================
// train_scaled_agent.cpp — 1M 细胞稀疏储层 + 蒸馏读出（规模智能体）
//
// 背景：仓库现有 1M bin 是压测拓扑（输出全 0），不是可用主干。
// 本工具**新建**一个真正有功能的 1M 细胞稀疏储层：
//   64 输入 sense + 999,936 hidden + 64 feature cells；3M DAG 突触。
//   输入广播编码 4 维任务观测；输出取最后 64 个 feature 细胞。
//
// 蒸馏：用现成任务冠军在 maze/cartpole/maglev 上采集 (obs, action)，
//       把 1M 储层的前向特征 + 原始 obs 拼成 69 维，ridge 回归到 2 维动作。
//       冻结储层，只训练 69x2 读出（138 参数）。
//
// 产出：
//   checkpoints/scaled_reservoir_1m.bin   （1M 细胞 / 3M 突触，可复现）
//   runs/scaled_agent_<date>.json         （champion vs agent SR）
//   runs/scaled_agent_adapter_<date>.json （读出权重）
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_cuda_runtime.hpp"
#include "tasks/control/cart_pole_task.hpp"
#include "tasks/control/domain_zoo.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

constexpr uint32_t kCells = 1'000'000;
constexpr uint32_t kSyns = 3'000'000;
constexpr uint32_t kIn = 64;
constexpr uint32_t kFeat = 64;
constexpr uint32_t kRaw = 4;
constexpr uint32_t kDim = kFeat + kRaw;  // 68 features + bias

static int g_collect_episodes = 10;
static bool g_include_raw = true;

struct Sample {
    std::array<float, kRaw> obs{};
    double a0{0.0};
    double a1{0.0};
};

struct TaskData {
    std::string name;
    std::vector<Sample> samples;
    int champion_ok = 0;
    int champion_total = 0;
};

struct Adapter {
    std::vector<double> mean, std;
    std::vector<double> w;  // (kDim+1) x 2, column-major [w0_0,w0_1,w1_0,...]
    double out0(const double* x) const {
        double s = w[kDim * 2 + 0];
        for (uint32_t i = 0; i < kDim; ++i) s += w[i * 2 + 0] * x[i];
        return s;
    }
    double out1(const double* x) const {
        double s = w[kDim * 2 + 1];
        for (uint32_t i = 0; i < kDim; ++i) s += w[i * 2 + 1] * x[i];
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

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else o.push_back(c);
    }
    return o;
}

// 构建有功能的 1M 稀疏 DAG 储层。
static CompactSoAGenome make_reservoir(uint32_t seed) {
    auto g = CompactSoAGenome::create_empty(kCells, kSyns, kIn, kFeat);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> gain(0.5, 1.5);
    std::normal_distribution<double> wdist(0.0, 0.25);

    const uint8_t ops[] = {
        SDSC_OP_DAMPER, SDSC_OP_INTEGRATE, SDSC_OP_DIFF, SDSC_OP_SUM,
        SDSC_OP_MULTIPLY, SDSC_OP_CLIP, SDSC_OP_ABS,
    };
    for (uint32_t i = 0; i < kCells; ++i) {
        g.op_types[i] = (i < kIn) ? SDSC_OP_PASSTHRU : ops[rng() % (sizeof(ops))];
        g.gains[i] = static_cast<float>(gain(rng));
        g.biases[i] = 0.0f;
    }
    // 输入细胞与 feature 细胞用稳定透传/求和
    for (uint32_t i = kCells - kFeat; i < kCells; ++i) {
        g.op_types[i] = SDSC_OP_SUM;
        g.gains[i] = 1.0f;
    }

    uint32_t cursor = 0;
    g.inc_off[0] = 0;
    for (uint32_t i = 0; i < kIn; ++i) g.inc_off[i + 1] = 0;
    for (uint32_t i = kIn; i < kCells; ++i) {
        // 每个 hidden 3 条入边；最后余量分给前面的细胞
        uint32_t edges = 3;
        if (cursor + edges > kSyns) edges = 0;
        // 保证总数正好 kSyns
        const uint32_t remaining_cells = kCells - i;
        const uint32_t remaining_edges = kSyns - cursor;
        if (remaining_cells > 0) edges = remaining_edges / remaining_cells;
        if (edges == 0) edges = 1;
        if (cursor + edges > kSyns) edges = kSyns - cursor;

        g.inc_off[i] = cursor;  // 注意：inc_off[i] 已在上一轮设置；此处仅防御
        for (uint32_t e = 0; e < edges; ++e) {
            uint32_t src = static_cast<uint32_t>(rng() % i);
            g.inc_from[cursor] = src;
            g.inc_weight[cursor] = static_cast<float>(
                (src < kIn) ? (0.8 + 0.4 * gain(rng)) : wdist(rng));
            ++cursor;
        }
        g.inc_off[i + 1] = cursor;
    }
    g.inc_off[kCells] = cursor;
    if (cursor != kSyns) {
        // 理论上不会发生；若发生，截断/补齐到 kSyns
        for (uint32_t k = cursor; k < kSyns; ++k) {
            g.inc_from[k] = 0;
            g.inc_weight[k] = 0.0f;
        }
        g.inc_off[kCells] = kSyns;
    }

    for (uint32_t k = 0; k < kIn; ++k) g.in_cell_ids[k] = k;
    for (uint32_t k = 0; k < kFeat; ++k) g.out_cell_ids[k] = kCells - kFeat + k;
    return g;
}

static std::string resolve_path(const std::string& p) {
    if (std::filesystem::exists(p)) return p;
    for (const char* prefix : {"../", "../../", "../../../"}) {
        std::string cand = std::string(prefix) + p;
        if (std::filesystem::exists(cand)) return cand;
    }
    return p;
}

struct Trunk {
    std::unique_ptr<SdscCUDAGraph> gpu;
    std::vector<float> in, out;
    bool load(const CompactSoAGenome& g) {
        gpu = std::make_unique<SdscCUDAGraph>();
        if (!gpu->upload(g)) return false;
        in.assign(kIn, 0.0f);
        out.assign(kFeat, 0.0f);
        return true;
    }
    void forward(const std::array<float, kRaw>& obs) {
        // 广播编码：64 路输入都携带任务观测，保证储层被激活。
        for (uint32_t i = 0; i < kIn; ++i) {
            in[i] = obs[i % kRaw] * ((i & 1u) ? -1.0f : 1.0f);
        }
        in[kIn - 1] = 1.0f;  // bias
        gpu->forward(in.data(), out.data());
    }
};

static void fill_features(const Trunk& trunk, const std::array<float, kRaw>& obs,
                          double* x) {
    for (uint32_t i = 0; i < kFeat; ++i) x[i] = trunk.out[i];
    for (uint32_t i = 0; i < kRaw; ++i) x[kFeat + i] = g_include_raw ? obs[i] : 0.0;
}

static void normalize_fit(std::vector<Sample>& samples) {
    // 在这里不做全局标准化；ridge 前按特征列计算 mean/std，见 train_adapter。
}

static bool solve_linear(std::vector<double>& A, std::vector<double>& B, int n, int m) {
    // A n x n, B n x m; Gauss-Jordan with partial pivot
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

static Adapter train_adapter(Trunk& trunk, const std::vector<Sample>& samples, double lambda) {
    const int F = kDim;
    const int N = F + 1;
    std::vector<double> A(N * N, 0.0), B(N * 2, 0.0);
    std::vector<double> mean(F, 0.0), sq(F, 0.0);

    std::vector<std::vector<double>> feats;
    feats.reserve(samples.size());
    for (const auto& s : samples) {
        trunk.forward(s.obs);
        std::vector<double> x(F + 1, 1.0);
        fill_features(trunk, s.obs, x.data());
        for (int i = 0; i < F; ++i) { mean[i] += x[i]; sq[i] += x[i] * x[i]; }
        feats.push_back(std::move(x));
    }
    const double invN = samples.empty() ? 0.0 : 1.0 / samples.size();
    for (int i = 0; i < F; ++i) {
        mean[i] *= invN;
        const double var = std::max(0.0, sq[i] * invN - mean[i] * mean[i]);
        sq[i] = std::sqrt(var) + 1e-6;
    }
    for (size_t k = 0; k < samples.size(); ++k) {
        auto& x = feats[k];
        for (int i = 0; i < F; ++i) x[i] = (x[i] - mean[i]) / sq[i];
    }
    for (size_t k = 0; k < samples.size(); ++k) {
        auto& x = feats[k];
        for (int i = 0; i <= F; ++i) {
            for (int j = 0; j <= F; ++j) A[i * N + j] += x[i] * x[j];
            B[i * 2 + 0] += x[i] * samples[k].a0;
            B[i * 2 + 1] += x[i] * samples[k].a1;
        }
    }
    for (int i = 0; i < N; ++i) A[i * N + i] += lambda;
    if (!solve_linear(A, B, N, 2)) {
        std::cerr << "[adapter] ridge solve failed\n";
    }
    Adapter ad;
    ad.mean = mean;
    ad.std = sq;
    ad.w = B;
    return ad;
}

static void collect_maze(Trunk& trunk, const std::string& champ, TaskData& out) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(champ);
    org.compile();
    for (int e = 0; e < g_collect_episodes; ++e) {
        const uint32_t seed = 90000u + static_cast<uint32_t>(e) * 17u;
        MazeTask task(11, 11, seed, 250, 0.15f);
        task.set_use_geodesic_bearing(true);
        task.reset(seed);
        org.reset_state(false);
        for (int t = 0; t < 250; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            Sample s;
            for (uint32_t i = 0; i < kRaw; ++i) s.obs[i] = obs[i];
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            out.samples.push_back(s);
            auto res = task.step_continuous(acts);
            if (res.done) { out.champion_ok += res.success ? 1 : 0; break; }
        }
        out.champion_total++;
    }
}

static void collect_cartpole(Trunk& trunk, const std::string& champ, TaskData& out) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(champ);
    org.compile();
    for (int e = 0; e < g_collect_episodes; ++e) {
        const uint32_t seed = 91000u + static_cast<uint32_t>(e) * 17u;
        CartPoleBalanceTask task;
        task.set_max_steps(300);
        task.reset(seed);
        org.reset_state(false);
        bool done = false;
        for (int t = 0; t < 300; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            Sample s;
            for (uint32_t i = 0; i < kRaw; ++i) s.obs[i] = obs[i];
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            out.samples.push_back(s);
            auto res = task.step_continuous(acts);
            if (res.done) { done = true; out.champion_ok += res.success ? 1 : 0; break; }
        }
        if (!done) ++out.champion_ok;
        out.champion_total++;
    }
}

static void collect_maglev(Trunk& trunk, const std::string& champ, TaskData& out) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(champ);
    org.compile();
    for (int e = 0; e < g_collect_episodes; ++e) {
        const uint32_t seed = 92000u + static_cast<uint32_t>(e) * 17u;
        ZooMaglev task;
        task.set_max_steps(200);
        task.reset(seed);
        org.reset_state(false);
        bool done = false;
        for (int t = 0; t < 200; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            Sample s;
            for (uint32_t i = 0; i < kRaw; ++i) s.obs[i] = obs[i];
            s.a0 = acts.positive_action;
            s.a1 = acts.negative_action;
            out.samples.push_back(s);
            auto res = task.step_continuous(acts);
            if (res.done) { done = true; out.champion_ok += res.success ? 1 : 0; break; }
        }
        if (!done) ++out.champion_ok;
        out.champion_total++;
    }
}

static bool eval_agent(Trunk& trunk, const Adapter& ad, const std::string& task,
                       int episodes, int* ok_out) {
    int ok = 0;
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = 93000u + static_cast<uint32_t>(e) * 17u;
        std::unique_ptr<EvolvableTask> env;
        int max_steps = 250;
        if (task == "maze") {
            auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
            t->set_use_geodesic_bearing(true);
            t->reset(seed);
            env.reset(t);
            max_steps = 250;
        } else if (task == "cartpole") {
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
        bool success = false;
        bool done = false;
        for (int t = 0; t < max_steps; ++t) {
            auto obs_v = env->current_observation();
            std::array<float, kRaw> obs{};
            for (uint32_t i = 0; i < kRaw; ++i) obs[i] = obs_v[i];
            trunk.forward(obs);
            double x[kDim];
            fill_features(trunk, obs, x);
            for (uint32_t i = 0; i < kDim; ++i) x[i] = (x[i] - ad.mean[i]) / ad.std[i];
            const double a0 = ad.out0(x);
            const double a1 = ad.out1(x);
            CellularOrganism::ActionOutputs acts;
            acts.positive_action = std::max(0.0, a0);
            acts.negative_action = std::max(0.0, a1);
            if (task == "maze") {
                acts.positive_action = a0;
                acts.negative_action = a1;
            }
            auto res = env->step_continuous(acts);
            if (res.done) { done = true; success = res.success; break; }
        }
        if (!done && task != "maze") success = true;  // 满步存活
        if (success) ++ok;
    }
    *ok_out = ok;
    return ok > 0;
}

static void save_adapter(const std::string& path, const Adapter& ad,
                         const std::string& task, double sr,
                         const std::string& reservoir) {
    std::ofstream os(path);
    if (!os) return;
    os << std::fixed << std::setprecision(8);
    os << "{\n  \"task\":\"" << task << "\",\n";
    os << "  \"reservoir\":\"" << reservoir << "\",\n";
    os << "  \"sr\":" << sr << ",\n";
    os << "  \"feature_dim\":" << kDim << ",\n";
    os << "  \"mean\":[";
    for (size_t i = 0; i < ad.mean.size(); ++i) { if (i) os << ","; os << ad.mean[i]; }
    os << "],\n  \"std\":[";
    for (size_t i = 0; i < ad.std.size(); ++i) { if (i) os << ","; os << ad.std[i]; }
    os << "],\n  \"w\":[";
    for (size_t i = 0; i < ad.w.size(); ++i) { if (i) os << ","; os << ad.w[i]; }
    os << "]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "runs/scaled_agent_" + today() + ".json";
    std::string reservoir_path = "checkpoints/scaled_reservoir_1m.bin";
    int eval_episodes = 20;
    uint32_t seed = 20261123u;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--out") out = next();
        else if (a == "--reservoir") reservoir_path = next();
        else if (a == "--eval-episodes") eval_episodes = std::stoi(next());
        else if (a == "--seed") seed = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--collect-episodes") g_collect_episodes = std::max(1, std::stoi(next()));
        else if (a == "--no-raw") g_include_raw = false;
        else if (a == "--help") {
            std::cout << "usage: train_scaled_agent [--out path] [--reservoir path] "
                         "[--eval-episodes N] [--seed N]\n";
            return 0;
        }
    }
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    std::cout << "Scaled agent | building 1M reservoir...\n";
    CompactSoAGenome genome;
    if (std::filesystem::exists(reservoir_path)) {
        // 有 bin 就直接用 SDSC-BIN 读取不方便；这里总是重建，保证可复现
    }
    genome = make_reservoir(seed);
    std::filesystem::create_directories(std::filesystem::path(reservoir_path).parent_path());
    if (!genome.save_binary(reservoir_path)) {
        std::cerr << "cannot save " << reservoir_path << "\n";
        return 2;
    }
    std::cout << "  saved " << reservoir_path << " cells=" << genome.num_cells
              << " syns=" << genome.num_synapses << "\n";

    Trunk trunk;
    if (!trunk.load(genome)) {
        std::cerr << "GPU upload failed (NVRTC path?)\n";
        return 77;
    }
    // probe
    {
        std::array<float, kRaw> obs{0.1f, -0.2f, 0.3f, -0.4f};
        trunk.forward(obs);
        std::cout << "  probe features:";
        for (uint32_t i = 0; i < 8; ++i) std::cout << " " << trunk.out[i];
        std::cout << "\n";
    }

    TaskData tasks[3];
    tasks[0].name = "maze";
    tasks[1].name = "cartpole";
    tasks[2].name = "maglev";
    collect_maze(trunk, resolve_path("checkpoints/maze_navigation_champion.bin"), tasks[0]);
    collect_cartpole(trunk, resolve_path("checkpoints/cartpole_balance_champion.bin"), tasks[1]);
    collect_maglev(trunk, resolve_path("checkpoints/zoo_maglev.bin"), tasks[2]);

    std::ofstream os(out);
    if (!os) { std::cerr << "cannot write " << out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"scaled agent 1M reservoir + distilled readout\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"reservoir\":\"" << json_escape(reservoir_path) << "\",\n";
    os << "  \"reservoir_cells\":" << kCells << ",\"reservoir_syns\":" << kSyns
       << ",\"readout_params\":" << (kDim + 1) * 2 << ",\n";
    os << "  \"tasks\":[";

    double champ_sr[3] = {0, 0, 0};
    double agent_sr[3] = {0, 0, 0};
    int agent_ok[3] = {0, 0, 0};
    for (int ti = 0; ti < 3; ++ti) {
        auto& data = tasks[ti];
        Adapter ad = train_adapter(trunk, data.samples, 1e-3);
        eval_agent(trunk, ad, data.name, eval_episodes, &agent_ok[ti]);
        champ_sr[ti] = data.champion_total ? double(data.champion_ok) / data.champion_total : 0.0;
        agent_sr[ti] = eval_episodes ? double(agent_ok[ti]) / eval_episodes : 0.0;
        if (ti) os << ",";
        os << "\n    {\"task\":\"" << data.name << "\",\"samples\":" << data.samples.size()
           << ",\"champion_sr\":" << champ_sr[ti] << ",\"agent_sr\":" << agent_sr[ti]
           << ",\"delta\":" << (agent_sr[ti] - champ_sr[ti])
           << ",\"champion_ok\":" << data.champion_ok << ",\"champion_total\":" << data.champion_total
           << ",\"agent_ok\":" << agent_ok[ti] << ",\"agent_total\":" << eval_episodes << "}";
        std::cout << "  " << data.name << ": champion=" << champ_sr[ti]
                  << " agent=" << agent_sr[ti] << " (samples=" << data.samples.size() << ")\n";
        {
            std::string ap = "runs/scaled_agent_adapter_" + data.name + "_" + today() + ".json";
            save_adapter(ap, ad, data.name, agent_sr[ti], reservoir_path);
        }
    }
    // 判定：至少一个任务 agent >= champion - 5pp，或 agent >= 80%
    bool pass = false;
    for (int ti = 0; ti < 3; ++ti) {
        if (agent_sr[ti] >= champ_sr[ti] - 0.05 || agent_sr[ti] >= 0.80) pass = true;
    }
    os << "\n  ],\n  \"gate\":{\"SCALED\":\"" << (pass ? "PASS" : "FAIL") << "\"}\n}\n";
    os.close();
    std::cout << "GATE: SCALED " << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "ARTIFACT: " << out << "\n";
    return 0;
}
