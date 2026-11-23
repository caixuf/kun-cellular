// ============================================================================
// bench_adapter_transfer.cpp — S1 冻结 1M 主干 + rank-8 适配器
//
// 目的：验证「冻结大主干 + 低秩适配器」能否用 <1% 参数接近 from-scratch 控制器。
//
// 主干：checkpoints/sdsc_mega_1million.bin (1,000,000 cells / 3,000,000 syns,
//       64 in / 16 out)，通过 sdsc_binary_runtime mmap 读取，转 CompactSoAGenome
//       上传 GPU (SdscCUDAGraph)。主干权重完全冻结。
//
// 适配器：16 维主干输出 -> rank-8 -> 2 维动作；参数 154 (<0.02% of 1M)。
// 训练：ES 只调适配器；主干 forward 不反传。
//
// 任务（3 个）：
//   maze     11x11, 250 steps, geodesic bearing
//   cartpole 300 steps
//   maglev   ZooMaglev, 200 steps
//
// 对照：现成 from-scratch 冠军（maze/cartpole/zoo_maglev）。
// 判据（预注册）：adapter SR >= champion SR - 5pp，且适配器参数 <1% 主干。
//
// 输出：runs/adapter_transfer_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/sdsc_binary_runtime.h"
#include "kun/cellular/sdsc_cuda_runtime.hpp"
#include "tasks/control/cart_pole_task.hpp"
#include "tasks/control/domain_zoo.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#include <algorithm>
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

constexpr size_t kRank = 8;
constexpr size_t kAct = 2;

struct Options {
    std::string trunk = "checkpoints/sdsc_mega_1million.bin";
    std::string out;
    int pop = 16;
    int gens = 40;
    int episodes_train = 4;
    int episodes_eval = 100;
    int seeds = 3;
    bool quick = false;
    bool cpu = false;
};

struct EvalResult {
    double sr = 0.0;
    int ok = 0;
    int total = 0;
    std::vector<int> bits;
};

struct TaskResult {
    std::string task;
    int seed = 0;
    double adapter_sr = 0.0;
    double champion_sr = 0.0;
    double delta = 0.0;
    double train_ms = 0.0;
    std::vector<int> adapter_bits;
    std::vector<int> champion_bits;
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

static std::string resolve_path(const std::string& p) {
    if (std::filesystem::exists(p)) return p;
    for (const char* prefix : {"../", "../../", "../../../"}) {
        std::string cand = std::string(prefix) + p;
        if (std::filesystem::exists(cand)) return cand;
    }
    return p;
}

// SDSC-BIN (mmap) -> CompactSoAGenome (GPU upload 用)
static CompactSoAGenome from_binary(const SDSCBinaryGraph* bg) {
    const auto& h = bg->header;
    CompactSoAGenome g = CompactSoAGenome::create_empty(
        h.num_cells, h.num_synapses, h.input_dim, h.output_dim);
    for (uint32_t i = 0; i < h.num_cells; ++i) {
        g.op_types[i] = bg->cells[i].op_type;
        g.gains[i] = static_cast<float>(bg->cells[i].param1_u8) / 255.0f * 4.0f;
        g.biases[i] = static_cast<float>(bg->cells[i].param2_u8) / 255.0f * 4.0f;
    }
    for (uint32_t i = 0; i <= h.num_cells; ++i) g.inc_off[i] = bg->row_ptr[i];
    for (uint32_t k = 0; k < h.num_synapses; ++k) {
        g.inc_from[k] = bg->col_idx[k];
        g.inc_weight[k] = bg->weights[k];
    }
    for (uint32_t k = 0; k < h.input_dim; ++k) {
        g.in_cell_ids[k] = bg->receptor_map ? bg->receptor_map[k] : k;
    }
    for (uint32_t k = 0; k < h.output_dim; ++k) {
        g.out_cell_ids[k] = h.num_cells - h.output_dim + k;
    }
    return g;
}

struct Adapter {
    size_t feat_dim{16};
    std::vector<double> genes;  // U(feat x rank), V(rank x act), b1(rank), b2(act)
    std::vector<double> feat_mean;
    std::vector<double> feat_std;

    explicit Adapter(size_t f = 16)
        : feat_dim(f),
          genes(f * kRank + kRank * kAct + kRank + kAct, 0.0) {}

    size_t param_count() const { return genes.size(); }

    void randomize(std::mt19937& rng) {
        std::normal_distribution<double> n(0.0, 0.3);
        for (auto& g : genes) g = n(rng);
    }

    void forward(const float* feat, double out[kAct]) const {
        std::vector<double> x(feat_dim);
        for (size_t i = 0; i < feat_dim; ++i) {
            x[i] = feat_mean.empty() ? feat[i]
                                     : (feat[i] - feat_mean[i]) / (feat_std[i] + 1e-6);
        }
        const size_t off_b1 = feat_dim * kRank + kRank * kAct;
        const size_t off_b2 = off_b1 + kRank;
        double h[kRank];
        for (size_t r = 0; r < kRank; ++r) {
            double s = genes[off_b1 + r];
            for (size_t i = 0; i < feat_dim; ++i) s += genes[i * kRank + r] * x[i];
            h[r] = std::tanh(s);
        }
        for (size_t a = 0; a < kAct; ++a) {
            double s = genes[off_b2 + a];
            for (size_t r = 0; r < kRank; ++r) {
                s += genes[feat_dim * kRank + r * kAct + a] * h[r];
            }
            out[a] = std::tanh(s);
        }
    }
};

struct Trunk {
    SDSCBinaryGraph* bg{nullptr};
    std::unique_ptr<SdscCUDAGraph> gpu;
    CellularOrganism cpu_org;
    bool cpu{false};
    bool ready{false};
    size_t in_dim{0};
    size_t out_dim{0};
    std::vector<float> in;
    std::vector<float> out;
    std::vector<double> cpu_in;
    std::vector<size_t> feat_cells;

    bool load(const std::string& path, bool use_cpu = false) {
        cpu = use_cpu;
        if (cpu) {
            cpu_org = CellularOrganism::load_checkpoint_bin(path);
            if (cpu_org.cells.empty()) { std::cerr << "[trunk] cpu load failed\n"; return false; }
            cpu_org.compile();
            bg = sdsc_binary_load(path.c_str());
            if (!bg) { std::cerr << "[trunk] header load failed\n"; return false; }
            in_dim = bg->header.input_dim;
            out_dim = bg->header.output_dim;
            for (size_t i = 0; i < cpu_org.cells.size(); ++i) {
                const int t = static_cast<int>(cpu_org.cells[i].type);
                if (t >= 30 && t <= 34) feat_cells.push_back(i);
            }
            if (feat_cells.size() < out_dim) {
                feat_cells.clear();
                const size_t start = cpu_org.cells.size() > out_dim ? cpu_org.cells.size() - out_dim : 0;
                for (size_t i = start; i < cpu_org.cells.size(); ++i) feat_cells.push_back(i);
            }
            in.assign(in_dim, 0.0f);
            cpu_in.assign(in_dim, 0.0);
            out.assign(out_dim, 0.0f);
            ready = true;
            return true;
        }
        bg = sdsc_binary_load(path.c_str());
        if (!bg) { std::cerr << "[trunk] sdsc_binary_load failed\n"; return false; }
        in_dim = bg->header.input_dim;
        out_dim = bg->header.output_dim;
        if (in_dim == 0 || out_dim == 0) { std::cerr << "[trunk] zero in/out dim\n"; return false; }
        CompactSoAGenome g = from_binary(bg);
        gpu = std::make_unique<SdscCUDAGraph>();
        if (!gpu->upload(g)) { std::cerr << "[trunk] gpu.upload failed\n"; return false; }
        in.assign(in_dim, 0.0f);
        out.assign(out_dim, 0.0f);
        ready = true;
        return true;
    }

    void step(const std::vector<float>& obs4) {
        std::fill(in.begin(), in.end(), 0.0f);
        for (size_t i = 0; i < obs4.size() && i < 4 && i < in_dim; ++i) in[i] = obs4[i];
        if (cpu) {
            for (size_t i = 0; i < in.size(); ++i) cpu_in[i] = in[i];
            cpu_org.forward_nd(cpu_in.data(), cpu_in.size(), false);
            for (size_t i = 0; i < out_dim && i < feat_cells.size(); ++i) {
                out[i] = static_cast<float>(cpu_org.cells[feat_cells[i]].output_val);
            }
        } else {
            gpu->forward(in.data(), out.data());
        }
    }
};

static void calibrate(Trunk& trunk, Adapter& ad, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const size_t nf = trunk.out_dim;
    std::vector<double> sum(nf, 0.0), sq(nf, 0.0);
    const int N = 64;
    for (int i = 0; i < N; ++i) {
        std::vector<float> obs(4);
        for (auto& v : obs) v = u(rng);
        trunk.step(obs);
        for (size_t f = 0; f < nf; ++f) {
            sum[f] += trunk.out[f];
            sq[f] += static_cast<double>(trunk.out[f]) * trunk.out[f];
        }
    }
    ad.feat_mean.resize(nf);
    ad.feat_std.resize(nf);
    for (size_t f = 0; f < nf; ++f) {
        ad.feat_mean[f] = sum[f] / N;
        const double var = std::max(0.0, sq[f] / N - ad.feat_mean[f] * ad.feat_mean[f]);
        ad.feat_std[f] = std::sqrt(var);
    }
}

enum class TaskId { Maze, CartPole, Maglev };

static std::string task_name(TaskId t) {
    switch (t) {
        case TaskId::Maze: return "maze";
        case TaskId::CartPole: return "cartpole";
        case TaskId::Maglev: return "maglev";
    }
    return "?";
}

static bool run_episode(Trunk& trunk, const Adapter& ad, TaskId task, uint32_t seed,
                        int* out_steps = nullptr) {
    std::unique_ptr<EvolvableTask> env;
    int max_steps = 250;
    if (task == TaskId::Maze) {
        auto* t = new MazeTask(11, 11, seed, 250, 0.15f);
        t->set_use_geodesic_bearing(true);
        t->reset(seed);
        env.reset(t);
        max_steps = 250;
    } else if (task == TaskId::CartPole) {
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

    for (int s = 0; s < max_steps; ++s) {
        auto obs = env->current_observation();
        trunk.step(obs);
        double a[kAct] = {0, 0};
        ad.forward(trunk.out.data(), a);

        CellularOrganism::ActionOutputs acts;
        if (task == TaskId::Maze) {
            acts.positive_action = a[0];
            acts.negative_action = a[1];
        } else {
            const double force = a[0] - a[1];
            acts.positive_action = std::max(0.0, force);
            acts.negative_action = std::max(0.0, -force);
        }
        auto res = env->step_continuous(acts);
        if (res.done) {
            if (out_steps) *out_steps = res.steps;
            return res.success;
        }
    }
    if (out_steps) *out_steps = max_steps;
    // CartPole / Maglev 的 success 语义是“满步存活”；Maze 是“到达目标”。
    return task != TaskId::Maze;
}

static EvalResult evaluate_adapter(Trunk& trunk, const Adapter& ad, TaskId task,
                                   int episodes, uint32_t seed) {
    EvalResult r;
    r.total = episodes;
    for (int e = 0; e < episodes; ++e) {
        const bool ok = run_episode(trunk, ad, task, seed + static_cast<uint32_t>(e) * 17u);
        if (ok) ++r.ok;
        r.bits.push_back(ok ? 1 : 0);
    }
    r.sr = episodes > 0 ? static_cast<double>(r.ok) / episodes : 0.0;
    return r;
}

static EvalResult evaluate_champion(const std::string& path, TaskId task,
                                    int episodes, uint32_t seed) {
    EvalResult r;
    r.total = episodes;
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    if (org.cells.empty()) return r;
    org.compile();
    for (int e = 0; e < episodes; ++e) {
        const uint32_t s = seed + static_cast<uint32_t>(e) * 17u;
        std::unique_ptr<EvolvableTask> env;
        int max_steps = 250;
        if (task == TaskId::Maze) {
            auto* t = new MazeTask(11, 11, s, 250, 0.15f);
            t->set_use_geodesic_bearing(true);
            t->reset(s);
            env.reset(t);
            max_steps = 250;
        } else if (task == TaskId::CartPole) {
            auto* t = new CartPoleBalanceTask();
            t->set_max_steps(300);
            t->reset(s);
            env.reset(t);
            max_steps = 300;
        } else {
            auto* t = new ZooMaglev();
            t->set_max_steps(200);
            t->reset(s);
            env.reset(t);
            max_steps = 200;
        }
        org.reset_state(false);
        bool ok = false;
        for (int t = 0; t < max_steps; ++t) {
            auto obs = env->current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            auto res = env->step_continuous(acts);
            if (res.done) { ok = res.success; break; }
        }
        if (!ok && task != TaskId::Maze) ok = true;  // 满步存活
        if (ok) ++r.ok;
        r.bits.push_back(ok ? 1 : 0);
    }
    r.sr = episodes > 0 ? static_cast<double>(r.ok) / episodes : 0.0;
    return r;
}

struct ESResult {
    Adapter adapter;
    double fitness = 0.0;
};

static ESResult train_adapter(Trunk& trunk, TaskId task, const Options& opt, uint32_t seed) {
    std::mt19937 rng(seed ^ 0x5bd1e995u);
    std::normal_distribution<double> mut(0.0, 0.2);

    auto fitness = [&](const Adapter& ad, uint32_t eval_seed) {
        return evaluate_adapter(trunk, ad, task, opt.episodes_train, eval_seed).sr;
    };

    std::vector<Adapter> pop(static_cast<size_t>(opt.pop), Adapter(trunk.out_dim));
    for (auto& a : pop) {
        a.randomize(rng);
        calibrate(trunk, a, seed);
    }

    Adapter best = pop[0];
    double best_fit = -1.0;
    const int elite = std::max(2, opt.pop / 6);

    for (int gen = 0; gen < opt.gens; ++gen) {
        const uint32_t eval_seed = seed + static_cast<uint32_t>(gen) * 7919u;
        std::vector<std::pair<double, size_t>> scored;
        for (size_t i = 0; i < pop.size(); ++i) {
            scored.emplace_back(fitness(pop[i], eval_seed), i);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (scored.front().first > best_fit) {
            best_fit = scored.front().first;
            best = pop[scored.front().second];
        }
        std::vector<Adapter> next;
        for (int i = 0; i < elite; ++i) next.push_back(pop[scored[i].second]);
        while (next.size() < pop.size()) {
            Adapter child = pop[scored[rng() % elite].second];
            for (auto& g : child.genes) {
                if ((rng() % 100) < 70) g += mut(rng);
                g = std::max(-4.0, std::min(4.0, g));
            }
            next.push_back(std::move(child));
        }
        pop.swap(next);
    }
    return {best, best_fit};
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--trunk") opt.trunk = next();
        else if (a == "--out") opt.out = next();
        else if (a == "--pop") opt.pop = std::stoi(next());
        else if (a == "--gens") opt.gens = std::stoi(next());
        else if (a == "--episodes") opt.episodes_eval = std::stoi(next());
        else if (a == "--seeds") opt.seeds = std::stoi(next());
        else if (a == "--cpu") opt.cpu = true;
        else if (a == "--quick") opt.quick = true;
        else if (a == "--help") {
            std::cout << "usage: bench_adapter_transfer [--quick] [--pop N] [--gens N] "
                         "[--seeds N] [--episodes N] [--out path]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.pop = 6;
        opt.gens = 6;
        opt.episodes_train = 2;
        opt.episodes_eval = 10;
        opt.seeds = 1;
    }
    if (opt.out.empty()) opt.out = "runs/adapter_transfer_" + today() + ".json";
    opt.trunk = resolve_path(opt.trunk);
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    Trunk trunk;
    if (!trunk.load(opt.trunk, opt.cpu)) {
        std::cerr << "cannot load/upload trunk: " << opt.trunk << "\n";
        std::cerr << "hint: set LD_LIBRARY_PATH to NVRTC builtins (see ctest env)\n";
        return 77;  // SKIP
    }
    const size_t adapter_params = trunk.out_dim * kRank + kRank * kAct + kRank + kAct;
    std::cout << "S1 adapter transfer | trunk=" << opt.trunk
              << " backend=" << (opt.cpu ? "cpu" : "cuda")
              << " cells=" << trunk.bg->header.num_cells
              << " syns=" << trunk.bg->header.num_synapses
              << " in=" << trunk.in_dim << " out=" << trunk.out_dim
              << " adapter_params=" << adapter_params << "\n";

    // 诊断：主干输出是否全零/饱和
    {
        std::vector<float> probe = {0.1f, -0.2f, 0.3f, -0.4f};
        trunk.step(probe);
        std::cout << "  trunk probe out:";
        for (size_t i = 0; i < trunk.out_dim; ++i) std::cout << " " << trunk.out[i];
        std::cout << "\n";
    }

    const char* champs[3] = {
        "checkpoints/maze_navigation_champion.bin",
        "checkpoints/cartpole_balance_champion.bin",
        "checkpoints/zoo_maglev.bin",
    };
    TaskId tasks[3] = {TaskId::Maze, TaskId::CartPole, TaskId::Maglev};

    std::vector<TaskResult> results;
    for (int ti = 0; ti < 3; ++ti) {
        for (int s = 0; s < opt.seeds; ++s) {
            const uint32_t seed = 20261123u + static_cast<uint32_t>(s) * 101u;
            std::cout << "  train " << task_name(tasks[ti]) << " seed=" << seed << " ... " << std::flush;
            const auto t0 = std::chrono::steady_clock::now();
            ESResult es = train_adapter(trunk, tasks[ti], opt, seed);
            const auto t1 = std::chrono::steady_clock::now();
            EvalResult a = evaluate_adapter(trunk, es.adapter, tasks[ti],
                                            opt.episodes_eval, seed ^ 0xABCDEFu);
            EvalResult c = evaluate_champion(resolve_path(champs[ti]), tasks[ti],
                                             opt.episodes_eval, seed ^ 0xABCDEFu);
            TaskResult r;
            r.task = task_name(tasks[ti]);
            r.seed = static_cast<int>(seed);
            r.adapter_sr = a.sr;
            r.champion_sr = c.sr;
            r.delta = a.sr - c.sr;
            r.train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            r.adapter_bits = a.bits;
            r.champion_bits = c.bits;
            std::cout << "adapter=" << a.sr << " champion=" << c.sr
                      << " delta=" << r.delta << " train=" << r.train_ms << "ms\n";
            results.push_back(std::move(r));
        }
    }

    auto mean = [&](const std::string& task, bool adapter) {
        double sum = 0.0;
        int n = 0;
        for (const auto& r : results) {
            if (r.task == task) { sum += adapter ? r.adapter_sr : r.champion_sr; ++n; }
        }
        return n ? sum / n : 0.0;
    };

    bool gate = true;
    for (const char* t : {"maze", "cartpole", "maglev"}) {
        const double a = mean(t, true);
        const double c = mean(t, false);
        if (a < c - 0.05) gate = false;
    }

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"S1 frozen 1M trunk + rank-8 adapter\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"pop\":" << opt.pop << ",\"gens\":" << opt.gens
       << ",\"seeds\":" << opt.seeds << ",\"episodes_train\":" << opt.episodes_train
       << ",\"episodes_eval\":" << opt.episodes_eval
       << ",\"adapter_params\":" << adapter_params
       << ",\"rank\":" << kRank << "},\n";
    os << "  \"results\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) os << ",";
        os << "\n    {\"task\":\"" << r.task << "\",\"seed\":" << r.seed
           << ",\"adapter_sr\":" << r.adapter_sr
           << ",\"champion_sr\":" << r.champion_sr
           << ",\"delta\":" << r.delta
           << ",\"train_ms\":" << r.train_ms
           << ",\"adapter_bits\":[";
        for (size_t j = 0; j < r.adapter_bits.size(); ++j) {
            if (j) os << ",";
            os << r.adapter_bits[j];
        }
        os << "],\"champion_bits\":[";
        for (size_t j = 0; j < r.champion_bits.size(); ++j) {
            if (j) os << ",";
            os << r.champion_bits[j];
        }
        os << "]}";
    }
    os << "\n  ],\n  \"aggregate\":{";
    bool first = true;
    for (const char* t : {"maze", "cartpole", "maglev"}) {
        if (!first) os << ",";
        first = false;
        os << "\"" << t << "_adapter\":" << mean(t, true)
           << ",\"" << t << "_champion\":" << mean(t, false);
    }
    os << "},\n  \"gate\":{\"S1\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"adapter_params\":" << adapter_params << "}\n}\n";
    os.close();

    std::cout << "\n== aggregate ==\n";
    for (const char* t : {"maze", "cartpole", "maglev"}) {
        std::cout << "  " << std::setw(9) << t
                  << " adapter=" << mean(t, true)
                  << " champion=" << mean(t, false)
                  << " delta=" << (mean(t, true) - mean(t, false)) << "\n";
    }
    std::cout << "GATE: S1 " << (gate ? "PASS" : "FAIL")
              << " (adapter_params=" << adapter_params << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}
