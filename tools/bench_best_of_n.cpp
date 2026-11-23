// ============================================================================
// bench_best_of_n.cpp — E-D Best-of-N + verifier（1 步前瞻，oracle verifier）
//
// 目的：验证“推理时搜索”是否能在不训练的情况下提升控制成功率。
//
// 形式：
//   - 基础策略：maze_navigation_champion（可加 policy noise 制造 headroom）
//   - 每步采样 N 个动作候选（围绕策略动作的高斯扰动）
//   - verifier = 真实迷宫模型 1 步前瞻（oracle），按 reward 选最优
//   - 执行最优候选；N=1 即无搜索基线
//
// 对照 / 口径：
//   - N ∈ {1,4,16,64}
//   - compute multiplier = N（每一步做 N 次模型推演）
//   - verifier 是 oracle simulator，不是学习模型；结论只对“有模型搜索”成立
//
// 输出：runs/best_of_n_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
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
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace kun;

namespace {

struct Options {
    std::string maze = "checkpoints/maze_navigation_champion.bin";
    std::string out;
    std::vector<int> ns{1, 4, 16, 64};
    int map = 21;
    int max_steps = 400;
    int episodes = 100;
    uint32_t seed_base = 80000;
    double policy_noise = 0.8;
    double cand_noise = 0.5;
    bool quick = false;
};

struct Result {
    int n = 1;
    int episodes = 0;
    int success = 0;
    double sr = 0.0;
    double mean_steps = 0.0;
    double wall_ms = 0.0;
    double compute_x = 1.0;
    std::vector<int> bits;
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

static double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

// 返回该 episode 是否成功；同时累加步数。
static bool run_episode(CellularOrganism& org, int n, int map_size, int max_steps,
                        uint32_t seed, double policy_noise, double cand_noise,
                        int* out_steps) {
    MazeTask task(map_size, map_size, seed, max_steps, 0.15f);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    org.reset_state(false);

    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::normal_distribution<double> pn(0.0, policy_noise);
    std::normal_distribution<double> cn(0.0, cand_noise);

    for (int t = 0; t < max_steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto policy = org.forward(inps, false);

        // 基础策略加噪：制造 headroom（N=1 也要承受同样噪声）
        CellularOrganism::ActionOutputs base = policy;
        base.positive_action += pn(rng);
        base.negative_action += pn(rng);
        base.defensive_reset = std::max(0.0, base.defensive_reset + pn(rng) * 0.5);

        CellularOrganism::ActionOutputs chosen = base;
        if (n > 1) {
            double best_score = -1e18;
            for (int i = 0; i < n; ++i) {
                CellularOrganism::ActionOutputs cand = base;
                if (i > 0) {
                    cand.positive_action += cn(rng);
                    cand.negative_action += cn(rng);
                    cand.defensive_reset = std::max(0.0, cand.defensive_reset + cn(rng) * 0.5);
                }

                MazeTask sim = task;  // 真实模型 1 步前瞻（oracle verifier）
                auto res = sim.step_continuous(cand);
                // 用测地势场距离打分；候选 0 是 base action，保证 N>1 不劣于无搜索。
                double score = -static_cast<double>(sim.current_geodesic_distance_to_goal());
                if (res.success) score += 1e6;
                score -= static_cast<double>(res.collision_count) * 0.01;
                if (score > best_score) {
                    best_score = score;
                    chosen = cand;
                }
            }
        }

        auto res = task.step_continuous(chosen);
        if (res.done) {
            if (out_steps) *out_steps = res.steps;
            return res.success;
        }
    }
    if (out_steps) *out_steps = max_steps;
    return false;
}

static Result evaluate(CellularOrganism& org, const Options& opt, int n) {
    Result r;
    r.n = n;
    r.episodes = opt.episodes;
    r.compute_x = n;
    long long total_steps = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int e = 0; e < opt.episodes; ++e) {
        const uint32_t seed = opt.seed_base + static_cast<uint32_t>(e) * 17u;
        int steps = 0;
        const bool ok = run_episode(org, n, opt.map, opt.max_steps, seed,
                                    opt.policy_noise, opt.cand_noise, &steps);
        if (ok) ++r.success;
        r.bits.push_back(ok ? 1 : 0);
        total_steps += steps;
    }
    const auto t1 = std::chrono::steady_clock::now();
    r.sr = opt.episodes > 0 ? static_cast<double>(r.success) / opt.episodes : 0.0;
    r.mean_steps = opt.episodes > 0 ? static_cast<double>(total_steps) / opt.episodes : 0.0;
    r.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--maze") opt.maze = next();
        else if (a == "--out") opt.out = next();
        else if (a == "--episodes") opt.episodes = std::max(1, std::stoi(next()));
        else if (a == "--map") opt.map = std::stoi(next());
        else if (a == "--steps") opt.max_steps = std::stoi(next());
        else if (a == "--seed") opt.seed_base = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--policy-noise") opt.policy_noise = std::stod(next());
        else if (a == "--cand-noise") opt.cand_noise = std::stod(next());
        else if (a == "--n") {
            opt.ns.clear();
            std::string s = next();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos) comma = s.size();
                opt.ns.push_back(std::stoi(s.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        } else if (a == "--quick") opt.quick = true;
        else if (a == "--help") {
            std::cout << "usage: bench_best_of_n [--quick] [--n 1,4,16,64] "
                         "[--episodes N] [--map 21] [--steps 400] "
                         "[--policy-noise 0.8] [--cand-noise 0.5] [--out path]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.ns = {1, 4};
        opt.episodes = 10;
        opt.map = 11;
        opt.max_steps = 250;
    }
    if (opt.out.empty()) opt.out = "runs/best_of_n_" + today() + ".json";
    opt.maze = resolve_path(opt.maze);
    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    CellularOrganism org = CellularOrganism::load_checkpoint_bin(opt.maze);
    if (org.cells.empty()) { std::cerr << "cannot load " << opt.maze << "\n"; return 2; }
    org.compile();

    std::cout << "E-D best-of-N | map=" << opt.map << " steps=" << opt.max_steps
              << " episodes=" << opt.episodes << " policy_noise=" << opt.policy_noise
              << " cand_noise=" << opt.cand_noise << "\n";

    std::vector<Result> results;
    for (int n : opt.ns) {
        std::cout << "  run N=" << n << " ... " << std::flush;
        Result r = evaluate(org, opt, n);
        std::cout << "SR=" << r.sr << " steps=" << r.mean_steps
                  << " wall=" << r.wall_ms << "ms\n";
        results.push_back(std::move(r));
    }

    const Result* base = nullptr;
    for (const auto& r : results) if (r.n == 1) base = &r;

    // 预注册判据：N=16 相对 N=1 提升 >=10pp，且 compute <=20x
    const Result* n16 = nullptr;
    for (const auto& r : results) if (r.n == 16) n16 = &r;
    bool gate = false;
    double delta16 = 0.0;
    if (base && n16) {
        delta16 = n16->sr - base->sr;
        gate = (delta16 >= 0.10) && (n16->compute_x <= 20.0);
    }

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }
    os << std::fixed << std::setprecision(6);
    os << "{\n  \"experiment\":\"E-D best-of-N oracle verifier\",\n";
    os << "  \"generated_at\":\"" << today() << "\",\n";
    os << "  \"commit\":\"" << json_escape(commit) << "\",\n";
    os << "  \"config\":{\"map\":" << opt.map << ",\"max_steps\":" << opt.max_steps
       << ",\"episodes\":" << opt.episodes << ",\"policy_noise\":" << opt.policy_noise
       << ",\"cand_noise\":" << opt.cand_noise << ",\"verifier\":\"oracle_1step\"},\n";
    os << "  \"results\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        if (i) os << ",";
        os << "\n    {\"n\":" << r.n << ",\"sr\":" << r.sr
           << ",\"success\":" << r.success << ",\"episodes\":" << r.episodes
           << ",\"mean_steps\":" << r.mean_steps
           << ",\"wall_ms\":" << r.wall_ms
           << ",\"compute_x\":" << r.compute_x << ",\"bits\":[";
        for (size_t j = 0; j < r.bits.size(); ++j) {
            if (j) os << ",";
            os << r.bits[j];
        }
        os << "]}";
    }
    os << "\n  ],\n  \"gate\":{\"ED\":\"" << (gate ? "PASS" : "FAIL")
       << "\",\"delta_n16_vs_n1\":" << delta16
       << ",\"compute_n16\":" << (n16 ? n16->compute_x : 0.0) << "}\n}\n";
    os.close();

    std::cout << "\n== summary ==\n";
    for (const auto& r : results) {
        std::cout << "  N=" << std::setw(3) << r.n << " SR=" << r.sr
                  << " steps=" << r.mean_steps << " compute=" << r.compute_x << "x\n";
    }
    std::cout << "GATE: ED " << (gate ? "PASS" : "FAIL")
              << " (N=16 vs N=1 delta=" << delta16
              << ", compute=" << (n16 ? n16->compute_x : 0.0) << "x)\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";
    return 0;
}
