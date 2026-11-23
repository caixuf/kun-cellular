// ============================================================================
// bench_ttc_scaling.cpp — E-A 推理时计算缩放 (Test-Time Compute Scaling)
//
// 预注册：docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md §2 E-A
//
// 只回答一个问题：在固定拓扑/权重下，增加 relaxation K 是否带来可复现的
// 成功率提升（H1），且该提升不能由“多算几步”本身或随机权重解释。
//
// 任务：
//   maze     11x11, 250 steps, geodesic bearing（主判据，K=1 有 headroom）
//   cartpole 300 steps ID + 500 steps OOD-stress（次判据）
//   fluid    复刻 tools/train_fluid_damper.py 的三相 cost（仅次级指标，不参与 H1）
//
// 对照：
//   random   champion 拓扑 + 随机权重/参数
//   shuffled champion 拓扑 + 打乱 op 类型（保留 sense/act）
//
// 输出：runs/ttc_scaling_<date>.json
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"
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
#include <sstream>
#include <string>
#include <vector>

using namespace kun;

namespace {

struct Options {
    std::string maze = "checkpoints/maze_navigation_champion.bin";
    std::string cartpole = "checkpoints/cartpole_balance_champion.bin";
    std::string fluid = "checkpoints/fluid_damper_champion.bin";
    std::string out;
    std::vector<int> ks{1, 2, 4, 8, 16, 32};
    std::vector<double> dampings{0.5, 0.8, 1.0};
    int episodes = 100;
    uint32_t seed_base = 70000;
    bool quick = false;
    bool require_pass = false;
};

struct TaskResult {
    std::string task;
    std::string variant;  // champion/random/shuffled
    int k = 1;
    double damping = 1.0;
    int episodes = 0;
    int ok = 0;
    double sr = 0.0;
    double baseline_sr = 0.0;
    double delta_sr = 0.0;
    double p_mcnemar = 1.0;
    double mean_steps = 0.0;
    double mean_relax_steps = 0.0;
    double converged_rate = 0.0;
    double bibo_rate = 1.0;
    double wall_ms = 0.0;
    double fluid_cost = 0.0;
    double best_k1_sr = 0.0;
    double delta_vs_best_k1 = 0.0;
    double p_vs_best_k1 = 1.0;
    std::vector<int> bits;
};

struct EvalStats {
    int ok = 0;
    double sum_steps = 0.0;
    double sum_relax_steps = 0.0;
    int converged = 0;
    int bibo_ok = 0;
    std::vector<int> bits;
    double fluid_cost = 0.0;
};

static double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(c); }
        else if (c == '\n') o += "\\n";
        else o.push_back(c);
    }
    return o;
}

static std::string today() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d", &tm);
    return buf;
}

static std::string resolve_path(const std::string& p) {
    if (std::filesystem::exists(p)) return p;
    for (const char* prefix : {"../", "../../", "../../../"}) {
        std::string cand = std::string(prefix) + p;
        if (std::filesystem::exists(cand)) return cand;
    }
    return p;
}

static CellularOrganism load_org(const std::string& path) {
    CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
    if (!org.cells.empty()) org.compile();
    return org;
}

static CellularOrganism randomize_weights(const CellularOrganism& src, uint32_t seed) {
    CellularOrganism org = src;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> w(-1.5, 1.5);
    for (auto& s : org.synapses) {
        if (!s.is_active) continue;
        s.weight = w(rng);
        s.initial_weight = s.weight;
    }
    for (auto& c : org.cells) {
        c.param1 = w(rng);
        c.param2 = w(rng);
    }
    org.compile();
    return org;
}

static CellularOrganism shuffle_op_types(const CellularOrganism& src, uint32_t seed) {
    CellularOrganism org = src;
    std::vector<size_t> idx;
    for (size_t i = 0; i < org.cells.size(); ++i) {
        const auto t = static_cast<uint8_t>(org.cells[i].type);
        if (t > 4 && t < 30) idx.push_back(i);  // 保留 sense / act
    }
    std::vector<CellType> types;
    types.reserve(idx.size());
    for (size_t i : idx) types.push_back(org.cells[i].type);
    std::shuffle(types.begin(), types.end(), std::mt19937(seed));
    for (size_t i = 0; i < idx.size(); ++i) org.cells[idx[i]].type = types[i];
    org.compile();
    return org;
}

static CellularOrganism::ActionOutputs forward_relax(CellularOrganism& org, const double* in, size_t in_dim,
                                   int k, double damping, RelaxationTelemetry* telem) {
    RelaxationConfig cfg;
    cfg.max_steps = static_cast<size_t>(std::max(1, k));
    cfg.damping_factor = damping;
    cfg.convergence_tol = 1e-5;
    cfg.enable_hebbian = false;
    return org.forward_with_relaxation(in, in_dim, cfg, telem);
}

static EvalStats eval_maze(CellularOrganism& org, int k, double damping,
                           int episodes, uint32_t seed_base) {
    EvalStats out;
    out.bits.reserve(episodes);
    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = seed_base + static_cast<uint32_t>(e) * 17u;
        MazeTask task(11, 11, seed, 250, 0.15f);
        task.set_use_geodesic_bearing(true);
        task.reset(seed);
        org.reset_state(false);

        bool success = false;
        int steps = 0;
        for (; steps < 250; ++steps) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            RelaxationTelemetry telem;
            auto acts = forward_relax(org, inps, 4, k, damping, &telem);
            out.sum_relax_steps += static_cast<double>(telem.actual_steps);
            out.converged += telem.converged ? 1 : 0;
            out.bibo_ok += telem.bibo_stable ? 1 : 0;
            auto res = task.step_continuous(acts);
            if (res.done) { success = res.success; ++steps; break; }
        }
        out.ok += success ? 1 : 0;
        out.bits.push_back(success ? 1 : 0);
        out.sum_steps += steps;
    }
    return out;
}

static EvalStats eval_cartpole(CellularOrganism& org, int k, double damping,
                               int episodes, uint32_t seed_base) {
    EvalStats out;
    out.bits.reserve(episodes);

    CartPoleBalanceTask::Params p;
    p.masspole = 0.2;
    p.length = 0.7;
    p.force_noise = 2.0;
    const int max_steps = 500;

    for (int e = 0; e < episodes; ++e) {
        const uint32_t seed = seed_base + static_cast<uint32_t>(e) * 19u;
        CartPoleBalanceTask env(p);
        env.set_max_steps(max_steps);
        env.reset(seed);
        org.reset_state(false);

        bool success = true;
        int steps = 0;
        for (; steps < max_steps; ++steps) {
            auto obs = env.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            RelaxationTelemetry telem;
            auto acts = forward_relax(org, inps, 4, k, damping, &telem);
            out.sum_relax_steps += static_cast<double>(telem.actual_steps);
            out.converged += telem.converged ? 1 : 0;
            out.bibo_ok += telem.bibo_stable ? 1 : 0;
            auto res = env.step_continuous(acts);
            if (res.done) { success = false; ++steps; break; }
        }
        out.ok += success ? 1 : 0;
        out.bits.push_back(success ? 1 : 0);
        out.sum_steps += steps;
    }
    return out;
}

// 复刻 tools/train_fluid_damper.py 的三相 cost（次级指标，不参与 H1）
static EvalStats eval_fluid(CellularOrganism& org, int k, double damping,
                            int episodes, uint32_t seed_base) {
    struct Phase { double wind; double mu; double rho; };
    const Phase phases[3] = {
        {300.0, 0.85, 1.225},
        {50.0, 0.35, 1000.0},
        {0.0, 0.90, 0.0},
    };
    EvalStats out;
    out.bits.reserve(episodes);
    double total_cost = 0.0;
    const double dt = 0.05;

    for (int e = 0; e < episodes; ++e) {
        for (const auto& ph : phases) {
            org.reset_state(false);
            double v = 12.0, x = 0.0, y = 0.4, psi = 0.0;
            double cum_cte = 0.0, max_cte = 0.0, prev_damper = 0.0, jitter = 0.0;
            const int steps = 200;
            for (int s = 0; s < steps; ++s) {
                const double t = s * dt;
                const double ref_y = 3.0 * std::sin(2.0 * M_PI * x / 250.0);
                const double ref_psi = std::atan2(
                    3.0 * 2.0 * M_PI / 250.0 * std::cos(2.0 * M_PI * x / 250.0), 1.0);
                const double cte = y - ref_y;
                const double dpsi = psi - ref_psi;
                const double wind = ph.wind > 0.0 ? ph.wind * std::sin(t * 1.5) : 0.0;
                const double drag = 0.5 * ph.rho * 0.3 * 2.2 * v * v * 0.001;

                double inps[6] = {
                    clampd(cte / 2.0, -1.0, 1.0),
                    clampd(dpsi / 0.5, -1.0, 1.0),
                    clampd(wind / 400.0, -1.0, 1.0),
                    clampd(drag / 50.0, -1.0, 1.0),
                    clampd((ph.mu - 0.5) * 2.0, -1.0, 1.0),
                    0.0,
                };
                RelaxationTelemetry telem;
                auto acts = forward_relax(org, inps, 6, k, damping, &telem);
                out.sum_relax_steps += static_cast<double>(telem.actual_steps);
                out.converged += telem.converged ? 1 : 0;
                out.bibo_ok += telem.bibo_stable ? 1 : 0;

                const double damper_gain = 1.0 + std::max(0.0, acts.positive_action) * 0.8;
                double steer = -0.6 * cte / damper_gain - 0.4 * dpsi;
                steer = clampd(steer, -0.45, 0.45);
                const double lat_force = -cte * 150.0 + wind;
                const double side_slip = lat_force / (1500.0 * 9.8 * ph.mu);

                x += v * std::cos(psi) * dt;
                y += (v * std::sin(psi) + side_slip * 2.0) * dt;
                psi += (v / 2.8) * std::tan(steer) * dt;

                cum_cte += std::abs(cte);
                max_cte = std::max(max_cte, std::abs(cte));
                jitter += std::abs(acts.positive_action - prev_damper);
                prev_damper = acts.positive_action;
                out.sum_steps += 1.0;
            }
            total_cost += (cum_cte / steps) * 20.0 + max_cte * 10.0 + (jitter / steps) * 5.0;
        }
        out.bits.push_back(0);
    }
    out.fluid_cost = episodes > 0 ? total_cost / episodes : 0.0;
    return out;
}

static double mcnemar_exact(int b, int c) {
    const int n = b + c;
    if (n == 0) return 1.0;
    const int k = std::min(b, c);
    double tail = 0.0;
    for (int i = 0; i <= k; ++i) {
        tail += std::exp(std::lgamma(n + 1.0) - std::lgamma(i + 1.0) -
                         std::lgamma(n - i + 1.0) - n * std::log(2.0));
    }
    return std::min(1.0, 2.0 * tail);
}

static void attach_baseline(std::vector<TaskResult>& rows, const std::string& task,
                            const std::string& variant) {
    for (auto& row : rows) {
        if (row.task != task || row.variant != variant) continue;
        const TaskResult* base = nullptr;
        for (const auto& b : rows) {
            if (b.task == task && b.variant == variant && b.k == 1 &&
                std::abs(b.damping - row.damping) < 1e-9) {
                base = &b;
                break;
            }
        }
        if (!base || base == &row) {
            row.baseline_sr = row.sr;
            row.delta_sr = 0.0;
            row.p_mcnemar = 1.0;
            continue;
        }
        row.baseline_sr = base->sr;
        row.delta_sr = row.sr - base->sr;
        int b = 0, c = 0;
        for (size_t i = 0; i < row.bits.size() && i < base->bits.size(); ++i) {
            if (base->bits[i] && !row.bits[i]) ++b;
            if (!base->bits[i] && row.bits[i]) ++c;
        }
        row.p_mcnemar = mcnemar_exact(b, c);
    }
}

static void attach_best_baseline(std::vector<TaskResult>& rows, const std::string& task) {
    const TaskResult* best = nullptr;
    for (const auto& r : rows) {
        if (r.task == task && r.variant == "champion" && r.k == 1) {
            if (!best || r.sr > best->sr) best = &r;
        }
    }
    if (!best) return;
    for (auto& r : rows) {
        if (r.task != task || r.variant != "champion") continue;
        r.best_k1_sr = best->sr;
        r.delta_vs_best_k1 = r.sr - best->sr;
        int b = 0, c = 0;
        for (size_t i = 0; i < r.bits.size() && i < best->bits.size(); ++i) {
            if (best->bits[i] && !r.bits[i]) ++b;
            if (!best->bits[i] && r.bits[i]) ++c;
        }
        r.p_vs_best_k1 = (r.k == 1 && std::abs(r.damping - best->damping) < 1e-9)
                             ? 1.0
                             : mcnemar_exact(b, c);
    }
}

static void write_results(std::ostream& os, const std::vector<TaskResult>& rows) {
    os << "[";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (i) os << ",";
        os << "\n    {\"task\":\"" << json_escape(r.task)
           << "\",\"variant\":\"" << json_escape(r.variant)
           << "\",\"k\":" << r.k
           << ",\"damping\":" << r.damping
           << ",\"episodes\":" << r.episodes
           << ",\"ok\":" << r.ok
           << ",\"sr\":" << r.sr
           << ",\"baseline_sr\":" << r.baseline_sr
           << ",\"delta_sr\":" << r.delta_sr
           << ",\"p_mcnemar\":" << r.p_mcnemar
           << ",\"mean_steps\":" << r.mean_steps
           << ",\"mean_relax_steps\":" << r.mean_relax_steps
           << ",\"converged_rate\":" << r.converged_rate
           << ",\"bibo_rate\":" << r.bibo_rate
           << ",\"wall_ms\":" << r.wall_ms
           << ",\"fluid_cost\":" << r.fluid_cost
           << ",\"best_k1_sr\":" << r.best_k1_sr
           << ",\"delta_vs_best_k1\":" << r.delta_vs_best_k1
           << ",\"p_vs_best_k1\":" << r.p_vs_best_k1
           << ",\"bits\":[";
        for (size_t j = 0; j < r.bits.size(); ++j) {
            if (j) os << ",";
            os << r.bits[j];
        }
        os << "]}";
    }
    os << "\n  ]";
}

static bool parse_int_list(const std::string& s, std::vector<int>& out) {
    std::stringstream ss(s);
    std::string tok;
    out.clear();
    while (std::getline(ss, tok, ',')) {
        try { out.push_back(std::stoi(tok)); } catch (...) { return false; }
    }
    return !out.empty();
}

static bool parse_double_list(const std::string& s, std::vector<double>& out) {
    std::stringstream ss(s);
    std::string tok;
    out.clear();
    while (std::getline(ss, tok, ',')) {
        try { out.push_back(std::stod(tok)); } catch (...) { return false; }
    }
    return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--maze") opt.maze = next();
        else if (a == "--cartpole") opt.cartpole = next();
        else if (a == "--fluid") opt.fluid = next();
        else if (a == "--out") opt.out = next();
        else if (a == "--episodes") opt.episodes = std::max(1, std::stoi(next()));
        else if (a == "--seed") opt.seed_base = static_cast<uint32_t>(std::stoul(next()));
        else if (a == "--k") { if (!parse_int_list(next(), opt.ks)) return 2; }
        else if (a == "--damping") { if (!parse_double_list(next(), opt.dampings)) return 2; }
        else if (a == "--quick") opt.quick = true;
        else if (a == "--require-pass") opt.require_pass = true;
        else if (a == "--help") {
            std::cout << "usage: bench_ttc_scaling [--quick] [--episodes N] [--k 1,2,4] "
                         "[--damping 0.5,1.0] [--out path] [--require-pass]\n";
            return 0;
        }
    }
    if (opt.quick) {
        opt.ks = {1, 4};
        opt.dampings = {1.0};
        opt.episodes = 10;
    }
    if (opt.out.empty()) opt.out = "runs/ttc_scaling_" + today() + ".json";
    opt.maze = resolve_path(opt.maze);
    opt.cartpole = resolve_path(opt.cartpole);
    opt.fluid = resolve_path(opt.fluid);

    const std::string commit = std::getenv("KUN_COMMIT") ? std::getenv("KUN_COMMIT") : "unknown";

    CellularOrganism maze_org = load_org(opt.maze);
    CellularOrganism cart_org = load_org(opt.cartpole);
    CellularOrganism fluid_org = load_org(opt.fluid);
    if (maze_org.cells.empty()) { std::cerr << "cannot load maze: " << opt.maze << "\n"; return 2; }
    if (cart_org.cells.empty()) { std::cerr << "cannot load cartpole: " << opt.cartpole << "\n"; return 2; }
    if (fluid_org.cells.empty()) { std::cerr << "cannot load fluid: " << opt.fluid << "\n"; return 2; }

    CellularOrganism maze_rand = randomize_weights(maze_org, 991);
    CellularOrganism cart_rand = randomize_weights(cart_org, 992);
    CellularOrganism maze_shuf = shuffle_op_types(maze_org, 993);
    CellularOrganism cart_shuf = shuffle_op_types(cart_org, 994);

    std::vector<TaskResult> rows;

    auto run_one = [&](const std::string& task, const std::string& variant,
                       CellularOrganism& org, int k, double damping) {
        const auto t0 = std::chrono::steady_clock::now();
        EvalStats st;
        if (task == "maze") st = eval_maze(org, k, damping, opt.episodes, opt.seed_base);
        else if (task == "cartpole") st = eval_cartpole(org, k, damping, opt.episodes, opt.seed_base);
        else st = eval_fluid(org, k, damping, opt.episodes, opt.seed_base);
        const auto t1 = std::chrono::steady_clock::now();

        TaskResult r;
        r.task = task;
        r.variant = variant;
        r.k = k;
        r.damping = damping;
        r.episodes = opt.episodes;
        r.ok = st.ok;
        r.sr = opt.episodes > 0 ? static_cast<double>(st.ok) / opt.episodes : 0.0;
        r.mean_steps = opt.episodes > 0 ? st.sum_steps / opt.episodes : 0.0;
        r.mean_relax_steps = opt.episodes > 0 ? st.sum_relax_steps / opt.episodes : 0.0;
        const double telem_n = std::max(1.0, st.sum_steps);
        r.converged_rate = st.converged / telem_n;
        r.bibo_rate = st.bibo_ok / telem_n;
        r.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.fluid_cost = st.fluid_cost;
        r.bits = st.bits;
        rows.push_back(std::move(r));
    };

    for (double d : opt.dampings) {
        for (int k : opt.ks) run_one("maze", "champion", maze_org, k, d);
        for (int k : opt.ks) run_one("cartpole", "champion", cart_org, k, d);
        for (int k : opt.ks) run_one("fluid", "champion", fluid_org, k, d);
    }

    // 反伪影：只跑 K={1,8,32} × damping={0.5,1.0}
    for (double d : {0.5, 1.0}) {
        for (int k : {1, 8, 32}) {
            run_one("maze", "random", maze_rand, k, d);
            run_one("cartpole", "random", cart_rand, k, d);
            run_one("maze", "shuffled", maze_shuf, k, d);
            run_one("cartpole", "shuffled", cart_shuf, k, d);
        }
    }

    attach_baseline(rows, "maze", "champion");
    attach_baseline(rows, "cartpole", "champion");
    attach_baseline(rows, "fluid", "champion");
    attach_baseline(rows, "maze", "random");
    attach_baseline(rows, "cartpole", "random");
    attach_baseline(rows, "maze", "shuffled");
    attach_baseline(rows, "cartpole", "shuffled");
    attach_best_baseline(rows, "maze");
    attach_best_baseline(rows, "cartpole");
    attach_best_baseline(rows, "fluid");

    // H1 机械判据（预注册）：同 damping 下 K* 相对 K=1 提升 >=10pp 且 p<0.05；
    // 同时要求 random/shuffled 对照不出现同类提升。
    bool gate = false;
    bool control_clean = true;
    TaskResult best;
    best.delta_sr = -1.0;
    for (const auto& r : rows) {
        if (r.variant != "champion") continue;
        if (r.task == "fluid") continue;  // 流体只有 cost，不进 H1
        if (r.delta_sr > best.delta_sr) best = r;
        if (r.delta_sr >= 0.10 && r.p_mcnemar < 0.05) gate = true;
    }
    for (const auto& r : rows) {
        if (r.variant != "random" && r.variant != "shuffled") continue;
        if (r.task == "fluid") continue;
        if (r.delta_sr >= 0.10 && r.p_mcnemar < 0.05) control_clean = false;
    }
    const bool h1_mechanical = gate && control_clean;

    // H1 能力判据（更严格，post-hoc）：必须超过该任务最佳 K=1 基线；
    // 这一条用来区分“真正的 test-time 能力增益”与“低 damping 下的收敛补偿”。
    bool cap_gate = false;
    TaskResult cap_best;
    cap_best.delta_vs_best_k1 = -1.0;
    for (const auto& r : rows) {
        if (r.variant != "champion" || r.task == "fluid") continue;
        if (r.delta_vs_best_k1 > cap_best.delta_vs_best_k1) cap_best = r;
        if (r.delta_vs_best_k1 >= 0.10 && r.p_vs_best_k1 < 0.05) cap_gate = true;
    }
    const bool h1_capability = cap_gate && control_clean;
    const bool h1_pass = h1_mechanical;  // 预注册门禁保持原口径；能力判据单独报告

    std::filesystem::path outp(opt.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream os(outp);
    if (!os) { std::cerr << "cannot write " << opt.out << "\n"; return 2; }

    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"experiment\": \"E-A test-time compute scaling\",\n";
    os << "  \"generated_at\": \"" << today() << "\",\n";
    os << "  \"commit\": \"" << json_escape(commit) << "\",\n";
    os << "  \"config\": {\"episodes\":" << opt.episodes
       << ",\"seed_base\":" << opt.seed_base
       << ",\"ks\":[";
    for (size_t i = 0; i < opt.ks.size(); ++i) { if (i) os << ","; os << opt.ks[i]; }
    os << "],\"dampings\":[";
    for (size_t i = 0; i < opt.dampings.size(); ++i) { if (i) os << ","; os << opt.dampings[i]; }
    os << "]},\n";
    os << "  \"assets\": {\"maze\":\"" << json_escape(opt.maze)
       << "\",\"cartpole\":\"" << json_escape(opt.cartpole)
       << "\",\"fluid\":\"" << json_escape(opt.fluid) << "\"},\n";
    os << "  \"results\": ";
    write_results(os, rows);
    os << ",\n";
    os << "  \"gate\": {\"H1\": \"" << (h1_pass ? "PASS" : "FAIL") << "\","
       << "\"H1_mechanical\": " << (h1_mechanical ? "true" : "false") << ","
       << "\"H1_capability_vs_best_k1\": " << (h1_capability ? "true" : "false") << ","
       << "\"control_clean\": " << (control_clean ? "true" : "false") << ","
       << "\"best_task\": \"" << json_escape(best.task) << "\","
       << "\"best_variant\": \"" << json_escape(best.variant) << "\","
       << "\"best_k\": " << best.k << ","
       << "\"best_damping\": " << best.damping << ","
       << "\"best_delta_sr\": " << best.delta_sr << ","
       << "\"best_p\": " << best.p_mcnemar << ","
       << "\"cap_best_task\": \"" << json_escape(cap_best.task) << "\","
       << "\"cap_best_k\": " << cap_best.k << ","
       << "\"cap_best_damping\": " << cap_best.damping << ","
       << "\"cap_best_delta_vs_best_k1\": " << cap_best.delta_vs_best_k1 << ","
       << "\"cap_best_p_vs_best_k1\": " << cap_best.p_vs_best_k1 << "}\n";
    os << "}\n";
    os.close();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "E-A summary (episodes=" << opt.episodes << ", commit=" << commit << ")\n";
    for (const auto& r : rows) {
        if (r.variant != "champion" && r.k != 1) continue;
        if (r.task == "fluid") {
            std::cout << "  " << r.task << "/" << r.variant << " K=" << r.k
                      << " d=" << r.damping << " cost=" << r.fluid_cost << "\n";
        } else {
            std::cout << "  " << r.task << "/" << r.variant << " K=" << r.k
                      << " d=" << r.damping << " SR=" << r.sr
                      << " dSR=" << r.delta_sr << " p=" << r.p_mcnemar
                      << " vsBestK1=" << r.delta_vs_best_k1
                      << " pBest=" << r.p_vs_best_k1 << "\n";
        }
    }
    std::cout << "GATE: H1(mechanical) " << (h1_mechanical ? "PASS" : "FAIL")
              << " | H1(capability vs best K=1) " << (h1_capability ? "PASS" : "FAIL")
              << " (best " << best.task << " K=" << best.k << " d=" << best.damping
              << " dSR=" << best.delta_sr << " p=" << best.p_mcnemar
              << "; capBest " << cap_best.task << " K=" << cap_best.k
              << " d=" << cap_best.damping << " dBest=" << cap_best.delta_vs_best_k1
              << " pBest=" << cap_best.p_vs_best_k1
              << ", control_clean=" << (control_clean ? "true" : "false") << ")\n";
    std::cout << "ARTIFACT: " << opt.out << "\n";

    if (opt.require_pass && !h1_pass) return 1;
    return 0;
}
