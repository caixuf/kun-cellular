// ============================================================================
// bench_cartpole_vs_pd.cpp — W1b shared cold eval: Gate-6 named PD vs cellular
//
// Slices (identical seeds for both controllers):
//   T4 ID:     default env, seeds 9000+i*17, N=20, ms=300
//   T4 OOD:    masspole=0.2 length=0.7 force_noise=2, seeds 9100+i*19, N=20, ms=300
//   Pack S:    masspole=0.2 length=0.7 force_noise=12, seeds 9300+i*29, N=40, ms=300
//
// Named PD formula FROZEN from tests/test_flow_cartpole_gate6_shadow.cpp pd_expert.
// Win: surv_cellular > surv_PD on S (strict) AND T4 floors ID>=19/20 OOD>=18/20.
//
// Compile:
//   g++ -O3 -march=native -std=c++20 -I include -I . \
//       tools/bench_cartpole_vs_pd.cpp -o build/bench_cartpole_vs_pd -pthread -lm
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "tasks/control/cart_pole_task.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace kun;

// Gate-6 pd_expert — byte-match formula; do not retune.
static CellularOrganism::ActionOutputs pd_expert(const std::vector<float>& obs) {
    const double theta = static_cast<double>(obs[0]) * 0.35;
    const double theta_dot = static_cast<double>(obs[1]) * 3.0;
    const double x = static_cast<double>(obs[2]) * 2.4;
    const double x_dot = static_cast<double>(obs[3]) * 3.0;
    const double u = std::clamp(
        12.0 * theta + 2.5 * theta_dot + 1.2 * x + 1.0 * x_dot, -1.0, 1.0);
    CellularOrganism::ActionOutputs a;
    if (u >= 0.0) {
        a.positive_action = u;
        a.negative_action = 0.0;
    } else {
        a.positive_action = 0.0;
        a.negative_action = -u;
    }
    return a;
}

struct EpResult {
    bool survived{false};
    int steps{0};
};

static EpResult run_pd(CartPoleBalanceTask& env, uint32_t seed, int max_steps) {
    env.reset(seed);
    EpResult ep;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        auto acts = pd_expert(obs);
        auto res = env.step_continuous(acts);
        if (res.done) {
            ep.survived = false;
            ep.steps = res.steps;
            return ep;
        }
    }
    ep.survived = true;
    ep.steps = max_steps;
    return ep;
}

static EpResult run_cellular(CellularOrganism& org, CartPoleBalanceTask& env,
                             uint32_t seed, int max_steps) {
    env.reset(seed);
    org.reset_state(false);
    EpResult ep;
    for (int t = 0; t < max_steps; ++t) {
        auto obs = env.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        auto res = env.step_continuous(acts);
        if (res.done) {
            ep.survived = false;
            ep.steps = res.steps;
            return ep;
        }
    }
    ep.survived = true;
    ep.steps = max_steps;
    return ep;
}

struct SliceStats {
    std::string name;
    int n{0};
    int pd_surv{0};
    int cell_surv{0};
    double pd_mean_steps{0};
    double cell_mean_steps{0};
    std::vector<uint32_t> seeds;
    std::vector<int> pd_steps;
    std::vector<int> cell_steps;
};

static SliceStats eval_slice(const char* name, CartPoleBalanceTask& env,
                             CellularOrganism& org,
                             const std::vector<uint32_t>& seeds, int max_steps) {
    SliceStats s;
    s.name = name;
    s.n = static_cast<int>(seeds.size());
    s.seeds = seeds;
    double pd_sum = 0, cell_sum = 0;
    for (uint32_t seed : seeds) {
        auto pd = run_pd(env, seed, max_steps);
        auto cell = run_cellular(org, env, seed, max_steps);
        if (pd.survived) s.pd_surv++;
        if (cell.survived) s.cell_surv++;
        pd_sum += pd.steps;
        cell_sum += cell.steps;
        s.pd_steps.push_back(pd.steps);
        s.cell_steps.push_back(cell.steps);
    }
    s.pd_mean_steps = pd_sum / std::max(1, s.n);
    s.cell_mean_steps = cell_sum / std::max(1, s.n);
    return s;
}

static void print_slice(const SliceStats& s) {
    std::printf("[%s] N=%d ms_mean PD=%.1f cell=%.1f | surv PD=%d/%d cell=%d/%d\n",
                s.name.c_str(), s.n, s.pd_mean_steps, s.cell_mean_steps,
                s.pd_surv, s.n, s.cell_surv, s.n);
}

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

static void write_slice_json(std::ostream& o, const SliceStats& s, int indent) {
    std::string pad(indent, ' ');
    std::string pad2(indent + 2, ' ');
    o << pad << "{\n";
    o << pad2 << "\"name\": \"" << json_escape(s.name) << "\",\n";
    o << pad2 << "\"n\": " << s.n << ",\n";
    o << pad2 << "\"pd_surv\": " << s.pd_surv << ",\n";
    o << pad2 << "\"cellular_surv\": " << s.cell_surv << ",\n";
    o << pad2 << "\"pd_mean_steps\": " << s.pd_mean_steps << ",\n";
    o << pad2 << "\"cellular_mean_steps\": " << s.cell_mean_steps << ",\n";
    o << pad2 << "\"seeds\": [";
    for (size_t i = 0; i < s.seeds.size(); ++i) {
        if (i) o << ", ";
        o << s.seeds[i];
    }
    o << "],\n";
    o << pad2 << "\"pd_steps\": [";
    for (size_t i = 0; i < s.pd_steps.size(); ++i) {
        if (i) o << ", ";
        o << s.pd_steps[i];
    }
    o << "],\n";
    o << pad2 << "\"cellular_steps\": [";
    for (size_t i = 0; i < s.cell_steps.size(); ++i) {
        if (i) o << ", ";
        o << s.cell_steps[i];
    }
    o << "]\n";
    o << pad << "}";
}

int main(int argc, char** argv) {
    const char* bin_path = "checkpoints/cartpole_balance_champion.bin";
    const char* out_path = "runs/cartpole_vs_pd_w1b.json";
    const int max_steps = 300;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bin" && i + 1 < argc) bin_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--help") {
            std::printf("Usage: %s [--bin PATH] [--out JSON]\n", argv[0]);
            return 0;
        }
    }

    CellularOrganism org = CellularOrganism::load_checkpoint_bin(bin_path);
    if (org.cells.empty()) {
        std::fprintf(stderr, "FAIL: cannot load %s\n", bin_path);
        return 2;
    }
    org.compile();

    CartPoleBalanceTask id_env;
    id_env.set_max_steps(max_steps);
    std::vector<uint32_t> id_seeds;
    for (int i = 0; i < 20; ++i)
        id_seeds.push_back(9000u + static_cast<uint32_t>(i) * 17u);

    CartPoleBalanceTask::Params ood_p;
    ood_p.masspole = 0.2;
    ood_p.length = 0.7;
    ood_p.force_noise = 2.0;
    CartPoleBalanceTask ood_env(ood_p);
    ood_env.set_max_steps(max_steps);
    std::vector<uint32_t> ood_seeds;
    for (int i = 0; i < 20; ++i)
        ood_seeds.push_back(9100u + static_cast<uint32_t>(i) * 19u);

    CartPoleBalanceTask::Params s_p;
    s_p.masspole = 0.2;
    s_p.length = 0.7;
    s_p.force_noise = 12.0;
    CartPoleBalanceTask s_env(s_p);
    s_env.set_max_steps(max_steps);
    std::vector<uint32_t> s_seeds;
    for (int i = 0; i < 40; ++i)
        s_seeds.push_back(9300u + static_cast<uint32_t>(i) * 29u);

    std::printf("======================================================================\n");
    std::printf("  W1b CartPole vs Gate-6 named PD | bin=%s\n", bin_path);
    std::printf("  cells=%zu syns=%zu | max_steps=%d\n",
                org.cells.size(), org.synapses.size(), max_steps);
    std::printf("======================================================================\n");

    auto t4_id = eval_slice("T4_ID", id_env, org, id_seeds, max_steps);
    print_slice(t4_id);
    auto t4_ood = eval_slice("T4_OOD", ood_env, org, ood_seeds, max_steps);
    print_slice(t4_ood);
    auto pack_s = eval_slice("Pack_S", s_env, org, s_seeds, max_steps);
    print_slice(pack_s);

    const bool t4_id_floor = t4_id.cell_surv >= 19;
    const bool t4_ood_floor = t4_ood.cell_surv >= 18;
    const bool s_strict = pack_s.cell_surv > pack_s.pd_surv;
    const bool win = t4_id_floor && t4_ood_floor && s_strict;

    std::printf("----------------------------------------------------------------------\n");
    std::printf("  T4 floors: ID %d/20 (>=19? %s) OOD %d/20 (>=18? %s)\n",
                t4_id.cell_surv, t4_id_floor ? "YES" : "NO",
                t4_ood.cell_surv, t4_ood_floor ? "YES" : "NO");
    std::printf("  Pack S: cellular=%d PD=%d strict_gt=%s\n",
                pack_s.cell_surv, pack_s.pd_surv, s_strict ? "YES" : "NO");
    std::printf("  VERDICT: %s\n", win ? "WIN (Prefer A)" : "NO_WIN");
    std::printf("======================================================================\n");

    std::ofstream jf(out_path);
    if (!jf) {
        std::fprintf(stderr, "WARN: cannot write %s\n", out_path);
    } else {
        jf << "{\n";
        jf << "  \"protocol\": \"W1b CartPole vs Gate-6 named PD\",\n";
        jf << "  \"bin\": \"" << json_escape(bin_path) << "\",\n";
        jf << "  \"cells\": " << org.cells.size() << ",\n";
        jf << "  \"synapses\": " << org.synapses.size() << ",\n";
        jf << "  \"max_steps\": " << max_steps << ",\n";
        jf << "  \"named_pd\": \"u=clamp(12*theta+2.5*theta_dot+1.2*x+1.0*x_dot,-1,1) Gate-6 split\",\n";
        jf << "  \"t4_id\": ";
        write_slice_json(jf, t4_id, 2);
        jf << ",\n";
        jf << "  \"t4_ood\": ";
        write_slice_json(jf, t4_ood, 2);
        jf << ",\n";
        jf << "  \"pack_s\": ";
        write_slice_json(jf, pack_s, 2);
        jf << ",\n";
        jf << "  \"floors\": {\n";
        jf << "    \"t4_id_ok\": " << (t4_id_floor ? "true" : "false") << ",\n";
        jf << "    \"t4_ood_ok\": " << (t4_ood_floor ? "true" : "false") << "\n";
        jf << "  },\n";
        jf << "  \"pack_s_strict_gt\": " << (s_strict ? "true" : "false") << ",\n";
        jf << "  \"prefer_a_win\": " << (win ? "true" : "false") << "\n";
        jf << "}\n";
        jf.close();
        std::printf("  Wrote %s\n", out_path);
    }

    return win ? 0 : 1;
}
