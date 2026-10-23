// ============================================================================
// train_cart_pole.cpp — CartPole balance trainer (W1b Prefer A: beat Gate-6 named PD)
//
// Prefer A: load cartpole_balance_champion.bin, evolve with stress OOD (force_noise=12)
// in fitness, write candidate to checkpoints/cartpole_balance_champion_w1b_*.bin.
// Does NOT overwrite cartpole_balance_champion.bin (promote only post-gate).
//
// Registered budget: POP=48 GENS=150 MAX_STEPS=300 seed=20261023
// Optional retry seed=20261024 same POP/GENS.
//
// Compile: g++ -O3 -march=native -std=c++20 -I include -I . \
//          tools/train_cart_pole.cpp -o build/train_cart_pole -pthread -lm
// ============================================================================

#include "tasks/control/cart_pole_task.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace kun;

static void enforce_strict_loop_gain(CellularOrganism& org, double target = 0.95) {
    struct Cycle { std::vector<std::pair<uint32_t, uint32_t>> edges; double gain{0.0}; };
    for (int iter = 0; iter < 32; ++iter) {
        auto rep = org.check_lyapunov_stability();
        if (rep.max_loop_gain < target) break;

        std::unordered_map<uint32_t, size_t> id2i;
        for (size_t i = 0; i < org.cells.size(); ++i) id2i[org.cells[i].id] = i;
        std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, double>>> adj;
        for (const auto& s : org.synapses) {
            if (!s.is_active) continue;
            if (id2i.count(s.from_cell_id) && id2i.count(s.to_cell_id))
                adj[s.from_cell_id].push_back({s.to_cell_id, s.weight});
        }

        std::unordered_map<uint32_t, int> vis;
        std::vector<uint32_t> path;
        std::vector<double> wp;
        Cycle best;
        std::function<void(uint32_t)> dfs = [&](uint32_t u) {
            vis[u] = 1; path.push_back(u);
            for (const auto& e : adj[u]) {
                wp.push_back(e.second);
                uint32_t v = e.first;
                if (vis[v] == 1) {
                    auto it = std::find(path.begin(), path.end(), v);
                    size_t st = static_cast<size_t>(std::distance(path.begin(), it));
                    double g = 1.0;
                    for (size_t k = st; k < path.size(); ++k)
                        g *= CellularOrganism::get_cell_operator_gain(org.cells[id2i.at(path[k])].type, 0.0);
                    for (size_t k = st; k < wp.size(); ++k) g *= std::fabs(wp[k]);
                    if (g > best.gain) {
                        best.gain = g; best.edges.clear();
                        for (size_t k = st; k + 1 < path.size(); ++k)
                            best.edges.push_back({path[k], path[k + 1]});
                        best.edges.push_back({path.back(), v});
                    }
                } else if (vis[v] == 0) {
                    dfs(v);
                }
                wp.pop_back();
            }
            path.pop_back(); vis[u] = 2;
        };
        for (const auto& c : org.cells) if (vis[c.id] == 0) dfs(c.id);
        if (best.edges.empty()) break;

        double f = std::pow(target / best.gain, 1.0 / static_cast<double>(best.edges.size()));
        std::unordered_set<uint64_t> es;
        for (const auto& e : best.edges) es.insert((static_cast<uint64_t>(e.first) << 32) | e.second);
        for (auto& s : org.synapses) {
            if (!s.is_active) continue;
            if (es.count((static_cast<uint64_t>(s.from_cell_id) << 32) | s.to_cell_id)) {
                s.weight *= f; s.initial_weight = s.weight;
            }
        }
        org.compile();
    }
    auto rep = org.check_lyapunov_stability();
    std::printf("  [Lyapunov] strict loop-gain fix rho=%.5f (target<%.2f)\n", rep.max_loop_gain, target);
}

static void usage(const char* argv0) {
    std::printf(
        "Usage: %s [--seed N] [--pop N] [--gens N] [--max-steps N]\n"
        "          [--start PATH] [--out PATH] [--handcrafted]\n"
        "W1b defaults: seed=20261023 pop=48 gens=150 max-steps=300\n"
        "  --start  load champ (default checkpoints/cartpole_balance_champion.bin)\n"
        "  --handcrafted  ignore --start; seed from HANDCRAFTED_PROGENITOR\n"
        "  --out    candidate bin (default checkpoints/cartpole_balance_champion_w1b_<seed>.bin)\n",
        argv0);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    size_t POP = 48;
    size_t GENS = 150;
    int MAX_STEPS = 300;
    uint32_t SEED = 20261023;
    std::string start_path = "checkpoints/cartpole_balance_champion.bin";
    std::string out_path;
    bool use_handcrafted = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seed") SEED = static_cast<uint32_t>(std::stoul(need("--seed")));
        else if (a == "--pop") POP = static_cast<size_t>(std::stoul(need("--pop")));
        else if (a == "--gens") GENS = static_cast<size_t>(std::stoul(need("--gens")));
        else if (a == "--max-steps") MAX_STEPS = std::stoi(need("--max-steps"));
        else if (a == "--start") start_path = need("--start");
        else if (a == "--out") out_path = need("--out");
        else if (a == "--handcrafted") use_handcrafted = true;
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (out_path.empty()) {
        out_path = "checkpoints/cartpole_balance_champion_w1b_" + std::to_string(SEED) + ".bin";
    }

    // Rails lock: do not raise beyond registered table without Prefer bounce.
    if (POP > 48 || GENS > 150) {
        std::fprintf(stderr, "REFUSE: POP=%zu GENS=%zu exceeds registered budget POP48/GENS150\n",
                     POP, GENS);
        return 2;
    }

    std::printf("==========================================================\n");
    std::printf("  CartPole W1b Prefer A · beat Gate-6 named PD on pack S\n");
    std::printf("  POP=%zu GENS=%zu MAX_STEPS=%d seed=%u\n", POP, GENS, MAX_STEPS, SEED);
    std::printf("  start=%s\n", use_handcrafted ? "HANDCRAFTED_PROGENITOR" : start_path.c_str());
    std::printf("  out=%s\n", out_path.c_str());
    std::printf("==========================================================\n");

    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_28;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;

    // Three envs: ID / mild OOD (T4-like) / stress OOD (pack S physics)
    CartPoleBalanceTask id_env;
    CartPoleBalanceTask::Params mild_p;
    mild_p.masspole = 0.2; mild_p.length = 0.7; mild_p.force_noise = 2.0;
    CartPoleBalanceTask mild_env(mild_p);
    CartPoleBalanceTask::Params stress_p;
    stress_p.masspole = 0.2; stress_p.length = 0.7; stress_p.force_noise = 12.0;
    CartPoleBalanceTask stress_env(stress_p);
    id_env.set_max_steps(MAX_STEPS);
    mild_env.set_max_steps(MAX_STEPS);
    stress_env.set_max_steps(MAX_STEPS);

    // Fitness seeds: ID must not be abandoned; stress must see force_noise>=12 / pack S.
    std::vector<uint32_t> id_seeds, mild_seeds, stress_seeds;
    for (uint32_t s = 101; s <= 110; ++s) id_seeds.push_back(s);
    for (uint32_t s = 201; s <= 210; ++s) mild_seeds.push_back(s);
    // Pack S seeds (N=40) so selection can beat PD on the decisive metric.
    for (int i = 0; i < 40; ++i)
        stress_seeds.push_back(9300u + static_cast<uint32_t>(i) * 29u);

    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);

    // ---- Start: load champ or HANDCRAFTED ----
    std::string start_mode = "HANDCRAFTED_PROGENITOR";
    if (!use_handcrafted) {
        CellularOrganism loaded = CellularOrganism::load_checkpoint_bin(start_path);
        if (loaded.cells.empty() || !loaded.compile()) {
            std::fprintf(stderr, "WARN: failed to load %s; falling back to HANDCRAFTED\n",
                         start_path.c_str());
            use_handcrafted = true;
        } else {
            for (auto& s : loaded.synapses) s.initial_weight = s.weight;
            loaded.compile();
            engine.population()[0] = loaded;
            for (size_t i = 1; i < engine.population().size(); ++i) {
                auto org = loaded;
                for (int m = 0; m < 2; ++m) engine.mutate(org);
                for (auto& s : org.synapses) s.initial_weight = s.weight;
                org.compile();
                engine.population()[i] = org;
            }
            start_mode = "LOAD_CHAMP:" + start_path;
            std::printf("[start] seeded pop from %s (%zu cells / %zu syns)\n",
                        start_path.c_str(), loaded.cells.size(), loaded.synapses.size());
        }
    }
    if (use_handcrafted) {
        start_mode = "HANDCRAFTED_PROGENITOR";
        std::printf("[start] HANDCRAFTED_PROGENITOR (engine default pop)\n");
    }

    auto fitness_of = [&](CellularOrganism& org) -> double {
        // Weighted: keep ID floor, push stress survival (decisive for Prefer A).
        auto mid = id_env.evaluate_organism(org, id_seeds, MAX_STEPS, true);
        auto mm = mild_env.evaluate_organism(org, mild_seeds, MAX_STEPS, true);
        auto ms = stress_env.evaluate_organism(org, stress_seeds, MAX_STEPS, true);
        // Emphasize stress success_rate + mean_fitness; retain ID/mild.
        return 0.25 * mid.mean_fitness
             + 0.15 * mm.mean_fitness
             + 0.20 * ms.mean_fitness
             + 0.40 * ms.success_rate;
    };

    double base_fit = 0.0;
    {
        auto& pop = engine.population();
        base_fit = fitness_of(pop[0]);
        auto mid = id_env.evaluate_organism(pop[0], id_seeds, MAX_STEPS, false);
        auto ms = stress_env.evaluate_organism(pop[0], stress_seeds, MAX_STEPS, false);
        std::printf("[baseline] fit=%.4f ID_SR=%.1f%% stress_SR=%.1f%% (%zu/%zu on pack-S seeds)\n\n",
                    base_fit, mid.success_rate * 100.0, ms.success_rate * 100.0,
                    static_cast<size_t>(ms.success_rate * stress_seeds.size() + 0.5),
                    stress_seeds.size());
    }

    double best_fit = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= GENS; ++gen) {
        auto& pop = engine.population();
        double gen_best = -1e9, gen_sum = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < pop.size(); ++i) {
            const double fit = fitness_of(pop[i]);
            pop[i].fitness_score = fit;
            gen_sum += fit;
            if (fit > gen_best) { gen_best = fit; best_idx = i; }
        }
        const bool improved = gen_best > best_fit;
        if (improved) { best_fit = gen_best; champion = pop[best_idx]; }

        if (gen % 5 == 0 || gen == 1 || improved) {
            std::printf("  Gen %3zu/%zu | best=%.4f mean=%.4f | %zu cells %zu syns%s\n",
                        gen, GENS, gen_best, gen_sum / static_cast<double>(POP),
                        pop[best_idx].cells.size(), pop[best_idx].synapses.size(),
                        improved ? " *" : "");
        }
        if (gen < GENS) engine.evolve_generation();
    }
    const double train_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    enforce_strict_loop_gain(champion, 0.95);

    // Final cold metrics (plasticity off) on registered slices
    auto fin_id = id_env.evaluate_organism(champion, id_seeds, MAX_STEPS, false);
    auto fin_mild = mild_env.evaluate_organism(champion, mild_seeds, MAX_STEPS, false);
    auto fin_stress = stress_env.evaluate_organism(champion, stress_seeds, MAX_STEPS, false);

    std::printf("----------------------------------------------------------\n");
    std::printf("  train_sec=%.1f | start_mode=%s | best_fit=%.4f\n",
                train_sec, start_mode.c_str(), best_fit);
    std::printf("  final ID_SR=%.1f%% mild_SR=%.1f%% stress_SR=%.1f%% (%zu/%zu)\n",
                fin_id.success_rate * 100.0, fin_mild.success_rate * 100.0,
                fin_stress.success_rate * 100.0,
                static_cast<size_t>(fin_stress.success_rate * stress_seeds.size() + 0.5),
                stress_seeds.size());
    std::printf("  champion: %zu cells %zu syns\n",
                champion.cells.size(), champion.synapses.size());

    for (auto& s : champion.synapses) s.initial_weight = s.weight;

    champion.save_checkpoint_bin(out_path);
    std::printf("  [SUCCESS] candidate saved: %s\n", out_path.c_str());
    std::printf("  (cartpole_balance_champion.bin NOT overwritten — promote post-gate only)\n");
    std::printf("==========================================================\n");
    return 0;
}
