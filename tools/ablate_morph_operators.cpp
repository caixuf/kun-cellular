// 任务层形态发生旋钮消融：ZooCartPole，不改 include/kun/cellular/，不覆盖 zoo_cartpole.bin。
// 可关：slow/medium 变异率、力敏、鲍德温。凋亡是 mutate() 内硬编码 5%，本工具关不掉。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include "tasks/control/domain_zoo.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace kun;

struct Arm {
    const char* id;
    double slow;
    double medium;
    bool mechano;
    bool baldwin;
};

static EvolutionConstraintConfig make_cfg(const Arm& a) {
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_28;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    cfg.slow_mutation_rate = a.slow;
    cfg.medium_mutation_rate = a.medium;
    cfg.enable_mechanotransduction = a.mechano;
    cfg.enable_baldwin_crystallization = a.baldwin;
    return cfg;
}

struct RunRow {
    std::string arm;
    uint32_t seed{0};
    double train_sr{0}, id_sr{0}, ood_sr{0};
    bool gate{false};
    size_t cells{0}, syns{0};
    double sec{0};
};

static RunRow run_arm(const Arm& arm, uint32_t seed, size_t pop, size_t gens) {
    ZooCartPole train_env(1.0), id_env(1.0), ood_env(2.0);
    const int MS = train_env.max_steps();
    train_env.set_max_steps(MS);
    id_env.set_max_steps(MS);
    ood_env.set_max_steps(MS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = "zoo_cartpole_ablate";
    split.max_steps_per_episode = MS;

    MorphogeneticEvolutionEngine engine(pop, seed, make_cfg(arm));
    double best = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= gens; ++gen) {
        auto& popv = engine.population();
        double gb = -1e9;
        size_t bi = 0;
        for (size_t i = 0; i < popv.size(); ++i) {
            auto m = train_env.evaluate_organism(popv[i], split.train_seeds, MS, false);
            popv[i].fitness_score = m.mean_fitness;
            if (m.mean_fitness > gb) {
                gb = m.mean_fitness;
                bi = i;
            }
        }
        if (gb > best) {
            best = gb;
            champion = popv[bi];
        }
        if (gen < gens) engine.evolve_generation();
    }
    RunRow row;
    row.arm = arm.id;
    row.seed = seed;
    row.sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    OOSReport rep = TaskEvaluator::evaluate_task_split(train_env, id_env, ood_env, champion, split, 0.70);
    if (rep.train_metrics.success_rate <= 0.0) {
        rep.passes_m1_gate = false;
    }
    row.train_sr = rep.train_metrics.success_rate;
    row.id_sr = rep.holdout_id_metrics.success_rate;
    row.ood_sr = rep.holdout_ood_metrics.success_rate;
    row.gate = rep.passes_m1_gate;
    row.cells = champion.cells.size();
    row.syns = champion.synapses.size();
    return row;
}

int main(int argc, char** argv) {
    size_t POP = 32;
    size_t GENS = 120;
    std::string out_path = "runs/morph_operator_ablation_cartpole_20260915.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--pop") POP = static_cast<size_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--gens") GENS = static_cast<size_t>(std::strtoul(next(), nullptr, 10));
        else if (a == "--out") out_path = next();
    }

    const Arm arms[] = {
        {"full", 0.35, 0.45, true, true},
        {"no_mitosis", 0.0, 0.45, true, true},
        {"no_rewire", 0.35, 0.0, true, true},
        {"param_only", 0.0, 0.0, true, true},
        {"no_baldwin", 0.35, 0.45, true, false},
        {"no_mechano", 0.35, 0.45, false, true},
    };
    const uint32_t seeds[] = {1, 2, 3};

    std::printf("morph-operator-ablation ZooCartPole pop=%zu gens=%zu\n", POP, GENS);
    std::printf("apoptosis=HARDCODED_5pct (cannot disable without touching base)\n");

    std::vector<RunRow> rows;
    for (const auto& arm : arms) {
        for (uint32_t seed : seeds) {
            auto r = run_arm(arm, seed, POP, GENS);
            std::printf("  %s seed=%u train=%.2f id=%.2f ood=%.2f gate=%d cells=%zu syns=%zu t=%.1fs\n",
                        r.arm.c_str(), r.seed, r.train_sr, r.id_sr, r.ood_sr,
                        r.gate ? 1 : 0, r.cells, r.syns, r.sec);
            rows.push_back(std::move(r));
        }
    }

    auto mean_id = [&](const char* id) {
        double s = 0;
        int n = 0;
        for (const auto& r : rows) {
            if (r.arm == id) {
                s += r.id_sr;
                n++;
            }
        }
        return n ? s / n : 0.0;
    };
    const double full_id = mean_id("full");
    std::printf("full mean ID SR=%.3f  (load-bearing if full-arm >= 0.10)\n", full_id);
    for (const auto& arm : arms) {
        if (std::string(arm.id) == "full") continue;
        const double d = full_id - mean_id(arm.id);
        std::printf("  vs %s  delta_id=%.3f  %s\n", arm.id, d, d >= 0.10 ? "LOAD_BEARING" : "NEGATIVE");
    }

    std::ofstream rf(out_path);
    rf << std::fixed;
    rf << "{\n  \"tool\": \"ablate_morph_operators\",\n  \"task\": \"zoo_cartpole\",\n";
    rf << "  \"pop\": " << POP << ", \"gens\": " << GENS << ",\n";
    rf << "  \"apoptosis\": \"hardcoded_0.05_cannot_disable\",\n";
    rf << "  \"overwrite_zoo_bin\": false,\n  \"runs\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        rf << "    {\"arm\":\"" << r.arm << "\",\"seed\":" << r.seed
           << ",\"train_sr\":" << r.train_sr << ",\"id_sr\":" << r.id_sr
           << ",\"ood_sr\":" << r.ood_sr << ",\"gate\":" << (r.gate ? "true" : "false")
           << ",\"cells\":" << r.cells << ",\"synapses\":" << r.syns
           << ",\"sec\":" << r.sec << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    rf << "  ]\n}\n";
    std::printf("[REPORT] %s\n", out_path.c_str());
    return 0;
}
