// ============================================================================
// train_domain_zoo.cpp — 12 域批量训练器 (管线横向复刻批量实证)
//
// 每域: 独立演化引擎 (解锁骨架, FULL_28) → Train/Holdout-ID/Holdout-OOD
// 三隔离门禁 (OOD = 同任务类 ood=2.0 工厂扰动)。SR=0 强制 FAIL。
//
// Maglev: 开环不稳定 + 质量 OOD 需课程化; 注入 PID 反射弧祖先并动态 ood 采样。
// 既有 zoo_maglev.bin 仅在复测仍过 M1 时复用, 否则重训 (杜绝失效检查点短路)。
// ============================================================================

#include "tasks/control/domain_zoo.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <random>

using namespace kun;

namespace {

struct ZooResult {
    std::string id;
    size_t cells = 0, syns = 0;
    double train_sr = 0, id_sr = 0, ood_sr = 0, id_ratio = 0, fit = 0;
    bool gate = false;
    double sec = 0;
};

CellularOrganism make_maglev_pid_progenitor() {
    CellularOrganism p = CellularOrganism::create_seed_organism(1);
    p.lineage_name = "Maglev-Progenitor";
    p.cells.clear();
    p.synapses.clear();
    p.cells.push_back({0, CellType::SENSE_RAW_INPUT_0, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f, -40.0f, 0.0f});
    p.cells.push_back({1, CellType::SENSE_RAW_INPUT_1, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,   0.0f, 0.0f});
    p.cells.push_back({2, CellType::SENSE_RAW_INPUT_2, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, -100.0f,  40.0f, 0.0f});
    p.cells.push_back({3, CellType::OP_INTEGRAL, 0.05, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -60.0f, 0.0f});
    p.cells.push_back({4, CellType::OP_DIFF,     1.00, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f, -20.0f, 0.0f});
    p.cells.push_back({5, CellType::OP_EMA,      0.30, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 0.0f,  20.0f, 0.0f});
    p.cells.push_back({6, CellType::OP_SUM,      1.00, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 60.0f,  0.0f, 0.0f});
    p.cells.push_back({7, CellType::ACT_PRIMARY_POSITIVE, 1.0, 0.0, 0.0, 0.0, false, 0.0, 0, 0, 120.0f, 0.0f, 0.0f});
    p.synapses.push_back({0, 6, 0, 1.5, true, 60.0f, -1.0f});
    p.synapses.push_back({0, 3, 0, 1.0, true, 60.0f, -1.0f});
    p.synapses.push_back({3, 6, 1, 1.2, true, 60.0f, -1.0f});
    p.synapses.push_back({1, 4, 0, 1.0, true, 60.0f, -1.0f});
    p.synapses.push_back({4, 6, 2, 0.8, true, 60.0f, -1.0f});
    p.synapses.push_back({6, 7, 0, 1.0, true, 60.0f, -1.0f});
    for (auto& s : p.synapses) {
        s.initial_weight = s.weight;
        s.hebbian_rate = 0.0;
    }
    p.compile();
    return p;
}

ZooResult train_one(const std::function<std::unique_ptr<ZooTask>(double ood)>& mk,
                    const char* ckpt_id, size_t pop, size_t gens, uint32_t seed) {
    const bool is_maglev = (std::string(ckpt_id) == "zoo_maglev");
    auto train_env = mk(1.0);
    auto id_env = mk(1.0);
    auto ood_env = mk(2.0);
    int MS = train_env->max_steps();
    if (is_maglev) MS = 600;  // 磁浮需更长稳态窗口
    train_env->set_max_steps(MS);
    id_env->set_max_steps(MS);
    ood_env->set_max_steps(MS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = ckpt_id;
    split.max_steps_per_episode = MS;

    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_28;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    MorphogeneticEvolutionEngine engine(pop, seed, cfg);

    double best = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();

    const std::string existing_ckpt = std::string("checkpoints/") + ckpt_id + ".bin";
    bool use_existing = false;
    if (is_maglev && std::ifstream(existing_ckpt).good()) {
        auto loaded = CellularOrganism::load_checkpoint_bin(existing_ckpt);
        if (!loaded.cells.empty() && loaded.compile()) {
            OOSReport probe = TaskEvaluator::evaluate_task_split(
                *train_env, *id_env, *ood_env, loaded, split, 0.70);
            if (probe.train_metrics.success_rate > 0.0 && probe.passes_m1_gate) {
                champion = loaded;
                use_existing = true;
                best = 1.0;
            }
        }
    }

    if (!use_existing) {
        if (is_maglev) {
            auto progenitor = make_maglev_pid_progenitor();
            engine.population()[0] = progenitor;
            for (size_t i = 1; i < engine.population().size(); ++i) {
                auto org = progenitor;
                for (int m = 0; m < 2; ++m) engine.mutate(org);
                engine.population()[i] = org;
            }
        }

        std::mt19937 rng(seed);
        for (size_t gen = 1; gen <= gens; ++gen) {
            auto& popv = engine.population();
            double gb = -1e9;
            size_t bi = 0;

            if (is_maglev) {
                const double min_ood = 1.0 - std::min(0.3, gen * 0.003);
                const double max_ood = 1.0 + std::min(0.8, gen * 0.008);
                std::uniform_real_distribution<double> dist_ood(min_ood, max_ood);
                ZooMaglev env1(dist_ood(rng));
                ZooMaglev env2(dist_ood(rng));
                env1.set_max_steps(MS);
                env2.set_max_steps(MS);
                for (size_t i = 0; i < popv.size(); ++i) {
                    auto m1 = env1.evaluate_organism(popv[i], split.train_seeds, MS, false);
                    auto m2 = env2.evaluate_organism(popv[i], split.train_seeds, MS, false);
                    const double fit = 0.5 * (m1.mean_fitness + m2.mean_fitness);
                    popv[i].fitness_score = fit;
                    if (fit > gb) { gb = fit; bi = i; }
                }
            } else {
                for (size_t i = 0; i < popv.size(); ++i) {
                    auto m = train_env->evaluate_organism(popv[i], split.train_seeds, MS, false);
                    popv[i].fitness_score = m.mean_fitness;
                    if (m.mean_fitness > gb) { gb = m.mean_fitness; bi = i; }
                }
            }
            if (gb > best) { best = gb; champion = popv[bi]; }
            if (gen < gens) engine.evolve_generation();
        }
    }
    const double sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    OOSReport rep = TaskEvaluator::evaluate_task_split(*train_env, *id_env, *ood_env,
                                                       champion, split, 0.70);
    if (rep.train_metrics.success_rate <= 0.0) {
        rep.passes_m1_gate = false;
        rep.verdict = "FAIL: train SR=0";
    }

    ZooResult r;
    r.id = ckpt_id;
    r.cells = champion.cells.size();
    r.syns = champion.synapses.size();
    r.train_sr = rep.train_metrics.success_rate;
    r.id_sr = rep.holdout_id_metrics.success_rate;
    r.ood_sr = rep.holdout_ood_metrics.success_rate;
    r.id_ratio = rep.id_generalization_ratio;
    r.fit = best;
    r.gate = rep.passes_m1_gate;
    r.sec = sec;

    champion.save_checkpoint_bin(std::string("checkpoints/") + ckpt_id + ".bin");
    return r;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=====================================================================\n");
    std::printf("  Domain Zoo — 12 域批量训练 (解锁骨架 / 三隔离门禁 / 每域独立引擎)\n");
    std::printf("=====================================================================\n");

    const size_t POP = 32, GENS = 120;
    const uint32_t SEED = 20260903;

    using Maker = std::function<std::unique_ptr<ZooTask>(double ood)>;
    const std::vector<std::pair<const char*, Maker>> zoo = {
        {"zoo_cartpole",     [](double o) { return std::make_unique<ZooCartPole>(o); }},
        {"zoo_ballbeam",     [](double o) { return std::make_unique<ZooBallBeam>(o); }},
        {"zoo_maglev",       [](double o) { return std::make_unique<ZooMaglev>(o); }},
        {"zoo_rocket_hover", [](double o) { return std::make_unique<ZooRocketHover>(o); }},
        {"zoo_cruise",       [](double o) { return std::make_unique<ZooCruise>(o); }},
        {"zoo_thermal",      [](double o) { return std::make_unique<ZooThermal>(o); }},
        {"zoo_water_tank",   [](double o) { return std::make_unique<ZooWaterTank>(o); }},
        {"zoo_dc_motor",     [](double o) { return std::make_unique<ZooDCMotor>(o); }},
        {"zoo_vibration",    [](double o) { return std::make_unique<ZooVibration>(o); }},
        {"zoo_servo",        [](double o) { return std::make_unique<ZooServo>(o); }},
        {"zoo_boiler",       [](double o) { return std::make_unique<ZooBoiler>(o); }},
        {"zoo_bicycle",      [](double o) { return std::make_unique<ZooBicycle>(o); }},
    };

    std::vector<ZooResult> results;
    uint32_t task_seed = SEED;
    for (const auto& [id, mk] : zoo) {
        auto r = train_one(mk, id, POP, GENS, task_seed++);
        results.push_back(r);
        std::printf("  %-16s %6.1fs | 训练SR %5.1f%% | ID %5.1f%% (x%.2f) | OOD %5.1f%% | "
                    "%zu 细胞 %zu 突触 | 门禁 %s\n",
                    r.id.c_str(), r.sec, r.train_sr * 100.0, r.id_sr * 100.0,
                    r.id_ratio, r.ood_sr * 100.0, r.cells, r.syns,
                    r.gate ? "PASS" : "FAIL");
    }

    size_t passed = 0;
    double total_sec = 0;
    for (auto& r : results) { passed += r.gate ? 1 : 0; total_sec += r.sec; }
    std::printf("---------------------------------------------------------------------\n");
    std::printf("  总计: %zu/%zu 域过 M1 门禁 | 总训练耗时 %.1fs\n", passed, results.size(), total_sec);

    std::ofstream rf("checkpoints/domain_zoo_report.json");
    rf << "{\n  \"gate\": " << (passed == results.size() ? "true" : "false") << ",\n";
    rf << "  \"passed\": " << passed << ", \"total\": " << results.size() << ",\n";
    rf << "  \"domains\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        rf << "    {\"id\": \"" << r.id << "\", \"cells\": " << r.cells
           << ", \"synapses\": " << r.syns
           << ", \"train_sr\": " << r.train_sr << ", \"id_sr\": " << r.id_sr
           << ", \"id_ratio\": " << r.id_ratio << ", \"ood_sr\": " << r.ood_sr
           << ", \"gate\": " << (r.gate ? "true" : "false")
           << ", \"train_sec\": " << r.sec << "}"
           << (i + 1 < results.size() ? "," : "") << "\n";
    }
    rf << "  ]\n}\n";
    rf.close();
    std::printf("  [SUCCESS] 批量冠军 12 份 + domain_zoo_report.json 已存盘\n");
    std::printf("=====================================================================\n");
    return (passed == results.size()) ? 0 : 1;
}
