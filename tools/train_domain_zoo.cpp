// ============================================================================
// train_domain_zoo.cpp — 12 域批量训练器 (管线横向复刻批量实证)
//
// 每域: 独立演化引擎 (解锁骨架, FULL_24) → Train/Holdout-ID/Holdout-OOD
// 三隔离门禁 (OOD = 同任务类 ood=2.0 工厂扰动)。SR=0 强制 FAIL。
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_domain_zoo.cpp -o bin/train_domain_zoo
// ============================================================================

#include "tasks/control/domain_zoo.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>

using namespace kun;

namespace {

struct ZooResult {
    std::string id;
    size_t cells = 0, syns = 0;
    double train_sr = 0, id_sr = 0, ood_sr = 0, id_ratio = 0, fit = 0;
    bool gate = false;
    double sec = 0;
};

ZooResult train_one(const std::function<std::unique_ptr<ZooTask>(double ood)>& mk,
                    const char* ckpt_id, size_t pop, size_t gens, uint32_t seed) {
    auto train_env = mk(1.0);
    auto id_env = mk(1.0);
    auto ood_env = mk(2.0);   // 跨物理参数扰动 (每类内部自定义语义)
    const int MS = train_env->max_steps();
    id_env->set_max_steps(MS);
    ood_env->set_max_steps(MS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = ckpt_id;
    split.max_steps_per_episode = MS;

    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;
    MorphogeneticEvolutionEngine engine(pop, seed, cfg);

    double best = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= gens; ++gen) {
        auto& popv = engine.population();
        double gb = -1e9;
        size_t bi = 0;
        for (size_t i = 0; i < popv.size(); ++i) {
            auto m = train_env->evaluate_organism(popv[i], split.train_seeds, MS, true);
            popv[i].fitness_score = m.mean_fitness;
            if (m.mean_fitness > gb) { gb = m.mean_fitness; bi = i; }
        }
        if (gb > best) { best = gb; champion = popv[bi]; }
        if (gen < gens) engine.evolve_generation();
    }
    const double sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    OOSReport rep = TaskEvaluator::evaluate_task_split(*train_env, *id_env, *ood_env,
                                                       champion, split, 0.70);
    if (rep.train_metrics.success_rate <= 0.0) {  // 生存型任务: SR=0 不得走距离回退虚报
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

    const std::string path = std::string("checkpoints/") + ckpt_id + ".json";
    champion.save_checkpoint_json(path);
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

    // 汇总
    size_t passed = 0;
    double total_sec = 0;
    for (auto& r : results) { passed += r.gate ? 1 : 0; total_sec += r.sec; }
    std::printf("---------------------------------------------------------------------\n");
    std::printf("  总计: %zu/%zu 域过 M1 门禁 | 总训练耗时 %.1fs\n", passed, results.size(), total_sec);

    // 汇总报告 JSON (诚实记录, 失败也留档)
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
    return 0;
}
