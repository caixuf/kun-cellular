// DomainZoo 评测隔离审计。任务层 only；不改底座热路径、不覆盖 zoo_*.bin、不改 domain_zoo_report.json。
// 2026-09-15 已在 ZooTask::reset 清 o_[]：cartpole ID 的 leaky=clone=fresh=0.1。
// 同日 reset_state 清 membrane_pores：堵住 evaluate_organism 复用有机体的代谢门控泄漏。
// JSON 12/12 与 id_sr=0.9 仍不可复现。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/evolvable_task.hpp"
#include "tasks/control/domain_zoo.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace kun;

static const char* kIds[] = {
    "zoo_cartpole", "zoo_ballbeam", "zoo_maglev", "zoo_rocket_hover",
    "zoo_cruise", "zoo_thermal", "zoo_water_tank", "zoo_dc_motor",
    "zoo_vibration", "zoo_servo", "zoo_boiler", "zoo_bicycle",
};

static std::unique_ptr<ZooTask> make_zoo(const char* id, double ood) {
    if (std::strcmp(id, "zoo_cartpole") == 0) return std::make_unique<ZooCartPole>(ood);
    if (std::strcmp(id, "zoo_ballbeam") == 0) return std::make_unique<ZooBallBeam>(ood);
    if (std::strcmp(id, "zoo_maglev") == 0) return std::make_unique<ZooMaglev>(ood);
    if (std::strcmp(id, "zoo_rocket_hover") == 0) return std::make_unique<ZooRocketHover>(ood);
    if (std::strcmp(id, "zoo_cruise") == 0) return std::make_unique<ZooCruise>(ood);
    if (std::strcmp(id, "zoo_thermal") == 0) return std::make_unique<ZooThermal>(ood);
    if (std::strcmp(id, "zoo_water_tank") == 0) return std::make_unique<ZooWaterTank>(ood);
    if (std::strcmp(id, "zoo_dc_motor") == 0) return std::make_unique<ZooDCMotor>(ood);
    if (std::strcmp(id, "zoo_vibration") == 0) return std::make_unique<ZooVibration>(ood);
    if (std::strcmp(id, "zoo_servo") == 0) return std::make_unique<ZooServo>(ood);
    if (std::strcmp(id, "zoo_boiler") == 0) return std::make_unique<ZooBoiler>(ood);
    if (std::strcmp(id, "zoo_bicycle") == 0) return std::make_unique<ZooBicycle>(ood);
    return nullptr;
}

static std::string find_bin(const char* id) {
    const std::string a = std::string("checkpoints/") + id + ".bin";
    const std::string b = std::string("../checkpoints/") + id + ".bin";
    for (const auto& p : {a, b}) {
        CellularOrganism probe = CellularOrganism::load_checkpoint_bin(p);
        if (!probe.cells.empty()) return p;
    }
    return {};
}

static bool m1_pass(double train_sr, double id_sr, double ood_sr) {
    const double base = std::max(0.001, train_sr);
    const bool train_ok = train_sr >= 0.70;
    const bool id_ok = (id_sr / base >= 0.70) && (id_sr >= 0.60);
    const bool ood_ok = (ood_sr / base >= 0.50) && (ood_sr >= 0.50);
    return train_ok && id_ok && ood_ok;
}

static TaskEvalMetrics eval_clone(const char* id, double ood, const CellularOrganism& tmpl,
                                  const std::vector<uint32_t>& seeds, int max_steps) {
    CellularOrganism org = tmpl;
    auto env = make_zoo(id, ood);
    env->set_max_steps(max_steps);
    return env->evaluate_organism(org, seeds, max_steps, false);
}

static TaskEvalMetrics eval_fresh(const std::string& path, const char* id, double ood,
                                  const std::vector<uint32_t>& seeds, int max_steps) {
    TaskEvalMetrics metrics;
    metrics.num_episodes = seeds.size();
    if (seeds.empty()) return metrics;
    size_t successes = 0;
    for (uint32_t seed : seeds) {
        CellularOrganism org = CellularOrganism::load_checkpoint_bin(path);
        org.compile();
        auto env = make_zoo(id, ood);
        env->set_max_steps(max_steps);
        org.reset_state(true);
        env->reset(seed);
        auto obs = env->current_observation();
        bool reached = false;
        for (int t = 0; t < max_steps; ++t) {
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            auto res = env->step_continuous(acts);
            obs = res.obs;
            if (res.success) reached = true;
            if (res.done) break;
        }
        if (reached) successes++;
    }
    metrics.success_episodes = successes;
    metrics.success_rate = static_cast<double>(successes) / static_cast<double>(seeds.size());
    return metrics;
}

struct SplitSR {
    double train{0}, id{0}, ood{0};
    bool m1{false};
};

int main() {
    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    int leaky_m1 = 0, clone_m1 = 0;
    double cartpole_leaky_id = -1, cartpole_clone_id = -1, cartpole_fresh_id = -1;
    double cartpole_leaky_ood = -1, cartpole_clone_ood = -1;
    double ballbeam_clone_ood = -1;

    std::cout << "id                 leaky train/id/ood/M1     clone train/id/ood/M1\n";
    for (const char* id : kIds) {
        const std::string path = find_bin(id);
        assert(!path.empty());
        const int ms = (std::strcmp(id, "zoo_maglev") == 0) ? 600 : 300;
        split.max_steps_per_episode = ms;

        CellularOrganism leaky = CellularOrganism::load_checkpoint_bin(path);
        assert(!leaky.cells.empty());
        leaky.compile();
        auto train_env = make_zoo(id, 1.0);
        auto id_env = make_zoo(id, 1.0);
        auto ood_env = make_zoo(id, 2.0);
        train_env->set_max_steps(ms);
        id_env->set_max_steps(ms);
        ood_env->set_max_steps(ms * 2);  // 与 evaluate 循环及 clone 对齐；success 判定是 steps_ >= max_steps_
        SplitSR L;
        L.train = train_env->evaluate_organism(leaky, split.train_seeds, ms, false).success_rate;
        L.id = id_env->evaluate_organism(leaky, split.holdout_id_seeds, ms, false).success_rate;
        L.ood = ood_env->evaluate_organism(leaky, split.holdout_ood_seeds, ms * 2, false).success_rate;
        L.m1 = m1_pass(L.train, L.id, L.ood);

        CellularOrganism tmpl = CellularOrganism::load_checkpoint_bin(path);
        tmpl.compile();
        SplitSR C;
        C.train = eval_clone(id, 1.0, tmpl, split.train_seeds, ms).success_rate;
        C.id = eval_clone(id, 1.0, tmpl, split.holdout_id_seeds, ms).success_rate;
        C.ood = eval_clone(id, 2.0, tmpl, split.holdout_ood_seeds, ms * 2).success_rate;
        C.m1 = m1_pass(C.train, C.id, C.ood);

        if (L.m1) leaky_m1++;
        if (C.m1) clone_m1++;
        if (std::strcmp(id, "zoo_cartpole") == 0) {
            cartpole_leaky_id = L.id;
            cartpole_clone_id = C.id;
            cartpole_leaky_ood = L.ood;
            cartpole_clone_ood = C.ood;
            cartpole_fresh_id = eval_fresh(path, id, 1.0, split.holdout_id_seeds, ms).success_rate;
        }
        if (std::strcmp(id, "zoo_ballbeam") == 0) {
            ballbeam_clone_ood = C.ood;
        }

        std::cout << "  " << id
                  << "  L " << L.train << "/" << L.id << "/" << L.ood << "/" << (L.m1 ? "P" : "F")
                  << "  C " << C.train << "/" << C.id << "/" << C.ood << "/" << (C.m1 ? "P" : "F")
                  << std::endl;
    }

    std::cout << "GATE_ISO leaky_m1=" << leaky_m1 << "/12 clone_m1=" << clone_m1 << "/12"
              << " cartpole_id L/C/F=" << cartpole_leaky_id << "/"
              << cartpole_clone_id << "/" << cartpole_fresh_id
              << " cartpole_ood L/C=" << cartpole_leaky_ood << "/" << cartpole_clone_ood
              << std::endl;

    // 数字先跑再锁。评测路径对齐后 leaky=clone；JSON 0.9 / JSON 12/12 仍不可复现。
    std::cout << "AUDIT domain zoo eval isolation (post o_[] + membrane_pores + eval-path hygiene)\n";
    assert(std::fabs(cartpole_leaky_id - 0.1) < 1e-9);
    assert(std::fabs(cartpole_clone_id - 0.1) < 1e-9);
    assert(std::fabs(cartpole_fresh_id - cartpole_clone_id) < 1e-9);
    assert(std::fabs(cartpole_leaky_ood - 0.1) < 1e-9);
    assert(std::fabs(cartpole_clone_ood - 0.1) < 1e-9);
    assert(std::fabs(ballbeam_clone_ood - 0.3) < 1e-9);
    assert(leaky_m1 == 10);
    assert(clone_m1 == 10);
    assert(clone_m1 < 12 && "isolated M1 must not be claimed 12/12");
    return 0;
}
