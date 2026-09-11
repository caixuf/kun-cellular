// ============================================================================
// train_cart_pole.cpp — 倒立摆平衡生命体训练器 (管线横向复刻第一证)
//
// 流程: 随机基线 → 形态发生演化 → Train/Holdout-ID/Holdout-OOD 三隔离门禁
// OOD = 更重摆锤(0.2kg) + 更长摆杆(0.7m) + 推力噪声(2N) —— 跨物理参数泛化
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_cart_pole.cpp -o bin/train_cart_pole
// ============================================================================

#include "tasks/control/cart_pole_task.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace kun;

// ============================================================================
// 严格李雅普诺夫环增益修复 (任务层，不动底座)
//
// 底座 CellularOrganism::check_lyapunov_stability() 对"环内含耗散门 (EMA/迟滞/死区)"
// 的反馈环宽容 (is_stable=true)，但形式化认证管线 kun_certify 的契约要求
// max_loop_gain < 1.0 才颁发 BIBO 证书。为保证演化产物可通过认证，这里对高增益
// 环做任务层权重缩放，把 max_loop_gain 压到 target 以下（幂律按环长分配，尽量小扰动）。
// ============================================================================
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
    std::printf("  [Lyapunov] 严格环增益修复后 rho=%.5f (target<%.2f)\n", rep.max_loop_gain, target);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("==========================================================\n");
    std::printf("  倒立摆平衡生命体 · 管线横向复刻验证 (C++20 Native)\n");
    std::printf("==========================================================\n");

    const size_t POP = 48;
    const size_t GENS = 150;
    const int MAX_STEPS = 300;
    const uint32_t SEED = 20260903;

    // 骨架解锁: 演化必须能长出新的感受器/效应器 (LOCKED 时摆杆信号进不来)
    EvolutionConstraintConfig cfg;
    cfg.skeleton_lock = SkeletonLockMode::UNLOCKED;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.seed_mode = SeedInitMode::HANDCRAFTED_PROGENITOR;

    // 三隔离环境: 训练 / 同分布留出 / 跨物理参数 OOD
    CartPoleBalanceTask train_env, id_env;
    CartPoleBalanceTask::Params ood_p;
    ood_p.masspole = 0.2; ood_p.length = 0.7; ood_p.force_noise = 2.0;
    CartPoleBalanceTask ood_env(ood_p);
    train_env.set_max_steps(MAX_STEPS); id_env.set_max_steps(MAX_STEPS);
    ood_env.set_max_steps(MAX_STEPS);

    TaskDatasetSplit split = TaskDatasetSplit::create_default_maze_split();
    split.task_name = "CartPoleBalance";
    split.max_steps_per_episode = MAX_STEPS;

    MorphogeneticEvolutionEngine engine(POP, SEED, cfg);

    // ---- 随机基线 (未演化祖细胞) ----
    double base_sr = 0.0;
    {
        auto& pop = engine.population();
        auto m = train_env.evaluate_organism(pop[0], split.train_seeds, MAX_STEPS, false);
        base_sr = m.success_rate;
        std::printf("[基线] 未演化祖细胞: 生存率 %.1f%% (适应度 %.3f)\n\n",
                    m.success_rate * 100.0, m.mean_fitness);
    }

    // ---- 代际演化 ----
    double best_fit = -1e9;
    CellularOrganism champion;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t gen = 1; gen <= GENS; ++gen) {
        auto& pop = engine.population();
        double gen_best = -1e9, gen_sum = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < pop.size(); ++i) {
            auto m = train_env.evaluate_organism(pop[i], split.train_seeds, MAX_STEPS, true);
            pop[i].fitness_score = m.mean_fitness;
            gen_sum += m.mean_fitness;
            if (m.mean_fitness > gen_best) { gen_best = m.mean_fitness; best_idx = i; }
        }
        const bool improved = gen_best > best_fit;
        if (improved) { best_fit = gen_best; champion = pop[best_idx]; }

        if (gen % 5 == 0 || gen == 1 || improved) {
            std::printf("  Gen %2zu/%zu | best=%.3f mean=%.3f | %zu 细胞 %zu 突触\n",
                        gen, GENS, gen_best, gen_sum / static_cast<double>(POP),
                        pop[best_idx].cells.size(), pop[best_idx].synapses.size());
        }
        if (gen < GENS) engine.evolve_generation();
    }
    const double train_sec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ---- 严格环增益修复 (保证可过 kun_certify 的 rho<1.0 契约) ----
    enforce_strict_loop_gain(champion, 0.95);

    // ---- 三隔离门禁终审 (TaskEvaluator 规范路径) ----
    OOSReport report = TaskEvaluator::evaluate_task_split(
        train_env, id_env, ood_env, champion, split, 0.70);

    // 诚实门禁: 生存型任务训练 SR=0 时, 距离回退分支会虚报 PASS —— 强制改判
    if (report.train_metrics.success_rate <= 0.0) {
        report.passes_m1_gate = false;
        report.verdict = "FAIL: 训练生存率为 0 (距离回退分支对生存型任务无效, 不得虚报通过)";
    }

    std::printf("----------------------------------------------------------\n");
    std::printf("  训练耗时 %.1fs | 随机基线生存率 %.1f%% → 冠军 %.1f%%\n",
                train_sec, base_sr * 100.0, report.train_metrics.success_rate * 100.0);
    std::printf("  %s\n", report.verdict.c_str());
    std::printf("  冠军: %zu 细胞 %zu 突触 | WL=%s\n",
                champion.cells.size(), champion.synapses.size(),
                TaskEvaluator::compute_topology_hash(champion).c_str());

    // 落盘前同步基因初始权重，避免 reset_state(true) 抹掉演化表型
    for (auto& s : champion.synapses) s.initial_weight = s.weight;

    champion.save_checkpoint_bin("checkpoints/cartpole_balance_champion.bin");
    std::ofstream rf("checkpoints/cartpole_balance_report.json");
    rf << report.to_json();
    rf.close();
    std::printf("  [SUCCESS] 冠军与门禁报告已存盘\n");
    std::printf("==========================================================\n");
    return report.passes_m1_gate ? 0 : 1;
}
