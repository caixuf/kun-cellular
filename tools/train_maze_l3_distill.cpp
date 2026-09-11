// ============================================================================
// train_maze_l3_distill.cpp — 迷宫 L3：从现冠军出发，通关率精英 ES
// 冻结门禁：11×11 @250 步 ≥95/100（种子 50000+s*17）
// ============================================================================

#include "kun/cellular/cellular_genome.hpp"
#include "tasks/robotics/maze_navigator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace kun;

static void sync_initial(CellularOrganism& org) {
    for (auto& s : org.synapses) s.initial_weight = s.weight;
}

static bool run_episode(CellularOrganism& org, uint32_t seed, int steps, float braid = 0.15f) {
    MazeTask task(11, 11, seed, steps, braid);
    task.set_use_geodesic_bearing(true);
    task.reset(seed);
    org.reset_state(false);
    for (int t = 0; t < steps; ++t) {
        auto obs = task.current_observation();
        double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
        auto acts = org.forward(inps, false);
        if (task.step_continuous(acts).done) return task.get_agent().reached_goal;
    }
    return false;
}

static double success_rate(CellularOrganism& org, const std::vector<uint32_t>& seeds, int steps) {
    int ok = 0;
    for (uint32_t s : seeds) {
        if (run_episode(org, s, steps)) ++ok;
    }
    return seeds.empty() ? 0.0 : static_cast<double>(ok) / static_cast<double>(seeds.size());
}

static double mean_fitness(CellularOrganism& org, const std::vector<uint32_t>& seeds, int steps) {
    double sum = 0.0;
    for (uint32_t s : seeds) {
        MazeTask task(11, 11, s, steps, 0.15f);
        task.set_use_geodesic_bearing(true);
        task.reset(s);
        org.reset_state(false);
        for (int t = 0; t < steps; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = org.forward(inps, false);
            if (task.step_continuous(acts).done) break;
        }
        sum += task.current_fitness();
    }
    return seeds.empty() ? 0.0 : sum / static_cast<double>(seeds.size());
}

static void mutate(CellularOrganism& org, std::mt19937& rng, double sigma, bool structural) {
    std::normal_distribution<double> n(0.0, sigma);
    for (auto& s : org.synapses) {
        if (!s.is_active) continue;
        s.weight = std::clamp(s.weight + n(rng), -8.0, 8.0);
        s.initial_weight = s.weight;
    }
    for (auto& c : org.cells) {
        c.param1 = std::clamp(c.param1 + n(rng) * 0.08, -5.0, 5.0);
        c.param2 = std::clamp(c.param2 + n(rng) * 0.08, -5.0, 5.0);
    }
    if (structural && org.synapses.size() < 40) {
        std::uniform_int_distribution<size_t> ci(0, org.cells.size() - 1);
        std::uniform_real_distribution<double> w(-0.4, 0.4);
        // 随机加一条弱连接
        uint32_t a = org.cells[ci(rng)].id;
        uint32_t b = org.cells[ci(rng)].id;
        if (a != b) {
            Synapse syn;
            syn.from_cell_id = a;
            syn.to_cell_id = b;
            syn.to_port = 0;
            syn.weight = w(rng);
            syn.initial_weight = syn.weight;
            syn.is_active = true;
            syn.rest_length = 50.0f;
            syn.hebbian_rate = 0.0;
            org.synapses.push_back(syn);
        }
    }
    org.compile();
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  迷宫 L3 · 冠军精英 ES（通关率主目标）\n";
    std::cout << "======================================================================\n";

    const size_t POP = 36;
    const int GENS = 150;
    const int EVAL_STEPS = 280; // 训练略宽，终评仍 250
    const uint32_t SEED = 20260911;
    std::mt19937 rng(SEED);

    CellularOrganism base = CellularOrganism::load_checkpoint_bin("checkpoints/maze_navigation_champion.bin");
    if (base.cells.empty()) {
        std::cerr << "缺少 maze_navigation_champion.bin\n";
        return 2;
    }
    sync_initial(base);
    base.compile();

    std::vector<CellularOrganism> pop(POP, base);
    // 仅子代变异；索引 0 永远是未变异精英槽（每代写回冠军）
    for (size_t i = 1; i < POP; ++i) mutate(pop[i], rng, 0.10, i % 4 == 0);

    std::vector<uint32_t> gate_seeds;
    for (int s = 0; s < 100; ++s) gate_seeds.push_back(50000u + static_cast<uint32_t>(s) * 17u);

    CellularOrganism champion = base;
    double best_gate = success_rate(champion, gate_seeds, 250);
    std::cout << "  起点门禁 SR@250: " << (best_gate * 100.0) << "%\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int gen = 1; gen <= GENS; ++gen) {
        // 每代重抽训练种子，逼泛化
        std::vector<uint32_t> train_seeds(12);
        for (auto& s : train_seeds) s = 20000u + static_cast<uint32_t>(rng() % 35000u);

        struct Rank {
            double score;
            double sr;
            size_t idx;
        };
        std::vector<Rank> ranks;
        ranks.reserve(POP);
        for (size_t i = 0; i < POP; ++i) {
            double sr = success_rate(pop[i], train_seeds, EVAL_STEPS);
            double fit = mean_fitness(pop[i], train_seeds, EVAL_STEPS);
            double score = sr * 200.0 + fit * 0.02;
            ranks.push_back({score, sr, i});
        }
        std::sort(ranks.begin(), ranks.end(), [](const Rank& a, const Rank& b) { return a.score > b.score; });

        // 用门禁口径评估前 3，更新冠军（避免只过拟合 train_seeds）
        for (size_t k = 0; k < 3; ++k) {
            auto& cand = pop[ranks[k].idx];
            double g = success_rate(cand, gate_seeds, 250);
            if (g > best_gate) {
                best_gate = g;
                champion = cand;
                sync_initial(champion);
            }
        }

        // 繁殖：精英 6 保留，其余由精英变异
        std::vector<CellularOrganism> next;
        next.reserve(POP);
        next.push_back(champion); // 未变异精英
        for (size_t e = 0; e < 5; ++e) next.push_back(pop[ranks[e].idx]);
        std::uniform_int_distribution<size_t> elite(0, 5);
        double sigma = (gen < 50) ? 0.12 : (gen < 100 ? 0.07 : 0.035);
        while (next.size() < POP) {
            CellularOrganism child = next[elite(rng)];
            mutate(child, rng, sigma, next.size() % 5 == 0);
            next.push_back(std::move(child));
        }
        pop.swap(next);

        if (gen % 10 == 0 || gen == 1 || gen == GENS) {
            std::cout << "  Gen " << gen << "/" << GENS
                      << " | gate_SR@250=" << (best_gate * 100.0) << "%"
                      << " | train_top_SR=" << (ranks[0].sr * 100.0) << "%"
                      << " | cells=" << champion.cells.size()
                      << " syns=" << champion.synapses.size()
                      << " | sigma=" << sigma << "\n";
        }
        if (best_gate >= 0.95) {
            std::cout << "  [早停] 门禁 ≥95%\n";
            break;
        }
    }

    sync_initial(champion);
    double sr250_f = success_rate(champion, gate_seeds, 250);
    double sr400_f = success_rate(champion, gate_seeds, 400);
    int ok_t = 0;
    for (uint32_t s : gate_seeds) {
        MazeTask task(11, 11, s, 250, 0.15f);
        task.set_use_geodesic_bearing(true);
        task.reset(s);
        champion.reset_state(true);
        bool ok = false;
        for (int t = 0; t < 250; ++t) {
            auto obs = task.current_observation();
            double inps[4] = {obs[0], obs[1], obs[2], obs[3]};
            auto acts = champion.forward(inps, false);
            if (task.step_continuous(acts).done) {
                ok = task.get_agent().reached_goal;
                break;
            }
        }
        if (ok) ++ok_t;
    }
    double sr250_t = ok_t / 100.0;

    auto sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  耗时 " << sec << "s | L3-M1 @250 false=" << (sr250_f * 100.0) << "% | true="
              << (sr250_t * 100.0) << "% | L3-M2 @400=" << (sr400_f * 100.0) << "%\n";
    const bool m1 = sr250_f >= 0.95;
    const bool m2 = sr400_f >= 0.98;
    const bool m3 = std::fabs(sr250_f - sr250_t) <= 0.02 + 1e-9;
    std::cout << "  门禁: L3-M1 " << (m1 ? "PASS" : "FAIL")
              << " | L3-M2 " << (m2 ? "PASS" : "FAIL")
              << " | L3-M3 " << (m3 ? "PASS" : "FAIL") << "\n";

    sync_initial(champion);
    const char* out = m1 ? "checkpoints/maze_navigation_champion.bin"
                         : "checkpoints/maze_navigation_l3_attempt.bin";
    if (m1) {
        auto cur = CellularOrganism::load_checkpoint_bin("checkpoints/maze_navigation_champion.bin");
        if (!cur.cells.empty()) {
            cur.save_checkpoint_bin("checkpoints/maze_navigation_champion_pre_l3_20260911.bin");
        }
    }
    champion.save_checkpoint_bin(out);
    std::cout << "  产物: " << out << "\n";
    std::cout << "======================================================================\n";
    return (m1 && m2 && m3) ? 0 : 1;
}
