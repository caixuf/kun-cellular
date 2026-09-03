// ============================================================================
// train_primordial_life.cpp — 无目标原始进化生命体训练器 (C++20 Native)
//
// 两个使命:
//  [Phase A] 规模真理实测: 在本机 (核数/内存/时钟) 上逐级发育生命体
//            (9 → 1,048,576 细胞)，测量发育耗时、内存足迹、forward 吞吐
//            与 Lyapunov 稳定性，实证"本机到底能训多大"。
//  [Phase B] 无目标演化: 在 Phase A 实证可行的最大规模上，以"生存"为唯一
//            选择压力运行代际演化 (红皇后熵压逐代抬升)，检验种群是否
//            自发收敛到内稳态 (Homeostasis) —— 门禁 2 的无目标版本。
//
// 编译: g++ -O3 -march=native -std=c++20 -I include \
//       tools/train_primordial_life.cpp -o bin/train_primordial_life
// ============================================================================

#include "kun/cellular/primordial_life.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

using namespace kun;

namespace {

double seconds_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

struct RampRecord {
    size_t target_cells;
    size_t actual_cells;
    size_t actual_synapses;
    double develop_sec;
    double mem_mb;
    double step_ms;          // 单步 forward 耗时 (毫秒)
    double cell_steps_per_s; // 吞吐: 细胞×步 / 秒
    bool   lyap_stable;
    double max_loop_gain;
    double smoke_survival;   // 无演化裸机生存率
};

// ---------------------------------------------------------------------------
// Phase A: 规模爬坡实测
// ---------------------------------------------------------------------------
RampRecord ramp_scale(size_t target_cells, uint32_t seed) {
    RampRecord rec;
    rec.target_cells = target_cells;

    auto org = CellularOrganism::create_by_mode(SeedInitMode::HANDCRAFTED_PROGENITOR, 1, seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    org.develop_to_scale(target_cells);
    rec.develop_sec = seconds_since(t0);

    rec.actual_cells = org.cells.size();
    rec.actual_synapses = org.synapses.size();
    rec.mem_mb = static_cast<double>(org.cells.size() * sizeof(Cell) +
                                     org.synapses.size() * sizeof(Synapse)) / (1024.0 * 1024.0);

    // forward 吞吐: 噪声驱动计时
    std::mt19937 rng(seed ^ 0xBEEF);
    std::normal_distribution<double> gauss(0.0, 1.0);
    double inputs[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {  // 预热
        for (double& v : inputs) v = gauss(rng);
        org.forward(inputs, false);
    }
    const int bench_steps = 20;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_steps; ++i) {
        for (double& v : inputs) v = gauss(rng);
        org.forward(inputs, false);
    }
    const double step_sec = seconds_since(t0) / bench_steps;
    rec.step_ms = step_sec * 1000.0;
    rec.cell_steps_per_s = static_cast<double>(org.cells.size()) / step_sec;

    auto lyap = org.check_lyapunov_stability();
    rec.lyap_stable = lyap.is_stable;
    rec.max_loop_gain = lyap.max_loop_gain;

    // 裸机生存率 (未演化、直接丢进熵场) + 峰值电位 (用于标定活力界限的裕度)
    PrimordialViabilityTask task(0.05, 1.004, 2.5);
    task.set_max_steps(400);
    std::vector<uint32_t> seeds = {101, 102, 103};
    auto m = task.evaluate_organism(org, seeds, 400, false);
    rec.smoke_survival = m.mean_fitness;

    PrimordialViabilityTask probe(0.05, 1.004, 1e9);  // 只测电位不死，看真实峰值
    probe.set_max_steps(400);
    probe.evaluate_organism(org, seeds, 400, false);

    std::printf("  N=%9zu 实际=%9zu 突触=%10zu | 发育=%6.2fs 内存=%8.1fMB | "
                "单步=%8.3fms 吞吐=%10.3e cell·step/s | Lyap=%s(增益%.2f) | "
                "裸机生存率=%.1f%% 峰值电位=%.2f\n",
                rec.target_cells, rec.actual_cells, rec.actual_synapses,
                rec.develop_sec, rec.mem_mb, rec.step_ms, rec.cell_steps_per_s,
                rec.lyap_stable ? "稳定" : "失稳", rec.max_loop_gain,
                rec.smoke_survival * 100.0, probe.last_peak_energy());
    return rec;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // 崩溃也不丢日志
    std::printf("==========================================================\n");
    std::printf("  SDSCC 无目标原始进化生命体 · 生存即选择 (C++20 Native)\n");
    std::printf("==========================================================\n");
    std::printf("  硬件: %u 核 | Cell=%zuB Synapse=%zuB\n\n",
                std::thread::hardware_concurrency(),
                sizeof(Cell), sizeof(Synapse));

    // ---------------- Phase A: 规模真理实测 ----------------
    std::printf("[Phase A] 规模爬坡实测 (发育 → 内存 → 吞吐 → 稳定性)\n");
    const std::vector<uint32_t> SEED = {20260903};
    const size_t scales[] = {12, 256, 4096, 65536, 262144, 1048576};
    std::vector<RampRecord> ramp;
    for (size_t n : scales) {
        auto rec = ramp_scale(n, SEED[0]);
        ramp.push_back(rec);
        if (rec.mem_mb > 2048.0) { std::printf("  (内存超预算 2GB, 终止爬坡)\n"); break; }
        if (rec.step_ms > 500.0)  { std::printf("  (单步超 500ms, 终止爬坡)\n"); break; }
    }

    // ---------------- 选择 Phase B 规模 (由机器实测决定) ----------------
    const size_t POP = 12, EVAL_SEEDS = 5, STEPS = 400, GENS = 30;
    const double GEN_TIME_BUDGET = 45.0, MEM_BUDGET_MB = 2048.0;
    size_t phase_b_scale = 0;
    for (const auto& rec : ramp) {
        const double gen_time = static_cast<double>(POP * EVAL_SEEDS * STEPS) * rec.step_ms / 1000.0;
        const double mem_total = rec.mem_mb * static_cast<double>(POP);
        if (gen_time <= GEN_TIME_BUDGET && mem_total <= MEM_BUDGET_MB) {
            phase_b_scale = rec.target_cells;  // 取满足预算的最大档
        }
    }
    if (phase_b_scale == 0) phase_b_scale = 256;
    std::printf("\n[裁决] 本机可行最大训练规模 = %zu 细胞 (种群 %zu, %zu 代预算内)\n\n",
                phase_b_scale, POP, GENS);

    // ---------------- Phase B: 无目标演化 ----------------
    std::printf("[Phase B] 无目标演化: 唯一选择压力=生存, 红皇后熵压逐代 ×1.30\n");
    PrimordialViabilityTask train_task(0.05, 1.004, 2.5);   // 熵压逐代抬升, 活力界限 2.5 (压力咬合)
    train_task.set_max_steps(static_cast<int>(STEPS));
    PrimordialViabilityTask ref_task(0.05, 1.004, 2.5);     // 固定参考熵压 (跨代可比指标)
    ref_task.set_max_steps(static_cast<int>(STEPS));
    std::vector<uint32_t> train_seeds = {101, 102, 103, 104, 105};
    std::vector<uint32_t> ref_seeds   = {901, 902, 903, 904, 905};

    MorphogeneticEvolutionEngine engine(POP, SEED[0], SeedInitMode::HANDCRAFTED_PROGENITOR);
    for (auto& org : engine.population()) org.develop_to_scale(phase_b_scale);

    double best_ref_survival = -1.0;
    CellularOrganism champion;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t gen = 1; gen <= GENS; ++gen) {
        train_task.set_base_pressure(0.05 * std::pow(1.30, static_cast<double>(gen - 1)));
        auto& pop = engine.population();

        double gen_best = 0.0, gen_sum = 0.0;
        size_t best_idx = 0;
        for (size_t i = 0; i < pop.size(); ++i) {
            auto m = train_task.evaluate_organism(pop[i], train_seeds, static_cast<int>(STEPS), true);
            pop[i].fitness_score = m.mean_fitness;
            gen_sum += m.mean_fitness;
            if (m.mean_fitness > gen_best) { gen_best = m.mean_fitness; best_idx = i; }
        }

        // 冠军在固定参考熵压 + 未见种子下的可比生存率
        auto ref_m = ref_task.evaluate_organism(pop[best_idx], ref_seeds, static_cast<int>(STEPS), false);
        if (ref_m.mean_fitness > best_ref_survival || gen == 1) {
            best_ref_survival = ref_m.mean_fitness;
            champion = pop[best_idx];
        }

        std::printf("  Gen %2zu/%zu | 熵压 %.3f | 训练生存 best=%.1f%% mean=%.1f%% "
                    "| 参考生存=%.1f%% | %zu 细胞 %zu 突触\n",
                    gen, GENS, train_task.base_pressure(),
                    gen_best * 100.0, gen_sum / static_cast<double>(POP) * 100.0,
                    ref_m.mean_fitness * 100.0,
                    pop[best_idx].cells.size(), pop[best_idx].synapses.size());

        if (gen < GENS) engine.evolve_generation();
    }
    const double total_sec = seconds_since(t_start);

    // ---------------- 冠军终审 ----------------
    auto lyap = champion.check_lyapunov_stability();
    std::printf("----------------------------------------------------------\n");
    std::printf("  演化完成: 总耗时 %.1fs (%.2fs/代)\n", total_sec, total_sec / GENS);
    std::printf("  冠军: %zu 细胞, %zu 突触, WL=%s\n", champion.cells.size(),
                champion.synapses.size(),
                TaskEvaluator::compute_topology_hash(champion).c_str());
    std::printf("  冠军参考生存率: %.1f%% | Lyapunov: %s (最大环增益 %.3f, 失稳环 %zu)\n",
                best_ref_survival * 100.0, lyap.is_stable ? "稳定" : "失稳",
                lyap.max_loop_gain, lyap.detected_cycles_count);
    std::printf("  [实证] 本机 (12核/27GB) 可训练 %zu 细胞无目标生命体, %.2fs/代\n",
                phase_b_scale, total_sec / GENS);

    if (champion.save_checkpoint_json("checkpoints/primordial_life_champion.json")) {
        std::printf("  [SUCCESS] 原始生命体冠军已存盘: checkpoints/primordial_life_champion.json\n");
    } else {
        std::printf("  [ERROR] 存盘失败\n");
        return 1;
    }
    std::printf("==========================================================\n");
    return 0;
}
