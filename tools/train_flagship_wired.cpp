// ============================================================================
// train_flagship_wired.cpp — 结构化发育接线 × FieldCML2D (T0 机制验证, Stage A)
//
// 目的: 检验「把随机接线换成结构化局部接线」能否让体素场有机体
//       ① 不坍缩 (活性集==全体) ② 超越持续性基线 ③ 优于同规模随机接线。
//
// 对照臂 (--arm):
//   structured (S, 默认): 层单调 2D lattice 局部 + 目标锚点强读出
//   noanchor   (S⁻ᵃ)     : 同上但锚点权重降为保命小权重 (剥离「喂答案」)
//   random     (R′)       : 结构化后打乱内部突触目标端点 (同细胞/突触数, 仅拓扑随机)
//
// 演化: L1 EcologyGrid (移植自系统层) + 权重-only 变异特化 (冻结拓扑, 隔离接线变量)。
// 注意: 本 trainer 不调用 L0 结构变异/凋亡 —— 否则会毁掉结构化拓扑。
//
// 用法: ./train_flagship_wired [N=128] [GENS=60] [ARM=structured]
// ============================================================================
#include "kun/cellular/field_cml_2d.hpp"
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/structured_wiring.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace kun;
using kun::population::ColumnLatticeSpec;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;
using kun::population::ReadoutAnchor;

// ── 权重-only 变异特化 (Stage A 冻结拓扑; 演化机器仍在 L1, 领域语义在此) ──────
namespace kun {
namespace population {
template <>
struct MutationTrait<CellularOrganism> {
    static void mutate(CellularOrganism& org, float rate, float sigma, std::mt19937& rng) {
        if (!org.is_compiled()) org.compile();
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::normal_distribution<double> n(0.0, static_cast<double>(sigma));
        // 权重: 同步改 weight 与 initial_weight (reset_state(true) 每 episode 从 initial_weight 复原)
        for (auto& cs : org.compiled_synapses_) {
            if (u(rng) >= rate) continue;
            auto& s = org.synapses[cs.raw_index];
            const double w = std::clamp(s.initial_weight + n(rng), -2.0, 2.0);
            s.weight = w;
            s.initial_weight = w;
            cs.weight = w;
            cs.initial_weight = w;
        }
        // 细胞参数 (不改变拓扑; dispatch 每拍直读 param1)
        for (auto& c : org.cells) {
            if (u(rng) < rate) c.param1 = std::clamp(c.param1 + n(rng) * 0.1, -2.0, 2.0);
        }
    }
};
}  // namespace population
}  // namespace kun

namespace {

double secs_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

std::vector<uint32_t> seed_range(uint32_t lo, uint32_t count) {
    std::vector<uint32_t> v;
    for (uint32_t i = 0; i < count; ++i) v.push_back(lo + i);
    return v;
}

struct StructStats {
    size_t cells = 0, synapses = 0;
    double active_ratio = 0.0;
};

StructStats stats_of(const CellularOrganism& org) {
    StructStats s;
    s.cells = org.cells.size();
    s.synapses = org.synapses.size();
    s.active_ratio = org.cells.empty() ? 0.0
                                       : (double)org.execution_order_.size() / (double)org.cells.size();
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t n = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 128;
    const size_t gens = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 60;
    const std::string arm = argc > 3 ? argv[3] : "structured";
    const uint32_t G = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 64;  // 列格边长
    // 训练期塑性开关 (Arg5; 默认 1 = 预注册协议)。0 = post-hoc 后续实验 (塑性已证为混淆变量)
    const bool train_plast = argc > 5 ? (std::atoi(argv[5]) != 0) : true;
    if (n % G != 0 || n == 0) {
        std::fprintf(stderr, "N 必须是 G(%u) 的倍数 (场边长 = G*b)\n", G);
        return 1;
    }
    const uint32_t b = static_cast<uint32_t>(n / G);
    const size_t POP = 12;
    const int MS = 200;
    const double coupling = 0.15, r_log = 3.7;
    const std::vector<uint32_t> train_seeds = seed_range(11, 16);
    const std::vector<uint32_t> id_seeds = seed_range(101, 16);
    const std::vector<uint32_t> ood_seeds = seed_range(201, 16);
    const std::vector<uint32_t> ood_regime_seeds = seed_range(301, 8);

    std::printf("=====================================================================\n");
    std::printf("  结构化发育接线 × FieldCML2D | N=%zu (G=%u,b=%u) | arm=%s | POP=%zu | %zu 代\n",
                n, G, b, arm.c_str(), POP, gens);
    std::printf("=====================================================================\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    FieldCML2DTask env(n, /*wide=*/true, coupling, r_log);
    env.set_max_steps(MS);
    const double persist = env.persistence_quality(101, MS);

    // ── 结构化个体工厂 ──
    auto make_org = [&](std::mt19937& rng) {
        ColumnLatticeSpec spec;
        spec.lattice_w = G;
        spec.lattice_h = G;
        spec.vox_per_column_axis = b;
        spec.layers = 4;
        spec.cells_per_layer = 4;
        spec.lateral_radius = 1;
        spec.seed = static_cast<uint32_t>(rng());
        std::vector<ReadoutAnchor> anchors = {{0, 0, 0, 0.5}, {G / 2, G / 2, 1, 0.5}};
        if (arm == "noanchor") {
            for (auto& a : anchors) a.weight = spec.w_readout;
        }
        CellularOrganism org = CellularOrganism::create_seed_organism();
        kun::population::build_columnar_structured_wiring(org, spec, anchors);
        if (arm == "random") {
            kun::population::randomize_internal_synapse_targets(org, static_cast<uint32_t>(rng()));
        }
        return org;
    };

    EcologyGrid<CellularOrganism> grid(1, POP, 20260910u);
    for (auto& d : grid.demes()) d.seed_population(POP, make_org);

    // 结构统计 (个体 0)
    {
        auto st = stats_of(grid.demes()[0].individuals()[0]);
        std::printf("  个体: %zu 细胞 | %zu 突触 | 活性比 %.4f (H1 需 ≥0.99) | 持续性基线 %.4f\n",
                    st.cells, st.synapses, st.active_ratio, persist);
    }

    // ── 每线程独立任务环境 (evaluate_organism 会 reset 任务) ──
    std::atomic<double> cur_scale{0.60};
    auto eval = [&](CellularOrganism& org) -> double {
        thread_local std::unique_ptr<FieldCML2DTask> tenv;
        if (!tenv) {
            tenv = std::make_unique<FieldCML2DTask>(n, true, coupling, r_log);
            tenv->set_max_steps(MS);
        }
        tenv->set_score_scale(cur_scale.load(std::memory_order_relaxed));
        return tenv->evaluate_organism(org, train_seeds, MS, train_plast).mean_fitness;
    };

    EvolutionParams params;
    params.migration_rate = 0.0f;
    params.elite_keep = 3;
    params.tournament_k = 3;
    params.mut_rate = 0.12f;
    params.mut_sigma = 0.20f;
    params.crossover_prob = 0.5f;

    double best = -1e9;
    for (size_t gen = 1; gen <= gens; ++gen) {
        // 课程退火 (与旧旗舰一致)
        if (gen <= gens * 0.5) cur_scale = 0.60;
        else if (gen <= gens * 0.75) cur_scale = 0.30;
        else cur_scale = 0.15;

        auto g0 = std::chrono::high_resolution_clock::now();
        grid.step_generation(eval, params);
        double gms = std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - g0).count();

        for (const auto& d : grid.demes()) {
            for (double f : d.fitness()) best = std::max(best, f);
        }
        if (gen % 10 == 0 || gen <= 2 || gen == gens) {
            std::printf("  代 %zu/%zu | 最佳 %.4f | %.0f ms/代 | %.0fs\n",
                        gen, gens, best, gms, secs_since(t0));
        }
    }

    // ── 冠军 (训练集最优) ──
    const CellularOrganism* champ = nullptr;
    double champ_fit = -1e18;
    for (const auto& d : grid.demes()) {
        for (size_t i = 0; i < d.individuals().size(); ++i) {
            if (d.fitness()[i] > champ_fit) {
                champ_fit = d.fitness()[i];
                champ = &d.individuals()[i];
            }
        }
    }
    CellularOrganism best_org = *champ;  // 拷贝以评测 (评测会改状态)

    auto st = stats_of(best_org);
    env.set_score_scale(0.15);
    const double un_id = env.evaluate_organism(best_org, id_seeds, MS, true).mean_fitness;
    const double un_id_nop = env.evaluate_organism(best_org, id_seeds, MS, false).mean_fitness;

    FieldCML2DTask ood_env(n, true, 0.22, 3.9);  // 同尺寸异动力学
    ood_env.set_max_steps(MS);
    ood_env.set_score_scale(0.15);
    const double un_ood = ood_env.evaluate_organism(best_org, ood_regime_seeds, MS, true).mean_fitness;
    const double un_ood_seed = env.evaluate_organism(best_org, ood_seeds, MS, true).mean_fitness;
    const double un_ood_nop = ood_env.evaluate_organism(best_org, ood_regime_seeds, MS, false).mean_fitness;
    const double un_ood_seed_nop = env.evaluate_organism(best_org, ood_seeds, MS, false).mean_fitness;

    std::printf("\n---------------------------------------------------------------------\n");
    std::printf("  arm=%s | 训练塑性=%d | 冠军: %zu 细胞 | %zu 突触 | 活性比 %.4f\n",
                arm.c_str(), train_plast ? 1 : 0, st.cells, st.synapses, st.active_ratio);
    std::printf("  持续性基线 P       : %.4f\n", persist);
    std::printf("  训练最佳           : %.4f\n", best);
    std::printf("  未见(id)越 PLASTIC : %.4f\n", un_id);
    std::printf("  未见(id)无 PLASTIC : %.4f   (H3 主口径: S-P ≥ +0.02)\n", un_id_nop);
    std::printf("  未见(ood 种子)无塑 : %.4f\n", un_ood_seed_nop);
    std::printf("  未见(ood 体制)无塑 : %.4f   (H7: /训练 需 ≥0.70)\n", un_ood_nop);
    std::printf("  [参考] 塑开口径    : id %.4f | ood种子 %.4f | ood体制 %.4f\n",
                un_id, un_ood_seed, un_ood);
    std::printf("  耗时 %.0fs\n", secs_since(t0));
    std::printf("---------------------------------------------------------------------\n");
    return 0;
}
