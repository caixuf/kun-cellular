// ============================================================================
// train_flagship_wired.cpp — 结构化发育接线 × FieldCML2D (T0 机制验证, Stage A)
//
// 目的: 检验「把随机接线换成结构化局部接线」能否让体素场有机体
//       ① 不坍缩 (活性集==全体) ② 超越持续性基线 ③ 优于同规模随机接线。
//
// 对照臂 (--arm):
//   structured (S, 默认): 层单调 2D lattice 局部 + 目标锚点强读出
//   noanchor   (S⁻ᵃ)     : 同上但锚点权重降为保命小权重 (剥离「喂答案」)
//   random     (R′)       : 结构化后打乱全部突触目标端点 (同细胞/突触数; 递归多为意外)
//   ampfix     (S^amp)    : 单变量① 幅度校准 — param1=1.0 + w_readout=w_receptor (拓扑同 S)
//   recur      (S^rec)    : 单变量② 层间反馈 — 同列 V_{l+1}→V_l (幅度同 S 默认)
//   rrac       (路线 A)   : 层间反馈 + 保 IO 内部随机化 + 活性闭包修复
//   rrac_dag   (RRAC⁻ʳ)   : 同上但不加层间反馈 (递归必要性消融)
//   rand_recv  (R′解剖)   : 仅打乱受体出边目标 (+闭包修复)
//   rand_eff   (R′解剖)   : 仅打乱效应器入边目标 (+闭包修复)
//   rand_io    (R′解剖)   : 打乱受体出边∪效应器入边 (+闭包修复)
//   io_mix     (发育默认) : 同 rand_io, L1 apply_io_mix_developmental 一等公民入口
//   io_evo     (形态发生) : 结构化骨架初始化 + 代际 IO 轴突重连 (论文口径, 非梯度投影)
//
// 演化: L1 EcologyGrid; 默认权重-only。io_evo 额外启用 IO 轴突重塑。
// 注意: 本 trainer 不调用 L0 结构变异/凋亡 —— 否则会毁掉结构化拓扑。
//
// 用法: ./train_flagship_wired [N=128] [GENS=60] [ARM=structured] [G=64] [train_plast=1] [evo_seed=20260910]
// ============================================================================
#include "kun/cellular/field_cml_2d.hpp"
#include "kun/cellular/population/ecology_grid.hpp"
#include "kun/cellular/population/structured_wiring.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace kun;
using kun::population::ColumnLatticeSpec;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;
using kun::population::ReadoutAnchor;
using kun::population::recurrent_synapse_count;

// 0=权重-only; 1=权重 + IO 轴突重连 (io_evo)
static std::atomic<int> g_wired_mut_mode{0};
static std::atomic<uint64_t> g_io_rewire_total{0};

// ── 变异特化 (Stage A 默认冻结内部拓扑; io_evo 仅开放 IO 轴突形态发生) ──────
namespace kun {
namespace population {
template <>
struct MutationTrait<CellularOrganism> {
    static void mutate(CellularOrganism& org, float rate, float sigma, std::mt19937& rng) {
        if (g_wired_mut_mode.load(std::memory_order_relaxed) == 1) {
            const size_t rw = rewire_io_axon_targets(org, rate, rng);
            if (rw > 0) {
                g_io_rewire_total.fetch_add(static_cast<uint64_t>(rw), std::memory_order_relaxed);
                ensure_active_closure(org);
            }
        }
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
    size_t cells = 0, synapses = 0, recurrent = 0;
    double active_ratio = 0.0;
};

StructStats stats_of(const CellularOrganism& org) {
    StructStats s;
    s.cells = org.cells.size();
    s.synapses = org.synapses.size();
    s.recurrent = recurrent_synapse_count(org);
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
    const uint32_t evo_seed = argc > 6 ? static_cast<uint32_t>(std::strtoul(argv[6], nullptr, 10))
                                       : 20260910u;
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
    std::printf("  结构化发育接线 × FieldCML2D | N=%zu (G=%u,b=%u) | arm=%s | POP=%zu | %zu 代 | seed=%u\n",
                n, G, b, arm.c_str(), POP, gens, evo_seed);
    std::printf("=====================================================================\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    FieldCML2DTask env(n, /*wide=*/true, coupling, r_log);
    env.set_max_steps(MS);
    const double persist = env.persistence_quality(101, MS);

    // 空间换时间: 场轨迹只走一遍, 种群评测直接传送观测帧
    const auto train_rollouts = FieldCML2DTask::record_rollouts(env, train_seeds, MS);
    const auto id_rollouts = FieldCML2DTask::record_rollouts(env, id_seeds, MS);
    const auto ood_seed_rollouts = FieldCML2DTask::record_rollouts(env, ood_seeds, MS);
    FieldCML2DTask ood_env(n, true, 0.22, 3.9);
    ood_env.set_max_steps(MS);
    ood_env.set_score_scale(0.15);
    const auto ood_regime_rollouts = FieldCML2DTask::record_rollouts(ood_env, ood_regime_seeds, MS);
    {
        const size_t bytes = train_rollouts.empty() ? 0
            : train_rollouts.size() * train_rollouts[0].frames.size() * sizeof(double);
        std::printf("  轨迹缓存: 训练 %zu 种子 × %d 步 | ~%.1f MiB | 预热+推进已走完\n",
                    train_rollouts.size(), MS, bytes / (1024.0 * 1024.0));
    }

    // ── 结构化个体工厂 ──
    size_t last_rewritten = 0, last_repaired = 0;
    bool rrac_diag_captured = false;
    auto make_org = [&](std::mt19937& rng) {
        ColumnLatticeSpec spec;
        spec.lattice_w = G;
        spec.lattice_h = G;
        spec.vox_per_column_axis = b;
        spec.layers = 4;
        spec.cells_per_layer = 4;
        spec.lateral_radius = 1;
        spec.seed = static_cast<uint32_t>(rng());
        // 单变量消融: ampfix 只改幅度; recur/rrac 加反馈; 其余保持 Stage A 默认
        if (arm == "ampfix") {
            spec.internal_param1 = 1.0f;
            spec.w_readout = spec.w_receptor;  // 0.15 — 解除读出衰减
        } else if (arm == "recur" || arm == "rrac") {
            spec.add_interlayer_feedback = true;
            spec.w_feedback = 0.10f;
        }
        // rrac_dag: 不加反馈 (A4 消融)
        std::vector<ReadoutAnchor> anchors = {{0, 0, 0, 0.5}, {G / 2, G / 2, 1, 0.5}};
        if (arm == "noanchor") {
            for (auto& a : anchors) a.weight = spec.w_readout;
        }
        CellularOrganism org = CellularOrganism::create_seed_organism();
        kun::population::build_columnar_structured_wiring(org, spec, anchors);
        if (arm == "random") {
            kun::population::randomize_internal_synapse_targets(org, static_cast<uint32_t>(rng()));
        } else if (arm == "rrac" || arm == "rrac_dag") {
            const size_t rw = kun::population::randomize_internal_preserving_io(
                org, static_cast<uint32_t>(rng()));
            const size_t rp = kun::population::ensure_active_closure(org);
            if (!rrac_diag_captured) {
                last_rewritten = rw;
                last_repaired = rp;
                rrac_diag_captured = true;
            }
        } else if (arm == "rand_recv" || arm == "rand_eff" || arm == "rand_io" ||
                   arm == "io_mix") {
            if (arm == "io_mix") {
                const auto [rw, rp] = kun::population::apply_io_mix_developmental(
                    org, static_cast<uint32_t>(rng()));
                if (!rrac_diag_captured) {
                    last_rewritten = rw;
                    last_repaired = rp;
                    rrac_diag_captured = true;
                }
            } else {
                kun::population::EdgeClassRandomizeFlags fl;
                fl.receptor_out = (arm == "rand_recv" || arm == "rand_io");
                fl.effector_in = (arm == "rand_eff" || arm == "rand_io");
                fl.internal = false;
                const size_t rw = kun::population::randomize_targets_by_edge_class(
                    org, fl, static_cast<uint32_t>(rng()));
                const size_t rp = kun::population::ensure_active_closure(org);
                if (!rrac_diag_captured) {
                    last_rewritten = rw;
                    last_repaired = rp;
                    rrac_diag_captured = true;
                }
            }
        }
        // io_evo: 保持结构化 IO 骨架, 代际由 MutationTrait 做轴突重连
        return org;
    };

    g_wired_mut_mode.store(arm == "io_evo" ? 1 : 0, std::memory_order_relaxed);
    g_io_rewire_total.store(0, std::memory_order_relaxed);

    EcologyGrid<CellularOrganism> grid(1, POP, evo_seed);
    for (auto& d : grid.demes()) d.seed_population(POP, make_org);

    // 结构统计 (个体 0)
    {
        auto st = stats_of(grid.demes()[0].individuals()[0]);
        auto bp0 = kun::population::build_receptor_bypass(grid.demes()[0].individuals()[0]);
        std::printf("  个体: %zu 细胞 | %zu 突触 | 递归边 %zu | 活性比 %.4f (A1/H1 需 ≥0.99) | 持续性基线 %.4f\n",
                    st.cells, st.synapses, st.recurrent, st.active_ratio, persist);
        std::printf("  受体直路: %s | 受体 %u | 注入边 %zu | 内部序 %zu\n",
                    bp0.ok ? "开" : "关", bp0.n_receptors, bp0.from_idx.size(),
                    bp0.internal_order.size());
        if (arm == "rrac" || arm == "rrac_dag" || arm == "rand_recv" ||
            arm == "rand_eff" || arm == "rand_io" || arm == "io_mix") {
            std::printf("  边类诊断: 改写边 %zu | 活性修复边 %zu\n",
                        last_rewritten, last_repaired);
        }
        if (arm == "io_evo") {
            std::printf("  形态发生: 代际 IO 轴突重连 开 (mut_mode=1) | 初始化=结构化骨架\n");
        }
    }

    // ── 传送评测: 只跑有机体前向, 场不再重算 ──
    std::atomic<double> cur_scale{0.60};
    auto eval = [&](CellularOrganism& org) -> double {
        const double scale = cur_scale.load(std::memory_order_relaxed);
        if (train_plast) {
            return env.score_rollouts(org, train_rollouts, true, scale);
        }
        const auto bypass = kun::population::build_receptor_bypass(org);
        return env.score_rollouts(org, train_rollouts, scale,
            [&](CellularOrganism& o, const double* x, size_t d) {
                return kun::population::forward_nd_skip_receptors(o, bypass, x, d);
            });
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
    const uint64_t io_rewires = g_io_rewire_total.load(std::memory_order_relaxed);
    env.set_score_scale(0.15);
    const double un_id = env.score_rollouts(best_org, id_rollouts, true, 0.15);
    const double un_id_nop = env.score_rollouts(best_org, id_rollouts, false, 0.15);
    const double un_ood = ood_env.score_rollouts(best_org, ood_regime_rollouts, true, 0.15);
    const double un_ood_seed = env.score_rollouts(best_org, ood_seed_rollouts, true, 0.15);
    const double un_ood_nop = ood_env.score_rollouts(best_org, ood_regime_rollouts, false, 0.15);
    const double un_ood_seed_nop = env.score_rollouts(best_org, ood_seed_rollouts, false, 0.15);

    std::printf("\n---------------------------------------------------------------------\n");
    std::printf("  arm=%s | seed=%u | 训练塑性=%d | 冠军: %zu 细胞 | %zu 突触 | 递归边 %zu | 活性比 %.4f\n",
                arm.c_str(), evo_seed, train_plast ? 1 : 0, st.cells, st.synapses, st.recurrent, st.active_ratio);
    std::printf("  持续性基线 P       : %.4f\n", persist);
    std::printf("  训练最佳           : %.4f\n", best);
    std::printf("  未见(id)越 PLASTIC : %.4f\n", un_id);
    std::printf("  未见(id)无 PLASTIC : %.4f   (H3 主口径: S-P ≥ +0.02)\n", un_id_nop);
    std::printf("  未见(ood 种子)无塑 : %.4f\n", un_ood_seed_nop);
    std::printf("  未见(ood 体制)无塑 : %.4f   (H7: /训练 需 ≥0.70)\n", un_ood_nop);
    std::printf("  [参考] 塑开口径    : id %.4f | ood种子 %.4f | ood体制 %.4f\n",
                un_id, un_ood_seed, un_ood);
    if (arm == "io_evo") {
        const uint64_t e5_floor = static_cast<uint64_t>(POP) * static_cast<uint64_t>(gens);
        std::printf("  IO轴突重连累计   : %llu  (E5 门槛 ≥ %llu)\n",
                    static_cast<unsigned long long>(io_rewires),
                    static_cast<unsigned long long>(e5_floor));
    }
    std::printf("  耗时 %.0fs\n", secs_since(t0));
    std::printf("  JSON {\"arm\":\"%s\",\"seed\":%u,\"n\":%zu,\"G\":%u,\"gens\":%zu,\"plast\":%d,"
                "\"train_best\":%.6f,\"un_id_nop\":%.6f,\"ood_regime_nop\":%.6f,"
                "\"active\":%.6f,\"persist\":%.6f,\"cells\":%zu,\"syns\":%zu,\"recurrent\":%zu,"
                "\"io_rewire_total\":%llu}\n",
                arm.c_str(), evo_seed, n, G, gens, train_plast ? 1 : 0,
                best, un_id_nop, un_ood_nop, st.active_ratio, persist,
                st.cells, st.synapses, st.recurrent,
                static_cast<unsigned long long>(io_rewires));
    {
        char path[256];
        std::snprintf(path, sizeof(path), "checkpoints/flagship_%s_n%zu_s%u.bin",
                      arm.c_str(), n, evo_seed);
        best_org.save_checkpoint_bin(path);
        std::printf("  冠军已存: %s\n", path);
    }
    std::printf("---------------------------------------------------------------------\n");
    return 0;
}
