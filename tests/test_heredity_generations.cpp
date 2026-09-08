// U3.1 + U3.2: 冠军 germline 出版 (assimilation 显式提交) + 代际实验
// Baldwin vs Lamarck 测量: 父代 BPTT 训练 → live 参数同化入先天模板 (语义版本化)
// → 子代从 germline 出生 (Lamarck 热启) vs 随机出生 (Baldwin 起点):
// 断言出生表现差 (量化遗传传递), 且 Baldwin 子代可训练收敛 (生命可塑性)。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/heredity.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/knowledge_module.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace kun;

namespace {

inline Cell make_cell(uint32_t id, CellType type, double param1 = 1.0, double param2 = 0.0) {
    Cell c;
    c.id = id;
    c.type = type;
    c.param1 = param1;
    c.param2 = param2;
    return c;
}

inline Synapse make_synapse(uint32_t from_id, uint32_t to_id, uint8_t port, double weight) {
    Synapse s;
    s.from_cell_id = from_id;
    s.to_cell_id = to_id;
    s.to_port = port;
    s.weight = weight;
    s.initial_weight = weight;
    s.is_active = true;
    return s;
}

// 可训练回归图 (容量足够): 0→2 双份 (2x0), 1→3 双份 (2x1), 3→5 权重 0.25 (0.5x1)
CellularOrganism build_regression_organism() {
    CellularOrganism org;
    org.organism_id = 7;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0));
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_1));
    org.cells.push_back(make_cell(2, CellType::OP_SUM));
    org.cells.push_back(make_cell(3, CellType::OP_SUM));
    org.cells.push_back(make_cell(4, CellType::ACT_CHANNEL, 1.0, 0.0));
    org.cells.push_back(make_cell(5, CellType::ACT_CHANNEL, 1.0, 1.0));
    org.synapses.push_back(make_synapse(0, 2, 0, 1.0));
    org.synapses.push_back(make_synapse(0, 2, 1, 1.0));
    org.synapses.push_back(make_synapse(1, 3, 0, 1.0));
    org.synapses.push_back(make_synapse(1, 3, 1, 1.0));
    org.synapses.push_back(make_synapse(2, 4, 0, 1.0));
    org.synapses.push_back(make_synapse(3, 5, 0, 0.25));
    org.compile();
    return org;
}

std::vector<std::vector<double>> make_stream(std::mt19937& rng, size_t n) {
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<std::vector<double>> stream;
    for (size_t t = 0; t < n; ++t) {
        std::vector<double> v(4);
        for (auto& x : v) x = uni(rng);
        stream.push_back(v);
    }
    return stream;
}

// 外部教师 (出生不可达): 通道0 = 3*x0 (出生 2x0), 通道1 = 0.8*x1 (出生 0.5x1)
double teacher_ch0(const std::vector<double>& x) { return 3.0 * x[0]; }
double teacher_ch1(const std::vector<double>& x) { return 0.8 * x[1]; }

// 随机种子 germline (出生图 + 出生种子)
core::GermlineResult make_random_germline(const CellularOrganism& org) {
    auto imported = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(1), core::GraphRevision(1));
    if (!imported.ok()) return {nullptr, {}};
    const auto& snap = *imported.snapshot;
    return core::Germline::create(
        snap.graph_definition(), snap.initial_parameter_values()->entries().size()
            ? core::InitialParameterSeeds{}
            : core::InitialParameterSeeds{});
    (void)snap;
}

// 全部边权重参数 binding (学习窗口)
std::vector<core::ParameterBinding> all_edge_weight_bindings(
    const core::CompiledGraph& plan) {
    std::vector<core::ParameterBinding> out;
    for (const auto& e : plan.edges()) {
        core::ParameterBinding b;
        b.kind = core::ParameterBindingKind::EdgeWeight;
        b.index = e.weight_parameter_index;
        b.edge = e.id;
        out.push_back(b);
    }
    return out;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(2026);
    const auto stream = make_stream(rng, 24);

    // ── 出生 germline (冠军的"基因组") ──
    auto org = build_regression_organism();
    auto imported = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(1), core::GraphRevision(1));
    assert(imported.ok());
    auto birth_germline = core::Germline::create(
        imported.snapshot->graph_definition(),
        kun::transfer::parameter_seeds(
            imported.snapshot->initial_parameter_values()->entries()));
    if (!birth_germline.ok()) {
        printf("[错误] 出生模板编译被拒: %s\n",
               birth_germline.error ? birth_germline.error->reason.c_str() : "?");
        return 1;
    }
    printf("[出生模板] 细胞=%zu 边=%zu 版本=%u\n",
           birth_germline.germline->graph()->cells().size(),
           birth_germline.germline->graph()->edges().size(),
           birth_germline.germline->version().value);

    // ── 父代: 从出生模板诞生 → BPTT 训练拟合教师 ──
    core::OffspringSpec spec;
    spec.organism_id = 100;
    spec.rng_seed = 42;
    core::LifecycleConfig lccfg;
    lccfg.apoptotic_resource = 5.0;
    lccfg.dormant_enter_resource = 10.0;
    lccfg.dormant_exit_resource = 100.0;
    spec.lifecycle_config = lccfg;
    core::ResourceLedgerConfig rcfg;
    rcfg.maintenance_cost = 0.01;
    rcfg.activity_cost = 0.05;
    spec.resource_config = rcfg;
    for (const auto& c : birth_germline.germline->graph()->cells())
        spec.resource_cells.push_back({c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
    spec.resource_compartments.push_back({core::ResourceCompartmentId{0}, 1e9});

    auto parent = core::Phenotype::create(birth_germline.germline, spec);
    assert(parent.ok());
    printf("[父代] 诞生 ✓ (organism=%llu)\n",
           (unsigned long long)spec.organism_id);

    // 学习: 全边权重窗口 + BPTT (executor 自备: 同 plan 绑定)
    auto parent_exec = core::CompiledExecutor::prepare(parent.phenotype->runtime().plan());
    assert(parent_exec.ok());
    auto bindings = all_edge_weight_bindings(*parent.phenotype->runtime().plan());
    auto window = core::LearningWindow::open(100, parent.phenotype->runtime(), bindings);
    CoreCellularBPTTEngine engine(32);
    engine.init_optimizer(parent.phenotype->runtime());
    const double lr = 0.05;
    double parent_loss = -1.0;
    for (int epoch = 0; epoch < 120; ++epoch) {
        auto reset = parent.phenotype->runtime().reset_episode();
        assert(!reset.error.has_value());
        engine.reset_tape();
        std::vector<std::vector<double>> targets;
        for (const auto& x : stream) {
            auto rec = engine.record_step(parent.phenotype->runtime(), *parent_exec.executor, x);
            assert(rec.ok());
            // 目标行: ACT 通道给教师, 其他细胞给自身当前输出 (零额外梯度)
            targets.push_back({teacher_ch0(x), teacher_ch1(x)});
        }
        CoreBPTTGradients grads;
        auto back = engine.backward(parent.phenotype->runtime(), targets, grads, &window);
        assert(back.ok());
        auto upd = engine.step_adam(parent.phenotype->runtime(), window, grads, lr);
        assert(upd.ok());
        parent_loss = grads.loss;
        if (epoch % 30 == 0 || epoch == 119)
            printf("  [父代训练] epoch %d loss=%.6f\n", epoch, grads.loss);
    }
    assert(parent_loss < 0.01);  // 父代必须真正学会 (图容量足够)

    // 父代表现 (出生模板的对照组: 未训练随机个体在同任务上的 loss)
    auto random_child = core::Phenotype::create(birth_germline.germline, spec);
    assert(random_child.ok());
    double random_loss = 0.0;
    {
        for (const auto& x : stream) {
            auto r = random_child.phenotype->step(x);
            assert(r.ok());
            for (const auto& c : random_child.phenotype->runtime().cell_states()) {
                if (c.type == CellType::ACT_CHANNEL)
                    random_loss += std::fabs(
                        c.output_val - (c.cell.value == 4 ? teacher_ch0(x) : teacher_ch1(x)));
            }
        }
    }
    printf("[对照] 随机出生 loss=%.4f | 父代训练后 loss=%.4f\n",
           random_loss, parent_loss);

    // ── U3.1 assimilation: 父代 live 参数显式同化入先天模板 (语义版本化) ──
    const auto live_def = kun::transfer::live_definition(parent.phenotype->runtime());
    const auto live_seeds = kun::transfer::parameter_seeds(parent.phenotype->runtime().parameters());
    const size_t assimilated_values =
        live_seeds.edge_weights.size() + live_seeds.cell_parameters.size();
    auto champion_germline = core::Germline::create(live_def, live_seeds, "doudizhu-champion-v1");
    assert(champion_germline.ok());
    printf("[U3.1 同化] 显式提交: %zu 参数写入先天模板 '%s' (版本 %u)\n",
           assimilated_values, champion_germline.germline->development_rules().c_str(),
           champion_germline.germline->version().value);
    // 同化保真: germline 种子 = 父代 live
    for (const auto& seed : champion_germline.germline->initial_values().entries()) {
        const auto& live = parent.phenotype->runtime().parameters()[seed.binding.index];
        if (auto cv = std::get_if<core::ContinuousValue>(&seed.value))
            assert(std::fabs(cv->value -
                std::get<core::ContinuousValue>(live.value).value) < 1e-12);
    }
    printf("[U3.1 同化保真] germline 种子 = 父代 live (位级) ✓\n");

    // ── U3.2 代际: Lamarck 子代 (冠军 germline 出生) vs Baldwin 子代 (随机出生) ──
    auto measure_birth_loss = [&](core::GermlineResult& g) {
        auto child = core::Phenotype::create(g.germline, spec);
        assert(child.ok());
        double loss = 0.0;
        for (const auto& x : stream) {
            auto r = child.phenotype->step(x);
            assert(r.ok());
            for (const auto& c : child.phenotype->runtime().cell_states()) {
                if (c.type == CellType::ACT_CHANNEL)
                    loss += std::fabs(
                        c.output_val - (c.cell.value == 4 ? teacher_ch0(x) : teacher_ch1(x)));
            }
        }
        return loss;
    };
    // Baldwin 对照: 出生种子扰动 (边权重 × U(0.5,1.5) — 发育噪声)
    core::InitialParameterSeeds noisy_seeds;
    {
        const auto& src_seeds = kun::transfer::parameter_seeds(
            imported.snapshot->initial_parameter_values()->entries());
        noisy_seeds.cell_parameters = src_seeds.cell_parameters;
        std::mt19937 nrng(99);
        std::uniform_real_distribution<double> nu(0.5, 1.5);
        for (const auto& w : src_seeds.edge_weights)
            noisy_seeds.edge_weights.push_back({w.edge, w.initial_weight * nu(nrng)});
    }
    auto noisy_germline = core::Germline::create(
        imported.snapshot->graph_definition(), noisy_seeds);
    assert(noisy_germline.ok());
    const double lamarck_birth = measure_birth_loss(champion_germline);
    const double baldwin_birth = measure_birth_loss(noisy_germline);
    printf("[U3.2 代际] Lamarck 出生 loss=%.4f | Baldwin 出生 loss=%.4f (比例 %.1fx)\n",
           lamarck_birth, baldwin_birth, baldwin_birth / std::max(lamarck_birth, 1e-9));
    assert(lamarck_birth < 0.1 * baldwin_birth);  // 遗传传递显著

    // Baldwin 子代可训练性 (生命可塑性): 30 epoch 后显著改善
    {
        auto baldwin = core::Phenotype::create(birth_germline.germline, spec);
        assert(baldwin.ok());
        auto baldwin_exec = core::CompiledExecutor::prepare(baldwin.phenotype->runtime().plan());
        assert(baldwin_exec.ok());
        auto w = core::LearningWindow::open(101, baldwin.phenotype->runtime(), bindings);
        CoreCellularBPTTEngine e2(32);
        e2.init_optimizer(baldwin.phenotype->runtime());
        double first_loss = -1.0, last_loss = -1.0;
        for (int epoch = 0; epoch < 120; ++epoch) {
            auto reset = baldwin.phenotype->runtime().reset_episode();
            assert(!reset.error.has_value());
            e2.reset_tape();
            std::vector<std::vector<double>> targets;
            for (const auto& x : stream) {
                auto rec = e2.record_step(baldwin.phenotype->runtime(), *baldwin_exec.executor, x);
                assert(rec.ok());
                targets.push_back({teacher_ch0(x), teacher_ch1(x)});
            }
            CoreBPTTGradients grads;
            auto back = e2.backward(baldwin.phenotype->runtime(), targets, grads, &w);
            assert(back.ok());
            auto upd = e2.step_adam(baldwin.phenotype->runtime(), w, grads, lr);
            assert(upd.ok());
            if (epoch == 0) first_loss = grads.loss;
            last_loss = grads.loss;
        }
        printf("[U3.2 可塑性] Baldwin 子代训练: %.4f → %.4f (生命可塑 ✓)\n",
               first_loss, last_loss);
        assert(last_loss < first_loss * 0.5);
    }

    printf("[U3 遗传与书] 全部通过\n");
    return 0;
}
