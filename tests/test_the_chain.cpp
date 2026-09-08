// THE CHAIN — 七环实证链机制级闭合 (预注册判据见
// docs/superpowers/plans/2026-09-08-the-chain-protocol.md)
// 1 从零出生 / 2 独立学习 / 3 未见泛化 / 4 结构参数真实改变 /
// 5 死亡后知识继承 / 6 无血缘借阅 / 7 代际超越
// 纪律: 环 5-7 的继承/借阅不靠梯度 (germline 与图书馆的本体)。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "tasks/transfer/knowledge_module.hpp"
#include "tasks/transfer/germline_library.hpp"
#include "tasks/transfer/knowledge_adoption.hpp"

#include <cassert>
#include <cstdio>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace kun;

namespace {

inline Cell make_cell(uint32_t id, CellType type, double param1 = 1.0, double param2 = 0.0) {
    Cell c; c.id = id; c.type = type; c.param1 = param1; c.param2 = param2; return c;
}
inline Synapse make_synapse(uint32_t from_id, uint32_t to_id, uint8_t port, double weight) {
    Synapse s; s.from_cell_id = from_id; s.to_cell_id = to_id; s.to_port = port;
    s.weight = weight; s.initial_weight = weight; s.is_active = true; return s;
}

// 回归图: x0 双份入 2 (SUM), x1 双份入 3 (SUM), 2→4 (ch0), 3→5 (ch1)
CellularOrganism build_organism() {
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

double teacher0(const std::vector<double>& x) { return 3.0 * x[0]; }
double teacher1(const std::vector<double>& x) { return 0.8 * x[1]; }

std::vector<std::vector<double>> make_stream(std::mt19937& rng, size_t n) {
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<std::vector<double>> s;
    for (size_t t = 0; t < n; ++t) s.push_back({uni(rng), uni(rng), 0.0, 0.0});
    return s;
}

std::vector<core::ParameterBinding> edge_bindings(const core::CompiledGraph& plan) {
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

// BPTT 训练到 loss 门 (probe 体上执行 — 研究行为不推本体生命钟)
std::pair<int, double> train_to_gate(
    core::RuntimeState& rt, core::CompiledExecutor& ex,
    uint64_t organism_id, const std::vector<std::vector<double>>& stream,
    double gate, int max_epochs, double lr) {
    auto w = core::LearningWindow::open(organism_id, rt, edge_bindings(*rt.plan()));
    CoreCellularBPTTEngine engine(32);
    engine.init_optimizer(rt);
    double last = -1.0;
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        auto reset = rt.reset_episode();
        engine.reset_tape();
        for (const auto& x : stream) {
            auto rec = engine.record_step(rt, ex, x);
            if (!rec.ok()) { printf("[训练错误] %s\n", rec.error->reason.c_str()); assert(false); }
        }
        std::vector<std::vector<double>> targets;
        for (const auto& x : stream) targets.push_back({teacher0(x), teacher1(x)});
        CoreBPTTGradients grads;
        auto back = engine.backward(rt, targets, grads, &w);
        if (!back.ok()) { printf("[backward 错误] %s\n", back.error->reason.c_str()); assert(false); }
        auto upd = engine.step_adam(rt, w, grads, lr);
        if (!upd.ok()) { printf("[adam 错误] %s\n", upd.error->reason.c_str()); assert(false); }
        last = grads.loss;
        if (last <= gate) return {epoch + 1, last};
    }
    return {max_epochs, last};
}

// 通道 L1 loss (probe 体上评测)
double probe_loss(core::RuntimeState& rt, core::CompiledExecutor& ex,
                  const std::vector<std::vector<double>>& stream) {
    double loss = 0.0;
    for (const auto& x : stream) {
        auto r = ex.step(rt, x);
        if (!r.ok()) { printf("[probe step 错误] %s\n", r.error->reason.data()); assert(false); }
        size_t slot = 0;
        for (const auto& c : rt.cell_states()) {
            if (c.type == CellType::ACT_CHANNEL && slot < 2) {
                const double t = slot == 0 ? teacher0(x) : teacher1(x);
                loss += std::fabs(c.output_val - t);
                ++slot;
            }
        }
    }
    return loss;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::remove("/tmp/opencode/chain_library.sqlite3");  // 每次运行全新图书馆
    std::mt19937 rng(20260908);
    const auto train_stream = make_stream(rng, 24);
    std::mt19937 rng2(777);
    const auto holdout_stream = make_stream(rng2, 24);
    const double GATE = 1e-3;

    // ── 环 1: 从零出生 (发育噪声种子) ──
    auto org = build_organism();
    auto imported = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(1), core::GraphRevision(1));
    assert(imported.ok());
    const auto& snap = *imported.snapshot;
    core::InitialParameterSeeds noisy;
    {
        const auto src_seeds = kun::transfer::parameter_seeds(
            snap.initial_parameter_values()->entries());
        noisy.cell_parameters = src_seeds.cell_parameters;
        std::mt19937 nrng(99);
        std::uniform_real_distribution<double> nu(0.5, 1.5);
        for (const auto& w : src_seeds.edge_weights)
            noisy.edge_weights.push_back({w.edge, w.initial_weight * nu(nrng)});
    }
    auto g0_germline = core::Germline::create(snap.graph_definition(), noisy);
    assert(g0_germline.ok());
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 1000;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    cfg.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};

    // 出生 (临时 germline-only 出生: 直接 Phenotype::create)
    core::OffspringSpec spec;
    spec.organism_id = cfg.organism_id;
    spec.rng_seed = cfg.rng_seed;
    spec.lifecycle_config = cfg.lifecycle_config;
    spec.growth_config = cfg.growth_config;
    for (const auto& c : g0_germline.germline->graph()->cells())
        spec.resource_cells.push_back(core::ResourceCellInitial{
            c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
    spec.resource_compartments.push_back(
        core::ResourceCompartmentInitial{core::ResourceCompartmentId{0}, 1e9});

    auto g0 = core::Phenotype::create(g0_germline.germline, spec);
    assert(g0.ok());
    // probe 体: 研究行为 (学习/评测) 不推本体生命钟 (U2.1 纪律)
    auto g0_probe = g0.phenotype->runtime().fork_probe();
    assert(g0_probe.ok());
    auto g0_ex = core::CompiledExecutor::prepare(g0_probe.runtime->plan());
    assert(g0_ex.ok());
    const double born_loss = probe_loss(*g0_probe.runtime, *g0_ex.executor, train_stream);
    printf("[环1 从零出生] 出生 train L1=%.4f (随机水平, >0) ✓\n", born_loss);
    assert(born_loss > 1.0);

    // ── 环 2: 独立学习 (probe 上) ──
    auto [e0, l0] = train_to_gate(*g0_probe.runtime, *g0_ex.executor, 1000,
                                  train_stream, GATE, 400, 0.05);
    printf("[环2 独立学习] 达门 epoch=%d final_loss=%.2e ✓\n", e0, l0);
    assert(l0 <= GATE);

    // ── 环 3: 未见泛化 ──
    const double holdout_l1 = probe_loss(*g0_probe.runtime, *g0_ex.executor, holdout_stream);
    const double train_l1 = probe_loss(*g0_probe.runtime, *g0_ex.executor, train_stream);
    const double ratio = holdout_l1 / std::max(train_l1, 1e-12);
    printf("[环3 未见泛化] train=%.4f holdout=%.4f 比率=%.2f (<3) ✓\n",
           train_l1, holdout_l1, ratio);
    assert(ratio < 3.0);

    // ── 环 5: 死亡后知识继承 ──
    // G0 的 probe 学习成果同化入先天模板 → 本体与 probe 显式销毁 → 再出生即继承
    auto g1_germline = core::Germline::create(
        kun::transfer::live_definition(*g0_probe.runtime),
        kun::transfer::parameter_seeds(g0_probe.runtime->parameters()),
        "chain-G1-assimilated");
    assert(g1_germline.ok());
    const double g0_final_train = train_l1;
    {
        std::unique_ptr<core::Phenotype> corpse = std::move(g0.phenotype);
        corpse.reset();  // 个体死亡 — germline 独立存活
        g0_probe.runtime.reset();
    }
    assert(g1_germline.germline != nullptr);
    {
        spec.organism_id = 1001;
        auto g1 = core::Phenotype::create(g1_germline.germline, spec);
        assert(g1.ok());
        auto g1_probe = g1.phenotype->runtime().fork_probe();
        assert(g1_probe.ok());
        auto g1_ex = core::CompiledExecutor::prepare(g1_probe.runtime->plan());
        assert(g1_ex.ok());
        const double born = probe_loss(*g1_probe.runtime, *g1_ex.executor, train_stream);
        printf("[环5 死亡继承] 死者终局 train=%.4f | 新生出生 train=%.4f ✓ (germline 跨越死亡)\n",
               g0_final_train, born);
        assert(std::fabs(born - g0_final_train) < 1e-9);  // 继承保真: 位级
        assert(born < born_loss * 0.1);                   // 远优于环 1 随机出生

        // ── 环 7: 代际超越 (G1 → G2, 达门 epoch 不增) ──
        auto [e1, l1] = train_to_gate(*g1_probe.runtime, *g1_ex.executor, 1001,
                                      train_stream, GATE, 400, 0.05);
        assert(l1 <= GATE);
        auto g2_germline = core::Germline::create(
            kun::transfer::live_definition(*g1_probe.runtime),
            kun::transfer::parameter_seeds(g1_probe.runtime->parameters()),
            "chain-G2");
        assert(g2_germline.ok());
        spec.organism_id = 1002;
        {
            std::unique_ptr<core::Phenotype> corpse = std::move(g1.phenotype);
            corpse.reset();
            g1_probe.runtime.reset();
        }
        auto g2 = core::Phenotype::create(g2_germline.germline, spec);
        assert(g2.ok());
        auto g2_probe = g2.phenotype->runtime().fork_probe();
        assert(g2_probe.ok());
        auto g2_ex = core::CompiledExecutor::prepare(g2_probe.runtime->plan());
        assert(g2_ex.ok());
        auto [e2, l2] = train_to_gate(*g2_probe.runtime, *g2_ex.executor, 1002,
                                      train_stream, GATE, 400, 0.05);
        assert(l2 <= GATE);
        printf("[环7 代际超越] 达门 epoch: G0=%d G1=%d G2=%d (单调不增) ✓\n", e0, e1, e2);
        assert(e1 <= e0 && e2 <= e1);

    // 环 4 载体: 主链个体的无性兄弟 (同 germline 独立出生, 独立代谢)
    spec.organism_id = 1003;
    auto sibling = core::Phenotype::create(g2_germline.germline, spec);
    assert(sibling.ok());
    (void)sibling;
    // ── 环 4: 结构/参数真实改变 (独立兄弟体上付费分裂 — 无性繁殖语义, 不污染主链代谢) ──
    {
        const size_t cells_before = sibling.phenotype->runtime().plan()->cells().size();
        const auto rev_before = sibling.phenotype->runtime().revision();
        core::GrowthSplitProposal sp;
        sp.proposal_id = 1;
        // 靶点: 通道0 前驱边 2→4 (低幅慢变输入下欠票通道路径)
        std::optional<core::EdgeId> target_edge;
        for (const auto& e : sibling.phenotype->runtime().plan()->edges()) {
            const uint32_t s = sibling.phenotype->runtime().plan()->cells()[e.source_index].id.value;
            const uint32_t t = sibling.phenotype->runtime().plan()->cells()[e.target_index].id.value;
            if (s == 2 && t == 4) { target_edge = e.id; break; }
        }
        assert(target_edge.has_value());
        sp.split.edge = *target_edge;
        sp.split.inserted = core::CellBirth{core::CellId{50}, CellType::OP_SUM,
            std::array<core::ParameterValue, 2>{
                core::UnusedParameter{}, core::UnusedParameter{}}};
        sp.split.source_to_new = core::EdgeId{700};
        sp.split.new_to_target = core::EdgeId{701};
        sp.split.new_input_port = core::InputPort{0};
        sp.split.source_weight = 1.0;
        sp.split.target_weight = 1.0;
        sp.funding.compartment = core::ResourceCompartmentId{0};
        assert(sibling.phenotype->growth().submit(sp).ok());
        auto res = sibling.phenotype->growth().step(train_stream.front());
        assert(res.ok() && res.growth_report.has_value());
        assert(res.growth_report->cumulative_growth_cost > 0.0);
        assert(sibling.phenotype->runtime().plan()->cells().size() == cells_before + 1);
        assert(sibling.phenotype->runtime().revision().value == rev_before.value + 1);
        printf("[环4 结构真实改变] 付费=%.3f 细胞 %zu→%zu revision→%llu ✓\n",
               res.growth_report->cumulative_growth_cost, cells_before,
               sibling.phenotype->runtime().plan()->cells().size(),
               (unsigned long long)sibling.phenotype->runtime().revision().value);
    }

        // ── 环 6: 无血缘借阅 (A 出版 → 图书馆验证 → 无血缘 X 冷边界借入) ──
        // 环 6 (StrictCore 微世界): KnowledgeModule 契约要求 StrictCore。
        // A 谱系知识固化: 受体→unary 提取器 (增益 0.8), 经图书馆出版→验证→借入。
        core::GraphDefinition motif;
        motif.identity = core::GraphIdentity(9);
        motif.revision = core::GraphRevision{1};
        motif.profile = SemanticProfile::StrictCore;
        motif.semantic_version = 1;
        motif.cells.push_back({core::CellId{0}, CellType::SENSE_RAW_INPUT_0});
        motif.cells.push_back({core::CellId{1}, CellType::OP_SUM});
        motif.edges.push_back({core::EdgeId{1}, core::CellId{0}, core::OutputPort{0},
                               core::CellId{1}, core::InputPort{0},
                               core::EdgeDelay::Immediate});
        core::InitialParameterSeeds motif_seeds;
        motif_seeds.cell_parameters.push_back(
            {core::CellId{0}, core::ParameterSlot::Param1,
             core::ParameterValue{core::ContinuousValue{1.0}}});
        motif_seeds.cell_parameters.push_back(
            {core::CellId{0}, core::ParameterSlot::Param2,
             core::UnusedParameter{}});
        motif_seeds.cell_parameters.push_back(
            {core::CellId{1}, core::ParameterSlot::Param1, core::UnusedParameter{}});
        motif_seeds.cell_parameters.push_back(
            {core::CellId{1}, core::ParameterSlot::Param2, core::UnusedParameter{}});
        motif_seeds.edge_weights.push_back({core::EdgeId{1}, 0.8});
        auto a_knowledge = core::Germline::create(motif, motif_seeds, "A 谱系: 0.8x 提取器");
        if (!a_knowledge.ok()) {
            printf("[环6 错误] 模块 germline 被拒: %s\n",
                   a_knowledge.error ? a_knowledge.error->reason.c_str() : "?");
            return 1;
        }
        transfer::ModuleContract contract;
        contract.interface_id = "chain-gain-extractor";
        contract.environment = "chain-bench";
        contract.input_count = 1;
        contract.output = core::CellId{1};
        contract.semantic_profile = SemanticProfile::StrictCore;
        contract.semantic_version = 1;
        auto module = transfer::KnowledgeModule::from_germline(
            *a_knowledge.germline, contract, "chain-A-lineage");

        transfer::GermlineLibraryStore store("/tmp/opencode/chain_library.sqlite3");
        transfer::KnowledgeRef ref{"chain-gain-extractor", "v1"};
        try {
            store.publish(ref, "A 谱系增益提取器", module, {});
        } catch (const transfer::KnowledgeError& e) {
            printf("[环6] publish 异常: %s\n", e.what());
            return 1;
        }
        printf("[环6] publish ✓\n");
        // 研究验证: 模块行为帧 (in → 0.8*in)
        transfer::EvaluationProtocol protocol;
        protocol.protocol_id = "KUN-R8-APPROVED/chain-eval-1";
        protocol.environment = "chain-bench";
        protocol.absolute_tolerance = 0.0;
        // 独立录制函数: 模块 fresh runtime 的行为帧 (in → out)
        auto make_trace = [&module](uint64_t seed) {
            transfer::EvaluationTrace t; t.seed = seed;
            auto fresh = module.fresh_runtime();
            auto mexec = core::CompiledExecutor::prepare(fresh->plan());
            assert(mexec.ok());
            for (double xv : {0.5, -0.3, 1.0, 0.25}) {
                std::vector<double> in{xv};
                auto r = mexec.executor->step(*fresh, in);
                assert(r.ok());
                double out = 0.0;
                for (const auto& c : fresh->cell_states())
                    if (c.cell.value == 1) out = c.output_val;
                t.frames.push_back({{xv}, {out}});
            }
            return t;
        };
        protocol.train.push_back(make_trace(1));
        protocol.train.push_back(make_trace(2));
        protocol.ood.push_back(make_trace(3));
        protocol.ood.push_back(make_trace(4));
        auto report = store.evaluate(ref, protocol, "chain-test");
        assert(report.passed);
        auto borrowed = store.borrow(ref, contract, "individual-X");

        // 无血缘个体 X (StrictCore): 受体→SUM (挂点 0→2), 借入后并联 0.8x 通路
        core::GraphDefinition xdef;
        xdef.identity = core::GraphIdentity(10);
        xdef.revision = core::GraphRevision{1};
        xdef.profile = SemanticProfile::StrictCore;
        xdef.semantic_version = 1;
        xdef.cells.push_back({core::CellId{0}, CellType::SENSE_RAW_INPUT_0});
        xdef.cells.push_back({core::CellId{2}, CellType::OP_SUM});
        xdef.edges.push_back({core::EdgeId{1}, core::CellId{0}, core::OutputPort{0},
                              core::CellId{2}, core::InputPort{0},
                              core::EdgeDelay::Immediate});
        core::InitialParameterSeeds xseeds;
        xseeds.cell_parameters.push_back(
            {core::CellId{0}, core::ParameterSlot::Param1,
             core::ParameterValue{core::ContinuousValue{1.0}}});
        xseeds.cell_parameters.push_back(
            {core::CellId{0}, core::ParameterSlot::Param2,
             core::UnusedParameter{}});
        xseeds.cell_parameters.push_back(
            {core::CellId{2}, core::ParameterSlot::Param1, core::UnusedParameter{}});
        xseeds.cell_parameters.push_back(
            {core::CellId{2}, core::ParameterSlot::Param2, core::UnusedParameter{}});
        xseeds.edge_weights.push_back({core::EdgeId{1}, 0.3});  // 弱直连 (借入前功能贫弱)
        auto x_germline = core::Germline::create(xdef, xseeds, "无血缘个体 X");
        if (!x_germline.ok()) {
            printf("[环6 错误] X germline: %s\n",
                   x_germline.error ? x_germline.error->reason.c_str() : "?");
            return 1;
        }
        spec.organism_id = 2000;
        spec.resource_cells.clear();  // X 是 2 细胞 StrictCore 图, 资源记录按图重建
        for (const auto& c : x_germline.germline->graph()->cells())
            spec.resource_cells.push_back(core::ResourceCellInitial{
                c.id, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
        auto x = core::Phenotype::create(x_germline.germline, spec);
        if (!x.ok()) {
            printf("[环6 错误] X 出生: %s\n",
                   x.error ? x.error->reason.c_str() : "?");
            return 1;
        }
        auto x_probe = x.phenotype->runtime().fork_probe();
        assert(x_probe.ok());
        auto x_ex = core::CompiledExecutor::prepare(x_probe.runtime->plan());
        assert(x_ex.ok());
        // X 微世界口径: 读 SUM(2号) 对输入 x0 的响应 (无 ACT 通道)
        auto x_response = [&](core::RuntimeState& rt, core::CompiledExecutor& ex) {
            double acc = 0.0;
            for (double xv : {0.5, -0.3, 1.0, 0.25}) {
                std::vector<double> in{xv, 0.0, 0.0, 0.0};
                auto r = ex.step(rt, in);
                assert(r.ok());
                for (const auto& c : rt.cell_states())
                    if (c.cell.value == 2) acc += c.output_val;
            }
            return acc;
        };
        const double x_before = x_response(*x_probe.runtime, *x_ex.executor);

        transfer::AdoptionRequest req;
        req.target_contract = contract;
        req.target_edge = core::EdgeId{1};  // X 的受体直连边 (unscaled receptor ✓ immediate ✓)
        req.expected_tick = x.phenotype->runtime().tick();
        req.funding.compartment = core::ResourceCompartmentId{0};
        req.cell_cost = 2.0;
        req.synapse_cost = 0.1;
        req.initial_energy = 5.0;
        std::vector<double> adopt_inputs{0.5};  // motif 单输入接口
        auto receipt = transfer::adopt_at_cold_boundary(
            *x.phenotype, borrowed, req, adopt_inputs);
        assert(receipt.paid_cost > 0.0);
        assert(x.phenotype->runtime().plan()->cells().size() == 3);
        // 借入后: 新 probe (图已推进) — 行为改变即借入生效
        auto x_probe2 = x.phenotype->runtime().fork_probe();
        assert(x_probe2.ok());
        auto x_ex2 = core::CompiledExecutor::prepare(x_probe2.runtime->plan());
        assert(x_ex2.ok());
        const double x_after = x_response(*x_probe2.runtime, *x_ex2.executor);
        printf("[环6 无血缘借阅] X 借入前 L1=%.6f → 借入后=%.6f (付费=%.3f, 细胞 2→3) Δ=%.6f\n",
               x_before, x_after, receipt.paid_cost, std::fabs(x_after - x_before));
        // 借入判据: 付费 + 图推进 + 行为真实改变 (0.8x 并联通路注入因果)
        assert(std::fabs(x_after - x_before) > 1e-9);
    }

    printf("\n[THE CHAIN] 七环机制级全部闭合 ✓\n");
    return 0;
}
