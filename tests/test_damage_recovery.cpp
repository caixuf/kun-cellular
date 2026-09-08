// U2.3 损伤恢复三组对照 (新核心机制级):
// C 无损参考 / A 静态受损 (GraphEditor 删除枢纽, 无恢复) / B 生命周期恢复
// (损伤后 growth 付费重建替代细胞 + 因果重接)。度量: 输出序列差异 + 归一化恢复比例。
// 预注册判据: B 的恢复比例 > 0 (机制非空转); A 显著受损。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/cellular_graph_edit.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/growth_controller.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"
#include "kun/cellular/cellular_bptt.hpp"
#include "kun/cellular/legacy/cold_assembly.hpp"

#include <cassert>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace kun;

namespace {

// 默认装配: 一个调用 (冷装配缝) — 底座内化 runtime/ledger/lifecycle/executor/growth
struct Rig {
    std::shared_ptr<core::RuntimeState> runtime;
    std::unique_ptr<core::Phenotype> phenotype;
    core::CellularGrowthController* growth{nullptr};  // 内生生长本能
    std::shared_ptr<core::CompiledExecutor> phenotype_executor;  // 图编辑后重绑
};

Rig assemble(const CellularOrganism& org) {
    kun::migration::ColdAssemblyConfig cfg;
    cfg.organism_id = 7;
    cfg.lifecycle_config = core::LifecycleConfig{5.0, 10.0, 100.0, 0, 0};
    cfg.growth_config = core::GrowthConfig{2.0, 0.1, 5.0, 10.0, 0.0};
    auto assembled = kun::migration::assemble_phenotype(
        org, core::GraphIdentity(1), core::GraphRevision(1), cfg);
    if (!assembled.ok()) {
        printf("[装配失败] %s\n", assembled.diagnostic.c_str());
        assert(false && "cold assembly failed");
    }
    Rig rig;
    rig.phenotype = std::move(assembled.phenotype);
    rig.runtime = std::shared_ptr<core::RuntimeState>(
        &rig.phenotype->runtime(), [](core::RuntimeState*) {});
    rig.growth = &rig.phenotype->growth();
    return rig;
}

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

CellularOrganism build_decision_organism() {
    CellularOrganism org;
    org.organism_id = 7;
    org.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0));
    org.cells.push_back(make_cell(1, CellType::SENSE_RAW_INPUT_1));
    org.cells.push_back(make_cell(2, CellType::OP_EMA, 0.3, 0.0));
    org.cells.push_back(make_cell(3, CellType::OP_SUM));
    org.cells.push_back(make_cell(4, CellType::OP_MULTIPLY));
    org.cells.push_back(make_cell(5, CellType::GATE_MIN_MAX, 1.0, 0.0));
    for (uint32_t ch = 0; ch < 4; ++ch)
        org.cells.push_back(make_cell(6 + ch, CellType::ACT_CHANNEL, 1.0, static_cast<double>(ch)));
    org.synapses.push_back(make_synapse(0, 2, 0, 1.0));
    org.synapses.push_back(make_synapse(1, 2, 0, 0.8));
    org.synapses.push_back(make_synapse(2, 3, 0, 1.0));
    org.synapses.push_back(make_synapse(0, 4, 0, 1.0));
    org.synapses.push_back(make_synapse(2, 4, 1, 1.0));
    org.synapses.push_back(make_synapse(4, 5, 0, 1.0));
    org.synapses.push_back(make_synapse(3, 5, 1, 1.0));
    for (uint32_t ch = 0; ch < 4; ++ch)
        org.synapses.push_back(make_synapse(5, 6 + ch, 0, 1.0 - 0.2 * ch));
    org.compile();
    return org;
}

// 回放输入流, 记录 4 通道 ACT 输出序列
std::vector<std::array<double, 4>> replay(
    Rig& rig, const std::vector<std::vector<double>>& stream, bool via_growth_step) {
    std::vector<std::array<double, 4>> trace;
    for (const auto& input : stream) {
        if (via_growth_step) {
            auto res = rig.growth->step(input);
            assert(res.ok());
        } else {
            core::CompiledExecutor* ex = rig.phenotype_executor
                ? rig.phenotype_executor.get() : &rig.phenotype->executor();
            auto res = ex->step(*rig.runtime, input);
            assert(res.ok());
        }
        std::array<double, 4> act{};
        size_t slot = 0;
        for (const auto& c : rig.runtime->cell_states()) {
            if (c.type == CellType::ACT_CHANNEL && slot < 4) act[slot++] = c.output_val;
        }
        trace.push_back(act);
    }
    return trace;
}

double trace_distance(const std::vector<std::array<double, 4>>& a,
                      const std::vector<std::array<double, 4>>& b) {
    double acc = 0.0;
    for (size_t t = 0; t < a.size(); ++t)
        for (size_t k = 0; k < 4; ++k)
            acc += std::fabs(a[t][k] - b[t][k]);
    return acc;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::vector<double>> stream;
    for (int t = 0; t < 30; ++t) {
        std::vector<double> v(4);
        for (auto& x : v) x = 0.2 + 0.6 * uni(rng);
        stream.push_back(v);
    }

    // ── C 无损参考 ──
    auto rig_c = assemble(build_decision_organism());
    auto trace_c = replay(rig_c, stream, false);
    printf("[C 无损] 回放 %zu 拍完成\n", trace_c.size());

    // ── A 静态受损: fork_probe 上 GraphEditor 删除枢纽 GATE_MIN_MAX (cell 5) ──
    std::vector<std::array<double, 4>> trace_a;
    {
        auto probe = rig_c.runtime->fork_probe();
        assert(probe.ok());
        core::GraphEditor editor(*probe.runtime);
        std::vector<core::GraphEditEvent> events;
        events.push_back(core::GraphEditEvent{
            1, core::GraphEditAction{core::RemoveCellAction{core::CellId{5}}}});
        auto edited = editor.apply(*probe.runtime, events);
        if (!edited.ok()) {
            printf("[A 错误] 删除被拒: %s\n",
                   edited.error ? edited.error->reason.c_str() : "?");
            return 1;
        }
        printf("[A 受损] 枢纽删除: 细胞 %zu→%zu, revision %llu→%llu\n",
               rig_c.runtime->plan()->cells().size(), edited.graph->cells().size(),
               (unsigned long long)edited.report.old_revision.value,
               (unsigned long long)edited.report.new_revision.value);
        // probe 重新绑定编译图后回放
        auto compiled = core::CompiledExecutor::prepare(edited.graph);
        assert(compiled.ok());
        std::vector<std::array<double, 4>> local;
        for (const auto& input : stream) {
            auto res = compiled.executor->step(*probe.runtime, input);
            if (!res.ok()) {
                printf("[A] 损伤后执行失败: %s\n", res.error->reason.data());
                return 1;
            }
            std::array<double, 4> act{};
            size_t slot = 0;
            for (const auto& c : probe.runtime->cell_states()) {
                if (c.type == CellType::ACT_CHANNEL && slot < 4) act[slot++] = c.output_val;
            }
            local.push_back(act);
        }
        trace_a = local;
    }
    const double dist_a = trace_distance(trace_a, trace_c);
    printf("[A 受损] 与无损差异累计 = %.4f\n", dist_a);
    assert(dist_a > 0.5);  // 损伤必须显著

    // ── B 生命周期恢复: 损伤同款 + growth 付费重建替代细胞 ──
    std::vector<std::array<double, 4>> trace_b;
    double recovery = 0.0;
    {
        auto rig_b = assemble(build_decision_organism());
        // 损伤 (GraphEditor on fork? growth 走 GraphEditor 内部路径 — 用 growth 的
        // 冷边界: 直接在主 runtime 上编辑, 因为 growth 需要主 runtime 生命力)
        // 机制: growth 的 graph edit 需要 lifecycle 冷边界; 生命周期恢复 = 
        // 损伤 → 诞生替代 (付费) → 重接因果 → 继续代谢
        // B 组: 诞生替代枢纽 (OP_SUM 吸收上游 2/3/4, 驱动全部 4 通道) — 付费重建
        core::GrowthCellBirthProposal bp;
        bp.proposal_id = 1;
        bp.birth = core::CellBirth{
            core::CellId{50}, CellType::OP_SUM,
            std::array<core::ParameterValue, 2>{
                core::UnusedParameter{}, core::UnusedParameter{}}};
        const std::array<uint32_t, 3> upstream{2, 3, 4};
        for (size_t u = 0; u < upstream.size(); ++u)
            bp.incoming_edges.push_back(core::EdgeBirth{
                core::EdgeId{600 + u}, core::CellId{upstream[u]},
                core::OutputPort{0}, core::CellId{50}, core::InputPort{0},
                core::EdgeDelay::Immediate, 0.8});
        for (size_t ch = 0; ch < 4; ++ch)
            bp.outgoing_edges.push_back(core::EdgeBirth{
                core::EdgeId{610 + ch}, core::CellId{50},
                core::OutputPort{0}, core::CellId{6 + static_cast<uint32_t>(ch)},
                core::InputPort{0}, core::EdgeDelay::Immediate,
                1.0 - 0.2 * ch});
        bp.funding.compartment = core::ResourceCompartmentId{0};
        auto submit = rig_b.growth->submit(bp);
        assert(submit.ok());
        auto res = rig_b.growth->step(stream.front());
        assert(res.ok());
        assert(rig_b.runtime->plan()->cells().size() == 11);
        // 旁路重建后: 通过 growth 删除枢纽 (5) — lifecycle 冷边界允许 remove?
        // growth variant 无 Remove 提案 → 用 GraphEditor on main runtime
        core::GraphEditor editor(*rig_b.runtime);
        std::vector<core::GraphEditEvent> evs;
        evs.push_back(core::GraphEditEvent{
            1, core::GraphEditAction{core::RemoveCellAction{core::CellId{5}}}});
        auto edited = editor.apply(*rig_b.runtime, evs);
        assert(edited.ok());
        auto compiled_b = core::CompiledExecutor::prepare(edited.graph);
        assert(compiled_b.ok());
        rig_b.phenotype_executor = std::move(compiled_b.executor);

        // 结构恢复基线 (无学习)
        double dist_b_struct = trace_distance(replay(rig_b, stream, false), trace_c);

        // ── 生长中学习: CoreCellularBPTTEngine, 教师 = C 组无损轨迹 ──
        // 学习窗口纪律: 只准动重建边 (600+u, 610+ch) 的权重参数
        std::vector<core::ParameterBinding> allowed;
        for (const auto& ed : edited.graph->edges()) {
            const uint64_t eid = ed.id.value;
            if ((eid >= 600 && eid < 603) || (eid >= 610 && eid < 614)) {
                core::ParameterBinding b;
                b.kind = core::ParameterBindingKind::EdgeWeight;
                b.index = ed.weight_parameter_index;
                b.edge = ed.id;
                allowed.push_back(b);
            }
        }
        assert(allowed.size() == 7);
        auto window = core::LearningWindow::open(7, *rig_b.runtime, allowed);
        CoreCellularBPTTEngine engine(32);
        engine.init_optimizer(*rig_b.runtime);
        // 教师轨迹: C 组无损重放, 行 = effector 通道目标 (CoreBPTT targets 语义)
        std::vector<std::vector<double>> teacher;
        {
            auto probe_t = rig_c.runtime->fork_probe();
            assert(probe_t.ok());
            for (const auto& input : stream) {
                auto r = rig_c.phenotype->executor().step(*probe_t.runtime, input);
                assert(r.ok());
                std::vector<double> row(4, 0.0);
                size_t slot = 0;
                for (const auto& c : probe_t.runtime->cell_states()) {
                    if (c.type == CellType::ACT_CHANNEL && slot < 4)
                        row[slot++] = c.output_val;  // ACT 序即通道槽 (param2 语义)
                }
                teacher.push_back(row);
            }
        }
        const double lr = 0.02;
        double last_loss = -1.0;
        for (int epoch = 0; epoch < 60; ++epoch) {
            auto reset = rig_b.runtime->reset_episode();
            assert(!reset.error.has_value());
            engine.reset_tape();
            for (size_t t = 0; t < stream.size(); ++t) {
                auto rec = engine.record_step(*rig_b.runtime, *rig_b.phenotype_executor, stream[t]);
                assert(rec.ok());
            }
            CoreBPTTGradients grads;
            auto back = engine.backward(*rig_b.runtime, teacher, grads, &window);
            if (!back.ok()) {
                printf("[B 错误] backward: %s\n", back.error->reason.c_str());
                return 1;
            }
            auto upd = engine.step_adam(*rig_b.runtime, window, grads, lr);
            if (!upd.ok()) {
                printf("[B 错误] step_adam: %s\n", upd.error->reason.c_str());
                return 1;
            }
            last_loss = grads.loss;
            if (epoch % 10 == 0 || epoch == 59)
                printf("  [学习] epoch %d loss=%.6f 更新参数=%zu\n",
                       epoch, grads.loss, upd.updated_values);
        }
        // 训练后回放
        auto reset2 = rig_b.runtime->reset_episode();
        assert(!reset2.error.has_value());
        { /* 编辑后回放使用编辑后执行器 */ } trace_b = replay(rig_b, stream, false);
        const double dist_b = trace_distance(trace_b, trace_c);
        recovery = (dist_a > 0.0) ? 1.0 - dist_b / dist_a : 0.0;
        // 位序纪律: BPTT 只是参数代谢工具, 贡献必须分解呈现
        const double recovery_struct = (dist_a > 0.0) ? 1.0 - dist_b_struct / dist_a : 0.0;
        printf("[B 恢复分解] 结构/代谢重建: %.3f | BPTT 参数代谢增量: %.3f | 合计: %.3f\n",
               recovery_struct, recovery - recovery_struct, recovery);
        printf("[B 恢复] 结构基线差异=%.4f → 学习后差异=%.4f (loss %.4f)\n",
               dist_b_struct, dist_b, last_loss);
    }
    assert(recovery > 0.0);  // 预注册: 恢复机制非空转

    printf("[U2.3 损伤恢复机制] 全部通过 (恢复比例 %.3f)\n", recovery);
    return 0;
}
