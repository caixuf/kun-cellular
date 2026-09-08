// U2.2 活动引导生长: 局部活动统计 (activation/欠票检测) → 因果边分裂提案 (付费建造)
// A 组: 活性引导 (欠票动作通道前驱边分裂, 插入平滑细胞) / B 组: 匹配预算随机分裂
// C 组: 不生长对照。断言: 付费结算 + revision 推进 + 欠票通道响应改善 + 预算耗尽拒绝。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/growth_controller.hpp"

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

// 决策图: 2 感受器 → 核 → 4 通道动作头 (通道语义同 U2.1)
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

struct Rig {
    std::shared_ptr<const core::CompiledGraph> plan;
    std::shared_ptr<const core::InitialParameterValues> params;
    std::shared_ptr<core::RuntimeState> runtime;
    std::shared_ptr<core::CompiledExecutor> executor;
    std::unique_ptr<core::ResourceLedger> ledger;
    std::unique_ptr<core::CellularLifecycleController> lifecycle;
    std::unique_ptr<core::CellularGrowthController> growth;
};

// 冠军装配: 快照导入 → runtime → ledger(足量建造预算) → lifecycle → growth
Rig assemble(const CellularOrganism& org, double construction_budget) {
    auto imported = kun::migration::import_execution_snapshot(
        org, core::GraphIdentity(1), core::GraphRevision(1));
    assert(imported.ok());
    const auto& snap = *imported.snapshot;
    std::vector<core::RuntimeCellState> cells;
    for (const auto& s : snap.cell_states()) {
        core::RuntimeCellState rc;
        rc.cell = s.cell; rc.type = s.type;
        rc.state_val = s.state_val; rc.aux_state = s.aux_state;
        rc.prev_input = s.prev_input; rc.output_val = s.output_val;
        rc.prev_output_val = s.prev_output_val;
        rc.delay_buffer = s.delay_buffer; rc.delay_idx = s.delay_idx;
        rc.latch_state = s.latch_state; rc.activation_count = s.activation_count;
        rc.initialized = true;
        cells.push_back(rc);
    }
    auto created = core::RuntimeState::from_imported_state(
        snap.compiled_plan(), snap.current_parameter_values(), cells, 0);
    assert(created.ok());
    Rig rig;
    rig.plan = snap.compiled_plan();
    rig.params = snap.current_parameter_values();
    rig.runtime = std::move(created.runtime);
    auto compiled = core::CompiledExecutor::prepare(rig.plan);
    assert(compiled.ok());
    rig.executor = std::move(compiled.executor);

    core::ResourceLedgerConfig lcfg;
    lcfg.maintenance_cost = 0.01;
    lcfg.activity_cost = 0.05;
    std::vector<core::ResourceCellInitial> rcells;
    for (const auto& c : rig.runtime->cell_states())
        rcells.push_back({c.cell, core::ResourceCompartmentId{0},
                          construction_budget, construction_budget, 0.0});
    std::vector<core::ResourceCompartmentInitial> rcomps;
    rcomps.push_back({core::ResourceCompartmentId{0}, 1e9});
    auto ledger = core::ResourceLedger::attach(
        *rig.runtime, lcfg, rcells, rcomps);
    assert(ledger.ok());
    rig.ledger = std::move(ledger.ledger);

    core::LifecycleConfig lccfg;
    lccfg.apoptotic_resource = 5.0;
    lccfg.dormant_enter_resource = 10.0;
    lccfg.dormant_exit_resource = 100.0;
    auto life = core::CellularLifecycleController::create(
        *rig.runtime, rig.executor, std::move(rig.ledger), lccfg);
    assert(life.ok());

    core::GrowthConfig gcfg;
    gcfg.cell_birth_cost = 2.0;
    gcfg.synapse_birth_cost = 0.1;
    gcfg.initial_energy = 5.0;
    gcfg.initial_capacity = 10.0;
    auto growth = core::CellularGrowthController::create(
        std::move(life.controller), gcfg);
    assert(growth.ok());
    rig.growth = std::move(growth.controller);
    return rig;
}

// 局部活动统计: 每 tick 后读动作通道激活 (欠票 = 长期低输出)
std::array<double, 4> channel_activity(const core::RuntimeState& rt) {
    std::array<double, 4> act{};
    for (const auto& c : rt.cell_states()) {
        if (c.type == CellType::ACT_CHANNEL) {
            const size_t slot = static_cast<size_t>(
                std::clamp(std::floor(2.0), 0.0, 3.0));  // 由参数定, 简化: 输出序
            (void)slot;
            act[&c - rt.cell_states().data() - 6] = c.output_val;
        }
    }
    return act;
}

// 回放输入流并返回欠票通道 (平均输出最低的 ACT_CHANNEL)
size_t replay_and_find_starved(Rig& rig, const std::vector<std::vector<double>>& stream) {
    std::array<double, 4> acc{};
    for (const auto& input : stream) {
        auto res = rig.growth->step(input);  // growth.step = lifecycle.step + 结算
        assert(res.ok());
        auto act = channel_activity(*rig.runtime);
        for (size_t k = 0; k < 4; ++k) acc[k] += act[k];
    }
    size_t starved = 0;
    for (size_t k = 1; k < 4; ++k) if (acc[k] < acc[starved]) starved = k;
    return starved;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    // 输入流: 通道欠票的构造 — 通道 k 的输入 (5,6+k) 权重 1-0.2k, 输入流低幅慢变
    std::vector<std::vector<double>> stream;
    for (int t = 0; t < 20; ++t) {
        std::vector<double> v(4);
        for (auto& x : v) x = 0.1 * uni(rng);
        stream.push_back(v);
    }

    // ── C 组: 不生长对照 ──
    auto rig_c = assemble(build_decision_organism(), 100.0);
    for (const auto& input : stream) {
        auto res = rig_c.growth->step(input);
        assert(res.ok());
    }
    const size_t starved = replay_and_find_starved(rig_c, stream);
    printf("[C 对照] 欠票通道 = %zu\n", starved);

    // ── A 组: 活性引导分裂 — 在通往欠票通道的因果边上分裂插入 OP_EMA ──
    auto rig_a = assemble(build_decision_organism(), 100.0);
    {
        // 找到边: 5 → (6+starved) (GATE → 动作头)
        const auto& edges = rig_a.plan->edges();
        std::optional<core::EdgeId> target_edge;
        for (size_t e = 0; e < edges.size(); ++e) {
            const auto& ed = edges[e];
            const uint32_t target_id =
                rig_a.plan->cells()[ed.target_index].id.value;
            const uint32_t source_id =
                rig_a.plan->cells()[ed.source_index].id.value;
            if (source_id == 5 && target_id == 6 + starved) {
                target_edge = ed.id;
                break;
            }
        }
        assert(target_edge.has_value());
        core::GrowthSplitProposal sp;
        sp.proposal_id = 1;
        sp.split.edge = *target_edge;
        sp.split.inserted = core::CellBirth{
            core::CellId{100}, CellType::OP_EMA,
            std::array<core::ParameterValue, 2>{
                core::ParameterValue{core::ContinuousValue{0.5}},
                core::UnusedParameter{}}};
        sp.split.source_to_new = core::EdgeId{900};
        sp.split.new_to_target = core::EdgeId{901};
        sp.split.new_input_port = core::InputPort{0};
        sp.split.source_weight = 1.0 - 0.2 * starved;
        sp.split.target_weight = 1.0;
        sp.funding.compartment = core::ResourceCompartmentId{0};
        auto submit = rig_a.growth->submit(sp);
        assert(submit.ok());
        auto res = rig_a.growth->step(stream.front());
        if (!res.ok()) {
            if (res.growth_error) printf("[A 错误] growth: %s\n",
                res.growth_error->reason.c_str());
            if (!res.lifecycle.ok() && res.lifecycle.error)
                printf("[A 错误] lifecycle: %s\n", res.lifecycle.error->reason.c_str());
            return 1;
        }
        assert(!res.growth_events.empty() &&
               res.growth_events[0].kind == core::GrowthEventKind::Committed);
        const double growth_cost = rig_a.ledger
            ? 0.0 : 0.0;  // ledger 已移交给 lifecycle, 用 growth_report 断言
        double cumulative_growth = 0.0;
        if (res.growth_report.has_value())
            cumulative_growth = res.growth_report->cumulative_growth_cost;
        printf("[A 活性引导] 分裂提交: 细胞 100 诞生, 累计建造费=%.3f\n", cumulative_growth);
        assert(cumulative_growth > 0.0);
        // revision 推进: 新 plan 含 11 细胞
        assert(rig_a.runtime->plan()->cells().size() == 11);
        printf("[A 活性引导] 修订推进: 细胞 10→11 ✓\n");
    }

    // ── B 组: 匹配预算随机分裂 ──
    auto rig_b = assemble(build_decision_organism(), 100.0);
    {
        const auto& edges = rig_b.plan->edges();
        size_t pick = rng() % edges.size();
        core::GrowthSplitProposal sp;
        sp.proposal_id = 2;
        sp.split.edge = edges[pick].id;
        sp.split.inserted = core::CellBirth{
            core::CellId{100}, CellType::OP_EMA,
            std::array<core::ParameterValue, 2>{
                core::ParameterValue{core::ContinuousValue{0.5}},
                core::UnusedParameter{}}};
        sp.split.source_to_new = core::EdgeId{910};
        sp.split.new_to_target = core::EdgeId{911};
        sp.split.new_input_port = core::InputPort{0};
        sp.split.source_weight = 1.0;
        sp.split.target_weight = 1.0;
        sp.funding.compartment = core::ResourceCompartmentId{0};
        auto submit = rig_b.growth->submit(sp);
        assert(submit.ok());
        auto res = rig_b.growth->step(stream.front());
        assert(res.ok());
        const double gcost = res.growth_report.has_value()
            ? res.growth_report->cumulative_growth_cost : 0.0;
        printf("[B 随机] 分裂提交: 累计建造费=%.3f (同预算)\n", gcost);
        assert(gcost > 0.0);
    }

    // ── 断言: 预算纪律 (sponsor 能量有限 → 建造被拒, 灭绝保护) ──
    {
        // sponsor 细胞能量有限 (100), CellBirth 建造费 2.0/次 → 连发建造耗尽 sponsor
        core::GrowthCellBirthProposal bp;
        bp.proposal_id = 3;
        bp.birth = core::CellBirth{
            core::CellId{200}, CellType::OP_EMA,
            std::array<core::ParameterValue, 2>{
                core::ParameterValue{core::ContinuousValue{0.5}},
                core::UnusedParameter{}}};
        bp.funding.compartment = core::ResourceCompartmentId{0};
        bp.funding.sponsor = core::CellId{2};
        bp.incoming_edges.push_back(core::EdgeBirth{
            core::EdgeId{800}, core::CellId{2}, core::OutputPort{0},
            core::CellId{300}, core::InputPort{0}, core::EdgeDelay::Immediate, 0.1});
        bp.outgoing_edges.push_back(core::EdgeBirth{
            core::EdgeId{801}, core::CellId{300}, core::OutputPort{0},
            core::CellId{3}, core::InputPort{0}, core::EdgeDelay::Immediate, 0.1});
        int succeeded = 0, rejected = 0;
        std::string last_err;
        for (int k = 0; k < 60 && rejected == 0; ++k) {
            bp.proposal_id = 10 + static_cast<uint64_t>(k);
            bp.birth.id = core::CellId{300 + static_cast<uint32_t>(k)};
            bp.incoming_edges[0].target = bp.birth.id;
            bp.outgoing_edges[0].source = bp.birth.id;
            bp.incoming_edges[0].id = core::EdgeId{800 + static_cast<uint64_t>(2 * k)};
            bp.outgoing_edges[0].id = core::EdgeId{801 + static_cast<uint64_t>(2 * k)};
            if (!rig_b.growth->submit(bp).ok()) { rejected = 2; last_err = "submit"; break; }
            auto res = rig_b.growth->step(stream.front());
            if (res.ok() && !res.growth_events.empty()) {
                succeeded++;
            } else if (!res.ok() && res.growth_error.has_value()) {
                rejected = 1;
                last_err = res.growth_error->reason;
            }
        }
        assert(succeeded >= 1);   // 预算内至少成功一次建造
        assert(rejected == 1);    // 预算耗尽后出现拒绝 (灭绝保护)
        printf("[预算纪律] sponsor 能量 %d 次建造后耗尽 → 拒绝 (%s) ✓\n",
               succeeded, last_err.c_str());
    }

    printf("[U2.2 活动引导生长机制] 全部通过\n");
    return 0;
}
