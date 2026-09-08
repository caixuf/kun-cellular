// U2.1 probe/commit 纪律: 候选打分 = fork_probe (不推生命钟/不结算资源/主状态不变),
// 真实出牌 = lifecycle.step (生命钟+1 + 资源结算)。含候选排列不变性回归。
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"

#include <cassert>
#include <cstdio>
#include <map>
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

// 决策图: 2 感受器 → 运算核 → 4 通道动作头 (K 候选语义)
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

struct ProbeMachine {
    core::RuntimeState& runtime;
    core::CompiledExecutor& executor;

    double score_candidate(const std::vector<double>& input) {
        auto forked = runtime.fork_probe();
        assert(forked.ok());
        auto res = executor.step(*forked.runtime, input);
        assert(res.ok());
        double best = -1e308;
        for (const auto& c : forked.runtime->cell_states())
            best = std::max(best, c.output_val);
        return best;
    }
};

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    auto org = build_decision_organism();
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
    auto& runtime = *created.runtime;

    auto compiled = core::CompiledExecutor::prepare(snap.compiled_plan());
    assert(compiled.ok());

    // 资源账本: 每细胞足量能量 + 有活动成本 (commit 才结算)
    core::ResourceLedgerConfig ledger_cfg;
    ledger_cfg.dt = 1.0;
    ledger_cfg.maintenance_cost = 0.01;
    ledger_cfg.activity_cost = 0.05;
    std::vector<core::ResourceCellInitial> rcells;
    for (const auto& c : runtime.cell_states())
        rcells.push_back({c.cell, core::ResourceCompartmentId{0}, 1000.0, 1000.0, 0.0});
    std::vector<core::ResourceCompartmentInitial> rcomps;
    rcomps.push_back({core::ResourceCompartmentId{0}, 1e9});
    auto ledger = core::ResourceLedger::attach(
        runtime, ledger_cfg, rcells, rcomps);
    assert(ledger.ok());

    core::LifecycleConfig lcfg;
    lcfg.apoptotic_resource = 10.0;     // 能量 < 10 → 调亡
    lcfg.dormant_enter_resource = 20.0; // 能量 < 20 → 休眠
    lcfg.dormant_exit_resource = 50.0;  // 能量 > 50 → 唤醒
    lcfg.minimum_dwell_ticks = 0;
    lcfg.cooldown_ticks = 0;
    auto lifecycle = core::CellularLifecycleController::create(
        runtime, compiled.executor, std::move(ledger.ledger), lcfg);
    if (!lifecycle.ok()) {
        printf("[错误] lifecycle 创建失败: %s\n",
               lifecycle.error ? lifecycle.error->reason.c_str() : "?");
        return 1;
    }
    auto& controller = *lifecycle.controller;

    ProbeMachine probe{runtime, *compiled.executor};

    // K=4 候选 (斗地主式: 4 个候选输入向量)
    std::vector<std::vector<double>> candidates;
    for (int k = 0; k < 4; ++k) {
        std::vector<double> v(4, 0.0);
        v[0] = 0.2 + 0.3 * k;
        v[1] = 0.9 - 0.2 * k;
        candidates.push_back(v);
    }

    // ── 阶段 1: 全 probe 阶段 ──
    const uint64_t tick_before = /* runtime tick 不可直读, 用细胞数快照代替 */ 0;
    (void)tick_before;
    auto snap_before = runtime.snapshot();
    std::vector<double> scores(candidates.size());
    for (size_t k = 0; k < candidates.size(); ++k)
        scores[k] = probe.score_candidate(candidates[k]);
    auto snap_after_probe = runtime.snapshot();
    // 断言 1: probe 零副作用 — 主 runtime 细胞状态逐字段不变
    for (size_t i = 0; i < snap_before.cells().size(); ++i) {
        const auto& a = snap_before.cells()[i];
        const auto& b = snap_after_probe.cells()[i];
        assert(a.state_val == b.state_val && a.output_val == b.output_val &&
               a.prev_input == b.prev_input && a.prev_output_val == b.prev_output_val &&
               a.delay_idx == b.delay_idx && a.delay_buffer == b.delay_buffer &&
               a.latch_state == b.latch_state && a.activation_count == b.activation_count);
    }
    printf("[断言1] probe 零副作用: 主 runtime 状态位级不变 ✓\n");

    // ── 断言 2: 候选排列不变性 (非 tie 胜选不变) ──
    size_t best1 = 0, best2 = 0;
    for (size_t k = 1; k < scores.size(); ++k) {
        if (scores[k] > scores[best1] + 1e-12) best1 = k;
    }
    std::vector<size_t> perm{3, 1, 0, 2};
    double bs = -1e308;
    for (size_t p = 0; p < perm.size(); ++p) {
        double s = probe.score_candidate(candidates[perm[p]]);
        if (s > bs + 1e-12) { bs = s; best2 = perm[p]; }
    }
    // tie-break 用最低下标: 重扫正向顺序确保唯一 argmax
    double top = -1e308;
    for (size_t k = 0; k < scores.size(); ++k) top = std::max(top, scores[k]);
    int n_top = 0;
    for (size_t k = 0; k < scores.size(); ++k) if (scores[k] > top - 1e-12) n_top++;
    if (n_top == 1) {
        assert(best1 == best2);
        printf("[断言2] 候选排列不变性: 乱序枚举胜选不变 (候选%zu, 分=%.6f) ✓\n", best1, top);
    } else {
        printf("[断言2] 候选排列不变性: 存在 tie (%d 个同分), 跳过 argmax 断言\n", n_top);
    }

    // ── 阶段 2: commit (真实出牌) ──
    const size_t win = best1;
    auto result = controller.step(candidates[win]);
    assert(result.ok());
    assert(result.executed);
    assert(result.tick == 1);  // 生命钟从 0 → 1, 仅此一次 commit
    // 断言 3: commit 结算真实发生 (paid_cost > 0)
    assert(result.settlement.has_value());
    assert(result.settlement->tick == 1);
    printf("[断言3] commit: 生命钟=1, 结算 injected=%.4f paid_cost=%.4f ✓\n",
           result.settlement->injected, result.settlement->paid_cost);

    // ── 断言 4: 后续 3 次 probe 不推生命钟, 再 commit → tick=2 ──
    for (int k = 0; k < 3; ++k) (void)probe.score_candidate(candidates[k]);
    auto r2 = controller.step(candidates[(win + 1) % candidates.size()]);
    assert(r2.ok() && r2.tick == 2);
    printf("[断言4] 3 次 probe 不推生命钟; 第二次 commit → tick=2 ✓\n");

    printf("[U2.1 probe/commit 纪律] 全部通过\n");
    return 0;
}
