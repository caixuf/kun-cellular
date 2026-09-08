#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/resource_ledger.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

using namespace kun;
using namespace kun::core;

namespace {

InitialParameterSeeds seeds_for(const GraphDefinition& graph) {
    InitialParameterSeeds seeds;
    for (const auto& cell : graph.cells) {
        const auto contract = contract_for(cell.type);
        assert(contract.has_value());
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            ParameterValue value = UnusedParameter{};
            switch (descriptor.value_type) {
                case ParameterValueType::Continuous:
                    value = ContinuousValue{
                        descriptor.name && std::string_view(descriptor.name) == "alpha"
                            ? 0.5
                            : 1.0};
                    break;
                case ParameterValueType::ChannelIndex:
                    value = ChannelIndex{0};
                    break;
                case ParameterValueType::DelayTicks:
                    value = DelayTicks{1};
                    break;
                case ParameterValueType::MinMaxMode:
                    value = MinMaxMode::Min;
                    break;
                case ParameterValueType::Unused:
                    break;
            }
            seeds.cell_parameters.push_back(
                CellParameterSeed{cell.id, static_cast<ParameterSlot>(slot), value});
        }
    }
    for (const auto& edge : graph.edges) {
        seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, 1.0});
    }
    return seeds;
}

struct Fixture {
    CompileResult compiled;
    std::shared_ptr<RuntimeState> runtime;
};

Fixture make_fixture(
    std::vector<CellDefinition> cells,
    SemanticProfile profile = SemanticProfile::StrictCore) {
    GraphDefinition graph;
    graph.identity = GraphIdentity{9001};
    graph.revision = GraphRevision{1};
    graph.profile = profile;
    graph.cells = std::move(cells);
    Fixture result;
    result.compiled = GraphCompiler{}.compile(graph, seeds_for(graph));
    assert(result.compiled.ok());
    const auto runtime =
        RuntimeState::create(result.compiled.graph, result.compiled.initial_values);
    assert(runtime.ok());
    result.runtime = std::move(runtime.runtime);
    return result;
}

ResourceLedgerConfig basic_config() {
    ResourceLedgerConfig config;
    config.dt = 1.0;
    config.maintenance_cost = 0.0;
    config.absorption_rate = 0.0;
    config.transmission_scale = 1.0;
    config.activity_scale = 1.0;
    config.activity_cost = 0.0;
    return config;
}

void test_real_execution_cost_is_paid_once_per_native_step() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto config = basic_config();
    config.execution_costs.push_back({CellType::OP_ABS, 1.0});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 10.0, 10.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);

    ReferenceExecutor executor;
    for (uint64_t tick = 1; tick <= 3; ++tick) {
        const auto native = executor.step(*fixture.runtime, {});
        assert(native.ok());
        const auto runtime_before_settlement = fixture.runtime->snapshot();
        const auto settled = ledger->settle(*fixture.runtime, native.measurement);
        assert(settled.ok());
        assert(settled.report.tick == tick);
        const auto runtime_after_settlement = fixture.runtime->snapshot();
        assert(runtime_after_settlement.tick() == runtime_before_settlement.tick());
        assert(std::equal(
            runtime_after_settlement.cells().begin(),
            runtime_after_settlement.cells().end(),
            runtime_before_settlement.cells().begin(),
            runtime_before_settlement.cells().end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cell == rhs.cell &&
                       lhs.type == rhs.type &&
                       lhs.state_val == rhs.state_val &&
                       lhs.aux_state == rhs.aux_state &&
                       lhs.prev_input == rhs.prev_input &&
                       lhs.output_val == rhs.output_val &&
                       lhs.prev_output_val == rhs.prev_output_val &&
                       lhs.delay_buffer == rhs.delay_buffer &&
                       lhs.delay_idx == rhs.delay_idx &&
                       lhs.latch_state == rhs.latch_state &&
                       lhs.activation_count == rhs.activation_count &&
                       lhs.initialized == rhs.initialized;
            }));
    }

    const auto snapshot = ledger->snapshot();
    assert(std::abs(snapshot.cell(CellId{11})->energy - 7.0) < 1e-12);
    assert(std::abs(snapshot.cumulative_paid_cost() - 3.0) < 1e-12);
    assert(std::abs(snapshot.cumulative_dissipation() - 3.0) < 1e-12);
    assert(snapshot.cell(CellId{11})->exhausted == false);
}

void test_activity_cost_is_optional_but_activity_is_recorded() {
    auto fixture = make_fixture({{CellId{11}, CellType::SENSE_RAW_INPUT_0}});
    auto config = basic_config();
    config.execution_costs.push_back({CellType::OP_ABS, 0.0});
    config.activity_cost = 0.0;
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 3.0, 3.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const double input = 0.75;
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {&input, 1});
    assert(native.ok());
    assert(ledger->settle(*fixture.runtime, native.measurement).ok());
    const auto snapshot = ledger->snapshot();
    assert(snapshot.cell(CellId{11})->cumulative_activity > 0.0);
    assert(snapshot.cumulative_paid_cost() == 0.0);

    auto charged_fixture = make_fixture({{CellId{11}, CellType::SENSE_RAW_INPUT_0}});
    auto charged_config = basic_config();
    charged_config.activity_cost = 2.0;
    auto charged = ResourceLedger::attach(
        *charged_fixture.runtime,
        charged_config,
        {{CellId{11}, ResourceCompartmentId{7}, 3.0, 3.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(charged.ok());
    auto charged_ledger = std::move(charged.ledger);
    const double charged_input = 0.75;
    const auto charged_native = ReferenceExecutor{}.step(
        *charged_fixture.runtime, {&charged_input, 1});
    assert(charged_native.ok());
    assert(charged_ledger->settle(
        *charged_fixture.runtime, charged_native.measurement).ok());
    assert(std::abs(
               charged_ledger->snapshot().cell(CellId{11})->cumulative_activity_cost -
               1.5) < 1e-12);
}

void test_compartment_demand_is_proportional_and_order_independent() {
    auto fixture = make_fixture({
        {CellId{101}, CellType::OP_ABS},
        {CellId{7}, CellType::OP_ABS},
    });
    auto config = basic_config();
    config.absorption_rate = 4.0;
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {
            {CellId{101}, ResourceCompartmentId{3}, 0.0, 2.0, 0.0},
            {CellId{7}, ResourceCompartmentId{3}, 0.0, 4.0, 0.0},
        },
        {{ResourceCompartmentId{3}, 2.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    assert(ledger->settle(
        *fixture.runtime,
        native.measurement,
        {{ResourceCompartmentId{3}, 0.0}}).ok());
    const auto snapshot = ledger->snapshot();
    assert(std::abs(snapshot.cell(CellId{7})->energy - 4.0 / 3.0) < 1e-12);
    assert(std::abs(snapshot.cell(CellId{101})->energy - 2.0 / 3.0) < 1e-12);
    assert(std::abs(snapshot.compartment(ResourceCompartmentId{3})->environmental_resource) <
           1e-12);
}

void test_rejects_replay_without_mutating_snapshot_and_supports_episode_rebind() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto config = basic_config();
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    assert(ledger->settle(*fixture.runtime, native.measurement).ok());
    const auto before = ledger->snapshot();
    const auto replay = ledger->settle(*fixture.runtime, native.measurement);
    assert(!replay.ok());
    assert(replay.error->code == ResourceErrorCode::DuplicateMeasurement);
    assert(ledger->snapshot() == before);

    auto probe_runtime_result = fixture.runtime->fork_probe();
    assert(probe_runtime_result.ok());
    auto probe_runtime = std::move(probe_runtime_result.runtime);
    auto probe_ledger = ledger->fork_probe();
    const auto probe_native = ReferenceExecutor{}.step(*probe_runtime, {});
    assert(probe_native.ok());
    assert(probe_ledger->settle(*probe_runtime, probe_native.measurement).ok());
    assert(ledger->snapshot() == before);

    assert(fixture.runtime->reset_episode().ok());
    assert(ledger->rebind_episode(*fixture.runtime).ok());
    const auto next = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(next.ok());
    assert(ledger->settle(*fixture.runtime, next.measurement).ok());
    assert(ledger->snapshot().cell(CellId{11})->age > before.cell(CellId{11})->age);
}

void test_compiled_measurement_view_is_consumed_without_copying_fake_activity() {
    auto fixture = make_fixture({{CellId{UINT64_C(0x100000001)}, CellType::SENSE_RAW_INPUT_0}});
    auto config = basic_config();
    config.execution_costs.push_back(
        {CellType::SENSE_RAW_INPUT_0, 0.25});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{UINT64_C(0x100000001)}, ResourceCompartmentId{9}, 4.0, 4.0, 0.0}},
        {{ResourceCompartmentId{9}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto prepared = CompiledExecutor::prepare(fixture.runtime->plan());
    assert(prepared.ok());
    auto compiled_executor = std::move(prepared.executor);
    const double input = 1.0;
    const auto native = compiled_executor->step(*fixture.runtime, {&input, 1});
    assert(native.ok());
    assert(ledger->settle(*fixture.runtime, native.measurement).ok());
    const auto snapshot = ledger->snapshot();
    assert(snapshot.cell(CellId{UINT64_C(0x100000001)})->execution_count == 1);
    assert(std::abs(snapshot.cell(CellId{UINT64_C(0x100000001)})->energy - 3.75) <
           1e-12);
}

void test_previous_tick_transmission_uses_measured_contribution_per_edge() {
    GraphDefinition graph;
    graph.identity = GraphIdentity{9010};
    graph.revision = GraphRevision{1};
    graph.profile = SemanticProfile::StrictCore;
    graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_ABS},
    };
    graph.edges = {
        {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::PreviousTick},
        {EdgeId{3}, CellId{2}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::PreviousTick},
    };
    const auto compiled = GraphCompiler{}.compile(graph, seeds_for(graph));
    assert(compiled.ok());
    const auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    auto config = basic_config();
    config.transmission_scale = 1.0;
    config.transmission_cost = 2.0;
    auto attached = ResourceLedger::attach(
        *runtime,
        config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
        },
        {{ResourceCompartmentId{1}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    ReferenceExecutor executor;
    const double first_input = 1.0;
    const auto first = executor.step(*runtime, {&first_input, 1});
    assert(first.ok());
    double expected_transmission_cost = 0.0;
    for (const auto& edge : first.measurement.edges) {
        expected_transmission_cost +=
            2.0 * std::min(1.0, std::abs(edge.contribution));
    }
    assert(ledger->settle(*runtime, first.measurement).ok());
    const double second_input = 0.25;
    const auto second = executor.step(*runtime, {&second_input, 1});
    assert(second.ok());
    double previous_edge_contribution = 0.0;
    for (const auto& edge : second.measurement.edges) {
        expected_transmission_cost +=
            2.0 * std::min(1.0, std::abs(edge.contribution));
        if (edge.edge == EdgeId{2}) previous_edge_contribution = edge.contribution;
    }
    assert(std::abs(previous_edge_contribution) > 0.5);
    assert(ledger->settle(*runtime, second.measurement).ok());
    assert(std::abs(ledger->snapshot().cumulative_paid_cost() -
                    expected_transmission_cost) < 1e-12);
}

void test_invalid_measurement_and_failed_native_step_are_atomic() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto config = basic_config();
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto before = ledger->snapshot();
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    const auto failed = ReferenceExecutor{}.step(*fixture.runtime, {&invalid, 1});
    assert(!failed.ok());
    assert(fixture.runtime->tick() == 0);
    assert(ledger->snapshot() == before);
    assert(!ledger->settle(*fixture.runtime, failed.measurement).ok());
    assert(ledger->snapshot() == before);

    ExecutionMeasurement malformed;
    malformed.tick = 1;
    malformed.identity = fixture.runtime->identity();
    malformed.revision = fixture.runtime->revision();
    malformed.profile = fixture.runtime->profile();
    malformed.cells.push_back(
        {CellId{99}, CellType::OP_ABS, true, 0.0});
    const auto rejected = ledger->settle(*fixture.runtime, malformed);
    assert(!rejected.ok());
    assert(rejected.error->code == ResourceErrorCode::InvalidMeasurement);
    assert(ledger->snapshot() == before);
}

void test_shortage_preserves_graph_member_and_conservation() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto config = basic_config();
    config.maintenance_cost = 2.0;
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 3.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    const auto settled = ledger->settle(*fixture.runtime, native.measurement);
    assert(settled.ok());
    assert(settled.report.unpaid_cost == 1.0);
    assert(settled.report.exhausted_cells == 1);
    assert(ledger->snapshot().cell(CellId{11}) != nullptr);
    const auto snapshot = ledger->snapshot();
    const double lhs = snapshot.cell(CellId{11})->energy +
                       snapshot.cell(CellId{11})->structural_reserve +
                       snapshot.compartment(ResourceCompartmentId{7})
                           ->environmental_resource +
                       snapshot.totals().cumulative_dissipation +
                       snapshot.totals().cumulative_export;
    const double rhs = snapshot.totals().initial_total +
                       snapshot.totals().cumulative_external_injection;
    assert(std::abs(lhs - rhs) <=
           1e-9 * std::max(1.0, snapshot.totals().initial_total));
}

void test_reversed_storage_order_has_identical_per_id_allocations() {
    auto first = make_fixture({
        {CellId{7}, CellType::OP_ABS},
        {CellId{101}, CellType::OP_ABS},
    });
    auto second = make_fixture({
        {CellId{101}, CellType::OP_ABS},
        {CellId{7}, CellType::OP_ABS},
    });
    auto config = basic_config();
    config.absorption_rate = 2.0;
    auto first_attach = ResourceLedger::attach(
        *first.runtime,
        config,
        {
            {CellId{101}, ResourceCompartmentId{3}, 0.0, 2.0, 0.0},
            {CellId{7}, ResourceCompartmentId{3}, 0.0, 2.0, 0.0},
        },
        {{ResourceCompartmentId{3}, 2.0}});
    auto second_attach = ResourceLedger::attach(
        *second.runtime,
        config,
        {
            {CellId{7}, ResourceCompartmentId{3}, 0.0, 2.0, 0.0},
            {CellId{101}, ResourceCompartmentId{3}, 0.0, 2.0, 0.0},
        },
        {{ResourceCompartmentId{3}, 2.0}});
    assert(first_attach.ok() && second_attach.ok());
    auto first_ledger = std::move(first_attach.ledger);
    auto second_ledger = std::move(second_attach.ledger);
    const auto first_native = ReferenceExecutor{}.step(*first.runtime, {});
    const auto second_native = ReferenceExecutor{}.step(*second.runtime, {});
    assert(first_native.ok() && second_native.ok());
    assert(first_ledger->settle(*first.runtime, first_native.measurement).ok());
    assert(second_ledger->settle(*second.runtime, second_native.measurement).ok());
    const auto first_snapshot = first_ledger->snapshot();
    const auto second_snapshot = second_ledger->snapshot();
    assert(first_snapshot.cell(CellId{7})->energy ==
           second_snapshot.cell(CellId{7})->energy);
    assert(first_snapshot.cell(CellId{101})->energy ==
           second_snapshot.cell(CellId{101})->energy);
    assert(first_snapshot.cell(CellId{7})->energy == 1.0);
    assert(first_snapshot.cell(CellId{101})->energy == 1.0);
    assert(first_snapshot.compartment(ResourceCompartmentId{3})->environmental_resource ==
           second_snapshot.compartment(ResourceCompartmentId{3})->environmental_resource);
}

void test_long_native_run_preserves_conservation_with_shortages_and_injections() {
    auto fixture = make_fixture(
        {{CellId{UINT64_C(0x100000001)}, CellType::SENSE_RAW_INPUT_0}});
    auto config = basic_config();
    config.dt = 0.25;
    config.maintenance_cost = 0.001;
    config.activity_cost = 0.0001;
    config.execution_costs.push_back(
        {CellType::SENSE_RAW_INPUT_0, 0.001});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{UINT64_C(0x100000001)}, ResourceCompartmentId{44}, 5.0, 5.0, 2.0}},
        {{ResourceCompartmentId{44}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    ReferenceExecutor executor;
    for (uint64_t tick = 0; tick < 10000; ++tick) {
        const double input = static_cast<double>(tick % 7) * 0.1;
        const auto native = executor.step(*fixture.runtime, {&input, 1});
        assert(native.ok());
        const double supply = tick % 5 == 0 ? 0.0002 : 0.0;
        const auto settled = ledger->settle(
            *fixture.runtime, native.measurement,
            {{ResourceCompartmentId{44}, supply}});
        assert(settled.ok());
        const auto snapshot = ledger->snapshot();
        const double scale = std::max(
            1.0,
            snapshot.totals().initial_total +
                snapshot.totals().cumulative_external_injection);
        assert(std::abs(settled.report.conservation_residual) <=
               1e-9 * scale);
    }
    assert(ledger->snapshot().totals().cumulative_unpaid_cost > 0.0);
}

void test_attach_rejects_duplicate_missing_and_invalid_resource_records() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    const auto config = basic_config();
    const auto duplicate = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {
            {CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0},
            {CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0},
        },
        {{ResourceCompartmentId{7}, 0.0}});
    assert(!duplicate.ok());
    assert(duplicate.error->code == ResourceErrorCode::InvalidCell);
    const auto missing_compartment = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{8}, 1.0, 1.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(!missing_compartment.ok());
    assert(missing_compartment.error->code == ResourceErrorCode::MissingId);
    auto invalid_config = basic_config();
    invalid_config.dt = std::numeric_limits<double>::quiet_NaN();
    const auto invalid = ResourceLedger::attach(
        *fixture.runtime,
        invalid_config,
        {{CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(!invalid.ok());
    assert(invalid.error->code == ResourceErrorCode::InvalidConfig);
}

GraphDefinition edge_id_order_graph() {
    GraphDefinition graph;
    graph.identity = GraphIdentity{9020};
    graph.revision = GraphRevision{1};
    graph.profile = SemanticProfile::StrictCore;
    graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_ABS},
        {CellId{3}, CellType::OP_ABS},
    };
    graph.edges = {
        {EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
         EdgeDelay::Immediate},
    };
    return graph;
}

void test_edge_measurements_use_stable_edge_lookup_for_reference_and_compiled() {
    const auto graph = edge_id_order_graph();
    const auto compiled = GraphCompiler{}.compile(graph, seeds_for(graph));
    assert(compiled.ok());
    auto reference_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    auto compiled_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(reference_result.ok() && compiled_result.ok());
    auto reference_runtime = std::move(reference_result.runtime);
    auto compiled_runtime = std::move(compiled_result.runtime);
    auto attached = ResourceLedger::attach(
        *reference_runtime,
        basic_config(),
        {
            {CellId{1}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
            {CellId{3}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
        },
        {{ResourceCompartmentId{1}, 0.0}});
    assert(attached.ok());
    auto reference_ledger = std::move(attached.ledger);
    auto compiled_ledger = reference_ledger->fork_probe();
    const double input = 1.0;
    const auto reference_measurement =
        ReferenceExecutor{}.step(*reference_runtime, {&input, 1});
    assert(reference_measurement.ok());
    auto reordered = reference_measurement.measurement;
    std::reverse(reordered.cells.begin(), reordered.cells.end());
    std::reverse(reordered.edges.begin(), reordered.edges.end());
    std::reverse(reordered.ports.begin(), reordered.ports.end());
    assert(reference_ledger->settle(*reference_runtime, reordered).ok());

    const auto prepared = CompiledExecutor::prepare(compiled.graph);
    assert(prepared.ok());
    auto executor = std::move(prepared.executor);
    const auto compiled_measurement =
        executor->step(*compiled_runtime, {&input, 1});
    assert(compiled_measurement.ok());
    assert(compiled_ledger->settle(
        *compiled_runtime, compiled_measurement.measurement).ok());
}

void test_attach_at_running_tick_starts_cursor_at_runtime_tick() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    const auto first = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(first.ok());
    assert(fixture.runtime->tick() == 1);
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        basic_config(),
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto second = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(second.ok());
    assert(ledger->settle(*fixture.runtime, second.measurement).ok());
    assert(ledger->snapshot().last_measurement_tick() == 2);
    assert(ledger->snapshot().cell(CellId{11})->age == 1.0);

    const std::array<RuntimeCellState, 1> cells{{
        RuntimeCellState{CellId{11}, CellType::OP_ABS},
    }};
    const auto imported = RuntimeState::from_imported_state(
        fixture.compiled.graph,
        fixture.compiled.initial_values,
        cells,
        std::numeric_limits<uint64_t>::max());
    assert(imported.ok());
    const auto max_attached = ResourceLedger::attach(
        *imported.runtime,
        basic_config(),
        {{CellId{11}, ResourceCompartmentId{7}, 0.0, 0.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(max_attached.ok());
    assert(max_attached.ledger->snapshot().last_measurement_tick() ==
           std::numeric_limits<uint64_t>::max());
}

void test_proportional_allocation_handles_zero_demand_and_large_finite_values() {
    auto fixture = make_fixture({
        {CellId{1}, CellType::OP_ABS},
        {CellId{2}, CellType::OP_ABS},
        {CellId{3}, CellType::OP_ABS},
        {CellId{4}, CellType::OP_ABS},
    });
    auto config = basic_config();
    config.absorption_rate = 1.0;
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 0.0, 0.1, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 0.0, 0.1, 0.0},
            {CellId{3}, ResourceCompartmentId{1}, 0.0, 0.1, 0.0},
            {CellId{4}, ResourceCompartmentId{1}, 0.0, 0.0, 0.0},
        },
        {{ResourceCompartmentId{1}, 0.3}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    const auto settled = ledger->settle(
        *fixture.runtime, native.measurement);
    assert(settled.ok());
    const auto snapshot = ledger->snapshot();
    assert(snapshot.cell(CellId{4})->energy == 0.0);
    assert(std::abs(
               snapshot.compartment(ResourceCompartmentId{1})
                   ->environmental_resource) < 1e-12);

    auto large_fixture = make_fixture({
        {CellId{1}, CellType::OP_ABS},
        {CellId{2}, CellType::OP_ABS},
    });
    auto large_config = basic_config();
    large_config.absorption_rate = 1.0e200;
    auto large_attached = ResourceLedger::attach(
        *large_fixture.runtime,
        large_config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 0.0, 1.0e200, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 0.0, 1.0e200, 0.0},
        },
        {{ResourceCompartmentId{1}, 1.0e200}});
    assert(large_attached.ok());
    auto large_ledger = std::move(large_attached.ledger);
    const auto large_native = ReferenceExecutor{}.step(*large_fixture.runtime, {});
    assert(large_native.ok());
    assert(large_ledger->settle(
        *large_fixture.runtime, large_native.measurement).ok());
    assert(std::abs(
               large_ledger->snapshot().compartment(ResourceCompartmentId{1})
                   ->environmental_resource) < 1e185);

    auto tiny_fixture = make_fixture({
        {CellId{1}, CellType::OP_ABS},
        {CellId{2}, CellType::OP_ABS},
        {CellId{3}, CellType::OP_ABS},
        {CellId{4}, CellType::OP_ABS},
    });
    auto tiny_config = basic_config();
    tiny_config.absorption_rate = 1.0;
    auto tiny_attached = ResourceLedger::attach(
        *tiny_fixture.runtime,
        tiny_config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 0.0, 0.1, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 0.0, 0.1, 0.0},
            {CellId{3}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0},
            {CellId{4}, ResourceCompartmentId{1}, 0.0, 1.0e-30, 0.0},
        },
        {{ResourceCompartmentId{1}, 0.03}});
    assert(tiny_attached.ok());
    auto tiny_ledger = std::move(tiny_attached.ledger);
    const auto tiny_native = ReferenceExecutor{}.step(*tiny_fixture.runtime, {});
    assert(tiny_native.ok());
    const auto tiny_settled = tiny_ledger->settle(
        *tiny_fixture.runtime, tiny_native.measurement);
    assert(tiny_settled.ok());
    const auto tiny_snapshot = tiny_ledger->snapshot();
    assert(tiny_snapshot.cell(CellId{4})->energy >= 0.0);
    assert(tiny_snapshot.compartment(ResourceCompartmentId{1})
               ->environmental_resource >= 0.0);
}

void test_exact_energy_depletion_is_reported_without_unpaid_cost() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto config = basic_config();
    config.execution_costs.push_back({CellType::OP_ABS, 1.0});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        config,
        {{CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    const auto settled = ledger->settle(*fixture.runtime, native.measurement);
    assert(settled.ok());
    assert(settled.report.unpaid_cost == 0.0);
    assert(ledger->snapshot().cell(CellId{11})->energy == 0.0);
    assert(ledger->snapshot().cell(CellId{11})->exhausted);
}

void test_validated_measurement_mutations_are_atomic_and_valid_retry_succeeds() {
    const auto graph = edge_id_order_graph();
    const auto compiled = GraphCompiler{}.compile(graph, seeds_for(graph));
    assert(compiled.ok());
    const auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    auto attached = ResourceLedger::attach(
        *runtime,
        basic_config(),
        {
            {CellId{1}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
            {CellId{2}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
            {CellId{3}, ResourceCompartmentId{1}, 2.0, 2.0, 0.0},
        },
        {{ResourceCompartmentId{1}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    const double input = 1.0;
    const auto native = ReferenceExecutor{}.step(*runtime, {&input, 1});
    assert(native.ok());
    const auto before = ledger->snapshot();
    const auto reject_without_mutating = [&](ExecutionMeasurement malformed) {
        auto probe = ledger->fork_probe();
        const auto rejected = probe->settle(*runtime, malformed);
        assert(!rejected.ok());
        assert(probe->snapshot() == before);
        assert(ledger->snapshot() == before);
    };

    auto unknown_cell = native.measurement;
    unknown_cell.cells[0].cell = CellId{99};
    reject_without_mutating(unknown_cell);
    auto duplicate_cell = native.measurement;
    duplicate_cell.cells[1].cell = duplicate_cell.cells[0].cell;
    reject_without_mutating(duplicate_cell);
    auto unknown_edge = native.measurement;
    unknown_edge.edges[0].edge = EdgeId{999};
    reject_without_mutating(unknown_edge);
    auto wrong_timing = native.measurement;
    wrong_timing.edges[0].source_mode = EdgeDelay::PreviousTick;
    reject_without_mutating(wrong_timing);
    auto nonfinite_edge = native.measurement;
    nonfinite_edge.edges[0].contribution =
        std::numeric_limits<double>::infinity();
    reject_without_mutating(nonfinite_edge);
    auto nonfinite_source = native.measurement;
    nonfinite_source.edges[0].source_value =
        std::numeric_limits<double>::quiet_NaN();
    reject_without_mutating(nonfinite_source);
    auto nonfinite_output = native.measurement;
    nonfinite_output.cells[0].output =
        std::numeric_limits<double>::quiet_NaN();
    reject_without_mutating(nonfinite_output);
    auto wrong_port = native.measurement;
    wrong_port.ports[0].port = InputPort{1};
    reject_without_mutating(wrong_port);
    assert(ledger->settle(*runtime, native.measurement).ok());
}

void test_tick_plan_and_injection_rejections_preserve_cursor() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        basic_config(),
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    ReferenceExecutor executor;
    const auto first = executor.step(*fixture.runtime, {});
    auto first_runtime_result = fixture.runtime->fork_probe();
    assert(first_runtime_result.ok());
    auto first_runtime = std::move(first_runtime_result.runtime);
    const auto second = executor.step(*fixture.runtime, {});
    assert(first.ok() && second.ok());
    const auto before = ledger->snapshot();
    assert(!ledger->settle(*fixture.runtime, second.measurement).ok());
    assert(ledger->snapshot() == before);
    assert(ledger->settle(*first_runtime, first.measurement).ok());
    const auto after_first = ledger->snapshot();
    assert(!ledger->settle(*fixture.runtime, first.measurement).ok());
    assert(ledger->snapshot() == after_first);
    assert(ledger->settle(*fixture.runtime, second.measurement).ok());
    const auto after_second = ledger->snapshot();
    assert(!ledger->settle(*fixture.runtime, first.measurement).ok());
    assert(ledger->snapshot() == after_second);

    auto injection_fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto injection_attached = ResourceLedger::attach(
        *injection_fixture.runtime,
        basic_config(),
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(injection_attached.ok());
    auto injection_ledger = std::move(injection_attached.ledger);
    const auto injection_native =
        ReferenceExecutor{}.step(*injection_fixture.runtime, {});
    assert(injection_native.ok());
    const auto injection_before = injection_ledger->snapshot();
    assert(!injection_ledger->settle(
        *injection_fixture.runtime, injection_native.measurement,
        {{ResourceCompartmentId{7}, 0.0}, {ResourceCompartmentId{7}, 0.0}}).ok());
    assert(injection_ledger->snapshot() == injection_before);
    assert(!injection_ledger->settle(
        *injection_fixture.runtime, injection_native.measurement,
        {{ResourceCompartmentId{99}, 0.0}}).ok());
    assert(injection_ledger->snapshot() == injection_before);
    assert(!injection_ledger->settle(
        *injection_fixture.runtime, injection_native.measurement,
        {{ResourceCompartmentId{7}, -1.0}}).ok());
    assert(injection_ledger->snapshot() == injection_before);
    assert(injection_ledger->settle(
        *injection_fixture.runtime, injection_native.measurement,
        {{ResourceCompartmentId{7}, 0.0}}).ok());
}

void test_same_metadata_different_plan_is_rejected() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    auto attached = ResourceLedger::attach(
        *fixture.runtime,
        basic_config(),
        {{CellId{11}, ResourceCompartmentId{7}, 2.0, 2.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(attached.ok());
    auto ledger = std::move(attached.ledger);
    GraphDefinition other_graph;
    other_graph.identity = fixture.runtime->identity();
    other_graph.revision = fixture.runtime->revision();
    other_graph.profile = fixture.runtime->profile();
    other_graph.cells = {
        {CellId{11}, CellType::OP_ABS},
        {CellId{12}, CellType::OP_ABS},
    };
    const auto other_compiled =
        GraphCompiler{}.compile(other_graph, seeds_for(other_graph));
    assert(other_compiled.ok());
    const auto other_runtime_result =
        RuntimeState::create(other_compiled.graph, other_compiled.initial_values);
    assert(other_runtime_result.ok());
    auto other_runtime = std::move(other_runtime_result.runtime);
    const auto other_measurement =
        ReferenceExecutor{}.step(*other_runtime, {});
    assert(other_measurement.ok());
    const auto before = ledger->snapshot();
    assert(!ledger->settle(
        *other_runtime, other_measurement.measurement).ok());
    assert(ledger->snapshot() == before);
}

void test_invalid_config_values_and_derived_overflow_are_rejected_atomically() {
    auto fixture = make_fixture({{CellId{11}, CellType::OP_ABS}});
    const auto invalid_config = [&](ResourceLedgerConfig config) {
        const auto result = ResourceLedger::attach(
            *fixture.runtime,
            std::move(config),
            {{CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0}},
            {{ResourceCompartmentId{7}, 0.0}});
        assert(!result.ok());
        assert(result.error->code == ResourceErrorCode::InvalidConfig);
    };
    auto zero_activity_scale = basic_config();
    zero_activity_scale.activity_scale = 0.0;
    invalid_config(zero_activity_scale);
    auto negative_rate = basic_config();
    negative_rate.absorption_rate = -1.0;
    invalid_config(negative_rate);
    auto negative_capacity = basic_config();
    const auto bad_capacity = ResourceLedger::attach(
        *fixture.runtime, negative_capacity,
        {{CellId{11}, ResourceCompartmentId{7}, 1.0, -1.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(!bad_capacity.ok());
    assert(bad_capacity.error->code == ResourceErrorCode::InvalidCell);
    auto negative_execution = basic_config();
    negative_execution.execution_costs.push_back({CellType::OP_ABS, -1.0});
    invalid_config(negative_execution);

    auto overflow_config = basic_config();
    overflow_config.dt = std::numeric_limits<double>::max();
    overflow_config.maintenance_cost = std::numeric_limits<double>::max();
    auto overflow_attached = ResourceLedger::attach(
        *fixture.runtime, overflow_config,
        {{CellId{11}, ResourceCompartmentId{7}, 1.0, 1.0, 0.0}},
        {{ResourceCompartmentId{7}, 0.0}});
    assert(overflow_attached.ok());
    auto overflow_ledger = std::move(overflow_attached.ledger);
    const auto native = ReferenceExecutor{}.step(*fixture.runtime, {});
    assert(native.ok());
    const auto before = overflow_ledger->snapshot();
    assert(!overflow_ledger->settle(*fixture.runtime, native.measurement).ok());
    assert(overflow_ledger->snapshot() == before);
}

void test_native_absorption_competition_runs_for_strict_and_legacy_profiles() {
    auto strict = make_fixture(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}});
    auto legacy = make_fixture(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        SemanticProfile::LegacyCompatible);
    auto config = basic_config();
    config.absorption_rate = 0.5;
    config.maintenance_cost = 0.6;
    config.execution_costs.push_back({CellType::OP_ABS, 0.2});
    auto strict_attached = ResourceLedger::attach(
        *strict.runtime, config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 0.8, 1.0, 1.0},
            {CellId{2}, ResourceCompartmentId{2}, 0.2, 1.0, 2.0},
        },
        {
            {ResourceCompartmentId{1}, 0.0},
            {ResourceCompartmentId{2}, 0.0},
        });
    auto legacy_attached = ResourceLedger::attach(
        *legacy.runtime, config,
        {
            {CellId{1}, ResourceCompartmentId{1}, 0.8, 1.0, 1.0},
            {CellId{2}, ResourceCompartmentId{2}, 0.2, 1.0, 2.0},
        },
        {
            {ResourceCompartmentId{1}, 0.0},
            {ResourceCompartmentId{2}, 0.0},
        });
    assert(strict_attached.ok() && legacy_attached.ok());
    auto strict_ledger = std::move(strict_attached.ledger);
    auto legacy_ledger = std::move(legacy_attached.ledger);
    ReferenceExecutor executor;
    double max_residual = 0.0;
    double max_scale = 1.0;
    for (uint64_t tick = 0; tick < 1000; ++tick) {
        const auto strict_native = executor.step(*strict.runtime, {});
        const auto legacy_native = executor.step(*legacy.runtime, {});
        assert(strict_native.ok() && legacy_native.ok());
        const double first_supply = tick % 3 == 0 ? 0.4 : 0.0;
        const double second_supply = tick % 5 == 0 ? 0.3 : 0.0;
        const std::array<ResourceSupply, 2> supplies{{
            {ResourceCompartmentId{1}, first_supply},
            {ResourceCompartmentId{2}, second_supply},
        }};
        const auto strict_settled = strict_ledger->settle(
            *strict.runtime, strict_native.measurement, supplies);
        const auto legacy_settled = legacy_ledger->settle(
            *legacy.runtime, legacy_native.measurement, supplies);
        assert(strict_settled.ok() && legacy_settled.ok());
        const auto current = strict_ledger->snapshot();
        const double scale = std::max(
            1.0,
            current.totals().initial_total +
                current.totals().cumulative_external_injection);
        max_residual = std::max(
            max_residual, std::abs(strict_settled.report.conservation_residual));
        max_scale = std::max(max_scale, scale);
    }
    const auto strict_snapshot = strict_ledger->snapshot();
    const auto legacy_snapshot = legacy_ledger->snapshot();
    assert(strict_snapshot.totals() == legacy_snapshot.totals());
    for (const auto& cell : strict_snapshot.cells()) {
        assert(*strict_snapshot.cell(cell.cell) == *legacy_snapshot.cell(cell.cell));
    }
    double lhs = 0.0;
    for (const auto& cell : strict_snapshot.cells()) {
        lhs += cell.energy + cell.structural_reserve;
    }
    for (const auto& compartment : strict_snapshot.compartments()) {
        lhs += compartment.environmental_resource;
    }
    lhs += strict_snapshot.totals().cumulative_dissipation;
    const double rhs = strict_snapshot.totals().initial_total +
                       strict_snapshot.totals().cumulative_external_injection;
    const double scale = std::max(1.0, rhs);
    assert(std::abs(lhs - rhs) <= 1e-9 * scale);
    assert(strict_snapshot.totals().cumulative_unpaid_cost > 0.0);
    assert(max_residual <= 1e-9 * max_scale);
}

}  // namespace

int main() {
    test_real_execution_cost_is_paid_once_per_native_step();
    test_activity_cost_is_optional_but_activity_is_recorded();
    test_compartment_demand_is_proportional_and_order_independent();
    test_rejects_replay_without_mutating_snapshot_and_supports_episode_rebind();
    test_compiled_measurement_view_is_consumed_without_copying_fake_activity();
    test_previous_tick_transmission_uses_measured_contribution_per_edge();
    test_invalid_measurement_and_failed_native_step_are_atomic();
    test_shortage_preserves_graph_member_and_conservation();
    test_edge_measurements_use_stable_edge_lookup_for_reference_and_compiled();
    test_attach_at_running_tick_starts_cursor_at_runtime_tick();
    test_proportional_allocation_handles_zero_demand_and_large_finite_values();
    test_exact_energy_depletion_is_reported_without_unpaid_cost();
    test_validated_measurement_mutations_are_atomic_and_valid_retry_succeeds();
    test_tick_plan_and_injection_rejections_preserve_cursor();
    test_same_metadata_different_plan_is_rejected();
    test_invalid_config_values_and_derived_overflow_are_rejected_atomically();
    test_native_absorption_competition_runs_for_strict_and_legacy_profiles();
    test_reversed_storage_order_has_identical_per_id_allocations();
    test_long_native_run_preserves_conservation_with_shortages_and_injections();
    test_attach_rejects_duplicate_missing_and_invalid_resource_records();
    return 0;
}
