#include "kun/cellular/core/cellular_graph_edit.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/runtime_state.hpp"

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
    GraphDefinition definition;
    CompileResult compiled;
    std::shared_ptr<RuntimeState> runtime;
};

Fixture make_fixture(
    std::vector<CellDefinition> cells,
    std::vector<EdgeDefinition> edges = {}) {
    Fixture fixture;
    fixture.definition = GraphDefinition{
        GraphIdentity{9100},
        GraphRevision{1},
        SemanticProfile::StrictCore,
        1,
        std::move(cells),
        std::move(edges)};
    fixture.compiled =
        GraphCompiler{}.compile(fixture.definition, seeds_for(fixture.definition));
    assert(fixture.compiled.ok());
    auto runtime =
        RuntimeState::create(fixture.compiled.graph, fixture.compiled.initial_values);
    assert(runtime.ok());
    fixture.runtime = std::move(runtime.runtime);
    return fixture;
}

ResourceLedgerConfig resource_config() {
    ResourceLedgerConfig config;
    config.dt = 1.0;
    config.maintenance_cost = 0.0;
    config.absorption_rate = 0.0;
    config.activity_scale = 1.0;
    config.activity_cost = 0.0;
    config.transmission_scale = 1.0;
    config.transmission_cost = 0.0;
    return config;
}

LifecycleConfig thresholds(
    double apoptosis,
    double dormant_enter,
    double dormant_exit,
    uint64_t dwell = 0,
    uint64_t cooldown = 0) {
    LifecycleConfig config;
    config.apoptotic_resource = apoptosis;
    config.dormant_enter_resource = dormant_enter;
    config.dormant_exit_resource = dormant_exit;
    config.minimum_dwell_ticks = dwell;
    config.cooldown_ticks = cooldown;
    return config;
}

std::unique_ptr<CellularLifecycleController> attach_controller(
    Fixture& fixture,
    ResourceLedgerConfig resource_config,
    std::vector<ResourceCellInitial> cells,
    LifecycleConfig lifecycle_config) {
    const std::array<ResourceCompartmentInitial, 1> compartments{
        ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0}};
    auto ledger = ResourceLedger::attach(
        *fixture.runtime,
        std::move(resource_config),
        std::span<const ResourceCellInitial>(cells.data(), cells.size()),
        std::span<const ResourceCompartmentInitial>(
            compartments.data(), compartments.size()));
    assert(ledger.ok());
    auto prepared = CompiledExecutor::prepare(fixture.runtime->plan());
    assert(prepared.ok());
    auto controller = CellularLifecycleController::create(
        *fixture.runtime,
        std::move(prepared.executor),
        std::move(ledger.ledger),
        lifecycle_config);
    assert(controller.ok());
    return std::move(controller.controller);
}

const EdgeTransmissionMeasurement* edge_measurement(
    const ExecutionMeasurementView& measurement,
    EdgeId id) {
    for (const auto& edge : measurement.edges) {
        if (edge.edge == id) return &edge;
    }
    return nullptr;
}

ParameterBinding edge_binding(const RuntimeState& runtime, EdgeId id) {
    for (const auto& parameter : runtime.parameters()) {
        if (parameter.binding.kind == ParameterBindingKind::EdgeWeight &&
            parameter.binding.edge == id) {
            return parameter.binding;
        }
    }
    assert(false);
    return {};
}

void test_resource_state_controls_native_execution() {
    auto abundant = make_fixture({{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto depleted = make_fixture({{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto abundant_controller = attach_controller(
        abundant,
        resource_config(),
        {{CellId{1}, ResourceCompartmentId{1}, 1.0, 1.0, 0.0}},
        thresholds(0.0, 0.1, 0.2));
    auto depleted_controller = attach_controller(
        depleted,
        resource_config(),
        {{CellId{1}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0}},
        thresholds(0.0, 0.1, 0.2));

    const double input = 2.0;
    const auto abundant_step = abundant_controller->step({&input, 1}, {});
    const auto depleted_step = depleted_controller->step({&input, 1}, {});
    assert(abundant_step.ok());
    assert(abundant_step.executed);
    assert(abundant_step.measurement.has_value());
    assert(depleted_step.ok());
    assert(depleted_step.extinct);
    assert(!depleted_step.executed);
    assert(abundant.runtime->cell_state(CellId{1})->output_val == 2.0);
    assert(depleted.runtime->plan()->cells().empty());
}

void test_lifecycle_config_rejects_non_finite_and_misordered_thresholds() {
    auto fixture = make_fixture({{CellId{1}, CellType::OP_ABS}});
    const std::array<ResourceCellInitial, 1> cells{
        ResourceCellInitial{CellId{1}, ResourceCompartmentId{1}, 1.0, 1.0, 0.0}};
    const std::array<ResourceCompartmentInitial, 1> compartments{
        ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0}};
    auto ledger = ResourceLedger::attach(
        *fixture.runtime,
        resource_config(),
        std::span<const ResourceCellInitial>(cells.data(), cells.size()),
        std::span<const ResourceCompartmentInitial>(
            compartments.data(), compartments.size()));
    assert(ledger.ok());
    auto prepared = CompiledExecutor::prepare(fixture.runtime->plan());
    assert(prepared.ok());
    auto invalid = thresholds(0.5, 0.2, 0.1);
    invalid.apoptotic_resource = std::numeric_limits<double>::quiet_NaN();
    const auto result = CellularLifecycleController::create(
        *fixture.runtime,
        std::move(prepared.executor),
        std::move(ledger.ledger),
        invalid);
    assert(!result.ok());
    assert(result.error->code == LifecycleErrorCode::InvalidConfig);
}

void test_dormant_clears_output_and_transmission_but_preserves_memory() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_EMA}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto resource = resource_config();
    resource.maintenance_cost = 0.1;
    resource.absorption_rate = 1.0;
    auto controller = attach_controller(
        fixture,
        resource,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 0.8, 1.0, 0.0}},
        thresholds(0.0, 0.75, 0.85));

    const double first_input = 1.0;
    const auto first = controller->step({&first_input, 1}, {});
    assert(first.ok());
    const auto before = *fixture.runtime->cell_state(CellId{2});
    assert(controller->state(CellId{2}) == LifecycleState::Dormant);

    const double second_input = 2.0;
    const auto second = controller->step({&second_input, 1}, {});
    assert(second.ok());
    assert(second.measurement.has_value());
    const auto* dormant = fixture.runtime->cell_state(CellId{2});
    assert(dormant->state_val == before.state_val);
    assert(dormant->aux_state == before.aux_state);
    assert(dormant->delay_buffer == before.delay_buffer);
    assert(dormant->output_val == 0.0);
    const auto* transmission = edge_measurement(*second.measurement, EdgeId{1});
    assert(transmission != nullptr);
    assert(transmission->contribution == 0.0);
}

void test_dormant_preserves_previous_output_and_delay_semantics_until_wake() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N},
         {CellId{3}, CellType::OP_ABS}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::PreviousTick}});
    auto resource = resource_config();
    resource.maintenance_cost = 0.3;
    resource.absorption_rate = 1.0;
    auto controller = attach_controller(
        fixture,
        resource,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 1.0, 1.0, 0.0},
         {CellId{3}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}},
        thresholds(0.0, 0.5, 0.65));

    const double first_input = 1.0;
    assert(controller->step({&first_input, 1}, {}).ok());
    const double second_input = 2.0;
    assert(controller->step({&second_input, 1}, {}).ok());
    const auto before_dormant = *fixture.runtime->cell_state(CellId{2});
    assert(std::abs(before_dormant.prev_output_val) > 0.5);
    assert(controller->state(CellId{2}) == LifecycleState::Dormant);

    const double third_input = 3.0;
    const auto dormant_tick =
        controller->step({&third_input, 1}, {});
    assert(dormant_tick.ok());
    assert(dormant_tick.measurement.has_value());
    const auto after_dormant = *fixture.runtime->cell_state(CellId{2});
    assert(after_dormant.output_val == 0.0);
    assert(after_dormant.state_val == before_dormant.state_val);
    assert(after_dormant.aux_state == before_dormant.aux_state);
    assert(after_dormant.prev_input == before_dormant.prev_input);
    assert(after_dormant.prev_output_val == before_dormant.prev_output_val);
    assert(after_dormant.delay_buffer == before_dormant.delay_buffer);
    assert(after_dormant.delay_idx == before_dormant.delay_idx);
    assert(after_dormant.latch_state == before_dormant.latch_state);
    const auto* dormant_edge =
        edge_measurement(*dormant_tick.measurement, EdgeId{2});
    assert(dormant_edge != nullptr);
    assert(dormant_edge->contribution == 0.0);

    const double wake_supply_input = 4.0;
    const auto wake_supply_tick = controller->step(
        {&wake_supply_input, 1}, {{ResourceCompartmentId{1}, 3.0}});
    assert(wake_supply_tick.ok());
    assert(controller->state(CellId{2}) == LifecycleState::Active);
    const double wake_input = 5.0;
    const auto wake_tick = controller->step({&wake_input, 1}, {});
    assert(wake_tick.ok());
    assert(wake_tick.measurement.has_value());
    const auto* wake_edge = edge_measurement(*wake_tick.measurement, EdgeId{2});
    assert(wake_edge != nullptr);
    assert(std::abs(
               wake_edge->contribution - before_dormant.prev_output_val) <
           1e-12);
}

void test_replenishment_wakes_dormant_cell() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_EMA}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto resource = resource_config();
    resource.maintenance_cost = 0.1;
    resource.absorption_rate = 1.0;
    auto controller = attach_controller(
        fixture,
        resource,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 0.8, 1.0, 0.0}},
        thresholds(0.0, 0.75, 0.85));

    const double input = 1.0;
    assert(controller->step({&input, 1}, {}).ok());
    assert(controller->state(CellId{2}) == LifecycleState::Dormant);
    assert(controller->step(
        {&input, 1}, {{ResourceCompartmentId{1}, 1.0}}).ok());
    assert(controller->state(CellId{2}) == LifecycleState::Active);
    const auto woke = controller->step({&input, 1}, {});
    assert(woke.ok());
    assert(woke.measurement.has_value());
    assert(fixture.runtime->cell_state(CellId{2})->output_val != 0.0);
}

void test_hysteresis_and_minimum_dwell_prevent_chatter() {
    auto fixture = make_fixture({{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto resource = resource_config();
    resource.absorption_rate = 1.0;
    auto controller = attach_controller(
        fixture,
        resource,
        {{CellId{1}, ResourceCompartmentId{1}, 0.5, 1.0, 0.0}},
        thresholds(0.0, 0.8, 0.9, 2, 3));
    const double input = 1.0;
    assert(controller->step({&input, 1}, {}).ok());
    assert(controller->state(CellId{1}) == LifecycleState::Active);
    assert(controller->step({&input, 1}, {}).ok());
    assert(controller->state(CellId{1}) == LifecycleState::Dormant);
    assert(controller->step(
        {&input, 1}, {{ResourceCompartmentId{1}, 1.0}}).ok());
    assert(controller->state(CellId{1}) == LifecycleState::Dormant);
    assert(controller->step({&input, 1}, {}).ok());
    assert(controller->state(CellId{1}) == LifecycleState::Dormant);
    assert(controller->step({&input, 1}, {}).ok());
    assert(controller->state(CellId{1}) == LifecycleState::Active);
}

void test_apoptosis_removes_incident_edges_and_preserves_survivor_identity() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N},
         {CellId{3}, CellType::OP_EMA}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::PreviousTick}});
    auto resource = resource_config();
    resource.maintenance_cost = 0.1;
    auto controller = attach_controller(
        fixture,
        resource,
        {{CellId{1}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{3}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}},
        thresholds(0.0, 0.0, 1.0));
    const auto stale_executor = controller->executor();
    const double input = 3.0;
    assert(controller->step({&input, 1}, {}).ok());
    assert(!stale_executor->step(*fixture.runtime, {&input, 1}).ok());
    const auto survivor_before = *fixture.runtime->cell_state(CellId{2});
    const auto edge_weight = edge_binding(*fixture.runtime, EdgeId{2});
    assert(fixture.runtime->set_parameter(
        edge_weight, ParameterValue{ContinuousValue{2.5}}).ok());
    const auto removed = controller->step({&input, 1}, {});
    assert(removed.ok());
    assert(fixture.runtime->cell_state(CellId{1}) == nullptr);
    assert(fixture.runtime->cell_state(CellId{2}) != nullptr);
    assert(fixture.runtime->cell_state(CellId{2})->state_val ==
           survivor_before.state_val);
    assert(fixture.runtime->cell_state(CellId{2})->delay_buffer ==
           survivor_before.delay_buffer);
    assert(fixture.runtime->plan()->edges().size() == 1);
    assert(fixture.runtime->plan()->edges()[0].id == EdgeId{2});
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{2}).index))
               .value == 2.5);
}

void test_failed_deletion_is_atomic() {
    auto fixture = make_fixture({{CellId{1}, CellType::OP_ABS}});
    auto before_runtime = fixture.runtime->snapshot();
    GraphEditor editor(*fixture.runtime);
    const auto result = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{1, RemoveCellAction{CellId{99}}}});
    assert(!result.ok());
    assert(fixture.runtime->revision() == before_runtime.plan()->revision());
    assert(fixture.runtime->plan()->cells().size() ==
           before_runtime.plan()->cells().size());
    assert(fixture.runtime->tick() == before_runtime.tick());
}

void test_all_cell_death_publishes_extinct_without_fallback() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}});
    auto controller = attach_controller(
        fixture,
        resource_config(),
        {{CellId{1}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0}},
        thresholds(0.0, 0.0, 1.0));
    const auto extinct = controller->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(extinct.extinct);
    assert(!extinct.executed);
    assert(fixture.runtime->plan()->cells().empty());
    assert(controller->overall_state() == LifecycleState::Extinct);
    assert(!controller->step(
        std::span<const double>{}, std::span<const ResourceSupply>{}).executed);
    assert(fixture.runtime->plan()->cells().empty());
}

}  // namespace

int main() {
    test_resource_state_controls_native_execution();
    test_lifecycle_config_rejects_non_finite_and_misordered_thresholds();
    test_dormant_clears_output_and_transmission_but_preserves_memory();
    test_dormant_preserves_previous_output_and_delay_semantics_until_wake();
    test_replenishment_wakes_dormant_cell();
    test_hysteresis_and_minimum_dwell_prevent_chatter();
    test_apoptosis_removes_incident_edges_and_preserves_survivor_identity();
    test_failed_deletion_is_atomic();
    test_all_cell_death_publishes_extinct_without_fallback();
    return 0;
}
