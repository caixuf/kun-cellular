#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/growth_controller.hpp"
#include "kun/cellular/core/lifecycle_controller.hpp"
#include "kun/cellular/core/resource_ledger.hpp"
#include "kun/cellular/core/runtime_state.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
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
                        descriptor.name &&
                                std::string_view(descriptor.name) == "alpha"
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
                CellParameterSeed{
                    cell.id, static_cast<ParameterSlot>(slot), value});
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
        GraphIdentity{9200},
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

LifecycleConfig lifecycle_config() {
    LifecycleConfig config;
    config.apoptotic_resource = 0.0;
    config.dormant_enter_resource = 0.0;
    config.dormant_exit_resource = 1.0;
    return config;
}

GrowthConfig growth_config() {
    GrowthConfig config;
    config.cell_birth_cost = 1.0;
    config.synapse_birth_cost = 0.25;
    config.initial_energy = 1.0;
    config.initial_capacity = 1.0;
    config.initial_structural_reserve = 0.0;
    return config;
}

std::unique_ptr<CellularGrowthController> attach_growth(
    Fixture& fixture,
    std::vector<ResourceCellInitial> cells,
    GrowthConfig config = growth_config()) {
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
    auto lifecycle = CellularLifecycleController::create(
        *fixture.runtime,
        std::move(prepared.executor),
        std::move(ledger.ledger),
        lifecycle_config());
    assert(lifecycle.ok());
    auto growth = CellularGrowthController::create(
        std::move(lifecycle.controller), config);
    assert(growth.ok());
    return std::move(growth.controller);
}

GrowthFunding funding(CellId sponsor = CellId{1}) {
    return GrowthFunding{
        sponsor.value == 0 ? std::optional<CellId>{}
                           : std::optional<CellId>{sponsor},
        ResourceCompartmentId{1}};
}

CellBirth sum_birth(CellId id) {
    return CellBirth{
        id,
        CellType::OP_SUM,
        {ParameterValue{UnusedParameter{}}, ParameterValue{UnusedParameter{}}}};
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

void test_funded_birth_changes_graph_and_executes_next_step() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_ABS}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}});
    const auto old_revision = fixture.runtime->revision();
    const auto result = growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            1,
            sum_birth(CellId{10}),
            {{EdgeBirth{
                EdgeId{100}, CellId{1}, OutputPort{0}, CellId{10},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{101}, CellId{10}, OutputPort{0}, CellId{2},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            funding()}});
    assert(result.ok());

    const double input = 2.0;
    const auto stepped = growth->step({&input, 1}, {});
    assert(stepped.ok());
    assert(stepped.growth_events.size() >= 1);
    assert(fixture.runtime->revision().value == old_revision.value + 1);
    assert(fixture.runtime->cell_state(CellId{10}) != nullptr);
    assert(stepped.lifecycle.measurement.has_value());
    assert(std::any_of(
        stepped.lifecycle.measurement->cells.begin(),
        stepped.lifecycle.measurement->cells.end(),
        [](const auto& cell) {
            return cell.cell == CellId{10} && cell.executed;
        }));
}

void test_unfunded_birth_is_atomic_and_releases_reservation() {
    auto fixture = make_fixture({{CellId{1}, CellType::OP_ABS}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 0.5, 0.5, 0.0}});
    const auto before_runtime = fixture.runtime->snapshot();
    const auto before_ledger = growth->lifecycle().ledger().snapshot();
    const auto before_executor = growth->lifecycle().executor();
    const auto result = growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            2,
            sum_birth(CellId{11}),
            {{EdgeBirth{
                EdgeId{110}, CellId{1}, OutputPort{0}, CellId{11},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{111}, CellId{11}, OutputPort{0}, CellId{1},
                InputPort{0}, EdgeDelay::PreviousTick, 1.0}}},
            funding()}});
    assert(result.ok());

    const auto stepped = growth->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(!stepped.ok());
    assert(stepped.growth_error.has_value());
    assert(stepped.growth_error->code == GrowthErrorCode::InsufficientResource);
    assert(stepped.growth_error->reason.find("limit") == std::string::npos);
    assert(stepped.growth_error->reason.find("count") == std::string::npos);
    assert(fixture.runtime->revision() == before_runtime.plan()->revision());
    assert(fixture.runtime->snapshot().tick() == before_runtime.tick());
    assert(growth->lifecycle().ledger().snapshot() == before_ledger);
    assert(growth->lifecycle().executor() == before_executor);
    assert(fixture.runtime->cell_state(CellId{11}) == nullptr);
}

void test_new_cell_affects_downstream_output() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_ABS}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}});
    assert(growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            3,
            sum_birth(CellId{12}),
            {{EdgeBirth{
                EdgeId{120}, CellId{1}, OutputPort{0}, CellId{12},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{121}, CellId{12}, OutputPort{0}, CellId{2},
                InputPort{0}, EdgeDelay::Immediate, 2.0}}},
            funding()}})
               .ok());

    const double input = 3.0;
    const auto stepped = growth->step({&input, 1}, {});
    assert(stepped.ok());
    const auto downstream = std::find_if(
        stepped.lifecycle.measurement->cells.begin(),
        stepped.lifecycle.measurement->cells.end(),
        [](const auto& cell) { return cell.cell == CellId{2}; });
    assert(downstream != stepped.lifecycle.measurement->cells.end());
    assert(std::abs(downstream->output) > 1e-12);
}

void test_synapse_addition_and_reclamation_update_measurements() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_SUM}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}});
    assert(growth->submit(GrowthProposal{
        GrowthSynapseProposal{
            4,
            EdgeBirth{
                EdgeId{2}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{1},
                EdgeDelay::Immediate, 2.0},
            funding(),
            0.25}})
               .ok());
    const double input = 1.0;
    const auto added = growth->step({&input, 1}, {});
    assert(added.ok());
    assert(edge_measurement(*added.lifecycle.measurement, EdgeId{2}) != nullptr);
    assert(std::abs(
               edge_measurement(*added.lifecycle.measurement, EdgeId{2})
                   ->contribution -
               2.0) < 1e-12);

    assert(growth->submit(GrowthProposal{
        GrowthReclaimProposal{5, EdgeId{2}}})
               .ok());
    const auto reclaimed = growth->step({&input, 1}, {});
    assert(reclaimed.ok());
    assert(edge_measurement(*reclaimed.lifecycle.measurement, EdgeId{2}) == nullptr);
    assert(std::none_of(
        fixture.runtime->plan()->edges().begin(),
        fixture.runtime->plan()->edges().end(),
        [](const auto& edge) { return edge.id == EdgeId{2}; }));
}

void test_growth_supports_more_than_sixty_four_cells_and_edges() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_SUM}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 1000.0, 1000.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 1000.0, 1000.0, 0.0}});
    std::vector<GrowthProposal> proposals;
    for (uint64_t index = 0; index < 70; ++index) {
        const CellId cell{100 + index};
        proposals.push_back(GrowthProposal{
            GrowthCellBirthProposal{
                1000 + index,
                sum_birth(cell),
                {{EdgeBirth{
                    EdgeId{2000 + index}, CellId{1}, OutputPort{0}, cell,
                    InputPort{0}, EdgeDelay::Immediate, 1.0}}},
                {{EdgeBirth{
                    EdgeId{3000 + index}, cell, OutputPort{0}, CellId{2},
                    InputPort{0}, EdgeDelay::Immediate, 1.0}}},
                funding()}});
    }
    assert(growth->submit(proposals).ok());
    const double input = 1.0;
    const auto stepped = growth->step({&input, 1}, {});
    assert(stepped.ok());
    assert(fixture.runtime->plan()->cells().size() == 72);
    assert(fixture.runtime->plan()->edges().size() == 140);
    assert(stepped.lifecycle.measurement->cells.size() == 72);
    assert(stepped.lifecycle.measurement->edges.size() == 140);
}

void test_natural_death_commits_before_failed_growth() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::OP_ABS},
         {CellId{2}, CellType::OP_ABS}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 0.0, 1.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 0.5, 0.5, 0.0}});
    assert(growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            6,
            sum_birth(CellId{20}),
            {{EdgeBirth{
                EdgeId{200}, CellId{2}, OutputPort{0}, CellId{20},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{201}, CellId{20}, OutputPort{0}, CellId{2},
                InputPort{0}, EdgeDelay::PreviousTick, 1.0}}},
            funding(CellId{2})}})
               .ok());

    const auto stepped = growth->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(!stepped.ok());
    assert(stepped.growth_error->code == GrowthErrorCode::InsufficientResource);
    assert(fixture.runtime->cell_state(CellId{1}) == nullptr);
    assert(fixture.runtime->cell_state(CellId{2}) != nullptr);
    assert(fixture.runtime->cell_state(CellId{20}) == nullptr);
    assert(stepped.lifecycle.events.end() !=
           std::find_if(
               stepped.lifecycle.events.begin(),
               stepped.lifecycle.events.end(),
               [](const auto& event) {
                   return event.kind == LifecycleEventKind::CellRemoved;
               }));
    const auto next = growth->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(next.lifecycle.error == std::nullopt);
    assert(fixture.runtime->cell_state(CellId{1}) == nullptr);
}

void test_growth_preserves_identity_parallel_edges_state_and_live_weights() {
    auto fixture = make_fixture(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N},
         {CellId{3}, CellType::OP_SUM}},
        {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{20}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::PreviousTick}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
         {CellId{3}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}});
    assert(fixture.runtime->set_parameter(
        edge_binding(*fixture.runtime, EdgeId{20}),
        ParameterValue{ContinuousValue{2.5}}).ok());
    const double first = 1.0;
    const double second = 2.0;
    assert(growth->step({&first, 1}, {}).ok());
    assert(growth->step({&second, 1}, {}).ok());
    auto control_runtime_result = fixture.runtime->fork_probe();
    assert(control_runtime_result.ok());
    auto control_runtime = std::move(control_runtime_result.runtime);
    auto control_executor_result =
        CompiledExecutor::prepare(control_runtime->plan());
    assert(control_executor_result.ok());
    auto control_executor = std::move(control_executor_result.executor);

    assert(growth->submit(GrowthProposal{
        GrowthSynapseProposal{
            7,
            EdgeBirth{
                EdgeId{21}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
                EdgeDelay::PreviousTick, 0.75},
            funding(),
            0.25}})
               .ok());
    assert(growth->step({&second, 1}, {}).ok());
    assert(control_executor->step(
        *control_runtime, std::span<const double>(&second, 1)).ok());
    const auto after = *fixture.runtime->cell_state(CellId{2});
    assert(after.state_val == control_runtime->cell_state(CellId{2})->state_val);
    assert(after.aux_state == control_runtime->cell_state(CellId{2})->aux_state);
    assert(after.prev_input == control_runtime->cell_state(CellId{2})->prev_input);
    assert(after.prev_output_val ==
           control_runtime->cell_state(CellId{2})->prev_output_val);
    assert(after.delay_buffer ==
           control_runtime->cell_state(CellId{2})->delay_buffer);
    assert(after.delay_idx == control_runtime->cell_state(CellId{2})->delay_idx);
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{20}).index))
               .value == 2.5);
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{21}).index))
               .value == 0.75);
}

void test_growth_conservation_and_failed_reservation_release() {
    auto fixture = make_fixture({{CellId{1}, CellType::OP_ABS}});
    auto growth = attach_growth(
        fixture,
        {{CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 2.0}});
    const auto before = growth->lifecycle().ledger().snapshot();
    assert(growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            8,
            sum_birth(CellId{30}),
            {{EdgeBirth{
                EdgeId{300}, CellId{1}, OutputPort{0}, CellId{30},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{301}, CellId{30}, OutputPort{0}, CellId{1},
                InputPort{0}, EdgeDelay::PreviousTick, 1.0}}},
            funding()}})
               .ok());
    const auto funded = growth->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(funded.ok());
    assert(funded.growth_report.has_value());
    assert(funded.growth_report->paid_cost > 0.0);
    assert(std::abs(funded.growth_report->conservation_residual) < 1e-9);

    auto poor = make_fixture({{CellId{1}, CellType::OP_ABS}});
    auto poor_growth = attach_growth(
        poor,
        {{CellId{1}, ResourceCompartmentId{1}, 0.1, 0.1, 0.0}});
    const auto poor_before = poor_growth->lifecycle().ledger().snapshot();
    assert(poor_growth->submit(GrowthProposal{
        GrowthCellBirthProposal{
            9,
            sum_birth(CellId{31}),
            {{EdgeBirth{
                EdgeId{310}, CellId{1}, OutputPort{0}, CellId{31},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            {{EdgeBirth{
                EdgeId{311}, CellId{31}, OutputPort{0}, CellId{1},
                InputPort{0}, EdgeDelay::Immediate, 1.0}}},
            funding()}})
               .ok());
    const auto rejected = poor_growth->step(
        std::span<const double>{}, std::span<const ResourceSupply>{});
    assert(!rejected.ok());
    assert(poor_growth->lifecycle().ledger().snapshot() == poor_before);
    assert(poor_growth->lifecycle().ledger().snapshot().cell(CellId{31}) == nullptr);
    assert(before.cell(CellId{1}) != nullptr);
}

}  // namespace

int main() {
    test_funded_birth_changes_graph_and_executes_next_step();
    test_unfunded_birth_is_atomic_and_releases_reservation();
    test_new_cell_affects_downstream_output();
    test_synapse_addition_and_reclamation_update_measurements();
    test_growth_supports_more_than_sixty_four_cells_and_edges();
    test_natural_death_commits_before_failed_growth();
    test_growth_preserves_identity_parallel_edges_state_and_live_weights();
    test_growth_conservation_and_failed_reservation_release();
    return 0;
}
