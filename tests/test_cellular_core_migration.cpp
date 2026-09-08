#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/runtime_migration.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/legacy/runtime_adapter.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

using namespace kun::core;
using namespace kun;

namespace {

GraphDefinition graph(
    GraphRevision revision,
    std::vector<CellDefinition> cells,
    std::vector<EdgeDefinition> edges = {}) {
    return GraphDefinition{
        GraphIdentity{700},
        revision,
        SemanticProfile::LegacyCompatible,
        1,
        std::move(cells),
        std::move(edges)};
}

InitialParameterSeeds seeds_for(const GraphDefinition& definition, double edge_seed = 1.0) {
    InitialParameterSeeds seeds;
    for (const auto& cell : definition.cells) {
        const auto contract = contract_for(cell.type);
        assert(contract.has_value());
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            ParameterValue value = UnusedParameter{};
            switch (descriptor.value_type) {
                case ParameterValueType::Continuous:
                    value = ContinuousValue{1.0};
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
    for (const auto& edge : definition.edges) {
        seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, edge_seed});
    }
    return seeds;
}

struct Compiled {
    std::shared_ptr<const CompiledGraph> graph;
    std::shared_ptr<const InitialParameterValues> seeds;
};

Compiled compile(const GraphDefinition& definition, double edge_seed = 1.0) {
    auto seeds = seeds_for(definition, edge_seed);
    const auto result = GraphCompiler{}.compile(definition, seeds);
    assert(result.ok());
    return {result.graph, result.initial_values};
}

Compiled compile(
    const GraphDefinition& definition,
    InitialParameterSeeds seeds) {
    const auto result = GraphCompiler{}.compile(definition, seeds);
    assert(result.ok());
    return {result.graph, result.initial_values};
}

void assert_same_state(const RuntimeCellState& lhs, const RuntimeCellState& rhs) {
    assert(lhs.cell == rhs.cell);
    assert(lhs.type == rhs.type);
    assert(lhs.state_val == rhs.state_val);
    assert(lhs.aux_state == rhs.aux_state);
    assert(lhs.prev_input == rhs.prev_input);
    assert(lhs.output_val == rhs.output_val);
    assert(lhs.prev_output_val == rhs.prev_output_val);
    assert(lhs.delay_buffer == rhs.delay_buffer);
    assert(lhs.delay_idx == rhs.delay_idx);
    assert(lhs.latch_state == rhs.latch_state);
    assert(lhs.activation_count == rhs.activation_count);
    assert(lhs.initialized == rhs.initialized);
}

const InitialParameterValue& seed_for(
    const InitialParameterValues& seeds,
    ParameterBindingKind kind,
    CellId cell,
    EdgeId edge,
    ParameterSlot slot) {
    for (const auto& entry : seeds.entries()) {
        if (entry.binding.kind == kind && entry.binding.cell == cell &&
            entry.binding.edge == edge && entry.binding.slot == slot) {
            return entry;
        }
    }
    assert(false);
    return seeds.at(0);
}

InitialParameterSeeds seeds_from_values(const InitialParameterValues& values) {
    InitialParameterSeeds seeds;
    for (const auto& entry : values.entries()) {
        if (entry.binding.kind == ParameterBindingKind::CellParameter) {
            seeds.cell_parameters.push_back(
                CellParameterSeed{entry.binding.cell, entry.binding.slot, entry.value});
        } else {
            assert(std::holds_alternative<ContinuousValue>(entry.value));
            seeds.edge_weights.push_back(
                EdgeParameterSeed{
                    entry.binding.edge,
                    std::get<ContinuousValue>(entry.value).value});
        }
    }
    return seeds;
}

void assert_same_parameter(
    const InitialParameterValue& lhs,
    const InitialParameterValue& rhs) {
    assert(lhs.binding.kind == rhs.binding.kind);
    assert(lhs.binding.index == rhs.binding.index);
    assert(lhs.binding.cell == rhs.binding.cell);
    assert(lhs.binding.edge == rhs.binding.edge);
    assert(lhs.binding.slot == rhs.binding.slot);
    assert(lhs.value.index() == rhs.value.index());
    std::visit(
        [](const auto& left, const auto& right) {
            using Left = std::decay_t<decltype(left)>;
            using Right = std::decay_t<decltype(right)>;
            if constexpr (std::is_same_v<Left, Right>) {
                if constexpr (std::is_same_v<Left, ContinuousValue>) {
                    assert(left.value == right.value);
                } else if constexpr (std::is_same_v<Left, ChannelIndex>) {
                    assert(left.value == right.value);
                } else if constexpr (std::is_same_v<Left, DelayTicks>) {
                    assert(left.value == right.value);
                } else if constexpr (std::is_same_v<Left, MinMaxMode>) {
                    assert(left == right);
                }
            } else {
                assert(false);
            }
        },
        lhs.value, rhs.value);
}

void assert_same_snapshot(
    const RuntimeSnapshot& lhs,
    const RuntimeSnapshot& rhs) {
    assert(lhs.plan());
    assert(rhs.plan());
    assert(detail::same_plan(*lhs.plan(), *rhs.plan()));
    assert(lhs.tick() == rhs.tick());
    assert(lhs.parameters().size() == rhs.parameters().size());
    for (std::size_t i = 0; i < lhs.parameters().size(); ++i) {
        assert_same_parameter(lhs.parameters()[i], rhs.parameters()[i]);
    }
    assert(lhs.cells().size() == rhs.cells().size());
    for (std::size_t i = 0; i < lhs.cells().size(); ++i) {
        assert_same_state(lhs.cells()[i], rhs.cells()[i]);
    }
}

const InitialParameterValue& parameter_for_edge(
    const RuntimeState& runtime,
    EdgeId edge) {
    for (const auto& value : runtime.parameters()) {
        if (value.binding.kind == ParameterBindingKind::EdgeWeight &&
            value.binding.edge == edge) {
            return value;
        }
    }
    assert(false);
    return runtime.parameters()[0];
}

std::shared_ptr<CellularOrganism> make_oja_source() {
    auto source = std::make_shared<CellularOrganism>();
    Cell raw;
    raw.id = 0;
    raw.type = CellType::SENSE_RAW_INPUT_0;
    raw.param1 = 1.0;
    Cell delay;
    delay.id = 1;
    delay.type = CellType::OP_DELAY_N;
    delay.param1 = 1.0;
    Cell ema;
    ema.id = 2;
    ema.type = CellType::OP_EMA;
    ema.param1 = 0.5;
    Cell diff;
    diff.id = 3;
    diff.type = CellType::OP_DIFF;
    Cell latch;
    latch.id = 4;
    latch.type = CellType::GATE_HYSTERESIS;
    latch.param1 = 0.5;
    latch.param2 = -0.5;
    Cell action;
    action.id = 5;
    action.type = CellType::ACT_PRIMARY_POSITIVE;
    source->cells = {raw, delay, ema, diff, latch, action};

    Synapse learned;
    learned.from_cell_id = 0;
    learned.to_cell_id = 1;
    learned.weight = 1.0;
    learned.initial_weight = 1.0;
    learned.hebbian_rate = 0.1;
    learned.hebbian_decay = 0.02;
    Synapse fixed = learned;
    fixed.weight = 0.6;
    fixed.initial_weight = 0.5;
    fixed.hebbian_rate = 0.0;
    source->synapses = {
        learned,
        fixed,
        Synapse{1, 2, 0, 1.0},
        Synapse{2, 3, 0, 1.0},
        Synapse{3, 4, 0, 1.0},
        Synapse{4, 5, 0, 1.0}};
    assert(source->compile());
    return source;
}

void test_noop_rebind_preserves_state_and_live_values() {
    const auto definition = graph(
        GraphRevision{1},
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_DELAY_N},
         {CellId{30}, CellType::OP_EMA},
         {CellId{40}, CellType::OP_DIFF},
         {CellId{50}, CellType::GATE_HYSTERESIS}},
        {{EdgeId{100}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{101}, CellId{20}, OutputPort{0}, CellId{30}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{102}, CellId{30}, OutputPort{0}, CellId{40}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{103}, CellId{40}, OutputPort{0}, CellId{50}, InputPort{0},
          EdgeDelay::Immediate}});
    auto old = compile(definition);
    auto created = RuntimeState::create(old.graph, old.seeds);
    assert(created.ok());
    auto runtime = std::move(created.runtime);
    const auto edge_index = old.graph->edges()[0].weight_parameter_index;
    assert(runtime->set_parameter(
                      runtime->parameters()[edge_index].binding,
                      ParameterValue{ContinuousValue{2.5}})
               .ok());

    ReferenceExecutor executor;
    for (double input : {1.0, 2.0, 3.0, 4.0}) {
        assert(executor.step(*runtime, std::span<const double>(&input, 1)).ok());
    }
    const auto before = runtime->snapshot();
    auto continuation = runtime->fork_probe();
    assert(continuation.ok());

    auto rebuilt = compile(definition, 99.0);
    const auto result = RuntimeMigration::rebind(*runtime, rebuilt.graph, rebuilt.seeds);
    assert(result.ok());
    assert(result.report.preserved_cells.size() == definition.cells.size());
    assert(result.report.preserved_edges.size() == definition.edges.size());
    assert(runtime->bound_to(*rebuilt.graph));
    assert(runtime->tick() == before.tick());
    for (std::size_t i = 0; i < before.cells().size(); ++i) {
        assert_same_state(runtime->cell_states()[i], before.cells()[i]);
    }
    assert(std::get<ContinuousValue>(*runtime->parameter_at(edge_index)).value == 2.5);
    assert(std::get<ContinuousValue>(
               seed_for(*rebuilt.seeds, ParameterBindingKind::EdgeWeight,
                        CellId{}, EdgeId{100}, ParameterSlot::Param1).value)
               .value == 99.0);
    assert(std::get<ContinuousValue>(
               seed_for(*old.seeds, ParameterBindingKind::EdgeWeight,
                        CellId{}, EdgeId{100}, ParameterSlot::Param1).value)
               .value == 1.0);

    for (double input : {5.0, 6.0, 7.0}) {
        assert(executor.step(*runtime, std::span<const double>(&input, 1)).ok());
        assert(executor.step(
                   *continuation.runtime, std::span<const double>(&input, 1))
                   .ok());
        assert(runtime->tick() == continuation.runtime->tick());
        for (std::size_t i = 0; i < runtime->cell_states().size(); ++i) {
            assert_same_state(runtime->cell_states()[i],
                              continuation.runtime->cell_states()[i]);
        }
    }
    assert(std::get<ContinuousValue>(*runtime->parameter_at(edge_index)).value == 2.5);
}

void test_imported_oja_source_survives_noop_rebind_and_restore() {
    const auto source = make_oja_source();
    for (double input = 1.0; input <= 20.0; input += 1.0) {
        source->forward_nd(&input, 1, true);
    }
    const double source_live_weight = source->compiled_synapses_[0].weight;
    assert(source_live_weight != 1.0);
    assert(source->compiled_synapses_[0].initial_weight == 1.0);

    const auto imported = migration::import_execution_snapshot(
        *source, GraphIdentity{700}, GraphRevision{1});
    assert(imported.ok());
    const auto& snapshot = *imported.snapshot;
    assert(!snapshot.edge_provenance().empty());
    assert(snapshot.edge_provenance()[0].raw_declared_initial_weight == 1.0);
    assert(snapshot.edge_provenance()[0].live_weight != 1.0);

    const auto runtime_result =
        migration::import_execution_snapshot_runtime(snapshot);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    const auto cold_values = std::vector<InitialParameterValue>(
        snapshot.current_parameter_values()->entries().begin(),
        snapshot.current_parameter_values()->entries().end());
    const auto imported_before = runtime->snapshot();
    auto reference_result = runtime->fork_probe();
    assert(reference_result.ok());
    auto reference = std::move(reference_result.runtime);

    const auto rebuilt = compile(
        snapshot.graph_definition(),
        seeds_from_values(*snapshot.current_parameter_values()));
    const auto migration = RuntimeMigration::rebind(
        *runtime, rebuilt.graph, rebuilt.seeds);
    assert(migration.ok());
    assert_same_snapshot(runtime->snapshot(), imported_before);
    for (const auto& value : snapshot.current_parameter_values()->entries()) {
        assert_same_parameter(
            value,
            cold_values[value.binding.index]);
    }
    for (const auto& value : runtime->parameters()) {
        assert_same_parameter(value, cold_values[value.binding.index]);
    }
    assert(std::get<ContinuousValue>(
               parameter_for_edge(*runtime, snapshot.edge_provenance()[0].edge)
                   .value)
               .value == source_live_weight);

    ReferenceExecutor executor;
    for (double input : {21.0, 22.0, 23.0, 24.0}) {
        assert(executor.step(*runtime, std::span<const double>(&input, 1)).ok());
        assert(executor.step(*reference, std::span<const double>(&input, 1)).ok());
        assert_same_snapshot(runtime->snapshot(), reference->snapshot());
    }
    const auto continuation = runtime->snapshot();
    assert(runtime->reset_episode().ok());
    for (std::size_t i = 0; i < runtime->parameters().size(); ++i) {
        assert_same_parameter(runtime->parameters()[i], continuation.parameters()[i]);
    }
    assert(runtime->restore_snapshot(continuation).ok());
    assert_same_snapshot(runtime->snapshot(), continuation);
    assert(snapshot.edge_provenance()[0].raw_declared_initial_weight == 1.0);
    assert(source->compiled_synapses_[0].initial_weight == 1.0);
    assert(source->compiled_synapses_[0].weight == source_live_weight);
}

void test_stable_ids_map_cells_and_parallel_edges() {
    const auto initial = graph(
        GraphRevision{1},
        {{CellId{1}, CellType::OP_SUM},
         {CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_DELAY_N},
         {CellId{30}, CellType::OP_EMA}},
        {{EdgeId{100}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{200}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{201}, CellId{20}, OutputPort{0}, CellId{30}, InputPort{0},
          EdgeDelay::Immediate}});
    auto initial_seeds = seeds_for(initial);
    for (auto& seed : initial_seeds.edge_weights) {
        if (seed.edge == EdgeId{100}) seed.initial_weight = 1.5;
        if (seed.edge == EdgeId{200}) seed.initial_weight = 2.5;
    }
    auto old = compile(initial, initial_seeds);
    auto created = RuntimeState::create(old.graph, old.seeds);
    assert(created.ok());
    auto runtime = std::move(created.runtime);
    for (double input : {1.0, 2.0, 3.0, 4.0}) {
        assert(ReferenceExecutor{}.step(
                   *runtime, std::span<const double>(&input, 1))
                   .ok());
    }
    const auto retained_delay = *runtime->cell_state(CellId{20});
    const auto retained_ema = *runtime->cell_state(CellId{30});
    assert(retained_delay.initialized);
    assert(retained_delay.delay_idx != 0);
    assert(std::any_of(
        retained_delay.delay_buffer.begin(), retained_delay.delay_buffer.end(),
        [](double value) { return value != 0.0; }));
    auto expected = runtime->fork_probe();
    assert(expected.ok());

    auto changed = initial;
    changed.revision = GraphRevision{2};
    changed.cells.erase(changed.cells.begin());
    changed.edges = {
        {EdgeId{300}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{200}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{201}, CellId{20}, OutputPort{0}, CellId{30}, InputPort{0},
         EdgeDelay::Immediate}};
    auto changed_seeds = seeds_for(changed, 7.0);
    auto rebuilt = compile(changed, changed_seeds);
    const auto migrated = RuntimeMigration::rebind(*runtime, rebuilt.graph, rebuilt.seeds);
    assert(migrated.ok());
    assert(RuntimeMigration::rebind(
               *expected.runtime, rebuilt.graph, rebuilt.seeds)
               .ok());
    assert(runtime->cell_state(CellId{1}) == nullptr);
    assert_same_state(*runtime->cell_state(CellId{20}), retained_delay);
    assert_same_state(*runtime->cell_state(CellId{30}), retained_ema);
    assert(std::find(migrated.report.preserved_edges.begin(),
                     migrated.report.preserved_edges.end(),
                     EdgeId{200}) != migrated.report.preserved_edges.end());
    assert(std::find(migrated.report.new_edges.begin(),
                     migrated.report.new_edges.end(),
                     EdgeId{300}) != migrated.report.new_edges.end());
    assert(std::find(migrated.report.removed_edges.begin(),
                     migrated.report.removed_edges.end(),
                     EdgeId{100}) != migrated.report.removed_edges.end());
    assert(std::get<ContinuousValue>(
               parameter_for_edge(*runtime, EdgeId{200}).value)
               .value == 2.5);
    assert(std::get<ContinuousValue>(
               parameter_for_edge(*runtime, EdgeId{300}).value)
               .value == 7.0);
    const double migrated_input = 5.0;
    assert(ReferenceExecutor{}.step(
               *runtime, std::span<const double>(&migrated_input, 1))
               .ok());
    assert(ReferenceExecutor{}.step(
               *expected.runtime, std::span<const double>(&migrated_input, 1))
               .ok());
    assert_same_snapshot(runtime->snapshot(), expected.runtime->snapshot());
}

void test_added_type_changed_and_discrete_changed_cells_reset_only_themselves() {
    auto initial = graph(
        GraphRevision{1},
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_EMA},
         {CellId{30}, CellType::SENSE_CHANNEL}},
        {{EdgeId{10}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate}});
    auto old = compile(initial);
    auto created = RuntimeState::create(old.graph, old.seeds);
    assert(created.ok());
    auto runtime = std::move(created.runtime);
    for (double input : {1.0, 2.0, 3.0}) {
        assert(ReferenceExecutor{}.step(
                   *runtime, std::span<const double>(&input, 1))
                   .ok());
    }
    const auto raw_before = *runtime->cell_state(CellId{10});
    const auto ema_before = *runtime->cell_state(CellId{20});
    const auto channel_before = *runtime->cell_state(CellId{30});
    assert(ema_before.initialized);
    assert(ema_before.state_val != 0.0);
    assert(channel_before.initialized);
    assert(channel_before.output_val == 3.0);

    auto changed = initial;
    changed.revision = GraphRevision{2};
    changed.cells = {
        {CellId{10}, CellType::SENSE_RAW_INPUT_0},
        {CellId{20}, CellType::OP_DIFF},
        {CellId{30}, CellType::SENSE_CHANNEL},
        {CellId{40}, CellType::OP_SUM}};
    auto rebuilt = compile(changed);
    const auto migrated = RuntimeMigration::rebind(*runtime, rebuilt.graph, rebuilt.seeds);
    assert(migrated.ok());
    assert_same_state(*runtime->cell_state(CellId{10}), raw_before);
    assert(!runtime->cell_state(CellId{20})->initialized);
    assert(runtime->cell_state(CellId{20})->state_val == 0.0);
    assert_same_state(*runtime->cell_state(CellId{30}), channel_before);
    assert(!runtime->cell_state(CellId{40})->initialized);
    assert(std::find(migrated.report.reset_cells.begin(),
                     migrated.report.reset_cells.end(),
                     CellId{20}) != migrated.report.reset_cells.end());
    assert(std::find(migrated.report.new_cells.begin(),
                     migrated.report.new_cells.end(),
                     CellId{40}) != migrated.report.new_cells.end());

    auto same_revision_discrete = changed;
    auto same_revision_seeds = seeds_for(same_revision_discrete);
    for (auto& seed : same_revision_seeds.cell_parameters) {
        if (seed.cell == CellId{30} && seed.slot == ParameterSlot::Param2) {
            seed.value = ChannelIndex{3};
        }
    }
    const auto same_revision_compiled = compile(
        same_revision_discrete, same_revision_seeds);
    const auto before_same_revision_rejection = runtime->snapshot();
    assert(!RuntimeMigration::rebind(
                    *runtime,
                    same_revision_compiled.graph,
                    same_revision_compiled.seeds)
                .ok());
    assert_same_snapshot(
        runtime->snapshot(), before_same_revision_rejection);

    auto discrete = changed;
    discrete.revision = GraphRevision{3};
    auto discrete_seeds = seeds_for(discrete);
    for (auto& seed : discrete_seeds.cell_parameters) {
        if (seed.cell == CellId{30} && seed.slot == ParameterSlot::Param2) {
            seed.value = ChannelIndex{3};
        }
    }
    const auto discrete_compiled = GraphCompiler{}.compile(discrete, discrete_seeds);
    assert(discrete_compiled.ok());
    const auto old_channel = *runtime->cell_state(CellId{30});
    const auto discrete_result = RuntimeMigration::rebind(
        *runtime, discrete_compiled.graph, discrete_compiled.initial_values);
    assert(discrete_result.ok());
    assert(!runtime->cell_state(CellId{30})->initialized);
    assert(runtime->cell_state(CellId{30})->output_val == 0.0);
    assert(std::get<ChannelIndex>(
               runtime->parameter(
                   CellId{30}, ParameterSlot::Param2).value())
               .value == 3);
    const auto& details = migrated.report.cell_reset_details;
    assert(std::find_if(
               details.begin(), details.end(),
               [](const auto& detail) {
                   return detail.cell == CellId{20} &&
                          detail.reason == MigrationResetReason::TypeChanged;
               }) != details.end());
    assert(std::find_if(
               discrete_result.report.cell_reset_details.begin(),
               discrete_result.report.cell_reset_details.end(),
               [](const auto& detail) {
                   return detail.cell == CellId{30} &&
                          detail.reason ==
                              MigrationResetReason::DiscreteConfigurationChanged;
               }) != discrete_result.report.cell_reset_details.end());
    assert(!runtime->cell_state(CellId{30})->initialized);
    assert(runtime->cell_state(CellId{30})->output_val == 0.0);
    assert(old_channel.cell == CellId{30});
}

void test_revision_identity_and_invalid_input_reject_atomically() {
    const auto definition = graph(
        GraphRevision{3},
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_SUM},
         {CellId{30}, CellType::OP_SUM}},
        {{EdgeId{1}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate}});
    auto compiled = compile(definition);
    auto created = RuntimeState::create(compiled.graph, compiled.seeds);
    assert(created.ok());
    auto runtime = std::move(created.runtime);
    const double input = 4.0;
    assert(ReferenceExecutor{}.step(*runtime, std::span<const double>(&input, 1)).ok());

    auto rewired = definition;
    rewired.edges[0].target = CellId{30};
    const auto same_revision = compile(rewired);
    const auto before_same_revision = runtime->snapshot();
    assert(!RuntimeMigration::rebind(*runtime, same_revision.graph, same_revision.seeds).ok());
    assert_same_snapshot(runtime->snapshot(), before_same_revision);
    assert(runtime->bound_to(*compiled.graph));

    auto advanced = rewired;
    advanced.revision = GraphRevision{4};
    const auto advanced_compiled = compile(advanced, 3.0);
    const auto advanced_migration = RuntimeMigration::rebind(
        *runtime, advanced_compiled.graph, advanced_compiled.seeds);
    assert(advanced_migration.ok());
    assert(!runtime->bound_to(*compiled.graph));
    assert(std::find(
               advanced_migration.report.reset_edges.begin(),
               advanced_migration.report.reset_edges.end(),
               EdgeId{1}) != advanced_migration.report.reset_edges.end());
    assert(std::find_if(
               advanced_migration.report.edge_reset_details.begin(),
               advanced_migration.report.edge_reset_details.end(),
               [](const auto& detail) {
                   return detail.edge == EdgeId{1} &&
                          detail.reason ==
                              MigrationResetReason::EdgeDefinitionChanged;
               }) != advanced_migration.report.edge_reset_details.end());
    assert(std::get<ContinuousValue>(
               parameter_for_edge(*runtime, EdgeId{1}).value)
               .value == 3.0);
    const double rewired_input = 2.0;
    assert(ReferenceExecutor{}.step(
               *runtime, std::span<const double>(&rewired_input, 1))
               .ok());
    assert(runtime->cell_state(CellId{20})->output_val == 0.0);
    assert(runtime->cell_state(CellId{30})->output_val == 6.0);
    const auto after_reused_edge_step = runtime->snapshot();
    const auto noop_rebind = RuntimeMigration::rebind(
        *runtime, advanced_compiled.graph, advanced_compiled.seeds);
    assert(noop_rebind.ok());
    assert_same_snapshot(runtime->snapshot(), after_reused_edge_step);

    auto foreign = advanced;
    foreign.identity = GraphIdentity{999};
    foreign.revision = GraphRevision{5};
    const auto foreign_compiled = compile(foreign);
    const auto stable = runtime->snapshot();
    assert(!RuntimeMigration::rebind(
                    *runtime, foreign_compiled.graph, foreign_compiled.seeds)
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);
    assert(runtime->bound_to(*advanced_compiled.graph));

    const auto regressed = compile(definition);
    assert(!RuntimeMigration::rebind(
                    *runtime, regressed.graph, regressed.seeds)
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);
    assert(runtime->bound_to(*advanced_compiled.graph));

    auto strict_definition = advanced;
    strict_definition.profile = SemanticProfile::StrictCore;
    const auto strict_compiled =
        GraphCompiler{}.compile(strict_definition, seeds_for(strict_definition));
    assert(strict_compiled.ok());
    assert(!RuntimeMigration::rebind(
                    *runtime, strict_compiled.graph, strict_compiled.initial_values)
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);
    assert(runtime->bound_to(*advanced_compiled.graph));

    auto invalid_values = std::vector<InitialParameterValue>(
        advanced_compiled.seeds->entries().begin(),
        advanced_compiled.seeds->entries().end());
    for (auto& value : invalid_values) {
        if (value.binding.cell == CellId{10} &&
            value.binding.slot == ParameterSlot::Param1) {
            value.value = ContinuousValue{
                std::numeric_limits<double>::quiet_NaN()};
        }
    }
    assert(!RuntimeMigration::rebind(
                    *runtime,
                    advanced_compiled.graph,
                    std::make_shared<InitialParameterValues>(invalid_values))
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);

    invalid_values = std::vector<InitialParameterValue>(
        advanced_compiled.seeds->entries().begin(),
        advanced_compiled.seeds->entries().end());
    for (auto& value : invalid_values) {
        if (value.binding.cell == CellId{10} &&
            value.binding.slot == ParameterSlot::Param1) {
            value.value = ContinuousValue{
                std::numeric_limits<double>::infinity()};
        }
    }
    assert(!RuntimeMigration::rebind(
                    *runtime,
                    advanced_compiled.graph,
                    std::make_shared<InitialParameterValues>(invalid_values))
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);

    invalid_values = std::vector<InitialParameterValue>(
        advanced_compiled.seeds->entries().begin(),
        advanced_compiled.seeds->entries().end());
    for (auto& value : invalid_values) {
        if (value.binding.cell == CellId{10} &&
            value.binding.slot == ParameterSlot::Param1) {
            value.value = ChannelIndex{1};
        }
    }
    assert(!RuntimeMigration::rebind(
                    *runtime,
                    advanced_compiled.graph,
                    std::make_shared<InitialParameterValues>(invalid_values))
                .ok());
    assert_same_snapshot(runtime->snapshot(), stable);
}

void test_empty_graph_and_reset_snapshot_boundaries() {
    const auto initial = graph(
        GraphRevision{1},
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0}});
    auto old = compile(initial);
    auto created = RuntimeState::create(old.graph, old.seeds);
    assert(created.ok());
    auto runtime = std::move(created.runtime);
    assert(runtime->set_parameter(
                      runtime->parameters()[0].binding,
                      ParameterValue{ContinuousValue{2.5}})
               .ok());
    const auto empty = graph(GraphRevision{2}, {});
    auto empty_compiled = compile(empty);
    assert(RuntimeMigration::rebind(
               *runtime, empty_compiled.graph, empty_compiled.seeds)
               .ok());
    assert(runtime->cell_states().empty());
    const auto empty_snapshot = runtime->snapshot();

    const auto restored = runtime->restore_snapshot(empty_snapshot);
    assert(restored.ok());
    const auto fresh = graph(
        GraphRevision{3},
        {{CellId{20}, CellType::SENSE_RAW_INPUT_0}});
    auto fresh_compiled = compile(fresh);
    assert(RuntimeMigration::rebind(
               *runtime, fresh_compiled.graph, fresh_compiled.seeds)
               .ok());
    assert(runtime->cell_state(CellId{20}) != nullptr);
    assert(!runtime->cell_state(CellId{20})->initialized);

    const auto checkpoint = runtime->snapshot();
    assert(runtime->reset_episode().ok());
    assert(runtime->restore_snapshot(checkpoint).ok());
    assert(std::get<ContinuousValue>(*runtime->parameter_at(0)).value == 1.0);

    auto probe = runtime->fork_probe();
    assert(probe.ok());
    auto next = fresh;
    next.revision = GraphRevision{4};
    next.cells.push_back({CellId{30}, CellType::OP_SUM});
    auto next_compiled = compile(next);
    assert(RuntimeMigration::rebind(
               *probe.runtime, next_compiled.graph, next_compiled.seeds)
               .ok());
    assert(!runtime->restore_snapshot(probe.runtime->snapshot()).ok());
}

}  // namespace

int main() {
    test_noop_rebind_preserves_state_and_live_values();
    test_imported_oja_source_survives_noop_rebind_and_restore();
    test_stable_ids_map_cells_and_parallel_edges();
    test_added_type_changed_and_discrete_changed_cells_reset_only_themselves();
    test_revision_identity_and_invalid_input_reject_atomically();
    test_empty_graph_and_reset_snapshot_boundaries();
    return 0;
}
