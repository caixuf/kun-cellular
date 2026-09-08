#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/runtime_migration.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/legacy/runtime_adapter.hpp"
#include "kun/cellular/cellular_genome.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

using namespace kun;
using namespace kun::core;

namespace {

std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocation_count{0};

void* allocate_or_throw(std::size_t size) {
    if (void* result = std::malloc(size == 0 ? 1 : size)) {
        if (count_allocations.load(std::memory_order_relaxed)) {
            allocation_count.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    throw std::bad_alloc();
}

void* allocate_aligned_or_throw(
    std::size_t size,
    std::align_val_t alignment) {
    const std::size_t align = static_cast<std::size_t>(alignment);
    const std::size_t rounded =
        ((size == 0 ? 1 : size) + align - 1) / align * align;
    if (void* result = std::aligned_alloc(align, rounded)) {
        if (count_allocations.load(std::memory_order_relaxed)) {
            allocation_count.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    throw std::bad_alloc();
}

}  // namespace

void* operator new(std::size_t size) { return allocate_or_throw(size); }
void* operator new[](std::size_t size) { return allocate_or_throw(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate_aligned_or_throw(size, alignment);
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocate_aligned_or_throw(size, alignment);
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete(
    void* pointer,
    std::size_t,
    std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](
    void* pointer,
    std::size_t,
    std::align_val_t) noexcept {
    std::free(pointer);
}

namespace {

struct AllocationScope {
    AllocationScope() {
        allocation_count.store(0, std::memory_order_relaxed);
        count_allocations.store(true, std::memory_order_relaxed);
    }
    ~AllocationScope() {
        count_allocations.store(false, std::memory_order_relaxed);
    }
    std::size_t count() const {
        return allocation_count.load(std::memory_order_relaxed);
    }
};

InitialParameterSeeds seeds_for(
    const GraphDefinition& graph,
    std::size_t channel_index = 1) {
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
                        descriptor.name && std::strcmp(descriptor.name, "alpha") == 0
                            ? 0.5
                            : 1.0};
                    break;
                case ParameterValueType::ChannelIndex:
                    value = ChannelIndex{channel_index};
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

GraphDefinition all30_graph(SemanticProfile profile, GraphRevision revision = GraphRevision{1}) {
    GraphDefinition graph;
    graph.identity = GraphIdentity{700};
    graph.revision = revision;
    graph.profile = profile;
    for (const auto& contract : all_cell_contracts()) {
        graph.cells.push_back(CellDefinition{
            CellId{static_cast<uint64_t>(100 + contract.code)}, contract.type});
    }
    uint64_t edge_id = 1;
    const std::array<CellId, 5> sensors{
        CellId{100}, CellId{101}, CellId{102}, CellId{103}, CellId{104}};
    for (std::size_t index = 0; index < graph.cells.size(); ++index) {
        const auto ports =
            contract_for(graph.cells[index].type)->get().input_port_count;
        if (ports > 0) {
            graph.edges.push_back(EdgeDefinition{
                EdgeId{edge_id++},
                sensors[(index + 1) % sensors.size()],
                OutputPort{0},
                graph.cells[index].id,
                InputPort{0},
                EdgeDelay::Immediate});
        }
        if (ports > 1) {
            graph.edges.push_back(EdgeDefinition{
                EdgeId{edge_id++},
                sensors[(index + 2) % sensors.size()],
                OutputPort{0},
                graph.cells[index].id,
                InputPort{1},
                EdgeDelay::PreviousTick});
        }
    }
    return graph;
}

GraphDefinition ema_graph(
    SemanticProfile profile,
    GraphRevision revision = GraphRevision{1},
    bool alternate = false) {
    GraphDefinition graph;
    graph.identity = GraphIdentity{701};
    graph.revision = revision;
    graph.profile = profile;
    graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_EMA},
        {CellId{3}, CellType::OP_DELAY_N},
    };
    graph.edges = {
        {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{2}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{3}, alternate ? CellId{3} : CellId{2}, OutputPort{0},
         CellId{3}, InputPort{0}, EdgeDelay::PreviousTick},
    };
    return graph;
}

InitialParameterSeeds ema_seeds(
    const GraphDefinition& graph,
    double first_weight = 1.0,
    std::size_t channel_index = 1) {
    auto seeds = seeds_for(graph, channel_index);
    for (auto& edge : seeds.edge_weights) {
        edge.initial_weight =
            edge.edge == EdgeId{1} ? first_weight
            : (edge.edge == EdgeId{2} ? 0.0 : 1.0);
    }
    return seeds;
}

struct CompiledFixture {
    CompileResult compiled;
    std::shared_ptr<RuntimeState> reference;
    std::shared_ptr<RuntimeState> optimized;
    std::shared_ptr<CompiledExecutor> executor;
};

std::shared_ptr<RuntimeState> make_imported_runtime(
    std::shared_ptr<const CompiledGraph> plan,
    std::shared_ptr<const InitialParameterValues> values,
    std::span<const RuntimeCellState> cells);

struct ImportedExecutorFixture {
    CompileResult compiled;
    std::shared_ptr<RuntimeState> reference;
    std::shared_ptr<RuntimeState> optimized;
    std::shared_ptr<CompiledExecutor> executor;
};

CompiledFixture make_fixture(
    const GraphDefinition& graph,
    const InitialParameterSeeds& seeds) {
    CompiledFixture result;
    result.compiled = GraphCompiler{}.compile(graph, seeds);
    assert(result.compiled.ok());
    auto reference = RuntimeState::create(
        result.compiled.graph, result.compiled.initial_values);
    auto optimized = RuntimeState::create(
        result.compiled.graph, result.compiled.initial_values);
    assert(reference.ok());
    assert(optimized.ok());
    result.reference = std::move(reference.runtime);
    result.optimized = std::move(optimized.runtime);
    const auto prepared = CompiledExecutor::prepare(result.compiled.graph);
    assert(prepared.ok());
    result.executor = std::move(prepared.executor);
    return result;
}

ImportedExecutorFixture make_imported_fixture(
    const GraphDefinition& graph,
    const InitialParameterSeeds& seeds,
    std::span<const RuntimeCellState> states) {
    ImportedExecutorFixture result;
    result.compiled = GraphCompiler{}.compile(graph, seeds);
    assert(result.compiled.ok());
    result.optimized = make_imported_runtime(
        result.compiled.graph, result.compiled.initial_values, states);
    const auto reference_result = result.optimized->fork_probe();
    assert(reference_result.ok());
    result.reference = std::move(reference_result.runtime);
    const auto prepared = CompiledExecutor::prepare(result.compiled.graph);
    assert(prepared.ok());
    result.executor = std::move(prepared.executor);
    return result;
}

std::size_t edge_parameter_index(
    const RuntimeState& runtime,
    EdgeId edge) {
    for (const auto& parameter : runtime.parameters()) {
        if (parameter.binding.kind == ParameterBindingKind::EdgeWeight &&
            parameter.binding.edge == edge) {
            return parameter.binding.index;
        }
    }
    assert(false);
    return 0;
}

void set_edge_weight(RuntimeState& runtime, EdgeId edge, double weight) {
    const auto index = edge_parameter_index(runtime, edge);
    assert(runtime.set_parameter(
        runtime.parameters()[index].binding,
        ParameterValue{ContinuousValue{weight}})
               .ok());
}

void set_cell_continuous(
    RuntimeState& runtime,
    CellId cell,
    ParameterSlot slot,
    double value) {
    for (const auto& parameter : runtime.parameters()) {
        if (parameter.binding.kind == ParameterBindingKind::CellParameter &&
            parameter.binding.cell == cell &&
            parameter.binding.slot == slot) {
            assert(runtime.set_parameter(
                parameter.binding,
                ParameterValue{ContinuousValue{value}})
                       .ok());
            return;
        }
    }
    assert(false);
}

void assert_cell_equal(
    const RuntimeCellState& lhs,
    const RuntimeCellState& rhs) {
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

void assert_parameter_value_equal(
    const ParameterValue& lhs,
    const ParameterValue& rhs) {
    assert(lhs.index() == rhs.index());
    std::visit(
        [&rhs](const auto& value) {
            using Value = std::decay_t<decltype(value)>;
            const auto& other = std::get<Value>(rhs);
            if constexpr (std::is_same_v<Value, ContinuousValue>) {
                assert(value.value == other.value);
            } else if constexpr (std::is_same_v<Value, ChannelIndex>) {
                assert(value.value == other.value);
            } else if constexpr (std::is_same_v<Value, DelayTicks>) {
                assert(value.value == other.value);
            } else if constexpr (std::is_same_v<Value, MinMaxMode>) {
                assert(value == other);
            }
        },
        lhs);
}

void assert_runtime_equal(
    const RuntimeState& lhs,
    const RuntimeState& rhs) {
    assert(lhs.bound_to(*rhs.plan()));
    assert(lhs.tick() == rhs.tick());
    assert(lhs.parameters().size() == rhs.parameters().size());
    for (std::size_t index = 0; index < lhs.parameters().size(); ++index) {
        assert(lhs.parameters()[index].binding.kind ==
               rhs.parameters()[index].binding.kind);
        assert(lhs.parameters()[index].binding.index ==
               rhs.parameters()[index].binding.index);
        assert(lhs.parameters()[index].binding.cell ==
               rhs.parameters()[index].binding.cell);
        assert(lhs.parameters()[index].binding.edge ==
               rhs.parameters()[index].binding.edge);
        assert(lhs.parameters()[index].binding.slot ==
               rhs.parameters()[index].binding.slot);
        assert_parameter_value_equal(
            lhs.parameters()[index].value, rhs.parameters()[index].value);
    }
    assert(lhs.cell_states().size() == rhs.cell_states().size());
    for (std::size_t index = 0; index < lhs.cell_states().size(); ++index) {
        assert_cell_equal(lhs.cell_states()[index], rhs.cell_states()[index]);
    }
}

void assert_snapshot_equal(
    const RuntimeSnapshot& lhs,
    const RuntimeSnapshot& rhs) {
    assert(lhs.plan() != nullptr);
    assert(rhs.plan() != nullptr);
    assert(detail::same_plan(*lhs.plan(), *rhs.plan()));
    assert(lhs.tick() == rhs.tick());
    assert(lhs.parameters().size() == rhs.parameters().size());
    for (std::size_t index = 0; index < lhs.parameters().size(); ++index) {
        assert(lhs.parameters()[index].binding.kind ==
               rhs.parameters()[index].binding.kind);
        assert(lhs.parameters()[index].binding.index ==
               rhs.parameters()[index].binding.index);
        assert(lhs.parameters()[index].binding.cell ==
               rhs.parameters()[index].binding.cell);
        assert(lhs.parameters()[index].binding.edge ==
               rhs.parameters()[index].binding.edge);
        assert(lhs.parameters()[index].binding.slot ==
               rhs.parameters()[index].binding.slot);
        assert_parameter_value_equal(
            lhs.parameters()[index].value, rhs.parameters()[index].value);
    }
    assert(lhs.cells().size() == rhs.cells().size());
    for (std::size_t index = 0; index < lhs.cells().size(); ++index) {
        assert_cell_equal(lhs.cells()[index], rhs.cells()[index]);
    }
}

template <typename ActualMeasurement>
void assert_measurement_equal(
    const ExecutionMeasurement& reference,
    const ActualMeasurement& optimized) {
    assert(reference.tick == optimized.tick);
    assert(reference.identity == optimized.identity);
    assert(reference.revision == optimized.revision);
    assert(reference.profile == optimized.profile);
    assert(reference.cells.size() == optimized.cells.size());
    assert(reference.ports.size() == optimized.ports.size());
    assert(reference.edges.size() == optimized.edges.size());
    for (std::size_t index = 0; index < reference.cells.size(); ++index) {
        assert(reference.cells[index].cell == optimized.cells[index].cell);
        assert(reference.cells[index].type == optimized.cells[index].type);
        assert(reference.cells[index].executed ==
               optimized.cells[index].executed);
        assert(reference.cells[index].output == optimized.cells[index].output);
    }
    for (std::size_t index = 0; index < reference.ports.size(); ++index) {
        assert(reference.ports[index].cell == optimized.ports[index].cell);
        assert(reference.ports[index].port == optimized.ports[index].port);
        assert(reference.ports[index].reduced_input ==
               optimized.ports[index].reduced_input);
    }
    for (std::size_t index = 0; index < reference.edges.size(); ++index) {
        assert(reference.edges[index].edge == optimized.edges[index].edge);
        assert(reference.edges[index].source == optimized.edges[index].source);
        assert(reference.edges[index].target == optimized.edges[index].target);
        assert(reference.edges[index].target_port ==
               optimized.edges[index].target_port);
        assert(reference.edges[index].source_mode ==
               optimized.edges[index].source_mode);
        assert(reference.edges[index].source_value ==
               optimized.edges[index].source_value);
        assert(reference.edges[index].contribution ==
               optimized.edges[index].contribution);
    }
}

ExecutionMeasurement copy_measurement(const ExecutionMeasurementView& view) {
    ExecutionMeasurement copy;
    copy.tick = view.tick;
    copy.identity = view.identity;
    copy.revision = view.revision;
    copy.profile = view.profile;
    copy.cells.assign(view.cells.begin(), view.cells.end());
    copy.ports.assign(view.ports.begin(), view.ports.end());
    copy.edges.assign(view.edges.begin(), view.edges.end());
    return copy;
}

void assert_measurement_view_equal(
    const ExecutionMeasurementView& actual,
    const ExecutionMeasurement& expected) {
    assert_measurement_equal(expected, actual);
}

InitialParameterSeeds seeds_from_values(
    const CompiledGraph& plan,
    std::span<const InitialParameterValue> values) {
    InitialParameterSeeds seeds;
    for (const auto& cell : plan.cells()) {
        for (std::size_t slot = 0; slot < 2; ++slot) {
            seeds.cell_parameters.push_back(CellParameterSeed{
                cell.id,
                static_cast<ParameterSlot>(slot),
                values[cell.parameter_indices[slot]].value});
        }
    }
    for (const auto& edge : plan.edges()) {
        assert(std::holds_alternative<ContinuousValue>(
            values[edge.weight_parameter_index].value));
        seeds.edge_weights.push_back(EdgeParameterSeed{
            edge.id,
            std::get<ContinuousValue>(
                values[edge.weight_parameter_index].value).value});
    }
    return seeds;
}

RuntimeCellState imported_cell(
    CellId id,
    CellType type,
    double output,
    bool initialized = true) {
    RuntimeCellState state;
    state.cell = id;
    state.type = type;
    state.output_val = output;
    state.prev_output_val = output;
    state.initialized = initialized;
    return state;
}

std::shared_ptr<RuntimeState> make_imported_runtime(
    std::shared_ptr<const CompiledGraph> plan,
    std::shared_ptr<const InitialParameterValues> values,
    std::span<const RuntimeCellState> cells) {
    const auto result =
        RuntimeState::from_imported_state(plan, values, cells);
    assert(result.ok());
    return std::move(result.runtime);
}

Cell legacy_cell(
    uint32_t id,
    CellType type,
    double param1 = 1.0,
    double param2 = 0.0) {
    Cell cell;
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

Synapse legacy_synapse(
    uint32_t from,
    uint32_t to,
    double weight,
    uint8_t port = 0) {
    Synapse synapse;
    synapse.from_cell_id = from;
    synapse.to_cell_id = to;
    synapse.to_port = port;
    synapse.weight = weight;
    synapse.initial_weight = weight;
    synapse.hebbian_rate = 0.0;
    synapse.hebbian_decay = 0.02;
    return synapse;
}

void test_all30_reference_and_compiled_match() {
    for (const SemanticProfile profile :
         {SemanticProfile::LegacyCompatible, SemanticProfile::StrictCore}) {
        const auto graph = all30_graph(profile);
        auto fixture = make_fixture(graph, seeds_for(graph));
        for (int tick = 0; tick < 20; ++tick) {
            const std::array<double, 4> inputs{
                0.25 + 0.031 * tick,
                -0.5 - 0.017 * tick,
                0.75 + 0.013 * tick,
                -1.0 + 0.029 * tick};
            const auto reference =
                ReferenceExecutor{}.step(*fixture.reference, inputs);
            const auto optimized = fixture.executor->step(
                *fixture.optimized, inputs);
            assert(reference.ok());
            assert(optimized.ok());
            assert_measurement_equal(
                reference.measurement, optimized.measurement);
            assert_runtime_equal(*fixture.reference, *fixture.optimized);
            assert(optimized.measurement.cells.size() == graph.cells.size());
            assert(optimized.measurement.ports.size() ==
                   fixture.compiled.graph->port_reductions().size());
            assert(optimized.measurement.edges.size() ==
                   fixture.compiled.graph->edges().size());
        }
        assert(fixture.optimized->cell_state(CellId{118})->delay_idx != 0);
    }
}

void test_strict_initialization_and_stateful_feedback() {
    const auto graph = ema_graph(SemanticProfile::StrictCore);
    auto fixture = make_fixture(graph, ema_seeds(graph));
    const std::array<double, 1> silent{0.0};
    const std::array<double, 1> input{2.0};
    assert(ReferenceExecutor{}.step(*fixture.reference, silent).ok());
    assert(fixture.executor->step(*fixture.optimized, silent).ok());
    assert_runtime_equal(*fixture.reference, *fixture.optimized);
    assert(fixture.optimized->cell_state(CellId{2})->initialized);
    assert(ReferenceExecutor{}.step(*fixture.reference, input).ok());
    assert(fixture.executor->step(*fixture.optimized, input).ok());
    assert_runtime_equal(*fixture.reference, *fixture.optimized);
    assert(std::abs(fixture.optimized->cell_state(CellId{2})->output_val - 1.0) <
           1e-12);

    const auto legacy_graph = ema_graph(SemanticProfile::LegacyCompatible);
    auto legacy = make_fixture(legacy_graph, ema_seeds(legacy_graph));
    assert(ReferenceExecutor{}.step(*legacy.reference, silent).ok());
    assert(legacy.executor->step(*legacy.optimized, silent).ok());
    assert(ReferenceExecutor{}.step(*legacy.reference, input).ok());
    assert(legacy.executor->step(*legacy.optimized, input).ok());
    assert_runtime_equal(*legacy.reference, *legacy.optimized);
    assert(std::abs(legacy.optimized->cell_state(CellId{2})->output_val - 2.0) <
           1e-12);
}

void test_graph_recurrent_oracles_delay_wrap_and_oscillator_birth() {
    for (const SemanticProfile profile :
         {SemanticProfile::LegacyCompatible, SemanticProfile::StrictCore}) {
        GraphDefinition recurrent;
        recurrent.identity = GraphIdentity{702};
        recurrent.profile = profile;
        recurrent.cells = {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::OP_SUM}};
        recurrent.edges = {
            {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
             EdgeDelay::Immediate},
            {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{2}, InputPort{1},
             EdgeDelay::PreviousTick}};
        auto recurrent_seeds = seeds_for(recurrent);
        for (auto& seed : recurrent_seeds.edge_weights) {
            if (seed.edge == EdgeId{2}) {
                seed.initial_weight = 0.5;
            }
        }
        auto recurrent_fixture = make_fixture(recurrent, recurrent_seeds);
        const std::array<double, 3> oracle{1.0, 0.0, 0.0};
        const std::array<double, 3> expected{1.0, 0.5, 0.25};
        const std::array<double, 3> previous_values{0.0, 1.0, 0.5};
        for (std::size_t index = 0; index < oracle.size(); ++index) {
            const std::array<double, 1> input{oracle[index]};
            const auto reference =
                ReferenceExecutor{}.step(*recurrent_fixture.reference, input);
            const auto optimized = recurrent_fixture.executor->step(
                *recurrent_fixture.optimized, input);
            assert(reference.ok());
            assert(optimized.ok());
            assert_measurement_equal(
                reference.measurement, optimized.measurement);
            const auto reference_feedback = std::find_if(
                reference.measurement.edges.begin(),
                reference.measurement.edges.end(),
                [](const auto& edge) { return edge.edge == EdgeId{2}; });
            const auto optimized_feedback = std::find_if(
                optimized.measurement.edges.begin(),
                optimized.measurement.edges.end(),
                [](const auto& edge) { return edge.edge == EdgeId{2}; });
            assert(reference_feedback != reference.measurement.edges.end());
            assert(optimized_feedback != optimized.measurement.edges.end());
            assert(reference_feedback->source_mode == EdgeDelay::PreviousTick);
            assert(optimized_feedback->source_mode == EdgeDelay::PreviousTick);
            assert(reference_feedback->source_value == previous_values[index]);
            assert(optimized_feedback->source_value == previous_values[index]);
            assert(std::abs(
                       recurrent_fixture.optimized->cell_state(CellId{2})->output_val -
                       expected[index]) < 1e-12);
            assert_runtime_equal(
                *recurrent_fixture.reference, *recurrent_fixture.optimized);
        }
    }

    GraphDefinition delayed;
    delayed.identity = GraphIdentity{703};
    delayed.profile = SemanticProfile::StrictCore;
    delayed.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_DELAY_N}};
    delayed.edges = {{
        EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
        EdgeDelay::Immediate}};
    auto delayed_seeds = seeds_for(delayed);
    for (auto& seed : delayed_seeds.cell_parameters) {
        if (seed.cell == CellId{2} && seed.slot == ParameterSlot::Param1) {
            seed.value = DelayTicks{16};
        }
    }
    auto delayed_fixture = make_fixture(delayed, delayed_seeds);
    for (int tick = 0; tick < 18; ++tick) {
        const std::array<double, 1> input{
            static_cast<double>(tick + 1)};
        const auto reference =
            ReferenceExecutor{}.step(*delayed_fixture.reference, input);
        const auto optimized =
            delayed_fixture.executor->step(*delayed_fixture.optimized, input);
        assert(reference.ok());
        assert(optimized.ok());
        assert_measurement_equal(
            reference.measurement, optimized.measurement);
        assert_runtime_equal(
            *delayed_fixture.reference, *delayed_fixture.optimized);
    }
    assert(delayed_fixture.optimized->cell_state(CellId{2})->delay_idx == 2);
    assert(delayed_fixture.optimized->cell_state(CellId{2})->delay_buffer[1] == 18.0);

    GraphDefinition oscillator;
    oscillator.identity = GraphIdentity{704};
    oscillator.profile = SemanticProfile::StrictCore;
    oscillator.cells = {{CellId{1}, CellType::OP_OSCILLATOR}};
    auto oscillator_fixture = make_fixture(oscillator, seeds_for(oscillator));
    const auto oscillator_reference =
        ReferenceExecutor{}.step(*oscillator_fixture.reference, {});
    const auto oscillator_optimized =
        oscillator_fixture.executor->step(*oscillator_fixture.optimized, {});
    assert(oscillator_reference.ok());
    assert(oscillator_optimized.ok());
    assert_measurement_equal(
        oscillator_reference.measurement, oscillator_optimized.measurement);
    assert(oscillator_fixture.reference->cell_state(CellId{1})->initialized);
    assert_runtime_equal(
        *oscillator_fixture.reference, *oscillator_fixture.optimized);

    const auto imported = make_imported_runtime(
        oscillator_fixture.compiled.graph,
        oscillator_fixture.compiled.initial_values,
        std::array<RuntimeCellState, 1>{
            imported_cell(CellId{1}, CellType::OP_OSCILLATOR, 0.0, true)});
    const auto imported_reference = imported->fork_probe();
    assert(imported_reference.ok());
    auto imported_executor =
        CompiledExecutor::prepare(oscillator_fixture.compiled.graph);
    assert(imported_executor.ok());
    const auto no_reseed =
        imported_executor.executor->step(*imported, {});
    const auto no_reseed_reference =
        ReferenceExecutor{}.step(*imported_reference.runtime, {});
    assert(no_reseed.ok());
    assert(no_reseed_reference.ok());
    assert_measurement_equal(
        no_reseed_reference.measurement, no_reseed.measurement);
    assert(imported->cell_state(CellId{1})->output_val == 0.0);
    assert_runtime_equal(*imported_reference.runtime, *imported);
}

void test_legacy_import_old_order_and_rebind_preserves_live_state() {
    std::shared_ptr<const kun::migration::ExecutionSnapshot> snapshot;
    std::shared_ptr<RuntimeState> imported;
    std::vector<kun::migration::SnapshotEdgeProvenance> provenance;
    {
        CellularOrganism source;
        source.cells = {
            legacy_cell(30, CellType::SENSE_RAW_INPUT_0),
            legacy_cell(10, CellType::SENSE_RAW_INPUT_1),
            legacy_cell(20, CellType::SENSE_RAW_INPUT_2),
            legacy_cell(40, CellType::OP_SUM),
            legacy_cell(50, CellType::ACT_PRIMARY_POSITIVE)};
        source.synapses = {
            legacy_synapse(30, 40, 1e16),
            legacy_synapse(10, 40, -1e16),
            legacy_synapse(20, 40, 1.0),
            legacy_synapse(40, 50, 1.0)};
        source.synapses.back().initial_weight = 1.0;
        assert(source.compile());
        source.compiled_synapses_.back().weight = 2.5;
        const auto result = kun::migration::import_execution_snapshot(
            source, GraphIdentity{0x2345}, GraphRevision{9});
        assert(result.ok());
        snapshot = result.snapshot;
        const auto runtime_result =
            kun::migration::import_execution_snapshot_runtime(*snapshot);
        assert(runtime_result.ok());
        imported = runtime_result.runtime;
        provenance = runtime_result.edge_provenance;
        assert(provenance.size() == 4);
        assert(provenance.back().live_weight == 2.5);
        assert(provenance.back().raw_declared_initial_weight == 1.0);
    }

    auto reference_result = imported->fork_probe();
    assert(reference_result.ok());
    auto reference = std::move(reference_result.runtime);
    const auto prepared = CompiledExecutor::prepare(imported->plan());
    assert(prepared.ok());
    auto executor = std::move(prepared.executor);
    const std::array<double, 3> input{1.0, 1.0, 1.0};
    const auto reference_frame = ReferenceExecutor{}.step(*reference, input);
    const auto optimized_frame = executor->step(*imported, input);
    assert(reference_frame.ok());
    assert(optimized_frame.ok());
    assert_measurement_equal(
        reference_frame.measurement, optimized_frame.measurement);
    assert_runtime_equal(*reference, *imported);
    const auto sum_measurement = std::find_if(
        optimized_frame.measurement.cells.begin(),
        optimized_frame.measurement.cells.end(),
        [](const auto& cell) { return cell.cell == CellId{40}; });
    assert(sum_measurement != optimized_frame.measurement.cells.end());
    assert(sum_measurement->output == 1.0);

    assert(imported->set_parameter(
        imported->parameters().back().binding,
        ParameterValue{ContinuousValue{3.5}})
        .ok());
    assert(reference->set_parameter(
        reference->parameters().back().binding,
        ParameterValue{ContinuousValue{3.5}})
        .ok());
    const auto before_noop = imported->snapshot();
    auto seeds = seeds_from_values(
        *snapshot->compiled_plan(), imported->parameters());
    const auto rebuilt = GraphCompiler{}.compile(
        snapshot->definition(), seeds);
    assert(rebuilt.ok());
    assert(RuntimeMigration::rebind(
        *imported, rebuilt.graph, rebuilt.initial_values)
               .ok());
    assert_snapshot_equal(imported->snapshot(), before_noop);
    const auto rebound = executor->reprepare(rebuilt.graph);
    assert(rebound.ok());
    executor = std::move(rebound.executor);

    const auto checkpoint = imported->snapshot();
    auto probe = imported->fork_probe();
    assert(probe.ok());
    assert(probe.runtime->reset_episode().ok());
    assert(imported->restore_snapshot(checkpoint).ok());
    assert_snapshot_equal(imported->snapshot(), checkpoint);
    const auto continued_reference =
        ReferenceExecutor{}.step(*reference, input);
    const auto continued_optimized = executor->step(*imported, input);
    assert(continued_reference.ok());
    assert(continued_optimized.ok());
    assert_measurement_equal(
        continued_reference.measurement, continued_optimized.measurement);
    assert_runtime_equal(*reference, *imported);
}

void assert_empty_measurement(const ExecutionMeasurement& measurement) {
    assert(measurement.tick == 0);
    assert(measurement.cells.empty());
    assert(measurement.ports.empty());
    assert(measurement.edges.empty());
}

void assert_empty_measurement(const ExecutionMeasurementView& measurement) {
    assert(measurement.tick == 0);
    assert(measurement.cells.empty());
    assert(measurement.ports.empty());
    assert(measurement.edges.empty());
}

template <typename Fixture>
void assert_atomic_failure(
    Fixture& fixture,
    std::span<const double> failing_inputs,
    CompiledExecutorErrorCode compiled_code,
    CellId diagnostic_cell,
    EdgeId diagnostic_edge,
    SdscCellKernelStatus kernel_status) {
    const auto before_reference = fixture.reference->snapshot();
    const auto before_optimized = fixture.optimized->snapshot();
    const auto committed = copy_measurement(fixture.executor->last_measurement());
    const auto reference_failure =
        ReferenceExecutor{}.step(*fixture.reference, failing_inputs);
    const auto optimized_failure =
        fixture.executor->step(*fixture.optimized, failing_inputs);
    assert(!reference_failure.ok());
    assert(!optimized_failure.ok());
    assert_empty_measurement(reference_failure.measurement);
    assert_empty_measurement(optimized_failure.measurement);
    assert(reference_failure.error->has_cell);
    assert(reference_failure.error->cell == diagnostic_cell);
    assert(reference_failure.error->has_edge ==
           (diagnostic_edge != EdgeId{0}));
    if (diagnostic_edge != EdgeId{0}) {
        assert(reference_failure.error->edge == diagnostic_edge);
    }
    assert(reference_failure.error->kernel_status == kernel_status);
    assert(optimized_failure.error->code == compiled_code);
    assert(optimized_failure.error->has_cell);
    assert(optimized_failure.error->cell == diagnostic_cell);
    assert(optimized_failure.error->has_edge ==
           (diagnostic_edge != EdgeId{0}));
    if (diagnostic_edge != EdgeId{0}) {
        assert(optimized_failure.error->edge == diagnostic_edge);
    }
    assert(optimized_failure.error->kernel_status == kernel_status);
    assert_snapshot_equal(fixture.reference->snapshot(), before_reference);
    assert_snapshot_equal(fixture.optimized->snapshot(), before_optimized);
    assert_runtime_equal(*fixture.reference, *fixture.optimized);
    assert_measurement_view_equal(
        fixture.executor->last_measurement(), committed);
}

void test_post_prefix_failures_are_atomic_and_measurement_stable() {
    {
        GraphDefinition graph;
        graph.identity = GraphIdentity{710};
        graph.profile = SemanticProfile::StrictCore;
        graph.cells = {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::OP_SUM}};
        graph.edges = {{
            EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
            EdgeDelay::Immediate}};
        auto seeds = seeds_for(graph);
        for (auto& edge : seeds.edge_weights) edge.initial_weight = 1e-308;
        auto fixture = make_imported_fixture(
            graph, seeds, std::array<RuntimeCellState, 2>{
                imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
                imported_cell(CellId{2}, CellType::OP_SUM, 0.0, false)});
        const std::array<double, 1> success{1e308};
        assert(ReferenceExecutor{}.step(*fixture.reference, success).ok());
        assert(fixture.executor->step(*fixture.optimized, success).ok());
        set_edge_weight(*fixture.reference, EdgeId{1}, 1e308);
        set_edge_weight(*fixture.optimized, EdgeId{1}, 1e308);
        assert_atomic_failure(
            fixture, success, CompiledExecutorErrorCode::InvalidInput,
            CellId{2}, EdgeId{1}, SDSC_CELL_KERNEL_OK);
    }

    {
        GraphDefinition graph;
        graph.identity = GraphIdentity{711};
        graph.profile = SemanticProfile::StrictCore;
        graph.cells = {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::SENSE_RAW_INPUT_1},
            {CellId{3}, CellType::OP_SUM}};
        graph.edges = {
            {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
             EdgeDelay::Immediate},
            {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
             EdgeDelay::Immediate}};
        auto seeds = seeds_for(graph);
        for (auto& edge : seeds.edge_weights) edge.initial_weight = 1e-308;
        auto fixture = make_imported_fixture(
            graph, seeds, std::array<RuntimeCellState, 3>{
                imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
                imported_cell(CellId{2}, CellType::SENSE_RAW_INPUT_1, 0.0, false),
                imported_cell(CellId{3}, CellType::OP_SUM, 0.0, false)});
        const std::array<double, 2> success{1e308, 1e308};
        assert(ReferenceExecutor{}.step(*fixture.reference, success).ok());
        assert(fixture.executor->step(*fixture.optimized, success).ok());
        set_edge_weight(*fixture.reference, EdgeId{1}, 1.0);
        set_edge_weight(*fixture.reference, EdgeId{2}, 1.0);
        set_edge_weight(*fixture.optimized, EdgeId{1}, 1.0);
        set_edge_weight(*fixture.optimized, EdgeId{2}, 1.0);
        assert_atomic_failure(
            fixture, success, CompiledExecutorErrorCode::InvalidInput,
            CellId{3}, EdgeId{2}, SDSC_CELL_KERNEL_OK);
    }

    {
        GraphDefinition graph;
        graph.identity = GraphIdentity{712};
        graph.profile = SemanticProfile::StrictCore;
        graph.cells = {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::OP_QUADRATIC}};
        graph.edges = {{
            EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
            EdgeDelay::Immediate}};
        auto seeds = seeds_for(graph);
        for (auto& edge : seeds.edge_weights) edge.initial_weight = 1e-308;
        auto fixture = make_imported_fixture(
            graph, seeds, std::array<RuntimeCellState, 2>{
                imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
                imported_cell(CellId{2}, CellType::OP_QUADRATIC, 0.0, false)});
        const std::array<double, 1> success{1e308};
        assert(ReferenceExecutor{}.step(*fixture.reference, success).ok());
        assert(fixture.executor->step(*fixture.optimized, success).ok());
        set_edge_weight(*fixture.reference, EdgeId{1}, 1.0);
        set_edge_weight(*fixture.optimized, EdgeId{1}, 1.0);
        set_cell_continuous(
            *fixture.reference, CellId{2}, ParameterSlot::Param1, 1e308);
        set_cell_continuous(
            *fixture.optimized, CellId{2}, ParameterSlot::Param1, 1e308);
        assert_atomic_failure(
            fixture, success, CompiledExecutorErrorCode::KernelFailure,
            CellId{2}, EdgeId{0}, SDSC_CELL_KERNEL_NUMERICAL_ERROR);
    }

    {
        GraphDefinition graph;
        graph.identity = GraphIdentity{713};
        graph.profile = SemanticProfile::StrictCore;
        graph.cells = {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::SENSE_CHANNEL}};
        auto fixture = make_fixture(graph, seeds_for(graph, 43));
        std::array<double, 44> success{};
        success[0] = 1.0;
        success[43] = 2.0;
        assert(ReferenceExecutor{}.step(*fixture.reference, success).ok());
        assert(fixture.executor->step(*fixture.optimized, success).ok());
        std::array<double, 43> missing{};
        missing[0] = 1.0;
        assert_atomic_failure(
            fixture, missing, CompiledExecutorErrorCode::KernelFailure,
            CellId{2}, EdgeId{0}, SDSC_CELL_KERNEL_INPUT_OUT_OF_RANGE);
    }
}

void test_shared_prepared_executor_isolated_between_runtimes() {
    const auto graph = ema_graph(SemanticProfile::StrictCore);
    auto seeds = ema_seeds(graph);
    auto compiled = GraphCompiler{}.compile(graph, seeds);
    assert(compiled.ok());
    auto first = RuntimeState::create(
        compiled.graph, compiled.initial_values);
    auto second = RuntimeState::create(
        compiled.graph, compiled.initial_values);
    assert(first.ok());
    assert(second.ok());
    auto first_reference = first.runtime->fork_probe();
    auto second_reference = second.runtime->fork_probe();
    assert(first_reference.ok());
    assert(second_reference.ok());
    set_edge_weight(*first.runtime, EdgeId{1}, 0.5);
    set_edge_weight(*first_reference.runtime, EdgeId{1}, 0.5);
    set_edge_weight(*second.runtime, EdgeId{1}, 2.0);
    set_edge_weight(*second_reference.runtime, EdgeId{1}, 2.0);
    const auto prepared = CompiledExecutor::prepare(compiled.graph);
    assert(prepared.ok());
    auto executor = std::move(prepared.executor);
    const std::array<double, 1> input{1.0};
    for (int tick = 0; tick < 3; ++tick) {
        const auto first_expected =
            ReferenceExecutor{}.step(*first_reference.runtime, input);
        const auto second_expected =
            ReferenceExecutor{}.step(*second_reference.runtime, input);
        const auto first_actual = executor->step(*first.runtime, input);
        assert(first_expected.ok());
        assert(second_expected.ok());
        assert(first_actual.ok());
        const auto first_frame = copy_measurement(first_actual.measurement);
        assert_measurement_equal(
            first_expected.measurement, first_actual.measurement);
        const auto second_actual = executor->step(*second.runtime, input);
        assert(second_actual.ok());
        assert_measurement_equal(first_expected.measurement, first_frame);
        assert_measurement_equal(
            second_expected.measurement, second_actual.measurement);
        assert_runtime_equal(*first_reference.runtime, *first.runtime);
        assert_runtime_equal(*second_reference.runtime, *second.runtime);
        assert(first.runtime->cell_state(CellId{2})->output_val !=
               second.runtime->cell_state(CellId{2})->output_val);
    }
}

void test_strict_migration_empty_graph_and_typed_channel_boundaries() {
    auto strict_graph = ema_graph(SemanticProfile::StrictCore);
    auto strict = make_fixture(strict_graph, ema_seeds(strict_graph));
    const std::array<double, 1> input{1.0};
    assert(strict.executor->step(*strict.optimized, input).ok());
    const double preserved_state =
        strict.optimized->cell_state(CellId{2})->state_val;

    auto changed = strict_graph;
    changed.revision = GraphRevision{2};
    changed.cells[1].type = CellType::OP_ABS;
    const auto changed_compiled =
        GraphCompiler{}.compile(changed, ema_seeds(changed));
    assert(changed_compiled.ok());
    assert(RuntimeMigration::rebind(
               *strict.optimized,
               changed_compiled.graph,
               changed_compiled.initial_values)
               .ok());
    assert(!strict.optimized->cell_state(CellId{2})->initialized);
    assert(strict.optimized->cell_state(CellId{1})->initialized);
    assert(strict.optimized->cell_state(CellId{1})->output_val != 0.0);
    assert(strict.optimized->cell_state(CellId{2})->state_val !=
           preserved_state);

    GraphDefinition discrete_graph;
    discrete_graph.identity = GraphIdentity{724};
    discrete_graph.profile = SemanticProfile::StrictCore;
    discrete_graph.cells = {
        {CellId{1}, CellType::SENSE_CHANNEL},
        {CellId{2}, CellType::OP_INTEGRAL}};
    discrete_graph.edges = {{
        EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
        EdgeDelay::Immediate}};
    auto discrete_fixture =
        make_fixture(discrete_graph, seeds_for(discrete_graph, 1));
    std::array<double, 2> discrete_input{0.0, 2.0};
    assert(discrete_fixture.executor->step(
        *discrete_fixture.optimized, discrete_input)
               .ok());
    const double unaffected_memory =
        discrete_fixture.optimized->cell_state(CellId{2})->state_val;
    auto discrete_changed = discrete_graph;
    discrete_changed.revision = GraphRevision{2};
    const auto discrete_recompiled = GraphCompiler{}.compile(
        discrete_changed, seeds_for(discrete_changed, 43));
    assert(discrete_recompiled.ok());
    assert(RuntimeMigration::rebind(
               *discrete_fixture.optimized,
               discrete_recompiled.graph,
               discrete_recompiled.initial_values)
               .ok());
    assert(!discrete_fixture.optimized->cell_state(CellId{1})->initialized);
    assert(discrete_fixture.optimized->cell_state(CellId{2})->initialized);
    assert(discrete_fixture.optimized->cell_state(CellId{2})->state_val ==
           unaffected_memory);

    auto cross_profile = changed;
    cross_profile.profile = SemanticProfile::LegacyCompatible;
    const auto cross_compiled =
        GraphCompiler{}.compile(cross_profile, ema_seeds(cross_profile));
    assert(cross_compiled.ok());
    const auto before_cross = strict.optimized->snapshot();
    assert(!RuntimeMigration::rebind(
                *strict.optimized,
                cross_compiled.graph,
                cross_compiled.initial_values)
                .ok());
    assert_snapshot_equal(strict.optimized->snapshot(), before_cross);

    GraphDefinition empty;
    empty.identity = GraphIdentity{720};
    empty.revision = GraphRevision{1};
    empty.profile = SemanticProfile::StrictCore;
    auto empty_fixture = make_fixture(empty, {});
    assert(empty_fixture.executor->step(*empty_fixture.optimized, {}).ok());
    assert(empty_fixture.optimized->tick() == 1);
    assert(empty_fixture.executor->last_measurement().cells.empty());

    GraphDefinition channel;
    channel.identity = GraphIdentity{721};
    channel.profile = SemanticProfile::StrictCore;
    channel.cells = {{CellId{1}, CellType::SENSE_CHANNEL}};
    auto channel_fixture = make_fixture(channel, seeds_for(channel, 43));
    const std::array<double, 1> missing{1.0};
    assert(!channel_fixture.executor->step(
        *channel_fixture.optimized, missing)
                 .ok());
    std::array<double, 44> available{};
    available[43] = 7.0;
    assert(channel_fixture.executor->step(
        *channel_fixture.optimized, available)
                 .ok());
    assert(channel_fixture.optimized->cell_state(CellId{1})->output_val == 7.0);

    GraphDefinition action;
    action.identity = GraphIdentity{722};
    action.profile = SemanticProfile::StrictCore;
    action.cells = {{CellId{1}, CellType::ACT_CHANNEL}};
    auto action_seeds = seeds_for(action, std::numeric_limits<std::size_t>::max());
    auto action_fixture = make_fixture(action, action_seeds);
    assert(action_fixture.executor->step(*action_fixture.optimized, {}).ok());
    assert(action_fixture.optimized->cell_state(CellId{1})->initialized);
}

void test_failure_is_atomic_and_last_measurement_is_stable() {
    const auto graph = ema_graph(SemanticProfile::StrictCore);
    auto fixture = make_fixture(graph, ema_seeds(graph));
    const std::array<double, 1> input{1.0};
    assert(fixture.executor->step(*fixture.optimized, input).ok());
    const auto before = fixture.optimized->snapshot();
    const auto committed = fixture.executor->last_measurement();
    const auto committed_tick = committed.tick;
    const auto committed_cells = committed.cells.size();
    const auto committed_ports = committed.ports.size();
    const auto committed_edges = committed.edges.size();

    const std::array<double, 1> bad{std::numeric_limits<double>::quiet_NaN()};
    const auto failed = fixture.executor->step(*fixture.optimized, bad);
    assert(!failed.ok());
    assert_snapshot_equal(fixture.optimized->snapshot(), before);
    const auto after_bad_input = fixture.executor->last_measurement();
    assert(after_bad_input.tick == committed_tick);
    assert(after_bad_input.cells.size() == committed_cells);
    assert(after_bad_input.ports.size() == committed_ports);
    assert(after_bad_input.edges.size() == committed_edges);

    auto late_graph = ema_graph(SemanticProfile::StrictCore);
    late_graph.cells.push_back({CellId{4}, CellType::SENSE_CHANNEL});
    auto late_seeds = ema_seeds(late_graph, 100);
    auto late = make_fixture(late_graph, late_seeds);
    assert(late.executor->step(*late.optimized, input).ok() == false);
    const auto late_before = late.optimized->snapshot();
    const auto late_failed = late.executor->step(*late.optimized, input);
    assert(!late_failed.ok());
    assert_snapshot_equal(late.optimized->snapshot(), late_before);
}

void test_reprepare_rejects_stale_plan_and_accepts_noop() {
    const auto original = ema_graph(SemanticProfile::StrictCore);
    auto fixture = make_fixture(original, ema_seeds(original));
    const std::array<double, 1> input{0.5};
    assert(fixture.executor->step(*fixture.optimized, input).ok());
    const auto before_migration = fixture.optimized->snapshot();

    auto rewired = ema_graph(SemanticProfile::StrictCore, GraphRevision{2}, true);
    const auto rebuilt = GraphCompiler{}.compile(rewired, ema_seeds(rewired));
    assert(rebuilt.ok());
    assert(RuntimeMigration::rebind(
               *fixture.optimized, rebuilt.graph, rebuilt.initial_values)
               .ok());
    assert(!fixture.executor->step(*fixture.optimized, input).ok());
    assert(fixture.optimized->tick() == before_migration.tick());

    const auto prepared = fixture.executor->reprepare(rebuilt.graph);
    assert(prepared.ok());
    fixture.executor = std::move(prepared.executor);
    const auto rewired_step = fixture.executor->step(*fixture.optimized, input);
    assert(rewired_step.ok());
    assert(rewired_step.measurement.edges.size() == rebuilt.graph->edges().size());
    const auto rewired_edge = std::find_if(
        rewired_step.measurement.edges.begin(),
        rewired_step.measurement.edges.end(),
        [](const auto& edge) { return edge.edge == EdgeId{3}; });
    assert(rewired_edge != rewired_step.measurement.edges.end());
    assert(rewired_edge->source == CellId{3});

    const auto separately_compiled =
        GraphCompiler{}.compile(rewired, ema_seeds(rewired));
    assert(separately_compiled.ok());
    const auto no_op = CompiledExecutor::prepare(separately_compiled.graph);
    assert(no_op.ok());
    auto second_runtime = RuntimeState::create(
        separately_compiled.graph, separately_compiled.initial_values);
    assert(second_runtime.ok());
    assert(no_op.executor->step(*second_runtime.runtime, input).ok());
}

void test_prepared_steps_do_not_allocate() {
    for (const SemanticProfile profile :
         {SemanticProfile::LegacyCompatible, SemanticProfile::StrictCore}) {
        const auto graph = ema_graph(profile);
        auto fixture = make_fixture(graph, ema_seeds(graph));
        const std::array<double, 1> input{0.25};
        {
            AllocationScope scope;
            for (int iteration = 0; iteration < 4; ++iteration) {
                assert(fixture.executor->step(*fixture.optimized, input).ok());
            }
            assert(scope.count() == 0);
        }
    }

    const auto graph = ema_graph(SemanticProfile::StrictCore);
    auto fixture = make_fixture(graph, ema_seeds(graph));
    const std::array<double, 1> bad{std::numeric_limits<double>::quiet_NaN()};
    {
        AllocationScope scope;
        for (int iteration = 0; iteration < 4; ++iteration) {
            assert(!fixture.executor->step(*fixture.optimized, bad).ok());
        }
        assert(scope.count() == 0);
    }

    auto late_graph = ema_graph(SemanticProfile::StrictCore);
    late_graph.cells.push_back({CellId{4}, CellType::SENSE_CHANNEL});
    auto late = make_fixture(late_graph, ema_seeds(late_graph, 100));
    const std::array<double, 1> finite_input{0.5};
    {
        AllocationScope late_scope;
        for (int iteration = 0; iteration < 4; ++iteration) {
            assert(!late.executor->step(*late.optimized, finite_input).ok());
        }
        assert(late_scope.count() == 0);
    }

    GraphDefinition overflow_graph;
    overflow_graph.identity = GraphIdentity{723};
    overflow_graph.profile = SemanticProfile::StrictCore;
    overflow_graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_SUM}};
    overflow_graph.edges = {{
        EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
        EdgeDelay::Immediate}};
    auto overflow_seeds = seeds_for(overflow_graph);
    for (auto& edge : overflow_seeds.edge_weights) edge.initial_weight = 1e-308;
    auto overflow = make_imported_fixture(
        overflow_graph, overflow_seeds, std::array<RuntimeCellState, 2>{
            imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
            imported_cell(CellId{2}, CellType::OP_SUM, 0.0, false)});
    const std::array<double, 1> overflow_input{1e308};
    assert(overflow.executor->step(*overflow.optimized, overflow_input).ok());
    set_edge_weight(*overflow.optimized, EdgeId{1}, 1e308);
    {
        AllocationScope overflow_scope;
        assert(!overflow.executor->step(
            *overflow.optimized, overflow_input)
                    .ok());
        assert(overflow_scope.count() == 0);
    }

    GraphDefinition sum_graph;
    sum_graph.identity = GraphIdentity{725};
    sum_graph.profile = SemanticProfile::StrictCore;
    sum_graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::SENSE_RAW_INPUT_1},
        {CellId{3}, CellType::OP_SUM}};
    sum_graph.edges = {
        {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
         EdgeDelay::Immediate},
        {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
         EdgeDelay::Immediate}};
    auto sum_seeds = seeds_for(sum_graph);
    for (auto& edge : sum_seeds.edge_weights) edge.initial_weight = 1e-308;
    auto sum = make_imported_fixture(
        sum_graph, sum_seeds, std::array<RuntimeCellState, 3>{
            imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
            imported_cell(CellId{2}, CellType::SENSE_RAW_INPUT_1, 0.0, false),
            imported_cell(CellId{3}, CellType::OP_SUM, 0.0, false)});
    const std::array<double, 2> sum_input{1e308, 1e308};
    assert(sum.executor->step(*sum.optimized, sum_input).ok());
    set_edge_weight(*sum.optimized, EdgeId{1}, 1.0);
    set_edge_weight(*sum.optimized, EdgeId{2}, 1.0);
    {
        AllocationScope sum_scope;
        assert(!sum.executor->step(*sum.optimized, sum_input).ok());
        assert(sum_scope.count() == 0);
    }

    GraphDefinition native_graph;
    native_graph.identity = GraphIdentity{726};
    native_graph.profile = SemanticProfile::StrictCore;
    native_graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::OP_QUADRATIC}};
    native_graph.edges = {{
        EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
        EdgeDelay::Immediate}};
    auto native_seeds = seeds_for(native_graph);
    for (auto& edge : native_seeds.edge_weights) edge.initial_weight = 1e-308;
    auto native = make_imported_fixture(
        native_graph, native_seeds, std::array<RuntimeCellState, 2>{
            imported_cell(CellId{1}, CellType::SENSE_RAW_INPUT_0, 0.0, false),
            imported_cell(CellId{2}, CellType::OP_QUADRATIC, 0.0, false)});
    assert(native.executor->step(*native.optimized, overflow_input).ok());
    set_edge_weight(*native.optimized, EdgeId{1}, 1.0);
    set_cell_continuous(
        *native.optimized, CellId{2}, ParameterSlot::Param1, 1e308);
    {
        AllocationScope native_scope;
        assert(!native.executor->step(*native.optimized, overflow_input).ok());
        assert(native_scope.count() == 0);
    }

    GraphDefinition missing_graph;
    missing_graph.identity = GraphIdentity{727};
    missing_graph.profile = SemanticProfile::StrictCore;
    missing_graph.cells = {
        {CellId{1}, CellType::SENSE_RAW_INPUT_0},
        {CellId{2}, CellType::SENSE_CHANNEL}};
    auto missing = make_fixture(missing_graph, seeds_for(missing_graph, 43));
    std::array<double, 44> missing_success{};
    missing_success[0] = 1.0;
    missing_success[43] = 2.0;
    assert(missing.executor->step(
        *missing.optimized, missing_success)
               .ok());
    std::array<double, 43> missing_input{};
    missing_input[0] = 1.0;
    {
        AllocationScope missing_scope;
        assert(!missing.executor->step(
            *missing.optimized, missing_input)
                    .ok());
        assert(missing_scope.count() == 0);
    }
}

}  // namespace

int main() {
    test_all30_reference_and_compiled_match();
    test_strict_initialization_and_stateful_feedback();
    test_graph_recurrent_oracles_delay_wrap_and_oscillator_birth();
    test_legacy_import_old_order_and_rebind_preserves_live_state();
    test_post_prefix_failures_are_atomic_and_measurement_stable();
    test_shared_prepared_executor_isolated_between_runtimes();
    test_strict_migration_empty_graph_and_typed_channel_boundaries();
    test_failure_is_atomic_and_last_measurement_is_stable();
    test_reprepare_rejects_stale_plan_and_accepts_noop();
    test_prepared_steps_do_not_allocate();
    return 0;
}
