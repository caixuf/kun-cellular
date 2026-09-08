#include "kun/cellular/cellular_graph_edit.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/runtime_state.hpp"

#include <cassert>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

using namespace kun::core;
using namespace kun;

namespace {
std::atomic<long long> allocation_fail_after{-1};
std::atomic<long long> allocation_count{0};

void* allocate_test_memory(std::size_t size) {
    const long long fail = allocation_fail_after.load(std::memory_order_relaxed);
    const long long index =
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (fail >= 0 && index >= fail) throw std::bad_alloc();
    if (void* result = std::malloc(size == 0 ? 1 : size)) return result;
    throw std::bad_alloc();
}

}  // namespace

void* operator new(std::size_t size) { return allocate_test_memory(size); }
void* operator new[](std::size_t size) { return allocate_test_memory(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    const auto align = static_cast<std::size_t>(alignment);
    const auto rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
    const long long fail = allocation_fail_after.load(std::memory_order_relaxed);
    const long long index =
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (fail >= 0 && index >= fail) throw std::bad_alloc();
    if (void* result = std::aligned_alloc(align, rounded)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

namespace {

GraphDefinition make_graph(GraphRevision revision = GraphRevision{1}) {
    return GraphDefinition{
        GraphIdentity{900},
        revision,
        SemanticProfile::LegacyCompatible,
        1,
        {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::OP_SUM},
            {CellId{3}, CellType::OP_EMA},
        },
        {
            {EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
             EdgeDelay::Immediate},
            {EdgeId{11}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{1},
             EdgeDelay::PreviousTick},
            {EdgeId{12}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
             EdgeDelay::Immediate},
        }};
}

InitialParameterSeeds make_seeds(const GraphDefinition& graph) {
    InitialParameterSeeds seeds;
    for (const auto& cell : graph.cells) {
        const auto contract = contract_for(cell.type);
        assert(contract.has_value());
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            ParameterValue value = UnusedParameter{};
            switch (descriptor.value_type) {
                case ParameterValueType::Continuous:
                    value = ContinuousValue{0.5};
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

void assert_same_snapshot(const RuntimeSnapshot& lhs, const RuntimeSnapshot& rhs) {
    assert(lhs.plan());
    assert(rhs.plan());
    assert(detail::same_plan(*lhs.plan(), *rhs.plan()));
    assert(lhs.tick() == rhs.tick());
    assert(lhs.parameters().size() == rhs.parameters().size());
    for (std::size_t i = 0; i < lhs.parameters().size(); ++i) {
        const auto& left = lhs.parameters()[i];
        const auto& right = rhs.parameters()[i];
        assert(left.binding.kind == right.binding.kind);
        assert(left.binding.index == right.binding.index);
        assert(left.binding.cell == right.binding.cell);
        assert(left.binding.edge == right.binding.edge);
        assert(left.binding.slot == right.binding.slot);
        assert(left.value.index() == right.value.index());
        std::visit(
            [](const auto& a, const auto& b) {
                using A = std::decay_t<decltype(a)>;
                using B = std::decay_t<decltype(b)>;
                if constexpr (std::is_same_v<A, B>) {
                    if constexpr (std::is_same_v<A, ContinuousValue>) {
                        assert(a.value == b.value);
                    } else if constexpr (std::is_same_v<A, ChannelIndex>) {
                        assert(a.value == b.value);
                    } else if constexpr (std::is_same_v<A, DelayTicks>) {
                        assert(a.value == b.value);
                    } else if constexpr (std::is_same_v<A, MinMaxMode>) {
                        assert(a == b);
                    }
                } else {
                    assert(false);
                }
            },
            left.value, right.value);
    }
    assert(lhs.cells().size() == rhs.cells().size());
    for (std::size_t i = 0; i < lhs.cells().size(); ++i) {
        const auto& left = lhs.cells()[i];
        const auto& right = rhs.cells()[i];
        assert(left.cell == right.cell);
        assert(left.type == right.type);
        assert(left.state_val == right.state_val);
        assert(left.aux_state == right.aux_state);
        assert(left.prev_input == right.prev_input);
        assert(left.output_val == right.output_val);
        assert(left.prev_output_val == right.prev_output_val);
        assert(left.delay_buffer == right.delay_buffer);
        assert(left.delay_idx == right.delay_idx);
        assert(left.latch_state == right.latch_state);
        assert(left.activation_count == right.activation_count);
        assert(left.initialized == right.initialized);
    }
}

Fixture make_fixture() {
    Fixture fixture{make_graph(), {}, nullptr};
    fixture.compiled =
        GraphCompiler{}.compile(fixture.definition, make_seeds(fixture.definition));
    assert(fixture.compiled.ok());
    auto runtime =
        RuntimeState::create(fixture.compiled.graph, fixture.compiled.initial_values);
    assert(runtime.ok());
    fixture.runtime = std::move(runtime.runtime);
    return fixture;
}

void test_split_selected_edges_preserves_time_and_executes_new_cell() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const CellBirth split_cell{
        CellId{20},
        CellType::OP_SUM,
        {ParameterValue{UnusedParameter{}}, ParameterValue{UnusedParameter{}}}};
    const std::vector<GraphEditEvent> events{
        GraphEditEvent{1, SplitEdgeAction{
            EdgeId{10}, split_cell, EdgeId{20}, EdgeId{21}, InputPort{0},
            1.25, 0.75}},
        GraphEditEvent{2, SplitEdgeAction{
            EdgeId{11},
            CellBirth{
                CellId{22},
                CellType::OP_EMA,
                {ParameterValue{ContinuousValue{0.5}},
                 ParameterValue{UnusedParameter{}}}},
            EdgeId{22}, EdgeId{23}, InputPort{0}, 0.8, 0.9}},
    };

    const auto result = editor.apply(*fixture.runtime, events);
    assert(result.ok());
    assert(result.graph);
    assert(result.graph->revision() == GraphRevision{2});
    assert(result.graph->edges().size() == 5);

    const auto find_edge = [&](EdgeId id) {
        for (const auto& edge : result.definition.edges) {
            if (edge.id == id) return edge;
        }
        assert(false);
        return EdgeDefinition{};
    };
    const auto first = find_edge(EdgeId{20});
    assert(first.source == CellId{1});
    assert(first.target == CellId{20});
    assert(first.source_port == OutputPort{0});
    assert(first.target_port == InputPort{0});
    assert(first.delay == EdgeDelay::Immediate);
    const auto second = find_edge(EdgeId{21});
    assert(second.source == CellId{20});
    assert(second.target == CellId{2});
    assert(second.target_port == InputPort{0});
    assert(second.delay == EdgeDelay::Immediate);
    const auto delayed_first = find_edge(EdgeId{22});
    assert(delayed_first.target == CellId{22});
    assert(delayed_first.delay == EdgeDelay::PreviousTick);
    const auto delayed_second = find_edge(EdgeId{23});
    assert(delayed_second.source == CellId{22});
    assert(delayed_second.target == CellId{2});
    assert(delayed_second.target_port == InputPort{1});
    assert(delayed_second.delay == EdgeDelay::Immediate);
    assert(fixture.runtime->revision() == GraphRevision{2});
    assert(!fixture.runtime->cell_state(CellId{20})->initialized);
    assert(!fixture.runtime->cell_state(CellId{22})->initialized);

    ReferenceExecutor reference;
    const double input = 2.0;
    const auto step = reference.step(*fixture.runtime, std::span<const double>(&input, 1));
    assert(step.ok());
    assert(fixture.runtime->cell_state(CellId{20})->initialized);
    assert(fixture.runtime->cell_state(CellId{22})->initialized);
    bool saw_split_sum = false;
    for (const auto& cell : step.measurement.cells) {
        if (cell.cell == CellId{20}) {
            saw_split_sum = true;
            assert(std::abs(cell.output - 1.25) < 1e-6);
        }
    }
    assert(saw_split_sum);
    const double next_input = 3.0;
    const auto next =
        reference.step(*fixture.runtime, std::span<const double>(&next_input, 1));
    assert(next.ok());
    bool saw_stateful_split = false;
    for (const auto& cell : next.measurement.cells) {
        if (cell.cell == CellId{22}) {
            saw_stateful_split = true;
            assert(std::isfinite(cell.output));
        }
        if (cell.cell == CellId{20}) {
            assert(std::abs(cell.output - 1.875) < 1e-6);
        }
    }
    assert(saw_stateful_split);
    auto probe = fixture.runtime->fork_probe();
    assert(probe.ok());
    const auto prepared = CompiledExecutor::prepare(result.graph);
    assert(prepared.ok());
    auto compiled_executor = std::move(prepared.executor);
    const auto compiled_step = compiled_executor->step(
        *probe.runtime, std::span<const double>(&next_input, 1));
    assert(compiled_step.ok());
    bool compiled_saw_split = false;
    for (const auto& cell : compiled_step.measurement.cells) {
        if (cell.cell == CellId{20}) {
            compiled_saw_split = true;
            assert(std::abs(cell.output - 1.875) < 1e-6);
        }
    }
    assert(compiled_saw_split);
}

void test_failed_batch_preserves_runtime_and_rejects_receptor_split() {
    auto fixture = make_fixture();
    const auto before = fixture.runtime->snapshot();
    GraphEditor editor(*fixture.runtime);
    const std::vector<GraphEditEvent> events{
        GraphEditEvent{10, SplitEdgeAction{
            EdgeId{10},
            CellBirth{
                CellId{30},
                CellType::SENSE_RAW_INPUT_0,
                {ParameterValue{ContinuousValue{1.0}},
                 ParameterValue{UnusedParameter{}}}},
            EdgeId{30}, EdgeId{31}, InputPort{0}, 1.0, 1.0}},
    };
    const auto result = editor.apply(*fixture.runtime, events);
    assert(!result.ok());
    assert(result.error.has_value());
    assert(result.error->code == GraphEditErrorCode::InvalidSplit);
    assert(fixture.runtime->revision() == before.plan()->revision());
    assert(fixture.runtime->cell_states().size() == before.cells().size());
    assert(fixture.runtime->parameters().size() == before.parameters().size());
}

void test_remove_commits_then_invalid_growth_cannot_restore_ids() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto enriched = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{0, AddEdgeAction{EdgeBirth{
                EdgeId{13}, CellId{2}, OutputPort{0}, CellId{2}, InputPort{0},
                EdgeDelay::PreviousTick, 0.4}}}});
    assert(enriched.ok());
    const std::vector<GraphEditEvent> remove{
        GraphEditEvent{1, RemoveCellAction{CellId{2}}}};
    const auto removed = editor.apply(*fixture.runtime, remove);
    assert(removed.ok());
    assert(fixture.runtime->cell_state(CellId{2}) == nullptr);
    assert(fixture.runtime->plan()->edges().empty());

    const std::vector<GraphEditEvent> illegal_growth{
        GraphEditEvent{2, AddEdgeAction{
            EdgeBirth{EdgeId{99}, CellId{3}, OutputPort{0}, CellId{3},
                       InputPort{0}, EdgeDelay::Immediate, 1.0}}}};
    const auto rejected = editor.apply(*fixture.runtime, illegal_growth);
    assert(!rejected.ok());
    assert(fixture.runtime->cell_state(CellId{2}) == nullptr);
    assert(fixture.runtime->revision() == GraphRevision{3});
}

void test_live_state_and_weights_survive_and_old_executor_is_stale() {
    auto fixture = make_fixture();
    const auto prepared = CompiledExecutor::prepare(fixture.runtime->plan());
    assert(prepared.ok());
    auto old_executor = std::move(prepared.executor);
    assert(fixture.runtime->set_parameter(
        edge_binding(*fixture.runtime, EdgeId{12}),
        ParameterValue{ContinuousValue{2.5}}).ok());
    assert(fixture.runtime->set_parameter(
        edge_binding(*fixture.runtime, EdgeId{10}),
        ParameterValue{ContinuousValue{2.5}}).ok());
    assert(fixture.runtime->set_parameter(
        edge_binding(*fixture.runtime, EdgeId{11}),
        ParameterValue{ContinuousValue{0.25}}).ok());
    ReferenceExecutor reference;
    for (double input : {1.0, 2.0, 3.0}) {
        assert(reference.step(*fixture.runtime, std::span<const double>(&input, 1)).ok());
    }
    const auto before_state = *fixture.runtime->cell_state(CellId{3});
    const std::vector<GraphEditEvent> events{
        GraphEditEvent{20, AddCellAction{CellBirth{
            CellId{40}, CellType::OP_EMA,
            {ParameterValue{ContinuousValue{0.5}},
             ParameterValue{UnusedParameter{}}}}}},
        GraphEditEvent{21, AddEdgeAction{EdgeBirth{
            EdgeId{40}, CellId{40}, OutputPort{0}, CellId{3}, InputPort{0},
            EdgeDelay::Immediate, 0.75}}},
    };
    GraphEditor editor(*fixture.runtime);
    const auto result = editor.apply(*fixture.runtime, events);
    assert(result.ok());
    assert(fixture.runtime->cell_state(CellId{40}) != nullptr);
    assert(!fixture.runtime->cell_state(CellId{40})->initialized);
    assert(fixture.runtime->cell_state(CellId{3})->state_val == before_state.state_val);
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{12}).index))
               .value == 2.5);
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{10}).index))
               .value == 2.5);
    assert(std::get<ContinuousValue>(
               *fixture.runtime->parameter_at(
                   edge_binding(*fixture.runtime, EdgeId{11}).index))
               .value == 0.25);
    const double input = 4.0;
    assert(!old_executor->step(
        *fixture.runtime, std::span<const double>(&input, 1)).ok());
    const auto rebound = old_executor->reprepare(result.graph);
    assert(rebound.ok());
    auto new_executor = std::move(rebound.executor);
    assert(new_executor->step(
        *fixture.runtime, std::span<const double>(&input, 1)).ok());
    assert(fixture.runtime->cell_state(CellId{40})->initialized);
}

void test_delay_ring_and_distinct_parallel_live_weights_survive() {
    GraphDefinition definition{
        GraphIdentity{904}, GraphRevision{1}, SemanticProfile::LegacyCompatible, 1,
        {
            {CellId{1}, CellType::SENSE_RAW_INPUT_0},
            {CellId{2}, CellType::OP_DELAY_N},
            {CellId{3}, CellType::OP_EMA},
        },
        {
            {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
             EdgeDelay::Immediate},
            {EdgeId{2}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
             EdgeDelay::Immediate},
            {EdgeId{3}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
             EdgeDelay::PreviousTick},
        }};
    auto seeds = make_seeds(definition);
    for (auto& cell : seeds.cell_parameters) {
        if (cell.cell == CellId{2} && cell.slot == ParameterSlot::Param1) {
            cell.value = DelayTicks{2};
        }
    }
    const auto compiled = GraphCompiler{}.compile(definition, seeds);
    assert(compiled.ok());
    auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    auto set_weight = [&](EdgeId id, double value) {
        assert(runtime->set_parameter(
            edge_binding(*runtime, id), ParameterValue{ContinuousValue{value}}).ok());
    };
    set_weight(EdgeId{1}, 2.5);
    set_weight(EdgeId{2}, -0.75);
    ReferenceExecutor reference;
    for (double input : {1.0, 2.0, 3.0}) {
        assert(reference.step(*runtime, std::span<const double>(&input, 1)).ok());
    }
    const auto before = *runtime->cell_state(CellId{2});
    GraphEditor editor(*runtime);
    const auto result = editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{40, AddCellAction{CellBirth{
                CellId{4}, CellType::OP_EMA,
                {ParameterValue{ContinuousValue{0.5}},
                 ParameterValue{UnusedParameter{}}}}}},
            GraphEditEvent{41, AddEdgeAction{EdgeBirth{
                EdgeId{4}, CellId{4}, OutputPort{0}, CellId{3}, InputPort{0},
                EdgeDelay::PreviousTick, 0.25}}}});
    assert(result.ok());
    const auto& after = *runtime->cell_state(CellId{2});
    assert(after.delay_idx == before.delay_idx);
    assert(after.delay_buffer == before.delay_buffer);
    assert(after.state_val == before.state_val);
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(edge_binding(*runtime, EdgeId{1}).index))
               .value == 2.5);
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(edge_binding(*runtime, EdgeId{2}).index))
               .value == -0.75);

    const auto split = editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{42, SplitEdgeAction{
                EdgeId{3},
                CellBirth{
                    CellId{5}, CellType::OP_EMA,
                    {ParameterValue{ContinuousValue{0.5}},
                     ParameterValue{UnusedParameter{}}}},
                EdgeId{5}, EdgeId{6}, InputPort{0}, 1.0, 1.0}}});
    assert(split.ok());
    assert(runtime->cell_state(CellId{2})->delay_buffer == before.delay_buffer);
    assert(runtime->cell_state(CellId{2})->delay_idx == before.delay_idx);
    assert(runtime->cell_state(CellId{5}) != nullptr);
    assert(!runtime->cell_state(CellId{5})->initialized);

    const auto removed = editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{43, RemoveCellAction{CellId{4}}}});
    assert(removed.ok());
    assert(runtime->cell_state(CellId{4}) == nullptr);
    assert(runtime->cell_state(CellId{2})->delay_buffer == before.delay_buffer);
    assert(runtime->cell_state(CellId{2})->delay_idx == before.delay_idx);
}

void test_empty_graph_and_retired_ids_are_explicit() {
    GraphDefinition definition{
        GraphIdentity{901}, GraphRevision{1}, SemanticProfile::LegacyCompatible, 1,
        {{CellId{1}, CellType::OP_SUM}}, {}};
    InitialParameterSeeds seeds;
    seeds.cell_parameters = {
        {CellId{1}, ParameterSlot::Param1, ParameterValue{UnusedParameter{}}},
        {CellId{1}, ParameterSlot::Param2, ParameterValue{UnusedParameter{}}}};
    const auto compiled = GraphCompiler{}.compile(definition, seeds);
    assert(compiled.ok());
    auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    GraphEditor editor(*runtime);
    const auto removed = editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{1, RemoveCellAction{CellId{1}}}});
    assert(removed.ok());
    assert(runtime->plan()->cells().empty());
    assert(runtime->plan()->edges().empty());
    const auto history = editor.history();
    GraphEditor resumed_editor(*runtime, history);
    const auto rebirth = resumed_editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{2, AddCellAction{CellBirth{
                CellId{1}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(!rebirth.ok());
    assert(rebirth.error->code == GraphEditErrorCode::RetiredId ||
           rebirth.error->code == GraphEditErrorCode::DuplicateId);
    assert(runtime->plan()->cells().empty());
}

void test_noop_and_invalid_batches_do_not_drift_revision_or_state() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto prepared = CompiledExecutor::prepare(fixture.runtime->plan());
    assert(prepared.ok());
    auto previous_executor = std::move(prepared.executor);
    const auto before = fixture.runtime->snapshot();
    const auto noop = editor.apply(*fixture.runtime, std::span<const GraphEditEvent>{});
    assert(noop.ok());
    assert(noop.report.no_op);
    assert(fixture.runtime->revision() == GraphRevision{1});
    assert(fixture.runtime->cell_states().size() == before.cells().size());

    const auto invalid = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{3, AddCellAction{CellBirth{
                CellId{50}, CellType::OP_EMA,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(!invalid.ok());
    assert(invalid.error->code == GraphEditErrorCode::InvalidParameter);
    assert(fixture.runtime->revision() == GraphRevision{1});
    assert(fixture.runtime->cell_states().size() == before.cells().size());
    assert(fixture.runtime->parameters().size() == before.parameters().size());
    assert_same_snapshot(before, fixture.runtime->snapshot());

    const auto cycle = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{4, AddEdgeAction{EdgeBirth{
                EdgeId{90}, CellId{3}, OutputPort{0}, CellId{1}, InputPort{0},
                EdgeDelay::Immediate, 1.0}}}});
    assert(!cycle.ok());
    assert(cycle.error->code == GraphEditErrorCode::CompilationFailed);
    assert(fixture.runtime->revision() == GraphRevision{1});
    assert_same_snapshot(before, fixture.runtime->snapshot());
    const double input = 1.0;
    assert(previous_executor->step(
        *fixture.runtime, std::span<const double>(&input, 1)).ok());
}

void test_parallel_edges_and_more_than_sixty_four_cells_are_supported() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto parallel = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{5, AddEdgeAction{EdgeBirth{
                EdgeId{13}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
                EdgeDelay::Immediate, 0.25}}}});
    assert(parallel.ok());
    bool saw10 = false;
    bool saw11 = false;
    bool saw13 = false;
    for (const auto& edge : parallel.graph->edges()) {
        saw10 |= edge.id == EdgeId{10};
        saw11 |= edge.id == EdgeId{11};
        saw13 |= edge.id == EdgeId{13};
    }
    assert(saw10 && saw11 && saw13);

    GraphDefinition large{
        GraphIdentity{902}, GraphRevision{1}, SemanticProfile::LegacyCompatible, 1,
        {}, {}};
    InitialParameterSeeds seeds;
    for (uint64_t id = 1; id <= 65; ++id) {
        large.cells.push_back({CellId{id}, CellType::OP_SUM});
        seeds.cell_parameters.push_back(
            {CellId{id}, ParameterSlot::Param1, ParameterValue{UnusedParameter{}}});
        seeds.cell_parameters.push_back(
            {CellId{id}, ParameterSlot::Param2, ParameterValue{UnusedParameter{}}});
    }
    const auto compiled = GraphCompiler{}.compile(large, seeds);
    assert(compiled.ok());
    auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    GraphEditor large_editor(*runtime);
    const auto added = large_editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{6, AddCellAction{CellBirth{
                CellId{66}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(added.ok());
    assert(runtime->plan()->cells().size() == 66);
}

void test_rejections_cover_identity_ports_weights_and_exhaustion() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto duplicate_event = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{7, AddCellAction{CellBirth{
                CellId{70}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}},
            GraphEditEvent{7, AddCellAction{CellBirth{
                CellId{71}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(!duplicate_event.ok());
    assert(duplicate_event.error->code == GraphEditErrorCode::DuplicateEventId);

    const auto missing = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{8, RemoveEdgeAction{EdgeId{999}}}});
    assert(!missing.ok());
    assert(missing.error->code == GraphEditErrorCode::MissingObject);

    const auto bad_port = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{9, AddEdgeAction{EdgeBirth{
                EdgeId{91}, CellId{1}, OutputPort{1}, CellId{3}, InputPort{0},
                EdgeDelay::Immediate, 1.0}}}});
    assert(!bad_port.ok());
    assert(bad_port.error->code == GraphEditErrorCode::InvalidPort);

    const auto bad_weight = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{10, AddEdgeAction{EdgeBirth{
                EdgeId{92}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
                EdgeDelay::Immediate,
                std::numeric_limits<double>::quiet_NaN()}}}});
    assert(!bad_weight.ok());
    assert(bad_weight.error->code == GraphEditErrorCode::NonFiniteSeed);

    const auto duplicate_id = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{11, AddCellAction{CellBirth{
                CellId{1}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(!duplicate_id.ok());
    assert(duplicate_id.error->code == GraphEditErrorCode::DuplicateId);

    const auto max_id = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{12, AddCellAction{CellBirth{
                CellId{std::numeric_limits<uint64_t>::max()},
                CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(max_id.ok());
    assert(fixture.runtime->cell_state(
               CellId{std::numeric_limits<uint64_t>::max()}) != nullptr);

    GraphDefinition max_revision = make_graph(
        GraphRevision{std::numeric_limits<uint64_t>::max()});
    const auto max_compiled =
        GraphCompiler{}.compile(max_revision, make_seeds(max_revision));
    assert(max_compiled.ok());
    auto max_runtime_result =
        RuntimeState::create(max_compiled.graph, max_compiled.initial_values);
    assert(max_runtime_result.ok());
    auto max_runtime = std::move(max_runtime_result.runtime);
    GraphEditor max_editor(*max_runtime);
    const auto max_before = max_runtime->snapshot();
    const auto max_noop =
        max_editor.apply(*max_runtime, std::span<const GraphEditEvent>{});
    assert(max_noop.ok());
    assert(max_noop.report.no_op);
    assert_same_snapshot(max_before, max_runtime->snapshot());
    const auto exhausted_revision = max_editor.apply(
        *max_runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{13, AddCellAction{CellBirth{
                CellId{100}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(!exhausted_revision.ok());
    assert(exhausted_revision.error->code == GraphEditErrorCode::RevisionExhausted);

    GraphDefinition other = make_graph();
    other.identity = GraphIdentity{903};
    const auto other_compiled =
        GraphCompiler{}.compile(other, make_seeds(other));
    assert(other_compiled.ok());
    auto other_runtime_result =
        RuntimeState::create(other_compiled.graph, other_compiled.initial_values);
    assert(other_runtime_result.ok());
    auto other_runtime = std::move(other_runtime_result.runtime);
    const auto wrong_source = editor.apply(
        *other_runtime,
        std::vector<GraphEditEvent>{});
    assert(!wrong_source.ok());
    assert(wrong_source.error->code == GraphEditErrorCode::SourceMismatch);
}

void test_independent_event_order_is_deterministic() {
    auto left = make_fixture();
    auto right = make_fixture();
    const GraphEditEvent add_a{30, AddCellAction{CellBirth{
        CellId{60}, CellType::OP_SUM,
        {ParameterValue{UnusedParameter{}},
         ParameterValue{UnusedParameter{}}}}}};
    const GraphEditEvent add_b{31, AddCellAction{CellBirth{
        CellId{61}, CellType::OP_EMA,
        {ParameterValue{ContinuousValue{0.5}},
         ParameterValue{UnusedParameter{}}}}}};
    GraphEditor left_editor(*left.runtime);
    GraphEditor right_editor(*right.runtime);
    const auto left_result = left_editor.apply(
        *left.runtime, std::vector<GraphEditEvent>{add_a, add_b});
    const auto right_result = right_editor.apply(
        *right.runtime, std::vector<GraphEditEvent>{add_b, add_a});
    assert(left_result.ok());
    assert(right_result.ok());
    assert(detail::same_plan(*left_result.graph, *right_result.graph));
    assert(left_result.report.diagnostics.size() == 2);
    assert(left_result.report.diagnostics[0].event_id == 30);
    assert(left_result.report.diagnostics[1].event_id == 31);
}

void test_split_birth_ids_are_checked_against_source_and_history() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto occupied = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{70, SplitEdgeAction{
                EdgeId{10},
                CellBirth{
                    CellId{3}, CellType::OP_SUM,
                    {ParameterValue{UnusedParameter{}},
                     ParameterValue{UnusedParameter{}}}},
                EdgeId{70}, EdgeId{71}, InputPort{0}, 1.0, 1.0}}});
    assert(!occupied.ok());
    assert(occupied.error->code == GraphEditErrorCode::DuplicateId);
    assert(occupied.error->event_id == 70);

    const auto first = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{71, SplitEdgeAction{
                EdgeId{10},
                CellBirth{
                    CellId{20}, CellType::OP_SUM,
                    {ParameterValue{UnusedParameter{}},
                     ParameterValue{UnusedParameter{}}}},
                EdgeId{72}, EdgeId{73}, InputPort{0}, 1.0, 1.0}}});
    assert(first.ok());
    const auto removed = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{72, RemoveCellAction{CellId{20}}}});
    assert(removed.ok());
    const auto history = editor.history();
    GraphEditor resumed(*fixture.runtime, history);
    const auto retired_split = resumed.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{73, SplitEdgeAction{
                EdgeId{11},
                CellBirth{
                    CellId{20}, CellType::OP_EMA,
                    {ParameterValue{ContinuousValue{0.5}},
                     ParameterValue{UnusedParameter{}}}},
                EdgeId{74}, EdgeId{75}, InputPort{0}, 1.0, 1.0}}});
    assert(!retired_split.ok());
    assert(retired_split.error->code == GraphEditErrorCode::RetiredId);
    assert(retired_split.error->event_id == 73);

    auto same_batch = make_fixture();
    GraphEditor same_batch_editor(*same_batch.runtime);
    const auto remove_and_rebirth = same_batch_editor.apply(
        *same_batch.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{74, RemoveCellAction{CellId{3}}},
            GraphEditEvent{75, SplitEdgeAction{
                EdgeId{10},
                CellBirth{
                    CellId{3}, CellType::OP_SUM,
                    {ParameterValue{UnusedParameter{}},
                     ParameterValue{UnusedParameter{}}}},
                EdgeId{76}, EdgeId{77}, InputPort{0}, 1.0, 1.0}}});
    assert(!remove_and_rebirth.ok());
    assert(remove_and_rebirth.error->code == GraphEditErrorCode::ConflictingRequest ||
           remove_and_rebirth.error->code == GraphEditErrorCode::DuplicateId);
    assert(!remove_and_rebirth.error->event_id.has_value());
}

void test_source_binding_checks_full_immutable_plan_not_pointer_identity() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);

    GraphDefinition rewired = make_graph();
    rewired.edges[2].delay = EdgeDelay::PreviousTick;
    const auto rewired_compiled =
        GraphCompiler{}.compile(rewired, make_seeds(rewired));
    assert(rewired_compiled.ok());
    auto rewired_runtime_result =
        RuntimeState::create(rewired_compiled.graph, rewired_compiled.initial_values);
    assert(rewired_runtime_result.ok());
    auto rewired_runtime = std::move(rewired_runtime_result.runtime);
    const auto rewired_result =
        editor.apply(*rewired_runtime, std::span<const GraphEditEvent>{});
    assert(!rewired_result.ok());
    assert(rewired_result.error->code == GraphEditErrorCode::SourceMismatch);

    GraphDefinition strict = make_graph();
    strict.profile = SemanticProfile::StrictCore;
    const auto strict_compiled =
        GraphCompiler{}.compile(strict, make_seeds(strict));
    assert(strict_compiled.ok());
    auto strict_runtime_result =
        RuntimeState::create(strict_compiled.graph, strict_compiled.initial_values);
    assert(strict_runtime_result.ok());
    auto strict_runtime = std::move(strict_runtime_result.runtime);
    const auto strict_result =
        editor.apply(*strict_runtime, std::span<const GraphEditEvent>{});
    assert(!strict_result.ok());
    assert(strict_result.error->code == GraphEditErrorCode::SourceMismatch);

    auto equivalent = fixture.runtime->fork_probe();
    assert(equivalent.ok());
    const auto equivalent_result =
        editor.apply(*equivalent.runtime, std::span<const GraphEditEvent>{});
    assert(equivalent_result.ok());

    GraphEditHistory inconsistent;
    inconsistent.retired_cells.push_back(CellId{1});
    GraphEditor invalid_history(*fixture.runtime, inconsistent);
    const auto history_result =
        invalid_history.apply(*fixture.runtime, std::span<const GraphEditEvent>{});
    assert(!history_result.ok());
    assert(history_result.error->code == GraphEditErrorCode::InvalidHistory);
}

void test_error_provenance_and_incident_edge_diagnostics_are_complete() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto invalid = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{77, AddEdgeAction{EdgeBirth{
                EdgeId{177}, CellId{999}, OutputPort{0}, CellId{3}, InputPort{0},
                EdgeDelay::Immediate, 1.0}}}});
    assert(!invalid.ok());
    assert(invalid.error->event_id == 77);
    assert(invalid.report.diagnostics.size() == 1);
    assert(invalid.report.diagnostics[0].event_id == 77);

    const auto before = fixture.runtime->snapshot();
    const auto removed = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{78, RemoveCellAction{CellId{2}}}});
    assert(removed.ok());
    assert(removed.report.diagnostics.size() == 1);
    const auto& diagnostic = removed.report.diagnostics[0];
    assert(diagnostic.event_id == 78);
    assert(diagnostic.cells.size() == 3);
    assert(diagnostic.cells[0] == CellId{1});
    assert(diagnostic.cells[1] == CellId{2});
    assert(diagnostic.cells[2] == CellId{3});
    assert(diagnostic.edges.size() == 3);
    assert(diagnostic.edges[0] == EdgeId{10});
    assert(diagnostic.edges[1] == EdgeId{11});
    assert(diagnostic.edges[2] == EdgeId{12});
    assert(before.plan()->edges().size() == 3);
}

void test_invalid_multi_event_requests_report_the_offending_event() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const GraphEditEvent valid_add{76, AddCellAction{CellBirth{
        CellId{76}, CellType::OP_SUM,
        {ParameterValue{UnusedParameter{}},
         ParameterValue{UnusedParameter{}}}}}};
    const GraphEditEvent bad_source_port{77, AddEdgeAction{EdgeBirth{
        EdgeId{177}, CellId{1}, OutputPort{1}, CellId{3}, InputPort{0},
        EdgeDelay::Immediate, 1.0}}};

    const auto forward = editor.apply(
        *fixture.runtime, std::vector<GraphEditEvent>{valid_add, bad_source_port});
    assert(!forward.ok());
    assert(forward.error->code == GraphEditErrorCode::InvalidPort);
    assert(forward.error->event_id == 77);

    const auto reversed = editor.apply(
        *fixture.runtime, std::vector<GraphEditEvent>{bad_source_port, valid_add});
    assert(!reversed.ok());
    assert(reversed.error->code == GraphEditErrorCode::InvalidPort);
    assert(reversed.error->event_id == 77);

    const GraphEditEvent bad_split_port{77, SplitEdgeAction{
        EdgeId{10},
        CellBirth{
            CellId{77}, CellType::OP_SUM,
            {ParameterValue{UnusedParameter{}},
             ParameterValue{UnusedParameter{}}}},
        EdgeId{178}, EdgeId{179}, InputPort{2}, 1.0, 1.0}};
    const auto split_port = editor.apply(
        *fixture.runtime, std::vector<GraphEditEvent>{valid_add, bad_split_port});
    assert(!split_port.ok());
    assert(split_port.error->code == GraphEditErrorCode::InvalidSplit);
    assert(split_port.error->event_id == 77);

    const GraphEditEvent bad_split_weight{77, SplitEdgeAction{
        EdgeId{10},
        CellBirth{
            CellId{78}, CellType::OP_SUM,
            {ParameterValue{UnusedParameter{}},
             ParameterValue{UnusedParameter{}}}},
        EdgeId{180}, EdgeId{181}, InputPort{0},
        std::numeric_limits<double>::quiet_NaN(), 1.0}};
    const auto split_weight = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{valid_add, bad_split_weight});
    assert(!split_weight.ok());
    assert(split_weight.error->code == GraphEditErrorCode::NonFiniteSeed);
    assert(split_weight.error->event_id == 77);
}

void test_rejected_duplicate_requests_do_not_expand_incident_diagnostics() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto result = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{90, RemoveCellAction{CellId{2}}},
            GraphEditEvent{91, RemoveCellAction{CellId{2}}}});
    assert(!result.ok());
    assert(result.error->code == GraphEditErrorCode::DuplicateId);
    assert(result.report.diagnostics.size() == 2);
    for (const auto& diagnostic : result.report.diagnostics) {
        assert(diagnostic.cells.size() == 1);
        assert(diagnostic.cells[0] == CellId{2});
        assert(diagnostic.edges.empty());
    }
}

void test_uint64_max_ids_are_valid_birth_ids() {
    auto fixture = make_fixture();
    GraphEditor editor(*fixture.runtime);
    const auto result = editor.apply(
        *fixture.runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{79, AddCellAction{CellBirth{
                CellId{std::numeric_limits<uint64_t>::max()},
                CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}}}}});
    assert(result.ok());
    assert(fixture.runtime->cell_state(
               CellId{std::numeric_limits<uint64_t>::max()}) != nullptr);
}

void test_all_valid_immediate_cycle_is_rejected_as_cycle() {
    GraphDefinition definition{
        GraphIdentity{905}, GraphRevision{1}, SemanticProfile::LegacyCompatible, 1,
        {{CellId{1}, CellType::OP_SUM}, {CellId{2}, CellType::OP_SUM}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}}};
    InitialParameterSeeds seeds;
    for (const auto id : {CellId{1}, CellId{2}}) {
        seeds.cell_parameters.push_back(
            {id, ParameterSlot::Param1, ParameterValue{UnusedParameter{}}});
        seeds.cell_parameters.push_back(
            {id, ParameterSlot::Param2, ParameterValue{UnusedParameter{}}});
    }
    seeds.edge_weights.push_back({EdgeId{1}, 1.0});
    const auto compiled = GraphCompiler{}.compile(definition, seeds);
    assert(compiled.ok());
    auto runtime_result =
        RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    GraphEditor editor(*runtime);
    const auto result = editor.apply(
        *runtime,
        std::vector<GraphEditEvent>{
            GraphEditEvent{80, AddEdgeAction{EdgeBirth{
                EdgeId{2}, CellId{2}, OutputPort{0}, CellId{1}, InputPort{0},
                EdgeDelay::Immediate, 1.0}}}});
    assert(!result.ok());
    assert(result.error->code == GraphEditErrorCode::CompilationFailed);
    assert(result.error->event_id == 80);
    assert(runtime->revision() == GraphRevision{1});
}

double measured_output(std::span<const ExecutedCellMeasurement> cells, CellId id) {
    for (const auto& cell : cells) {
        if (cell.cell == id) return cell.output;
    }
    assert(false);
    return 0.0;
}

void test_split_numeric_oracle_matches_unsplit_for_both_profiles() {
    for (const auto profile : {
             SemanticProfile::LegacyCompatible,
             SemanticProfile::StrictCore}) {
        GraphDefinition definition{
            GraphIdentity{906}, GraphRevision{1}, profile, 1,
            {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
             {CellId{2}, CellType::OP_EMA}},
            {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
              EdgeDelay::PreviousTick}}};
        auto seeds = make_seeds(definition);
        for (auto& seed : seeds.cell_parameters) {
            if (seed.cell == CellId{1} && seed.slot == ParameterSlot::Param1) {
                seed.value = ContinuousValue{1.0};
            }
        }
        const auto compiled = GraphCompiler{}.compile(definition, seeds);
        assert(compiled.ok());
        auto baseline_result =
            RuntimeState::create(compiled.graph, compiled.initial_values);
        auto split_result =
            RuntimeState::create(compiled.graph, compiled.initial_values);
        assert(baseline_result.ok());
        assert(split_result.ok());
        auto baseline = std::move(baseline_result.runtime);
        auto split = std::move(split_result.runtime);
        GraphEditor editor(*split);
        const auto edited = editor.apply(
            *split,
            std::vector<GraphEditEvent>{
                GraphEditEvent{81, SplitEdgeAction{
                    EdgeId{10},
                    CellBirth{
                        CellId{3}, CellType::OP_SUM,
                        {ParameterValue{UnusedParameter{}},
                         ParameterValue{UnusedParameter{}}}},
                    EdgeId{11}, EdgeId{12}, InputPort{0}, 1.0, 1.0}}});
        assert(edited.ok());
        bool old_edge_absent = true;
        bool delayed_edge_exact = false;
        bool immediate_edge_exact = false;
        for (const auto& edge : edited.graph->edges()) {
            old_edge_absent &= edge.id != EdgeId{10};
            delayed_edge_exact |=
                edge.id == EdgeId{11} &&
                edge.source_index == 0 &&
                edge.target_index == 2 &&
                edge.delay == EdgeDelay::PreviousTick;
            immediate_edge_exact |=
                edge.id == EdgeId{12} &&
                edge.source_index == 2 &&
                edge.target_index == 1 &&
                edge.delay == EdgeDelay::Immediate;
        }
        assert(old_edge_absent && delayed_edge_exact && immediate_edge_exact);

        auto baseline_probe = baseline->fork_probe();
        auto split_probe = split->fork_probe();
        assert(baseline_probe.ok());
        assert(split_probe.ok());
        const auto baseline_prepared = CompiledExecutor::prepare(baseline->plan());
        const auto split_prepared = CompiledExecutor::prepare(split->plan());
        assert(baseline_prepared.ok());
        assert(split_prepared.ok());
        auto baseline_executor = std::move(baseline_prepared.executor);
        auto split_executor = std::move(split_prepared.executor);

        for (const double input : {1.0, 2.0, 3.0}) {
            const auto reference_baseline = ReferenceExecutor{}.step(
                *baseline, std::span<const double>(&input, 1));
            const auto reference_split = ReferenceExecutor{}.step(
                *split, std::span<const double>(&input, 1));
            assert(reference_baseline.ok());
            assert(reference_split.ok());
            const double reference_unsplit_output =
                measured_output(reference_baseline.measurement.cells, CellId{2});
            const double reference_split_output =
                measured_output(reference_split.measurement.cells, CellId{2});
            assert(std::abs(reference_unsplit_output - reference_split_output) < 1e-9);
            const double expected_target[] =
                {0.0, 1.0, 1.5};
            const double expected_strict_target[] =
                {0.0, 0.5, 1.25};
            const double expected_inserted[] = {0.0, 1.0, 2.0};
            const std::size_t tick = static_cast<std::size_t>(
                baseline->tick() - 1);
            const double target_expected =
                profile == SemanticProfile::LegacyCompatible
                    ? expected_target[tick]
                    : expected_strict_target[tick];
            assert(std::abs(reference_unsplit_output - target_expected) < 1e-9);
            assert(std::abs(reference_split_output - target_expected) < 1e-9);
            const double reference_inserted_output =
                measured_output(reference_split.measurement.cells, CellId{3});
            assert(std::abs(reference_inserted_output - expected_inserted[tick]) < 1e-9);

            const auto compiled_baseline = baseline_executor->step(
                *baseline_probe.runtime, std::span<const double>(&input, 1));
            const auto compiled_split = split_executor->step(
                *split_probe.runtime, std::span<const double>(&input, 1));
            assert(compiled_baseline.ok());
            assert(compiled_split.ok());
            const double compiled_unsplit_output =
                measured_output(compiled_baseline.measurement.cells, CellId{2});
            const double compiled_split_output =
                measured_output(compiled_split.measurement.cells, CellId{2});
            assert(std::abs(compiled_unsplit_output - compiled_split_output) < 1e-9);
            assert(std::abs(reference_unsplit_output - compiled_unsplit_output) < 1e-9);
            assert(std::abs(reference_split_output - compiled_split_output) < 1e-9);
            assert(std::abs(
                       measured_output(compiled_split.measurement.cells, CellId{3}) -
                       expected_inserted[tick]) < 1e-9);
        }
    }
}

void test_allocation_failure_never_publishes_partial_edit() {
    const std::vector<GraphEditEvent> events{
        GraphEditEvent{82, RemoveEdgeAction{EdgeId{11}}},
        GraphEditEvent{83, SplitEdgeAction{
            EdgeId{10},
            CellBirth{
                CellId{82}, CellType::OP_SUM,
                {ParameterValue{UnusedParameter{}},
                 ParameterValue{UnusedParameter{}}}},
            EdgeId{182}, EdgeId{183}, InputPort{0}, 0.75, 1.25}}};

    auto success_fixture = make_fixture();
    assert(ReferenceExecutor{}.step(
        *success_fixture.runtime,
        std::span<const double>(std::array<double, 1>{2.0})).ok());
    assert(success_fixture.runtime->set_parameter(
        edge_binding(*success_fixture.runtime, EdgeId{10}),
        ParameterValue{ContinuousValue{2.5}}).ok());
    GraphEditor success_editor(*success_fixture.runtime);
    allocation_count.store(0, std::memory_order_relaxed);
    allocation_fail_after.store(-1, std::memory_order_relaxed);
    const auto success = success_editor.apply(*success_fixture.runtime, events);
    const long long success_allocations =
        allocation_count.load(std::memory_order_relaxed);
    assert(success.ok());
    assert(success_allocations > 0);
    assert(!success_editor.history().retired_edges.empty());
    allocation_fail_after.store(-1, std::memory_order_relaxed);

    bool observed_injected_failure = false;
    bool observed_success_boundary = false;
    for (long long fail_after = 0; fail_after <= success_allocations; ++fail_after) {
        auto fixture = make_fixture();
        assert(ReferenceExecutor{}.step(
            *fixture.runtime,
            std::span<const double>(std::array<double, 1>{2.0})).ok());
        assert(fixture.runtime->set_parameter(
            edge_binding(*fixture.runtime, EdgeId{10}),
            ParameterValue{ContinuousValue{2.5}}).ok());
        GraphEditor editor(*fixture.runtime);
        const auto before = fixture.runtime->snapshot();
        allocation_count.store(0, std::memory_order_relaxed);
        allocation_fail_after.store(fail_after, std::memory_order_relaxed);
        bool threw = false;
        try {
            (void)editor.apply(*fixture.runtime, events);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        allocation_fail_after.store(-1, std::memory_order_relaxed);
        if (threw) {
            observed_injected_failure = true;
            assert_same_snapshot(before, fixture.runtime->snapshot());
            assert(editor.history().retired_cells.empty());
            assert(editor.history().retired_edges.empty());
            const auto retry = editor.apply(*fixture.runtime, events);
            assert(retry.ok());
            assert(!editor.history().retired_edges.empty());
            const auto reuse = editor.apply(
                *fixture.runtime,
                std::vector<GraphEditEvent>{
                    GraphEditEvent{84, AddEdgeAction{EdgeBirth{
                        EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2},
                        InputPort{0}, EdgeDelay::Immediate, 1.0}}}});
            assert(!reuse.ok());
            assert(reuse.error->code == GraphEditErrorCode::RetiredId);
            const auto reuse_removed = editor.apply(
                *fixture.runtime,
                std::vector<GraphEditEvent>{
                    GraphEditEvent{85, AddEdgeAction{EdgeBirth{
                        EdgeId{11}, CellId{1}, OutputPort{0}, CellId{2},
                        InputPort{1}, EdgeDelay::PreviousTick, 1.0}}}});
            assert(!reuse_removed.ok());
            assert(reuse_removed.error->code == GraphEditErrorCode::RetiredId);
        } else {
            observed_success_boundary = true;
            assert(fail_after == success_allocations);
            assert(editor.history().retired_edges.size() == 2);
            break;
        }
    }
    assert(observed_injected_failure);
    assert(observed_success_boundary);
}

void test_committed_runtime_owns_candidate_after_cold_owners_die() {
    auto fixture = make_fixture();
    std::shared_ptr<const CompiledGraph> committed_plan;
    {
        GraphEditor editor(*fixture.runtime);
        auto result = editor.apply(
            *fixture.runtime,
            std::vector<GraphEditEvent>{
                GraphEditEvent{84, AddCellAction{CellBirth{
                    CellId{84}, CellType::OP_SUM,
                    {ParameterValue{UnusedParameter{}},
                     ParameterValue{UnusedParameter{}}}}}}});
        assert(result.ok());
        committed_plan = result.graph;
    }
    assert(committed_plan);
    assert(fixture.runtime->bound_to(*committed_plan));
    const auto prepared = CompiledExecutor::prepare(committed_plan);
    assert(prepared.ok());
    auto executor = std::move(prepared.executor);
    const double input = 1.0;
    assert(executor->step(
        *fixture.runtime, std::span<const double>(&input, 1)).ok());
    assert(fixture.runtime->cell_state(CellId{84})->initialized);
}

}  // namespace

int main() {
    test_split_selected_edges_preserves_time_and_executes_new_cell();
    test_failed_batch_preserves_runtime_and_rejects_receptor_split();
    test_remove_commits_then_invalid_growth_cannot_restore_ids();
    test_live_state_and_weights_survive_and_old_executor_is_stale();
    test_delay_ring_and_distinct_parallel_live_weights_survive();
    test_empty_graph_and_retired_ids_are_explicit();
    test_noop_and_invalid_batches_do_not_drift_revision_or_state();
    test_parallel_edges_and_more_than_sixty_four_cells_are_supported();
    test_rejections_cover_identity_ports_weights_and_exhaustion();
    test_independent_event_order_is_deterministic();
    test_split_birth_ids_are_checked_against_source_and_history();
    test_source_binding_checks_full_immutable_plan_not_pointer_identity();
    test_error_provenance_and_incident_edge_diagnostics_are_complete();
    test_invalid_multi_event_requests_report_the_offending_event();
    test_rejected_duplicate_requests_do_not_expand_incident_diagnostics();
    test_uint64_max_ids_are_valid_birth_ids();
    test_all_valid_immediate_cycle_is_rejected_as_cycle();
    test_split_numeric_oracle_matches_unsplit_for_both_profiles();
    test_allocation_failure_never_publishes_partial_edit();
    test_committed_runtime_owns_candidate_after_cold_owners_die();
}
