#include "kun/cellular/core/reference_executor.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/legacy/runtime_adapter.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <vector>

using namespace kun;
using namespace kun::core;

namespace {

GraphDefinition one_input_graph() {
    GraphDefinition graph;
    graph.identity = GraphIdentity{101};
    graph.revision = GraphRevision{7};
    graph.cells = {{CellId{10}, CellType::SENSE_RAW_INPUT_0}};
    return graph;
}

InitialParameterSeeds one_input_seeds() {
    return InitialParameterSeeds{
        {{CellId{10}, ParameterSlot::Param1, ContinuousValue{2.0}},
         {CellId{10}, ParameterSlot::Param2, UnusedParameter{}}},
        {}};
}

InitialParameterSeeds seeds_for(const GraphDefinition& graph) {
    InitialParameterSeeds seeds;
    for (const auto& cell : graph.cells) {
        const auto& contract = contract_for(cell.type)->get();
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract.parameters[slot];
            ParameterValue value = UnusedParameter{};
            switch (descriptor.value_type) {
                case ParameterValueType::Continuous:
                    value = ContinuousValue{1.0};
                    break;
                case ParameterValueType::ChannelIndex:
                    value = ChannelIndex{43};
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

GraphDefinition graph_with(
    std::vector<CellDefinition> cells,
    std::vector<EdgeDefinition> edges = {}) {
    GraphDefinition graph;
    graph.identity = GraphIdentity{101};
    graph.revision = GraphRevision{7};
    graph.cells = std::move(cells);
    graph.edges = std::move(edges);
    return graph;
}

void assert_near(double actual, double expected, double epsilon = 1e-12) {
    assert(std::abs(actual - expected) <= epsilon);
}

void assert_runtime_cell_matches(
    const RuntimeCellState& actual,
    const Cell& expected) {
    assert(actual.cell.value == expected.id);
    assert(actual.type == expected.type);
    assert(actual.initialized);
    assert_near(actual.state_val, expected.state_val);
    assert_near(actual.aux_state, expected.aux_state);
    assert_near(actual.prev_input, expected.prev_input);
    assert_near(actual.output_val, expected.output_val);
    assert_near(actual.prev_output_val, expected.prev_output_val);
    for (std::size_t i = 0; i < 16; ++i) {
        assert_near(actual.delay_buffer[i], expected.delay_buffer[i]);
    }
    assert(actual.delay_idx == expected.delay_idx);
    assert(actual.latch_state == expected.latch_state);
    assert(actual.activation_count == expected.activation_count);
}

void assert_runtime_state_equal(
    const RuntimeCellState& actual,
    const RuntimeCellState& expected) {
    assert(actual.cell == expected.cell);
    assert(actual.type == expected.type);
    assert_near(actual.state_val, expected.state_val);
    assert_near(actual.aux_state, expected.aux_state);
    assert_near(actual.prev_input, expected.prev_input);
    assert_near(actual.output_val, expected.output_val);
    assert_near(actual.prev_output_val, expected.prev_output_val);
    assert(actual.delay_buffer == expected.delay_buffer);
    assert(actual.delay_idx == expected.delay_idx);
    assert(actual.latch_state == expected.latch_state);
    assert(actual.activation_count == expected.activation_count);
    assert(actual.initialized == expected.initialized);
}

ParameterValue typed_parameter(
    const CellContract& contract,
    ParameterSlot slot,
    double raw) {
    const auto& descriptor = contract.parameters[static_cast<std::size_t>(slot)];
    switch (descriptor.value_type) {
        case ParameterValueType::Continuous:
            return ContinuousValue{raw};
        case ParameterValueType::ChannelIndex:
            return ChannelIndex{static_cast<std::size_t>(raw)};
        case ParameterValueType::DelayTicks:
            return DelayTicks{static_cast<uint64_t>(raw * 16.0)};
        case ParameterValueType::MinMaxMode:
            return raw > 0.5 ? ParameterValue{MinMaxMode::Max}
                             : ParameterValue{MinMaxMode::Min};
        case ParameterValueType::Unused:
            return UnusedParameter{};
    }
    return UnusedParameter{};
}

std::pair<double, double> parameters_for_type(CellType type) {
    switch (type) {
        case CellType::SENSE_CHANNEL:
        case CellType::ACT_CHANNEL:
            return {1.0, 43.0};
        case CellType::OP_EMA:
            return {0.5, 0.0};
        case CellType::OP_DELAY_N:
            return {1.0 / 16.0, 0.0};
        case CellType::OP_OSCILLATOR:
            return {1.0, 0.05};
        case CellType::OP_QUADRATIC:
            return {0.5, 0.25};
        case CellType::GATE_THRESHOLD:
            return {0.25, 0.0};
        case CellType::GATE_HYSTERESIS:
            return {0.5, -0.5};
        case CellType::GATE_DEADZONE:
            return {0.1, 0.0};
        case CellType::GATE_MIN_MAX:
            return {1.0, 0.0};
        default:
            return {1.0, 0.0};
    }
}

void test_runtime_executes_and_resets_without_resetting_parameters() {
    const auto compiled = GraphCompiler{}.compile(one_input_graph(), one_input_seeds());
    assert(compiled.ok());
    auto runtime_result = RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    ReferenceExecutor executor;

    const double input = 3.0;
    const auto frame = executor.step(*runtime, std::span<const double>(&input, 1));
    assert(frame.ok());
    assert(runtime->cell_state(CellId{10})->output_val == 6.0);

    runtime->reset_episode();
    assert(runtime->cell_state(CellId{10})->output_val == 0.0);
    const auto parameter =
        runtime->parameter(CellId{10}, ParameterSlot::Param1);
    assert(parameter.has_value());
    assert(std::holds_alternative<ContinuousValue>(*parameter));
    assert(std::get<ContinuousValue>(*parameter).value == 2.0);
}

void test_binding_identity_and_strict_profile_are_supported() {
    const auto compiled = GraphCompiler{}.compile(one_input_graph(), one_input_seeds());
    assert(compiled.ok());
    auto valid_runtime = RuntimeState::create(
        compiled.graph, compiled.initial_values);
    assert(valid_runtime.ok());
    assert(valid_runtime.runtime->set_continuous_parameter(
        CellId{10}, ParameterSlot::Param1, 1.5).ok());
    assert(!valid_runtime.runtime->set_continuous_parameter(
        CellId{10}, ParameterSlot::Param2, 1.5).ok());
    auto unrelated = std::vector<InitialParameterValue>(
        compiled.initial_values->entries().begin(),
        compiled.initial_values->entries().end());
    unrelated[0].binding.cell = CellId{999};
    auto unrelated_values =
        std::make_shared<const InitialParameterValues>(std::move(unrelated));
    assert(!RuntimeState::create(compiled.graph, unrelated_values).ok());

    auto strict_graph = one_input_graph();
    strict_graph.profile = SemanticProfile::StrictCore;
    const auto strict = GraphCompiler{}.compile(strict_graph, one_input_seeds());
    assert(strict.ok());
    auto strict_runtime =
        RuntimeState::create(strict.graph, strict.initial_values);
    assert(strict_runtime.ok());
    const double strict_input = 3.0;
    assert(ReferenceExecutor{}.step(
        *strict_runtime.runtime,
        std::span<const double>(&strict_input, 1))
        .ok());
    assert(strict_runtime.runtime->cell_state(CellId{10})->initialized);
}

void test_invalid_parameter_enums_and_nonfinite_updates_are_rejected() {
    const auto compiled = GraphCompiler{}.compile(one_input_graph(), one_input_seeds());
    assert(compiled.ok());
    auto runtime_result = RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);

    const auto invalid_slot_two = static_cast<ParameterSlot>(2);
    assert(!runtime->parameter(CellId{10}, invalid_slot_two).has_value());
    assert(!runtime->set_continuous_parameter(
        CellId{10}, invalid_slot_two, 1.0).ok());
    const auto invalid_slot = static_cast<ParameterSlot>(255);
    assert(!runtime->parameter(CellId{10}, invalid_slot).has_value());
    assert(!runtime->set_continuous_parameter(
        CellId{10}, invalid_slot, 1.0).ok());
    assert(!runtime->set_parameter(
        ParameterBinding{
            ParameterBindingKind::CellParameter, 0, CellId{10}, EdgeId{0},
            invalid_slot},
        ParameterValue{ContinuousValue{1.0}})
        .ok());
    const auto parameter_binding = runtime->parameters()[0].binding;
    assert(!runtime->set_parameter(
        parameter_binding, ParameterValue{ChannelIndex{43}}).ok());
    assert(!legacy_kernel_parameter(
        ParameterValue{static_cast<MinMaxMode>(255)}));
    assert(!runtime->set_continuous_parameter(
        CellId{10}, ParameterSlot::Param1,
        std::numeric_limits<double>::quiet_NaN())
        .ok());
    assert(!runtime->set_continuous_parameter(
        CellId{10}, ParameterSlot::Param1,
        std::numeric_limits<double>::infinity())
        .ok());
}

void test_recurrence_and_actual_measurement() {
    const auto graph = graph_with(
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_SUM}},
        {{EdgeId{100}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{101}, CellId{20}, OutputPort{0}, CellId{20}, InputPort{1},
          EdgeDelay::PreviousTick}});
    auto seeds = seeds_for(graph);
    seeds.edge_weights[0].initial_weight = 1.0;
    seeds.edge_weights[1].initial_weight = 0.5;
    const auto compiled = GraphCompiler{}.compile(graph, seeds);
    assert(compiled.ok());
    auto runtime_result = RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    ReferenceExecutor executor;
    std::size_t frame_index = 0;
    const auto output = [&](double input) {
        const auto frame = executor.step(*runtime, std::span<const double>(&input, 1));
        assert(frame.ok());
        assert(frame.measurement.cells.size() == 2);
        assert(frame.measurement.ports.size() == 2);
        assert(frame.measurement.edges.size() == 2);
        assert(frame.measurement.edges[0].source_mode == EdgeDelay::Immediate);
        assert(frame.measurement.edges[0].source_value == input);
        assert(frame.measurement.edges[0].contribution == input);
        assert(frame.measurement.edges[1].source_mode == EdgeDelay::PreviousTick);
        const double previous =
            frame_index == 0 ? 0.0 : (frame_index == 1 ? 1.0 : 0.5);
        assert(frame.measurement.edges[1].source_value == previous);
        assert(frame.measurement.edges[1].contribution == previous * 0.5);
        const auto reduced = std::find_if(
            frame.measurement.ports.begin(), frame.measurement.ports.end(),
            [](const auto& measurement) {
                return measurement.cell == CellId{20} &&
                       measurement.port == InputPort{0};
            });
        assert(reduced != frame.measurement.ports.end());
        assert(reduced->reduced_input == input);
        const auto feedback = std::find_if(
            frame.measurement.ports.begin(), frame.measurement.ports.end(),
            [](const auto& measurement) {
                return measurement.cell == CellId{20} &&
                       measurement.port == InputPort{1};
            });
        assert(feedback != frame.measurement.ports.end());
        assert(feedback->reduced_input == previous * 0.5);
        ++frame_index;
        return runtime->cell_state(CellId{20})->output_val;
    };
    assert(output(1.0) == 1.0);
    assert(output(0.0) == 0.5);
    assert(output(0.0) == 0.25);
    assert(runtime->tick() == 3);
}

void test_multiport_channel_delay_and_hysteresis() {
    const auto arithmetic = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::SENSE_RAW_INPUT_1},
         {CellId{3}, CellType::OP_SUM}},
        {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{20}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{1},
          EdgeDelay::Immediate}});
    auto arithmetic_result =
        GraphCompiler{}.compile(arithmetic, seeds_for(arithmetic));
    assert(arithmetic_result.ok());
    auto arithmetic_runtime =
        RuntimeState::create(arithmetic_result.graph, arithmetic_result.initial_values);
    assert(arithmetic_runtime.ok());
    auto arithmetic_state = std::move(arithmetic_runtime.runtime);
    const std::array<double, 2> inputs{2.0, 3.0};
    assert(ReferenceExecutor{}.step(*arithmetic_state, inputs).ok());
    assert(arithmetic_state->cell_state(CellId{3})->output_val == 5.0);

    auto channel = graph_with({{CellId{4}, CellType::SENSE_CHANNEL}});
    auto channel_seeds = seeds_for(channel);
    channel_seeds.cell_parameters[0].value = ContinuousValue{2.0};
    channel_seeds.cell_parameters[1].value = ChannelIndex{43};
    const auto channel_result =
        GraphCompiler{}.compile(channel, channel_seeds);
    assert(channel_result.ok());
    auto channel_runtime =
        RuntimeState::create(channel_result.graph, channel_result.initial_values);
    assert(channel_runtime.ok());
    auto channel_state = std::move(channel_runtime.runtime);
    std::array<double, 44> channel_inputs{};
    channel_inputs[43] = 3.0;
    assert(ReferenceExecutor{}.step(*channel_state, channel_inputs).ok());
    assert(channel_state->cell_state(CellId{4})->output_val == 6.0);

    auto delay = graph_with(
        {{CellId{5}, CellType::SENSE_RAW_INPUT_0},
         {CellId{6}, CellType::OP_DELAY_N}},
        {{EdgeId{30}, CellId{5}, OutputPort{0}, CellId{6}, InputPort{0},
          EdgeDelay::Immediate}});
    auto delay_seeds = seeds_for(delay);
    delay_seeds.cell_parameters[2].value = DelayTicks{1};
    const auto delay_result = GraphCompiler{}.compile(delay, delay_seeds);
    assert(delay_result.ok());
    auto delay_runtime =
        RuntimeState::create(delay_result.graph, delay_result.initial_values);
    assert(delay_runtime.ok());
    auto delay_state = std::move(delay_runtime.runtime);
    for (double input : {1.0, 2.0, 3.0}) {
        assert(ReferenceExecutor{}.step(*delay_state, std::span<const double>(&input, 1)).ok());
        const double expected = input - 1.0;
        assert(delay_state->cell_state(CellId{6})->output_val == expected);
    }

    auto hysteresis = graph_with(
        {{CellId{7}, CellType::SENSE_RAW_INPUT_0},
         {CellId{8}, CellType::GATE_HYSTERESIS}},
        {{EdgeId{40}, CellId{7}, OutputPort{0}, CellId{8}, InputPort{0},
          EdgeDelay::Immediate}});
    auto hysteresis_seeds = seeds_for(hysteresis);
    hysteresis_seeds.cell_parameters[2].value = ContinuousValue{0.5};
    hysteresis_seeds.cell_parameters[3].value = ContinuousValue{-0.5};
    const auto hysteresis_result =
        GraphCompiler{}.compile(hysteresis, hysteresis_seeds);
    assert(hysteresis_result.ok());
    auto hysteresis_runtime = RuntimeState::create(
        hysteresis_result.graph, hysteresis_result.initial_values);
    assert(hysteresis_runtime.ok());
    auto hysteresis_state = std::move(hysteresis_runtime.runtime);
    double input = 1.0;
    assert(ReferenceExecutor{}.step(*hysteresis_state, std::span<const double>(&input, 1)).ok());
    assert(hysteresis_state->cell_state(CellId{8})->output_val == 1.0);
    input = 0.0;
    assert(ReferenceExecutor{}.step(*hysteresis_state, std::span<const double>(&input, 1)).ok());
    assert(hysteresis_state->cell_state(CellId{8})->output_val == 1.0);
    input = -1.0;
    assert(ReferenceExecutor{}.step(*hysteresis_state, std::span<const double>(&input, 1)).ok());
    assert(hysteresis_state->cell_state(CellId{8})->output_val == -1.0);
}

void test_all_generated_cell_types_use_the_shared_kernel() {
    constexpr std::array<CellType, 30> types{{
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::SENSE_RAW_INPUT_2, CellType::SENSE_RAW_INPUT_3,
        CellType::SENSE_CHANNEL, CellType::OP_EMA, CellType::OP_DIFF,
        CellType::OP_INTEGRAL, CellType::OP_SUM, CellType::OP_SUB,
        CellType::OP_MULTIPLY, CellType::OP_RATIO, CellType::OP_ABS,
        CellType::OP_DELAY_N, CellType::OP_OSCILLATOR, CellType::OP_QUADRATIC,
        CellType::GATE_THRESHOLD, CellType::GATE_HYSTERESIS, CellType::GATE_AND,
        CellType::GATE_INHIBIT, CellType::GATE_DEADZONE, CellType::GATE_MIN_MAX,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE,
        CellType::ACT_DEFENSIVE_RESET, CellType::ACT_IMMUNE_BLOCK,
        CellType::ACT_CHANNEL, CellType::PREDICT_SENSE_0,
        CellType::PREDICT_SENSE_1, CellType::ASSOCIATION_HUB}};
    const std::array<std::array<double, 44>, 3> inputs{{
        [] {
            std::array<double, 44> value{};
            value[0] = 2.0;
            value[1] = -1.0;
            value[2] = 3.0;
            value[3] = 4.0;
            value[43] = 5.0;
            return value;
        }(),
        [] {
            std::array<double, 44> value{};
            value[0] = -1.0;
            value[1] = 2.0;
            value[2] = 0.5;
            value[3] = 3.0;
            value[43] = 1.5;
            return value;
        }(),
        [] {
            std::array<double, 44> value{};
            value[0] = 0.25;
            value[1] = -3.0;
            value[2] = 2.5;
            value[3] = -2.0;
            value[43] = 7.0;
            return value;
        }(),
    }};
    ReferenceExecutor executor;
    for (std::size_t type_index = 0; type_index < types.size(); ++type_index) {
        const CellType type = types[type_index];
        const bool is_sensor = type_index < 5;
        GraphDefinition graph = graph_with({});
        CellularOrganism legacy;
        if (is_sensor) {
            graph.cells = {{CellId{1}, type}, {CellId{2}, CellType::OP_ABS}};
            legacy.cells = {Cell{1, type}, Cell{2, CellType::OP_ABS}};
            const auto [param1, param2] = parameters_for_type(type);
            legacy.cells[0].param1 = param1;
            legacy.cells[0].param2 = param2;
            legacy.synapses = {Synapse{1, 2, 0, 1.0}};
            graph.edges = {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2},
                            InputPort{0}, EdgeDelay::Immediate}};
        } else {
            graph.cells = {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
                           {CellId{2}, type}, {CellId{3}, CellType::OP_ABS}};
            legacy.cells = {Cell{1, CellType::SENSE_RAW_INPUT_0},
                            Cell{2, type}, Cell{3, CellType::OP_ABS}};
            const auto [param1, param2] = parameters_for_type(type);
            legacy.cells[1].param1 = param1;
            legacy.cells[1].param2 = param2;
            legacy.synapses = {Synapse{1, 2, 0, 1.0}, Synapse{2, 3, 0, 1.0}};
            graph.edges = {
                {EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2},
                 InputPort{0}, EdgeDelay::Immediate},
                {EdgeId{2}, CellId{2}, OutputPort{0}, CellId{3},
                 InputPort{0}, EdgeDelay::Immediate}};
            if (is_effector_cell(type) ||
                type == CellType::PREDICT_SENSE_0 ||
                type == CellType::PREDICT_SENSE_1) {
                legacy.cells.pop_back();
                legacy.synapses.pop_back();
                graph.cells.pop_back();
                graph.edges.pop_back();
            }
        }
        assert(legacy.compile());
        auto seeds = InitialParameterSeeds{};
        for (const auto& cell : legacy.cells) {
            const auto contract = contract_for(cell.type)->get();
            for (std::size_t slot = 0; slot < 2; ++slot) {
                seeds.cell_parameters.push_back(CellParameterSeed{
                    CellId{cell.id}, static_cast<ParameterSlot>(slot),
                    typed_parameter(contract, static_cast<ParameterSlot>(slot),
                                    slot == 0 ? cell.param1 : cell.param2)});
            }
        }
        for (const auto& edge : graph.edges) {
            seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, 1.0});
        }
        const auto compiled = GraphCompiler{}.compile(graph, seeds);
        assert(compiled.ok());
        auto runtime = RuntimeState::create(compiled.graph, compiled.initial_values);
        assert(runtime.ok());
        for (const auto& input : inputs) {
            legacy.forward_nd(input.data(), input.size(), false);
            const auto frame = executor.step(*runtime.runtime, input);
            assert(frame.ok());
            assert(frame.measurement.cells.size() == graph.cells.size());
            for (const auto& cell : frame.measurement.cells) {
                assert(cell.executed);
                const auto expected = std::find_if(
                    legacy.cells.begin(), legacy.cells.end(),
                    [&cell](const auto& candidate) {
                        return candidate.id == cell.cell.value;
                    });
                assert(expected != legacy.cells.end());
                assert_runtime_cell_matches(
                    *runtime.runtime->cell_state(cell.cell), *expected);
            }
        }
    }
}

void test_overflowed_delay_ring_step_is_atomic() {
    const auto graph = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto seeds = seeds_for(graph);
    seeds.edge_weights[0].initial_weight = 2.0;
    seeds.cell_parameters[2].value = DelayTicks{1};
    const auto compiled = GraphCompiler{}.compile(graph, seeds);
    assert(compiled.ok());
    auto runtime_result = RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    const auto before = runtime->snapshot();
    const double input = std::numeric_limits<double>::max();
    const auto frame = ReferenceExecutor{}.step(
        *runtime, std::span<const double>(&input, 1));
    assert(!frame.ok());
    assert(frame.error->has_cell);
    assert(frame.error->cell == CellId{2});
    assert(runtime->tick() == before.tick());
    assert(runtime->cell_state(CellId{1})->output_val ==
           before.cells()[0].output_val);
    assert(runtime->cell_state(CellId{2})->delay_buffer ==
           before.cells()[1].delay_buffer);
    assert(frame.measurement.cells.empty());
    assert(frame.measurement.edges.empty());
}

void test_delay16_wrap_and_silent_state_initialization() {
    const auto delay_graph = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto delay_seeds = seeds_for(delay_graph);
    delay_seeds.cell_parameters[2].value = DelayTicks{16};
    const auto delay_compiled =
        GraphCompiler{}.compile(delay_graph, delay_seeds);
    assert(delay_compiled.ok());
    auto delay_runtime =
        RuntimeState::create(delay_compiled.graph, delay_compiled.initial_values);
    assert(delay_runtime.ok());
    CellularOrganism legacy;
    Cell source;
    source.id = 1;
    source.type = CellType::SENSE_RAW_INPUT_0;
    source.param1 = 1.0;
    Cell delayed;
    delayed.id = 2;
    delayed.type = CellType::OP_DELAY_N;
    delayed.param1 = 1.0;
    legacy.cells = {source, delayed};
    legacy.synapses = {Synapse{1, 2, 0, 1.0}};
    assert(legacy.compile());
    ReferenceExecutor executor;
    for (std::size_t tick = 0; tick < 18; ++tick) {
        const double input = static_cast<double>(tick + 1);
        legacy.forward_nd(&input, 1, false);
        const auto frame = executor.step(
            *delay_runtime.runtime, std::span<const double>(&input, 1));
        assert(frame.ok());
        assert_runtime_cell_matches(
            *delay_runtime.runtime->cell_state(CellId{2}),
            legacy.cells[1]);
        const double expected = tick < 16 ? 0.0 : static_cast<double>(tick - 15);
        assert(delay_runtime.runtime->cell_state(CellId{2})->output_val == expected);
    }

    const auto silent_graph = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_EMA},
         {CellId{3}, CellType::OP_OSCILLATOR}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{2}, CellId{1}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::Immediate}});
    auto silent_seeds = seeds_for(silent_graph);
    silent_seeds.cell_parameters[2].value = ContinuousValue{0.5};
    silent_seeds.cell_parameters[5].value = ContinuousValue{0.05};
    const auto silent_compiled =
        GraphCompiler{}.compile(silent_graph, silent_seeds);
    assert(silent_compiled.ok());
    auto silent_runtime =
        RuntimeState::create(silent_compiled.graph, silent_compiled.initial_values);
    assert(silent_runtime.ok());
    double input = 0.0;
    assert(ReferenceExecutor{}.step(
               *silent_runtime.runtime, std::span<const double>(&input, 1))
               .ok());
    assert(silent_runtime.runtime->cell_state(CellId{2})->initialized);
    assert(silent_runtime.runtime->cell_state(CellId{3})->initialized);
    assert(silent_runtime.runtime->cell_state(CellId{2})->activation_count == 0);
    assert(silent_runtime.runtime->cell_state(CellId{3})->activation_count ==
           (std::abs(silent_runtime.runtime->cell_state(CellId{3})->output_val) >
                    1e-6
                ? 1
                : 0));
    input = 2.0;
    assert(ReferenceExecutor{}.step(
               *silent_runtime.runtime, std::span<const double>(&input, 1))
               .ok());
    assert(silent_runtime.runtime->cell_state(CellId{2})->activation_count > 0);
    assert(silent_runtime.runtime->cell_state(CellId{3})->activation_count > 0);
}

void test_imported_old_order_and_measurement_values() {
    CellularOrganism source;
    Cell first;
    first.id = 30;
    first.type = CellType::SENSE_RAW_INPUT_0;
    first.param1 = 1.0;
    Cell second;
    second.id = 10;
    second.type = CellType::SENSE_RAW_INPUT_1;
    second.param1 = 1.0;
    Cell third;
    third.id = 20;
    third.type = CellType::SENSE_RAW_INPUT_2;
    third.param1 = 1.0;
    Cell sum;
    sum.id = 40;
    sum.type = CellType::OP_SUM;
    Cell action;
    action.id = 50;
    action.type = CellType::ACT_PRIMARY_POSITIVE;
    source.cells = {first, second, third, sum, action};
    source.synapses = {
        Synapse{30, 40, 0, 1e16},
        Synapse{10, 40, 0, -1e16},
        Synapse{20, 40, 0, 1.0},
        Synapse{40, 50, 0, 1.0}};
    assert(source.compile());
    const auto imported = migration::import_execution_snapshot(
        source, GraphIdentity{900}, GraphRevision{1});
    assert(imported.ok());
    const auto adapter =
        migration::import_execution_snapshot_runtime(*imported.snapshot);
    assert(adapter.ok());
    const std::array<double, 3> input{1.0, 1.0, 1.0};
    CellularOrganism legacy = source;
    legacy.forward_nd(input.data(), input.size(), false);
    const auto frame = ReferenceExecutor{}.step(*adapter.runtime, input);
    assert(frame.ok());
    assert_near(adapter.runtime->cell_state(CellId{40})->output_val, 1.0);
    assert_near(adapter.runtime->cell_state(CellId{40})->output_val,
                legacy.cells[3].output_val);
    assert(frame.measurement.edges.size() == 4);
    assert(frame.measurement.edges[0].edge.value == 1);
    assert(frame.measurement.edges[0].source_value == 1.0);
    assert(frame.measurement.edges[0].contribution == 1e16);
    assert(frame.measurement.edges[1].source_value == 1.0);
    assert(frame.measurement.edges[1].contribution == -1e16);
    assert(frame.measurement.edges[2].source_value == 1.0);
    assert(frame.measurement.edges[2].contribution == 1.0);
    assert(frame.measurement.ports.size() == 3);
    const auto reduced = std::find_if(
        frame.measurement.ports.begin(), frame.measurement.ports.end(),
        [](const auto& measurement) {
            return measurement.cell == CellId{40} &&
                   measurement.port == InputPort{0};
        });
    assert(reduced != frame.measurement.ports.end());
    assert(reduced->reduced_input == 1.0);
    const auto sum_port1 = std::find_if(
        frame.measurement.ports.begin(), frame.measurement.ports.end(),
        [](const auto& measurement) {
            return measurement.cell == CellId{40} &&
                   measurement.port == InputPort{1};
        });
    assert(sum_port1 != frame.measurement.ports.end());
    assert(sum_port1->reduced_input == 0.0);
    const auto action_port = std::find_if(
        frame.measurement.ports.begin(), frame.measurement.ports.end(),
        [](const auto& measurement) {
            return measurement.cell == CellId{50} &&
                   measurement.port == InputPort{0};
        });
    assert(action_port != frame.measurement.ports.end());
    assert(action_port->reduced_input == 1.0);
}

void test_silent_frames_reset_probe_snapshot_and_atomic_restore() {
    const auto graph = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::ACT_PRIMARY_POSITIVE}},
        {{EdgeId{5}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto seeds = seeds_for(graph);
    seeds.edge_weights[0].initial_weight = 1.0;
    const auto compiled = GraphCompiler{}.compile(graph, seeds);
    assert(compiled.ok());
    auto runtime_result = RuntimeState::create(compiled.graph, compiled.initial_values);
    assert(runtime_result.ok());
    auto runtime = std::move(runtime_result.runtime);
    const auto edge_parameter = runtime->parameters()[
        compiled.graph->edges()[0].weight_parameter_index].binding;
    assert(runtime->set_parameter(edge_parameter, ContinuousValue{2.5}).ok());
    ReferenceExecutor executor;
    double input = 0.0;
    assert(executor.step(*runtime, std::span<const double>(&input, 1)).ok());
    assert(runtime->cell_state(CellId{1})->activation_count == 0);
    assert(runtime->cell_state(CellId{1})->initialized);
    input = 2.0;
    assert(executor.step(*runtime, std::span<const double>(&input, 1)).ok());
    assert(runtime->cell_state(CellId{1})->activation_count == 1);
    assert(runtime->cell_state(CellId{2})->output_val == 5.0);

    const auto checkpoint = runtime->snapshot();
    auto restored_result = RuntimeState::from_snapshot(checkpoint);
    assert(restored_result.ok());
    assert(std::get<ContinuousValue>(
               *restored_result.runtime->parameter_at(edge_parameter.index))
               .value == 2.5);
    assert(runtime->set_parameter(edge_parameter, ContinuousValue{3.5}).ok());
    assert(std::get<ContinuousValue>(
               *restored_result.runtime->parameter_at(edge_parameter.index))
               .value == 2.5);
    assert(runtime->set_parameter(edge_parameter, ContinuousValue{2.5}).ok());
    auto fork_result = runtime->fork_probe();
    assert(fork_result.ok());
    assert(fork_result.runtime->cell_states().size() == checkpoint.cells().size());
    for (std::size_t i = 0; i < checkpoint.cells().size(); ++i) {
        assert_runtime_state_equal(
            fork_result.runtime->cell_states()[i], checkpoint.cells()[i]);
    }
    assert(fork_result.runtime->set_parameter(edge_parameter, ContinuousValue{1.0}).ok());
    assert(executor.step(*fork_result.runtime, std::span<const double>(&input, 1)).ok());
    assert(runtime->cell_state(CellId{2})->output_val == 5.0);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(edge_parameter.index)).value == 2.5);
    runtime->reset_episode();
    assert(runtime->tick() == 0);
    assert(runtime->cell_state(CellId{2})->output_val == 0.0);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(edge_parameter.index)).value == 2.5);
    assert(runtime->restore_snapshot(checkpoint).ok());
    assert(runtime->tick() == checkpoint.tick());
    for (std::size_t i = 0; i < checkpoint.cells().size(); ++i) {
        assert_runtime_state_equal(
            runtime->cell_states()[i], checkpoint.cells()[i]);
    }

    auto altered_graph = graph;
    altered_graph.cells[1].type = CellType::ACT_PRIMARY_NEGATIVE;
    const auto altered = GraphCompiler{}.compile(altered_graph, seeds);
    assert(altered.ok());
    auto altered_runtime =
        RuntimeState::create(altered.graph, altered.initial_values);
    assert(altered_runtime.ok());
    const auto bad = altered_runtime.runtime->snapshot();
    assert(!runtime->restore_snapshot(bad).ok());
    assert(runtime->cell_state(CellId{2})->output_val == 5.0);
}

void test_empty_graph_and_legacy_snapshot_connection() {
    const auto empty = graph_with({});
    const auto empty_result = GraphCompiler{}.compile(empty, {});
    assert(empty_result.ok());
    auto empty_runtime = RuntimeState::create(
        empty_result.graph, empty_result.initial_values);
    assert(empty_runtime.ok());
    assert(ReferenceExecutor{}.step(*empty_runtime.runtime, {}).ok());
    assert(empty_runtime.runtime->tick() == 1);

    std::shared_ptr<RuntimeState> imported;
    std::vector<migration::SnapshotEdgeProvenance> provenance;
    {
        CellularOrganism source;
        source.cells = {
            Cell{0, CellType::SENSE_RAW_INPUT_0},
            Cell{1, CellType::ACT_PRIMARY_POSITIVE}};
        Synapse edge;
        edge.from_cell_id = 0;
        edge.to_cell_id = 1;
        edge.weight = 2.5;
        edge.initial_weight = 1.0;
        edge.hebbian_rate = 0.0;
        source.synapses = {edge};
        assert(source.compile());
        const double value = 2.0;
        source.forward_nd(&value, 1, false);
        const auto snapshot = migration::import_execution_snapshot(
            source, GraphIdentity{500}, GraphRevision{2});
        assert(snapshot.ok());
        const auto runtime = migration::import_execution_snapshot_runtime(*snapshot.snapshot);
        assert(runtime.ok());
        imported = runtime.runtime;
        provenance = runtime.edge_provenance;
        assert(provenance.size() == 1);
        assert(provenance[0].raw_declared_initial_weight == 1.0);
    }
    assert(imported);
    assert(imported->cell_state(CellId{1})->output_val == 0.5);
    const double detached_input = 1.0;
    const auto detached_frame = ReferenceExecutor{}.step(
        *imported, std::span<const double>(&detached_input, 1));
    assert(detached_frame.ok());
    assert(imported->tick() == 1);
}

void test_imported_live_weights_continue_without_online_learning() {
    CellularOrganism source;
    source.cells = {
        Cell{0, CellType::SENSE_RAW_INPUT_0},
        Cell{1, CellType::ACT_PRIMARY_POSITIVE}};
    Synapse first;
    first.from_cell_id = 0;
    first.to_cell_id = 1;
    first.weight = 2.5;
    first.initial_weight = 1.0;
    first.hebbian_rate = 0.1;
    first.hebbian_decay = 0.02;
    Synapse second = first;
    second.weight = 0.75;
    second.initial_weight = 0.25;
    second.hebbian_rate = 0.0;
    source.synapses = {first, second};
    assert(source.compile());
    const double initial_input = 2.0;
    source.forward_nd(&initial_input, 1, true);
    const double live_first_weight = source.compiled_synapses_[0].weight;
    const double live_second_weight = source.compiled_synapses_[1].weight;
    assert(live_first_weight != live_second_weight);
    const auto snapshot = migration::import_execution_snapshot(
        source, GraphIdentity{700}, GraphRevision{3});
    assert(snapshot.ok());
    const auto runtime_result =
        migration::import_execution_snapshot_runtime(*snapshot.snapshot);
    assert(runtime_result.ok());
    auto runtime = runtime_result.runtime;
    const auto imported_plan = runtime->plan();
    assert(imported_plan->edges().size() == 2);
    const auto first_weight_index =
        imported_plan->edges()[0].weight_parameter_index;
    const auto second_weight_index =
        imported_plan->edges()[1].weight_parameter_index;
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(first_weight_index))
               .value == live_first_weight);
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(second_weight_index))
               .value == live_second_weight);
    CellularOrganism continuation = source;
    ReferenceExecutor executor;
    for (const double input : {-1.5, 0.25, 3.0}) {
        continuation.forward_nd(&input, 1, false);
        const auto frame = executor.step(*runtime, std::span<const double>(&input, 1));
        assert(frame.ok());
        for (const auto& state : runtime->cell_states()) {
            const auto it = std::find_if(
                continuation.cells.begin(), continuation.cells.end(),
                [&state](const auto& cell) {
                    return cell.id == state.cell.value;
                });
            assert(it != continuation.cells.end());
            assert_runtime_cell_matches(state, *it);
        }
        assert(std::get<ContinuousValue>(
                   *runtime->parameter_at(first_weight_index))
                   .value == live_first_weight);
        assert(std::get<ContinuousValue>(
                   *runtime->parameter_at(second_weight_index))
                   .value == live_second_weight);
    }
    const auto continuation_snapshot = runtime->snapshot();
    assert(runtime->reset_episode().ok());
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(first_weight_index))
               .value == live_first_weight);
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(second_weight_index))
               .value == live_second_weight);
    assert(runtime->restore_snapshot(continuation_snapshot).ok());
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(first_weight_index))
               .value == live_first_weight);
    assert(std::get<ContinuousValue>(
               *runtime->parameter_at(second_weight_index))
               .value == live_second_weight);
}

void test_mixed_stateful_snapshot_probe_restore_continuation() {
    CellularOrganism source;
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
    source.cells = {raw, delay, ema, diff, latch, action};

    Synapse learned;
    learned.from_cell_id = 0;
    learned.to_cell_id = 1;
    learned.weight = 1.4;
    learned.initial_weight = 1.0;
    learned.hebbian_rate = 0.1;
    learned.hebbian_decay = 0.02;
    Synapse fixed = learned;
    fixed.weight = 0.6;
    fixed.initial_weight = 0.5;
    fixed.hebbian_rate = 0.0;
    source.synapses = {
        learned,
        fixed,
        Synapse{1, 2, 0, 1.0},
        Synapse{2, 3, 0, 1.0},
        Synapse{3, 4, 0, 1.0},
        Synapse{4, 5, 0, 1.0}};
    assert(source.compile());

    for (double input = 1.0; input <= 20.0; input += 1.0) {
        source.forward_nd(&input, 1, true);
    }
    assert(source.cells[1].delay_idx != 0);
    assert(std::any_of(
        std::begin(source.cells[1].delay_buffer),
        std::end(source.cells[1].delay_buffer),
        [](double value) { return std::abs(value) > 1e-12; }));
    assert(std::abs(source.cells[2].state_val) > 1e-12);
    assert(std::abs(source.cells[3].prev_input) > 1e-12);
    assert(source.cells[4].latch_state);
    const double learned_weight = source.compiled_synapses_[0].weight;
    const double fixed_weight = source.compiled_synapses_[1].weight;
    assert(learned_weight != fixed_weight);

    CellularOrganism reference = source;
    const auto imported = migration::import_execution_snapshot(
        source, GraphIdentity{701}, GraphRevision{4});
    assert(imported.ok());
    const auto runtime_result =
        migration::import_execution_snapshot_runtime(*imported.snapshot);
    assert(runtime_result.ok());
    auto runtime = runtime_result.runtime;
    const auto plan = runtime->plan();
    assert(plan->edges().size() == 6);
    const auto learned_index = plan->edges()[0].weight_parameter_index;
    const auto fixed_index = plan->edges()[1].weight_parameter_index;
    assert(std::get<ContinuousValue>(*runtime->parameter_at(learned_index)).value ==
           learned_weight);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(fixed_index)).value ==
           fixed_weight);

    const auto checkpoint = runtime->snapshot();
    auto probe = runtime->fork_probe();
    assert(probe.ok());
    const double probe_input = 21.0;
    assert(ReferenceExecutor{}.step(
               *probe.runtime, std::span<const double>(&probe_input, 1))
               .ok());
    assert(runtime->tick() == checkpoint.tick());
    for (std::size_t i = 0; i < checkpoint.cells().size(); ++i) {
        assert_runtime_state_equal(runtime->cell_states()[i], checkpoint.cells()[i]);
    }
    assert(std::get<ContinuousValue>(*probe.runtime->parameter_at(learned_index)).value ==
           learned_weight);
    assert(std::get<ContinuousValue>(*probe.runtime->parameter_at(fixed_index)).value ==
           fixed_weight);

    assert(runtime->reset_episode().ok());
    assert(runtime->tick() == 0);
    assert(runtime->cell_state(CellId{1})->delay_idx == 0);
    const std::array<double, 16> zero_delay{};
    assert(runtime->cell_state(CellId{1})->delay_buffer == zero_delay);
    assert(runtime->cell_state(CellId{2})->state_val == 0.0);
    assert(runtime->cell_state(CellId{3})->prev_input == 0.0);
    assert(!runtime->cell_state(CellId{4})->latch_state);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(learned_index)).value ==
           learned_weight);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(fixed_index)).value ==
           fixed_weight);
    assert(runtime->restore_snapshot(checkpoint).ok());
    for (std::size_t i = 0; i < checkpoint.cells().size(); ++i) {
        assert_runtime_state_equal(runtime->cell_states()[i], checkpoint.cells()[i]);
    }

    ReferenceExecutor executor;
    for (double input = 21.0; input <= 24.0; input += 1.0) {
        reference.forward_nd(&input, 1, false);
        const auto frame = executor.step(
            *runtime, std::span<const double>(&input, 1));
        assert(frame.ok());
        for (const auto& state : runtime->cell_states()) {
            const auto expected = std::find_if(
                reference.cells.begin(), reference.cells.end(),
                [&state](const auto& cell) {
                    return cell.id == state.cell.value;
                });
            assert(expected != reference.cells.end());
            assert_runtime_cell_matches(state, *expected);
        }
    }
    assert(std::get<ContinuousValue>(*runtime->parameter_at(learned_index)).value ==
           learned_weight);
    assert(std::get<ContinuousValue>(*runtime->parameter_at(fixed_index)).value ==
           fixed_weight);
}

}  // namespace

int main() {
    test_runtime_executes_and_resets_without_resetting_parameters();
    test_binding_identity_and_strict_profile_are_supported();
    test_invalid_parameter_enums_and_nonfinite_updates_are_rejected();
    test_recurrence_and_actual_measurement();
    test_multiport_channel_delay_and_hysteresis();
    test_all_generated_cell_types_use_the_shared_kernel();
    test_overflowed_delay_ring_step_is_atomic();
    test_delay16_wrap_and_silent_state_initialization();
    test_imported_old_order_and_measurement_values();
    test_silent_frames_reset_probe_snapshot_and_atomic_restore();
    test_empty_graph_and_legacy_snapshot_connection();
    test_imported_live_weights_continue_without_online_learning();
    test_mixed_stateful_snapshot_probe_restore_continuation();
    return 0;
}
