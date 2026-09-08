#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace kun;
using namespace kun::core;

CellParameterSeed parameter_seed(CellId cell, ParameterSlot slot, ParameterValue value) {
    return CellParameterSeed{cell, slot, std::move(value)};
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
                    value = UnusedParameter{};
                    break;
            }
            seeds.cell_parameters.push_back(
                parameter_seed(cell.id, static_cast<ParameterSlot>(slot), std::move(value)));
        }
    }
    for (const auto& edge : graph.edges) {
        seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, 1.0});
    }
    return seeds;
}

GraphDefinition graph_with(std::vector<CellDefinition> cells,
                           std::vector<EdgeDefinition> edges = {}) {
    GraphDefinition graph;
    graph.identity = GraphIdentity{77};
    graph.revision = GraphRevision{9};
    graph.profile = SemanticProfile::LegacyCompatible;
    graph.semantic_version = 1;
    graph.cells = std::move(cells);
    graph.edges = std::move(edges);
    return graph;
}

void expect_error(const GraphDefinition& graph,
                  const InitialParameterSeeds& seeds,
                  CompileErrorCode code,
                  uint64_t id = 0) {
    const auto result = GraphCompiler{}.compile(graph, seeds);
    assert(!result.ok());
    assert(result.error.has_value());
    assert(result.error->code == code);
    if (id != 0) {
        assert(result.error->diagnostic().find(std::to_string(id)) != std::string::npos);
    }
}

const InitialParameterValue& initial(const CompileResult& result, std::size_t index) {
    return result.initial_values->at(index);
}

std::optional<double> legacy_kernel_parameter(
    const ParameterValue& value) {
    if (std::holds_alternative<ContinuousValue>(value)) {
        return std::get<ContinuousValue>(value).value;
    }
    if (std::holds_alternative<ChannelIndex>(value)) {
        constexpr std::size_t max_exact_integer =
            static_cast<std::size_t>(1ULL << 53);
        const auto channel = std::get<ChannelIndex>(value).value;
        if (channel > max_exact_integer) return std::nullopt;
        return static_cast<double>(channel);
    }
    if (std::holds_alternative<DelayTicks>(value)) {
        return static_cast<double>(std::get<DelayTicks>(value).value) / 16.0;
    }
    if (std::holds_alternative<MinMaxMode>(value)) {
        return std::get<MinMaxMode>(value) == MinMaxMode::Max ? 1.0 : 0.0;
    }
    return 0.0;
}

double initial_double(const CompileResult& result, std::size_t index) {
    const auto converted = legacy_kernel_parameter(initial(result, index).value);
    assert(converted.has_value());
    return *converted;
}

double edge_weight(const CompileResult& result, std::size_t index) {
    const auto& value = initial(result, index).value;
    assert(std::holds_alternative<ContinuousValue>(value));
    return std::get<ContinuousValue>(value).value;
}

void assert_same_initial_value(const InitialParameterValue& lhs,
                               const InitialParameterValue& rhs) {
    assert(lhs.binding.kind == rhs.binding.kind);
    assert(lhs.binding.index == rhs.binding.index);
    assert(lhs.binding.cell == rhs.binding.cell);
    assert(lhs.binding.edge == rhs.binding.edge);
    assert(lhs.binding.slot == rhs.binding.slot);
    assert(lhs.value.index() == rhs.value.index());
    if (std::holds_alternative<ContinuousValue>(lhs.value)) {
        assert(std::get<ContinuousValue>(lhs.value).value ==
               std::get<ContinuousValue>(rhs.value).value);
    } else if (std::holds_alternative<ChannelIndex>(lhs.value)) {
        assert(std::get<ChannelIndex>(lhs.value).value ==
               std::get<ChannelIndex>(rhs.value).value);
    } else if (std::holds_alternative<DelayTicks>(lhs.value)) {
        assert(std::get<DelayTicks>(lhs.value).value ==
               std::get<DelayTicks>(rhs.value).value);
    } else if (std::holds_alternative<MinMaxMode>(lhs.value)) {
        assert(std::get<MinMaxMode>(lhs.value) ==
               std::get<MinMaxMode>(rhs.value));
    }
}

void test_empty_and_declared_disconnected_graphs() {
    const GraphDefinition empty = graph_with({});
    const auto empty_result = GraphCompiler{}.compile(empty, {});
    assert(empty_result.ok());
    assert(empty_result.graph->cells().empty());
    assert(empty_result.graph->edges().empty());
    assert(empty_result.graph->execution_order().empty());
    assert(empty_result.initial_values->entries().empty());

    const GraphDefinition graph = graph_with({
        {CellId{9}, CellType::ACT_PRIMARY_POSITIVE},
        {CellId{2}, CellType::SENSE_RAW_INPUT_0},
        {CellId{100}, CellType::OP_ABS},
    });
    const auto result = GraphCompiler{}.compile(graph, seeds_for(graph));
    assert(result.ok());
    assert(result.graph->cells().size() == 3);
    assert(result.graph->execution_order().size() == 3);
    assert(result.graph->cells()[0].id == CellId{2});
    assert(result.graph->cells()[1].id == CellId{9});
    assert(result.graph->cells()[2].id == CellId{100});
}

void test_ids_parallel_edges_and_determinism() {
    const CellId high{0x100000000ULL + 17};
    GraphDefinition graph = graph_with({
        {high, CellType::OP_ABS},
        {CellId{3}, CellType::SENSE_RAW_INPUT_0},
        {CellId{4}, CellType::SENSE_RAW_INPUT_1},
    }, {
        {EdgeId{20}, CellId{3}, OutputPort{0}, high, InputPort{0}, EdgeDelay::Immediate},
        {EdgeId{10}, CellId{3}, OutputPort{0}, high, InputPort{0}, EdgeDelay::Immediate},
    });
    auto seeds = seeds_for(graph);
    seeds.edge_weights[0].initial_weight = 2.0;
    seeds.edge_weights[1].initial_weight = 3.0;
    const auto result = GraphCompiler{}.compile(graph, seeds);
    assert(result.ok());
    assert(result.graph->edges().size() == 2);
    assert(result.graph->edges()[0].id == EdgeId{10});
    assert(result.graph->edges()[1].id == EdgeId{20});
    assert(result.graph->edges()[0].source_index == result.graph->edges()[1].source_index);
    assert(edge_weight(result, result.graph->edges()[0].weight_parameter_index) == 3.0);
    assert(edge_weight(result, result.graph->edges()[1].weight_parameter_index) == 2.0);
    assert(result.graph->port_reductions().size() == 1);
    const auto reduction = result.graph->port_reductions()[0];
    assert(reduction.edge_end - reduction.edge_begin == 2);

    GraphDefinition permuted = graph;
    std::reverse(permuted.cells.begin(), permuted.cells.end());
    std::reverse(permuted.edges.begin(), permuted.edges.end());
    std::reverse(seeds.cell_parameters.begin(), seeds.cell_parameters.end());
    std::reverse(seeds.edge_weights.begin(), seeds.edge_weights.end());
    const auto permuted_result = GraphCompiler{}.compile(permuted, seeds);
    assert(permuted_result.ok());
    assert(permuted_result.graph->execution_order().size() ==
           result.graph->execution_order().size());
    for (std::size_t i = 0; i < result.graph->execution_order().size(); ++i) {
        const auto lhs = result.graph->cells()[result.graph->execution_order()[i]].id;
        const auto rhs =
            permuted_result.graph
                ->cells()[permuted_result.graph->execution_order()[i]]
                .id;
        assert(lhs == rhs);
    }
    for (std::size_t i = 0; i < result.graph->edges().size(); ++i) {
        assert(result.graph->edges()[i].id == permuted_result.graph->edges()[i].id);
        assert(result.graph->edges()[i].target_port ==
               permuted_result.graph->edges()[i].target_port);
        assert(result.graph->edges()[i].weight_parameter_index ==
               permuted_result.graph->edges()[i].weight_parameter_index);
    }
    assert(result.graph->port_reductions().size() ==
           permuted_result.graph->port_reductions().size());
    for (std::size_t i = 0; i < result.graph->port_reductions().size(); ++i) {
        const auto& lhs = result.graph->port_reductions()[i];
        const auto& rhs = permuted_result.graph->port_reductions()[i];
        assert(lhs.target_index == rhs.target_index);
        assert(lhs.target_port == rhs.target_port);
        assert(lhs.edge_end - lhs.edge_begin == rhs.edge_end - rhs.edge_begin);
        for (std::size_t offset = 0; offset < lhs.edge_end - lhs.edge_begin; ++offset) {
            assert(result.graph->edges()[lhs.edge_begin + offset].id ==
                   permuted_result.graph->edges()[rhs.edge_begin + offset].id);
        }
    }
    for (std::size_t i = 0; i < result.initial_values->size(); ++i) {
        assert_same_initial_value(initial(result, i), initial(permuted_result, i));
    }
}

void test_validation_diagnostics() {
    GraphDefinition duplicate_cell = graph_with({
        {CellId{8}, CellType::OP_ABS},
        {CellId{8}, CellType::OP_ABS},
    });
    expect_error(duplicate_cell, seeds_for(duplicate_cell),
                 CompileErrorCode::DuplicateCellId, 8);

    GraphDefinition duplicate_edge = graph_with(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        {{EdgeId{6}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{6}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::PreviousTick}});
    expect_error(duplicate_edge, seeds_for(duplicate_edge),
                 CompileErrorCode::DuplicateEdgeId, 6);

    GraphDefinition dangling = graph_with(
        {{CellId{1}, CellType::OP_ABS}},
        {{EdgeId{0x100000000ULL + 1}, CellId{9}, OutputPort{0},
          CellId{1}, InputPort{0}, EdgeDelay::Immediate}});
    auto dangling_seeds = seeds_for(dangling);
    dangling_seeds.edge_weights.push_back({EdgeId{0x100000000ULL + 1}, 1.0});
    expect_error(dangling, dangling_seeds, CompileErrorCode::DanglingEndpoint,
                 0x100000000ULL + 1);

    GraphDefinition bad_ports = graph_with(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        {{EdgeId{3}, CellId{2}, OutputPort{1}, CellId{1}, InputPort{0},
          EdgeDelay::Immediate}});
    expect_error(bad_ports, seeds_for(bad_ports),
                 CompileErrorCode::UnsupportedSourcePort, 3);
    bad_ports.edges[0].source_port = OutputPort{0};
    bad_ports.edges[0].target_port = InputPort{1};
    expect_error(bad_ports, seeds_for(bad_ports),
                 CompileErrorCode::UnsupportedTargetPort, 3);

    GraphDefinition bad_delay = graph_with(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        {{EdgeId{4}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          static_cast<EdgeDelay>(7)}});
    expect_error(bad_delay, seeds_for(bad_delay),
                 CompileErrorCode::UnknownDelay, 4);

    GraphDefinition bad_profile = graph_with({{CellId{1}, CellType::OP_ABS}});
    bad_profile.profile = static_cast<SemanticProfile>(255);
    bad_profile.identity = GraphIdentity{0x100000000ULL + 9};
    expect_error(bad_profile, seeds_for(bad_profile),
                 CompileErrorCode::UnknownProfile, 0x100000000ULL + 9);
    assert(GraphCompiler{}.compile(bad_profile, seeds_for(bad_profile))
               .error->diagnostic()
               .find("graph") != std::string::npos);
    GraphDefinition bad_version = graph_with({{CellId{1}, CellType::OP_ABS}});
    bad_version.semantic_version = 2;
    expect_error(bad_version, seeds_for(bad_version),
                 CompileErrorCode::UnsupportedSemanticVersion,
                 bad_version.identity.value);
    GraphDefinition strict = graph_with({{CellId{1}, CellType::OP_ABS}});
    strict.profile = SemanticProfile::StrictCore;
    const auto strict_result = GraphCompiler{}.compile(strict, seeds_for(strict));
    assert(strict_result.ok());
    assert(strict_result.graph->profile() == SemanticProfile::StrictCore);

    GraphDefinition bad_type = graph_with(
        {{CellId{1}, static_cast<CellType>(255)}});
    expect_error(bad_type, {}, CompileErrorCode::UnknownCellType, 1);

    GraphDefinition bad_channel = graph_with(
        {{CellId{1}, CellType::SENSE_CHANNEL}});
    auto channel_seeds = seeds_for(bad_channel);
    channel_seeds.cell_parameters[1].value = ContinuousValue{43.5};
    expect_error(bad_channel, channel_seeds, CompileErrorCode::InvalidParameter, 1);

    GraphDefinition bad_delay_parameter = graph_with(
        {{CellId{1}, CellType::OP_DELAY_N}});
    auto delay_seeds = seeds_for(bad_delay_parameter);
    delay_seeds.cell_parameters[0].value = DelayTicks{17};
    expect_error(bad_delay_parameter, delay_seeds,
                 CompileErrorCode::InvalidParameter, 1);
    delay_seeds.cell_parameters[0].value = DelayTicks{0};
    expect_error(bad_delay_parameter, delay_seeds,
                 CompileErrorCode::InvalidParameter, 1);

    GraphDefinition bad_mode = graph_with(
        {{CellId{1}, CellType::GATE_MIN_MAX}});
    auto bad_mode_seeds = seeds_for(bad_mode);
    bad_mode_seeds.cell_parameters[0].value =
        static_cast<MinMaxMode>(255);
    expect_error(bad_mode, bad_mode_seeds,
                 CompileErrorCode::InvalidParameter, 1);

    GraphDefinition bad_slot = graph_with(
        {{CellId{1}, CellType::OP_ABS}});
    auto bad_slot_seeds = seeds_for(bad_slot);
    bad_slot_seeds.cell_parameters[0].slot =
        static_cast<ParameterSlot>(255);
    expect_error(bad_slot, bad_slot_seeds,
                 CompileErrorCode::InvalidParameter, 1);

    GraphDefinition bad_nan = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto bad_nan_seeds = seeds_for(bad_nan);
    bad_nan_seeds.cell_parameters[0].value =
        ContinuousValue{std::numeric_limits<double>::quiet_NaN()};
    expect_error(bad_nan, bad_nan_seeds,
                 CompileErrorCode::InvalidParameter, 1);

    GraphDefinition broad_gain = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto broad_gain_seeds = seeds_for(broad_gain);
    broad_gain_seeds.cell_parameters[0].value = ContinuousValue{-9.0};
    const auto broad_gain_result =
        GraphCompiler{}.compile(broad_gain, broad_gain_seeds);
    assert(broad_gain_result.ok());
    broad_gain_seeds.cell_parameters[0].value = ContinuousValue{9.0};
    assert(GraphCompiler{}.compile(broad_gain, broad_gain_seeds).ok());

    GraphDefinition missing = graph_with({{CellId{1}, CellType::OP_ABS}});
    auto missing_seeds = seeds_for(missing);
    missing_seeds.cell_parameters.pop_back();
    expect_error(missing, missing_seeds, CompileErrorCode::MissingParameter, 1);
    auto duplicate_seeds = seeds_for(missing);
    duplicate_seeds.cell_parameters.push_back(duplicate_seeds.cell_parameters[0]);
    expect_error(missing, duplicate_seeds, CompileErrorCode::DuplicateParameter, 1);
    auto extra_seeds = seeds_for(missing);
    extra_seeds.cell_parameters.push_back(
        parameter_seed(CellId{99}, ParameterSlot::Param1, UnusedParameter{}));
    expect_error(missing, extra_seeds, CompileErrorCode::ExtraParameter, 99);

    GraphDefinition bad_weight = graph_with(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        {{EdgeId{5}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto bad_weight_seeds = seeds_for(bad_weight);
    bad_weight_seeds.edge_weights[0].initial_weight =
        std::numeric_limits<double>::quiet_NaN();
    expect_error(bad_weight, bad_weight_seeds,
                 CompileErrorCode::InvalidEdgeWeight, 5);
    auto missing_weight = seeds_for(bad_weight);
    missing_weight.edge_weights.clear();
    expect_error(bad_weight, missing_weight,
                 CompileErrorCode::MissingEdgeWeight, 5);
    auto duplicate_weight = seeds_for(bad_weight);
    duplicate_weight.edge_weights.push_back(duplicate_weight.edge_weights[0]);
    expect_error(bad_weight, duplicate_weight,
                 CompileErrorCode::DuplicateEdgeWeight, 5);
    auto extra_weight = seeds_for(bad_weight);
    extra_weight.edge_weights.push_back({EdgeId{99}, 1.0});
    expect_error(bad_weight, extra_weight,
                 CompileErrorCode::ExtraEdgeWeight, 99);
}

void test_cycles_and_explicit_recurrence() {
    GraphDefinition self_cycle = graph_with(
        {{CellId{1}, CellType::OP_ABS}},
        {{EdgeId{11}, CellId{1}, OutputPort{0}, CellId{1}, InputPort{0},
          EdgeDelay::Immediate}});
    expect_error(self_cycle, seeds_for(self_cycle),
                 CompileErrorCode::ImmediateCycle, 11);

    GraphDefinition two_cycle = graph_with(
        {{CellId{1}, CellType::OP_ABS}, {CellId{2}, CellType::OP_ABS}},
        {{EdgeId{11}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{12}, CellId{2}, OutputPort{0}, CellId{1}, InputPort{0},
          EdgeDelay::Immediate}});
    const auto two_cycle_result =
        GraphCompiler{}.compile(two_cycle, seeds_for(two_cycle));
    assert(!two_cycle_result.ok());
    assert(two_cycle_result.error->code == CompileErrorCode::ImmediateCycle);
    assert(two_cycle_result.error->object_id == 11 ||
           two_cycle_result.error->object_id == 12);

    GraphDefinition mixed = two_cycle;
    mixed.edges[1].delay = EdgeDelay::PreviousTick;
    const auto mixed_result = GraphCompiler{}.compile(mixed, seeds_for(mixed));
    assert(mixed_result.ok());
    assert(mixed_result.graph->execution_order().size() == 2);

    GraphDefinition cycle_with_downstream = graph_with(
        {{CellId{1}, CellType::OP_ABS},
         {CellId{2}, CellType::OP_ABS},
         {CellId{3}, CellType::OP_ABS}},
        {{EdgeId{10}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{20}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{30}, CellId{2}, OutputPort{0}, CellId{1}, InputPort{0},
          EdgeDelay::Immediate}});
    const auto downstream_result =
        GraphCompiler{}.compile(cycle_with_downstream,
                                seeds_for(cycle_with_downstream));
    assert(!downstream_result.ok());
    assert(downstream_result.error->code == CompileErrorCode::ImmediateCycle);
    assert(downstream_result.error->object_id != 10);
}

void test_sparse_graph_and_empty_port_ranges() {
    constexpr std::size_t cell_count = 128;
    GraphDefinition sparse;
    sparse.identity = GraphIdentity{88};
    sparse.revision = GraphRevision{4};
    for (std::size_t i = 0; i < cell_count; ++i) {
        sparse.cells.push_back(CellDefinition{
            CellId{static_cast<uint64_t>(i + 1)}, CellType::OP_ABS});
        if (i != 0) {
            sparse.edges.push_back(EdgeDefinition{
                EdgeId{static_cast<uint64_t>(i)},
                CellId{static_cast<uint64_t>(i)},
                OutputPort{0},
                CellId{static_cast<uint64_t>(i + 1)},
                InputPort{0},
                EdgeDelay::Immediate});
        }
    }
    const auto result = GraphCompiler{}.compile(sparse, seeds_for(sparse));
    assert(result.ok());
    assert(result.graph->cells().size() == cell_count);
    assert(result.graph->edges().size() == cell_count - 1);
    assert(result.graph->port_reductions().size() == cell_count);
    assert(result.graph->port_reductions()[0].edge_begin ==
           result.graph->port_reductions()[0].edge_end);
    for (std::size_t i = 1; i < cell_count; ++i) {
        assert(result.graph->port_reductions()[i].edge_end -
                   result.graph->port_reductions()[i].edge_begin ==
               1);
    }
}

struct TestCellState {
    double state{0.0};
    double aux{0.0};
    double previous{0.0};
    double output{0.0};
    double delay[16]{};
    bool latch{false};
    uint8_t delay_index{0};
    uint32_t activations{0};
};

std::vector<double> run_plan(const CompileResult& result,
                             const std::vector<std::vector<double>>& external_inputs,
                             std::size_t ticks,
                             std::size_t output_index) {
    const auto& plan = *result.graph;
    std::vector<TestCellState> state(plan.cells().size());
    std::vector<double> edge_weights(plan.edges().size(), 0.0);
    for (std::size_t edge_index = 0; edge_index < plan.edges().size(); ++edge_index) {
        edge_weights[edge_index] =
            edge_weight(result, plan.edges()[edge_index].weight_parameter_index);
    }
    std::vector<double> outputs(plan.cells().size(), 0.0);
    std::vector<double> previous_outputs(plan.cells().size(), 0.0);
    std::vector<double> observed;
    observed.reserve(ticks);

    for (std::size_t tick = 0; tick < ticks; ++tick) {
        previous_outputs = outputs;
        for (const std::size_t dense_index : plan.execution_order()) {
            double port_inputs[2]{0.0, 0.0};
            for (const auto& reduction : plan.port_reductions()) {
                if (reduction.target_index != dense_index) continue;
                for (std::size_t edge_index = reduction.edge_begin;
                     edge_index < reduction.edge_end; ++edge_index) {
                    const auto& edge = plan.edges()[edge_index];
                    const double source =
                        edge.delay == EdgeDelay::PreviousTick
                            ? previous_outputs[edge.source_index]
                            : state[edge.source_index].output;
                    port_inputs[edge.target_port.value] += source * edge_weights[edge_index];
                }
            }

            double param1 = 0.0;
            double param2 = 0.0;
            const auto& cell = plan.cells()[dense_index];
            const auto param1_value =
                legacy_kernel_parameter(initial(result, cell.parameter_indices[0]).value);
            const auto param2_value =
                legacy_kernel_parameter(initial(result, cell.parameter_indices[1]).value);
            assert(param1_value.has_value());
            assert(param2_value.has_value());
            param1 = *param1_value;
            param2 = *param2_value;
            SdscCellKernelStateView view{
                &state[dense_index].state,
                &state[dense_index].aux,
                &state[dense_index].previous,
                &state[dense_index].output,
                state[dense_index].delay,
                &state[dense_index].latch,
                &state[dense_index].delay_index,
                state[dense_index].activations,
            };
            const auto status = sdsc_cell_kernel_step(
                static_cast<uint8_t>(cell.type), param1, param2,
                port_inputs[0], port_inputs[1], external_inputs[tick].size(),
                external_inputs[tick].data(), &view);
            assert(status == SDSC_CELL_KERNEL_OK);
            if (std::abs(state[dense_index].output) > 1e-6) {
                ++state[dense_index].activations;
            }
            outputs[dense_index] = state[dense_index].output;
        }
        observed.push_back(outputs[output_index]);
    }
    return observed;
}

void test_recurrence_and_reference_reduction_order() {
    GraphDefinition recurrence = graph_with(
        {{CellId{10}, CellType::SENSE_RAW_INPUT_0},
         {CellId{20}, CellType::OP_SUM}},
        {{EdgeId{100}, CellId{10}, OutputPort{0}, CellId{20}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{101}, CellId{20}, OutputPort{0}, CellId{20}, InputPort{1},
          EdgeDelay::PreviousTick}});
    auto recurrence_seeds = seeds_for(recurrence);
    recurrence_seeds.edge_weights[0].initial_weight = 1.0;
    recurrence_seeds.edge_weights[1].initial_weight = 0.5;
    const auto recurrence_result =
        GraphCompiler{}.compile(recurrence, recurrence_seeds);
    assert(recurrence_result.ok());
    const auto y_index = std::find_if(
        recurrence_result.graph->cells().begin(),
        recurrence_result.graph->cells().end(),
        [](const auto& cell) { return cell.id == CellId{20}; }) -
        recurrence_result.graph->cells().begin();
    const auto values = run_plan(
        recurrence_result, {{1.0}, {0.0}, {0.0}}, 3, y_index);
    assert(std::abs(values[0] - 1.0) < 1e-12);
    assert(std::abs(values[1] - 0.5) < 1e-12);
    assert(std::abs(values[2] - 0.25) < 1e-12);
    GraphDefinition recurrence_permuted = recurrence;
    std::reverse(recurrence_permuted.cells.begin(), recurrence_permuted.cells.end());
    std::reverse(recurrence_permuted.edges.begin(), recurrence_permuted.edges.end());
    auto recurrence_seeds_permuted = recurrence_seeds;
    std::reverse(recurrence_seeds_permuted.cell_parameters.begin(),
                 recurrence_seeds_permuted.cell_parameters.end());
    std::reverse(recurrence_seeds_permuted.edge_weights.begin(),
                 recurrence_seeds_permuted.edge_weights.end());
    const auto recurrence_permuted_result =
        GraphCompiler{}.compile(recurrence_permuted, recurrence_seeds_permuted);
    assert(recurrence_permuted_result.ok());
    const auto permuted_values = run_plan(
        recurrence_permuted_result, {{1.0}, {0.0}, {0.0}}, 3, y_index);
    for (std::size_t i = 0; i < values.size(); ++i) {
        assert(values[i] == permuted_values[i]);
    }
    for (std::size_t i = 0; i < recurrence_result.initial_values->size(); ++i) {
        assert_same_initial_value(initial(recurrence_result, i),
                                  initial(recurrence_permuted_result, i));
    }

    GraphDefinition immediate_recurrence = recurrence;
    immediate_recurrence.edges[1].delay = EdgeDelay::Immediate;
    expect_error(immediate_recurrence, recurrence_seeds,
                 CompileErrorCode::ImmediateCycle, 101);

    GraphDefinition reduction = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::SENSE_RAW_INPUT_1},
         {CellId{3}, CellType::SENSE_RAW_INPUT_2},
         {CellId{4}, CellType::ACT_PRIMARY_POSITIVE}},
        {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{4}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{20}, CellId{2}, OutputPort{0}, CellId{4}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{30}, CellId{3}, OutputPort{0}, CellId{4}, InputPort{0},
          EdgeDelay::Immediate}});
    auto reduction_seeds = seeds_for(reduction);
    for (auto& seed : reduction_seeds.edge_weights) seed.initial_weight = 1.0;
    const auto reduction_result =
        GraphCompiler{}.compile(reduction, reduction_seeds);
    assert(reduction_result.ok());
    const auto abs_index = std::find_if(
        reduction_result.graph->cells().begin(),
        reduction_result.graph->cells().end(),
        [](const auto& cell) { return cell.id == CellId{4}; }) -
        reduction_result.graph->cells().begin();
    const auto reduced = run_plan(
        reduction_result, {{1.0e16, -1.0e16, 1.0}}, 1, abs_index);
    assert(std::abs(reduced[0] - 1.0) < 1e-12);
    const auto& reduction_range = reduction_result.graph->port_reductions()[0];
    assert(reduction_range.edge_end - reduction_range.edge_begin == 3);
    assert(reduction_result.graph->edges()[reduction_range.edge_begin].id == EdgeId{10});
    assert(reduction_result.graph->edges()[reduction_range.edge_begin + 1].id == EdgeId{20});
    assert(reduction_result.graph->edges()[reduction_range.edge_begin + 2].id == EdgeId{30});
    GraphDefinition reduction_permuted = reduction;
    std::reverse(reduction_permuted.cells.begin(), reduction_permuted.cells.end());
    std::reverse(reduction_permuted.edges.begin(), reduction_permuted.edges.end());
    auto reduction_seeds_permuted = reduction_seeds;
    std::reverse(reduction_seeds_permuted.cell_parameters.begin(),
                 reduction_seeds_permuted.cell_parameters.end());
    std::reverse(reduction_seeds_permuted.edge_weights.begin(),
                 reduction_seeds_permuted.edge_weights.end());
    const auto reduction_permuted_result =
        GraphCompiler{}.compile(reduction_permuted, reduction_seeds_permuted);
    assert(reduction_permuted_result.ok());
    const auto reduced_permuted = run_plan(
        reduction_permuted_result, {{1.0e16, -1.0e16, 1.0}}, 1, abs_index);
    assert(reduced[0] == reduced_permuted[0]);
    for (std::size_t i = 0; i < reduction_result.initial_values->size(); ++i) {
        assert_same_initial_value(initial(reduction_result, i),
                                  initial(reduction_permuted_result, i));
    }
}

void test_legacy_stateful_adapter_semantics() {
    assert(!legacy_kernel_parameter(
        ParameterValue{ChannelIndex{static_cast<std::size_t>(1ULL << 54)}}));

    GraphDefinition delay = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_DELAY_N}},
        {{EdgeId{1}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto delay_one_seeds = seeds_for(delay);
    for (auto& seed : delay_one_seeds.cell_parameters) {
        if (seed.cell == CellId{2} && seed.slot == ParameterSlot::Param1) {
            seed.value = DelayTicks{1};
        }
    }
    const auto delay_one_result =
        GraphCompiler{}.compile(delay, delay_one_seeds);
    assert(delay_one_result.ok());
    const auto delay_index = std::find_if(
        delay_one_result.graph->cells().begin(),
        delay_one_result.graph->cells().end(),
        [](const auto& cell) { return cell.id == CellId{2}; }) -
        delay_one_result.graph->cells().begin();
    const auto delay_one_values = run_plan(
        delay_one_result, {{1.0}, {2.0}, {3.0}}, 3, delay_index);
    assert(delay_one_values[0] == 0.0);
    assert(delay_one_values[1] == 1.0);
    assert(delay_one_values[2] == 2.0);

    auto delay_sixteen_seeds = delay_one_seeds;
    for (auto& seed : delay_sixteen_seeds.cell_parameters) {
        if (seed.cell == CellId{2} && seed.slot == ParameterSlot::Param1) {
            seed.value = DelayTicks{16};
        }
    }
    const auto delay_sixteen_result =
        GraphCompiler{}.compile(delay, delay_sixteen_seeds);
    assert(delay_sixteen_result.ok());
    std::vector<std::vector<double>> delay_inputs;
    for (std::size_t i = 0; i < 17; ++i) {
        delay_inputs.push_back({static_cast<double>(i + 1)});
    }
    const auto delay_sixteen_values = run_plan(
        delay_sixteen_result, delay_inputs, delay_inputs.size(), delay_index);
    for (std::size_t i = 0; i < 16; ++i) {
        assert(delay_sixteen_values[i] == 0.0);
    }
    assert(delay_sixteen_values[16] == 1.0);

    GraphDefinition ema = graph_with(
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_EMA}},
        {{EdgeId{2}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate}});
    auto ema_seeds = seeds_for(ema);
    for (auto& seed : ema_seeds.cell_parameters) {
        if (seed.cell == CellId{2} && seed.slot == ParameterSlot::Param1) {
            seed.value = ContinuousValue{0.5};
        }
    }
    const auto ema_result = GraphCompiler{}.compile(ema, ema_seeds);
    assert(ema_result.ok());
    const auto ema_index = std::find_if(
        ema_result.graph->cells().begin(),
        ema_result.graph->cells().end(),
        [](const auto& cell) { return cell.id == CellId{2}; }) -
        ema_result.graph->cells().begin();
    const auto ema_values = run_plan(ema_result, {{0.0}, {2.0}}, 2, ema_index);
    assert(ema_values[0] == 0.0);
    assert(ema_values[1] == 2.0);
}

void test_immutable_detached_plan_and_initial_ownership() {
    GraphDefinition graph = graph_with({{CellId{1}, CellType::SENSE_RAW_INPUT_0}});
    auto seeds = seeds_for(graph);
    const auto result = GraphCompiler{}.compile(graph, seeds);
    assert(result.ok());
    const auto original_id = result.graph->cells()[0].id;
    const auto original_gain = initial_double(result, result.graph->cells()[0].parameter_indices[0]);
    graph.cells[0].id = CellId{99};
    graph.cells[0].type = CellType::OP_ABS;
    seeds.cell_parameters[0].value = ContinuousValue{42.0};
    assert(result.graph->cells()[0].id == original_id);
    assert(initial_double(result, result.graph->cells()[0].parameter_indices[0]) ==
           original_gain);

    std::vector<double> runtime_parameters(result.initial_values->size());
    for (std::size_t i = 0; i < runtime_parameters.size(); ++i) {
        runtime_parameters[i] = initial_double(result, i);
    }
    runtime_parameters[result.graph->cells()[0].parameter_indices[0]] = 7.0;
    assert(initial_double(result, result.graph->cells()[0].parameter_indices[0]) ==
           original_gain);
    const auto again = GraphCompiler{}.compile(graph_with({{CellId{1}, CellType::SENSE_RAW_INPUT_0}}),
                                                seeds_for(graph_with({{CellId{1}, CellType::SENSE_RAW_INPUT_0}})));
    assert(again.ok());
    assert(runtime_parameters[result.graph->cells()[0].parameter_indices[0]] == 7.0);
}

} // namespace

int main() {
    test_empty_and_declared_disconnected_graphs();
    test_ids_parallel_edges_and_determinism();
    test_validation_diagnostics();
    test_cycles_and_explicit_recurrence();
    test_sparse_graph_and_empty_port_ranges();
    test_recurrence_and_reference_reduction_order();
    test_legacy_stateful_adapter_semantics();
    test_immutable_detached_plan_and_initial_ownership();
    return 0;
}
