#include "kun/cellular/legacy/execution_snapshot.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::Synapse;
using kun::core::CellId;
using kun::core::CompiledGraph;
using kun::core::ContinuousValue;
using kun::core::DelayTicks;
using kun::core::InitialParameterValues;
using kun::core::ParameterValue;
using kun::migration::ExecutionSnapshot;
using kun::migration::ExecutionSnapshotImportCode;
using kun::migration::ExecutionSnapshotImportResult;

Cell make_cell(uint32_t id, CellType type, double param1 = 1.0, double param2 = 0.0) {
    Cell cell;
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

Synapse make_synapse(
    uint32_t from,
    uint32_t to,
    uint8_t port = 0,
    double weight = 1.0) {
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

void assert_near(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

const Cell& cell_by_id(const CellularOrganism& organism, uint32_t id) {
    for (const auto& cell : organism.cells) {
        if (cell.id == id) return cell;
    }
    assert(false);
    return organism.cells.front();
}

const kun::migration::SnapshotCellState& snapshot_cell_by_id(
    const ExecutionSnapshot& snapshot,
    uint32_t id) {
    for (const auto& cell : snapshot.cell_states()) {
        if (cell.cell == CellId{id}) return cell;
    }
    assert(false);
    return snapshot.cell_states().front();
}

const kun::migration::SnapshotCellState& snapshot_cell_by_id(
    std::span<const kun::migration::SnapshotCellState> states,
    uint32_t id) {
    for (const auto& cell : states) {
        if (cell.cell == CellId{id}) return cell;
    }
    assert(false);
    return states.front();
}

double parameter_as_double(const ParameterValue& value) {
    if (std::holds_alternative<ContinuousValue>(value)) {
        return std::get<ContinuousValue>(value).value;
    }
    if (std::holds_alternative<DelayTicks>(value)) {
        return static_cast<double>(std::get<DelayTicks>(value).value) / 16.0;
    }
    if (std::holds_alternative<kun::core::ChannelIndex>(value)) {
        return static_cast<double>(std::get<kun::core::ChannelIndex>(value).value);
    }
    if (std::holds_alternative<kun::core::MinMaxMode>(value)) {
        return std::get<kun::core::MinMaxMode>(value) ==
                       kun::core::MinMaxMode::Max
                   ? 1.0
                   : 0.0;
    }
    return 0.0;
}

struct WalkerCell {
    CellId id{};
    CellType type{CellType::OP_EMA};
    double param1{0.0};
    double param2{0.0};
    double state_val{0.0};
    double aux_state{0.0};
    double prev_input{0.0};
    double output_val{0.0};
    double prev_output_val{0.0};
    double delay_buffer[16]{};
    bool latch_state{false};
    uint8_t delay_idx{0};
    uint32_t activation_count{0};
};

struct TestOnlyCompiledPlanWalker {
    std::shared_ptr<const CompiledGraph> plan;
    std::vector<WalkerCell> cells;

    explicit TestOnlyCompiledPlanWalker(const ExecutionSnapshot& snapshot)
        : TestOnlyCompiledPlanWalker(
              snapshot.compiled_plan(),
              snapshot.current_parameter_values(),
              snapshot.cell_states()) {}

    TestOnlyCompiledPlanWalker(
        std::shared_ptr<const CompiledGraph> imported_plan,
        std::shared_ptr<const InitialParameterValues> imported_values,
        std::span<const kun::migration::SnapshotCellState> states)
        : plan(std::move(imported_plan)),
          current_values_(std::move(imported_values)) {
        const auto values = current_values_;
        cells.reserve(plan->cells().size());
        for (const auto& compiled_cell : plan->cells()) {
            WalkerCell cell;
            cell.id = compiled_cell.id;
            cell.type = compiled_cell.type;
            cell.param1 = parameter_as_double(
                values->at(compiled_cell.parameter_indices[0]).value);
            cell.param2 = parameter_as_double(
                values->at(compiled_cell.parameter_indices[1]).value);
            const auto& captured = snapshot_cell_by_id(
                states, static_cast<uint32_t>(cell.id.value));
            cell.state_val = captured.state_val;
            cell.aux_state = captured.aux_state;
            cell.prev_input = captured.prev_input;
            cell.output_val = captured.output_val;
            cell.prev_output_val = captured.prev_output_val;
            std::copy(std::begin(captured.delay_buffer),
                      std::end(captured.delay_buffer), std::begin(cell.delay_buffer));
            cell.delay_idx = captured.delay_idx;
            cell.latch_state = captured.latch_state;
            cell.activation_count = captured.activation_count;
            cells.push_back(cell);
        }
    }

    const WalkerCell& cell_by_id(uint64_t id) const {
        for (const auto& cell : cells) {
            if (cell.id.value == id) return cell;
        }
        assert(false);
        return cells.front();
    }

    void step(const double* inputs, size_t input_count) {
        std::vector<double> ports(cells.size() * 2, 0.0);
        for (const size_t cell_index : plan->execution_order()) {
            for (const auto& reduction : plan->port_reductions()) {
                if (reduction.target_index != cell_index) continue;
                double input = 0.0;
                for (size_t edge_index = reduction.edge_begin;
                     edge_index < reduction.edge_end; ++edge_index) {
                    const auto& edge = plan->edges()[edge_index];
                    const double source_value =
                        edge.delay == kun::core::EdgeDelay::PreviousTick
                            ? cells[edge.source_index].prev_output_val
                            : cells[edge.source_index].output_val;
                    input += source_value *
                             parameter_as_double(
                                 plan_weight(edge.weight_parameter_index));
                }
                ports[cell_index * 2 + reduction.target_port.value] = input;
            }
            auto& cell = cells[cell_index];
            SdscCellKernelStateView state{
                &cell.state_val,
                &cell.aux_state,
                &cell.prev_input,
                &cell.output_val,
                cell.delay_buffer,
                &cell.latch_state,
                &cell.delay_idx,
                cell.activation_count};
            const auto status = sdsc_cell_kernel_step(
                static_cast<uint8_t>(cell.type),
                cell.param1,
                cell.param2,
                ports[cell_index * 2],
                ports[cell_index * 2 + 1],
                input_count,
                inputs,
                &state);
            assert(status == SDSC_CELL_KERNEL_OK);
            if (std::abs(cell.output_val) > 1e-6) ++cell.activation_count;
        }
        for (auto& cell : cells) cell.prev_output_val = cell.output_val;
    }

private:
    const ParameterValue& plan_weight(size_t index) const {
        for (const auto& value : current_values_->entries()) {
            if (value.binding.kind == kun::core::ParameterBindingKind::EdgeWeight &&
                value.binding.index == index) {
                return value.value;
            }
        }
        assert(false);
        return current_values_->entries().front().value;
    }

    std::shared_ptr<const InitialParameterValues> current_values_{
        nullptr};
};

// This constructor helper keeps the walker deliberately test-only: it copies
// immutable snapshot data and runs the existing native C11 kernel directly.
TestOnlyCompiledPlanWalker make_walker(const ExecutionSnapshot& snapshot) {
    return TestOnlyCompiledPlanWalker(snapshot);
}

void assert_walker_matches_source(
    const TestOnlyCompiledPlanWalker& walker,
    const CellularOrganism& source) {
    for (const auto& cell : walker.cells) {
        const auto& expected = cell_by_id(
            source, static_cast<uint32_t>(cell.id.value));
        assert_near(cell.state_val, expected.state_val);
        assert_near(cell.aux_state, expected.aux_state);
        assert_near(cell.prev_input, expected.prev_input);
        assert_near(cell.output_val, expected.output_val);
        assert_near(cell.prev_output_val, expected.prev_output_val);
        assert(cell.latch_state == expected.latch_state);
        assert(cell.delay_idx == expected.delay_idx);
        assert(cell.activation_count == expected.activation_count);
        for (size_t index = 0; index < 16; ++index) {
            assert_near(cell.delay_buffer[index],
                        expected.delay_buffer[index]);
        }
    }
}

void assert_source_unchanged(
    const CellularOrganism& actual,
    const CellularOrganism& before) {
    assert(actual.cells.size() == before.cells.size());
    assert(actual.synapses.size() == before.synapses.size());
    for (size_t i = 0; i < actual.cells.size(); ++i) {
        assert(actual.cells[i].id == before.cells[i].id);
        assert(actual.cells[i].type == before.cells[i].type);
        assert(actual.cells[i].param1 == before.cells[i].param1);
        assert(actual.cells[i].param2 == before.cells[i].param2);
        assert(actual.cells[i].state_val == before.cells[i].state_val);
        assert(actual.cells[i].aux_state == before.cells[i].aux_state);
        assert(actual.cells[i].prev_input == before.cells[i].prev_input);
        assert(actual.cells[i].output_val == before.cells[i].output_val);
        assert(actual.cells[i].prev_output_val == before.cells[i].prev_output_val);
        assert(actual.cells[i].latch_state == before.cells[i].latch_state);
        assert(actual.cells[i].delay_idx == before.cells[i].delay_idx);
        assert(actual.cells[i].activation_count == before.cells[i].activation_count);
        for (size_t j = 0; j < 16; ++j) {
            assert(actual.cells[i].delay_buffer[j] ==
                   before.cells[i].delay_buffer[j]);
        }
    }
    for (size_t i = 0; i < actual.synapses.size(); ++i) {
        assert(actual.synapses[i].from_cell_id == before.synapses[i].from_cell_id);
        assert(actual.synapses[i].to_cell_id == before.synapses[i].to_cell_id);
        assert(actual.synapses[i].to_port == before.synapses[i].to_port);
        assert(actual.synapses[i].weight == before.synapses[i].weight);
        assert(actual.synapses[i].initial_weight == before.synapses[i].initial_weight);
    }
    assert(actual.compiled_synapses_.size() == before.compiled_synapses_.size());
    for (size_t i = 0; i < actual.compiled_synapses_.size(); ++i) {
        assert(actual.compiled_synapses_[i].weight ==
               before.compiled_synapses_[i].weight);
        assert(actual.compiled_synapses_[i].initial_weight ==
               before.compiled_synapses_[i].initial_weight);
        assert(actual.compiled_synapses_[i].is_recurrent ==
               before.compiled_synapses_[i].is_recurrent);
    }
    assert(actual.compiled_actions_.size() == before.compiled_actions_.size());
    for (size_t i = 0; i < actual.compiled_actions_.size(); ++i) {
        assert(actual.compiled_actions_[i].cell_idx ==
               before.compiled_actions_[i].cell_idx);
        assert(actual.compiled_actions_[i].type ==
               before.compiled_actions_[i].type);
    }
    assert(actual.execution_order_ == before.execution_order_);
    assert(actual.out_start_ == before.out_start_);
    assert(actual.out_edges_ == before.out_edges_);
    assert(actual.flat_port_inputs_ == before.flat_port_inputs_);
    assert(actual.relaxation_steps_ == before.relaxation_steps_);
    assert(actual.relaxation_damping_ == before.relaxation_damping_);
    assert(actual.is_compiled_ == before.is_compiled_);
}

ExecutionSnapshotImportResult import(const CellularOrganism& source) {
    return kun::migration::import_execution_snapshot(
        source, kun::core::GraphIdentity{0x1234}, kun::core::GraphRevision{7});
}

void test_imports_a_compiled_ordinary_dag_and_detaches() {
    std::shared_ptr<const ExecutionSnapshot> snapshot;
    {
        CellularOrganism source;
        source.cells = {
            make_cell(10, CellType::SENSE_RAW_INPUT_0),
            make_cell(20, CellType::ACT_PRIMARY_POSITIVE),
        };
        source.synapses = {make_synapse(10, 20)};
        assert(source.compile());
        const CellularOrganism before = source;
        const auto result = import(source);
        assert(result.ok());
        snapshot = result.snapshot;
        assert_source_unchanged(source, before);
        assert(snapshot->graph_definition().identity ==
               kun::core::GraphIdentity{0x1234});
        assert(snapshot->graph_definition().revision ==
               kun::core::GraphRevision{7});
        assert(snapshot->compiled_plan()->cells().size() == 2);
        assert(snapshot->compiled_plan()->edges().size() == 1);
        assert(snapshot->cell_states().size() == 2);

        source.cells[0].param1 = 99.0;
        source.cells[0].output_val = 123.0;
        source.synapses[0].weight = -4.0;
    }

    assert_near(snapshot->cell_states()[0].output_val, 0.0);
    assert_near(
        std::get<ContinuousValue>(
            snapshot->current_parameter_values()->entries()[0].value)
            .value,
        1.0);
    assert_near(
        std::get<ContinuousValue>(
            snapshot->current_parameter_values()->entries().back().value)
            .value,
        1.0);
}

void test_walker_uses_imported_old_reduction_order() {
    CellularOrganism source;
    source.cells = {
            make_cell(30, CellType::SENSE_RAW_INPUT_0),
            make_cell(10, CellType::SENSE_RAW_INPUT_1),
            make_cell(20, CellType::SENSE_RAW_INPUT_2),
            make_cell(40, CellType::OP_SUM),
            make_cell(50, CellType::ACT_PRIMARY_POSITIVE),
    };
    source.synapses = {
            make_synapse(30, 40, 0, 1e16),
            make_synapse(10, 40, 0, -1e16),
            make_synapse(20, 40, 0, 1.0),
            make_synapse(40, 50),
    };
    assert(source.compile());
    const auto result = import(source);
    assert(result.ok());

    const double inputs[] = {1.0, 1.0, 1.0};
    CellularOrganism old_path = source;
    old_path.forward_nd(inputs, 3, false);
    auto walker = make_walker(*result.snapshot);
    walker.step(inputs, 3);
    assert_near(cell_by_id(old_path, 40).output_val, 1.0);
    assert_near(
            walker.cell_by_id(40).output_val,
            cell_by_id(old_path, 40).output_val);
}

void test_delay_import_matches_source_kernel_boundary() {
    CellularOrganism source;
    source.cells = {
            make_cell(0, CellType::SENSE_RAW_INPUT_0),
            make_cell(1, CellType::OP_DELAY_N,
                       std::numeric_limits<double>::max()),
            make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
    };
    source.synapses = {make_synapse(0, 1), make_synapse(1, 2)};
    assert(source.compile());

    CellularOrganism source_execution = source;
    const double input = 1.0;
    bool kernel_rejected = false;
    try {
            source_execution.forward_nd(&input, 1, false);
    } catch (const std::invalid_argument&) {
            kernel_rejected = true;
    }
    assert(kernel_rejected);

    const CellularOrganism before = source;
    const auto result = import(source);
    assert(!result.ok());
    assert(result.error->code ==
               kun::migration::ExecutionSnapshotImportCode::InvalidParameter);
    assert_source_unchanged(source, before);

    const auto make_boundary_source = [](double delay) {
        CellularOrganism boundary;
        boundary.cells = {
                make_cell(0, CellType::SENSE_RAW_INPUT_0),
                make_cell(1, CellType::OP_DELAY_N, delay),
                make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
        };
        boundary.synapses = {
                make_synapse(0, 1),
                make_synapse(1, 2),
        };
        assert(boundary.compile());
        return boundary;
    };
    const double accepted_delay =
        (static_cast<double>(INT_MAX) + 0.5) / 16.0;
    CellularOrganism accepted = make_boundary_source(accepted_delay);
    CellularOrganism accepted_execution = accepted;
    accepted_execution.forward_nd(&input, 1, false);
    const auto accepted_result = import(accepted);
    assert(accepted_result.ok());
    assert(std::get<DelayTicks>(
                   accepted_result.snapshot->current_parameter_values()
                       ->entries()[2]
                       .value)
                   .value == 16);

    const double rejected_delay =
        (static_cast<double>(INT_MAX) + 1.0) / 16.0;
    CellularOrganism rejected = make_boundary_source(rejected_delay);
    CellularOrganism rejected_execution = rejected;
    bool rejected_by_kernel = false;
    try {
        rejected_execution.forward_nd(&input, 1, false);
    } catch (const std::invalid_argument&) {
        rejected_by_kernel = true;
    }
    assert(rejected_by_kernel);
    const CellularOrganism rejected_before = rejected;
    const auto rejected_result = import(rejected);
    assert(!rejected_result.ok());
    assert(rejected_result.error->code ==
               kun::migration::ExecutionSnapshotImportCode::InvalidParameter);
    assert_source_unchanged(rejected, rejected_before);
}

void test_rejects_not_compiled_and_pruned_sources_without_mutation() {
    CellularOrganism not_compiled;
    not_compiled.cells = {make_cell(1, CellType::OP_ABS)};
    const CellularOrganism not_compiled_before = not_compiled;
    const auto not_compiled_result = import(not_compiled);
    assert(!not_compiled_result.ok());
    assert(not_compiled_result.error->code ==
           ExecutionSnapshotImportCode::NotCompiled);
    assert_source_unchanged(not_compiled, not_compiled_before);

    CellularOrganism pruned;
    pruned.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::ACT_PRIMARY_POSITIVE),
        make_cell(2, CellType::OP_ABS),
    };
    pruned.synapses = {make_synapse(0, 1)};
    assert(pruned.compile());
    const CellularOrganism pruned_before = pruned;
    const auto pruned_result = import(pruned);
    assert(!pruned_result.ok());
    assert(pruned_result.error->code ==
           ExecutionSnapshotImportCode::UnsupportedPrunedExecution);
    assert_source_unchanged(pruned, pruned_before);
}

void test_generic_channel_delay_and_latch_continuation() {
    CellularOrganism channel;
    channel.cells = {
        make_cell(0, CellType::SENSE_CHANNEL, 1.0, 43.0),
        make_cell(1, CellType::ACT_CHANNEL, 1.0, 43.0),
    };
    channel.synapses = {make_synapse(0, 1)};
    assert(channel.compile());
    std::array<double, 44> input{};
    input[43] = 7.25;
    channel.forward_nd(input.data(), input.size(), false);
    const auto channel_result = import(channel);
    assert(channel_result.ok());
    const auto& channel_state = snapshot_cell_by_id(*channel_result.snapshot, 0);
    assert(channel_state.type == CellType::SENSE_CHANNEL);
    assert(std::get<kun::core::ChannelIndex>(
               channel_result.snapshot->current_parameter_values()
                   ->entries()[1]
                   .value)
               .value == 43);

    CellularOrganism delay;
    delay.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::OP_DELAY_N, 1.0 / 16.0),
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
    };
    delay.synapses = {make_synapse(0, 1), make_synapse(1, 2)};
    assert(delay.compile());
    const double sequence[] = {10.0, 20.0, 30.0};
    for (double value : sequence) delay.forward_nd(&value, 1, false);
    const auto delay_result = import(delay);
    assert(delay_result.ok());
    const auto& delay_value = delay_result.snapshot->current_parameter_values()
                                  ->entries()[2]
                                  .value;
    assert(std::holds_alternative<DelayTicks>(delay_value));
    assert(std::get<DelayTicks>(delay_value).value == 1);

    auto walker = make_walker(*delay_result.snapshot);
    CellularOrganism continuation = delay;
    double next = 40.0;
    continuation.forward_nd(&next, 1, false);
    walker.step(&next, 1);
    assert_near(
        walker.cell_by_id(1).output_val,
        cell_by_id(continuation, 1).output_val);

    CellularOrganism latch;
    latch.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::GATE_HYSTERESIS, 0.5, -0.5),
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
    };
    latch.synapses = {make_synapse(0, 1), make_synapse(1, 2)};
    assert(latch.compile());
    const double latch_inputs[] = {0.0, 0.75, 0.0, -0.75, 0.0};
    for (double value : latch_inputs) latch.forward_nd(&value, 1, false);
    const auto latch_result = import(latch);
    assert(latch_result.ok());
    assert(snapshot_cell_by_id(*latch_result.snapshot, 1).latch_state == false);
    auto latch_walker = make_walker(*latch_result.snapshot);
    CellularOrganism latch_continuation = latch;
    double latch_next = 0.75;
    latch_continuation.forward_nd(&latch_next, 1, false);
    latch_walker.step(&latch_next, 1);
    assert_near(
        latch_walker.cell_by_id(1).output_val,
        cell_by_id(latch_continuation, 1).output_val);
}

void test_recurrent_flag_order_and_numerically_sensitive_reduction() {
    CellularOrganism source;
    source.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::SENSE_RAW_INPUT_1),
        make_cell(2, CellType::SENSE_RAW_INPUT_2),
        make_cell(3, CellType::OP_SUM),
        make_cell(4, CellType::ACT_PRIMARY_POSITIVE),
    };
    source.synapses = {
        make_synapse(2, 3, 0, 1.0),
        make_synapse(1, 3, 0, -1e16),
        make_synapse(0, 3, 0, 1e16),
        make_synapse(3, 3, 0, 0.5),
        make_synapse(3, 4, 0, 1.0),
    };
    source.synapses[3].is_recurrent = false;
    assert(source.compile());
    source.cells[3].prev_output_val = 0.5;
    const CellularOrganism before = source;
    const auto result = import(source);
    assert(result.ok());
    assert_source_unchanged(source, before);
    assert(result.snapshot->edge_provenance()[0].recurrent);
    assert(result.snapshot->compiled_plan()->edges()[0].delay ==
           kun::core::EdgeDelay::PreviousTick);

    auto permuted_definition = result.snapshot->graph_definition();
    auto permuted_seeds = kun::core::InitialParameterSeeds{};
    for (const auto& value :
         result.snapshot->current_parameter_values()->entries()) {
        if (value.binding.kind ==
            kun::core::ParameterBindingKind::CellParameter) {
            permuted_seeds.cell_parameters.push_back(
                {value.binding.cell, value.binding.slot, value.value});
        } else {
            permuted_seeds.edge_weights.push_back(
                {value.binding.edge,
                 std::get<ContinuousValue>(value.value).value});
        }
    }
    std::reverse(permuted_definition.cells.begin(),
                permuted_definition.cells.end());
    std::reverse(permuted_definition.edges.begin(),
                permuted_definition.edges.end());
    std::reverse(permuted_seeds.cell_parameters.begin(),
                 permuted_seeds.cell_parameters.end());
    std::reverse(permuted_seeds.edge_weights.begin(),
                 permuted_seeds.edge_weights.end());
    const auto permuted = kun::core::GraphCompiler{}.compile(
        permuted_definition, permuted_seeds);
    assert(permuted.ok());
    assert(permuted.graph->edges().size() ==
           result.snapshot->compiled_plan()->edges().size());
    for (size_t i = 0; i < permuted.graph->edges().size(); ++i) {
        const auto& expected = result.snapshot->compiled_plan()->edges()[i];
        const auto& actual = permuted.graph->edges()[i];
        assert(actual.id == expected.id);
        assert(actual.source_index == expected.source_index);
        assert(actual.target_index == expected.target_index);
        assert(actual.target_port == expected.target_port);
        assert(actual.delay == expected.delay);
    }

    CellularOrganism old_continuation = source;
    auto walker = make_walker(*result.snapshot);
    TestOnlyCompiledPlanWalker permuted_walker(
        permuted.graph, permuted.initial_values,
        result.snapshot->cell_states());
    const std::array<std::array<double, 3>, 3> inputs{{
        {{1.0, 1.0, 1.0}},
        {{2.0, -3.0, 0.5}},
        {{-4.0, 5.0, 2.0}},
    }};
    size_t tick_index = 0;
    for (const auto& tick : inputs) {
        old_continuation.forward_nd(tick.data(), tick.size(), false);
        walker.step(tick.data(), tick.size());
        permuted_walker.step(tick.data(), tick.size());
        assert_walker_matches_source(walker, old_continuation);
        assert_walker_matches_source(permuted_walker, old_continuation);
        if (tick_index++ == 0) {
            assert_near(walker.cell_by_id(3).output_val, 1.0);
        }
    }
}

void test_parallel_provenance_and_oja_live_weight() {
    CellularOrganism source;
    source.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::ACT_PRIMARY_POSITIVE),
    };
    source.synapses = {
        make_synapse(0, 1, 0, 2.5),
        make_synapse(0, 1, 0, 0.75),
    };
    source.synapses[0].initial_weight = 1.0;
    source.synapses[1].initial_weight = 0.25;
    assert(source.compile());
    // The legacy raw array is allowed to be reordered after compilation; the
    // compile-time weight anchors must still recover the parallel ancestry.
    std::reverse(source.synapses.begin(), source.synapses.end());
    source.compiled_synapses_[0].hebbian_rate = 0.1;
    source.compiled_synapses_[0].hebbian_decay = 0.02;
    const double input = 2.0;
    source.forward_nd(&input, 1, true);
    assert(source.compiled_synapses_[0].weight !=
           source.synapses[0].weight);
    const auto result = import(source);
    assert(result.ok());
    assert(result.snapshot->edge_provenance().size() == 2);
    bool saw_first = false;
    bool saw_second = false;
    for (const auto& edge : result.snapshot->edge_provenance()) {
        if (edge.raw_declared_initial_weight == 1.0) {
            saw_first = true;
            assert(edge.compile_time_weight == 2.5);
        }
        if (edge.raw_declared_initial_weight == 0.25) {
            saw_second = true;
            assert(edge.compile_time_weight == 0.75);
        }
    }
    assert(saw_first && saw_second);
    assert(result.snapshot->edge_provenance()[0].live_weight ==
           source.compiled_synapses_[0].weight);

    CellularOrganism signed_zero;
    signed_zero.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::ACT_PRIMARY_POSITIVE),
    };
    signed_zero.synapses = {make_synapse(
        0, 1, 0, -0.0)};
    assert(signed_zero.compile());
    signed_zero.compiled_synapses_[0].initial_weight = 0.0;
    assert(import(signed_zero).ok());

    CellularOrganism old_continuation = source;
    auto walker = make_walker(*result.snapshot);
    const std::array<std::array<double, 1>, 3> continuation_inputs{{
        {{-1.5}},
        {{0.25}},
        {{3.0}},
    }};
    for (const auto& tick : continuation_inputs) {
        old_continuation.forward_nd(tick.data(), tick.size(), false);
        walker.step(tick.data(), tick.size());
        assert_walker_matches_source(walker, old_continuation);
    }

    CellularOrganism ambiguous = source;
    ambiguous.synapses[0].weight = 1.0;
    ambiguous.synapses[1].weight = 1.0;
    ambiguous.compiled_synapses_[0].initial_weight = 1.0;
    ambiguous.compiled_synapses_[1].initial_weight = 1.0;
    const CellularOrganism ambiguous_before = ambiguous;
    const auto ambiguous_result = import(ambiguous);
    assert(!ambiguous_result.ok());
    assert(ambiguous_result.error->code ==
           ExecutionSnapshotImportCode::AmbiguousParallelEdgeProvenance);
    assert_source_unchanged(ambiguous, ambiguous_before);
}

void test_rejects_malformed_metadata_and_unsupported_features() {
    CellularOrganism malformed;
    malformed.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::ACT_PRIMARY_POSITIVE),
    };
    malformed.synapses = {make_synapse(0, 1)};
    assert(malformed.compile());
    malformed.out_edges_[0] = 999999;
    const CellularOrganism malformed_before = malformed;
    const auto malformed_result = import(malformed);
    assert(!malformed_result.ok());
    assert(malformed_result.error->code == ExecutionSnapshotImportCode::InvalidCsr);
    assert_source_unchanged(malformed, malformed_before);

    CellularOrganism bad_endpoint = malformed;
    bad_endpoint.out_edges_[0] = 0;
    bad_endpoint.compiled_synapses_[0].to_idx = 99;
    const auto endpoint_result = import(bad_endpoint);
    assert(!endpoint_result.ok());
    assert(endpoint_result.error->code ==
           ExecutionSnapshotImportCode::InvalidCompiledEdge);

    CellularOrganism bad_order = malformed;
    bad_order.out_edges_[0] = 0;
    bad_order.execution_order_[0] = bad_order.execution_order_[1];
    const auto order_result = import(bad_order);
    assert(!order_result.ok());
    assert(order_result.error->code ==
           ExecutionSnapshotImportCode::InvalidExecutionOrder);

    CellularOrganism bad_state = malformed;
    bad_state.out_edges_[0] = 0;
    bad_state.cells[0].state_val = std::numeric_limits<double>::quiet_NaN();
    const auto state_result = import(bad_state);
    assert(!state_result.ok());
    assert(state_result.error->code ==
           ExecutionSnapshotImportCode::InvalidComputationalState);

    CellularOrganism ignored_port;
    ignored_port.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::SENSE_RAW_INPUT_1),
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
    };
    ignored_port.synapses = {make_synapse(0, 1), make_synapse(1, 2)};
    assert(ignored_port.compile());
    const auto ignored_result = import(ignored_port);
    assert(!ignored_result.ok());
    assert(ignored_result.error->code ==
           ExecutionSnapshotImportCode::UnsupportedIgnoredPort);

    CellularOrganism advanced;
    advanced.cells = {make_cell(0, CellType::OP_ABS)};
    assert(advanced.compile());
    advanced.enable_mla(1, 1);
    const CellularOrganism advanced_before = advanced;
    const auto advanced_result = import(advanced);
    assert(!advanced_result.ok());
    assert(advanced_result.error->code ==
           ExecutionSnapshotImportCode::UnsupportedAdvancedOrgan);
    assert_source_unchanged(advanced, advanced_before);
}

void test_rejects_incomplete_action_metadata() {
    CellularOrganism source;
    source.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::OP_SUM),
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE),
    };
    source.synapses = {make_synapse(0, 1), make_synapse(1, 2)};
    assert(source.compile());
    source.compiled_actions_.push_back({1, CellType::OP_SUM});
    const CellularOrganism before = source;
    const auto result = import(source);
    assert(!result.ok());
    assert(result.error->code ==
           ExecutionSnapshotImportCode::InvalidActionMetadata);
    assert_source_unchanged(source, before);
}

void test_imports_a_modest_sparse_graph_without_edge_caps() {
    constexpr size_t cell_count = 256;
    CellularOrganism source;
    source.cells.reserve(cell_count);
    source.synapses.reserve(cell_count - 1);
    for (size_t index = 0; index < cell_count; ++index) {
        const uint32_t id = static_cast<uint32_t>(100000 - index);
        const CellType type =
            index == 0
                ? CellType::SENSE_RAW_INPUT_0
                : (index + 1 == cell_count
                       ? CellType::ACT_PRIMARY_POSITIVE
                       : CellType::OP_ABS);
        source.cells.push_back(make_cell(id, type));
        if (index != 0) {
            source.synapses.push_back(make_synapse(
                static_cast<uint32_t>(100000 - (index - 1)),
                id, 0, 1.0 + static_cast<double>(index)));
        }
    }
    assert(source.compile());
    const auto result = import(source);
    assert(result.ok());
    assert(result.snapshot->cell_states().size() == cell_count);
    assert(result.snapshot->edge_provenance().size() == cell_count - 1);
}

void test_empty_compiled_graph_is_supported() {
    CellularOrganism empty;
    assert(empty.compile());
    const auto result = import(empty);
    assert(result.ok());
    assert(result.snapshot->graph_definition().cells.empty());
    assert(result.snapshot->compiled_plan()->edges().empty());
    assert(result.snapshot->cell_states().empty());
}

}  // namespace

int main() {
    test_imports_a_compiled_ordinary_dag_and_detaches();
    test_walker_uses_imported_old_reduction_order();
    test_delay_import_matches_source_kernel_boundary();
    test_rejects_not_compiled_and_pruned_sources_without_mutation();
    test_generic_channel_delay_and_latch_continuation();
    test_recurrent_flag_order_and_numerically_sensitive_reduction();
    test_parallel_provenance_and_oja_live_weight();
    test_rejects_malformed_metadata_and_unsupported_features();
    test_rejects_incomplete_action_metadata();
    test_imports_a_modest_sparse_graph_without_edge_caps();
    test_empty_compiled_graph_is_supported();
    return 0;
}
