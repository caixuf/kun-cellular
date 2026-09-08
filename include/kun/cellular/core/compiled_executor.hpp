#pragma once

#include "kun/cellular/core/execution_measurement.hpp"
#include "kun/cellular/core/execution_control.hpp"
#include "kun/cellular/core/kernel_bridge.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/strict_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace kun::core {

enum class CompiledExecutorErrorCode : uint8_t {
    InvalidPlan,
    UnsupportedProfile,
    BindingMismatch,
    InvalidInput,
    InvalidParameter,
    InvalidGraphLayout,
    InvalidExecutionControl,
    KernelFailure,
};

struct CompiledExecutorError {
    CompiledExecutorErrorCode code{CompiledExecutorErrorCode::InvalidPlan};
    std::string_view reason;
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
    SdscCellKernelStatus kernel_status{SDSC_CELL_KERNEL_OK};
};

struct CompiledExecutionResult {
    ExecutionMeasurementView measurement{};
    std::optional<CompiledExecutorError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct CompiledPrepareResult {
    std::shared_ptr<class CompiledExecutor> executor;
    std::optional<CompiledExecutorError> error;

    bool ok() const { return executor != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class CompiledExecutor final {
public:
    static CompiledPrepareResult prepare(
        std::shared_ptr<const CompiledGraph> plan) {
        if (!plan) {
            return {nullptr, CompiledExecutorError{
                CompiledExecutorErrorCode::InvalidPlan,
                "compiled executor requires an owned immutable plan"}};
        }
        if (!semantic_profile_from_code(static_cast<uint8_t>(plan->profile()))) {
            return {nullptr, CompiledExecutorError{
                CompiledExecutorErrorCode::UnsupportedProfile,
                "compiled executor plan uses an unknown semantic profile"}};
        }
        auto candidate = std::shared_ptr<CompiledExecutor>(
            new CompiledExecutor(std::move(plan)));
        if (const auto error = candidate->prepare_layout()) {
            return {nullptr, *error};
        }
        return {std::move(candidate), std::nullopt};
    }

    static CompiledPrepareResult prepare(const CompileResult& compiled) {
        return prepare(compiled.graph);
    }

    CompiledPrepareResult reprepare(
        std::shared_ptr<const CompiledGraph> plan) const {
        return prepare(std::move(plan));
    }

    const std::shared_ptr<const CompiledGraph>& plan() const { return plan_; }

    // The returned spans refer to executor-owned committed storage. They remain
    // stable across failed steps and expire on the next successful step or
    // reprepare.
    ExecutionMeasurementView last_measurement() const {
        return ExecutionMeasurementView{
            committed_tick_,
            std::span<const ExecutedCellMeasurement>(committed_cells_),
            std::span<const ReducedPortMeasurement>(committed_ports_),
            std::span<const EdgeTransmissionMeasurement>(committed_edges_),
            plan_->identity(),
            plan_->revision(),
            plan_->profile()};
    }

    CompiledExecutionResult step(
        RuntimeState& runtime,
        std::span<const double> inputs) {
        return step(runtime, inputs, ExecutionControlView{});
    }

    CompiledExecutionResult step(
        RuntimeState& runtime,
        std::span<const double> inputs,
        ExecutionControlView control) {
        const auto binding_error = check_binding(runtime);
        if (binding_error.has_value()) return {{}, binding_error};
        if (!control.valid_for(cells_.size())) {
            return failure(
                CompiledExecutorErrorCode::InvalidExecutionControl,
                "execution control must be empty or cover every compiled cell");
        }
        if (!inputs.empty() && inputs.data() == nullptr) {
            return failure(
                CompiledExecutorErrorCode::InvalidInput,
                "non-empty input span has a null data pointer");
        }
        if (profile() == SemanticProfile::StrictCore) {
            for (const double input : inputs) {
                if (!std::isfinite(input)) {
                    return failure(
                        CompiledExecutorErrorCode::InvalidInput,
                        "submitted strict input tensor contains a non-finite value");
                }
            }
        }

        std::copy(
            runtime.cells_.begin(), runtime.cells_.end(), working_cells_.begin());
        pending_cells_.clear();
        pending_ports_.clear();
        pending_edges_.clear();
        const uint64_t next_tick = runtime.tick_ + 1;

        for (const std::size_t dense_index : execution_order_) {
            const PreparedCell& prepared = cells_[dense_index];
            auto& cell = working_cells_[dense_index];
            double port_inputs[2]{0.0, 0.0};
            for (uint32_t port = 0; port < prepared.input_port_count; ++port) {
                const PortRange range = prepared.ports[port];
                double reduced = 0.0;
                for (std::size_t edge_index = range.begin;
                     edge_index < range.end; ++edge_index) {
                    const PreparedEdge& edge = edges_[edge_index];
                    const auto& source = working_cells_[edge.source_index];
                    const bool muted =
                        control.dormant(edge.source_index) ||
                        control.dormant(edge.target_index);
                    const double source_value = muted
                        ? 0.0
                        : (edge.delay == EdgeDelay::PreviousTick
                            ? source.prev_output_val
                            : source.output_val);
                    const auto* parameter =
                        runtime.parameter_at(edge.weight_parameter_index);
                    if (!parameter) {
                        return failure(
                            CompiledExecutorErrorCode::InvalidParameter,
                            "compiled edge refers to a missing live weight",
                            true, cell.cell, true, edge.id);
                    }

                    double contribution = 0.0;
                    if (muted) {
                        pending_edges_.push_back(EdgeTransmissionMeasurement{
                            edge.id,
                            source.cell,
                            cell.cell,
                            InputPort{port},
                            edge.delay,
                            source_value,
                            contribution});
                        continue;
                    }
                    if (profile() == SemanticProfile::StrictCore) {
                        if (!std::holds_alternative<ContinuousValue>(*parameter)) {
                            return failure(
                                CompiledExecutorErrorCode::InvalidParameter,
                                "strict edge weight is not a continuous live value",
                                true, cell.cell, true, edge.id);
                        }
                        const double weight =
                            std::get<ContinuousValue>(*parameter).value;
                        if (!std::isfinite(weight) ||
                            !std::isfinite(source_value)) {
                            return failure(
                                CompiledExecutorErrorCode::InvalidInput,
                                "strict edge source or weight is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                        contribution = source_value * weight;
                        if (!std::isfinite(contribution)) {
                            return failure(
                                CompiledExecutorErrorCode::InvalidInput,
                                "strict edge contribution is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                    } else {
                        const auto weight = legacy_kernel_parameter(*parameter);
                        if (!weight.has_value() || !std::isfinite(*weight)) {
                            return failure(
                                CompiledExecutorErrorCode::InvalidParameter,
                                "live edge weight cannot be represented by the legacy kernel",
                                true, cell.cell, true, edge.id);
                        }
                        contribution = source_value * *weight;
                    }
                    reduced += contribution;
                    if (profile() == SemanticProfile::StrictCore &&
                        !std::isfinite(reduced)) {
                        return failure(
                            CompiledExecutorErrorCode::InvalidInput,
                            "strict port reduction is non-finite",
                            true, cell.cell, true, edge.id);
                    }
                    pending_edges_.push_back(EdgeTransmissionMeasurement{
                        edge.id,
                        source.cell,
                        cell.cell,
                        InputPort{port},
                        edge.delay,
                        source_value,
                        contribution});
                }
                port_inputs[port] = reduced;
                pending_ports_.push_back(
                    ReducedPortMeasurement{cell.cell, InputPort{port}, reduced});
            }

            const auto* first =
                runtime.parameter_at(prepared.parameter_indices[0]);
            const auto* second =
                runtime.parameter_at(prepared.parameter_indices[1]);
            if (!first || !second) {
                return failure(
                    CompiledExecutorErrorCode::InvalidParameter,
                    "compiled cell refers to a missing typed parameter",
                    true, cell.cell);
            }

            if (control.dormant(dense_index)) {
                cell.output_val = 0.0;
                pending_cells_.push_back(
                    ExecutedCellMeasurement{cell.cell, cell.type, false, 0.0});
                continue;
            }

            SdscCellKernelStatus status = SDSC_CELL_KERNEL_OK;
            if (profile() == SemanticProfile::StrictCore) {
                SdscCellKernelStrictStateView state{
                    &cell.state_val,
                    &cell.aux_state,
                    &cell.prev_input,
                    &cell.output_val,
                    cell.delay_buffer.data(),
                    &cell.latch_state,
                    &cell.delay_idx,
                    &cell.initialized,
                    &cell.activation_count};
                status = strict_cell_kernel_step(
                    prepared.type,
                    *first,
                    *second,
                    port_inputs[0],
                    port_inputs[1],
                    inputs,
                    &state);
            } else {
                const auto param1 = legacy_kernel_parameter(*first);
                const auto param2 = legacy_kernel_parameter(*second);
                if (!param1.has_value() || !param2.has_value()) {
                    return failure(
                        CompiledExecutorErrorCode::InvalidParameter,
                        "typed parameter cannot be represented by the legacy kernel",
                        true, cell.cell);
                }
                SdscCellKernelStateView state{
                    &cell.state_val,
                    &cell.aux_state,
                    &cell.prev_input,
                    &cell.output_val,
                    cell.delay_buffer.data(),
                    &cell.latch_state,
                    &cell.delay_idx,
                    cell.activation_count};
                status = sdsc_cell_kernel_step(
                    static_cast<uint8_t>(prepared.type),
                    *param1,
                    *param2,
                    port_inputs[0],
                    port_inputs[1],
                    inputs.size(),
                    inputs.data(),
                    &state);
                if (status == SDSC_CELL_KERNEL_OK) cell.initialized = true;
            }
            if (status != SDSC_CELL_KERNEL_OK) {
                return failure(
                    CompiledExecutorErrorCode::KernelFailure,
                    sdsc_cell_kernel_status_string(status),
                    true, cell.cell, false, EdgeId{0}, status);
            }
            if (!detail::valid_runtime_cell(cell)) {
                return failure(
                    CompiledExecutorErrorCode::KernelFailure,
                    "cell kernel produced state outside the finite runtime domain",
                    true, cell.cell);
            }
            if (std::abs(cell.output_val) > 1e-6) ++cell.activation_count;
            pending_cells_.push_back(
                ExecutedCellMeasurement{cell.cell, cell.type, true, cell.output_val});
        }

        for (std::size_t index = 0; index < working_cells_.size(); ++index) {
            auto& cell = working_cells_[index];
            if (!std::isfinite(cell.output_val)) {
                return failure(
                    CompiledExecutorErrorCode::KernelFailure,
                    "graph produced a non-finite output");
            }
            if (!control.dormant(index)) cell.prev_output_val = cell.output_val;
        }

        std::copy(
            working_cells_.begin(), working_cells_.end(), runtime.cells_.begin());
        runtime.tick_ = next_tick;
        pending_cells_.swap(committed_cells_);
        pending_ports_.swap(committed_ports_);
        pending_edges_.swap(committed_edges_);
        committed_tick_ = next_tick;
        return {last_measurement(), std::nullopt};
    }

    CompiledExecutionResult step(
        RuntimeState& runtime,
        const double* inputs,
        std::size_t input_count) {
        if (inputs == nullptr && input_count != 0) {
            return failure(
                CompiledExecutorErrorCode::InvalidInput,
                "non-zero input count has a null data pointer");
        }
        return step(runtime, std::span<const double>(inputs, input_count));
    }

private:
    struct PortRange {
        std::size_t begin{0};
        std::size_t end{0};
    };

    struct PreparedCell {
        std::size_t dense_index{0};
        CellId id{};
        CellType type{CellType::OP_EMA};
        std::array<std::size_t, 2> parameter_indices{};
        uint8_t input_port_count{0};
        std::array<PortRange, 2> ports{};
    };

    struct PreparedEdge {
        EdgeId id{};
        std::size_t source_index{0};
        std::size_t target_index{0};
        EdgeDelay delay{EdgeDelay::Immediate};
        std::size_t weight_parameter_index{0};
    };

    explicit CompiledExecutor(std::shared_ptr<const CompiledGraph> plan)
        : plan_(std::move(plan)) {}

    SemanticProfile profile() const { return plan_->profile(); }

    std::optional<CompiledExecutorError> prepare_layout() {
        const auto graph_cells = plan_->cells();
        const auto graph_edges = plan_->edges();
        const auto graph_reductions = plan_->port_reductions();
        const auto order = plan_->execution_order();
        if (graph_cells.size() != order.size() ||
            graph_reductions.size() >
                graph_cells.size() * 2 ||
            plan_->parameter_binding_count() <
                graph_cells.size() * 2 + graph_edges.size()) {
            return CompiledExecutorError{
                CompiledExecutorErrorCode::InvalidGraphLayout,
                "compiled plan dimensions are inconsistent"};
        }

        cells_.resize(graph_cells.size());
        execution_order_.assign(order.begin(), order.end());
        std::vector<bool> seen(graph_cells.size(), false);
        for (std::size_t position = 0; position < order.size(); ++position) {
            const std::size_t dense_index = order[position];
            if (dense_index >= graph_cells.size() || seen[dense_index]) {
                return CompiledExecutorError{
                    CompiledExecutorErrorCode::InvalidGraphLayout,
                    "compiled execution order is not a permutation"};
            }
            seen[dense_index] = true;
        }
        for (std::size_t index = 0; index < graph_cells.size(); ++index) {
            const auto& cell = graph_cells[index];
            cells_[index] = PreparedCell{
                index, cell.id, cell.type, cell.parameter_indices,
                cell.input_port_count, {}};
        }

        for (const auto& reduction : graph_reductions) {
            if (reduction.target_index >= cells_.size() ||
                reduction.target_port.value >= 2 ||
                reduction.edge_begin > reduction.edge_end ||
                reduction.edge_end > graph_edges.size()) {
                return CompiledExecutorError{
                    CompiledExecutorErrorCode::InvalidGraphLayout,
                    "compiled port reduction range is invalid"};
            }
            auto& range = cells_[reduction.target_index]
                              .ports[reduction.target_port.value];
            if (range.begin != 0 || range.end != 0) {
                return CompiledExecutorError{
                    CompiledExecutorErrorCode::InvalidGraphLayout,
                    "compiled plan contains duplicate port reductions"};
            }
            range = PortRange{reduction.edge_begin, reduction.edge_end};
        }

        edges_.resize(graph_edges.size());
        for (std::size_t index = 0; index < graph_edges.size(); ++index) {
            const auto& edge = graph_edges[index];
            if (edge.source_index >= cells_.size() ||
                edge.target_index >= cells_.size() ||
                edge.target_port.value >= 2) {
                return CompiledExecutorError{
                    CompiledExecutorErrorCode::InvalidGraphLayout,
                    "compiled edge endpoint is outside the dense cell store",
                    false, CellId{0}, true, edge.id};
            }
            const auto range =
                cells_[edge.target_index].ports[edge.target_port.value];
            if (index < range.begin || index >= range.end) {
                return CompiledExecutorError{
                    CompiledExecutorErrorCode::InvalidGraphLayout,
                    "compiled edge is outside its cached port reduction",
                    false, CellId{0}, true, edge.id};
            }
            edges_[index] = PreparedEdge{
                edge.id,
                edge.source_index,
                edge.target_index,
                edge.delay,
                edge.weight_parameter_index};
        }

        working_cells_.resize(cells_.size());
        pending_cells_.reserve(cells_.size());
        pending_ports_.reserve(graph_reductions.size());
        pending_edges_.reserve(graph_edges.size());
        committed_cells_.reserve(cells_.size());
        committed_ports_.reserve(graph_reductions.size());
        committed_edges_.reserve(graph_edges.size());
        return std::nullopt;
    }

    std::optional<CompiledExecutorError> check_binding(
        const RuntimeState& runtime) const {
        if (!runtime.bound_to(*plan_) ||
            runtime.cells_.size() != cells_.size()) {
            return CompiledExecutorError{
                CompiledExecutorErrorCode::BindingMismatch,
                "runtime is bound to a different compiled plan; prepare or reprepare explicitly"};
        }
        return std::nullopt;
    }

    static CompiledExecutionResult failure(
        CompiledExecutorErrorCode code,
        std::string_view reason,
        bool has_cell = false,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0},
        SdscCellKernelStatus kernel_status = SDSC_CELL_KERNEL_OK) {
        return {
            {},
            CompiledExecutorError{
                code, reason, has_cell, cell, has_edge, edge, kernel_status}};
    }

    std::shared_ptr<const CompiledGraph> plan_;
    std::vector<PreparedCell> cells_;
    std::vector<PreparedEdge> edges_;
    std::vector<std::size_t> execution_order_;
    std::vector<RuntimeCellState> working_cells_;
    std::vector<ExecutedCellMeasurement> pending_cells_;
    std::vector<ReducedPortMeasurement> pending_ports_;
    std::vector<EdgeTransmissionMeasurement> pending_edges_;
    std::vector<ExecutedCellMeasurement> committed_cells_;
    std::vector<ReducedPortMeasurement> committed_ports_;
    std::vector<EdgeTransmissionMeasurement> committed_edges_;
    uint64_t committed_tick_{0};
};

}  // namespace kun::core
