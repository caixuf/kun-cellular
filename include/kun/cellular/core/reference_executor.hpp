#pragma once

#include "kun/cellular/core/execution_measurement.hpp"
#include "kun/cellular/core/execution_control.hpp"
#include "kun/cellular/core/kernel_bridge.hpp"
#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/core/strict_kernel.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace kun::core {

enum class ReferenceExecutionErrorCode : uint8_t {
    UnsupportedProfile,
    InvalidInput,
    InvalidParameter,
    KernelFailure,
};

struct ReferenceExecutionError {
    ReferenceExecutionErrorCode code{ReferenceExecutionErrorCode::KernelFailure};
    std::string reason;
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
    SdscCellKernelStatus kernel_status{SDSC_CELL_KERNEL_OK};
};

struct ReferenceExecutionResult {
    ExecutionMeasurement measurement;
    std::optional<ReferenceExecutionError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class ReferenceExecutor final {
public:
    ReferenceExecutionResult step(
        RuntimeState& runtime,
        std::span<const double> inputs) const {
        return step(runtime, inputs, ExecutionControlView{});
    }

    ReferenceExecutionResult step(
        RuntimeState& runtime,
        std::span<const double> inputs,
        ExecutionControlView control) const {
        if (runtime.profile() == SemanticProfile::StrictCore) {
            return step_profile<true>(runtime, inputs, control);
        }
        if (runtime.profile() != SemanticProfile::LegacyCompatible) {
            return failure(
                ReferenceExecutionErrorCode::UnsupportedProfile,
                "runtime uses an unknown semantic profile");
        }
        return step_profile<false>(runtime, inputs, control);
    }

    ReferenceExecutionResult step(
        RuntimeState& runtime,
        const double* inputs,
        std::size_t input_count) const {
        if (inputs == nullptr && input_count != 0) {
            return failure(
                ReferenceExecutionErrorCode::InvalidInput,
                "non-zero input count has a null data pointer");
        }
        return step(runtime, std::span<const double>(inputs, input_count));
    }

private:
    template <bool Strict>
    ReferenceExecutionResult step_profile(
        RuntimeState& runtime,
        std::span<const double> inputs,
        ExecutionControlView control) const {
        if (!control.valid_for(runtime.plan_->cells().size())) {
            return failure(
                ReferenceExecutionErrorCode::InvalidInput,
                "execution control must be empty or cover every compiled cell");
        }
        if (!inputs.empty() && inputs.data() == nullptr) {
            return failure(
                ReferenceExecutionErrorCode::InvalidInput,
                "non-empty input span has a null data pointer");
        }
        if constexpr (Strict) {
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                if (!std::isfinite(inputs[index])) {
                    return failure(
                        ReferenceExecutionErrorCode::InvalidInput,
                        "submitted strict input tensor contains a non-finite value");
                }
            }
        }

        std::vector<RuntimeCellState> working(
            runtime.cells_.begin(), runtime.cells_.end());
        ExecutionMeasurement measurement;
        measurement.tick = runtime.tick_ + 1;
        measurement.identity = runtime.plan_->identity();
        measurement.revision = runtime.plan_->revision();
        measurement.profile = runtime.plan_->profile();
        measurement.cells.reserve(runtime.plan_->execution_order().size());
        measurement.ports.reserve(runtime.plan_->port_reductions().size());
        measurement.edges.reserve(runtime.plan_->edges().size());

        for (const std::size_t dense_index : runtime.plan_->execution_order()) {
            if (dense_index >= working.size()) {
                return failure(
                    ReferenceExecutionErrorCode::KernelFailure,
                    "compiled execution order contains an out-of-range cell");
            }
            auto& cell = working[dense_index];
            double port_inputs[2]{0.0, 0.0};
            for (const auto& reduction : runtime.plan_->port_reductions()) {
                if (reduction.target_index != dense_index) continue;
                if (reduction.target_port.value >= 2 ||
                    reduction.edge_begin > reduction.edge_end ||
                    reduction.edge_end > runtime.plan_->edges().size()) {
                    return failure(
                        ReferenceExecutionErrorCode::KernelFailure,
                        "compiled port reduction range is invalid", true, cell.cell);
                }
                double reduced = 0.0;
                for (std::size_t edge_index = reduction.edge_begin;
                     edge_index < reduction.edge_end; ++edge_index) {
                    const auto& edge = runtime.plan_->edges()[edge_index];
                    if (edge.source_index >= working.size() ||
                        edge.target_index != dense_index ||
                        edge.target_port != reduction.target_port) {
                        return failure(
                            ReferenceExecutionErrorCode::KernelFailure,
                            "compiled edge does not match its port reduction",
                            true, cell.cell, true, edge.id);
                    }
                    const auto* parameter =
                        runtime.parameter_at(edge.weight_parameter_index);
                    if (!parameter) {
                        return failure(
                            ReferenceExecutionErrorCode::InvalidParameter,
                            "compiled edge refers to a missing live weight",
                            true, cell.cell, true, edge.id);
                    }
                    const auto& source = working[edge.source_index];
                    const bool muted =
                        control.dormant(edge.source_index) ||
                        control.dormant(edge.target_index);
                    const double source_value = muted
                        ? 0.0
                        : (edge.delay == EdgeDelay::PreviousTick
                            ? source.prev_output_val
                            : source.output_val);
                    double contribution = 0.0;
                    if (muted) {
                        measurement.edges.push_back(EdgeTransmissionMeasurement{
                            edge.id,
                            source.cell,
                            cell.cell,
                            edge.target_port,
                            edge.delay,
                            source_value,
                            contribution});
                        continue;
                    }
                    if constexpr (Strict) {
                        if (!std::holds_alternative<ContinuousValue>(*parameter)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidParameter,
                                "strict edge weight is not a continuous live value",
                                true, cell.cell, true, edge.id);
                        }
                        const double weight =
                            std::get<ContinuousValue>(*parameter).value;
                        if (!std::isfinite(weight)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidParameter,
                                "strict edge weight is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                        if (!std::isfinite(source_value)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidInput,
                                "strict edge source value is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                        contribution = source_value * weight;
                        if (!std::isfinite(contribution)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidInput,
                                "strict edge contribution is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                    } else {
                        const auto weight = legacy_kernel_parameter(*parameter);
                        if (!weight.has_value() || !std::isfinite(*weight)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidParameter,
                                "live edge weight cannot be represented by the legacy kernel",
                                true, cell.cell, true, edge.id);
                        }
                        contribution = source_value * *weight;
                    }
                    reduced += contribution;
                    if constexpr (Strict) {
                        if (!std::isfinite(reduced)) {
                            return failure(
                                ReferenceExecutionErrorCode::InvalidInput,
                                "strict port reduction is non-finite",
                                true, cell.cell, true, edge.id);
                        }
                    }
                    measurement.edges.push_back(EdgeTransmissionMeasurement{
                        edge.id,
                        source.cell,
                        cell.cell,
                        edge.target_port,
                        edge.delay,
                        source_value,
                        contribution});
                }
                port_inputs[reduction.target_port.value] = reduced;
                measurement.ports.push_back(
                    ReducedPortMeasurement{cell.cell, reduction.target_port, reduced});
            }

            const auto& compiled_cell = runtime.plan_->cells()[dense_index];
            const auto* first =
                runtime.parameter_at(compiled_cell.parameter_indices[0]);
            const auto* second =
                runtime.parameter_at(compiled_cell.parameter_indices[1]);
            if (!first || !second) {
                return failure(
                    ReferenceExecutionErrorCode::InvalidParameter,
                    "compiled cell refers to a missing typed parameter",
                    true, cell.cell);
            }

            if (control.dormant(dense_index)) {
                cell.output_val = 0.0;
                measurement.cells.push_back(
                    ExecutedCellMeasurement{cell.cell, cell.type, false, 0.0});
                continue;
            }

            SdscCellKernelStatus status = SDSC_CELL_KERNEL_OK;
            if constexpr (Strict) {
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
                    compiled_cell.type,
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
                        ReferenceExecutionErrorCode::InvalidParameter,
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
                    static_cast<uint8_t>(compiled_cell.type),
                    *param1,
                    *param2,
                    port_inputs[0],
                    port_inputs[1],
                    inputs.size(),
                    inputs.data(),
                    &state);
            }
            if (status != SDSC_CELL_KERNEL_OK) {
                if constexpr (Strict) {
                    return failure(
                        ReferenceExecutionErrorCode::KernelFailure,
                        sdsc_cell_kernel_status_string(status),
                        true, cell.cell, false, EdgeId{0}, status);
                } else {
                    return failure(
                        ReferenceExecutionErrorCode::KernelFailure,
                        std::string("cell kernel failed: ") +
                            sdsc_cell_kernel_status_string(status),
                        true, cell.cell, false, EdgeId{0}, status);
                }
            }
            if (!detail::valid_runtime_cell(cell)) {
                return failure(
                    ReferenceExecutionErrorCode::KernelFailure,
                    "cell kernel produced state outside the finite runtime domain",
                    true, cell.cell);
            }
            if constexpr (!Strict) cell.initialized = true;
            if (std::abs(cell.output_val) > 1e-6) ++cell.activation_count;
            measurement.cells.push_back(
                ExecutedCellMeasurement{cell.cell, cell.type, true, cell.output_val});
        }

        for (std::size_t index = 0; index < working.size(); ++index) {
            auto& cell = working[index];
            if constexpr (Strict) {
                if (!std::isfinite(cell.output_val)) {
                    return failure(
                        ReferenceExecutionErrorCode::KernelFailure,
                        "strict graph produced a non-finite output");
                }
            }
            if (!control.dormant(index)) cell.prev_output_val = cell.output_val;
        }
        runtime.cells_ = std::move(working);
        ++runtime.tick_;
        return {std::move(measurement), std::nullopt};
    }

    static ReferenceExecutionResult failure(
        ReferenceExecutionErrorCode code,
        std::string reason,
        bool has_cell = false,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0},
        SdscCellKernelStatus kernel_status = SDSC_CELL_KERNEL_OK) {
        return {
            {},
            ReferenceExecutionError{
                code, std::move(reason), has_cell, cell, has_edge, edge,
                kernel_status}};
    }
};

}  // namespace kun::core
