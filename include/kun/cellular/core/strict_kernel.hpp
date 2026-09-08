#pragma once

#include "kun/cellular/core/primitive_contract.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <span>

namespace kun::core {

inline bool strict_parameter_to_c(
    const ParameterValue& value, SdscCellKernelStrictParameter* result) {
    if (result == nullptr) return false;
    if (const auto* continuous = std::get_if<ContinuousValue>(&value)) {
        result->kind = SDSC_CELL_PARAMETER_CONTINUOUS;
        result->value.continuous = continuous->value;
        return true;
    }
    if (const auto* channel = std::get_if<ChannelIndex>(&value)) {
        result->kind = SDSC_CELL_PARAMETER_CHANNEL_INDEX;
        result->value.channel_index = channel->value;
        return true;
    }
    if (const auto* delay = std::get_if<DelayTicks>(&value)) {
        result->kind = SDSC_CELL_PARAMETER_DELAY_TICKS;
        result->value.delay_ticks = delay->value;
        return true;
    }
    if (const auto* mode = std::get_if<MinMaxMode>(&value)) {
        result->kind = SDSC_CELL_PARAMETER_MIN_MAX_MODE;
        result->value.min_max_mode = static_cast<uint8_t>(*mode);
        return true;
    }
    result->kind = SDSC_CELL_PARAMETER_UNUSED;
    return std::holds_alternative<UnusedParameter>(value);
}

inline SdscCellKernelStatus strict_cell_kernel_step(
    CellType type,
    const ParameterValue& param1,
    const ParameterValue& param2,
    double in0,
    double in1,
    std::span<const double> inputs,
    SdscCellKernelStrictStateView* state) {
    SdscCellKernelStrictParameters parameters{};
    if (!strict_parameter_to_c(param1, &parameters.param1) ||
        !strict_parameter_to_c(param2, &parameters.param2)) {
        return SDSC_CELL_KERNEL_INVALID_PARAMETER_KIND;
    }
    return sdsc_cell_kernel_step_strict(
        static_cast<uint8_t>(type), &parameters, in0, in1, inputs.size(),
        inputs.data(), state);
}

}  // namespace kun::core
