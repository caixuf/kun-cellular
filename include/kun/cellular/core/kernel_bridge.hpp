#pragma once

#include "kun/cellular/core/primitive_contract.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace kun::core {

inline std::optional<double> legacy_kernel_parameter(const ParameterValue& value) {
    if (const auto* continuous = std::get_if<ContinuousValue>(&value)) {
        return std::isfinite(continuous->value)
                   ? std::optional<double>{continuous->value}
                   : std::nullopt;
    }
    if (const auto* channel = std::get_if<ChannelIndex>(&value)) {
        const double encoded = static_cast<double>(channel->value);
        const double exclusive_upper_bound =
            std::ldexp(1.0, std::numeric_limits<std::size_t>::digits);
        if (!std::isfinite(encoded) || encoded >= exclusive_upper_bound ||
            std::floor(encoded) != encoded ||
            static_cast<std::size_t>(encoded) != channel->value) {
            return std::nullopt;
        }
        return encoded;
    }
    if (const auto* delay = std::get_if<DelayTicks>(&value)) {
        if (delay->value == 0 || delay->value > 16) return std::nullopt;
        return static_cast<double>(delay->value) / 16.0;
    }
    if (const auto* mode = std::get_if<MinMaxMode>(&value)) {
        if (*mode != MinMaxMode::Min && *mode != MinMaxMode::Max) {
            return std::nullopt;
        }
        return *mode == MinMaxMode::Max ? 1.0 : 0.0;
    }
    if (std::holds_alternative<UnusedParameter>(value)) return 0.0;
    return std::nullopt;
}

inline bool valid_runtime_state(const double value) {
    return std::isfinite(value);
}

}  // namespace kun::core
