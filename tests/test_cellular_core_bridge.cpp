#include "kun/cellular/core/kernel_bridge.hpp"

#include <cassert>
#include <cstdint>
#include <limits>

using namespace kun::core;

int main() {
    const auto native_max = std::numeric_limits<std::size_t>::max();
    if constexpr (std::numeric_limits<std::size_t>::digits <=
                  std::numeric_limits<double>::digits) {
        assert(legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{native_max}})
                   .has_value());
    } else {
        assert(!legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{native_max}})
                    .has_value());
    }
    if constexpr (std::numeric_limits<std::size_t>::digits > 53) {
        const auto exact = static_cast<std::size_t>(UINT64_C(1) << 53);
        assert(legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{exact}})
                   .has_value());
        assert(legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{exact + 2}})
                   .has_value());
        assert(!legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{exact + 1}})
                    .has_value());
    }
    if constexpr (std::numeric_limits<std::size_t>::digits > 60) {
        const auto exact_60 = static_cast<std::size_t>(UINT64_C(1) << 60);
        assert(legacy_kernel_parameter(
                   ParameterValue{ChannelIndex{exact_60}})
                   .has_value());
    }
    assert(!legacy_kernel_parameter(
        ParameterValue{static_cast<MinMaxMode>(255)}));
    return 0;
}
