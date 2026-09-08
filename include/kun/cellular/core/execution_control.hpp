#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace kun::core {

enum class ExecutionDisposition : uint8_t {
    Active = 0,
    Dormant = 1,
};

struct ExecutionControlView {
    std::span<const ExecutionDisposition> dispositions{};

    bool empty() const { return dispositions.empty(); }
    bool valid_for(std::size_t cell_count) const {
        return dispositions.empty() || dispositions.size() == cell_count;
    }
    bool dormant(std::size_t dense_index) const {
        return !dispositions.empty() &&
               dispositions[dense_index] == ExecutionDisposition::Dormant;
    }
};

}  // namespace kun::core
