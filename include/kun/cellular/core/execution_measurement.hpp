#pragma once

#include "kun/cellular/core/graph_ir.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace kun::core {

struct ExecutedCellMeasurement {
    CellId cell{};
    CellType type{CellType::OP_EMA};
    bool executed{false};
    double output{0.0};
};

struct ReducedPortMeasurement {
    CellId cell{};
    InputPort port{};
    double reduced_input{0.0};
};

struct EdgeTransmissionMeasurement {
    EdgeId edge{};
    CellId source{};
    CellId target{};
    InputPort target_port{};
    EdgeDelay source_mode{EdgeDelay::Immediate};
    double source_value{0.0};
    double contribution{0.0};
};

struct ExecutionMeasurement {
    uint64_t tick{0};
    std::vector<ExecutedCellMeasurement> cells;
    std::vector<ReducedPortMeasurement> ports;
    std::vector<EdgeTransmissionMeasurement> edges;
    GraphIdentity identity{};
    GraphRevision revision{};
    SemanticProfile profile{SemanticProfile::LegacyCompatible};
};

struct ExecutionMeasurementView {
    uint64_t tick{0};
    std::span<const ExecutedCellMeasurement> cells;
    std::span<const ReducedPortMeasurement> ports;
    std::span<const EdgeTransmissionMeasurement> edges;
    GraphIdentity identity{};
    GraphRevision revision{};
    SemanticProfile profile{SemanticProfile::LegacyCompatible};
};

}  // namespace kun::core
