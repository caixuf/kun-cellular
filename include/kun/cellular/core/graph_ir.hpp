#pragma once

#include "kun/cellular/core/primitive_contract.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <vector>

namespace kun::core {

struct CellId {
    uint64_t value{0};
    explicit constexpr CellId(uint64_t v = 0) : value(v) {}
    friend constexpr bool operator==(CellId, CellId) = default;
    friend constexpr auto operator<=>(CellId, CellId) = default;
};

struct EdgeId {
    uint64_t value{0};
    explicit constexpr EdgeId(uint64_t v = 0) : value(v) {}
    friend constexpr bool operator==(EdgeId, EdgeId) = default;
    friend constexpr auto operator<=>(EdgeId, EdgeId) = default;
};

struct InputPort {
    uint32_t value{0};
    explicit constexpr InputPort(uint32_t v = 0) : value(v) {}
    friend constexpr bool operator==(InputPort, InputPort) = default;
    friend constexpr auto operator<=>(InputPort, InputPort) = default;
};

struct OutputPort {
    uint32_t value{0};
    explicit constexpr OutputPort(uint32_t v = 0) : value(v) {}
    friend constexpr bool operator==(OutputPort, OutputPort) = default;
    friend constexpr auto operator<=>(OutputPort, OutputPort) = default;
};

struct GraphIdentity {
    uint64_t value{0};
    explicit constexpr GraphIdentity(uint64_t v = 0) : value(v) {}
    friend constexpr bool operator==(GraphIdentity, GraphIdentity) = default;
    friend constexpr auto operator<=>(GraphIdentity, GraphIdentity) = default;
};

struct GraphRevision {
    uint64_t value{0};
    explicit constexpr GraphRevision(uint64_t v = 0) : value(v) {}
    friend constexpr bool operator==(GraphRevision, GraphRevision) = default;
    friend constexpr auto operator<=>(GraphRevision, GraphRevision) = default;
};

enum class EdgeDelay : uint8_t {
    Immediate = 0,
    PreviousTick = 1,
};

struct CellDefinition {
    CellId id{};
    CellType type{CellType::OP_EMA};
};

struct EdgeDefinition {
    EdgeId id{};
    CellId source{};
    OutputPort source_port{};
    CellId target{};
    InputPort target_port{};
    EdgeDelay delay{EdgeDelay::Immediate};
};

struct CellParameterSeed {
    CellId cell{};
    ParameterSlot slot{ParameterSlot::Param1};
    ParameterValue value{UnusedParameter{}};
};

struct EdgeParameterSeed {
    EdgeId edge{};
    double initial_weight{0.0};
};

struct InitialParameterSeeds {
    std::vector<CellParameterSeed> cell_parameters;
    std::vector<EdgeParameterSeed> edge_weights;
};

struct GraphDefinition {
    GraphIdentity identity{};
    GraphRevision revision{};
    SemanticProfile profile{SemanticProfile::LegacyCompatible};
    uint32_t semantic_version{1};
    std::vector<CellDefinition> cells;
    std::vector<EdgeDefinition> edges;
};

} // namespace kun::core
