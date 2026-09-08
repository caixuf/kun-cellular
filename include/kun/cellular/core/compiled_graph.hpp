#pragma once

#include "kun/cellular/core/graph_ir.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kun::core {

enum class ParameterBindingKind : uint8_t {
    CellParameter,
    EdgeWeight,
};

struct ParameterBinding {
    ParameterBindingKind kind{ParameterBindingKind::CellParameter};
    std::size_t index{0};
    CellId cell{};
    EdgeId edge{};
    ParameterSlot slot{ParameterSlot::Param1};
};

struct InitialParameterValue {
    ParameterBinding binding{};
    ParameterValue value{UnusedParameter{}};
};

// Immutable birth values are deliberately separate from CompiledGraph. They
// are copied from the caller so future runtime state cannot alias graph input.
class InitialParameterValues final {
public:
    explicit InitialParameterValues(std::vector<InitialParameterValue> values)
        : values_(std::move(values)) {}

    std::span<const InitialParameterValue> entries() const { return values_; }
    const InitialParameterValue& at(std::size_t index) const { return values_.at(index); }
    std::size_t size() const { return values_.size(); }

private:
    std::vector<InitialParameterValue> values_;
};

struct CompiledCell {
    CellId id{};
    CellType type{CellType::OP_EMA};
    std::array<std::size_t, 2> parameter_indices{};
    uint8_t input_port_count{0};
};

struct CompiledEdge {
    EdgeId id{};
    std::size_t source_index{0};
    OutputPort source_port{};
    std::size_t target_index{0};
    InputPort target_port{};
    EdgeDelay delay{EdgeDelay::Immediate};
    std::size_t weight_parameter_index{0};
};

struct PortReduction {
    std::size_t target_index{0};
    InputPort target_port{};
    std::size_t edge_begin{0};
    std::size_t edge_end{0};
};

class CompiledGraph final {
public:
    GraphIdentity identity() const { return identity_; }
    GraphRevision revision() const { return revision_; }
    SemanticProfile profile() const { return profile_; }
    uint32_t semantic_version() const { return semantic_version_; }

    std::span<const CompiledCell> cells() const { return cells_; }
    std::span<const CompiledEdge> edges() const { return edges_; }
    std::span<const std::size_t> execution_order() const { return execution_order_; }
    std::span<const PortReduction> port_reductions() const { return port_reductions_; }
    std::size_t parameter_binding_count() const { return parameter_binding_count_; }

private:
    friend class GraphCompiler;

    CompiledGraph(
        GraphIdentity identity,
        GraphRevision revision,
        SemanticProfile profile,
        uint32_t semantic_version,
        std::vector<CompiledCell> cells,
        std::vector<CompiledEdge> edges,
        std::vector<std::size_t> execution_order,
        std::vector<PortReduction> port_reductions,
        std::size_t parameter_binding_count)
        : identity_(identity),
          revision_(revision),
          profile_(profile),
          semantic_version_(semantic_version),
          cells_(std::move(cells)),
          edges_(std::move(edges)),
          execution_order_(std::move(execution_order)),
          port_reductions_(std::move(port_reductions)),
          parameter_binding_count_(parameter_binding_count) {}

    GraphIdentity identity_{};
    GraphRevision revision_{};
    SemanticProfile profile_{SemanticProfile::LegacyCompatible};
    uint32_t semantic_version_{0};
    std::vector<CompiledCell> cells_;
    std::vector<CompiledEdge> edges_;
    std::vector<std::size_t> execution_order_;
    std::vector<PortReduction> port_reductions_;
    std::size_t parameter_binding_count_{0};
};

enum class CompileObjectKind : uint8_t {
    Graph,
    Cell,
    Edge,
    CellParameter,
    EdgeParameter,
};

enum class CompileErrorCode : uint8_t {
    InvalidGraph,
    UnknownProfile,
    UnsupportedSemanticVersion,
    DuplicateCellId,
    DuplicateEdgeId,
    UnknownCellType,
    DanglingEndpoint,
    UnsupportedSourcePort,
    UnsupportedTargetPort,
    UnknownDelay,
    MissingParameter,
    DuplicateParameter,
    ExtraParameter,
    InvalidParameter,
    MissingEdgeWeight,
    DuplicateEdgeWeight,
    ExtraEdgeWeight,
    InvalidEdgeWeight,
    ImmediateCycle,
};

inline constexpr std::string_view compile_object_kind_name(CompileObjectKind kind) {
    switch (kind) {
        case CompileObjectKind::Graph: return "graph";
        case CompileObjectKind::Cell: return "cell";
        case CompileObjectKind::Edge: return "edge";
        case CompileObjectKind::CellParameter: return "cell parameter";
        case CompileObjectKind::EdgeParameter: return "edge parameter";
    }
    return "unknown object";
}

struct CompileError {
    CompileErrorCode code{CompileErrorCode::InvalidGraph};
    CompileObjectKind object_kind{CompileObjectKind::Graph};
    bool has_id{false};
    uint64_t object_id{0};
    std::string reason;

    std::string diagnostic() const {
        std::string result{compile_object_kind_name(object_kind)};
        if (has_id) result += " id=" + std::to_string(object_id);
        result += ": " + reason;
        return result;
    }
};

struct CompileResult {
    std::shared_ptr<const CompiledGraph> graph;
    std::shared_ptr<const InitialParameterValues> initial_values;
    std::optional<CompileError> error;

    bool ok() const { return graph != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

} // namespace kun::core
