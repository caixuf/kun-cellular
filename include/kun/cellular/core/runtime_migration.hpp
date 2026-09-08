#pragma once

#include "kun/cellular/core/runtime_state.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace kun::core {

enum class MigrationResetReason : uint8_t {
    TypeChanged,
    DiscreteConfigurationChanged,
    EdgeDefinitionChanged,
};

struct CellMigrationReset {
    CellId cell{};
    MigrationResetReason reason{MigrationResetReason::TypeChanged};
};

struct EdgeMigrationReset {
    EdgeId edge{};
    MigrationResetReason reason{MigrationResetReason::EdgeDefinitionChanged};
};

struct RuntimeMigrationReport {
    std::vector<CellId> preserved_cells;
    std::vector<CellId> new_cells;
    std::vector<CellId> removed_cells;
    std::vector<CellId> reset_cells;
    std::vector<CellMigrationReset> cell_reset_details;

    std::vector<EdgeId> preserved_edges;
    std::vector<EdgeId> new_edges;
    std::vector<EdgeId> removed_edges;
    std::vector<EdgeId> reset_edges;
    std::vector<EdgeMigrationReset> edge_reset_details;
};

struct RuntimeMigrationResult {
    RuntimeMigrationReport report;
    std::optional<RuntimeError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class RuntimeMigration final {
public:
    // Rebinds one live RuntimeState to a compiler-produced immutable graph.
    // Validation and the complete candidate state are built before any field
    // in runtime is published. Birth seeds are used only for new/reset IDs.
    static RuntimeMigrationResult rebind(
        RuntimeState& runtime,
        std::shared_ptr<const CompiledGraph> graph,
        std::shared_ptr<const InitialParameterValues> seeds) {
        RuntimeMigrationResult result;
        if (!graph || !seeds) {
            return failure(
                RuntimeErrorCode::MigrationInvalidInput,
                "migration requires an owned compiled graph and typed seed store");
        }
        if (!semantic_profile_from_code(static_cast<uint8_t>(graph->profile()))) {
            return failure(
                RuntimeErrorCode::UnsupportedProfile,
                "migration graph uses an unknown semantic profile");
        }
        if (graph->profile() != runtime.profile()) {
            return failure(
                RuntimeErrorCode::UnsupportedProfile,
                "runtime migration cannot cross semantic profiles");
        }
        if (graph->identity() != runtime.identity()) {
            return failure(
                RuntimeErrorCode::MigrationIdentityMismatch,
                "a graph with a different identity is not a runtime rebind");
        }
        if (graph->revision() < runtime.revision()) {
            return failure(
                RuntimeErrorCode::MigrationRevisionMismatch,
                "an older graph revision cannot overwrite the live runtime");
        }
        if (graph->revision() == runtime.revision() &&
            !detail::same_plan(*runtime.plan(), *graph)) {
            return failure(
                RuntimeErrorCode::MigrationRevisionMismatch,
                "same-revision graph is not an identical actual plan");
        }
        if (const auto error = detail::validate_parameter_bindings(
                *graph, seeds->entries())) {
            return {RuntimeMigrationReport{}, error};
        }
        if (graph->revision() == runtime.revision()) {
            for (const auto& cell : graph->cells()) {
                const auto old_index = find_cell(*runtime.plan(), cell.id);
                if (!old_index.has_value() ||
                    !same_discrete_parameters(runtime, *old_index, *seeds, cell)) {
                    return failure(
                        RuntimeErrorCode::MigrationRevisionMismatch,
                        "same-revision graph has incompatible discrete configuration");
                }
            }
        }

        RuntimeState candidate(graph, seeds);
        candidate.tick_ = runtime.tick_;
        candidate.cells_.clear();
        candidate.cells_.reserve(graph->cells().size());
        candidate.parameters_.assign(
            seeds->entries().begin(), seeds->entries().end());

        const auto old_plan = runtime.plan_;
        const bool no_op = detail::same_plan(*old_plan, *graph);

        for (const auto& new_cell : graph->cells()) {
            const auto old_index = find_cell(*old_plan, new_cell.id);
            if (!old_index.has_value()) {
                candidate.cells_.push_back(RuntimeCellState{new_cell.id, new_cell.type});
                result.report.new_cells.push_back(new_cell.id);
                continue;
            }

            const auto& old_cell = old_plan->cells()[*old_index];
            const bool same_type = old_cell.type == new_cell.type;
            const bool same_discrete = no_op || (same_type &&
                same_discrete_parameters(runtime, *old_index, *seeds, new_cell));
            if (!same_type || !same_discrete) {
                candidate.cells_.push_back(RuntimeCellState{new_cell.id, new_cell.type});
                RuntimeCellState& reset = candidate.cells_.back();
                RuntimeState::reset_cell(reset, new_cell.type);
                result.report.reset_cells.push_back(new_cell.id);
                const auto reason = !same_type
                    ? MigrationResetReason::TypeChanged
                    : MigrationResetReason::DiscreteConfigurationChanged;
                result.report.cell_reset_details.push_back({new_cell.id, reason});
            } else {
                candidate.cells_.push_back(runtime.cells_[*old_index]);
                candidate.cells_.back().cell = new_cell.id;
                candidate.cells_.back().type = new_cell.type;
                result.report.preserved_cells.push_back(new_cell.id);
            }

            if (same_type && same_discrete) {
                for (std::size_t slot = 0; slot < 2; ++slot) {
                    candidate.parameters_[new_cell.parameter_indices[slot]].value =
                        runtime.parameters_[old_cell.parameter_indices[slot]].value;
                }
            }
        }

        for (const auto& old_cell : old_plan->cells()) {
            if (!find_cell(*graph, old_cell.id).has_value()) {
                result.report.removed_cells.push_back(old_cell.id);
            }
        }

        const auto old_edge_order = edge_order(*old_plan);
        const auto new_edge_order = edge_order(*graph);
        for (const auto& new_edge : graph->edges()) {
            const auto old_index = find_edge(*old_plan, old_edge_order, new_edge.id);
            if (!old_index.has_value()) {
                result.report.new_edges.push_back(new_edge.id);
                continue;
            }
            const auto& old_edge = old_plan->edges()[*old_index];
            if (no_op || edge_definition_matches(*old_plan, old_edge, *graph, new_edge)) {
                candidate.parameters_[new_edge.weight_parameter_index].value =
                    runtime.parameters_[old_edge.weight_parameter_index].value;
                result.report.preserved_edges.push_back(new_edge.id);
            } else {
                result.report.reset_edges.push_back(new_edge.id);
                result.report.edge_reset_details.push_back(
                    {new_edge.id, MigrationResetReason::EdgeDefinitionChanged});
            }
        }
        for (const auto& old_edge : old_plan->edges()) {
            if (!find_edge(*graph, new_edge_order, old_edge.id).has_value()) {
                result.report.removed_edges.push_back(old_edge.id);
            }
        }
        sort_report(result.report);

        if (const auto error = candidate.validate_bindings()) {
            return {RuntimeMigrationReport{}, error};
        }
        for (const auto& cell : candidate.cells_) {
            if (!detail::valid_runtime_cell(cell)) {
                return {RuntimeMigrationReport{}, RuntimeError{
                    RuntimeErrorCode::InvalidState,
                    "migration would publish a non-finite or invalid computational state",
                    true, cell.cell, false, EdgeId{0}}};
            }
        }

        runtime.plan_ = std::move(candidate.plan_);
        runtime.parameters_ = std::move(candidate.parameters_);
        runtime.cells_ = std::move(candidate.cells_);
        runtime.tick_ = candidate.tick_;
        return result;
    }

private:
    static RuntimeMigrationResult failure(RuntimeErrorCode code, std::string reason) {
        return {RuntimeMigrationReport{},
                RuntimeError{code, std::move(reason), false, CellId{0},
                             false, EdgeId{0}}};
    }

    static void sort_report(RuntimeMigrationReport& report) {
        const auto sort_ids = [](auto& ids) {
            std::sort(ids.begin(), ids.end());
        };
        sort_ids(report.preserved_cells);
        sort_ids(report.new_cells);
        sort_ids(report.removed_cells);
        sort_ids(report.reset_cells);
        sort_ids(report.preserved_edges);
        sort_ids(report.new_edges);
        sort_ids(report.removed_edges);
        sort_ids(report.reset_edges);
        std::sort(
            report.cell_reset_details.begin(), report.cell_reset_details.end(),
            [](const CellMigrationReset& lhs, const CellMigrationReset& rhs) {
                return lhs.cell < rhs.cell;
            });
        std::sort(
            report.edge_reset_details.begin(), report.edge_reset_details.end(),
            [](const EdgeMigrationReset& lhs, const EdgeMigrationReset& rhs) {
                return lhs.edge < rhs.edge;
            });
    }

    static std::optional<std::size_t> find_cell(
        const CompiledGraph& graph,
        CellId id) {
        const auto cells = graph.cells();
        const auto it = std::lower_bound(
            cells.begin(), cells.end(), id,
            [](const CompiledCell& cell, CellId sought) {
                return cell.id < sought;
            });
        if (it == cells.end() || it->id != id) return std::nullopt;
        return static_cast<std::size_t>(it - cells.begin());
    }

    static std::vector<std::size_t> edge_order(const CompiledGraph& graph) {
        std::vector<std::size_t> order(graph.edges().size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&graph](std::size_t lhs, std::size_t rhs) {
            return graph.edges()[lhs].id < graph.edges()[rhs].id;
        });
        return order;
    }

    static std::optional<std::size_t> find_edge(
        const CompiledGraph& graph,
        const std::vector<std::size_t>& order,
        EdgeId id) {
        const auto it = std::lower_bound(
            order.begin(), order.end(), id,
            [&graph](std::size_t index, EdgeId sought) {
                return graph.edges()[index].id < sought;
            });
        if (it == order.end() || graph.edges()[*it].id != id) return std::nullopt;
        return *it;
    }

    static bool same_discrete_parameters(
        const RuntimeState& runtime,
        std::size_t old_cell_index,
        const InitialParameterValues& new_seeds,
        const CompiledCell& new_cell) {
        const auto& old_cell = runtime.plan_->cells()[old_cell_index];
        const auto old_contract = contract_for(old_cell.type);
        const auto new_contract = contract_for(new_cell.type);
        if (!old_contract.has_value() || !new_contract.has_value()) return false;
        for (std::size_t slot = 0; slot < 2; ++slot) {
            if (old_contract->get().parameters[slot].value_type ==
                ParameterValueType::Continuous) {
                continue;
            }
            const auto& old_value =
                runtime.parameters_[old_cell.parameter_indices[slot]].value;
            const auto& new_value =
                new_seeds.at(new_cell.parameter_indices[slot]).value;
            if (!parameter_equal(old_value, new_value)) return false;
        }
        return true;
    }

    static bool parameter_equal(
        const ParameterValue& lhs,
        const ParameterValue& rhs) {
        if (lhs.index() != rhs.index()) return false;
        return std::visit(
            [](const auto& left, const auto& right) {
                using Left = std::decay_t<decltype(left)>;
                using Right = std::decay_t<decltype(right)>;
                if constexpr (!std::is_same_v<Left, Right>) {
                    return false;
                } else if constexpr (std::is_same_v<Left, ContinuousValue>) {
                    return left.value == right.value;
                } else if constexpr (std::is_same_v<Left, ChannelIndex>) {
                    return left.value == right.value;
                } else if constexpr (std::is_same_v<Left, DelayTicks>) {
                    return left.value == right.value;
                } else if constexpr (std::is_same_v<Left, MinMaxMode>) {
                    return left == right;
                } else {
                    return true;
                }
            },
            lhs, rhs);
    }

    static bool edge_definition_matches(
        const CompiledGraph& old_graph,
        const CompiledEdge& old_edge,
        const CompiledGraph& new_graph,
        const CompiledEdge& new_edge) {
        if (old_edge.source_port != new_edge.source_port ||
            old_edge.target_port != new_edge.target_port ||
            old_edge.delay != new_edge.delay) {
            return false;
        }
        return old_graph.cells()[old_edge.source_index].id ==
                   new_graph.cells()[new_edge.source_index].id &&
               old_graph.cells()[old_edge.target_index].id ==
                   new_graph.cells()[new_edge.target_index].id;
    }
};

}  // namespace kun::core
