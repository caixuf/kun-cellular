#pragma once

#include "kun/cellular/core/compiled_graph.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <string>
#include <utility>

namespace kun::core {

class GraphCompiler final {
public:
    CompileResult compile(
        const GraphDefinition& definition,
        const InitialParameterSeeds& seeds) const {
        if (const auto profile_error = validate_profile(definition)) {
            return failure(*profile_error);
        }

        std::vector<CellDefinition> cells = definition.cells;
        std::sort(cells.begin(), cells.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.id < rhs.id;
        });
        if (const auto duplicate = duplicate_cell(cells)) {
            return failure(*duplicate);
        }

        for (const auto& cell : cells) {
            if (!contract_for(cell.type).has_value()) {
                return failure(make_error(
                    CompileErrorCode::UnknownCellType, CompileObjectKind::Cell,
                    cell.id.value, "cell type is not present in generated metadata"));
            }
        }

        std::vector<EdgeDefinition> edges = definition.edges;
        std::sort(edges.begin(), edges.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.id < rhs.id;
        });
        if (const auto duplicate = duplicate_edge(edges)) {
            return failure(*duplicate);
        }

        std::vector<InitialParameterValue> initial_values;
        initial_values.reserve(cells.size() * 2 + edges.size());
        std::vector<std::array<std::size_t, 2>> cell_parameter_indices;
        cell_parameter_indices.resize(cells.size());

        if (const auto parameter_error = bind_cell_parameters(
                cells, seeds.cell_parameters, initial_values, cell_parameter_indices)) {
            return failure(*parameter_error);
        }

        std::vector<std::size_t> source_indices(edges.size());
        std::vector<std::size_t> target_indices(edges.size());
        std::vector<std::size_t> edge_parameter_indices(edges.size());
        std::vector<std::size_t> indegrees(cells.size(), 0);
        std::vector<std::vector<std::size_t>> outgoing(cells.size());

        for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
            const auto& edge = edges[edge_index];
            const auto source = find_cell(cells, edge.source);
            const auto target = find_cell(cells, edge.target);
            if (!source.has_value() || !target.has_value()) {
                return failure(make_error(
                    CompileErrorCode::DanglingEndpoint, CompileObjectKind::Edge,
                    edge.id.value, "edge endpoint does not name a declared cell"));
            }
            if (edge.source_port.value != 0) {
                return failure(make_error(
                    CompileErrorCode::UnsupportedSourcePort, CompileObjectKind::Edge,
                    edge.id.value, "source port must be output port 0"));
            }
            const auto& target_contract = contract_for(cells[*target].type)->get();
            if (edge.target_port.value >= target_contract.input_port_count) {
                return failure(make_error(
                    CompileErrorCode::UnsupportedTargetPort, CompileObjectKind::Edge,
                    edge.id.value, "target port is not consumed by the cell type"));
            }
            if (edge.delay != EdgeDelay::Immediate &&
                edge.delay != EdgeDelay::PreviousTick) {
                return failure(make_error(
                    CompileErrorCode::UnknownDelay, CompileObjectKind::Edge,
                    edge.id.value, "edge delay enum is not recognized"));
            }
            source_indices[edge_index] = *source;
            target_indices[edge_index] = *target;
            if (edge.delay == EdgeDelay::Immediate) {
                ++indegrees[*target];
                outgoing[*source].push_back(edge_index);
            }
        }

        if (const auto weight_error = bind_edge_weights(
                edges, seeds.edge_weights, initial_values, edge_parameter_indices)) {
            return failure(*weight_error);
        }

        std::vector<std::size_t> execution_order;
        execution_order.reserve(cells.size());
        std::vector<bool> emitted(cells.size(), false);
        auto ready = [&cells](std::size_t lhs, std::size_t rhs) {
            return cells[rhs].id < cells[lhs].id;
        };
        std::priority_queue<std::size_t, std::vector<std::size_t>, decltype(ready)> queue(ready);
        for (std::size_t index = 0; index < indegrees.size(); ++index) {
            if (indegrees[index] == 0) queue.push(index);
        }
        for (auto& outgoing_edges : outgoing) {
            std::sort(outgoing_edges.begin(), outgoing_edges.end(),
                      [&edges](std::size_t lhs, std::size_t rhs) {
                          return edges[lhs].id < edges[rhs].id;
                      });
        }
        while (!queue.empty()) {
            const std::size_t current = queue.top();
            queue.pop();
            execution_order.push_back(current);
            emitted[current] = true;
            for (const std::size_t edge_index : outgoing[current]) {
                const std::size_t target = target_indices[edge_index];
                if (--indegrees[target] == 0) queue.push(target);
            }
        }
        if (execution_order.size() != cells.size()) {
            const auto cycle_edge = find_cycle_edge(outgoing, target_indices, emitted, cells);
            if (cycle_edge.has_value()) {
                const auto& edge = edges[*cycle_edge];
                return failure(make_error(
                    CompileErrorCode::ImmediateCycle, CompileObjectKind::Edge,
                    edge.id.value,
                    "immediate edges contain a cycle involving cells " +
                        std::to_string(edge.source.value) + " and " +
                        std::to_string(edge.target.value)));
            }
            return failure(make_error(
                CompileErrorCode::ImmediateCycle, CompileObjectKind::Graph,
                definition.identity.value,
                "an immediate edge is blocked by a cycle"));
        }

        std::vector<std::size_t> canonical_edge_order(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) canonical_edge_order[i] = i;
        std::sort(canonical_edge_order.begin(), canonical_edge_order.end(),
                  [&edges, &target_indices](std::size_t lhs, std::size_t rhs) {
                      if (target_indices[lhs] != target_indices[rhs]) {
                          return target_indices[lhs] < target_indices[rhs];
                      }
                      if (edges[lhs].target_port != edges[rhs].target_port) {
                          return edges[lhs].target_port < edges[rhs].target_port;
                      }
                      return edges[lhs].id < edges[rhs].id;
                  });

        std::vector<CompiledEdge> compiled_edges;
        compiled_edges.reserve(edges.size());
        for (const std::size_t edge_index : canonical_edge_order) {
            const auto& edge = edges[edge_index];
            compiled_edges.push_back(CompiledEdge{
                edge.id,
                source_indices[edge_index],
                edge.source_port,
                target_indices[edge_index],
                edge.target_port,
                edge.delay,
                edge_parameter_indices[edge_index],
            });
        }

        std::vector<PortReduction> reductions;
        std::size_t reduction_edge_cursor = 0;
        for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
            const auto& contract = contract_for(cells[cell_index].type)->get();
            for (uint32_t port = 0; port < contract.input_port_count; ++port) {
                const std::size_t begin = reduction_edge_cursor;
                while (reduction_edge_cursor < compiled_edges.size() &&
                       compiled_edges[reduction_edge_cursor].target_index == cell_index &&
                       compiled_edges[reduction_edge_cursor].target_port.value == port) {
                    ++reduction_edge_cursor;
                }
                reductions.push_back(PortReduction{
                    cell_index,
                    InputPort{port},
                    begin,
                    reduction_edge_cursor,
                });
            }
        }

        std::vector<CompiledCell> compiled_cells;
        compiled_cells.reserve(cells.size());
        for (std::size_t index = 0; index < cells.size(); ++index) {
            const auto& contract = contract_for(cells[index].type)->get();
            compiled_cells.push_back(CompiledCell{
                cells[index].id,
                cells[index].type,
                cell_parameter_indices[index],
                contract.input_port_count,
            });
        }

        auto graph = std::shared_ptr<const CompiledGraph>(new CompiledGraph(
            definition.identity,
            definition.revision,
            definition.profile,
            definition.semantic_version,
            std::move(compiled_cells),
            std::move(compiled_edges),
            std::move(execution_order),
            std::move(reductions),
            initial_values.size()));
        return CompileResult{
            std::move(graph),
            std::make_shared<const InitialParameterValues>(std::move(initial_values)),
            std::nullopt,
        };
    }

private:
    static CompileResult failure(CompileError error) {
        return CompileResult{nullptr, nullptr, std::move(error)};
    }

    static std::optional<CompileError> validate_profile(const GraphDefinition& definition) {
        if (!semantic_profile_from_code(static_cast<uint8_t>(definition.profile))) {
            return make_error(
                CompileErrorCode::UnknownProfile, CompileObjectKind::Graph,
                definition.identity.value,
                "semantic profile code is not recognized");
        }
        if (definition.semantic_version != 1) {
            return make_error(
                CompileErrorCode::UnsupportedSemanticVersion, CompileObjectKind::Graph,
                definition.identity.value,
                "semantic version is not present in generated metadata");
        }
        return std::nullopt;
    }

    static std::optional<std::size_t> find_cycle_edge(
        const std::vector<std::vector<std::size_t>>& outgoing,
        const std::vector<std::size_t>& target_indices,
        const std::vector<bool>& emitted,
        const std::vector<CellDefinition>& cells) {
        struct Frame {
            std::size_t node;
            std::size_t next_edge;
        };
        std::vector<uint8_t> color(cells.size(), 0);
        for (std::size_t start = 0; start < cells.size(); ++start) {
            if (emitted[start] || color[start] != 0) continue;
            std::vector<Frame> stack{{start, 0}};
            color[start] = 1;
            while (!stack.empty()) {
                auto& frame = stack.back();
                if (frame.next_edge == outgoing[frame.node].size()) {
                    color[frame.node] = 2;
                    stack.pop_back();
                    continue;
                }
                const std::size_t edge_index =
                    outgoing[frame.node][frame.next_edge++];
                const std::size_t target = target_indices[edge_index];
                if (emitted[target]) continue;
                if (color[target] == 1) return edge_index;
                if (color[target] == 0) {
                    color[target] = 1;
                    stack.push_back(Frame{target, 0});
                }
            }
        }
        return std::nullopt;
    }

    static std::optional<CompileError> duplicate_cell(
        const std::vector<CellDefinition>& cells) {
        for (std::size_t i = 1; i < cells.size(); ++i) {
            if (cells[i - 1].id == cells[i].id) {
                return make_error(
                    CompileErrorCode::DuplicateCellId, CompileObjectKind::Cell,
                    cells[i].id.value, "duplicate cell ID");
            }
        }
        return std::nullopt;
    }

    static std::optional<CompileError> duplicate_edge(
        const std::vector<EdgeDefinition>& edges) {
        for (std::size_t i = 1; i < edges.size(); ++i) {
            if (edges[i - 1].id == edges[i].id) {
                return make_error(
                    CompileErrorCode::DuplicateEdgeId, CompileObjectKind::Edge,
                    edges[i].id.value, "duplicate edge ID");
            }
        }
        return std::nullopt;
    }

    static std::optional<std::size_t> find_cell(
        const std::vector<CellDefinition>& cells, CellId id) {
        const auto it = std::lower_bound(
            cells.begin(), cells.end(), id,
            [](const CellDefinition& cell, CellId sought) { return cell.id < sought; });
        if (it == cells.end() || it->id != id) return std::nullopt;
        return static_cast<std::size_t>(it - cells.begin());
    }

    static std::optional<CompileError> bind_cell_parameters(
        const std::vector<CellDefinition>& cells,
        const std::vector<CellParameterSeed>& seeds,
        std::vector<InitialParameterValue>& values,
        std::vector<std::array<std::size_t, 2>>& indices) {
        std::vector<CellParameterSeed> sorted = seeds;
        std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.cell != rhs.cell) return lhs.cell < rhs.cell;
            return lhs.slot < rhs.slot;
        });
        std::size_t cursor = 0;
        for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
            const auto& cell = cells[cell_index];
            const auto contract = contract_for(cell.type);
            while (cursor < sorted.size() && sorted[cursor].cell < cell.id) {
                return make_error(
                    CompileErrorCode::ExtraParameter, CompileObjectKind::CellParameter,
                    sorted[cursor].cell.value, "parameter seed names an undeclared cell");
            }
            std::array<bool, 2> found{false, false};
            std::array<const CellParameterSeed*, 2> entries{nullptr, nullptr};
            while (cursor < sorted.size() && sorted[cursor].cell == cell.id) {
                const auto raw_slot = static_cast<uint8_t>(sorted[cursor].slot);
                if (raw_slot >= 2) {
                    return make_error(
                        CompileErrorCode::InvalidParameter, CompileObjectKind::CellParameter,
                        cell.id.value, "parameter slot enum is not recognized");
                }
                const std::size_t slot_index = raw_slot;
                if (found[slot_index]) {
                    return make_error(
                        CompileErrorCode::DuplicateParameter, CompileObjectKind::CellParameter,
                        cell.id.value, "duplicate cell parameter slot");
                }
                found[slot_index] = true;
                entries[slot_index] = &sorted[cursor];
                ++cursor;
            }
            for (std::size_t slot_index = 0; slot_index < 2; ++slot_index) {
                const auto slot = static_cast<ParameterSlot>(slot_index);
                if (!found[slot_index]) {
                    return make_error(
                        CompileErrorCode::MissingParameter, CompileObjectKind::CellParameter,
                        cell.id.value, "cell parameter seed is missing");
                }
                if (!validate_parameter(
                        contract->get().parameters[slot_index], entries[slot_index]->value)) {
                    return make_error(
                        CompileErrorCode::InvalidParameter, CompileObjectKind::CellParameter,
                        cell.id.value, "parameter value does not match generated cell metadata");
                }
                const std::size_t binding_index = values.size();
                indices[cell_index][slot_index] = binding_index;
                values.push_back(InitialParameterValue{
                    ParameterBinding{
                        ParameterBindingKind::CellParameter,
                        binding_index,
                        cell.id,
                        EdgeId{},
                        slot,
                    },
                    entries[slot_index]->value,
                });
            }
        }
        if (cursor != sorted.size()) {
            return make_error(
                CompileErrorCode::ExtraParameter, CompileObjectKind::CellParameter,
                sorted[cursor].cell.value, "parameter seed names an undeclared cell");
        }
        return std::nullopt;
    }

    static std::optional<CompileError> bind_edge_weights(
        const std::vector<EdgeDefinition>& edges,
        const std::vector<EdgeParameterSeed>& seeds,
        std::vector<InitialParameterValue>& values,
        std::vector<std::size_t>& indices) {
        std::vector<EdgeParameterSeed> sorted = seeds;
        std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.edge < rhs.edge;
        });
        std::size_t cursor = 0;
        for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
            const auto& edge = edges[edge_index];
            while (cursor < sorted.size() && sorted[cursor].edge < edge.id) {
                return make_error(
                    CompileErrorCode::ExtraEdgeWeight, CompileObjectKind::EdgeParameter,
                    sorted[cursor].edge.value, "edge weight seed names an undeclared edge");
            }
            if (cursor >= sorted.size() || sorted[cursor].edge != edge.id) {
                return make_error(
                    CompileErrorCode::MissingEdgeWeight, CompileObjectKind::EdgeParameter,
                    edge.id.value, "edge initial weight seed is missing");
            }
            if (cursor + 1 < sorted.size() && sorted[cursor + 1].edge == edge.id) {
                return make_error(
                    CompileErrorCode::DuplicateEdgeWeight, CompileObjectKind::EdgeParameter,
                    edge.id.value, "duplicate edge initial weight seed");
            }
            if (!std::isfinite(sorted[cursor].initial_weight)) {
                return make_error(
                    CompileErrorCode::InvalidEdgeWeight, CompileObjectKind::EdgeParameter,
                    edge.id.value, "edge initial weight must be finite");
            }
            const std::size_t binding_index = values.size();
            indices[edge_index] = binding_index;
            values.push_back(InitialParameterValue{
                ParameterBinding{
                    ParameterBindingKind::EdgeWeight,
                    binding_index,
                    CellId{},
                    edge.id,
                    ParameterSlot::Param1,
                },
                ContinuousValue{sorted[cursor].initial_weight},
            });
            ++cursor;
        }
        if (cursor != sorted.size()) {
            return make_error(
                CompileErrorCode::ExtraEdgeWeight, CompileObjectKind::EdgeParameter,
                sorted[cursor].edge.value, "edge weight seed names an undeclared edge");
        }
        return std::nullopt;
    }

    static CompileError make_error(
        CompileErrorCode code,
        CompileObjectKind object_kind,
        uint64_t object_id,
        std::string reason) {
        return CompileError{code, object_kind, true, object_id, std::move(reason)};
    }
};

} // namespace kun::core
