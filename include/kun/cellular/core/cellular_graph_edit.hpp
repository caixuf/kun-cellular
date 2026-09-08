#pragma once

#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/runtime_migration.hpp"
#include "kun/cellular/core/runtime_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace kun::core {

struct CellBirth {
    CellId id{};
    CellType type{CellType::OP_EMA};
    std::array<ParameterValue, 2> parameters{};
};

struct EdgeBirth {
    EdgeId id{};
    CellId source{};
    OutputPort source_port{};
    CellId target{};
    InputPort target_port{};
    EdgeDelay delay{EdgeDelay::Immediate};
    double initial_weight{0.0};
};

struct RemoveCellAction { CellId id{}; };
struct RemoveEdgeAction { EdgeId id{}; };
struct AddCellAction { CellBirth birth{}; };
struct AddEdgeAction { EdgeBirth birth{}; };

struct SplitEdgeAction {
    EdgeId edge{};
    CellBirth inserted{};
    EdgeId source_to_new{};
    EdgeId new_to_target{};
    InputPort new_input_port{};
    // The original edge delay is assigned to source_to_new; new_to_target is
    // always Immediate, preserving the original one-tick causal boundary.
    double source_weight{0.0};
    double target_weight{0.0};
};

using GraphEditAction = std::variant<
    RemoveCellAction,
    RemoveEdgeAction,
    AddCellAction,
    AddEdgeAction,
    SplitEdgeAction>;

struct GraphEditEvent {
    uint64_t event_id{0};
    GraphEditAction action{};
};

enum class GraphEditActionKind : uint8_t {
    RemoveCell,
    RemoveEdge,
    AddCell,
    AddEdge,
    SplitEdge,
};

enum class GraphEditErrorCode : uint8_t {
    SourceMismatch,
    RevisionExhausted,
    DuplicateEventId,
    DuplicateId,
    RetiredId,
    MissingObject,
    ConflictingRequest,
    InvalidHistory,
    InvalidType,
    InvalidParameter,
    InvalidPort,
    NonFiniteSeed,
    InvalidSplit,
    CompilationFailed,
    MigrationFailed,
    InvalidRequest,
};

struct GraphEditError {
    GraphEditErrorCode code{GraphEditErrorCode::InvalidRequest};
    std::string reason;
    std::optional<uint64_t> event_id{std::nullopt};
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
};

struct GraphEditDiagnostic {
    uint64_t event_id{0};
    GraphEditActionKind action{GraphEditActionKind::AddCell};
    bool success{false};
    GraphRevision old_revision{};
    GraphRevision new_revision{};
    std::vector<CellId> cells;
    std::vector<EdgeId> edges;
    std::string reason;
};

struct GraphEditReport {
    GraphRevision old_revision{};
    GraphRevision new_revision{};
    bool no_op{false};
    bool committed{false};
    std::vector<GraphEditDiagnostic> diagnostics;
    RuntimeMigrationReport migration;
};

struct GraphEditHistory {
    std::vector<CellId> retired_cells;
    std::vector<EdgeId> retired_edges;
};

struct GraphEditResult {
    GraphDefinition definition{};
    std::shared_ptr<const CompiledGraph> graph;
    std::shared_ptr<const InitialParameterValues> seeds;
    std::optional<GraphEditError> error;
    GraphEditReport report;

    bool ok() const { return committed() && !error.has_value(); }
    explicit operator bool() const { return ok(); }
    bool committed() const { return report.committed || report.no_op; }
};

// Cold, explicit structural editing boundary. It owns only the retired-ID
// ledger and the source binding; RuntimeState remains the sole live parameter
// authority. The default ledger starts with no knowledge beyond the bound
// source; callers importing a graph with older retired-ID history must pass
// GraphEditHistory explicitly (and can persist history() in a later owner).
// A failed attempt records diagnostics in its returned report but reserves no
// birth and retires no object.
class GraphEditor final {
public:
    explicit GraphEditor(
        const RuntimeState& runtime,
        GraphEditHistory history = {})
        : identity_(runtime.identity()),
          expected_revision_(runtime.revision()),
          expected_plan_(runtime.plan()),
          retired_cells_(std::move(history.retired_cells)),
          retired_edges_(std::move(history.retired_edges)) {
        normalize(retired_cells_);
        normalize(retired_edges_);
        for (const auto id : retired_cells_) {
            if (runtime.cell_state(id) != nullptr) {
                history_error_ = GraphEditError{
                    GraphEditErrorCode::InvalidHistory,
                    "retired cell history conflicts with the live source plan",
                    std::nullopt, true, id, false, EdgeId{0}};
                break;
            }
        }
        if (!history_error_) {
            std::vector<EdgeId> live_edge_ids;
            live_edge_ids.reserve(runtime.plan()->edges().size());
            for (const auto& edge : runtime.plan()->edges()) {
                live_edge_ids.push_back(edge.id);
            }
            normalize(live_edge_ids);
            for (const auto id : retired_edges_) {
                if (std::binary_search(live_edge_ids.begin(), live_edge_ids.end(), id)) {
                    history_error_ = GraphEditError{
                        GraphEditErrorCode::InvalidHistory,
                        "retired edge history conflicts with the live source plan",
                        std::nullopt, false, CellId{0}, true, id};
                    break;
                }
            }
        }
    }

    GraphEditHistory history() const {
        return GraphEditHistory{retired_cells_, retired_edges_};
    }

    GraphEditResult apply(
        RuntimeState& runtime,
        std::span<const GraphEditEvent> events) {
        GraphEditResult result;
        result.report.old_revision = runtime.revision();
        result.report.new_revision = runtime.revision();
        const GraphRevision old_revision = runtime.revision();
        initialize_diagnostics(result.report, events, old_revision);

        if (runtime.identity() != identity_ ||
            runtime.revision() != expected_revision_ ||
            !expected_plan_ || !runtime.bound_to(*expected_plan_)) {
            return reject(
                result,
                GraphEditErrorCode::SourceMismatch,
                "editor source binding does not match the runtime identity or revision");
        }
        if (history_error_) {
            return reject(
                result,
                history_error_->code,
                history_error_->reason,
                history_error_->event_id,
                history_error_->has_cell,
                history_error_->cell,
                history_error_->has_edge,
                history_error_->edge);
        }

        const auto source = source_bundle(runtime);
        result.definition = source.definition;
        if (const auto error = validate_event_ids(events)) {
            return reject(result, error->code, error->reason, error->event_id);
        }
        if (!events.empty() &&
            runtime.revision().value == std::numeric_limits<uint64_t>::max()) {
            return reject(
                result,
                GraphEditErrorCode::RevisionExhausted,
                "graph revision cannot advance past uint64_t maximum");
        }

        if (events.empty()) {
            const auto source_compiled =
                GraphCompiler{}.compile(source.definition, source.seeds);
            if (!source_compiled.ok()) {
                return reject(
                    result,
                    GraphEditErrorCode::CompilationFailed,
                    source_compiled.error ? source_compiled.error->diagnostic()
                                          : "source graph could not be rebound");
            }
            result.graph = source_compiled.graph;
            result.seeds = source_compiled.initial_values;
            result.report.no_op = true;
            result.report.committed = true;
            return result;
        }

        GraphDefinition candidate = source.definition;
        InitialParameterSeeds candidate_seeds;
        std::vector<CellId> removed_cells;
        std::vector<EdgeId> removed_edges;
        std::vector<CellBirth> added_cells;
        std::vector<EdgeBirth> added_edges;
        std::vector<SplitEdgeAction> splits;

        if (const auto error = collect_and_validate(
                source.definition,
                events,
                removed_cells,
                removed_edges,
                added_cells,
                added_edges,
                splits)) {
            const auto event_id = responsible_event(events, *error);
            return reject(
                result, error->code, error->reason, event_id,
                error->has_cell, error->cell, error->has_edge, error->edge);
        }
        enrich_diagnostics(result.report, events, source.definition);
        const auto source_compiled =
            GraphCompiler{}.compile(source.definition, source.seeds);
        if (!source_compiled.ok()) {
            return reject(
                result,
                GraphEditErrorCode::CompilationFailed,
                source_compiled.error ? source_compiled.error->diagnostic()
                                      : "source graph could not be rebound");
        }
        result.seeds = source_compiled.initial_values;
        std::sort(removed_cells.begin(), removed_cells.end());
        std::sort(removed_edges.begin(), removed_edges.end());
        std::sort(added_cells.begin(), added_cells.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
        std::sort(added_edges.begin(), added_edges.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
        std::sort(splits.begin(), splits.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.edge < rhs.edge;
                  });

        std::vector<EdgeId> structurally_removed_edges = removed_edges;
        for (const auto& split : splits) {
            structurally_removed_edges.push_back(split.edge);
        }
        normalize(structurally_removed_edges);
        remove_objects(candidate, removed_cells, structurally_removed_edges);
        for (const auto& split : splits) {
            const auto old = find_edge(source.definition.edges, split.edge);
            if (!old.has_value()) {
                const GraphEditError missing{
                    GraphEditErrorCode::MissingObject,
                    "split edge disappeared during candidate construction",
                    std::nullopt, false, CellId{0}, true, split.edge};
                return reject(
                    result,
                    missing.code,
                    missing.reason,
                    responsible_event(events, missing),
                    false, CellId{0}, true, split.edge);
            }
            candidate.cells.push_back(
                CellDefinition{split.inserted.id, split.inserted.type});
            candidate.edges.push_back(EdgeDefinition{
                split.source_to_new,
                old->source,
                old->source_port,
                split.inserted.id,
                split.new_input_port,
                old->delay});
            candidate.edges.push_back(EdgeDefinition{
                split.new_to_target,
                split.inserted.id,
                OutputPort{0},
                old->target,
                old->target_port,
                EdgeDelay::Immediate});
        }
        for (const auto& cell : added_cells) {
            candidate.cells.push_back(CellDefinition{cell.id, cell.type});
        }
        for (const auto& edge : added_edges) {
            candidate.edges.push_back(EdgeDefinition{
                edge.id, edge.source, edge.source_port, edge.target,
                edge.target_port, edge.delay});
        }
        candidate.revision = GraphRevision{runtime.revision().value + 1};
        candidate_seeds = seeds_for_candidate(
            source.seeds,
            candidate,
            added_cells,
            added_edges,
            splits);

        const auto compiled = GraphCompiler{}.compile(candidate, candidate_seeds);
        if (!compiled.ok()) {
            const auto event_id = compiled.error
                ? compile_event_id(events, *compiled.error)
                : std::nullopt;
            return reject(
                result,
                GraphEditErrorCode::CompilationFailed,
                compiled.error ? compiled.error->diagnostic()
                               : "candidate graph compilation failed",
                event_id);
        }

        std::vector<EdgeId> removed_by_cells;
        for (const auto& edge : source.definition.edges) {
            if (contains(removed_cells, edge.source) ||
                contains(removed_cells, edge.target)) {
                removed_by_cells.push_back(edge.id);
            }
        }
        normalize(removed_by_cells);
        GraphEditHistory staged_history = history();
        staged_history.retired_cells.insert(
            staged_history.retired_cells.end(),
            removed_cells.begin(), removed_cells.end());
        staged_history.retired_edges.insert(
            staged_history.retired_edges.end(),
            removed_edges.begin(), removed_edges.end());
        staged_history.retired_edges.insert(
            staged_history.retired_edges.end(),
            removed_by_cells.begin(), removed_by_cells.end());
        for (const auto& split : splits) {
            staged_history.retired_edges.push_back(split.edge);
        }
        normalize(staged_history.retired_cells);
        normalize(staged_history.retired_edges);

        result.definition = std::move(candidate);
        result.graph = compiled.graph;
        result.seeds = compiled.initial_values;
        result.report.new_revision = compiled.graph->revision();
        mark_success_diagnostics(
            result.report, old_revision, result.report.new_revision);

        auto migration =
            RuntimeMigration::rebind(runtime, compiled.graph, compiled.initial_values);
        if (!migration.ok()) {
            const auto event_id = responsible_event(
                events,
                GraphEditError{
                    GraphEditErrorCode::MigrationFailed,
                    migration.error ? migration.error->diagnostic()
                                     : "candidate runtime migration failed"});
            return reject(
                result,
                GraphEditErrorCode::MigrationFailed,
                migration.error ? migration.error->diagnostic()
                                 : "candidate runtime migration failed",
                event_id);
        }

        result.report.committed = true;
        result.report.migration = std::move(migration.report);
        retired_cells_ = std::move(staged_history.retired_cells);
        retired_edges_ = std::move(staged_history.retired_edges);
        expected_revision_ = compiled.graph->revision();
        expected_plan_ = compiled.graph;
        return result;
    }

private:
    struct SourceBundle {
        GraphDefinition definition;
        InitialParameterSeeds seeds;
    };

    static GraphEditResult reject(
        GraphEditResult result,
        GraphEditErrorCode code,
        std::string reason,
        std::optional<uint64_t> event_id = std::nullopt,
        bool has_cell = false,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0}) {
        result.error = GraphEditError{
            code, std::move(reason), event_id, has_cell, cell, has_edge, edge};
        for (auto& diagnostic : result.report.diagnostics) {
            diagnostic.success = false;
            diagnostic.reason = result.error->reason;
        }
        return result;
    }

    static SourceBundle source_bundle(const RuntimeState& runtime) {
        SourceBundle source;
        source.definition.identity = runtime.identity();
        source.definition.revision = runtime.revision();
        source.definition.profile = runtime.profile();
        source.definition.semantic_version = runtime.plan()->semantic_version();
        for (const auto& cell : runtime.plan()->cells()) {
            source.definition.cells.push_back(CellDefinition{cell.id, cell.type});
            for (std::size_t slot = 0; slot < 2; ++slot) {
                source.seeds.cell_parameters.push_back(CellParameterSeed{
                    cell.id,
                    static_cast<ParameterSlot>(slot),
                    runtime.parameters()[cell.parameter_indices[slot]].value});
            }
        }
        for (const auto& edge : runtime.plan()->edges()) {
            const auto source_cell = runtime.plan()->cells()[edge.source_index].id;
            const auto target_cell = runtime.plan()->cells()[edge.target_index].id;
            source.definition.edges.push_back(EdgeDefinition{
                edge.id, source_cell, edge.source_port, target_cell,
                edge.target_port, edge.delay});
            const auto& value = runtime.parameters()[edge.weight_parameter_index].value;
            const double weight = std::holds_alternative<ContinuousValue>(value)
                ? std::get<ContinuousValue>(value).value
                : std::numeric_limits<double>::quiet_NaN();
            source.seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, weight});
        }
        std::sort(
            source.definition.edges.begin(), source.definition.edges.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });
        return source;
    }

    static std::optional<uint64_t> responsible_event(
        std::span<const GraphEditEvent> events,
        const GraphEditError& error) {
        if (error.event_id.has_value()) return error.event_id;
        std::optional<uint64_t> responsible;
        for (const auto& event : events) {
            bool matches = false;
            std::visit(
                [&](const auto& action) {
                    using Action = std::decay_t<decltype(action)>;
                    if constexpr (std::is_same_v<Action, RemoveCellAction>) {
                        matches = error.has_cell && action.id == error.cell;
                    } else if constexpr (std::is_same_v<Action, RemoveEdgeAction>) {
                        matches = error.has_edge && action.id == error.edge;
                    } else if constexpr (std::is_same_v<Action, AddCellAction>) {
                        matches = error.has_cell && action.birth.id == error.cell;
                    } else if constexpr (std::is_same_v<Action, AddEdgeAction>) {
                        matches = error.has_edge && action.birth.id == error.edge;
                    } else {
                        matches = (error.has_edge &&
                                   (action.edge == error.edge ||
                                    action.source_to_new == error.edge ||
                                    action.new_to_target == error.edge)) ||
                            (error.has_cell && action.inserted.id == error.cell);
                    }
                },
                event.action);
            if (matches) {
                if (responsible.has_value()) return std::nullopt;
                responsible = event.event_id;
            }
        }
        if (responsible.has_value()) return responsible;
        return events.size() == 1
            ? std::optional<uint64_t>(events.front().event_id)
            : std::nullopt;
    }

    static std::optional<uint64_t> compile_event_id(
        std::span<const GraphEditEvent> events,
        const CompileError& error) {
        GraphEditError mapped{
            GraphEditErrorCode::CompilationFailed,
            error.reason,
            std::nullopt,
            error.object_kind == CompileObjectKind::Cell ||
                error.object_kind == CompileObjectKind::CellParameter,
            CellId{error.object_id},
            error.object_kind == CompileObjectKind::Edge ||
                error.object_kind == CompileObjectKind::EdgeParameter,
            EdgeId{error.object_id}};
        return responsible_event(events, mapped);
    }

    static std::optional<GraphEditError> validate_event_ids(
        std::span<const GraphEditEvent> events) {
        std::vector<uint64_t> ids;
        ids.reserve(events.size());
        for (const auto& event : events) ids.push_back(event.event_id);
        std::sort(ids.begin(), ids.end());
        const auto duplicate = std::adjacent_find(ids.begin(), ids.end());
        if (duplicate != ids.end()) {
            return GraphEditError{
                GraphEditErrorCode::DuplicateEventId,
                "event IDs must be unique within one edit attempt",
                *duplicate};
        }
        return std::nullopt;
    }

    std::optional<GraphEditError> collect_and_validate(
        const GraphDefinition& source,
        std::span<const GraphEditEvent> events,
        std::vector<CellId>& removed_cells,
        std::vector<EdgeId>& removed_edges,
        std::vector<CellBirth>& added_cells,
        std::vector<EdgeBirth>& added_edges,
        std::vector<SplitEdgeAction>& splits) const {
        std::vector<CellId> new_cells;
        std::vector<EdgeId> new_edges;
        std::vector<EdgeId> split_targets;
        std::map<CellId, uint64_t> cell_event_ids;
        std::map<EdgeId, uint64_t> edge_event_ids;
        std::map<EdgeId, uint64_t> split_event_ids;
        for (const auto& event : events) {
            std::visit(
                [&](const auto& action) {
                    using Action = std::decay_t<decltype(action)>;
                    if constexpr (std::is_same_v<Action, RemoveCellAction>) {
                        removed_cells.push_back(action.id);
                        cell_event_ids.emplace(action.id, event.event_id);
                    } else if constexpr (std::is_same_v<Action, RemoveEdgeAction>) {
                        removed_edges.push_back(action.id);
                        edge_event_ids.emplace(action.id, event.event_id);
                    } else if constexpr (std::is_same_v<Action, AddCellAction>) {
                        added_cells.push_back(action.birth);
                        new_cells.push_back(action.birth.id);
                        cell_event_ids.emplace(action.birth.id, event.event_id);
                    } else if constexpr (std::is_same_v<Action, AddEdgeAction>) {
                        added_edges.push_back(action.birth);
                        new_edges.push_back(action.birth.id);
                        edge_event_ids.emplace(action.birth.id, event.event_id);
                    } else {
                        splits.push_back(action);
                        split_targets.push_back(action.edge);
                        new_cells.push_back(action.inserted.id);
                        new_edges.push_back(action.source_to_new);
                        new_edges.push_back(action.new_to_target);
                        cell_event_ids.emplace(action.inserted.id, event.event_id);
                        split_event_ids.emplace(action.edge, event.event_id);
                    }
                },
                event.action);
        }
        std::sort(removed_cells.begin(), removed_cells.end());
        std::sort(removed_edges.begin(), removed_edges.end());
        std::sort(new_cells.begin(), new_cells.end());
        std::sort(new_edges.begin(), new_edges.end());
        std::sort(split_targets.begin(), split_targets.end());

        if (const auto error = duplicate_or_existing_cells(
                source, removed_cells, new_cells)) {
            return error;
        }
        if (const auto error = duplicate_or_existing_edges(
                source, removed_edges, split_targets, new_edges)) {
            return error;
        }
        for (const auto& id : removed_cells) {
            if (!find_cell(source.cells, id).has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::MissingObject,
                    "remove cell refers to a missing source cell",
                    cell_event_ids.at(id), true, id};
            }
        }
        for (const auto& id : removed_edges) {
            if (!find_edge(source.edges, id).has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::MissingObject,
                    "remove edge refers to a missing source edge",
                    edge_event_ids.at(id), false, CellId{0}, true, id};
            }
        }
        for (const auto& split : splits) {
            const auto old = find_edge(source.edges, split.edge);
            if (!old.has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::MissingObject,
                    "split refers to a missing source edge",
                    split_event_ids.at(split.edge),
                    false, CellId{0}, true, split.edge};
            }
            if (contains(removed_edges, split.edge)) {
                return GraphEditError{
                    GraphEditErrorCode::ConflictingRequest,
                    "an edge cannot be removed and split in one batch",
                    split_event_ids.at(split.edge),
                    false, CellId{0}, true, split.edge};
            }
            if (contains(removed_cells, old->source) ||
                contains(removed_cells, old->target)) {
                return GraphEditError{
                    GraphEditErrorCode::ConflictingRequest,
                    "a split cannot retain an edge incident to a removed cell",
                    split_event_ids.at(split.edge),
                    false, CellId{0}, true, split.edge};
            }
            if (!std::isfinite(split.source_weight) ||
                !std::isfinite(split.target_weight)) {
                return GraphEditError{
                    GraphEditErrorCode::NonFiniteSeed,
                    "split edge weights must be finite",
                    split_event_ids.at(split.edge),
                    false, CellId{0}, true, split.edge};
            }
            const auto contract = contract_for(split.inserted.type);
            if (!contract.has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::InvalidType,
                    "split inserted cell type is not in the generated schema",
                    cell_event_ids.at(split.inserted.id),
                    true, split.inserted.id};
            }
            if (split.new_input_port.value >= contract->get().input_port_count) {
                return GraphEditError{
                    GraphEditErrorCode::InvalidSplit,
                    "split inserted cell must consume the explicitly selected input port",
                    cell_event_ids.at(split.inserted.id),
                    true, split.inserted.id};
            }
            if (split.inserted.id == old->source ||
                split.inserted.id == old->target) {
                return GraphEditError{
                    GraphEditErrorCode::InvalidSplit,
                    "split inserted cell ID must be distinct from both edge endpoints",
                    cell_event_ids.at(split.inserted.id),
                    true, split.inserted.id};
            }
            if (auto error = validate_birth(split.inserted)) {
                error->event_id = cell_event_ids.at(split.inserted.id);
                return error;
            }
        }
        for (const auto& cell : added_cells) {
            if (auto error = validate_birth(cell)) {
                error->event_id = cell_event_ids.at(cell.id);
                return error;
            }
        }
        for (const auto& edge : added_edges) {
            if (!std::isfinite(edge.initial_weight)) {
                return GraphEditError{
                    GraphEditErrorCode::NonFiniteSeed,
                    "new edge weight must be finite",
                    edge_event_ids.at(edge.id),
                    false, CellId{0}, true, edge.id};
            }
            if (edge.source_port.value != 0) {
                return GraphEditError{
                    GraphEditErrorCode::InvalidPort,
                    "source output port must be zero",
                    edge_event_ids.at(edge.id),
                    false, CellId{0}, true, edge.id};
            }
        }
        for (const auto id : removed_cells) {
            if (contains(removed_cells, id) &&
                (contains(new_cells, id) || is_retired(retired_cells_, id))) {
                return GraphEditError{
                    GraphEditErrorCode::ConflictingRequest,
                    "a removed cell ID cannot be reborn in the same batch or editor history"};
            }
        }
        for (const auto id : removed_edges) {
            if (contains(new_edges, id) || is_retired(retired_edges_, id)) {
                return GraphEditError{
                    GraphEditErrorCode::ConflictingRequest,
                    "a removed edge ID cannot be reborn in the same batch or editor history"};
            }
        }
        for (const auto& edge : added_edges) {
            if (contains(removed_cells, edge.source) ||
                contains(removed_cells, edge.target)) {
                return GraphEditError{
                    GraphEditErrorCode::ConflictingRequest,
                    "new edge cannot reference a cell removed in the same batch",
                    edge_event_ids.at(edge.id),
                    false, CellId{0}, true, edge.id};
            }
        }
        return std::nullopt;
    }

    std::optional<GraphEditError> duplicate_or_existing_cells(
        const GraphDefinition& source,
        const std::vector<CellId>& removed,
        const std::vector<CellId>& all_new) const {
        if (has_duplicate(removed) || has_duplicate(all_new)) {
            return GraphEditError{
                GraphEditErrorCode::DuplicateId,
                "cell IDs are duplicated within the edit batch"};
        }
        for (const auto id : all_new) {
            if (find_cell(source.cells, id).has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::DuplicateId,
                    "new cell ID already exists in the source graph",
                    std::nullopt, true, id};
            }
            if (is_retired(retired_cells_, id)) {
                return GraphEditError{
                    GraphEditErrorCode::RetiredId,
                    "new cell ID was retired by an earlier committed edit",
                    std::nullopt, true, id};
            }
        }
        return std::nullopt;
    }

    std::optional<GraphEditError> duplicate_or_existing_edges(
        const GraphDefinition& source,
        const std::vector<EdgeId>& removed,
        const std::vector<EdgeId>& split_targets,
        const std::vector<EdgeId>& all_new) const {
        if (has_duplicate(removed) || has_duplicate(split_targets) ||
            has_duplicate(all_new)) {
            return GraphEditError{
                GraphEditErrorCode::DuplicateId,
                "edge IDs are duplicated within the edit batch"};
        }
        for (const auto id : all_new) {
            if (find_edge(source.edges, id).has_value()) {
                return GraphEditError{
                    GraphEditErrorCode::DuplicateId,
                    "new edge ID already exists in the source graph",
                    std::nullopt, false, CellId{0}, true, id};
            }
            if (is_retired(retired_edges_, id)) {
                return GraphEditError{
                    GraphEditErrorCode::RetiredId,
                    "new edge ID was retired by an earlier committed edit",
                    std::nullopt, false, CellId{0}, true, id};
            }
        }
        return std::nullopt;
    }

    static std::optional<GraphEditError> validate_birth(const CellBirth& birth) {
        const auto contract = contract_for(birth.type);
        if (!contract.has_value()) {
            return GraphEditError{
                GraphEditErrorCode::InvalidType,
                "cell type is not in the generated schema",
                std::nullopt, true, birth.id};
        }
        for (std::size_t slot = 0; slot < 2; ++slot) {
            if (!validate_parameter(
                    contract->get().parameters[slot], birth.parameters[slot])) {
                return GraphEditError{
                    GraphEditErrorCode::InvalidParameter,
                    "cell birth parameters do not match the generated typed schema",
                    std::nullopt, true, birth.id};
            }
        }
        return std::nullopt;
    }

    static void remove_objects(
        GraphDefinition& candidate,
        const std::vector<CellId>& cells,
        const std::vector<EdgeId>& edges) {
        candidate.cells.erase(
            std::remove_if(
                candidate.cells.begin(), candidate.cells.end(),
                [&](const CellDefinition& cell) {
                    return contains(cells, cell.id);
                }),
            candidate.cells.end());
        candidate.edges.erase(
            std::remove_if(
                candidate.edges.begin(), candidate.edges.end(),
                [&](const EdgeDefinition& edge) {
                    return contains(edges, edge.id) ||
                        contains(cells, edge.source) ||
                        contains(cells, edge.target);
                }),
            candidate.edges.end());
    }

    static InitialParameterSeeds seeds_for_candidate(
        const InitialParameterSeeds& source_seeds,
        const GraphDefinition& candidate,
        const std::vector<CellBirth>& added_cells,
        const std::vector<EdgeBirth>& added_edges,
        const std::vector<SplitEdgeAction>& splits) {
        std::map<CellId, std::array<ParameterValue, 2>> cell_values;
        for (const auto& seed : source_seeds.cell_parameters) {
            cell_values[seed.cell][static_cast<std::size_t>(seed.slot)] =
                seed.value;
        }
        for (const auto& birth : added_cells) {
            cell_values[birth.id] = birth.parameters;
        }
        for (const auto& split : splits) {
            cell_values[split.inserted.id] = split.inserted.parameters;
        }

        std::map<EdgeId, double> edge_values;
        for (const auto& seed : source_seeds.edge_weights) {
            edge_values[seed.edge] = seed.initial_weight;
        }
        for (const auto& birth : added_edges) {
            edge_values[birth.id] = birth.initial_weight;
        }
        for (const auto& split : splits) {
            edge_values[split.source_to_new] = split.source_weight;
            edge_values[split.new_to_target] = split.target_weight;
        }

        InitialParameterSeeds seeds;
        for (const auto& cell : candidate.cells) {
            const auto value = cell_values.find(cell.id);
            if (value != cell_values.end()) {
                for (std::size_t slot = 0; slot < 2; ++slot) {
                    seeds.cell_parameters.push_back(
                        CellParameterSeed{
                            cell.id, static_cast<ParameterSlot>(slot),
                            value->second[slot]});
                }
            }
        }
        for (const auto& edge : candidate.edges) {
            const auto value = edge_values.find(edge.id);
            if (value != edge_values.end()) {
                seeds.edge_weights.push_back(
                    EdgeParameterSeed{edge.id, value->second});
            }
        }
        return seeds;
    }

    static void initialize_diagnostics(
        GraphEditReport& report,
        std::span<const GraphEditEvent> events,
        GraphRevision old_revision) {
        report.diagnostics.reserve(events.size());
        for (const auto& event : events) {
            GraphEditDiagnostic diagnostic;
            diagnostic.event_id = event.event_id;
            diagnostic.action = action_kind(event.action);
            diagnostic.success = false;
            diagnostic.old_revision = old_revision;
            diagnostic.new_revision = old_revision;
            touched(event.action, diagnostic.cells, diagnostic.edges);
            diagnostic.reason = "not committed";
            report.diagnostics.push_back(std::move(diagnostic));
        }
        std::sort(
            report.diagnostics.begin(), report.diagnostics.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.event_id < rhs.event_id;
            });
    }

    static void enrich_diagnostics(
        GraphEditReport& report,
        std::span<const GraphEditEvent> events,
        const GraphDefinition& source) {
        std::map<uint64_t, std::size_t> diagnostic_index;
        std::map<CellId, std::vector<std::size_t>> incident_cells;
        std::map<EdgeId, std::size_t> referenced_edges;
        for (std::size_t index = 0; index < report.diagnostics.size(); ++index) {
            diagnostic_index.emplace(report.diagnostics[index].event_id, index);
        }
        for (const auto& event : events) {
            std::visit(
                [&](const auto& action) {
                    using Action = std::decay_t<decltype(action)>;
                    const auto found = diagnostic_index.find(event.event_id);
                    if (found == diagnostic_index.end()) return;
                    if constexpr (std::is_same_v<Action, RemoveCellAction>) {
                        incident_cells[action.id].push_back(found->second);
                    } else if constexpr (std::is_same_v<Action, RemoveEdgeAction>) {
                        referenced_edges[action.id] = found->second;
                    } else if constexpr (std::is_same_v<Action, SplitEdgeAction>) {
                        referenced_edges[action.edge] = found->second;
                    }
                },
                event.action);
        }
        for (const auto& edge : source.edges) {
            const auto source_diags = incident_cells.find(edge.source);
            const auto target_diags = incident_cells.find(edge.target);
            auto add_incident = [&](const auto& found) {
                if (found == incident_cells.end()) return;
                for (const auto index : found->second) {
                    report.diagnostics[index].edges.push_back(edge.id);
                    report.diagnostics[index].cells.push_back(edge.source);
                    report.diagnostics[index].cells.push_back(edge.target);
                }
            };
            add_incident(source_diags);
            if (edge.target != edge.source) add_incident(target_diags);
            const auto referenced = referenced_edges.find(edge.id);
            if (referenced != referenced_edges.end()) {
                auto& diagnostic = report.diagnostics[referenced->second];
                diagnostic.cells.push_back(edge.source);
                diagnostic.cells.push_back(edge.target);
            }
        }
        for (std::size_t index = 0; index < events.size(); ++index) {
            const auto diagnostic_it = diagnostic_index.find(events[index].event_id);
            if (diagnostic_it == diagnostic_index.end()) continue;
            std::visit(
                [&](const auto& action) {
                    using Action = std::decay_t<decltype(action)>;
                    if constexpr (std::is_same_v<Action, SplitEdgeAction>) {
                        auto& diagnostic = report.diagnostics[diagnostic_it->second];
                        diagnostic.edges.push_back(action.edge);
                        diagnostic.edges.push_back(action.source_to_new);
                        diagnostic.edges.push_back(action.new_to_target);
                    }
                },
                events[index].action);
        }
        for (auto& diagnostic : report.diagnostics) {
            normalize(diagnostic.cells);
            normalize(diagnostic.edges);
        }
    }

    static void mark_success_diagnostics(
        GraphEditReport& report,
        GraphRevision old_revision,
        GraphRevision new_revision) {
        for (auto& diagnostic : report.diagnostics) {
            diagnostic.success = true;
            diagnostic.old_revision = old_revision;
            diagnostic.new_revision = new_revision;
            diagnostic.reason = "committed";
        }
    }

    static GraphEditActionKind action_kind(const GraphEditAction& action) {
        return std::visit(
            [](const auto& value) -> GraphEditActionKind {
                using Action = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Action, RemoveCellAction>) {
                    return GraphEditActionKind::RemoveCell;
                } else if constexpr (std::is_same_v<Action, RemoveEdgeAction>) {
                    return GraphEditActionKind::RemoveEdge;
                } else if constexpr (std::is_same_v<Action, AddCellAction>) {
                    return GraphEditActionKind::AddCell;
                } else if constexpr (std::is_same_v<Action, AddEdgeAction>) {
                    return GraphEditActionKind::AddEdge;
                } else {
                    return GraphEditActionKind::SplitEdge;
                }
            },
            action);
    }

    static void touched(
        const GraphEditAction& action,
        std::vector<CellId>& cells,
        std::vector<EdgeId>& edges) {
        std::visit(
            [&](const auto& value) {
                using Action = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Action, RemoveCellAction>) {
                    cells.push_back(value.id);
                } else if constexpr (std::is_same_v<Action, RemoveEdgeAction>) {
                    edges.push_back(value.id);
                } else if constexpr (std::is_same_v<Action, AddCellAction>) {
                    cells.push_back(value.birth.id);
                } else if constexpr (std::is_same_v<Action, AddEdgeAction>) {
                    edges.push_back(value.birth.id);
                    cells.push_back(value.birth.source);
                    cells.push_back(value.birth.target);
                } else {
                    cells.push_back(value.inserted.id);
                    edges.push_back(value.edge);
                    edges.push_back(value.source_to_new);
                    edges.push_back(value.new_to_target);
                }
            },
            action);
        std::sort(cells.begin(), cells.end());
        std::sort(edges.begin(), edges.end());
    }

    static bool has_duplicate(const std::vector<CellId>& ids) {
        auto sorted = ids;
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
    }

    static bool has_duplicate(const std::vector<EdgeId>& ids) {
        auto sorted = ids;
        std::sort(sorted.begin(), sorted.end());
        return std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end();
    }

    static bool contains(const std::vector<CellId>& ids, CellId id) {
        return std::binary_search(ids.begin(), ids.end(), id);
    }

    static bool contains(const std::vector<EdgeId>& ids, EdgeId id) {
        return std::binary_search(ids.begin(), ids.end(), id);
    }

    template <typename T>
    static bool is_retired(const std::vector<T>& ids, T id) {
        return std::binary_search(ids.begin(), ids.end(), id);
    }

    template <typename T>
    static void normalize(std::vector<T>& ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    }

    static std::optional<CellDefinition> find_cell(
        const std::vector<CellDefinition>& cells,
        CellId id) {
        const auto it = std::lower_bound(
            cells.begin(), cells.end(), id,
            [](const auto& cell, CellId sought) { return cell.id < sought; });
        return it != cells.end() && it->id == id
            ? std::optional<CellDefinition>(*it)
            : std::nullopt;
    }

    static std::optional<EdgeDefinition> find_edge(
        const std::vector<EdgeDefinition>& edges,
        EdgeId id) {
        const auto it = std::lower_bound(
            edges.begin(), edges.end(), id,
            [](const auto& edge, EdgeId sought) { return edge.id < sought; });
        return it != edges.end() && it->id == id
            ? std::optional<EdgeDefinition>(*it)
            : std::nullopt;
    }

    GraphIdentity identity_{};
    GraphRevision expected_revision_{};
    std::shared_ptr<const CompiledGraph> expected_plan_;
    std::optional<GraphEditError> history_error_;
    std::vector<CellId> retired_cells_;
    std::vector<EdgeId> retired_edges_;
};

}  // namespace kun::core
