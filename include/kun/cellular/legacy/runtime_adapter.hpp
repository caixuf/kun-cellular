#pragma once

#include "kun/cellular/core/runtime_state.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace kun::migration {

enum class RuntimeImportCode : uint8_t {
    Accepted,
    InvalidSnapshot,
    RuntimeRejected,
    MissingCellState,
    CellTypeMismatch,
};

struct RuntimeImportError {
    RuntimeImportCode code{RuntimeImportCode::InvalidSnapshot};
    std::string reason;
    bool has_cell{false};
    core::CellId cell{};
};

struct RuntimeImportResult {
    std::shared_ptr<core::RuntimeState> runtime;
    std::vector<SnapshotReadout> readouts;
    std::vector<SnapshotEdgeProvenance> edge_provenance;
    std::optional<RuntimeImportError> error;

    bool ok() const { return runtime != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

inline RuntimeImportResult import_execution_snapshot_runtime(
    const ExecutionSnapshot& snapshot) {
    std::vector<const SnapshotCellState*> states_by_id;
    states_by_id.reserve(snapshot.cell_states().size());
    for (const auto& state : snapshot.cell_states()) states_by_id.push_back(&state);
    std::sort(states_by_id.begin(), states_by_id.end(),
              [](const auto* lhs, const auto* rhs) {
                  return lhs->cell < rhs->cell;
              });
    std::vector<core::RuntimeCellState> states;
    states.reserve(snapshot.compiled_plan()->cells().size());
    std::size_t cursor = 0;
    for (const auto& compiled : snapshot.compiled_plan()->cells()) {
        while (cursor < states_by_id.size() &&
               states_by_id[cursor]->cell < compiled.id) {
            ++cursor;
        }
        if (cursor == states_by_id.size() ||
            states_by_id[cursor]->cell != compiled.id) {
            return {
                nullptr,
                {},
                {},
                RuntimeImportError{
                    RuntimeImportCode::MissingCellState,
                    "imported snapshot has no state for a declared cell",
                    true,
                    compiled.id}};
        }
        const auto* imported = states_by_id[cursor++];
        if (cursor < states_by_id.size() &&
            states_by_id[cursor]->cell == compiled.id) {
            return {
                nullptr,
                {},
                {},
                RuntimeImportError{
                    RuntimeImportCode::InvalidSnapshot,
                    "imported snapshot contains duplicate stable cell IDs",
                    true,
                    compiled.id}};
        }
        if (imported->type != compiled.type) {
            return {
                nullptr,
                {},
                {},
                RuntimeImportError{
                    RuntimeImportCode::CellTypeMismatch,
                    "imported computational state type does not match the plan",
                    true,
                    compiled.id}};
        }
        states.push_back(core::RuntimeCellState{
            imported->cell,
            imported->type,
            imported->state_val,
            imported->aux_state,
            imported->prev_input,
            imported->output_val,
            imported->prev_output_val,
            imported->delay_buffer,
            imported->delay_idx,
            imported->latch_state,
            imported->activation_count,
            true});
    }

    auto restored = core::RuntimeState::from_imported_state(
        snapshot.compiled_plan(),
        snapshot.current_parameter_values(),
        states);
    if (!restored.ok()) {
        return {
            nullptr,
            {},
            {},
            RuntimeImportError{
                RuntimeImportCode::RuntimeRejected,
                restored.error ? restored.error->diagnostic()
                                : "runtime rejected imported state",
                false,
            core::CellId{0}}};
    }
    return {
        std::move(restored.runtime),
        std::vector<SnapshotReadout>(
            snapshot.readouts().begin(), snapshot.readouts().end()),
        std::vector<SnapshotEdgeProvenance>(
            snapshot.edge_provenance().begin(), snapshot.edge_provenance().end()),
        std::nullopt};
}

}  // namespace kun::migration
