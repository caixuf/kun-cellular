#pragma once

#include "kun/cellular/core/compiled_graph.hpp"
#include "kun/cellular/core/kernel_bridge.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace kun::core {

class CompiledExecutor;

struct RuntimeCellState {
    CellId cell{};
    CellType type{CellType::OP_EMA};
    double state_val{0.0};
    double aux_state{0.0};
    double prev_input{0.0};
    double output_val{0.0};
    double prev_output_val{0.0};
    std::array<double, 16> delay_buffer{};
    uint8_t delay_idx{0};
    bool latch_state{false};
    uint32_t activation_count{0};
    bool initialized{false};
};

enum class RuntimeErrorCode : uint8_t {
    InvalidPlan,
    InvalidBindings,
    InvalidState,
    UnsupportedProfile,
    BindingMismatch,
    ParameterNotWritable,
    InvalidParameter,
    SnapshotMismatch,
    MigrationIdentityMismatch,
    MigrationRevisionMismatch,
    MigrationInvalidInput,
};

struct RuntimeError {
    RuntimeErrorCode code{RuntimeErrorCode::InvalidPlan};
    std::string reason;
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};

    std::string diagnostic() const {
        std::string result;
        if (has_cell) result += "cell id=" + std::to_string(cell.value) + ": ";
        if (has_edge) result += "edge id=" + std::to_string(edge.value) + ": ";
        result += reason;
        return result;
    }
};

class RuntimeSnapshot final {
public:
    std::shared_ptr<const CompiledGraph> plan() const { return plan_; }
    std::span<const InitialParameterValue> parameters() const { return parameters_; }
    std::span<const RuntimeCellState> cells() const { return cells_; }
    uint64_t tick() const { return tick_; }

private:
    friend class RuntimeState;

    RuntimeSnapshot(
        std::shared_ptr<const CompiledGraph> plan,
        std::vector<InitialParameterValue> parameters,
        std::vector<RuntimeCellState> cells,
        uint64_t tick)
        : plan_(std::move(plan)),
          parameters_(std::move(parameters)),
          cells_(std::move(cells)),
          tick_(tick) {}

    std::shared_ptr<const CompiledGraph> plan_;
    std::vector<InitialParameterValue> parameters_;
    std::vector<RuntimeCellState> cells_;
    uint64_t tick_{0};
};

struct RuntimeResult {
    std::shared_ptr<class RuntimeState> runtime;
    std::optional<RuntimeError> error;

    bool ok() const { return runtime != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct RuntimeMutationResult {
    std::optional<RuntimeError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

namespace detail {

inline RuntimeMutationResult runtime_failure(
    RuntimeErrorCode code,
    std::string reason,
    bool has_cell = false,
    CellId cell = CellId{0},
    bool has_edge = false,
    EdgeId edge = EdgeId{0}) {
    return {RuntimeError{code, std::move(reason), has_cell, cell, has_edge, edge}};
}

inline RuntimeResult runtime_creation_failure(
    RuntimeErrorCode code,
    std::string reason) {
    return {nullptr, RuntimeError{code, std::move(reason)}};
}

inline bool same_plan(
    const CompiledGraph& lhs,
    const CompiledGraph& rhs) {
    if (lhs.identity() != rhs.identity() ||
        lhs.revision() != rhs.revision() ||
        lhs.profile() != rhs.profile() ||
        lhs.semantic_version() != rhs.semantic_version() ||
        lhs.parameter_binding_count() != rhs.parameter_binding_count() ||
        lhs.cells().size() != rhs.cells().size() ||
        lhs.edges().size() != rhs.edges().size() ||
        lhs.execution_order().size() != rhs.execution_order().size() ||
        lhs.port_reductions().size() != rhs.port_reductions().size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.cells().size(); ++i) {
        const auto& a = lhs.cells()[i];
        const auto& b = rhs.cells()[i];
        if (a.id != b.id || a.type != b.type ||
            a.parameter_indices != b.parameter_indices ||
            a.input_port_count != b.input_port_count) {
            return false;
        }
    }
    for (std::size_t i = 0; i < lhs.edges().size(); ++i) {
        const auto& a = lhs.edges()[i];
        const auto& b = rhs.edges()[i];
        if (a.id != b.id || a.source_index != b.source_index ||
            a.source_port != b.source_port || a.target_index != b.target_index ||
            a.target_port != b.target_port || a.delay != b.delay ||
            a.weight_parameter_index != b.weight_parameter_index) {
            return false;
        }
    }
    if (!std::equal(lhs.execution_order().begin(), lhs.execution_order().end(),
                    rhs.execution_order().begin())) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.port_reductions().size(); ++i) {
        const auto& a = lhs.port_reductions()[i];
        const auto& b = rhs.port_reductions()[i];
        if (a.target_index != b.target_index ||
            a.target_port != b.target_port ||
            a.edge_begin != b.edge_begin || a.edge_end != b.edge_end) {
            return false;
        }
    }
    return true;
}

inline bool valid_runtime_cell(const RuntimeCellState& state) {
    if (state.delay_idx >= 16) return false;
    if (!valid_runtime_state(state.state_val) ||
        !valid_runtime_state(state.aux_state) ||
        !valid_runtime_state(state.prev_input) ||
        !valid_runtime_state(state.output_val) ||
        !valid_runtime_state(state.prev_output_val)) {
        return false;
    }
    return std::all_of(
        state.delay_buffer.begin(), state.delay_buffer.end(), valid_runtime_state);
}

inline bool valid_runtime_parameter(
    const CompiledGraph& plan,
    const ParameterBinding& binding,
    const ParameterValue& value) {
    if (binding.kind != ParameterBindingKind::CellParameter &&
        binding.kind != ParameterBindingKind::EdgeWeight) {
        return false;
    }
    if (binding.kind == ParameterBindingKind::EdgeWeight) {
        return std::holds_alternative<ContinuousValue>(value) &&
               std::isfinite(std::get<ContinuousValue>(value).value);
    }
    if (static_cast<uint8_t>(binding.slot) >= 2) return false;
    const auto cells = plan.cells();
    const auto it = std::lower_bound(
        cells.begin(), cells.end(), binding.cell,
        [](const CompiledCell& cell, CellId sought) {
            return cell.id < sought;
        });
    if (it == cells.end() || it->id != binding.cell ||
        it->parameter_indices[static_cast<std::size_t>(binding.slot)] != binding.index) {
        return false;
    }
    const auto contract = contract_for(it->type);
    return contract.has_value() &&
           validate_parameter(
               contract->get().parameters[static_cast<std::size_t>(binding.slot)],
               value);
}

inline std::optional<RuntimeError> validate_parameter_bindings(
    const CompiledGraph& plan,
    std::span<const InitialParameterValue> parameters) {
    if (parameters.size() != plan.parameter_binding_count()) {
        return RuntimeError{
            RuntimeErrorCode::InvalidBindings,
            "runtime parameter store size does not match the compiled plan",
            false, CellId{0}, false, EdgeId{0}};
    }
    for (const auto& cell : plan.cells()) {
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto index = cell.parameter_indices[slot];
            if (index >= parameters.size()) {
                return RuntimeError{
                    RuntimeErrorCode::InvalidBindings,
                    "compiled cell parameter index is outside the parameter store",
                    true, cell.id, false, EdgeId{0}};
            }
            const auto& actual = parameters[index];
            if (actual.binding.kind != ParameterBindingKind::CellParameter ||
                actual.binding.index != index ||
                actual.binding.cell != cell.id ||
                actual.binding.edge != EdgeId{0} ||
                actual.binding.slot != static_cast<ParameterSlot>(slot) ||
                !valid_runtime_parameter(plan, actual.binding, actual.value)) {
                return RuntimeError{
                    RuntimeErrorCode::BindingMismatch,
                    "runtime parameter binding is not the exact binding emitted by the plan",
                    true, cell.id, false, EdgeId{0}};
            }
        }
    }
    for (const auto& edge : plan.edges()) {
        const auto index = edge.weight_parameter_index;
        if (index >= parameters.size()) {
            return RuntimeError{
                RuntimeErrorCode::InvalidBindings,
                "compiled edge weight index is outside the parameter store",
                false, CellId{0}, true, edge.id};
        }
        const auto& actual = parameters[index];
        if (actual.binding.kind != ParameterBindingKind::EdgeWeight ||
            actual.binding.index != index ||
            actual.binding.edge != edge.id ||
            actual.binding.cell != CellId{0} ||
            actual.binding.slot != ParameterSlot::Param1 ||
            !valid_runtime_parameter(plan, actual.binding, actual.value)) {
            return RuntimeError{
                RuntimeErrorCode::BindingMismatch,
                "runtime parameter binding is not the exact binding emitted by the plan",
                false, CellId{0}, true, edge.id};
        }
    }
    return std::nullopt;
}

}  // namespace detail

class RuntimeState final {
public:
    static RuntimeResult create(
        std::shared_ptr<const CompiledGraph> plan,
        std::shared_ptr<const InitialParameterValues> initial_values) {
        if (!plan || !initial_values) {
            return detail::runtime_creation_failure(
                RuntimeErrorCode::InvalidPlan,
                "runtime requires an owned compiled plan and typed initial values");
        }
        if (!semantic_profile_from_code(static_cast<uint8_t>(plan->profile()))) {
            return detail::runtime_creation_failure(
                RuntimeErrorCode::UnsupportedProfile,
                "runtime graph uses an unknown semantic profile");
        }
        RuntimeState candidate(std::move(plan), std::move(initial_values));
        if (const auto error = candidate.validate_bindings()) {
            return {nullptr, error};
        }
        return {std::make_shared<RuntimeState>(std::move(candidate)), std::nullopt};
    }

    static RuntimeResult from_snapshot(const RuntimeSnapshot& snapshot) {
        if (!snapshot.plan_) {
            return detail::runtime_creation_failure(
                RuntimeErrorCode::InvalidPlan,
                "runtime snapshot has no owned compiled plan");
        }
        auto parameters = std::make_shared<const InitialParameterValues>(
            snapshot.parameters_);
        auto created = create(snapshot.plan_, std::move(parameters));
        if (!created.ok()) return created;
        if (const auto restored = created.runtime->restore_snapshot(snapshot);
            !restored.ok()) {
            return {nullptr, restored.error};
        }
        return created;
    }

    // Cold adapters use this seam to connect an already validated external
    // computational snapshot without exposing a public mutable snapshot
    // constructor or cell-state setter.
    static RuntimeResult from_imported_state(
        std::shared_ptr<const CompiledGraph> plan,
        std::shared_ptr<const InitialParameterValues> parameters,
        std::span<const RuntimeCellState> cells,
        uint64_t tick = 0) {
        if (!plan || !parameters) {
            return detail::runtime_creation_failure(
                RuntimeErrorCode::InvalidPlan,
                "imported runtime state requires an owned plan and typed parameters");
        }
        auto created = create(plan, parameters);
        if (!created.ok()) return created;
        const RuntimeSnapshot snapshot(
            std::move(plan),
            std::vector<InitialParameterValue>(
                parameters->entries().begin(), parameters->entries().end()),
            std::vector<RuntimeCellState>(cells.begin(), cells.end()),
            tick);
        if (const auto restored = created.runtime->restore_snapshot(snapshot);
            !restored.ok()) {
            return {nullptr, restored.error};
        }
        return created;
    }

    std::shared_ptr<const CompiledGraph> plan() const { return plan_; }
    SemanticProfile profile() const { return plan_->profile(); }
    GraphIdentity identity() const { return plan_->identity(); }
    GraphRevision revision() const { return plan_->revision(); }
    uint64_t tick() const { return tick_; }
    bool bound_to(const CompiledGraph& graph) const {
        return plan_ && detail::same_plan(*plan_, graph);
    }

    std::span<const RuntimeCellState> cell_states() const { return cells_; }
    std::span<const InitialParameterValue> parameters() const { return parameters_; }

    const RuntimeCellState* cell_state(CellId id) const {
        const auto index = cell_index(id);
        return index.has_value() ? &cells_[*index] : nullptr;
    }

    std::optional<ParameterValue> parameter(CellId cell, ParameterSlot slot) const {
        const auto index = cell_parameter_index(cell, slot);
        if (!index.has_value()) return std::nullopt;
        return parameters_[*index].value;
    }

    const ParameterValue* parameter_at(std::size_t index) const {
        return index < parameters_.size() ? &parameters_[index].value : nullptr;
    }

    RuntimeMutationResult set_parameter(
        ParameterBinding binding,
        ParameterValue value) {
        if (binding.index >= parameters_.size()) {
            return detail::runtime_failure(
                RuntimeErrorCode::BindingMismatch,
                "parameter binding index is outside the owned runtime store");
        }
        const auto& expected = parameters_[binding.index].binding;
        if (expected.kind != binding.kind || expected.index != binding.index ||
            expected.cell != binding.cell || expected.edge != binding.edge ||
            expected.slot != binding.slot) {
            return detail::runtime_failure(
                RuntimeErrorCode::BindingMismatch,
                "parameter binding does not belong to this compiled plan");
        }
        if (!validate_runtime_parameter(binding, value)) {
            return detail::runtime_failure(
                RuntimeErrorCode::InvalidParameter,
                "parameter value does not match the generated typed contract",
                binding.kind == ParameterBindingKind::CellParameter,
                binding.cell,
                binding.kind == ParameterBindingKind::EdgeWeight,
                binding.edge);
        }
        parameters_[binding.index].value = std::move(value);
        return {};
    }

    RuntimeMutationResult set_continuous_parameter(
        CellId cell,
        ParameterSlot slot,
        double value) {
        const auto index = cell_parameter_index(cell, slot);
        if (!index.has_value()) {
            return detail::runtime_failure(
                RuntimeErrorCode::BindingMismatch,
                "cell parameter is not present in this plan", true, cell);
        }
        const auto& contract = contract_for(plan_->cells()[*cell_index(cell)].type)->get();
        const auto& descriptor = contract.parameters[static_cast<std::size_t>(slot)];
        if (!is_continuous_trainable(descriptor)) {
            return detail::runtime_failure(
                RuntimeErrorCode::ParameterNotWritable,
                "cell parameter is not an eligible continuous learning slot",
                true, cell);
        }
        return set_parameter(
            parameters_[*index].binding,
            ParameterValue{ContinuousValue{value}});
    }

    RuntimeMutationResult reset_episode() {
        for (std::size_t i = 0; i < cells_.size(); ++i) {
            reset_cell(cells_[i], plan_->cells()[i].type);
        }
        tick_ = 0;
        return {};
    }

    RuntimeResult fork_probe() const {
        auto copy = std::make_shared<RuntimeState>(*this);
        return {std::move(copy), std::nullopt};
    }

    RuntimeSnapshot snapshot() const {
        return RuntimeSnapshot(
            plan_, parameters_, std::vector<RuntimeCellState>(cells_.begin(), cells_.end()),
            tick_);
    }

    RuntimeMutationResult restore_snapshot(const RuntimeSnapshot& snapshot) {
        if (!snapshot.plan_ || !detail::same_plan(*plan_, *snapshot.plan_)) {
            return detail::runtime_failure(
                RuntimeErrorCode::SnapshotMismatch,
                "snapshot graph identity, revision, topology, or execution configuration differs");
        }
        if (snapshot.parameters_.size() != parameters_.size() ||
            snapshot.cells_.size() != cells_.size()) {
            return detail::runtime_failure(
                RuntimeErrorCode::SnapshotMismatch,
                "snapshot dimensions do not match the owned runtime");
        }
        for (std::size_t i = 0; i < snapshot.cells_.size(); ++i) {
            if (snapshot.cells_[i].cell != plan_->cells()[i].id ||
                snapshot.cells_[i].type != plan_->cells()[i].type) {
                return detail::runtime_failure(
                    RuntimeErrorCode::SnapshotMismatch,
                    "snapshot cell state is not bound to the plan's stable cell ID",
                    true, snapshot.cells_[i].cell);
            }
        }
        RuntimeState candidate(*this);
        candidate.parameters_ = snapshot.parameters_;
        candidate.cells_ = snapshot.cells_;
        candidate.tick_ = snapshot.tick_;
        if (const auto error = candidate.validate_bindings()) return {error};
        for (const auto& cell : candidate.cells_) {
            if (!detail::valid_runtime_cell(cell)) {
                return detail::runtime_failure(
                    RuntimeErrorCode::InvalidState,
                    "snapshot contains a non-finite state or invalid delay index",
                    true, cell.cell);
            }
        }
        parameters_ = std::move(candidate.parameters_);
        cells_ = std::move(candidate.cells_);
        tick_ = candidate.tick_;
        return {};
    }

private:
    friend class ReferenceExecutor;
    friend class CompiledExecutor;
    friend class RuntimeMigration;

    RuntimeState(
        std::shared_ptr<const CompiledGraph> plan,
        std::shared_ptr<const InitialParameterValues> initial_values)
        : plan_(std::move(plan)) {
        parameters_.assign(initial_values->entries().begin(), initial_values->entries().end());
        cells_.reserve(plan_->cells().size());
        for (const auto& cell : plan_->cells()) {
            cells_.push_back(RuntimeCellState{cell.id, cell.type});
        }
    }

    std::optional<std::size_t> cell_index(CellId id) const {
        const auto cells = plan_->cells();
        const auto it = std::lower_bound(
            cells.begin(), cells.end(), id,
            [](const CompiledCell& cell, CellId sought) {
                return cell.id < sought;
            });
        if (it == cells.end() || it->id != id) return std::nullopt;
        return static_cast<std::size_t>(it - cells.begin());
    }

    std::optional<std::size_t> cell_parameter_index(
        CellId cell,
        ParameterSlot slot) const {
        if (static_cast<uint8_t>(slot) >= 2) return std::nullopt;
        const auto index = cell_index(cell);
        if (!index.has_value()) return std::nullopt;
        return plan_->cells()[*index].parameter_indices[static_cast<std::size_t>(slot)];
    }

    std::optional<RuntimeError> validate_bindings() const {
        return detail::validate_parameter_bindings(*plan_, parameters_);
    }

    bool validate_runtime_parameter(
        const ParameterBinding& binding,
        const ParameterValue& value) const {
        return detail::valid_runtime_parameter(*plan_, binding, value);
    }

    static void reset_cell(RuntimeCellState& state, CellType type) {
        const auto contract = contract_for(type);
        if (!contract.has_value()) {
            state = RuntimeCellState{state.cell, type};
            return;
        }
        const auto reset_double = [](ResetAction action, double& value) {
            if (action == ResetAction::Zero) value = 0.0;
        };
        const auto reset_bool = [](ResetAction action, bool& value) {
            if (action == ResetAction::FalseValue) value = false;
        };
        const auto& metadata = contract->get().reset;
        reset_double(metadata.state_value, state.state_val);
        reset_double(metadata.aux_state, state.aux_state);
        reset_double(metadata.previous_input, state.prev_input);
        if (metadata.delay_buffer == ResetAction::Zero) {
            state.delay_buffer.fill(0.0);
        }
        if (metadata.delay_index == ResetAction::Zero) state.delay_idx = 0;
        reset_bool(metadata.latch_state, state.latch_state);
        if (metadata.activation_count == ResetAction::Zero) state.activation_count = 0;
        reset_double(metadata.output_value, state.output_val);
        state.prev_output_val = 0.0;
        state.initialized = false;
        // Unused fields are not observable by the typed kernel, but clearing
        // them prevents stale computational state from crossing an episode.
        if (metadata.aux_state == ResetAction::Unused) state.aux_state = 0.0;
        if (metadata.previous_input == ResetAction::Unused) state.prev_input = 0.0;
        if (metadata.delay_buffer == ResetAction::Unused) state.delay_buffer.fill(0.0);
        if (metadata.delay_index == ResetAction::Unused) state.delay_idx = 0;
        if (metadata.latch_state == ResetAction::Unused) state.latch_state = false;
        if (metadata.state_value == ResetAction::Unused) state.state_val = 0.0;
    }

    std::shared_ptr<const CompiledGraph> plan_;
    std::vector<InitialParameterValue> parameters_;
    std::vector<RuntimeCellState> cells_;
    uint64_t tick_{0};
};

}  // namespace kun::core
