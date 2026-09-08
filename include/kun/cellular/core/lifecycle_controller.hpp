#pragma once

#include "kun/cellular/core/cellular_graph_edit.hpp"
#include "kun/cellular/core/compiled_executor.hpp"
#include "kun/cellular/core/execution_control.hpp"
#include "kun/cellular/core/resource_ledger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace kun::core {

class CellularGrowthController;

enum class LifecycleState : uint8_t {
    Active = 0,
    Dormant = 1,
    Apoptotic = 2,
    Extinct = 3,
};

enum class LifecycleEventKind : uint8_t {
    BecameActive,
    BecameDormant,
    BecameApoptotic,
    CellRemoved,
    Extinct,
};

struct LifecycleConfig {
    double apoptotic_resource{0.0};
    double dormant_enter_resource{0.0};
    double dormant_exit_resource{0.0};
    uint64_t minimum_dwell_ticks{0};
    uint64_t cooldown_ticks{0};
};

enum class LifecycleErrorCode : uint8_t {
    InvalidConfig,
    BindingMismatch,
    InvalidState,
    NativeExecutionFailed,
    ResourceSettlementFailed,
    GrowthFailed,
    GraphEditFailed,
    ExecutorPrepareFailed,
    ResourceMigrationFailed,
    RevisionExhausted,
};

struct LifecycleError {
    LifecycleErrorCode code{LifecycleErrorCode::InvalidConfig};
    std::string reason;
    bool has_cell{false};
    CellId cell{};
};

struct LifecycleCellState {
    CellId cell{};
    LifecycleState state{LifecycleState::Active};
    uint64_t entered_tick{0};
    uint64_t cooldown_until{0};
};

struct LifecycleEvent {
    uint64_t tick{0};
    LifecycleEventKind kind{LifecycleEventKind::BecameActive};
    CellId cell{};
    LifecycleState from{LifecycleState::Active};
    LifecycleState to{LifecycleState::Active};
};

struct LifecycleStepResult {
    std::optional<ExecutionMeasurementView> measurement;
    std::optional<ResourceSettlementReport> settlement;
    std::vector<LifecycleEvent> events;
    std::optional<LifecycleError> error;
    uint64_t tick{0};
    bool executed{false};
    bool extinct{false};

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct LifecycleCreateResult {
    std::unique_ptr<class CellularLifecycleController> controller;
    std::optional<LifecycleError> error;

    bool ok() const { return controller != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class CellularLifecycleController final {
public:
    static LifecycleCreateResult create(
        RuntimeState& runtime,
        std::shared_ptr<CompiledExecutor> executor,
        std::unique_ptr<ResourceLedger> ledger,
        LifecycleConfig config) {
        if (!executor || !ledger) {
            return {nullptr, failure(
                LifecycleErrorCode::BindingMismatch,
                "lifecycle controller requires an executor and resource ledger")};
        }
        if (const auto error = validate_config(config)) return {nullptr, error};
        if (!runtime.bound_to(*executor->plan()) ||
            !runtime.bound_to(*ledger->plan())) {
            return {nullptr, failure(
                LifecycleErrorCode::BindingMismatch,
                "lifecycle controller dependencies are not bound to the runtime")};
        }
        auto candidate = std::unique_ptr<CellularLifecycleController>(
            new CellularLifecycleController(
                runtime, std::move(executor), std::move(ledger), config));
        return {std::move(candidate), std::nullopt};
    }

    LifecycleStepResult step(
        std::span<const double> inputs,
        std::span<const ResourceSupply> supplies = {}) {
        return step_with_cold_boundary(inputs, supplies, {});
    }

    LifecycleStepResult step_with_cold_boundary(
        std::span<const double> inputs,
        std::span<const ResourceSupply> supplies,
        std::function<std::optional<LifecycleError>(
            CellularLifecycleController&)> boundary_action) {
        // Resource settlement commits after native execution. State changes
        // observed here therefore govern the next tick; APOPTOTIC cells are
        // removed only at this step's cold graph boundary before execution.
        LifecycleStepResult result;
        result.events.reserve(states_.size() * 4 + 1);
        result.tick = runtime_->tick() == std::numeric_limits<uint64_t>::max()
            ? runtime_->tick()
            : runtime_->tick() + 1;
        // 本能自愈: 个体形态被生长/编辑推进后, 执行视图随之重备 —
        // 个体永远执行自己的当前形态, 不允许绑死旧 revision。
        if (executor_->plan() != runtime_->plan()) {
            auto reprepared = CompiledExecutor::prepare(runtime_->plan());
            if (!reprepared.ok()) {
                result.error = failure(
                    LifecycleErrorCode::ExecutorPrepareFailed,
                    reprepared.error ? std::string(reprepared.error->reason)
                                     : "executor re-preparation failed");
                return result;
            }
            executor_ = std::move(reprepared.executor);
        }
        if (const auto error = check_binding()) {
            result.error = error;
            return result;
        }
        if (extinct_) {
            result.extinct = true;
            return result;
        }

        auto candidate_states = states_;
        if (const auto error = reconcile(
                candidate_states,
                ledger_->snapshot(),
                result.tick,
                result.events)) {
            result.error = error;
            return result;
        }
        if (const auto error = apply_pending_apoptosis(
                candidate_states, result.events, result.tick)) {
            result.error = error;
            return result;
        }
        if (extinct_) {
            result.extinct = true;
            states_ = std::move(candidate_states);
            return result;
        }
        if (boundary_action) {
            if (const auto error = boundary_action(*this)) {
                states_ = std::move(candidate_states);
                result.error = error;
                return result;
            }
            synchronize_states(candidate_states, result.tick);
            states_ = candidate_states;
        }

        std::vector<ExecutionDisposition> dispositions;
        dispositions.reserve(runtime_->plan()->cells().size());
        for (const auto& cell : runtime_->plan()->cells()) {
            const auto state = find_state(candidate_states, cell.id);
            if (!state.has_value() ||
                candidate_states[*state].state == LifecycleState::Apoptotic ||
                candidate_states[*state].state == LifecycleState::Extinct) {
                result.error = failure(
                    LifecycleErrorCode::InvalidState,
                    "non-executable lifecycle state reached a live graph cell",
                    true, cell.id);
                return result;
            }
            dispositions.push_back(
                candidate_states[*state].state == LifecycleState::Dormant
                    ? ExecutionDisposition::Dormant
                    : ExecutionDisposition::Active);
        }

        const auto native = executor_->step(
            *runtime_, inputs, ExecutionControlView{dispositions});
        if (!native.ok()) {
            result.error = failure(
                LifecycleErrorCode::NativeExecutionFailed,
                native.error ? std::string(native.error->reason)
                             : "native execution failed");
            return result;
        }
        result.executed = true;
        result.measurement = native.measurement;
        const auto settled = ledger_->settle(*runtime_, native.measurement, supplies);
        if (!settled.ok()) {
            result.error = failure(
                LifecycleErrorCode::ResourceSettlementFailed,
                settled.error ? settled.error->reason
                              : "resource settlement failed");
            return result;
        }
        result.settlement = settled.report;
        if (const auto error = reconcile(
                candidate_states,
                ledger_->snapshot(),
                native.measurement.tick,
                result.events)) {
            result.error = error;
            return result;
        }
        states_ = std::move(candidate_states);
        return result;
    }

    LifecycleStepResult step(
        const double* inputs,
        std::size_t input_count,
        std::span<const ResourceSupply> supplies = {}) {
        if (inputs == nullptr && input_count != 0) {
            LifecycleStepResult result;
            result.tick = runtime_->tick() + 1;
            result.error = failure(
                LifecycleErrorCode::NativeExecutionFailed,
                "non-zero input count has a null data pointer");
            return result;
        }
        return step(std::span<const double>(inputs, input_count), supplies);
    }

    LifecycleStepResult step(
        std::span<const double> inputs,
        std::initializer_list<ResourceSupply> supplies) {
        return step(
            inputs,
            std::span<const ResourceSupply>(supplies.begin(), supplies.size()));
    }

    std::optional<LifecycleState> state(CellId cell) const {
        const auto index = find_state(states_, cell);
        if (!index.has_value()) return std::nullopt;
        return states_[*index].state;
    }

    LifecycleState overall_state() const {
        if (extinct_ || states_.empty()) return LifecycleState::Extinct;
        bool has_active = false;
        bool has_apoptotic = false;
        for (const auto& state : states_) {
            has_active |= state.state == LifecycleState::Active;
            has_apoptotic |= state.state == LifecycleState::Apoptotic;
        }
        if (has_active) return LifecycleState::Active;
        if (has_apoptotic) return LifecycleState::Apoptotic;
        return LifecycleState::Dormant;
    }

    std::span<const LifecycleCellState> states() const { return states_; }
    GraphEditHistory edit_history() const { return editor_.history(); }
    const ResourceLedger& ledger() const { return *ledger_; }
    const std::shared_ptr<CompiledExecutor>& executor() const { return executor_; }
    const LifecycleConfig& config() const { return config_; }

    std::optional<LifecycleError> restore_states(
        std::span<const LifecycleCellState> states) {
        if (states.size() != runtime_->plan()->cells().size()) {
            return failure(
                LifecycleErrorCode::InvalidState,
                "lifecycle checkpoint state count does not match the graph");
        }
        std::vector<LifecycleCellState> candidate(states.begin(), states.end());
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            if (candidate[i].cell != runtime_->plan()->cells()[i].id ||
                candidate[i].state == LifecycleState::Extinct) {
                return failure(
                    LifecycleErrorCode::InvalidState,
                    "lifecycle checkpoint state is not bound to the live graph",
                    true, candidate[i].cell);
            }
        }
        states_ = std::move(candidate);
        extinct_ = states_.empty();
        return std::nullopt;
    }

private:
    friend class CellularGrowthController;
    CellularLifecycleController(
        RuntimeState& runtime,
        std::shared_ptr<CompiledExecutor> executor,
        std::unique_ptr<ResourceLedger> ledger,
        LifecycleConfig config)
        : runtime_(&runtime),
          executor_(std::move(executor)),
          ledger_(std::move(ledger)),
          config_(config),
          editor_(runtime) {
        extinct_ = runtime.plan()->cells().empty();
        states_.reserve(runtime.plan()->cells().size());
        for (const auto& cell : runtime.plan()->cells()) {
            states_.push_back(
                LifecycleCellState{cell.id, LifecycleState::Active,
                                   runtime.tick(), runtime.tick()});
        }
    }

    static std::optional<LifecycleError> validate_config(
        const LifecycleConfig& config) {
        if (!std::isfinite(config.apoptotic_resource) ||
            !std::isfinite(config.dormant_enter_resource) ||
            !std::isfinite(config.dormant_exit_resource) ||
            config.apoptotic_resource < 0.0 ||
            config.dormant_enter_resource < 0.0 ||
            config.dormant_exit_resource < 0.0 ||
            config.apoptotic_resource > config.dormant_enter_resource ||
            config.dormant_enter_resource >= config.dormant_exit_resource) {
            return failure(
                LifecycleErrorCode::InvalidConfig,
                "lifecycle thresholds must be finite, non-negative, and ordered as apoptosis <= dormancy entry < wake");
        }
        return std::nullopt;
    }

    static LifecycleError failure(
        LifecycleErrorCode code,
        std::string reason,
        bool has_cell = false,
        CellId cell = CellId{0}) {
        return LifecycleError{code, std::move(reason), has_cell, cell};
    }

    std::optional<LifecycleError> check_binding() const {
        if (!runtime_->bound_to(*executor_->plan()) ||
            !runtime_->bound_to(*ledger_->plan())) {
            return failure(
                LifecycleErrorCode::BindingMismatch,
                "runtime, executor, and resource ledger are not bound to one graph");
        }
        return std::nullopt;
    }

    static std::optional<std::size_t> find_state(
        std::span<const LifecycleCellState> states,
        CellId cell) {
        const auto it = std::lower_bound(
            states.begin(), states.end(), cell,
            [](const LifecycleCellState& state, CellId sought) {
                return state.cell < sought;
            });
        if (it == states.end() || it->cell != cell) return std::nullopt;
        return static_cast<std::size_t>(it - states.begin());
    }

    static bool can_transition(
        const LifecycleCellState& state,
        uint64_t tick,
        const LifecycleConfig& config) {
        return tick >= state.entered_tick &&
               tick - state.entered_tick >= config.minimum_dwell_ticks &&
               tick >= state.cooldown_until;
    }

    static uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
        if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
            return std::numeric_limits<uint64_t>::max();
        }
        return lhs + rhs;
    }

    static LifecycleEventKind event_kind(LifecycleState state) {
        switch (state) {
            case LifecycleState::Active: return LifecycleEventKind::BecameActive;
            case LifecycleState::Dormant: return LifecycleEventKind::BecameDormant;
            case LifecycleState::Apoptotic:
                return LifecycleEventKind::BecameApoptotic;
            case LifecycleState::Extinct:
                return LifecycleEventKind::Extinct;
        }
        return LifecycleEventKind::Extinct;
    }

    std::optional<LifecycleError> reconcile(
        std::vector<LifecycleCellState>& states,
        const ResourceSnapshot& resources,
        uint64_t tick,
        std::vector<LifecycleEvent>& events) const {
        for (auto& state : states) {
            if (state.state == LifecycleState::Apoptotic ||
                state.state == LifecycleState::Extinct) {
                continue;
            }
            const auto* resource = resources.cell(state.cell);
            if (!resource) {
                return failure(
                    LifecycleErrorCode::BindingMismatch,
                    "lifecycle state has no resource record",
                    true, state.cell);
            }
            LifecycleState next = state.state;
            if (resource->energy <= config_.apoptotic_resource) {
                next = LifecycleState::Apoptotic;
            } else if (state.state == LifecycleState::Active &&
                       resource->energy <= config_.dormant_enter_resource &&
                       can_transition(state, tick, config_)) {
                next = LifecycleState::Dormant;
            } else if (state.state == LifecycleState::Dormant &&
                       resource->energy >= config_.dormant_exit_resource &&
                       can_transition(state, tick, config_)) {
                next = LifecycleState::Active;
            }
            if (next == state.state) continue;
            const LifecycleState previous = state.state;
            state.state = next;
            state.entered_tick = tick;
            state.cooldown_until = saturating_add(tick, config_.cooldown_ticks);
            events.push_back(
                LifecycleEvent{tick, event_kind(next), state.cell, previous, next});
        }
        return std::nullopt;
    }

    void synchronize_states(
        std::vector<LifecycleCellState>& candidate_states,
        uint64_t tick) const {
        std::vector<LifecycleCellState> synchronized;
        synchronized.reserve(runtime_->plan()->cells().size());
        for (const auto& cell : runtime_->plan()->cells()) {
            const auto found = find_state(candidate_states, cell.id);
            if (found.has_value()) {
                synchronized.push_back(candidate_states[*found]);
            } else {
                synchronized.push_back(
                    LifecycleCellState{
                        cell.id, LifecycleState::Active, tick, tick});
            }
        }
        candidate_states.swap(synchronized);
    }

    std::optional<LifecycleError> apply_pending_apoptosis(
        std::vector<LifecycleCellState>& candidate_states,
        std::vector<LifecycleEvent>& events,
        uint64_t tick) {
        std::vector<CellId> dying;
        for (const auto& state : candidate_states) {
            if (state.state == LifecycleState::Apoptotic) dying.push_back(state.cell);
        }
        if (dying.empty()) return std::nullopt;
        if (runtime_->revision().value == std::numeric_limits<uint64_t>::max()) {
            return failure(
                LifecycleErrorCode::RevisionExhausted,
                "apoptotic removal cannot advance graph revision");
        }

        std::vector<GraphEditEvent> edit_events;
        edit_events.reserve(dying.size());
        uint64_t event_id = next_event_id_;
        for (const auto cell : dying) {
            edit_events.push_back(
                GraphEditEvent{event_id++, RemoveCellAction{cell}});
        }

        auto probe_result = runtime_->fork_probe();
        if (!probe_result.ok()) {
            return failure(
                LifecycleErrorCode::GraphEditFailed,
                "runtime probe for apoptotic removal could not be created");
        }
        GraphEditor probe_editor = editor_;
        const auto probe_edit =
            probe_editor.apply(*probe_result.runtime, edit_events);
        if (!probe_edit.ok()) {
            return failure(
                LifecycleErrorCode::GraphEditFailed,
                probe_edit.error ? probe_edit.error->reason
                                 : "apoptotic graph edit failed during preflight");
        }
        auto probe_ledger = ledger_->fork_probe();
        const auto probe_rebind =
            probe_ledger->rebind(*probe_result.runtime, probe_edit.graph);
        if (!probe_rebind.ok()) {
            return failure(
                LifecycleErrorCode::ResourceMigrationFailed,
                probe_rebind.error ? probe_rebind.error->reason
                                   : "resource migration failed during preflight");
        }
        const auto prepared = CompiledExecutor::prepare(probe_edit.graph);
        if (!prepared.ok()) {
            return failure(
                LifecycleErrorCode::ExecutorPrepareFailed,
                prepared.error ? std::string(prepared.error->reason)
                                : "executor preparation failed during preflight");
        }

        const auto actual_edit = editor_.apply(*runtime_, edit_events);
        if (!actual_edit.ok()) {
            return failure(
                LifecycleErrorCode::GraphEditFailed,
                actual_edit.error ? actual_edit.error->reason
                                  : "apoptotic graph edit failed");
        }
        ledger_->adopt_from(std::move(*probe_ledger));
        executor_ = prepared.executor;
        next_event_id_ = event_id;

        for (const auto cell : dying) {
            events.push_back(
                LifecycleEvent{
                    tick, LifecycleEventKind::CellRemoved, cell,
                    LifecycleState::Apoptotic, LifecycleState::Extinct});
        }
        candidate_states.erase(
            std::remove_if(
                candidate_states.begin(), candidate_states.end(),
                [&](const LifecycleCellState& state) {
                    return std::binary_search(dying.begin(), dying.end(), state.cell);
                }),
            candidate_states.end());
        if (candidate_states.empty()) {
            extinct_ = true;
            events.push_back(
                LifecycleEvent{
                    tick, LifecycleEventKind::Extinct, CellId{0},
                    LifecycleState::Apoptotic, LifecycleState::Extinct});
        }
        return std::nullopt;
    }

    RuntimeState* runtime_{nullptr};
    std::shared_ptr<CompiledExecutor> executor_;
    std::unique_ptr<ResourceLedger> ledger_;
    LifecycleConfig config_{};
    GraphEditor editor_;
    std::vector<LifecycleCellState> states_;
    uint64_t next_event_id_{1};
    bool extinct_{false};
};

}  // namespace kun::core
