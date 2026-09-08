#pragma once

#include "kun/cellular/core/lifecycle_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace kun::core {

struct GrowthFunding {
    std::optional<CellId> sponsor;
    ResourceCompartmentId compartment{};
};

struct GrowthCellBirthProposal {
    uint64_t proposal_id{0};
    CellBirth birth{};
    std::vector<EdgeBirth> incoming_edges;
    std::vector<EdgeBirth> outgoing_edges;
    GrowthFunding funding{};
};

struct GrowthSplitProposal {
    uint64_t proposal_id{0};
    SplitEdgeAction split{};
    GrowthFunding funding{};
};

struct GrowthSynapseProposal {
    uint64_t proposal_id{0};
    EdgeBirth edge{};
    GrowthFunding funding{};
    double cost{0.0};
};

struct GrowthReclaimProposal {
    uint64_t proposal_id{0};
    EdgeId edge{};
};

using GrowthProposal = std::variant<
    GrowthCellBirthProposal,
    GrowthSplitProposal,
    GrowthSynapseProposal,
    GrowthReclaimProposal>;

struct GrowthConfig {
    double cell_birth_cost{0.0};
    double synapse_birth_cost{0.0};
    double initial_energy{0.0};
    double initial_capacity{0.0};
    double initial_structural_reserve{0.0};
};

enum class GrowthErrorCode : uint8_t {
    InvalidConfig,
    InvalidProposal,
    DuplicateProposal,
    InsufficientResource,
    BindingMismatch,
    GraphEditFailed,
    ResourceMigrationFailed,
    ExecutorPrepareFailed,
    Extinct,
    RevisionExhausted,
};

struct GrowthError {
    GrowthErrorCode code{GrowthErrorCode::InvalidConfig};
    std::string reason;
    uint64_t proposal_id{0};
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
};

struct GrowthSubmitResult {
    std::optional<GrowthError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

enum class GrowthEventKind : uint8_t {
    Committed,
    Rejected,
    ReservationReleased,
};

struct GrowthEvent {
    uint64_t tick{0};
    GrowthEventKind kind{GrowthEventKind::Committed};
    uint64_t proposal_id{0};
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
    double paid_cost{0.0};
    std::string reason;
};

struct GrowthStepResult {
    LifecycleStepResult lifecycle;
    std::vector<GrowthEvent> growth_events;
    std::optional<ResourceGrowthReport> growth_report;
    std::optional<GrowthError> growth_error;

    bool ok() const {
        return lifecycle.ok() && !growth_error.has_value();
    }
    explicit operator bool() const { return ok(); }
};

struct GrowthCreateResult {
    std::unique_ptr<class CellularGrowthController> controller;
    std::optional<GrowthError> error;

    bool ok() const { return controller != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

class CellularGrowthController final {
public:
    static GrowthCreateResult create(
        CellularLifecycleController& lifecycle,
        GrowthConfig config) {
        if (const auto error = validate_config(config)) {
            return {nullptr, error};
        }
        auto candidate = std::unique_ptr<CellularGrowthController>(
            new CellularGrowthController(nullptr, lifecycle, config));
        return {std::move(candidate), std::nullopt};
    }

    static GrowthCreateResult create(
        std::unique_ptr<CellularLifecycleController> lifecycle,
        GrowthConfig config) {
        if (!lifecycle) {
            return {nullptr, failure(
                GrowthErrorCode::BindingMismatch,
                "growth controller requires an owned lifecycle controller")};
        }
        if (const auto error = validate_config(config)) {
            return {nullptr, error};
        }
        auto* lifecycle_ptr = lifecycle.get();
        auto candidate = std::unique_ptr<CellularGrowthController>(
            new CellularGrowthController(
                std::move(lifecycle), *lifecycle_ptr, config));
        return {std::move(candidate), std::nullopt};
    }

    GrowthSubmitResult submit(const GrowthProposal& proposal) {
        const auto proposal_id = id_of(proposal);
        if (std::any_of(
                pending_.begin(), pending_.end(),
                [&](const GrowthProposal& existing) {
                    return id_of(existing) == proposal_id;
                })) {
            return GrowthSubmitResult{failure(
                GrowthErrorCode::DuplicateProposal,
                "growth proposal ID is already pending",
                proposal_id)};
        }
        pending_.push_back(proposal);
        return {};
    }

    GrowthSubmitResult submit(std::span<const GrowthProposal> proposals) {
        auto candidate = pending_;
        for (const auto& proposal : proposals) {
            const auto proposal_id = id_of(proposal);
            if (std::any_of(
                    candidate.begin(), candidate.end(),
                    [&](const GrowthProposal& existing) {
                        return id_of(existing) == proposal_id;
                    })) {
                return GrowthSubmitResult{failure(
                    GrowthErrorCode::DuplicateProposal,
                    "growth proposal ID is already pending",
                    proposal_id)};
            }
            candidate.push_back(proposal);
        }
        pending_ = std::move(candidate);
        return {};
    }

    GrowthSubmitResult submit(
        std::initializer_list<GrowthProposal> proposals) {
        return submit(
            std::span<const GrowthProposal>(
                proposals.begin(), proposals.size()));
    }

    GrowthStepResult step(
        std::span<const double> inputs,
        std::span<const ResourceSupply> supplies = {}) {
        growth_events_.clear();
        growth_report_.reset();
        growth_error_.reset();
        LifecycleStepResult lifecycle_result;
        if (pending_.empty()) {
            lifecycle_result = lifecycle_->step(inputs, supplies);
        } else {
            lifecycle_result = lifecycle_->step_with_cold_boundary(
                inputs,
                supplies,
                [this](CellularLifecycleController& owner) {
                    return commit_pending(owner);
                });
        }
        if (lifecycle_result.extinct && !pending_.empty()) {
            reject_pending(
                lifecycle_result.tick,
                GrowthErrorCode::Extinct,
                "growth cannot revive an extinct lifecycle");
        }
        return GrowthStepResult{
            std::move(lifecycle_result),
            std::move(growth_events_),
            std::move(growth_report_),
            std::move(growth_error_)};
    }

    GrowthStepResult step(
        const double* inputs,
        std::size_t input_count,
        std::span<const ResourceSupply> supplies = {}) {
        if (inputs == nullptr && input_count != 0) {
            GrowthStepResult result;
            result.growth_error = GrowthError{
                GrowthErrorCode::InvalidProposal,
                "non-zero input count has a null data pointer"};
            return result;
        }
        return step(std::span<const double>(inputs, input_count), supplies);
    }

    GrowthStepResult step(
        std::span<const double> inputs,
        std::initializer_list<ResourceSupply> supplies) {
        return step(
            inputs,
            std::span<const ResourceSupply>(supplies.begin(), supplies.size()));
    }

    CellularLifecycleController& lifecycle() { return *lifecycle_; }
    const CellularLifecycleController& lifecycle() const { return *lifecycle_; }

private:
    struct EditBundle {
        std::vector<GraphEditEvent> events;
        std::vector<ResourceCellBirth> births;
        std::vector<ResourceGrowthCharge> charges;
        std::vector<uint64_t> proposal_ids;
    };

    CellularGrowthController(
        std::unique_ptr<CellularLifecycleController> lifecycle_owner,
        CellularLifecycleController& lifecycle,
        GrowthConfig config)
        : lifecycle_owner_(std::move(lifecycle_owner)),
          lifecycle_(&lifecycle),
          config_(config) {}

    static std::optional<GrowthError> validate_config(
        const GrowthConfig& config) {
        if (!finite_nonnegative(config.cell_birth_cost) ||
            !finite_nonnegative(config.synapse_birth_cost) ||
            !finite_nonnegative(config.initial_energy) ||
            !finite_nonnegative(config.initial_capacity) ||
            !finite_nonnegative(config.initial_structural_reserve) ||
            config.initial_energy > config.initial_capacity) {
            return GrowthError{
                GrowthErrorCode::InvalidConfig,
                "growth costs, endowments, and capacity must be finite and non-negative"};
        }
        return std::nullopt;
    }

    static bool finite_nonnegative(double value) {
        return std::isfinite(value) && value >= 0.0;
    }

    static bool nonzero_finite(double value) {
        return std::isfinite(value) && value != 0.0;
    }

    static uint64_t id_of(const GrowthProposal& proposal) {
        return std::visit(
            [](const auto& value) { return value.proposal_id; }, proposal);
    }

    static GrowthError failure(
        GrowthErrorCode code,
        std::string reason,
        uint64_t proposal_id = 0,
        bool has_cell = false,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0}) {
        return GrowthError{
            code, std::move(reason), proposal_id,
            has_cell, cell, has_edge, edge};
    }

    std::optional<LifecycleError> commit_pending(
        CellularLifecycleController& owner) {
        if (pending_.empty()) return std::nullopt;
        if (owner.extinct_) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::Extinct,
                "growth cannot revive an extinct lifecycle");
        }
        if (owner.runtime_->revision().value == std::numeric_limits<uint64_t>::max()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::RevisionExhausted,
                "growth cannot advance graph revision");
        }

        auto bundle = build_bundle(*owner.runtime_);
        if (!bundle.has_value()) return boundary_failure();

        auto probe_result = owner.runtime_->fork_probe();
        if (!probe_result.ok()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::GraphEditFailed,
                "growth runtime probe could not be created");
        }
        GraphEditor probe_editor = owner.editor_;
        const auto probe_edit =
            probe_editor.apply(*probe_result.runtime, bundle->events);
        if (!probe_edit.ok()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::GraphEditFailed,
                probe_edit.error ? probe_edit.error->reason
                                 : "growth graph edit failed during preflight");
        }
        auto probe_ledger = owner.ledger_->fork_probe();
        const auto growth = probe_ledger->grow(
            *probe_result.runtime,
            probe_edit.graph,
            bundle->births,
            bundle->charges);
        if (!growth.ok()) {
            return reject_pending(
                owner.runtime_->tick(),
                growth.error && growth.error->code ==
                        ResourceErrorCode::InsufficientResource
                    ? GrowthErrorCode::InsufficientResource
                    : GrowthErrorCode::ResourceMigrationFailed,
                growth.error ? growth.error->reason
                             : "resource growth failed during preflight");
        }
        const auto prepared = CompiledExecutor::prepare(probe_edit.graph);
        if (!prepared.ok()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::ExecutorPrepareFailed,
                prepared.error ? std::string(prepared.error->reason)
                                : "growth executor preparation failed");
        }

        const auto actual_edit =
            owner.editor_.apply(*owner.runtime_, bundle->events);
        if (!actual_edit.ok()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::GraphEditFailed,
                actual_edit.error ? actual_edit.error->reason
                                  : "growth graph edit failed");
        }
        if (bundle->events.empty()) {
            return reject_pending(
                owner.runtime_->tick(),
                GrowthErrorCode::InvalidProposal,
                "growth transaction contains no graph edits");
        }
        owner.ledger_->adopt_from(std::move(*probe_ledger));
        owner.executor_ = prepared.executor;
        growth_report_ = growth.report;
        const uint64_t tick = owner.runtime_->tick() + 1;
        for (const auto& proposal : pending_) {
            GrowthEvent event;
            event.tick = tick;
            event.kind = GrowthEventKind::Committed;
            event.proposal_id = id_of(proposal);
            std::visit(
                [&](const auto& value) {
                    using Value = std::decay_t<decltype(value)>;
                    if constexpr (
                        std::is_same_v<Value, GrowthCellBirthProposal>) {
                        event.has_cell = true;
                        event.cell = value.birth.id;
                    } else if constexpr (
                        std::is_same_v<Value, GrowthSplitProposal>) {
                        event.has_cell = true;
                        event.cell = value.split.inserted.id;
                    } else if constexpr (
                        std::is_same_v<Value, GrowthSynapseProposal>) {
                        event.has_edge = true;
                        event.edge = value.edge.id;
                    } else {
                        event.has_edge = true;
                        event.edge = value.edge;
                    }
                },
                proposal);
            growth_events_.push_back(std::move(event));
        }
        pending_.clear();
        next_event_id_ = bundle->events.back().event_id + 1;
        return std::nullopt;
    }

    std::optional<EditBundle> build_bundle(const RuntimeState& runtime) {
        auto ordered = pending_;
        std::sort(
            ordered.begin(), ordered.end(),
            [](const GrowthProposal& lhs, const GrowthProposal& rhs) {
                return id_of(lhs) < id_of(rhs);
            });
        EditBundle bundle;
        uint64_t event_id = next_event_id_;
        for (const auto& proposal : ordered) {
            const auto proposal_id = id_of(proposal);
            bundle.proposal_ids.push_back(proposal_id);
            const auto error = append_proposal(
                runtime, proposal, proposal_id, event_id, bundle);
            if (error.has_value()) {
                reject_pending(runtime.tick(), error->code, error->reason);
                return std::nullopt;
            }
        }
        return bundle;
    }

    std::optional<GrowthError> append_proposal(
        const RuntimeState& runtime,
        const GrowthProposal& proposal,
        uint64_t proposal_id,
        uint64_t& event_id,
        EditBundle& bundle) const {
        (void)runtime;
        return std::visit(
            [&](const auto& value) -> std::optional<GrowthError> {
                using Value = std::decay_t<decltype(value)>;
                if constexpr (
                    std::is_same_v<Value, GrowthCellBirthProposal>) {
                    if (value.incoming_edges.empty() ||
                        value.outgoing_edges.empty()) {
                        return failure(
                            GrowthErrorCode::InvalidProposal,
                            "a birth must have incoming and outgoing causal edges",
                            proposal_id, true, value.birth.id);
                    }
                    for (const auto& edge : value.incoming_edges) {
                        if (edge.target != value.birth.id ||
                            edge.source == value.birth.id ||
                            !nonzero_finite(edge.initial_weight)) {
                            return failure(
                                GrowthErrorCode::InvalidProposal,
                                "birth incoming edge is not an explainable causal edge",
                                proposal_id, true, value.birth.id,
                                true, edge.id);
                        }
                        bundle.events.push_back(
                            GraphEditEvent{
                                event_id++, AddEdgeAction{edge}});
                    }
                    for (const auto& edge : value.outgoing_edges) {
                        if (edge.source != value.birth.id ||
                            edge.target == value.birth.id ||
                            !nonzero_finite(edge.initial_weight)) {
                            return failure(
                                GrowthErrorCode::InvalidProposal,
                                "birth outgoing edge is not an explainable causal edge",
                                proposal_id, true, value.birth.id,
                                true, edge.id);
                        }
                        bundle.events.push_back(
                            GraphEditEvent{
                                event_id++, AddEdgeAction{edge}});
                    }
                    bundle.events.push_back(
                        GraphEditEvent{
                            event_id++, AddCellAction{value.birth}});
                    bundle.births.push_back(
                        ResourceCellBirth{
                            value.birth.id,
                            value.funding.compartment,
                            config_.initial_energy,
                            config_.initial_capacity,
                            config_.initial_structural_reserve,
                            config_.cell_birth_cost,
                            value.funding.sponsor});
                    return std::nullopt;
                } else if constexpr (
                    std::is_same_v<Value, GrowthSplitProposal>) {
                    const auto& split = value.split;
                    if (!nonzero_finite(split.source_weight) ||
                        !nonzero_finite(split.target_weight)) {
                        return failure(
                            GrowthErrorCode::InvalidProposal,
                            "split birth weights must be finite and non-zero",
                            proposal_id, true, split.inserted.id,
                            true, split.edge);
                    }
                    bundle.events.push_back(
                        GraphEditEvent{event_id++, split});
                    bundle.births.push_back(
                        ResourceCellBirth{
                            split.inserted.id,
                            value.funding.compartment,
                            config_.initial_energy,
                            config_.initial_capacity,
                            config_.initial_structural_reserve,
                            config_.cell_birth_cost,
                            value.funding.sponsor});
                    return std::nullopt;
                } else if constexpr (
                    std::is_same_v<Value, GrowthSynapseProposal>) {
                    const double cost = value.cost == 0.0
                        ? config_.synapse_birth_cost
                        : value.cost;
                    if (!nonzero_finite(value.edge.initial_weight) ||
                        !finite_nonnegative(cost)) {
                        return failure(
                            GrowthErrorCode::InvalidProposal,
                            "synapse edge weight and cost must be finite and valid",
                            proposal_id, false, CellId{0},
                            true, value.edge.id);
                    }
                    bundle.events.push_back(
                        GraphEditEvent{
                            event_id++, AddEdgeAction{value.edge}});
                    bundle.charges.push_back(
                        ResourceGrowthCharge{
                            value.funding.compartment,
                            cost,
                            value.funding.sponsor,
                            cost});
                    return std::nullopt;
                } else {
                    bundle.events.push_back(
                        GraphEditEvent{
                            event_id++, RemoveEdgeAction{value.edge}});
                    return std::nullopt;
                }
            },
            proposal);
    }

    std::optional<LifecycleError> reject_pending(
        uint64_t tick,
        GrowthErrorCode code,
        std::string reason) {
        const auto pending = std::move(pending_);
        pending_.clear();
        growth_error_ = failure(code, reason);
        for (const auto& proposal : pending) {
            const auto proposal_id = id_of(proposal);
            growth_events_.push_back(
                GrowthEvent{
                    tick,
                    GrowthEventKind::Rejected,
                    proposal_id,
                    false, CellId{0}, false, EdgeId{0},
                    0.0,
                    reason});
            growth_events_.push_back(
                GrowthEvent{
                    tick,
                    GrowthEventKind::ReservationReleased,
                    proposal_id,
                    false, CellId{0}, false, EdgeId{0},
                    0.0,
                    "growth reservation released"});
        }
        return LifecycleError{
            LifecycleErrorCode::GrowthFailed,
            growth_error_ ? growth_error_->reason : "growth proposal rejected"};
    }

    std::optional<LifecycleError> boundary_failure() const {
        return LifecycleError{
            LifecycleErrorCode::GrowthFailed,
            growth_error_ ? growth_error_->reason : "growth proposal rejected"};
    }

    std::unique_ptr<CellularLifecycleController> lifecycle_owner_;
    CellularLifecycleController* lifecycle_{nullptr};
    GrowthConfig config_{};
    std::vector<GrowthProposal> pending_;
    std::vector<GrowthEvent> growth_events_;
    std::optional<ResourceGrowthReport> growth_report_;
    std::optional<GrowthError> growth_error_;
    uint64_t next_event_id_{1};
};

}  // namespace kun::core
