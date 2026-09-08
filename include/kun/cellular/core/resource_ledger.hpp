#pragma once

#include "kun/cellular/core/execution_measurement.hpp"
#include "kun/cellular/core/runtime_state.hpp"

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
#include <string_view>
#include <utility>
#include <vector>

namespace kun::core {

struct ResourceCompartmentId {
    uint64_t value{0};
    explicit constexpr ResourceCompartmentId(uint64_t v = 0) : value(v) {}
    friend constexpr bool operator==(ResourceCompartmentId, ResourceCompartmentId) = default;
    friend constexpr auto operator<=>(ResourceCompartmentId, ResourceCompartmentId) = default;
};

struct ResourceCellInitial {
    CellId cell{};
    ResourceCompartmentId compartment{};
    double initial_energy{0.0};
    double capacity{0.0};
    double structural_reserve{0.0};
};

struct ResourceCompartmentInitial {
    ResourceCompartmentId compartment{};
    double initial_environmental_resource{0.0};
};

struct ResourceSupply {
    ResourceCompartmentId compartment{};
    double amount{0.0};
};

struct ResourceExecutionCost {
    CellType type{CellType::OP_EMA};
    double cost{0.0};
};

struct ResourceCellBirth {
    CellId cell{};
    ResourceCompartmentId compartment{};
    double initial_energy{0.0};
    double capacity{0.0};
    double structural_reserve{0.0};
    double birth_cost{0.0};
    std::optional<CellId> sponsor;
};

struct ResourceGrowthCharge {
    ResourceCompartmentId compartment{};
    double amount{0.0};
    std::optional<CellId> sponsor;
    double dissipated{0.0};
};

struct ResourceLedgerConfig {
    // All values are dimensionless model units; no physical energy or CPU
    // interpretation is implied. A settlement absorbs before charging costs.
    double dt{1.0};
    double maintenance_cost{0.0};
    double absorption_rate{0.0};
    double activity_scale{1.0};
    double activity_cost{0.0};
    double transmission_scale{1.0};
    double transmission_cost{0.0};
    std::vector<ResourceExecutionCost> execution_costs;
};

enum class ResourceErrorCode : uint8_t {
    InvalidConfig,
    InvalidCell,
    InvalidCompartment,
    DuplicateId,
    MissingId,
    BindingMismatch,
    InvalidMeasurement,
    MissingCoverage,
    DuplicateMeasurement,
    StaleMeasurement,
    SkippedMeasurement,
    Overflow,
    EpisodeMismatch,
    UnsupportedOperation,
    MigrationInvalidInput,
    MigrationIdentityMismatch,
    MigrationRevisionMismatch,
    InsufficientResource,
};

struct ResourceError {
    ResourceErrorCode code{ResourceErrorCode::InvalidConfig};
    std::string reason;
    bool has_cell{false};
    CellId cell{};
    bool has_edge{false};
    EdgeId edge{};
    bool has_compartment{false};
    ResourceCompartmentId compartment{};
};

struct ResourceCellState {
    CellId cell{};
    ResourceCompartmentId compartment{};
    double capacity{0.0};
    double energy{0.0};
    double structural_reserve{0.0};
    double initial_structural_reserve{0.0};
    double age{0.0};
    double cumulative_activity{0.0};
    double cumulative_transmission{0.0};
    double cumulative_execution_cost{0.0};
    double cumulative_maintenance_cost{0.0};
    double cumulative_activity_cost{0.0};
    double cumulative_transmission_cost{0.0};
    double cumulative_paid_cost{0.0};
    double cumulative_unpaid_cost{0.0};
    double cumulative_growth_cost{0.0};
    uint64_t execution_count{0};
    bool exhausted{false};
    // P2 records the default active birth state only. It does not control
    // native execution; sleep/death transitions belong to the P3 owner.
    bool active{true};

    friend bool operator==(const ResourceCellState&, const ResourceCellState&) = default;
};

struct ResourceCompartmentState {
    ResourceCompartmentId compartment{};
    double environmental_resource{0.0};

    friend bool operator==(const ResourceCompartmentState&, const ResourceCompartmentState&) =
        default;
};

struct ResourceTotals {
    double initial_total{0.0};
    double cumulative_external_injection{0.0};
    double cumulative_dissipation{0.0};
    double cumulative_export{0.0};
    double cumulative_paid_cost{0.0};
    double cumulative_unpaid_cost{0.0};
    double cumulative_growth_cost{0.0};

    friend bool operator==(const ResourceTotals&, const ResourceTotals&) = default;
};

class ResourceSnapshot final {
public:
    static ResourceSnapshot from_parts(
        std::vector<ResourceCellState> cells,
        std::vector<ResourceCompartmentState> compartments,
        ResourceTotals totals,
        uint64_t last_measurement_tick,
        GraphIdentity identity,
        GraphRevision revision,
        SemanticProfile profile) {
        ResourceSnapshot result;
        result.cells_ = std::move(cells);
        result.compartments_ = std::move(compartments);
        result.totals_ = totals;
        result.last_measurement_tick_ = last_measurement_tick;
        result.identity_ = identity;
        result.revision_ = revision;
        result.profile_ = profile;
        return result;
    }

    std::span<const ResourceCellState> cells() const { return cells_; }
    std::span<const ResourceCompartmentState> compartments() const { return compartments_; }
    const ResourceCellState* cell(CellId id) const {
        const auto it = std::lower_bound(
            cells_.begin(), cells_.end(), id,
            [](const ResourceCellState& state, CellId sought) {
                return state.cell < sought;
            });
        return it == cells_.end() || it->cell != id ? nullptr : &*it;
    }
    const ResourceCompartmentState* compartment(ResourceCompartmentId id) const {
        const auto it = std::lower_bound(
            compartments_.begin(), compartments_.end(), id,
            [](const ResourceCompartmentState& state, ResourceCompartmentId sought) {
                return state.compartment < sought;
            });
        return it == compartments_.end() || it->compartment != id ? nullptr : &*it;
    }
    const ResourceTotals& totals() const { return totals_; }
    double cumulative_paid_cost() const { return totals_.cumulative_paid_cost; }
    double cumulative_dissipation() const { return totals_.cumulative_dissipation; }
    double cumulative_growth_cost() const {
        return totals_.cumulative_growth_cost;
    }
    uint64_t last_measurement_tick() const { return last_measurement_tick_; }
    GraphIdentity identity() const { return identity_; }
    GraphRevision revision() const { return revision_; }
    SemanticProfile profile() const { return profile_; }

    friend bool operator==(const ResourceSnapshot&, const ResourceSnapshot&) = default;

private:
    friend class ResourceLedger;

    std::vector<ResourceCellState> cells_;
    std::vector<ResourceCompartmentState> compartments_;
    ResourceTotals totals_{};
    uint64_t last_measurement_tick_{0};
    GraphIdentity identity_{};
    GraphRevision revision_{};
    SemanticProfile profile_{SemanticProfile::LegacyCompatible};
};

struct ResourceCellSettlement {
    CellId cell{};
    double absorbed{0.0};
    double activity{0.0};
    double execution_cost{0.0};
    double maintenance_cost{0.0};
    double activity_cost{0.0};
    double transmission_cost{0.0};
    double requested_cost{0.0};
    double paid_cost{0.0};
    double unpaid_cost{0.0};
    bool executed{false};
    bool exhausted{false};
};

struct ResourceSettlementReport {
    uint64_t tick{0};
    double injected{0.0};
    double absorbed{0.0};
    // Requested terms are pre-payment demands; paid terms are what entered
    // dissipation. Unpaid cost creates no debt and no energy.
    double requested_cost{0.0};
    double paid_cost{0.0};
    double unpaid_cost{0.0};
    double execution_cost{0.0};
    double maintenance_cost{0.0};
    double activity_cost{0.0};
    double transmission_cost{0.0};
    double cumulative_dissipation{0.0};
    double cumulative_export{0.0};
    double conservation_residual{0.0};
    std::size_t exhausted_cells{0};
    std::vector<ResourceCellSettlement> cells;
};

struct ResourceAttachResult {
    std::unique_ptr<class ResourceLedger> ledger;
    std::optional<ResourceError> error;
    bool ok() const { return ledger != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct ResourceSettlementResult {
    ResourceSettlementReport report;
    std::optional<ResourceError> error;
    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct ResourceGrowthPayment {
    std::optional<CellId> sponsor;
    ResourceCompartmentId compartment{};
    double requested{0.0};
    double from_structural_reserve{0.0};
    double from_energy{0.0};
    double from_environment{0.0};
    double paid{0.0};
};

struct ResourceGrowthReport {
    double requested_cost{0.0};
    double paid_cost{0.0};
    double cumulative_growth_cost{0.0};
    double conservation_residual{0.0};
    std::vector<ResourceGrowthPayment> payments;
};

struct ResourceGrowthResult {
    ResourceGrowthReport report;
    std::optional<ResourceError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

struct ResourceRebindResult {
    std::optional<ResourceError> error;

    bool ok() const { return !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

namespace resource_detail {

inline ResourceError failure(
    ResourceErrorCode code,
    std::string reason,
    bool has_cell = false,
    CellId cell = CellId{0},
    bool has_edge = false,
    EdgeId edge = EdgeId{0},
    bool has_compartment = false,
    ResourceCompartmentId compartment = ResourceCompartmentId{0}) {
    return ResourceError{
        code, std::move(reason), has_cell, cell, has_edge, edge,
        has_compartment, compartment};
}

inline bool finite_nonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

inline bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

inline bool checked_add(double lhs, double rhs, double& result) {
    result = lhs + rhs;
    return std::isfinite(result);
}

inline bool checked_mul(double lhs, double rhs, double& result) {
    result = lhs * rhs;
    return std::isfinite(result);
}

inline bool checked_increment(uint64_t value, uint64_t& result) {
    if (value == std::numeric_limits<uint64_t>::max()) return false;
    result = value + 1;
    return true;
}

inline double bounded_magnitude(double value, double scale) {
    const double magnitude = std::abs(value);
    if (magnitude >= scale) return 1.0;
    return magnitude / scale;
}

}  // namespace resource_detail

class CellularLifecycleController;
class CellularGrowthController;

class ResourceLedger final {
public:
    static ResourceAttachResult attach(
        const RuntimeState& runtime,
        ResourceLedgerConfig config,
        std::span<const ResourceCellInitial> cells,
        std::span<const ResourceCompartmentInitial> compartments) {
        if (!runtime.plan()) {
            return {nullptr, resource_detail::failure(
                ResourceErrorCode::BindingMismatch,
                "resource ledger requires a bound compiled runtime")};
        }
        if (const auto error = validate_config(config)) {
            return {nullptr, error};
        }
        auto candidate = std::unique_ptr<ResourceLedger>(
            new ResourceLedger(runtime.plan(), std::move(config)));
        if (const auto error =
                candidate->initialize(runtime, cells, compartments)) {
            return {nullptr, error};
        }
        return {std::move(candidate), std::nullopt};
    }

    static ResourceAttachResult attach(
        const RuntimeState& runtime,
        ResourceLedgerConfig config,
        std::initializer_list<ResourceCellInitial> cells,
        std::initializer_list<ResourceCompartmentInitial> compartments) {
        return attach(
            runtime, std::move(config),
            std::span<const ResourceCellInitial>(cells.begin(), cells.size()),
            std::span<const ResourceCompartmentInitial>(
                compartments.begin(), compartments.size()));
    }

    static ResourceAttachResult from_snapshot(
        const RuntimeState& runtime,
        ResourceLedgerConfig config,
        const ResourceSnapshot& snapshot) {
        if (!runtime.plan() ||
            snapshot.identity_ != runtime.identity() ||
            snapshot.revision_ != runtime.revision() ||
            snapshot.profile_ != runtime.profile()) {
            return {nullptr, resource_detail::failure(
                ResourceErrorCode::BindingMismatch,
                "resource checkpoint identity does not match the runtime")};
        }
        if (const auto error = validate_config(config)) {
            return {nullptr, error};
        }
        if (snapshot.cells_.size() != runtime.plan()->cells().size()) {
            return {nullptr, resource_detail::failure(
                ResourceErrorCode::MissingCoverage,
                "resource checkpoint cell coverage does not match the graph")};
        }
        auto candidate = std::unique_ptr<ResourceLedger>(
            new ResourceLedger(runtime.plan(), std::move(config)));
        candidate->cells_ = snapshot.cells_;
        candidate->compartments_ = snapshot.compartments_;
        candidate->totals_ = snapshot.totals_;
        candidate->last_measurement_tick_ = snapshot.last_measurement_tick_;
        if (!runtime.bound_to(*candidate->plan_)) {
            return {nullptr, resource_detail::failure(
                ResourceErrorCode::BindingMismatch,
                "resource checkpoint runtime binding is invalid")};
        }
        for (std::size_t i = 0; i < candidate->cells_.size(); ++i) {
            if (candidate->cells_[i].cell != runtime.plan()->cells()[i].id ||
                !resource_detail::finite_nonnegative(
                    candidate->cells_[i].energy) ||
                !resource_detail::finite_nonnegative(
                    candidate->cells_[i].structural_reserve) ||
                !resource_detail::finite_nonnegative(
                    candidate->cells_[i].capacity)) {
                return {nullptr, resource_detail::failure(
                    ResourceErrorCode::InvalidCell,
                    "resource checkpoint contains invalid cell state",
                    true, candidate->cells_[i].cell)};
            }
        }
        if (!resource_detail::finite_nonnegative(snapshot.totals_.initial_total) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_external_injection) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_dissipation) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_export) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_paid_cost) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_unpaid_cost) ||
            !resource_detail::finite_nonnegative(
                snapshot.totals_.cumulative_growth_cost)) {
            return {nullptr, resource_detail::failure(
                ResourceErrorCode::InvalidMeasurement,
                "resource checkpoint contains invalid lifetime totals")};
        }
        for (const auto& compartment : candidate->compartments_) {
            if (!resource_detail::finite_nonnegative(
                    compartment.environmental_resource)) {
                return {nullptr, resource_detail::failure(
                    ResourceErrorCode::InvalidCompartment,
                    "resource checkpoint contains invalid compartment state",
                    false, CellId{0}, false, EdgeId{0}, true,
                    compartment.compartment)};
            }
        }
        return {std::move(candidate), std::nullopt};
    }

    ResourceSnapshot snapshot() const {
        ResourceSnapshot result;
        result.cells_ = cells_;
        result.compartments_ = compartments_;
        result.totals_ = totals_;
        result.last_measurement_tick_ = last_measurement_tick_;
        result.identity_ = plan_->identity();
        result.revision_ = plan_->revision();
        result.profile_ = plan_->profile();
        return result;
    }

    const ResourceLedgerConfig& config() const { return config_; }
    std::shared_ptr<const CompiledGraph> plan() const { return plan_; }
    // P2 has no export operation; cumulative export is explicitly zero.
    bool export_supported() const { return false; }
    std::unique_ptr<ResourceLedger> fork_probe() const {
        return std::unique_ptr<ResourceLedger>(new ResourceLedger(*this));
    }

    // A cold graph edit may remove cells, but it must not silently birth
    // resource state. The candidate is fully validated before this ledger is
    // changed, and surviving state is matched by stable CellId.
    ResourceRebindResult rebind(
        const RuntimeState& runtime,
        std::shared_ptr<const CompiledGraph> graph) {
        if (!graph) {
            return {resource_detail::failure(
                ResourceErrorCode::MigrationInvalidInput,
                "resource rebind requires an owned compiled graph")};
        }
        if (!runtime.bound_to(*graph)) {
            return {resource_detail::failure(
                ResourceErrorCode::BindingMismatch,
                "resource rebind runtime is not bound to the candidate graph")};
        }
        if (graph->identity() != plan_->identity()) {
            return {resource_detail::failure(
                ResourceErrorCode::MigrationIdentityMismatch,
                "resource rebind cannot cross graph identities")};
        }
        if (graph->revision() < plan_->revision()) {
            return {resource_detail::failure(
                ResourceErrorCode::MigrationRevisionMismatch,
                "resource rebind cannot move to an older graph revision")};
        }
        if (graph->revision() == plan_->revision() &&
            !detail::same_plan(*plan_, *graph)) {
            return {resource_detail::failure(
                ResourceErrorCode::MigrationRevisionMismatch,
                "same-revision resource rebind is not an identical graph")};
        }

        std::vector<ResourceCellState> candidate_cells;
        candidate_cells.reserve(graph->cells().size());
        for (const auto& cell : graph->cells()) {
            const auto current = std::lower_bound(
                cells_.begin(), cells_.end(), cell.id,
                [](const ResourceCellState& state, CellId sought) {
                    return state.cell < sought;
                });
            if (current == cells_.end() || current->cell != cell.id) {
                return {resource_detail::failure(
                    ResourceErrorCode::UnsupportedOperation,
                    "resource rebind cannot add a cell without an explicit birth allocation",
                    true, cell.id)};
            }
            candidate_cells.push_back(*current);
        }

        plan_ = std::move(graph);
        cells_.swap(candidate_cells);
        return {};
    }

    ResourceGrowthResult grow(
        const RuntimeState& runtime,
        std::shared_ptr<const CompiledGraph> graph,
        std::span<const ResourceCellBirth> births,
        std::span<const ResourceGrowthCharge> charges) {
        if (!graph) {
            return growth_failure(
                ResourceErrorCode::MigrationInvalidInput,
                "resource growth requires an owned compiled graph");
        }
        if (!runtime.bound_to(*graph)) {
            return growth_failure(
                ResourceErrorCode::BindingMismatch,
                "resource growth runtime is not bound to the candidate graph");
        }
        if (graph->identity() != plan_->identity()) {
            return growth_failure(
                ResourceErrorCode::MigrationIdentityMismatch,
                "resource growth cannot cross graph identities");
        }
        if (graph->revision() <= plan_->revision()) {
            return growth_failure(
                ResourceErrorCode::MigrationRevisionMismatch,
                "resource growth must advance the graph revision");
        }

        std::vector<ResourceCellBirth> sorted_births(
            births.begin(), births.end());
        std::sort(
            sorted_births.begin(), sorted_births.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cell < rhs.cell;
            });
        for (std::size_t i = 0; i < sorted_births.size(); ++i) {
            const auto& birth = sorted_births[i];
            if (i != 0 && sorted_births[i - 1].cell == birth.cell) {
                return growth_failure(
                    ResourceErrorCode::DuplicateId,
                    "resource growth contains a duplicate birth cell",
                    true, birth.cell);
            }
            const auto existing = std::lower_bound(
                cells_.begin(), cells_.end(), birth.cell,
                [](const ResourceCellState& state, CellId sought) {
                    return state.cell < sought;
                });
            if (existing != cells_.end() && existing->cell == birth.cell) {
                return growth_failure(
                    ResourceErrorCode::DuplicateId,
                    "resource growth attempts to rebirth an existing cell",
                    true, birth.cell);
            }
            if (!resource_detail::finite_nonnegative(birth.initial_energy) ||
                !resource_detail::finite_nonnegative(birth.capacity) ||
                !resource_detail::finite_nonnegative(birth.structural_reserve) ||
                !resource_detail::finite_nonnegative(birth.birth_cost) ||
                birth.initial_energy > birth.capacity) {
                return growth_failure(
                    ResourceErrorCode::InvalidCell,
                    "resource birth values must be finite and non-negative",
                    true, birth.cell);
            }
            const auto compartment = std::lower_bound(
                compartments_.begin(), compartments_.end(), birth.compartment,
                [](const ResourceCompartmentState& state,
                   ResourceCompartmentId sought) {
                    return state.compartment < sought;
                });
            if (compartment == compartments_.end() ||
                compartment->compartment != birth.compartment) {
                return growth_failure(
                    ResourceErrorCode::MissingId,
                    "resource birth refers to a missing compartment",
                    true, birth.cell, false, EdgeId{0}, true,
                    birth.compartment);
            }
            const auto candidate_cell = std::lower_bound(
                graph->cells().begin(), graph->cells().end(), birth.cell,
                [](const CompiledCell& cell, CellId sought) {
                    return cell.id < sought;
                });
            if (candidate_cell == graph->cells().end() ||
                candidate_cell->id != birth.cell) {
                return growth_failure(
                    ResourceErrorCode::MissingId,
                    "resource birth is not present in the candidate graph",
                    true, birth.cell);
            }
        }

        std::vector<ResourceCellState> candidate_cells;
        candidate_cells.reserve(graph->cells().size());
        for (const auto& cell : graph->cells()) {
            const auto current = std::lower_bound(
                cells_.begin(), cells_.end(), cell.id,
                [](const ResourceCellState& state, CellId sought) {
                    return state.cell < sought;
                });
            if (current != cells_.end() && current->cell == cell.id) {
                candidate_cells.push_back(*current);
                continue;
            }
            const auto birth = std::lower_bound(
                sorted_births.begin(), sorted_births.end(), cell.id,
                [](const ResourceCellBirth& value, CellId sought) {
                    return value.cell < sought;
                });
            if (birth == sorted_births.end() || birth->cell != cell.id) {
                return growth_failure(
                    ResourceErrorCode::MissingId,
                    "candidate graph contains a cell without resource birth state",
                    true, cell.id);
            }
            candidate_cells.push_back(ResourceCellState{
                birth->cell,
                birth->compartment,
                birth->capacity,
                birth->initial_energy,
                birth->structural_reserve,
                birth->structural_reserve});
        }

        std::vector<ResourceGrowthCharge> all_charges(
            charges.begin(), charges.end());
        for (const auto& birth : sorted_births) {
            double endowment = 0.0;
            if (!resource_detail::checked_add(
                    birth.initial_energy, birth.structural_reserve, endowment) ||
                !resource_detail::checked_add(
                    endowment, birth.birth_cost, endowment)) {
                return growth_failure(
                    ResourceErrorCode::Overflow,
                    "resource birth funding overflows",
                    true, birth.cell);
            }
            all_charges.push_back(
                ResourceGrowthCharge{
                    birth.compartment, endowment, birth.sponsor,
                    birth.birth_cost});
        }
        std::sort(
            all_charges.begin(), all_charges.end(),
            [](const auto& lhs, const auto& rhs) {
                const uint64_t lhs_sponsor =
                    lhs.sponsor.has_value() ? lhs.sponsor->value : 0;
                const uint64_t rhs_sponsor =
                    rhs.sponsor.has_value() ? rhs.sponsor->value : 0;
                if (lhs_sponsor != rhs_sponsor) return lhs_sponsor < rhs_sponsor;
                if (lhs.compartment != rhs.compartment) {
                    return lhs.compartment < rhs.compartment;
                }
                return lhs.amount < rhs.amount;
            });

        std::vector<ResourceCompartmentState> staged_compartments = compartments_;
        ResourceTotals staged_totals = totals_;
        ResourceGrowthReport report;
        report.payments.reserve(all_charges.size());
        for (const auto& charge : all_charges) {
            if (!resource_detail::finite_nonnegative(charge.amount) ||
                !resource_detail::finite_nonnegative(charge.dissipated) ||
                charge.dissipated > charge.amount) {
                return growth_failure(
                    ResourceErrorCode::InvalidCell,
                    "growth charge must be finite, non-negative, and bounded");
            }
            auto compartment = std::lower_bound(
                staged_compartments.begin(), staged_compartments.end(),
                charge.compartment,
                [](const ResourceCompartmentState& state,
                   ResourceCompartmentId sought) {
                    return state.compartment < sought;
                });
            if (compartment == staged_compartments.end() ||
                compartment->compartment != charge.compartment) {
                return growth_failure(
                    ResourceErrorCode::MissingId,
                    "growth charge refers to a missing compartment",
                    false, CellId{0}, false, EdgeId{0}, true,
                    charge.compartment);
            }

            ResourceCellState* sponsor = nullptr;
            if (charge.sponsor.has_value()) {
                const auto found = std::lower_bound(
                    candidate_cells.begin(), candidate_cells.end(),
                    *charge.sponsor,
                    [](const ResourceCellState& state, CellId sought) {
                        return state.cell < sought;
                    });
                if (found == candidate_cells.end() ||
                    found->cell != *charge.sponsor) {
                    return growth_failure(
                        ResourceErrorCode::InvalidCell,
                        "growth charge sponsor is not a surviving cell",
                        true, *charge.sponsor);
                }
                sponsor = &*found;
            }

            double remaining = charge.amount;
            const double from_reserve = sponsor == nullptr
                ? 0.0
                : std::min(sponsor->structural_reserve, remaining);
            if (sponsor != nullptr) sponsor->structural_reserve -= from_reserve;
            remaining -= from_reserve;
            const double from_energy = sponsor == nullptr
                ? 0.0
                : std::min(sponsor->energy, remaining);
            if (sponsor != nullptr) sponsor->energy -= from_energy;
            remaining -= from_energy;
            const double from_environment =
                std::min(compartment->environmental_resource, remaining);
            compartment->environmental_resource -= from_environment;
            remaining -= from_environment;
            if (remaining > 0.0) {
                return growth_failure(
                    ResourceErrorCode::InsufficientResource,
                    "growth charge exceeds sponsor and compartment resources",
                    charge.sponsor.has_value(), charge.sponsor.value_or(CellId{0}),
                    false, EdgeId{0}, true, charge.compartment);
            }
            const double paid =
                from_reserve + from_energy + from_environment;
            if (sponsor != nullptr) {
                sponsor->cumulative_growth_cost += paid;
            }
            report.payments.push_back(
                ResourceGrowthPayment{
                    charge.sponsor,
                    charge.compartment,
                    charge.amount,
                    from_reserve,
                    from_energy,
                    from_environment,
                    paid});
            if (!resource_detail::checked_add(
                    report.requested_cost, charge.amount,
                    report.requested_cost) ||
                !resource_detail::checked_add(
                    report.paid_cost, paid, report.paid_cost) ||
                !resource_detail::checked_add(
                    staged_totals.cumulative_growth_cost,
                    charge.dissipated, staged_totals.cumulative_growth_cost) ||
                !resource_detail::checked_add(
                    staged_totals.cumulative_dissipation,
                    charge.dissipated, staged_totals.cumulative_dissipation)) {
                return growth_failure(
                    ResourceErrorCode::Overflow,
                    "growth accounting aggregate overflows");
            }
        }
        for (auto& cell : candidate_cells) {
            cell.exhausted = cell.energy == 0.0;
        }

        double lhs = 0.0;
        for (const auto& cell : candidate_cells) {
            if (!resource_detail::checked_add(lhs, cell.energy, lhs) ||
                !resource_detail::checked_add(
                    lhs, cell.structural_reserve, lhs)) {
                return growth_failure(
                    ResourceErrorCode::Overflow,
                    "growth conservation aggregate overflows");
            }
        }
        for (const auto& compartment : staged_compartments) {
            if (!resource_detail::checked_add(
                    lhs, compartment.environmental_resource, lhs)) {
                return growth_failure(
                    ResourceErrorCode::Overflow,
                    "growth environmental aggregate overflows");
            }
        }
        if (!resource_detail::checked_add(
                lhs, staged_totals.cumulative_dissipation, lhs) ||
            !resource_detail::checked_add(
                lhs, staged_totals.cumulative_export, lhs)) {
            return growth_failure(
                ResourceErrorCode::Overflow,
                "growth conservation total overflows");
        }
        double rhs = 0.0;
        if (!resource_detail::checked_add(
                staged_totals.initial_total,
                staged_totals.cumulative_external_injection,
                rhs)) {
            return growth_failure(
                ResourceErrorCode::Overflow,
                "growth conservation right-hand side overflows");
        }
        report.cumulative_growth_cost =
            staged_totals.cumulative_growth_cost;
        report.conservation_residual = lhs - rhs;
        if (!std::isfinite(report.conservation_residual)) {
            return growth_failure(
                ResourceErrorCode::Overflow,
                "growth conservation residual is non-finite");
        }

        plan_ = std::move(graph);
        cells_.swap(candidate_cells);
        compartments_.swap(staged_compartments);
        totals_ = staged_totals;
        return {std::move(report), std::nullopt};
    }

    // The caller must pass the measurement returned by the successful native
    // executor step for this runtime, while its committed view is live.
    // Public measurement structs cannot prove provenance cryptographically;
    // this ledger validates the full graph/tick/coverage contract and never
    // treats a missing or mismatched record as a zero-cost success.
    // A failed settlement leaves this ledger unchanged but does not roll back
    // the already-committed native RuntimeState step.
    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurement& measurement) {
        return settle_view(runtime, ExecutionMeasurementView{
            measurement.tick,
            measurement.cells,
            measurement.ports,
            measurement.edges,
            measurement.identity,
            measurement.revision,
            measurement.profile});
    }

    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurementView& measurement) {
        return settle_view(runtime, measurement);
    }

    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurement& measurement,
        std::span<const ResourceSupply> supplies) {
        return settle_with_supplies(runtime, measurement, supplies);
    }

    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurementView& measurement,
        std::span<const ResourceSupply> supplies) {
        return settle_view(runtime, measurement, supplies);
    }

    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurement& measurement,
        std::initializer_list<ResourceSupply> supplies) {
        return settle_with_supplies(runtime, measurement, supplies);
    }

    ResourceSettlementResult settle(
        const RuntimeState& runtime,
        const ExecutionMeasurementView& measurement,
        std::initializer_list<ResourceSupply> supplies) {
        return settle_view(runtime, measurement, supplies);
    }

    // RuntimeState::reset_episode resets only computational state. The ledger
    // must be explicitly handed the new episode so resources and lifetime
    // statistics are preserved while the measurement cursor starts at zero.
    ResourceSettlementResult rebind_episode(const RuntimeState& runtime) {
        if (!runtime.bound_to(*plan_)) {
            return failure_result(ResourceErrorCode::BindingMismatch,
                                  "episode runtime is bound to a different full graph");
        }
        if (runtime.tick() != 0) {
            return failure_result(ResourceErrorCode::EpisodeMismatch,
                                  "episode rebind requires a runtime reset to tick zero");
        }
        last_measurement_tick_ = 0;
        return {};
    }

private:
    friend class CellularLifecycleController;
    friend class CellularGrowthController;

    void adopt_from(ResourceLedger&& candidate) noexcept {
        plan_ = std::move(candidate.plan_);
        cells_.swap(candidate.cells_);
        compartments_.swap(candidate.compartments_);
        totals_ = candidate.totals_;
        last_measurement_tick_ = candidate.last_measurement_tick_;
    }

    ResourceLedger(
        std::shared_ptr<const CompiledGraph> plan,
        ResourceLedgerConfig config)
        : plan_(std::move(plan)), config_(std::move(config)) {}

    static std::optional<ResourceError> validate_config(
        const ResourceLedgerConfig& config) {
        if (!resource_detail::finite_positive(config.dt) ||
            !resource_detail::finite_nonnegative(config.maintenance_cost) ||
            !resource_detail::finite_nonnegative(config.absorption_rate) ||
            !resource_detail::finite_positive(config.activity_scale) ||
            !resource_detail::finite_nonnegative(config.activity_cost) ||
            !resource_detail::finite_positive(config.transmission_scale) ||
            !resource_detail::finite_nonnegative(config.transmission_cost)) {
            return resource_detail::failure(
                ResourceErrorCode::InvalidConfig,
                "resource rates, costs, dt, and scales must be finite and non-negative; scales and dt are positive");
        }
        std::vector<CellType> seen;
        seen.reserve(config.execution_costs.size());
        for (const auto& entry : config.execution_costs) {
            if (!contract_for(entry.type).has_value() ||
                !resource_detail::finite_nonnegative(entry.cost)) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidConfig,
                    "execution cost has an unknown cell type or invalid value");
            }
            if (std::find(seen.begin(), seen.end(), entry.type) != seen.end()) {
                return resource_detail::failure(
                    ResourceErrorCode::DuplicateId,
                    "execution cost table contains a duplicate cell type");
            }
            seen.push_back(entry.type);
        }
        return std::nullopt;
    }

    std::optional<ResourceError> initialize(
        const RuntimeState& runtime,
        std::span<const ResourceCellInitial> initial_cells,
        std::span<const ResourceCompartmentInitial> initial_compartments) {
        const auto graph_cells = plan_->cells();
        if (initial_cells.size() != graph_cells.size()) {
            return resource_detail::failure(
                initial_cells.size() < graph_cells.size()
                    ? ResourceErrorCode::MissingId
                    : ResourceErrorCode::InvalidCell,
                "resource cell records must cover every real graph cell exactly once");
        }
        std::vector<ResourceCellInitial> sorted_cells(
            initial_cells.begin(), initial_cells.end());
        std::sort(sorted_cells.begin(), sorted_cells.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.cell < rhs.cell; });
        for (std::size_t i = 1; i < sorted_cells.size(); ++i) {
            if (sorted_cells[i - 1].cell == sorted_cells[i].cell) {
                return resource_detail::failure(
                    ResourceErrorCode::DuplicateId,
                    "resource cell records contain a duplicate stable ID",
                    true, sorted_cells[i].cell);
            }
        }
        std::vector<ResourceCompartmentInitial> sorted_compartments(
            initial_compartments.begin(), initial_compartments.end());
        std::sort(
            sorted_compartments.begin(), sorted_compartments.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.compartment < rhs.compartment;
            });
        for (std::size_t i = 1; i < sorted_compartments.size(); ++i) {
            if (sorted_compartments[i - 1].compartment ==
                sorted_compartments[i].compartment) {
                return resource_detail::failure(
                    ResourceErrorCode::DuplicateId,
                    "resource compartments contain a duplicate stable ID",
                    false, CellId{0}, false, EdgeId{0}, true,
                    sorted_compartments[i].compartment);
            }
        }
        for (const auto& cell : graph_cells) {
            const auto it = std::lower_bound(
                sorted_cells.begin(), sorted_cells.end(), cell.id,
                [](const auto& initial, CellId sought) {
                    return initial.cell < sought;
                });
            if (it == sorted_cells.end() || it->cell != cell.id) {
                return resource_detail::failure(
                    ResourceErrorCode::MissingId,
                    "resource cells must use the graph's real stable IDs",
                    true, cell.id);
            }
            if (!contract_for(cell.type).has_value()) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidCell,
                    "graph cell type has no generated primitive contract",
                    true, cell.id);
            }
        }
        if (graph_cells.empty() && !sorted_cells.empty()) {
            return resource_detail::failure(
                ResourceErrorCode::InvalidCell,
                "empty graph cannot have phantom resource cells");
        }
        for (const auto& initial : sorted_cells) {
            const auto compartment = std::lower_bound(
                sorted_compartments.begin(), sorted_compartments.end(),
                initial.compartment,
                [](const auto& value, ResourceCompartmentId sought) {
                    return value.compartment < sought;
                });
            if (compartment == sorted_compartments.end() ||
                compartment->compartment != initial.compartment) {
                return resource_detail::failure(
                    ResourceErrorCode::MissingId,
                    "resource cell refers to a missing compartment",
                    true, initial.cell, false, EdgeId{0}, true,
                    initial.compartment);
            }
            if (!resource_detail::finite_nonnegative(initial.initial_energy) ||
                !resource_detail::finite_nonnegative(initial.capacity) ||
                !resource_detail::finite_nonnegative(initial.structural_reserve) ||
                initial.initial_energy > initial.capacity) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidCell,
                    "cell energy, capacity, and structural reserve must be finite and valid",
                    true, initial.cell);
            }
        }
        for (const auto& compartment : sorted_compartments) {
            if (!resource_detail::finite_nonnegative(
                    compartment.initial_environmental_resource)) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidCompartment,
                    "initial environmental resource must be finite and non-negative",
                    false, CellId{0}, false, EdgeId{0}, true,
                    compartment.compartment);
            }
        }
        cells_.reserve(sorted_cells.size());
        compartments_.reserve(sorted_compartments.size());
        for (const auto& initial : sorted_cells) {
            ResourceCellState state{
                initial.cell, initial.compartment, initial.capacity,
                initial.initial_energy, initial.structural_reserve,
                initial.structural_reserve};
            state.exhausted = state.energy == 0.0;
            cells_.push_back(state);
        }
        for (const auto& initial : sorted_compartments) {
            compartments_.push_back(ResourceCompartmentState{
                initial.compartment, initial.initial_environmental_resource});
        }
        totals_.initial_total = 0.0;
        for (const auto& cell : cells_) {
            if (!resource_detail::checked_add(
                    totals_.initial_total, cell.energy, totals_.initial_total) ||
                !resource_detail::checked_add(
                    totals_.initial_total, cell.structural_reserve,
                    totals_.initial_total)) {
                return resource_detail::failure(
                    ResourceErrorCode::Overflow,
                    "initial cell resource aggregate overflows");
            }
        }
        for (const auto& compartment : compartments_) {
            if (!resource_detail::checked_add(
                    totals_.initial_total,
                    compartment.environmental_resource,
                    totals_.initial_total)) {
                return resource_detail::failure(
                    ResourceErrorCode::Overflow,
                    "initial environmental resource aggregate overflows");
            }
        }
        if (!runtime.bound_to(*plan_)) {
            return resource_detail::failure(
                ResourceErrorCode::BindingMismatch,
                "resource ledger runtime binding does not match the full compiled graph");
        }
        // Attachment starts a fresh resource lifetime at the runtime's
        // committed cursor; it never invents metabolism history before attach.
        last_measurement_tick_ = runtime.tick();
        return std::nullopt;
    }

    double execution_cost(CellType type) const {
        for (const auto& entry : config_.execution_costs) {
            if (entry.type == type) return entry.cost;
        }
        return 0.0;
    }

    ResourceSettlementResult settle_with_supplies(
        const RuntimeState& runtime,
        const ExecutionMeasurement& measurement,
        std::span<const ResourceSupply> supplies) {
        return settle_view(runtime, ExecutionMeasurementView{
            measurement.tick,
            measurement.cells,
            measurement.ports,
            measurement.edges,
            measurement.identity,
            measurement.revision,
            measurement.profile},
            supplies);
    }

    ResourceSettlementResult settle_view(
        const RuntimeState& runtime,
        const ExecutionMeasurementView& measurement,
        std::span<const ResourceSupply> supplies = {}) {
        if (!runtime.bound_to(*plan_)) {
            return failure_result(
                ResourceErrorCode::BindingMismatch,
                "measurement runtime is not bound to this ledger's full graph");
        }
        if (measurement.identity != plan_->identity() ||
            measurement.revision != plan_->revision() ||
            measurement.profile != plan_->profile()) {
            return failure_result(
                ResourceErrorCode::BindingMismatch,
                "measurement graph identity, revision, or semantic profile differs");
        }
        if (measurement.tick != runtime.tick()) {
            return failure_result(
                ResourceErrorCode::InvalidMeasurement,
                "measurement tick must equal the committed runtime tick");
        }
        if (measurement.tick == 0) {
            return failure_result(
                ResourceErrorCode::InvalidMeasurement,
                "tick zero is not a successful native execution measurement");
        }
        if (measurement.tick < last_measurement_tick_) {
            return failure_result(
                ResourceErrorCode::StaleMeasurement,
                "measurement tick is older than the committed ledger cursor");
        }
        if (measurement.tick == last_measurement_tick_) {
            return failure_result(
                ResourceErrorCode::DuplicateMeasurement,
                "successful native measurement has already been settled");
        }
        if (last_measurement_tick_ == std::numeric_limits<uint64_t>::max() ||
            measurement.tick != last_measurement_tick_ + 1) {
            return failure_result(
                ResourceErrorCode::SkippedMeasurement,
                "measurement tick skips an unsettled committed native step");
        }
        if (const auto error = validate_measurement(measurement)) {
            return {{}, *error};
        }
        std::vector<ResourceSupply> injection(compartments_.size());
        std::vector<bool> seen_injection(compartments_.size(), false);
        for (std::size_t i = 0; i < compartments_.size(); ++i) {
            injection[i].compartment = compartments_[i].compartment;
        }
        if (!supplies.empty()) {
            if (supplies.size() != compartments_.size()) {
                return failure_result(
                    supplies.size() < compartments_.size()
                        ? ResourceErrorCode::MissingId
                        : ResourceErrorCode::InvalidCompartment,
                    "supply injection must cover every compartment exactly once");
            }
            for (const auto& supply : supplies) {
                if (!resource_detail::finite_nonnegative(supply.amount)) {
                    return failure_result(
                        ResourceErrorCode::InvalidCompartment,
                        "supply injection must be finite and non-negative",
                        false, CellId{0}, false, EdgeId{0}, true,
                        supply.compartment);
                }
                const auto it = std::lower_bound(
                    injection.begin(), injection.end(), supply.compartment,
                    [](const auto& value, ResourceCompartmentId sought) {
                        return value.compartment < sought;
                    });
                if (it == injection.end() || it->compartment != supply.compartment) {
                    return failure_result(
                        ResourceErrorCode::MissingId,
                        "supply injection refers to an unknown compartment",
                        false, CellId{0}, false, EdgeId{0}, true,
                        supply.compartment);
                }
                const std::size_t index =
                    static_cast<std::size_t>(it - injection.begin());
                if (seen_injection[index]) {
                    return failure_result(
                        ResourceErrorCode::DuplicateId,
                        "supply injection contains a duplicate compartment ID",
                        false, CellId{0}, false, EdgeId{0}, true,
                        supply.compartment);
                }
                it->amount = supply.amount;
                seen_injection[index] = true;
            }
        }

        std::vector<ExecutedCellMeasurement> measured_cells(
            measurement.cells.begin(), measurement.cells.end());
        std::sort(
            measured_cells.begin(), measured_cells.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cell < rhs.cell;
            });
        std::vector<EdgeTransmissionMeasurement> measured_edges(
            measurement.edges.begin(), measurement.edges.end());
        std::sort(
            measured_edges.begin(), measured_edges.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.edge < rhs.edge;
            });
        std::vector<ResourceCellState> staged_cells = cells_;
        std::vector<ResourceCompartmentState> staged_compartments = compartments_;
        ResourceTotals staged_totals = totals_;
        ResourceSettlementReport report;
        report.tick = measurement.tick;
        report.cells.reserve(staged_cells.size());
        std::vector<double> demand(staged_cells.size(), 0.0);
        std::vector<double> absorbed(staged_cells.size(), 0.0);
        std::vector<double> requested(staged_cells.size(), 0.0);
        std::vector<double> transmission(staged_cells.size(), 0.0);
        std::vector<double> transmission_costs(staged_cells.size(), 0.0);
        std::vector<double> activity(staged_cells.size(), 0.0);
        std::vector<bool> executed(staged_cells.size(), false);

        double injected_total = 0.0;
        for (std::size_t i = 0; i < injection.size(); ++i) {
            if (!resource_detail::checked_add(
                    staged_compartments[i].environmental_resource,
                    injection[i].amount,
                    staged_compartments[i].environmental_resource) ||
                !resource_detail::checked_add(
                    injected_total, injection[i].amount, injected_total)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "environmental resource or injection aggregate overflows");
            }
        }

        for (std::size_t i = 0; i < staged_cells.size(); ++i) {
            const auto& measured = measured_cells[i];
            const double room = staged_cells[i].capacity - staged_cells[i].energy;
            if (!resource_detail::finite_nonnegative(room)) {
                return failure_result(
                    ResourceErrorCode::InvalidCell,
                    "committed resource state has invalid remaining capacity",
                    true, staged_cells[i].cell);
            }
            double max_absorption = 0.0;
            if (!resource_detail::checked_mul(
                    config_.absorption_rate, config_.dt, max_absorption)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "absorption demand overflows",
                    true, staged_cells[i].cell);
            }
            demand[i] = std::min(room, max_absorption);
            activity[i] = resource_detail::bounded_magnitude(
                measured.output, config_.activity_scale);
            executed[i] = measured.executed;
            if (executed[i]) {
                if (!resource_detail::checked_add(
                        requested[i], execution_cost(measured.type), requested[i])) {
                    return failure_result(
                        ResourceErrorCode::Overflow,
                        "execution cost overflows",
                        true, staged_cells[i].cell);
                }
            }
            double maintenance = 0.0;
            if (!resource_detail::checked_mul(
                    config_.maintenance_cost, config_.dt, maintenance) ||
                !resource_detail::checked_add(requested[i], maintenance, requested[i])) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "maintenance cost overflows",
                    true, staged_cells[i].cell);
            }
            double activity_charge = 0.0;
            if (!resource_detail::checked_mul(
                    config_.activity_cost, activity[i], activity_charge) ||
                !resource_detail::checked_add(
                    requested[i], activity_charge, requested[i])) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "activity cost overflows",
                    true, staged_cells[i].cell);
            }
        }
        for (const auto& edge : measured_edges) {
            const auto source = std::lower_bound(
                staged_cells.begin(), staged_cells.end(), edge.source,
                [](const auto& cell, CellId sought) { return cell.cell < sought; });
            if (source == staged_cells.end() || source->cell != edge.source) {
                return failure_result(
                    ResourceErrorCode::InvalidMeasurement,
                    "edge measurement source is not a real cell",
                    false, CellId{0}, true, edge.edge);
            }
            const std::size_t source_index =
                static_cast<std::size_t>(source - staged_cells.begin());
            const double bounded = resource_detail::bounded_magnitude(
                edge.contribution, config_.transmission_scale);
            if (!resource_detail::checked_add(
                    transmission[source_index], bounded, transmission[source_index])) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "cumulative transmission measure overflows",
                    true, edge.source, true, edge.edge);
            }
            double charge = 0.0;
            if (!resource_detail::checked_mul(
                    config_.transmission_cost, bounded, charge) ||
                !resource_detail::checked_add(
                    transmission_costs[source_index], charge,
                    transmission_costs[source_index])) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "transmission cost overflows",
                    true, edge.source, true, edge.edge);
            }
            if (!resource_detail::checked_add(
                    requested[source_index], charge, requested[source_index])) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "total cell cost overflows",
                    true, edge.source, true, edge.edge);
            }
        }

        std::vector<std::vector<std::size_t>> members_by_compartment(
            staged_compartments.size());
        for (std::size_t i = 0; i < staged_cells.size(); ++i) {
            const auto compartment = std::lower_bound(
                staged_compartments.begin(), staged_compartments.end(),
                staged_cells[i].compartment,
                [](const auto& value, ResourceCompartmentId sought) {
                    return value.compartment < sought;
                });
            if (compartment == staged_compartments.end() ||
                compartment->compartment != staged_cells[i].compartment) {
                return failure_result(
                    ResourceErrorCode::BindingMismatch,
                    "committed cell refers to a missing compartment",
                    true, staged_cells[i].cell);
            }
            members_by_compartment[
                static_cast<std::size_t>(
                    compartment - staged_compartments.begin())]
                .push_back(i);
        }
        for (std::size_t compartment_index = 0;
             compartment_index < staged_compartments.size();
             ++compartment_index) {
            double total_demand = 0.0;
            const auto& members = members_by_compartment[compartment_index];
            for (const std::size_t i : members) {
                if (!resource_detail::checked_add(
                        total_demand, demand[i], total_demand)) {
                    return failure_result(
                        ResourceErrorCode::Overflow,
                        "compartment demand aggregate overflows");
                }
            }
            const double available =
                staged_compartments[compartment_index].environmental_resource;
            if (!resource_detail::finite_nonnegative(available)) {
                return failure_result(
                    ResourceErrorCode::InvalidCompartment,
                    "environmental resource is invalid during settlement",
                    false, CellId{0}, false, EdgeId{0}, true,
                    staged_compartments[compartment_index].compartment);
            }
            double allocated = 0.0;
            std::vector<std::size_t> positive_members;
            positive_members.reserve(members.size());
            for (const std::size_t i : members) {
                if (demand[i] > 0.0) positive_members.push_back(i);
            }
            if (total_demand <= available) {
                for (const std::size_t i : positive_members) {
                    absorbed[i] = demand[i];
                    if (!resource_detail::checked_add(
                            allocated, absorbed[i], allocated)) {
                        return failure_result(
                            ResourceErrorCode::Overflow,
                            "compartment allocation aggregate overflows");
                    }
                }
            } else if (!positive_members.empty()) {
                const double ratio = available / total_demand;
                if (!resource_detail::finite_nonnegative(ratio) || ratio > 1.0) {
                    return failure_result(
                        ResourceErrorCode::Overflow,
                        "proportional compartment allocation ratio is invalid");
                }
                for (const std::size_t i : positive_members) {
                    double value = ratio * demand[i];
                    if (!resource_detail::finite_nonnegative(value)) {
                        return failure_result(
                            ResourceErrorCode::Overflow,
                            "proportional compartment allocation is not representable");
                    }
                    if (value > demand[i]) value = demand[i];
                    absorbed[i] = value;
                    if (!resource_detail::checked_add(
                            allocated, value, allocated)) {
                        return failure_result(
                            ResourceErrorCode::Overflow,
                            "compartment allocation aggregate overflows");
                    }
                }
                if (allocated > available) {
                    double excess = allocated - available;
                    for (auto it = positive_members.rbegin();
                         it != positive_members.rend() && excess > 0.0;
                         ++it) {
                        const double reduction = std::min(excess, absorbed[*it]);
                        absorbed[*it] -= reduction;
                        allocated -= reduction;
                        excess -= reduction;
                    }
                    if (excess > 0.0) {
                        return failure_result(
                            ResourceErrorCode::Overflow,
                            "proportional compartment allocation exceeds supply");
                    }
                    // Every correction above is bounded by an already
                    // allocated positive share; the remaining environmental
                    // amount is kept from the corrected aggregate rather
                    // than assigned to one arbitrarily selected cell.
                    if (allocated < 0.0 || !std::isfinite(allocated)) {
                        return failure_result(
                            ResourceErrorCode::Overflow,
                            "proportional allocation correction is invalid");
                    }
                }
            }
            staged_compartments[compartment_index].environmental_resource =
                available - allocated;
            if (!resource_detail::finite_nonnegative(
                    staged_compartments[compartment_index].environmental_resource)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "environmental resource became invalid after allocation",
                    false, CellId{0}, false, EdgeId{0}, true,
                    staged_compartments[compartment_index].compartment);
            }
        }

        double report_absorbed = 0.0;
        double report_requested = 0.0;
        double report_paid = 0.0;
        double report_unpaid = 0.0;
        for (std::size_t i = 0; i < staged_cells.size(); ++i) {
            auto& cell = staged_cells[i];
            if (!resource_detail::checked_add(
                    cell.energy, absorbed[i], cell.energy) ||
                !resource_detail::finite_nonnegative(cell.energy) ||
                cell.energy > cell.capacity) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "cell energy overflows during absorption",
                    true, cell.cell);
            }
            double maintenance = 0.0;
            if (!resource_detail::checked_mul(
                    config_.maintenance_cost, config_.dt, maintenance)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "maintenance cost overflows", true, cell.cell);
            }
            const double paid = std::min(cell.energy, requested[i]);
            const double unpaid = requested[i] - paid;
            if (!resource_detail::finite_nonnegative(paid) ||
                !resource_detail::finite_nonnegative(unpaid)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "cost payment is invalid", true, cell.cell);
            }
            cell.energy -= paid;
            if (cell.energy < 0.0 || !std::isfinite(cell.energy)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "cell energy became negative after payment", true, cell.cell);
            }
            if (!resource_detail::checked_add(
                    cell.age, config_.dt, cell.age) ||
                !resource_detail::checked_add(
                    cell.cumulative_activity, activity[i],
                    cell.cumulative_activity) ||
                !resource_detail::checked_add(
                    cell.cumulative_transmission, transmission[i],
                    cell.cumulative_transmission) ||
                !resource_detail::checked_add(
                    cell.cumulative_execution_cost,
                    executed[i] ? execution_cost(measured_cells[i].type) : 0.0,
                    cell.cumulative_execution_cost) ||
                !resource_detail::checked_add(
                    cell.cumulative_maintenance_cost, maintenance,
                    cell.cumulative_maintenance_cost) ||
                !resource_detail::checked_add(
                    cell.cumulative_activity_cost,
                    config_.activity_cost * activity[i],
                    cell.cumulative_activity_cost) ||
                !resource_detail::checked_add(
                    cell.cumulative_transmission_cost,
                    transmission_costs[i],
                    cell.cumulative_transmission_cost) ||
                !resource_detail::checked_add(
                    cell.cumulative_paid_cost, paid, cell.cumulative_paid_cost) ||
                !resource_detail::checked_add(
                    cell.cumulative_unpaid_cost, unpaid,
                    cell.cumulative_unpaid_cost)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "cell cumulative accounting overflows", true, cell.cell);
            }
            if (executed[i]) {
                if (!resource_detail::checked_increment(
                        cell.execution_count, cell.execution_count)) {
                    return failure_result(
                        ResourceErrorCode::Overflow,
                        "cell execution counter overflows", true, cell.cell);
                }
            }
            cell.exhausted = cell.energy == 0.0;
            report.cells.push_back(ResourceCellSettlement{
                cell.cell, absorbed[i], activity[i],
                executed[i] ? execution_cost(measured_cells[i].type) : 0.0,
                maintenance, config_.activity_cost * activity[i],
                transmission_costs[i], requested[i], paid, unpaid,
                executed[i], cell.exhausted});
            if (cell.exhausted) ++report.exhausted_cells;
            if (!resource_detail::checked_add(
                    report_absorbed, absorbed[i], report_absorbed) ||
                !resource_detail::checked_add(
                    report_requested, requested[i], report_requested) ||
                !resource_detail::checked_add(
                    report_paid, paid, report_paid) ||
                !resource_detail::checked_add(
                    report_unpaid, unpaid, report_unpaid) ||
                !resource_detail::checked_add(
                    report.execution_cost,
                    executed[i] ? execution_cost(measured_cells[i].type) : 0.0,
                    report.execution_cost) ||
                !resource_detail::checked_add(
                    report.maintenance_cost, maintenance, report.maintenance_cost) ||
                !resource_detail::checked_add(
                    report.activity_cost,
                    config_.activity_cost * activity[i], report.activity_cost) ||
                !resource_detail::checked_add(
                    report.transmission_cost, transmission_costs[i],
                    report.transmission_cost)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "settlement report aggregate overflows");
            }
        }
        if (!resource_detail::checked_add(
                staged_totals.cumulative_external_injection,
                injected_total, staged_totals.cumulative_external_injection) ||
            !resource_detail::checked_add(
                staged_totals.cumulative_dissipation,
                report_paid, staged_totals.cumulative_dissipation) ||
            !resource_detail::checked_add(
                staged_totals.cumulative_paid_cost,
                report_paid, staged_totals.cumulative_paid_cost) ||
            !resource_detail::checked_add(
                staged_totals.cumulative_unpaid_cost,
                report_unpaid, staged_totals.cumulative_unpaid_cost)) {
            return failure_result(
                ResourceErrorCode::Overflow,
                "cumulative ledger totals overflow");
        }
        double rhs = 0.0;
        if (!resource_detail::checked_add(
                staged_totals.initial_total,
                staged_totals.cumulative_external_injection,
                rhs)) {
            return failure_result(
                ResourceErrorCode::Overflow,
                "conservation right-hand side overflows");
        }
        double lhs = 0.0;
        for (const auto& cell : staged_cells) {
            if (!resource_detail::checked_add(lhs, cell.energy, lhs) ||
                !resource_detail::checked_add(lhs, cell.structural_reserve, lhs)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "conservation aggregate overflows");
            }
        }
        for (const auto& compartment : staged_compartments) {
            if (!resource_detail::checked_add(
                    lhs, compartment.environmental_resource, lhs)) {
                return failure_result(
                    ResourceErrorCode::Overflow,
                    "conservation environmental aggregate overflows");
            }
        }
        if (!resource_detail::checked_add(
                lhs, staged_totals.cumulative_dissipation, lhs) ||
            !resource_detail::checked_add(
                lhs, staged_totals.cumulative_export, lhs)) {
            return failure_result(
                ResourceErrorCode::Overflow,
                "conservation total overflows");
        }
        const double residual = lhs - rhs;
        if (!std::isfinite(residual)) {
            return failure_result(
                ResourceErrorCode::Overflow,
                "conservation residual is non-finite");
        }
        report.injected = injected_total;
        report.absorbed = report_absorbed;
        report.requested_cost = report_requested;
        report.paid_cost = report_paid;
        report.unpaid_cost = report_unpaid;
        report.cumulative_dissipation = staged_totals.cumulative_dissipation;
        report.cumulative_export = staged_totals.cumulative_export;
        report.conservation_residual = residual;
        cells_.swap(staged_cells);
        compartments_.swap(staged_compartments);
        totals_ = staged_totals;
        last_measurement_tick_ = measurement.tick;
        return {std::move(report), std::nullopt};
    }

    std::optional<ResourceError> validate_measurement(
        const ExecutionMeasurementView& measurement) const {
        const auto graph_cells = plan_->cells();
        const auto graph_edges = plan_->edges();
        const auto reductions = plan_->port_reductions();
        if (measurement.cells.size() != graph_cells.size() ||
            measurement.ports.size() != reductions.size() ||
            measurement.edges.size() != graph_edges.size()) {
            return resource_detail::failure(
                ResourceErrorCode::MissingCoverage,
                "measurement must cover every real cell, reduced port, and edge exactly once");
        }
        std::vector<ExecutedCellMeasurement> cells(
            measurement.cells.begin(), measurement.cells.end());
        std::sort(cells.begin(), cells.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.cell < rhs.cell; });
        for (std::size_t i = 0; i < graph_cells.size(); ++i) {
            if (cells[i].cell != graph_cells[i].id ||
                cells[i].type != graph_cells[i].type ||
                !std::isfinite(cells[i].output)) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidMeasurement,
                    "cell measurement does not match the full graph binding",
                    true, cells[i].cell);
            }
        }
        std::vector<ReducedPortMeasurement> ports(
            measurement.ports.begin(), measurement.ports.end());
        std::sort(
            ports.begin(), ports.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.cell != rhs.cell) return lhs.cell < rhs.cell;
                return lhs.port < rhs.port;
            });
        std::vector<ReducedPortMeasurement> expected_ports;
        expected_ports.reserve(reductions.size());
        for (const auto& reduction : reductions) {
            expected_ports.push_back(ReducedPortMeasurement{
                graph_cells[reduction.target_index].id,
                reduction.target_port,
                0.0});
        }
        std::sort(
            expected_ports.begin(), expected_ports.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.cell != rhs.cell) return lhs.cell < rhs.cell;
                return lhs.port < rhs.port;
            });
        for (std::size_t i = 0; i < ports.size(); ++i) {
            if (ports[i].cell != expected_ports[i].cell ||
                ports[i].port != expected_ports[i].port ||
                !std::isfinite(ports[i].reduced_input)) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidMeasurement,
                    "port measurement does not match the full graph binding",
                    true, ports[i].cell);
            }
        }
        std::vector<EdgeTransmissionMeasurement> edges(
            measurement.edges.begin(), measurement.edges.end());
        std::sort(edges.begin(), edges.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.edge < rhs.edge; });
        std::vector<std::size_t> expected_edge_indices(graph_edges.size());
        for (std::size_t i = 0; i < expected_edge_indices.size(); ++i) {
            expected_edge_indices[i] = i;
        }
        std::sort(
            expected_edge_indices.begin(), expected_edge_indices.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                return graph_edges[lhs].id < graph_edges[rhs].id;
            });
        for (std::size_t i = 0; i < graph_edges.size(); ++i) {
            const auto& expected = graph_edges[expected_edge_indices[i]];
            const auto& actual = edges[i];
            if (actual.edge != expected.id ||
                actual.source != graph_cells[expected.source_index].id ||
                actual.target != graph_cells[expected.target_index].id ||
                actual.target_port != expected.target_port ||
                actual.source_mode != expected.delay ||
                !std::isfinite(actual.source_value) ||
                !std::isfinite(actual.contribution)) {
                return resource_detail::failure(
                    ResourceErrorCode::InvalidMeasurement,
                    "edge measurement does not match the full graph binding",
                    false, CellId{0}, true, actual.edge);
            }
        }
        return std::nullopt;
    }

    static ResourceSettlementResult failure_result(
        ResourceErrorCode code,
        std::string reason) {
        return {ResourceSettlementReport{},
                resource_detail::failure(code, std::move(reason))};
    }

    static ResourceSettlementResult failure_result(
        ResourceErrorCode code,
        std::string reason,
        bool has_cell,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0},
        bool has_compartment = false,
        ResourceCompartmentId compartment = ResourceCompartmentId{0}) {
        return {ResourceSettlementReport{},
                resource_detail::failure(
                    code, std::move(reason), has_cell, cell, has_edge, edge,
                    has_compartment, compartment)};
    }

    static ResourceGrowthResult growth_failure(
        ResourceErrorCode code,
        std::string reason,
        bool has_cell = false,
        CellId cell = CellId{0},
        bool has_edge = false,
        EdgeId edge = EdgeId{0},
        bool has_compartment = false,
        ResourceCompartmentId compartment = ResourceCompartmentId{0}) {
        return {
            {},
            resource_detail::failure(
                code, std::move(reason), has_cell, cell, has_edge, edge,
                has_compartment, compartment)};
    }

    std::shared_ptr<const CompiledGraph> plan_;
    ResourceLedgerConfig config_;
    std::vector<ResourceCellState> cells_;
    std::vector<ResourceCompartmentState> compartments_;
    ResourceTotals totals_{};
    uint64_t last_measurement_tick_{0};
};

}  // namespace kun::core
