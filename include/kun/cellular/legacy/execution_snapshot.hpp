#pragma once

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/sdsc_cell_kernel.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kun::migration {

// R2b is a one-time, cold-data boundary. It does not create a runtime,
// checkpoint loader, or continuously synchronized copy of CellularOrganism.
// The importer never calls compile(), reset_state(), forward_nd(), or
// plasticity on its source.
enum class ExecutionSnapshotImportCode : uint8_t {
    Accepted = 0,
    InvalidRequest,
    NotCompiled,
    UnsupportedAdvancedOrgan,
    InvalidCellMetadata,
    InvalidExecutionOrder,
    UnsupportedPrunedExecution,
    InvalidCompiledEdge,
    UnsupportedIgnoredPort,
    InvalidCsr,
    UnsupportedInactiveEdge,
    EdgeProvenanceMismatch,
    AmbiguousParallelEdgeProvenance,
    InvalidComputationalState,
    InvalidActionMetadata,
    InvalidParameter,
    CompilerRejected,
};

struct ExecutionSnapshotImportError {
    ExecutionSnapshotImportCode code{ExecutionSnapshotImportCode::CompilerRejected};
    std::string reason;
    bool has_object_id{false};
    uint64_t object_id{0};

    std::string diagnostic() const {
        std::string result;
        if (has_object_id) {
            result = "object id=" + std::to_string(object_id) + ": ";
        }
        result += reason;
        return result;
    }
};

struct SnapshotCellState {
    // Only computational continuation state is retained. Physical/glow
    // telemetry, environment, RNG, optimizer, and task state are excluded.
    core::CellId cell{};
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
};

struct SnapshotCellParameterProvenance {
    core::CellId cell{};
    core::ParameterSlot slot{core::ParameterSlot::Param1};
    // Legacy Cell has no separate ancestral parameter copy. This is the
    // import-time/current value, never a recovered germline value.
    double raw_current_value{0.0};
    bool captured_current_value{true};
    bool ancestral_value_recovered{false};
};

struct SnapshotEdgeProvenance {
    core::EdgeId edge{};
    core::CellId source{};
    core::CellId target{};
    core::InputPort target_port{};
    bool recurrent{false};
    double live_weight{0.0};
    double compile_time_weight{0.0};
    // This is the raw declared initial_weight sidecar. It is intentionally
    // separate from the current live seed and is not treated as genealogy.
    double raw_declared_initial_weight{0.0};
    double hebbian_rate{0.0};
    double hebbian_decay{0.0};
    size_t raw_synapse_index{0};
};

struct SnapshotReadout {
    core::CellId cell{};
    CellType type{CellType::OP_EMA};
};

struct ExecutionSnapshotImportResult;

class ExecutionSnapshot final {
public:
    const core::GraphDefinition& graph_definition() const { return definition_; }
    const core::GraphDefinition& definition() const { return definition_; }
    std::shared_ptr<const core::CompiledGraph> compiled_plan() const { return plan_; }
    std::shared_ptr<const core::CompiledGraph> graph() const { return plan_; }
    std::shared_ptr<const core::InitialParameterValues> current_parameter_values() const {
        return current_values_;
    }
    std::shared_ptr<const core::InitialParameterValues> initial_parameter_values() const {
        return current_values_;
    }
    std::span<const SnapshotCellState> cell_states() const { return cell_states_; }
    std::span<const SnapshotCellParameterProvenance> cell_parameter_provenance() const {
        return cell_parameter_provenance_;
    }
    std::span<const SnapshotEdgeProvenance> edge_provenance() const {
        return edge_provenance_;
    }
    std::span<const SnapshotReadout> readouts() const { return readouts_; }

private:
    friend ExecutionSnapshotImportResult import_execution_snapshot(
        const CellularOrganism&, core::GraphIdentity, core::GraphRevision);

    ExecutionSnapshot(
        core::GraphDefinition definition,
        std::shared_ptr<const core::CompiledGraph> plan,
        std::shared_ptr<const core::InitialParameterValues> current_values,
        std::vector<SnapshotCellState> cell_states,
        std::vector<SnapshotCellParameterProvenance> cell_parameter_provenance,
        std::vector<SnapshotEdgeProvenance> edge_provenance,
        std::vector<SnapshotReadout> readouts)
        : definition_(std::move(definition)),
          plan_(std::move(plan)),
          current_values_(std::move(current_values)),
          cell_states_(std::move(cell_states)),
          cell_parameter_provenance_(std::move(cell_parameter_provenance)),
          edge_provenance_(std::move(edge_provenance)),
          readouts_(std::move(readouts)) {}

    core::GraphDefinition definition_;
    std::shared_ptr<const core::CompiledGraph> plan_;
    std::shared_ptr<const core::InitialParameterValues> current_values_;
    std::vector<SnapshotCellState> cell_states_;
    std::vector<SnapshotCellParameterProvenance> cell_parameter_provenance_;
    std::vector<SnapshotEdgeProvenance> edge_provenance_;
    std::vector<SnapshotReadout> readouts_;
};

struct ExecutionSnapshotImportResult {
    std::shared_ptr<const ExecutionSnapshot> snapshot;
    std::optional<ExecutionSnapshotImportError> error;

    bool ok() const { return snapshot != nullptr && !error.has_value(); }
    explicit operator bool() const { return ok(); }
};

namespace detail {

inline ExecutionSnapshotImportResult snapshot_failure(
    ExecutionSnapshotImportCode code,
    std::string reason,
    bool has_object_id = false,
    uint64_t object_id = 0) {
    return {
        nullptr,
        ExecutionSnapshotImportError{
            code, std::move(reason), has_object_id, object_id}};
}

inline bool finite_cell_state(const Cell& cell) {
    if (!std::isfinite(cell.state_val) || !std::isfinite(cell.aux_state) ||
        !std::isfinite(cell.prev_input) || !std::isfinite(cell.output_val) ||
        !std::isfinite(cell.prev_output_val)) {
        return false;
    }
    for (const double value : cell.delay_buffer) {
        if (!std::isfinite(value)) return false;
    }
    return cell.delay_idx < 16;
}

inline bool valid_channel_value(double value) {
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
        return false;
    }
    const long double exclusive_limit =
        std::ldexp(1.0L, std::numeric_limits<size_t>::digits);
    return static_cast<long double>(value) < exclusive_limit;
}

inline std::optional<core::ParameterValue> convert_parameter(
    const core::ParameterDescriptor& descriptor,
    double raw_value) {
    switch (descriptor.value_type) {
        case core::ParameterValueType::Continuous:
            if (descriptor.kind != core::ParameterKind::Continuous ||
                !std::isfinite(raw_value)) {
                return std::nullopt;
            }
            return core::ParameterValue{core::ContinuousValue{raw_value}};
        case core::ParameterValueType::ChannelIndex:
            if (descriptor.kind != core::ParameterKind::ChannelIndex ||
                !valid_channel_value(raw_value)) {
                return std::nullopt;
            }
            return core::ParameterValue{core::ChannelIndex{
                static_cast<size_t>(raw_value)}};
        case core::ParameterValueType::DelayTicks: {
            if (descriptor.kind != core::ParameterKind::DelayTicks ||
                !std::isfinite(raw_value)) {
                return std::nullopt;
            }
            int delay_ticks = 0;
            if (!sdsc_cell_kernel_normalize_delay(raw_value, &delay_ticks)) {
                return std::nullopt;
            }
            return core::ParameterValue{core::DelayTicks{
                static_cast<uint64_t>(delay_ticks)}};
        }
        case core::ParameterValueType::MinMaxMode:
            if (descriptor.kind != core::ParameterKind::Enum ||
                !std::isfinite(raw_value)) {
                return std::nullopt;
            }
            return core::ParameterValue{raw_value > 0.5
                                            ? core::MinMaxMode::Max
                                            : core::MinMaxMode::Min};
        case core::ParameterValueType::Unused:
            if (descriptor.kind != core::ParameterKind::Unused) {
                return std::nullopt;
            }
            return core::ParameterValue{core::UnusedParameter{}};
    }
    return std::nullopt;
}

inline bool advanced_organs_present(const CellularOrganism& organism) {
    return organism.has_mla() || organism.has_rope() ||
           organism.moe_router() != nullptr ||
           organism.speculative_engine() != nullptr ||
           organism.draft_organism() != nullptr ||
           !organism.moe_experts().empty() ||
           organism.relaxation_steps_ != 1 ||
           organism.relaxation_damping_ != 0.5;
}

inline bool valid_snapshot_readout_type(CellType type) {
    switch (type) {
        case CellType::ACT_PRIMARY_POSITIVE:
        case CellType::ACT_PRIMARY_NEGATIVE:
        case CellType::ACT_DEFENSIVE_RESET:
        case CellType::ACT_IMMUNE_BLOCK:
        case CellType::ACT_CHANNEL:
        case CellType::PREDICT_SENSE_0:
        case CellType::PREDICT_SENSE_1:
            return true;
        default:
            return false;
    }
}

inline uint64_t normalized_double_bits(double value) {
    if (value == 0.0) return 0;
    return std::bit_cast<uint64_t>(value);
}

using RawEdgeKey = std::tuple<uint32_t, uint32_t, uint8_t, uint64_t>;

struct RawEdgeIndexEntry {
    RawEdgeKey key;
    size_t raw_index{0};
};

}  // namespace detail

inline ExecutionSnapshotImportResult import_execution_snapshot(
    const CellularOrganism& organism,
    core::GraphIdentity identity,
    core::GraphRevision revision) {
    if (!organism.is_compiled_) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::NotCompiled,
            "legacy source is not compiled; import never compiles a source copy");
    }
    if (detail::advanced_organs_present(organism)) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::UnsupportedAdvancedOrgan,
            "snapshot domain excludes MLA, RoPE, MoE, speculative, and relaxation organs");
    }

    const size_t cell_count = organism.cells.size();
    std::unordered_set<uint32_t> cell_ids;
    cell_ids.reserve(cell_count);
    std::vector<size_t> execution_rank(cell_count, cell_count);
    for (size_t index = 0; index < cell_count; ++index) {
        const auto& cell = organism.cells[index];
        if (!cell_ids.insert(cell.id).second) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCellMetadata,
                "declared cells contain duplicate IDs", true, cell.id);
        }
        if (!core::contract_for(cell.type).has_value()) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCellMetadata,
                "declared cell type is absent from the typed contract", true, cell.id);
        }
        if (!std::isfinite(cell.param1) || !std::isfinite(cell.param2)) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidParameter,
                "declared cell parameters must be finite", true, cell.id);
        }
        if (!detail::finite_cell_state(cell)) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidComputationalState,
                "cell computational state contains a non-finite value or invalid delay index",
                true, cell.id);
        }
    }

    if (cell_count == 0) {
        if (!organism.execution_order_.empty() ||
            !organism.compiled_synapses_.empty() ||
            !organism.compiled_actions_.empty() ||
            !organism.out_start_.empty() ||
            !organism.out_edges_.empty()) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidExecutionOrder,
                "empty compiled source contains non-empty execution metadata");
        }
    } else {
        if (organism.execution_order_.size() != cell_count) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::UnsupportedPrunedExecution,
                "legacy execution order omits one or more declared cells");
        }
        for (size_t rank = 0; rank < organism.execution_order_.size(); ++rank) {
            const size_t index = organism.execution_order_[rank];
            if (index >= cell_count || execution_rank[index] != cell_count) {
                return detail::snapshot_failure(
                    ExecutionSnapshotImportCode::InvalidExecutionOrder,
                    "execution order contains an out-of-range or duplicate cell index");
            }
            execution_rank[index] = rank;
        }
    }

    std::vector<SnapshotReadout> readouts;
    std::unordered_set<size_t> action_indices;
    readouts.reserve(organism.compiled_actions_.size());
    for (const auto& action : organism.compiled_actions_) {
        if (action.cell_idx >= cell_count ||
            organism.cells[action.cell_idx].type != action.type ||
            !detail::valid_snapshot_readout_type(action.type) ||
            !action_indices.insert(action.cell_idx).second) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidActionMetadata,
                "compiled action metadata has an invalid, mismatched, or duplicate cell index");
        }
        readouts.push_back(SnapshotReadout{
            core::CellId{organism.cells[action.cell_idx].id}, action.type});
    }

    if ((cell_count != 0 &&
         (organism.out_start_.size() != cell_count + 1 ||
          organism.out_start_.front() != 0 ||
          organism.out_start_.back() != organism.compiled_synapses_.size())) ||
        organism.out_edges_.size() != organism.compiled_synapses_.size()) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::InvalidCsr,
            "compiled outgoing CSR has invalid dimensions or terminal range");
    }
    for (size_t index = 1; index < organism.out_start_.size(); ++index) {
        if (organism.out_start_[index - 1] > organism.out_start_[index]) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCsr,
                "compiled outgoing CSR ranges are not monotonic");
        }
    }
    std::vector<uint8_t> edge_references(organism.compiled_synapses_.size(), 0);
    for (size_t cell_index = 0; cell_index < cell_count; ++cell_index) {
        for (size_t cursor = organism.out_start_[cell_index];
             cursor < organism.out_start_[cell_index + 1]; ++cursor) {
            const size_t edge_index = organism.out_edges_[cursor];
            if (edge_index >= organism.compiled_synapses_.size() ||
                organism.compiled_synapses_[edge_index].from_idx != cell_index ||
                ++edge_references[edge_index] != 1) {
                return detail::snapshot_failure(
                    ExecutionSnapshotImportCode::InvalidCsr,
                    "compiled outgoing CSR does not reference each edge exactly once by source bucket");
            }
        }
    }
    for (const uint8_t references : edge_references) {
        if (references != 1) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCsr,
                "compiled outgoing CSR omits a compiled edge");
        }
    }

    for (size_t edge_index = 0;
         edge_index < organism.compiled_synapses_.size(); ++edge_index) {
        const auto& edge = organism.compiled_synapses_[edge_index];
        if (edge.from_idx >= cell_count || edge.to_idx >= cell_count ||
            edge.to_port > 1 || !std::isfinite(edge.weight) ||
            !std::isfinite(edge.initial_weight) ||
            !std::isfinite(edge.hebbian_rate) ||
            !std::isfinite(edge.hebbian_decay)) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCompiledEdge,
                "compiled edge has an invalid endpoint, port, weight, or plasticity value",
                true, edge_index);
        }
        const auto target_contract =
            core::contract_for(organism.cells[edge.to_idx].type);
        if (!target_contract.has_value() ||
            edge.to_port >= target_contract->get().input_port_count) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::UnsupportedIgnoredPort,
                "compiled edge targets a port not consumed by the typed cell contract",
                true, edge_index);
        }
        if (!edge.is_recurrent &&
            execution_rank[edge.from_idx] >= execution_rank[edge.to_idx]) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::InvalidCompiledEdge,
                "nonrecurrent compiled edge is not strictly forward in execution order",
                true, edge_index);
        }
    }

    for (size_t raw_index = 0; raw_index < organism.synapses.size(); ++raw_index) {
        const auto& raw = organism.synapses[raw_index];
        if (!raw.is_active) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::UnsupportedInactiveEdge,
                "inactive raw synapses are outside the first snapshot domain",
                true, raw_index);
        }
        if (!std::isfinite(raw.weight) || !std::isfinite(raw.initial_weight) ||
            !std::isfinite(raw.hebbian_rate) ||
            !std::isfinite(raw.hebbian_decay)) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::EdgeProvenanceMismatch,
                "raw edge provenance contains a non-finite weight or plasticity value",
                true, raw_index);
        }
    }
    if (organism.synapses.size() != organism.compiled_synapses_.size()) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::EdgeProvenanceMismatch,
            "active raw and compiled edge counts differ");
    }

    std::vector<size_t> compiled_to_raw(
        organism.compiled_synapses_.size(), organism.synapses.size());
    std::vector<uint8_t> raw_used(organism.synapses.size(), 0);
    std::vector<detail::RawEdgeIndexEntry> raw_edge_index;
    raw_edge_index.reserve(organism.synapses.size());
    for (size_t raw_index = 0; raw_index < organism.synapses.size(); ++raw_index) {
        const auto& raw = organism.synapses[raw_index];
        raw_edge_index.push_back(detail::RawEdgeIndexEntry{
            detail::RawEdgeKey{
                raw.from_cell_id,
                raw.to_cell_id,
                raw.to_port,
                detail::normalized_double_bits(raw.weight)},
            raw_index});
    }
    std::sort(
        raw_edge_index.begin(), raw_edge_index.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.key < rhs.key; });

    for (size_t compiled_index = 0;
         compiled_index < organism.compiled_synapses_.size(); ++compiled_index) {
        const auto& compiled = organism.compiled_synapses_[compiled_index];
        const uint32_t source_id = organism.cells[compiled.from_idx].id;
        const uint32_t target_id = organism.cells[compiled.to_idx].id;
        const detail::RawEdgeKey key{
            source_id,
            target_id,
            compiled.to_port,
            detail::normalized_double_bits(compiled.initial_weight)};
        const auto first = std::lower_bound(
            raw_edge_index.begin(), raw_edge_index.end(), key,
            [](const auto& entry, const auto& sought) {
                return entry.key < sought;
            });
        size_t candidate = organism.synapses.size();
        size_t candidate_count = 0;
        for (auto it = first;
             it != raw_edge_index.end() && !(key < it->key); ++it) {
            candidate = it->raw_index;
            ++candidate_count;
        }
        if (candidate_count > 1) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::AmbiguousParallelEdgeProvenance,
                "parallel raw edges share the compile-time weight anchor; ancestry cannot be proven",
                true, compiled_index);
        }
        if (candidate_count == 0 || raw_used[candidate] != 0) {
            return detail::snapshot_failure(
                ExecutionSnapshotImportCode::EdgeProvenanceMismatch,
                "compiled edge has no unique raw compile-time provenance anchor",
                true, compiled_index);
        }
        raw_used[candidate] = 1;
        compiled_to_raw[compiled_index] = candidate;
    }

    core::GraphDefinition definition;
    definition.identity = identity;
    definition.revision = revision;
    definition.profile = SemanticProfile::LegacyCompatible;
    definition.semantic_version = 1;
    definition.cells.reserve(cell_count);
    core::InitialParameterSeeds seeds;
    seeds.cell_parameters.reserve(cell_count * 2);

    std::vector<SnapshotCellState> cell_states;
    std::vector<SnapshotCellParameterProvenance> cell_parameter_provenance;
    cell_states.reserve(cell_count);
    cell_parameter_provenance.reserve(cell_count * 2);
    for (const auto& cell : organism.cells) {
        const core::CellId id{cell.id};
        definition.cells.push_back(core::CellDefinition{id, cell.type});
        cell_states.push_back(SnapshotCellState{
            id, cell.type, cell.state_val, cell.aux_state, cell.prev_input,
            cell.output_val, cell.prev_output_val,
            [&cell] {
                std::array<double, 16> values{};
                std::copy(std::begin(cell.delay_buffer),
                          std::end(cell.delay_buffer), values.begin());
                return values;
            }(),
            cell.delay_idx, cell.latch_state, cell.activation_count});

        const auto contract = core::contract_for(cell.type)->get();
        const std::array<double, 2> raw_parameters{cell.param1, cell.param2};
        for (size_t slot_index = 0; slot_index < raw_parameters.size(); ++slot_index) {
            const auto slot = static_cast<core::ParameterSlot>(slot_index);
            const auto value = detail::convert_parameter(
                contract.parameters[slot_index], raw_parameters[slot_index]);
            if (!value.has_value() ||
                !core::validate_parameter(contract.parameters[slot_index], *value)) {
                return detail::snapshot_failure(
                    ExecutionSnapshotImportCode::InvalidParameter,
                    "cell parameter cannot be represented by the typed contract",
                    true, cell.id);
            }
            seeds.cell_parameters.push_back(
                core::CellParameterSeed{id, slot, *value});
            cell_parameter_provenance.push_back(
                SnapshotCellParameterProvenance{
                    id, slot, raw_parameters[slot_index], true, false});
        }
    }

    std::vector<size_t> emission_order;
    emission_order.reserve(organism.compiled_synapses_.size());
    std::vector<uint8_t> emitted(organism.compiled_synapses_.size(), 0);
    for (size_t compiled_index = 0;
         compiled_index < organism.compiled_synapses_.size(); ++compiled_index) {
        if (organism.compiled_synapses_[compiled_index].is_recurrent) {
            emission_order.push_back(compiled_index);
            emitted[compiled_index] = 1;
        }
    }
    for (const size_t cell_index : organism.execution_order_) {
        for (size_t cursor = organism.out_start_[cell_index];
             cursor < organism.out_start_[cell_index + 1]; ++cursor) {
            const size_t compiled_index = organism.out_edges_[cursor];
            if (!organism.compiled_synapses_[compiled_index].is_recurrent) {
                if (emitted[compiled_index] != 0) {
                    return detail::snapshot_failure(
                        ExecutionSnapshotImportCode::InvalidCsr,
                        "old edge emission order references an immediate edge more than once");
                }
                emission_order.push_back(compiled_index);
                emitted[compiled_index] = 1;
            }
        }
    }
    if (emission_order.size() != organism.compiled_synapses_.size() ||
        std::any_of(emitted.begin(), emitted.end(),
                    [](uint8_t value) { return value == 0; })) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::InvalidCsr,
            "old recurrent/immediate emission order does not cover every edge");
    }

    std::vector<SnapshotEdgeProvenance> edge_provenance;
    edge_provenance.reserve(emission_order.size());
    seeds.edge_weights.reserve(emission_order.size());
    definition.edges.reserve(emission_order.size());
    for (size_t emission_index = 0; emission_index < emission_order.size();
         ++emission_index) {
        const size_t compiled_index = emission_order[emission_index];
        const auto& compiled = organism.compiled_synapses_[compiled_index];
        const size_t raw_index = compiled_to_raw[compiled_index];
        const auto& raw = organism.synapses[raw_index];
        const core::EdgeId edge_id{
            static_cast<uint64_t>(emission_index) + 1};
        definition.edges.push_back(core::EdgeDefinition{
            edge_id,
            core::CellId{organism.cells[compiled.from_idx].id},
            core::OutputPort{0},
            core::CellId{organism.cells[compiled.to_idx].id},
            core::InputPort{compiled.to_port},
            compiled.is_recurrent ? core::EdgeDelay::PreviousTick
                                  : core::EdgeDelay::Immediate});
        seeds.edge_weights.push_back(
            core::EdgeParameterSeed{edge_id, compiled.weight});
        edge_provenance.push_back(SnapshotEdgeProvenance{
            edge_id,
            core::CellId{organism.cells[compiled.from_idx].id},
            core::CellId{organism.cells[compiled.to_idx].id},
            core::InputPort{compiled.to_port},
            compiled.is_recurrent,
            compiled.weight,
            compiled.initial_weight,
            raw.initial_weight,
            compiled.hebbian_rate,
            compiled.hebbian_decay,
            raw_index});
    }

    const core::CompileResult compiled =
        core::GraphCompiler{}.compile(definition, seeds);
    if (!compiled.ok()) {
        return detail::snapshot_failure(
            ExecutionSnapshotImportCode::CompilerRejected,
            compiled.error.has_value()
                ? compiled.error->diagnostic()
                : "typed graph compiler rejected imported execution snapshot");
    }

    auto snapshot = std::shared_ptr<const ExecutionSnapshot>(
        new ExecutionSnapshot(
            std::move(definition),
            compiled.graph,
            compiled.initial_values,
            std::move(cell_states),
            std::move(cell_parameter_provenance),
            std::move(edge_provenance),
            std::move(readouts)));
    return {std::move(snapshot), std::nullopt};
}

}  // namespace kun::migration
