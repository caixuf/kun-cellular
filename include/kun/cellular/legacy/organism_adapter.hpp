#pragma once

#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/core/heredity.hpp"
#include "kun/cellular/core/semantic_profile.hpp"
#include "kun/cellular/legacy/execution_snapshot.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kun::migration {

enum class MigrationOperation : uint8_t {
    ImportBirthTemplate = 0,
    RestoreCheckpoint = 1,
    ExportCheckpoint = 2,
};

enum class MigrationArtifactKind : uint8_t {
    LegacyGenome = 0,
    LegacyCheckpoint = 1,
    StrictCoreGraph = 2,
};

enum class MigrationPreflightCode : uint8_t {
    Accepted = 0,
    UnknownProfile = 1,
    UnsupportedProfile = 2,
    InvalidOrganism = 3,
    UnsupportedArtifact = 4,
    RuntimeStateUnavailable = 5,
    UnsupportedFeature = 6,
    UnsupportedFormat = 7,
    InvalidRequest = 8,
};

struct MigrationPreflightResult {
    MigrationPreflightCode code{MigrationPreflightCode::Accepted};
    const char* diagnostic{"accepted"};

    constexpr bool accepted() const {
        return code == MigrationPreflightCode::Accepted;
    }
};

struct LegacyGermlineImportResult {
    std::shared_ptr<const core::Germline> germline;
    MigrationPreflightResult preflight{};
    std::string reason;

    bool ok() const { return germline != nullptr && preflight.accepted(); }
    explicit operator bool() const { return ok(); }
};

// Request metadata is intentionally independent of the inspected object. This
// keeps the preflight safe when callers construct an organism temporarily.
struct MigrationPreflightRequest {
    MigrationOperation operation{MigrationOperation::ImportBirthTemplate};
    MigrationArtifactKind artifact_kind{MigrationArtifactKind::LegacyGenome};
    SemanticProfile profile{SemanticProfile::LegacyCompatible};
    uint32_t checkpoint_version{3};
};

inline bool is_valid_channel_index_parameter(double value) {
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
        return false;
    }

    // A double at or above 2^digits cannot be represented by size_t. The
    // range check deliberately happens before any conversion to an index.
    const long double exclusive_limit =
        std::ldexp(1.0L, std::numeric_limits<size_t>::digits);
    return static_cast<long double>(value) < exclusive_limit;
}

inline bool has_actual_cycle(const CellularOrganism& organism) {
    std::unordered_map<uint32_t, size_t> id_to_index;
    id_to_index.reserve(organism.cells.size());
    for (size_t i = 0; i < organism.cells.size(); ++i) {
        id_to_index.emplace(organism.cells[i].id, i);
    }

    std::vector<size_t> indegrees(organism.cells.size(), 0);
    std::vector<std::vector<size_t>> adjacency(organism.cells.size());
    for (const auto& synapse : organism.synapses) {
        if (!synapse.is_active) {
            continue;
        }
        const auto from = id_to_index.find(synapse.from_cell_id);
        const auto to = id_to_index.find(synapse.to_cell_id);
        if (from == id_to_index.end() || to == id_to_index.end()) {
            continue;
        }
        adjacency[from->second].push_back(to->second);
        ++indegrees[to->second];
    }

    std::vector<size_t> queue;
    queue.reserve(organism.cells.size());
    for (size_t i = 0; i < indegrees.size(); ++i) {
        if (indegrees[i] == 0) {
            queue.push_back(i);
        }
    }

    size_t processed = 0;
    for (size_t head = 0; head < queue.size(); ++head) {
        const size_t node = queue[head];
        ++processed;
        for (const size_t next : adjacency[node]) {
            if (--indegrees[next] == 0) {
                queue.push_back(next);
            }
        }
    }
    return processed != organism.cells.size();
}

inline MigrationPreflightResult validate_legacy_birth_template(
    const CellularOrganism& organism) {
    if (organism.cells.empty()) {
        return {MigrationPreflightCode::InvalidOrganism,
                "birth-template inspection rejects an empty organism"};
    }

    std::unordered_set<uint32_t> cell_ids;
    cell_ids.reserve(organism.cells.size());
    for (const auto& cell : organism.cells) {
        if (!cell_ids.insert(cell.id).second) {
            return {MigrationPreflightCode::InvalidOrganism,
                    "birth template contains duplicate cell IDs"};
        }
        if (!is_valid_cell_type_code(static_cast<uint8_t>(cell.type))) {
            return {MigrationPreflightCode::InvalidOrganism,
                    "birth template contains an unknown cell type"};
        }
        if (!std::isfinite(cell.param1) || !std::isfinite(cell.param2) ||
            !std::isfinite(cell.x) || !std::isfinite(cell.y) ||
            !std::isfinite(cell.z)) {
            return {MigrationPreflightCode::InvalidOrganism,
                    "birth template contains a non-finite gene parameter or position"};
        }
        if ((cell.type == CellType::SENSE_CHANNEL ||
             cell.type == CellType::ACT_CHANNEL) &&
            !is_valid_channel_index_parameter(cell.param2)) {
            return {MigrationPreflightCode::InvalidOrganism,
                    "channel parameter must be a non-negative representable integer"};
        }
    }

    for (const auto& synapse : organism.synapses) {
        if (!cell_ids.count(synapse.from_cell_id) ||
            !cell_ids.count(synapse.to_cell_id) ||
            synapse.to_port > 1 ||
            !std::isfinite(synapse.weight) ||
            !std::isfinite(synapse.initial_weight) ||
            !std::isfinite(synapse.hebbian_rate) ||
            !std::isfinite(synapse.hebbian_decay) ||
            !std::isfinite(synapse.rest_length) ||
            !std::isfinite(synapse.photon_pos)) {
            return {MigrationPreflightCode::InvalidOrganism,
                    "birth template contains an invalid reference, port, or parameter"};
        }
        if (synapse.is_recurrent) {
            return {MigrationPreflightCode::UnsupportedFeature,
                    "birth-template scope supports ordinary DAGs, not recurrent metadata"};
        }
    }

    if (organism.mla_engine_ || organism.rope_table_ || organism.moe_router_ ||
        organism.speculative_engine_ || organism.draft_organism_ ||
        !organism.moe_experts_.empty() || organism.relaxation_steps_ != 1 ||
        organism.relaxation_damping_ != 0.5) {
        return {MigrationPreflightCode::UnsupportedFeature,
                "birth-template scope excludes advanced runtime organs and settings"};
    }

    if (has_actual_cycle(organism)) {
        return {MigrationPreflightCode::UnsupportedFeature,
                "birth-template scope supports ordinary DAG topology only"};
    }
    return {};
}

inline MigrationPreflightResult validate_migration_preflight(
    const CellularOrganism& organism,
    const MigrationPreflightRequest& request) {
    const auto parsed_profile = semantic_profile_from_code(
        static_cast<uint8_t>(request.profile));
    if (!parsed_profile.has_value()) {
        return {MigrationPreflightCode::UnknownProfile,
                "semantic profile code is not recognized"};
    }
    if (*parsed_profile == SemanticProfile::StrictCore) {
        return {MigrationPreflightCode::UnsupportedProfile,
                "StrictCore is declared but strict migration is not implemented"};
    }

    if (request.operation != MigrationOperation::ImportBirthTemplate &&
        request.operation != MigrationOperation::RestoreCheckpoint &&
        request.operation != MigrationOperation::ExportCheckpoint) {
        return {MigrationPreflightCode::InvalidRequest,
                "migration operation code is not recognized"};
    }
    if (request.artifact_kind != MigrationArtifactKind::LegacyGenome &&
        request.artifact_kind != MigrationArtifactKind::LegacyCheckpoint &&
        request.artifact_kind != MigrationArtifactKind::StrictCoreGraph) {
        return {MigrationPreflightCode::UnsupportedArtifact,
                "migration artifact kind is not recognized"};
    }

    if (request.operation == MigrationOperation::ImportBirthTemplate) {
        if (request.artifact_kind == MigrationArtifactKind::StrictCoreGraph) {
            return {MigrationPreflightCode::UnsupportedArtifact,
                    "StrictCoreGraph migration is reserved for the strict compiler"};
        }
        if (request.artifact_kind != MigrationArtifactKind::LegacyGenome) {
            return {MigrationPreflightCode::InvalidRequest,
                    "birth-template inspection requires a LegacyGenome artifact"};
        }
        return validate_legacy_birth_template(organism);
    }

    if (request.operation == MigrationOperation::RestoreCheckpoint) {
        if (request.artifact_kind != MigrationArtifactKind::LegacyCheckpoint) {
            return {MigrationPreflightCode::InvalidRequest,
                    "checkpoint restoration requires a LegacyCheckpoint artifact"};
        }
        if (request.checkpoint_version != 2 &&
            request.checkpoint_version != 3) {
            return {MigrationPreflightCode::UnsupportedFormat,
                    "only committed legacy checkpoint metadata versions v2 and v3 are known"};
        }
        return {MigrationPreflightCode::RuntimeStateUnavailable,
                "R0 does not restore complete runtime checkpoint state"};
    }

    if (request.artifact_kind != MigrationArtifactKind::LegacyCheckpoint) {
        return {MigrationPreflightCode::InvalidRequest,
                "checkpoint export requires a LegacyCheckpoint artifact"};
    }
    return {MigrationPreflightCode::UnsupportedFeature,
            "R0 does not prove or provide lossless binary export"};
}

// Narrow generic migration seam: legacy CellularOrganism is consumed as a
// birth template only. Live weights, runtime memory, resources, and legacy
// checkpoint state are intentionally excluded from the new germline.
inline LegacyGermlineImportResult import_legacy_birth_template_germline(
    const CellularOrganism& organism,
    core::GraphIdentity identity,
    core::GraphRevision revision) {
    const auto preflight = validate_migration_preflight(
        organism,
        MigrationPreflightRequest{
            MigrationOperation::ImportBirthTemplate,
            MigrationArtifactKind::LegacyGenome,
            SemanticProfile::LegacyCompatible,
            0});
    if (!preflight.accepted()) return {nullptr, preflight, preflight.diagnostic};

    core::GraphDefinition definition{
        identity, revision, SemanticProfile::LegacyCompatible, 1, {}, {}};
    core::InitialParameterSeeds seeds;
    for (const auto& cell : organism.cells) {
        definition.cells.push_back(
            core::CellDefinition{
                core::CellId{cell.id},
                static_cast<CellType>(cell.type)});
        const auto contract = core::contract_for(
            static_cast<CellType>(cell.type));
        if (!contract.has_value()) {
            return {nullptr,
                    {MigrationPreflightCode::InvalidOrganism,
                     "legacy cell type has no generated core contract"},
                    "legacy cell type has no generated core contract"};
        }
        const double raw_values[2] = {cell.param1, cell.param2};
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            core::ParameterValue value = core::UnusedParameter{};
            switch (descriptor.value_type) {
                case core::ParameterValueType::Continuous:
                    value = core::ContinuousValue{raw_values[slot]};
                    break;
                case core::ParameterValueType::ChannelIndex:
                    value = core::ChannelIndex{
                        static_cast<std::size_t>(raw_values[slot])};
                    break;
                case core::ParameterValueType::DelayTicks:
                    value = core::DelayTicks{
                        static_cast<uint64_t>(std::max(0.0, raw_values[slot]))};
                    break;
                case core::ParameterValueType::MinMaxMode:
                    value = raw_values[slot] > 0.5
                        ? core::ParameterValue{core::MinMaxMode::Max}
                        : core::ParameterValue{core::MinMaxMode::Min};
                    break;
                case core::ParameterValueType::Unused:
                    break;
            }
            seeds.cell_parameters.push_back(
                core::CellParameterSeed{
                    core::CellId{cell.id},
                    static_cast<core::ParameterSlot>(slot),
                    value});
        }
    }
    for (const auto& synapse : organism.synapses) {
        if (!synapse.is_active) continue;
        definition.edges.push_back(
            core::EdgeDefinition{
                core::EdgeId{definition.edges.size()},
                core::CellId{synapse.from_cell_id},
                core::OutputPort{0},
                core::CellId{synapse.to_cell_id},
                core::InputPort{synapse.to_port},
                core::EdgeDelay::Immediate});
        seeds.edge_weights.push_back(
            core::EdgeParameterSeed{
                definition.edges.back().id, synapse.initial_weight});
    }
    const auto germline = core::Germline::create(
        std::move(definition),
        std::move(seeds),
        "legacy-birth-template-import");
    if (!germline.ok()) {
        return {
            nullptr,
            {MigrationPreflightCode::InvalidOrganism,
             "legacy template could not compile as a core germline"},
            germline.error ? germline.error->reason
                           : "legacy template could not compile"};
    }
    return {std::move(germline.germline), preflight, {}};
}

}  // namespace kun::migration
