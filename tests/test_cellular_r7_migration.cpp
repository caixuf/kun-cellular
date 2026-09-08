#include "kun/cellular/legacy/organism_adapter.hpp"
#include "kun/cellular/legacy/support_manifest.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <span>

using namespace kun;
using namespace kun::migration;
using namespace kun::core;

namespace {

CellularOrganism make_generic_legacy_consumer() {
    CellularOrganism organism;
    organism.cells = {
        Cell{1, CellType::SENSE_RAW_INPUT_0, 1.0},
        Cell{2, CellType::OP_ABS, 0.0}};
    organism.synapses = {
        Synapse{1, 2, 0, 1.0, true, 60.0f, -1.0f}};
    organism.synapses[0].initial_weight = 1.0;
    assert(organism.compile());
    return organism;
}

OffspringSpec offspring_spec() {
    OffspringSpec spec;
    spec.organism_id = 77;
    spec.rng_seed = 77;
    spec.lifecycle_config.apoptotic_resource = 0.0;
    spec.lifecycle_config.dormant_enter_resource = 0.0;
    spec.lifecycle_config.dormant_exit_resource = 1.0;
    spec.resource_config.activity_scale = 1.0;
    spec.resource_config.transmission_scale = 1.0;
    spec.resource_compartments.push_back(
        ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0});
    spec.resource_cells = {
        {CellId{1}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0},
        {CellId{2}, ResourceCompartmentId{1}, 10.0, 10.0, 0.0}};
    return spec;
}

void test_generic_legacy_template_migrates_without_live_sync() {
    auto legacy = make_generic_legacy_consumer();
    const auto before_weight = legacy.synapses[0].weight;
    const auto before_state = legacy.cells[1].state_val;
    const auto imported = import_legacy_birth_template_germline(
        legacy, GraphIdentity{9500}, GraphRevision{1});
    assert(imported.ok());
    assert(imported.germline->graph()->cells().size() == 2);
    assert(imported.germline->graph()->edges().size() == 1);

    auto phenotype_result = imported.germline->spawn_offspring(offspring_spec());
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const double input = -2.0;
    const auto step = phenotype->step({&input, 1}, {});
    assert(step.ok());
    assert(step.measurement.has_value());
    const auto target = std::find_if(
        step.measurement->cells.begin(),
        step.measurement->cells.end(),
        [](const auto& cell) { return cell.cell == CellId{2}; });
    assert(target != step.measurement->cells.end());
    assert(std::isfinite(target->output));
    assert(phenotype->runtime().parameters().size() == 5);

    assert(legacy.synapses[0].weight == before_weight);
    assert(legacy.cells[1].state_val == before_state);
}

void test_legacy_malformed_template_and_support_boundaries_are_explicit() {
    auto malformed = make_generic_legacy_consumer();
    malformed.cells.push_back(malformed.cells.front());
    const auto rejected = import_legacy_birth_template_germline(
        malformed, GraphIdentity{9501}, GraphRevision{1});
    assert(!rejected.ok());
    assert(rejected.preflight.code == MigrationPreflightCode::InvalidOrganism);

    assert(support_status(SupportCapability::CppLifecycleRuntime).supported);
    assert(support_status(
        SupportCapability::CppFullLifecycleCheckpoint).supported);
    assert(support_status(SupportCapability::FrozenC11GraphRuntime).supported);
    assert(support_status(
        SupportCapability::LegacyGenomeTemplateImport).supported);
    assert(!support_status(
        SupportCapability::LegacyCheckpointFullRestore).supported);
    assert(!support_status(
        SupportCapability::OptimizerCheckpointRestore).supported);
}

}  // namespace

int main() {
    test_generic_legacy_template_migrates_without_live_sync();
    test_legacy_malformed_template_and_support_boundaries_are_explicit();
    return 0;
}
