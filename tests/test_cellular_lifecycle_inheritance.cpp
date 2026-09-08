#include "kun/cellular/core/cellular_graph_edit.hpp"
#include "kun/cellular/core/graph_compiler.hpp"
#include "kun/cellular/core/heredity.hpp"
#include "kun/cellular/core/lifecycle_persistence.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

using namespace kun;
using namespace kun::core;

namespace {

GraphDefinition make_graph() {
    return GraphDefinition{
        GraphIdentity{9300},
        GraphRevision{1},
        SemanticProfile::StrictCore,
        1,
        {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
         {CellId{2}, CellType::OP_EMA},
         {CellId{3}, CellType::OP_ABS}},
        {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
          EdgeDelay::Immediate},
         {EdgeId{11}, CellId{2}, OutputPort{0}, CellId{3}, InputPort{0},
          EdgeDelay::PreviousTick}}};
}

InitialParameterSeeds make_seeds(const GraphDefinition& graph) {
    InitialParameterSeeds seeds;
    for (const auto& cell : graph.cells) {
        const auto contract = contract_for(cell.type);
        assert(contract.has_value());
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const auto& descriptor = contract->get().parameters[slot];
            ParameterValue value = UnusedParameter{};
            switch (descriptor.value_type) {
                case ParameterValueType::Continuous:
                    value = ContinuousValue{
                        descriptor.name &&
                                std::string_view(descriptor.name) == "alpha"
                            ? 0.5
                            : 1.0};
                    break;
                case ParameterValueType::ChannelIndex:
                    value = ChannelIndex{0};
                    break;
                case ParameterValueType::DelayTicks:
                    value = DelayTicks{1};
                    break;
                case ParameterValueType::MinMaxMode:
                    value = MinMaxMode::Min;
                    break;
                case ParameterValueType::Unused:
                    break;
            }
            seeds.cell_parameters.push_back(
                CellParameterSeed{
                    cell.id, static_cast<ParameterSlot>(slot), value});
        }
    }
    for (const auto& edge : graph.edges) {
        seeds.edge_weights.push_back(EdgeParameterSeed{edge.id, 1.0});
    }
    return seeds;
}

OffspringSpec offspring_spec(uint64_t seed) {
    OffspringSpec spec;
    spec.organism_id = seed;
    spec.rng_seed = seed;
    spec.lifecycle_config.apoptotic_resource = 0.0;
    spec.lifecycle_config.dormant_enter_resource = 0.0;
    spec.lifecycle_config.dormant_exit_resource = 1.0;
    spec.resource_config.dt = 1.0;
    spec.resource_config.transmission_scale = 1.0;
    spec.resource_config.activity_scale = 1.0;
    for (const CellId id : {CellId{1}, CellId{2}, CellId{3}}) {
        spec.resource_cells.push_back(
            ResourceCellInitial{
                id, ResourceCompartmentId{1}, 10.0, 10.0, 0.0});
    }
    spec.resource_compartments.push_back(
        ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0});
    return spec;
}

std::shared_ptr<const Germline> make_germline() {
    auto result = Germline::create(
        make_graph(), make_seeds(make_graph()), "r5-development-v1");
    assert(result.ok());
    return std::move(result.germline);
}

void test_checkpoint_accepts_graph_over_sixty_four_cells() {
    GraphDefinition graph{
        GraphIdentity{9301},
        GraphRevision{1},
        SemanticProfile::StrictCore,
        1,
        {},
        {}};
    for (uint64_t id = 1; id <= 70; ++id) {
        graph.cells.push_back(
            CellDefinition{CellId{id}, CellType::OP_ABS});
    }
    auto germline_result = Germline::create(
        graph, make_seeds(graph), "r5-large-template");
    assert(germline_result.ok());
    OffspringSpec spec;
    spec.organism_id = 700;
    spec.rng_seed = 700;
    spec.lifecycle_config.apoptotic_resource = 0.0;
    spec.lifecycle_config.dormant_enter_resource = 0.0;
    spec.lifecycle_config.dormant_exit_resource = 1.0;
    spec.resource_config.transmission_scale = 1.0;
    spec.resource_config.activity_scale = 1.0;
    spec.resource_compartments.push_back(
        ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0});
    for (const auto& cell : graph.cells) {
        spec.resource_cells.push_back(
            ResourceCellInitial{
                cell.id, ResourceCompartmentId{1}, 10.0, 10.0, 0.0});
    }
    auto phenotype_result = germline_result.germline->spawn_offspring(spec);
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const auto checkpoint = LifecyclePersistence::save(*phenotype);
    assert(checkpoint.ok());
    auto restored_result = germline_result.germline->spawn_offspring(spec);
    assert(restored_result.ok());
    auto restored = std::move(restored_result.phenotype);
    assert(LifecyclePersistence::restore_into(
        *restored, checkpoint.bytes).ok());
    assert(restored->runtime().plan()->cells().size() == 70);
}

void test_default_offspring_is_fresh_and_template_only() {
    const auto germline = make_germline();
    auto parent_result = germline->spawn_offspring(offspring_spec(100));
    assert(parent_result.ok());
    auto parent = std::move(parent_result.phenotype);
    const auto edge = parent->runtime().parameters()[4].binding;
    assert(parent->runtime().set_parameter(
        edge, ParameterValue{ContinuousValue{3.5}}).ok());
    const double input = 2.0;
    assert(parent->step({&input, 1}, {}).ok());
    assert(parent->runtime().tick() == 1);

    auto child_result = germline->spawn_offspring(offspring_spec(200));
    assert(child_result.ok());
    auto child = std::move(child_result.phenotype);
    assert(child->runtime().tick() == 0);
    assert(std::get<ContinuousValue>(
               *child->runtime().parameter_at(edge.index))
               .value == 1.0);
    assert(child->runtime().cell_state(CellId{2})->state_val == 0.0);
    assert(child->ledger().snapshot().cell(CellId{1})->energy == 10.0);
    assert(parent->ledger().snapshot().cell(CellId{1})->energy != 10.0 ||
           parent->runtime().tick() != child->runtime().tick());
    assert(parent->next_random() != child->next_random());
}

void test_assimilation_is_explicit_and_declared() {
    const auto germline = make_germline();
    auto phenotype_result = germline->spawn_offspring(offspring_spec(300));
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const auto edge = phenotype->runtime().parameters()[4].binding;
    assert(phenotype->runtime().set_parameter(
        edge, ParameterValue{ContinuousValue{4.25}}).ok());

    const std::array<AssimilationField, 1> declared{
        AssimilationField::LiveParameters};
    const auto assimilated = germline->assimilate(
        *phenotype,
        std::span<const AssimilationField>(
            declared.data(), declared.size()));
    assert(assimilated.ok());
    assert(assimilated.report.version_after.value ==
           germline->version().value + 1);
    assert(assimilated.report.copied_fields ==
           AssimilationFieldMask::LiveParameters);
    assert(std::get<ContinuousValue>(
               assimilated.germline->initial_values().at(edge.index).value)
               .value == 4.25);

    const std::array<AssimilationField, 1> forbidden{
        AssimilationField::RuntimeMemory};
    const auto rejected = germline->assimilate(
        *phenotype,
        std::span<const AssimilationField>(
            forbidden.data(), forbidden.size()));
    assert(!rejected.ok());
    assert(rejected.error->code == HeredityErrorCode::UndeclaredAssimilation);
}

void test_learning_window_rejects_structural_revision_and_resets_once() {
    const auto germline = make_germline();
    auto phenotype_result = germline->spawn_offspring(offspring_spec(400));
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const auto binding = phenotype->runtime().parameters()[4].binding;
    auto window = LearningWindow::open(
        phenotype->organism_id(),
        phenotype->runtime(),
        {binding});
    assert(window.validate(phenotype->runtime()).ok());

    GraphEditor editor(phenotype->runtime());
    const auto edit = editor.apply(
        phenotype->runtime(),
        std::vector<GraphEditEvent>{
            GraphEditEvent{
                1,
                RemoveEdgeAction{EdgeId{10}}},
            GraphEditEvent{
                2,
                AddEdgeAction{EdgeBirth{
                    EdgeId{12}, CellId{1}, OutputPort{0}, CellId{2},
                    InputPort{0}, EdgeDelay::Immediate, 1.0}}}});
    assert(edit.ok());
    assert(edit.graph->cells().size() == germline->graph()->cells().size());
    assert(edit.graph->revision() != germline->graph()->revision());
    assert(!window.validate(phenotype->runtime()).ok());
    const auto reset = window.consume_after_graph_edit(phenotype->runtime());
    assert(reset.ok());
    assert(reset.report.reset_count == 1);
    assert(!window.validate(phenotype->runtime()).ok());
}

void test_learning_window_applies_only_declared_continuous_updates() {
    const auto germline = make_germline();
    auto phenotype_result = germline->spawn_offspring(offspring_spec(450));
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const auto binding = phenotype->runtime().parameters()[4].binding;
    auto window = LearningWindow::open(
        phenotype->organism_id(), phenotype->runtime(), {binding});

    const auto update = window.apply_sgd(
        phenotype->runtime(),
        std::array<LearningGradient, 1>{
            LearningGradient{binding, 2.0}},
        0.25);
    assert(update.ok());
    assert(update.report.updated_values == 1);
    assert(std::get<ContinuousValue>(
               *phenotype->runtime().parameter_at(binding.index)).value == 0.5);

    const auto forbidden = phenotype->runtime().parameters()[0].binding;
    const auto rejected = window.apply_sgd(
        phenotype->runtime(),
        std::array<LearningGradient, 1>{
            LearningGradient{forbidden, 1.0}},
        0.25);
    assert(!rejected.ok());
    assert(rejected.error->code == LearningWindowErrorCode::ParameterNotAllowed);
}

void test_checkpoint_exact_resume_and_atomic_rejection() {
    const auto germline = make_germline();
    auto original_result = germline->spawn_offspring(offspring_spec(500));
    assert(original_result.ok());
    auto original = std::move(original_result.phenotype);
    const double first = 1.0;
    assert(original->step({&first, 1}, {}).ok());
    const auto checkpoint = LifecyclePersistence::save(*original);
    assert(checkpoint.ok());

    const double second = 2.0;
    assert(original->step({&second, 1}, {}).ok());
    const auto expected = original->runtime().snapshot();

    auto restored_result = germline->spawn_offspring(offspring_spec(999));
    assert(restored_result.ok());
    auto restored = std::move(restored_result.phenotype);
    const auto restore = LifecyclePersistence::restore_into(
        *restored, checkpoint.bytes);
    assert(restore.ok());
    assert(restored->step({&second, 1}, {}).ok());
    assert(restored->runtime().snapshot().tick() == expected.tick());
    assert(restored->runtime().cell_states().size() == expected.cells().size());
    for (std::size_t i = 0; i < expected.cells().size(); ++i) {
        assert(restored->runtime().cell_states()[i].state_val ==
               expected.cells()[i].state_val);
        assert(restored->runtime().cell_states()[i].delay_buffer ==
               expected.cells()[i].delay_buffer);
    }

    const auto before = restored->runtime().snapshot();
    auto corrupt = checkpoint.bytes;
    corrupt.resize(corrupt.size() / 2);
    const auto rejected = LifecyclePersistence::restore_into(
        *restored, corrupt);
    assert(!rejected.ok());
    assert(
        rejected.error->code == PersistenceErrorCode::Truncated ||
        rejected.error->code == PersistenceErrorCode::Corrupt);
    assert(restored->runtime().snapshot().tick() == before.tick());

    auto external_window = LearningWindow::open(
        restored->organism_id(),
        restored->runtime(),
        {restored->runtime().parameters()[0].binding});
    const auto rejected_save = LifecyclePersistence::save(
        *restored,
        CheckpointMode::NoOptimizer,
        &external_window);
    assert(!rejected_save.ok());
    assert(rejected_save.error->code == PersistenceErrorCode::UnsupportedMode);

    const auto valid_restore = LifecyclePersistence::restore_into(
        *restored, checkpoint.bytes, external_window);
    assert(valid_restore.ok());
    assert(!external_window.validate(restored->runtime()).ok());
}

void test_optimizer_mode_is_explicitly_unsupported() {
    const auto germline = make_germline();
    auto phenotype_result = germline->spawn_offspring(offspring_spec(600));
    assert(phenotype_result.ok());
    auto phenotype = std::move(phenotype_result.phenotype);
    const auto rejected = LifecyclePersistence::save(
        *phenotype, CheckpointMode::WithOptimizerState);
    assert(!rejected.ok());
    assert(rejected.error->code == PersistenceErrorCode::UnsupportedMode);
}

}  // namespace

int main() {
    test_checkpoint_accepts_graph_over_sixty_four_cells();
    test_default_offspring_is_fresh_and_template_only();
    test_assimilation_is_explicit_and_declared();
    test_learning_window_rejects_structural_revision_and_resets_once();
    test_learning_window_applies_only_declared_continuous_updates();
    test_checkpoint_exact_resume_and_atomic_rejection();
    test_optimizer_mode_is_explicitly_unsupported();
    return 0;
}
