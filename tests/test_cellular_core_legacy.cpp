#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/core/semantic_profile.hpp"
#include "kun/cellular/legacy/organism_adapter.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::Synapse;
using kun::migration::MigrationArtifactKind;
using kun::migration::MigrationOperation;
using kun::migration::MigrationPreflightCode;
using kun::migration::MigrationPreflightRequest;
using kun::migration::validate_migration_preflight;

constexpr double kTolerance = 1e-12;

void expect_near(double actual, double expected) {
    assert(std::abs(actual - expected) < kTolerance);
}

Cell make_cell(
    uint32_t id,
    CellType type,
    double param1 = 1.0,
    double param2 = 0.0) {
    Cell cell;
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

Synapse make_synapse(
    uint32_t from,
    uint32_t to,
    uint8_t port = 0,
    double weight = 1.0) {
    Synapse synapse;
    synapse.from_cell_id = from;
    synapse.to_cell_id = to;
    synapse.to_port = port;
    synapse.weight = weight;
    synapse.initial_weight = weight;
    synapse.hebbian_rate = 0.0;
    return synapse;
}

CellularOrganism make_two_port_arithmetic(CellType operation) {
    CellularOrganism organism;
    organism.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::SENSE_RAW_INPUT_1),
        make_cell(2, operation),
        make_cell(3, CellType::ACT_CHANNEL, 1.0, 0.0),
    };
    organism.synapses = {
        make_synapse(0, 2, 0),
        make_synapse(1, 2, 1),
        make_synapse(2, 3),
    };
    return organism;
}

CellularOrganism make_single_input_chain(
    CellType operation,
    double param1 = 1.0,
    double param2 = 0.0) {
    CellularOrganism organism;
    organism.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, operation, param1, param2),
        make_cell(2, CellType::ACT_CHANNEL, 1.0, 0.0),
    };
    organism.synapses = {
        make_synapse(0, 1),
        make_synapse(1, 2),
    };
    return organism;
}

MigrationPreflightRequest make_birth_request(
    kun::SemanticProfile profile = kun::SemanticProfile::LegacyCompatible) {
    return MigrationPreflightRequest{
        MigrationOperation::ImportBirthTemplate,
        MigrationArtifactKind::LegacyGenome,
        profile,
    };
}

void assert_compiled_synapses_equal(
    const std::vector<CellularOrganism::CompiledSynapse>& actual,
    const std::vector<CellularOrganism::CompiledSynapse>& expected) {
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(actual[i].from_idx == expected[i].from_idx);
        assert(actual[i].to_idx == expected[i].to_idx);
        assert(actual[i].to_port == expected[i].to_port);
        expect_near(actual[i].weight, expected[i].weight);
        expect_near(actual[i].initial_weight, expected[i].initial_weight);
        expect_near(actual[i].hebbian_rate, expected[i].hebbian_rate);
        expect_near(actual[i].hebbian_decay, expected[i].hebbian_decay);
        assert(actual[i].is_recurrent == expected[i].is_recurrent);
    }
}

void assert_source_unchanged(
    const CellularOrganism& actual,
    const CellularOrganism& before) {
    assert(actual.cells.size() == before.cells.size());
    assert(actual.synapses.size() == before.synapses.size());
    assert(std::memcmp(actual.cells.data(), before.cells.data(),
                       actual.cells.size() * sizeof(Cell)) == 0);
    assert(std::memcmp(actual.synapses.data(), before.synapses.data(),
                       actual.synapses.size() * sizeof(Synapse)) == 0);
    assert_compiled_synapses_equal(actual.compiled_synapses_,
                                   before.compiled_synapses_);
    assert(actual.compiled_actions_.size() == before.compiled_actions_.size());
    for (size_t i = 0; i < actual.compiled_actions_.size(); ++i) {
        assert(actual.compiled_actions_[i].cell_idx ==
               before.compiled_actions_[i].cell_idx);
        assert(actual.compiled_actions_[i].type ==
               before.compiled_actions_[i].type);
    }
    assert(actual.execution_order_ == before.execution_order_);
    assert(actual.out_start_ == before.out_start_);
    assert(actual.out_edges_ == before.out_edges_);
    assert(actual.flat_port_inputs_ == before.flat_port_inputs_);
    assert(actual.is_compiled_ == before.is_compiled_);
}

void test_two_port_arithmetic_characterization() {
    const std::vector<std::pair<CellType, double>> cases = {
        {CellType::OP_SUM, 5.0},
        {CellType::OP_SUB, -1.0},
        {CellType::OP_MULTIPLY, 6.0},
    };
    for (const auto& [operation, expected] : cases) {
        CellularOrganism organism = make_two_port_arithmetic(operation);
        assert(organism.compile());
        const double inputs[] = {2.0, 3.0};
        organism.forward_nd(inputs, 2, false);
        expect_near(organism.cells[2].output_val, expected);
        expect_near(organism.cells[3].output_val, expected);
        expect_near(organism.cells[2].param1, 1.0);
        expect_near(organism.cells[2].param2, 0.0);
    }
}

void test_generic_channel43_characterization() {
    CellularOrganism organism =
        make_single_input_chain(CellType::SENSE_CHANNEL, 1.0, 43.0);
    organism.cells[2].type = CellType::ACT_CHANNEL;
    organism.cells[2].param2 = 43.0;
    organism.synapses[1] = make_synapse(1, 2);
    assert(organism.compile());

    std::vector<double> inputs(44, 0.0);
    inputs[43] = 7.25;
    organism.forward_nd(inputs.data(), inputs.size(), false);
    expect_near(organism.cells[1].param2, 43.0);
    expect_near(organism.cells[1].output_val, 7.25);
    expect_near(organism.cells[2].param2, 43.0);
    expect_near(organism.cells[2].output_val, 7.25);
}

void test_delay_sequence_and_state() {
    CellularOrganism organism =
        make_single_input_chain(CellType::OP_DELAY_N, 1.0 / 16.0);
    assert(organism.compile());
    organism.reset_state();

    const double inputs[] = {10.0, 20.0, 30.0};
    const double expected_outputs[] = {0.0, 10.0, 20.0};
    for (size_t step = 0; step < 3; ++step) {
        organism.forward_nd(&inputs[step], 1, false);
        expect_near(organism.cells[1].output_val, expected_outputs[step]);
    }
    assert(organism.cells[1].delay_idx == 3);
    expect_near(organism.cells[1].delay_buffer[0], 10.0);
    expect_near(organism.cells[1].delay_buffer[1], 20.0);
    expect_near(organism.cells[1].delay_buffer[2], 30.0);
    expect_near(organism.cells[1].param1, 1.0 / 16.0);
}

void test_hysteresis_sequence_and_latch() {
    CellularOrganism organism =
        make_single_input_chain(CellType::GATE_HYSTERESIS, 0.5, -0.5);
    assert(organism.compile());
    organism.reset_state();

    const double inputs[] = {0.0, 0.75, 0.0, -0.75, 0.0};
    const double expected_outputs[] = {-1.0, 1.0, 1.0, -1.0, -1.0};
    const bool expected_latches[] = {false, true, true, false, false};
    for (size_t step = 0; step < 5; ++step) {
        organism.forward_nd(&inputs[step], 1, false);
        expect_near(organism.cells[1].output_val, expected_outputs[step]);
        assert(organism.cells[1].latch_state == expected_latches[step]);
        expect_near(organism.cells[1].state_val, expected_outputs[step]);
    }
}

void test_recurrent_sequence_characterization() {
    CellularOrganism organism;
    organism.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::OP_SUM),
        make_cell(2, CellType::ACT_CHANNEL, 1.0, 0.0),
    };
    organism.synapses = {
        make_synapse(0, 1),
        make_synapse(1, 1, 0, 0.5),
        make_synapse(1, 2),
    };
    assert(organism.compile());
    organism.reset_state();

    const double inputs[] = {1.0, 0.0, 0.0};
    const double expected_outputs[] = {1.0, 0.5, 0.25};
    for (size_t step = 0; step < 3; ++step) {
        organism.forward_nd(&inputs[step], 1, false);
        expect_near(organism.cells[1].output_val, expected_outputs[step]);
        expect_near(organism.cells[1].prev_output_val, expected_outputs[step]);
    }
    assert(organism.compiled_synapses_[1].is_recurrent);
}

void test_parallel_edges_preserve_distinct_weights() {
    CellularOrganism organism;
    organism.cells = {
        make_cell(0, CellType::SENSE_RAW_INPUT_0),
        make_cell(1, CellType::ACT_CHANNEL, 1.0, 43.0),
    };
    organism.synapses = {
        make_synapse(0, 1, 0, 0.25),
        make_synapse(0, 1, 0, 0.75),
    };
    assert(organism.compile());
    const double input = 4.0;
    organism.forward_nd(&input, 1, false);
    expect_near(organism.cells[1].output_val, 4.0);
    assert(organism.compiled_synapses_.size() == 2);
    expect_near(organism.compiled_synapses_[0].weight, 0.25);
    expect_near(organism.compiled_synapses_[1].weight, 0.75);
}

void test_profile_and_supported_birth_template_preflight() {
    static_assert(static_cast<uint8_t>(kun::SemanticProfile::LegacyCompatible) == 0);
    static_assert(static_cast<uint8_t>(kun::SemanticProfile::StrictCore) == 1);
    assert(kun::semantic_profile_from_code(0) ==
           kun::SemanticProfile::LegacyCompatible);
    assert(kun::semantic_profile_from_code(1) == kun::SemanticProfile::StrictCore);
    assert(!kun::semantic_profile_from_code(255).has_value());

    CellularOrganism valid = make_single_input_chain(CellType::OP_SUM);
    assert(valid.compile());
    const double input = 2.5;
    valid.forward_nd(&input, 1, false);
    const CellularOrganism before = valid;
    const auto result =
        validate_migration_preflight(valid, make_birth_request());
    assert(result.accepted());
    assert_source_unchanged(valid, before);

    CellularOrganism sensor_only;
    sensor_only.cells = {make_cell(0, CellType::SENSE_RAW_INPUT_0)};
    assert(validate_migration_preflight(sensor_only, make_birth_request()).accepted());
}

void test_channel43_is_valid_for_sense_and_act_preflight() {
    for (const CellType channel_type :
         {CellType::SENSE_CHANNEL, CellType::ACT_CHANNEL}) {
        CellularOrganism organism =
            make_single_input_chain(channel_type, 1.0, 43.0);
        const CellularOrganism before = organism;
        assert(validate_migration_preflight(organism, make_birth_request()).accepted());
        assert_source_unchanged(organism, before);
    }
}

void test_malformed_birth_templates_are_rejected_without_mutation() {
    const auto assert_invalid = [](CellularOrganism organism) {
        const CellularOrganism before = organism;
        assert(validate_migration_preflight(organism, make_birth_request()).code ==
               MigrationPreflightCode::InvalidOrganism);
        assert_source_unchanged(organism, before);
    };

    CellularOrganism duplicate_ids = make_single_input_chain(CellType::OP_SUM);
    duplicate_ids.cells[1].id = duplicate_ids.cells[0].id;
    assert_invalid(duplicate_ids);

    CellularOrganism dangling = make_single_input_chain(CellType::OP_SUM);
    dangling.synapses[0].from_cell_id = 99;
    assert_invalid(dangling);

    CellularOrganism port2 = make_single_input_chain(CellType::OP_SUM);
    port2.synapses[0].to_port = 2;
    assert_invalid(port2);

    CellularOrganism unknown_type = make_single_input_chain(CellType::OP_SUM);
    unknown_type.cells[1].type = static_cast<CellType>(255);
    assert_invalid(unknown_type);

    CellularOrganism nan_gene = make_single_input_chain(CellType::OP_SUM);
    nan_gene.cells[1].param1 = std::numeric_limits<double>::quiet_NaN();
    assert_invalid(nan_gene);

    CellularOrganism nan_weight = make_single_input_chain(CellType::OP_SUM);
    nan_weight.synapses[0].weight = std::numeric_limits<double>::quiet_NaN();
    assert_invalid(nan_weight);

    CellularOrganism nan_geometry = make_single_input_chain(CellType::OP_SUM);
    nan_geometry.cells[1].x = std::numeric_limits<float>::quiet_NaN();
    assert_invalid(nan_geometry);

    for (const CellType channel_type :
         {CellType::SENSE_CHANNEL, CellType::ACT_CHANNEL}) {
        for (const double channel :
             {-1.0, 1.5, std::numeric_limits<double>::max()}) {
            CellularOrganism invalid_channel =
                make_single_input_chain(channel_type, 1.0, channel);
            assert_invalid(invalid_channel);
        }
    }
}

void test_unsupported_actual_features_and_inferred_cycle() {
    CellularOrganism advanced = make_single_input_chain(CellType::OP_SUM);
    advanced.enable_mla(1, 1);
    const CellularOrganism advanced_before = advanced;
    assert(validate_migration_preflight(advanced, make_birth_request()).code ==
           MigrationPreflightCode::UnsupportedFeature);
    assert_source_unchanged(advanced, advanced_before);

    CellularOrganism cycle = make_single_input_chain(CellType::OP_SUM);
    cycle.synapses.push_back(make_synapse(1, 1, 0, 0.5));
    assert(!cycle.synapses.back().is_recurrent);
    assert(validate_migration_preflight(cycle, make_birth_request()).code ==
           MigrationPreflightCode::UnsupportedFeature);
}

void test_profile_and_operation_boundaries() {
    CellularOrganism valid = make_single_input_chain(CellType::OP_SUM);

    MigrationPreflightRequest unknown_profile = make_birth_request(
        static_cast<kun::SemanticProfile>(255));
    assert(validate_migration_preflight(valid, unknown_profile).code ==
           MigrationPreflightCode::UnknownProfile);

    MigrationPreflightRequest strict_profile = make_birth_request(
        kun::SemanticProfile::StrictCore);
    assert(validate_migration_preflight(valid, strict_profile).code ==
           MigrationPreflightCode::UnsupportedProfile);

    MigrationPreflightRequest strict_graph{
        MigrationOperation::ImportBirthTemplate,
        MigrationArtifactKind::StrictCoreGraph,
        kun::SemanticProfile::LegacyCompatible,
    };
    assert(validate_migration_preflight(valid, strict_graph).code ==
           MigrationPreflightCode::UnsupportedArtifact);

    MigrationPreflightRequest restore{
        MigrationOperation::RestoreCheckpoint,
        MigrationArtifactKind::LegacyCheckpoint,
        kun::SemanticProfile::LegacyCompatible,
        3,
    };
    assert(validate_migration_preflight(valid, restore).code ==
           MigrationPreflightCode::RuntimeStateUnavailable);

    MigrationPreflightRequest export_request{
        MigrationOperation::ExportCheckpoint,
        MigrationArtifactKind::LegacyCheckpoint,
        kun::SemanticProfile::LegacyCompatible,
        3,
    };
    assert(validate_migration_preflight(valid, export_request).code ==
           MigrationPreflightCode::UnsupportedFeature);
}

}  // namespace

int main() {
    test_two_port_arithmetic_characterization();
    test_generic_channel43_characterization();
    test_delay_sequence_and_state();
    test_hysteresis_sequence_and_latch();
    test_recurrent_sequence_characterization();
    test_parallel_edges_preserve_distinct_weights();
    test_profile_and_supported_birth_template_preflight();
    test_channel43_is_valid_for_sense_and_act_preflight();
    test_malformed_birth_templates_are_rejected_without_mutation();
    test_unsupported_actual_features_and_inferred_cycle();
    test_profile_and_operation_boundaries();
    std::cout << "[PASS] legacy substrate characterization and migration preflight\n";
    return 0;
}
