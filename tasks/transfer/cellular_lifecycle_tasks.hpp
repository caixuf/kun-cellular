#pragma once

#include "kun/cellular/core/growth_controller.hpp"
#include "kun/cellular/core/heredity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace kun::tasks::transfer {

using namespace kun::core;

enum class LifecycleCapabilityArm : uint8_t {
    FixedGraph,
    DevelopmentOnly,
    HeredityOnly,
    DualLayer,
    RandomStructuralProposal,
};

struct LifecycleTaskManifest {
    uint64_t master_seed{0xC011AUL};
    uint64_t train_seed{0};
    uint64_t ood_seed{0};
    uint64_t resource_budget{100};
    std::size_t train_ticks{8};
    std::size_t ood_ticks{8};

    void derive_seed_split() {
        train_seed = master_seed ^ UINT64_C(0x9E3779B97F4A7C15);
        ood_seed = master_seed ^ UINT64_C(0xD1B54A32D192ED03);
    }
};

struct LifecycleMechanismEvidence {
    uint64_t organism_id{0};
    uint64_t graph_revision_before{0};
    uint64_t graph_revision_after{0};
    std::size_t cell_count_before{0};
    std::size_t cell_count_after{0};
    std::size_t edge_count_before{0};
    std::size_t edge_count_after{0};
    bool identity_changed{false};
    bool structural_changed{false};
    bool functional_changed{false};
    std::size_t forward_steps{0};
    std::size_t cold_boundary_transactions{0};
    double forward_resource_paid{0.0};
    double lifecycle_resource_paid{0.0};
    double growth_resource_paid{0.0};
    bool extinct{false};
    bool independent_replicas{false};
};

struct LifecycleCapabilityEvidence {
    double baseline_score{0.0};
    double damaged_score{0.0};
    double recovery_score{0.0};
    double ood_recovery_score{0.0};
    std::size_t recovery_ticks{0};
    double resource_use{0.0};
    bool negative_result{false};
    std::string conclusion;
};

struct LifecycleCapabilityReport {
    LifecycleCapabilityArm arm{LifecycleCapabilityArm::FixedGraph};
    uint64_t train_seed{0};
    uint64_t ood_seed{0};
    LifecycleMechanismEvidence mechanism;
    LifecycleCapabilityEvidence research;
};

class CellularLifecycleCapabilityExperiment final {
public:
    static LifecycleCapabilityReport run(
        LifecycleCapabilityArm arm,
        LifecycleTaskManifest manifest) {
        manifest.derive_seed_split();
        auto germline_result = make_germline();
        if (!germline_result.ok()) {
            return failure_report(
                arm, manifest, "germline construction failed");
        }
        const auto germline = std::move(germline_result.germline);
        auto baseline_result = germline->spawn_offspring(
            offspring_spec(manifest, 1));
        auto damaged_result = germline->spawn_offspring(
            offspring_spec(manifest, 2));
        auto recovery_result = germline->spawn_offspring(
            offspring_spec(manifest, 3));
        auto ood_result = germline->spawn_offspring(
            offspring_spec(manifest, 4));
        if (!baseline_result.ok() || !damaged_result.ok() ||
            !recovery_result.ok() || !ood_result.ok()) {
            return failure_report(
                arm, manifest, "independent replica construction failed");
        }
        auto baseline_phenotype = std::move(baseline_result.phenotype);
        auto damaged_phenotype = std::move(damaged_result.phenotype);
        auto recovery_phenotype = std::move(recovery_result.phenotype);
        auto ood_phenotype = std::move(ood_result.phenotype);

        LifecycleCapabilityReport report;
        report.arm = arm;
        report.train_seed = manifest.train_seed;
        report.ood_seed = manifest.ood_seed;
        report.mechanism.organism_id = recovery_phenotype->organism_id();
        report.mechanism.graph_revision_before =
            recovery_phenotype->runtime().revision().value;
        report.mechanism.cell_count_before =
            recovery_phenotype->runtime().plan()->cells().size();
        report.mechanism.edge_count_before =
            recovery_phenotype->runtime().plan()->edges().size();
        report.mechanism.independent_replicas = true;

        const auto train = sequence(manifest.train_seed, manifest.train_ticks);
        const auto ood = sequence(manifest.ood_seed, manifest.ood_ticks);
        const auto baseline = run_sequence(*baseline_phenotype, train);
        record_sequence(report, baseline);
        report.research.baseline_score = baseline.score;
        if (baseline.failed) {
            report.mechanism.extinct = true;
            report.research.negative_result = true;
            report.research.conclusion =
                "baseline execution failed or extinct under the declared resource budget";
            return report;
        }
        if (!damage(*damaged_phenotype) ||
            !damage(*recovery_phenotype) ||
            !damage(*ood_phenotype)) {
            return failure_report(
                arm, manifest, "functional damage could not be injected");
        }
        const auto damaged = run_sequence(*damaged_phenotype, train);
        record_sequence(report, damaged);
        report.research.damaged_score = damaged.score;
        report.mechanism.functional_changed =
            report.research.baseline_score != report.research.damaged_score;

        switch (arm) {
            case LifecycleCapabilityArm::FixedGraph: {
                const auto recovery = run_sequence(*recovery_phenotype, train);
                const auto ood_recovery = run_sequence(*ood_phenotype, ood);
                record_sequence(report, recovery);
                record_sequence(report, ood_recovery);
                report.research.recovery_score = recovery.score;
                report.research.ood_recovery_score = ood_recovery.score;
                break;
            }
            case LifecycleCapabilityArm::HeredityOnly: {
                auto child = germline->spawn_offspring(
                    offspring_spec(manifest, 2));
                auto ood_child = germline->spawn_offspring(
                    offspring_spec(manifest, 4));
                if (!child.ok() || !ood_child.ok()) {
                    return failure_report(
                        arm, manifest, "heredity-only offspring failed");
                }
                report.mechanism.identity_changed = true;
                const auto recovery = run_sequence(*child.phenotype, train);
                const auto ood_recovery =
                    run_sequence(*ood_child.phenotype, ood);
                record_sequence(report, recovery);
                record_sequence(report, ood_recovery);
                report.research.recovery_score = recovery.score;
                report.research.ood_recovery_score = ood_recovery.score;
                break;
            }
            case LifecycleCapabilityArm::DevelopmentOnly:
            case LifecycleCapabilityArm::DualLayer:
            case LifecycleCapabilityArm::RandomStructuralProposal: {
                auto growth = CellularGrowthController::create(
                    recovery_phenotype->lifecycle(), growth_config());
                auto ood_growth = CellularGrowthController::create(
                    ood_phenotype->lifecycle(), growth_config());
                if (!growth.ok() || !ood_growth.ok()) {
                    return failure_report(
                        arm, manifest, "growth controller construction failed");
                }
                const EdgeId edge_id{arm == LifecycleCapabilityArm::RandomStructuralProposal
                        ? 90 + (manifest.train_seed % 7)
                        : 80};
                const double weight =
                    arm == LifecycleCapabilityArm::RandomStructuralProposal &&
                            (manifest.train_seed & 1U)
                        ? -1.0
                        : 1.0;
                const GrowthProposal proposal{
                    GrowthSynapseProposal{
                        1,
                        EdgeBirth{
                            edge_id, CellId{1}, OutputPort{0}, CellId{2},
                            InputPort{0}, EdgeDelay::Immediate, weight},
                        GrowthFunding{
                            CellId{1}, ResourceCompartmentId{1}},
                        0.1}};
                if (!growth.controller->submit(proposal).ok() ||
                    !ood_growth.controller->submit(proposal).ok()) {
                    return failure_report(
                        arm, manifest, "growth proposal submission failed");
                }
                const auto repaired = growth.controller->step(
                    std::span<const double>{},
                    std::span<const ResourceSupply>{});
                const auto ood_repaired = ood_growth.controller->step(
                    std::span<const double>{},
                    std::span<const ResourceSupply>{});
                report.mechanism.cold_boundary_transactions = 2;
                if (!repaired.ok() || !ood_repaired.ok()) {
                    report.research.negative_result = true;
                    report.research.conclusion =
                        "growth proposal rejected; no capability conclusion";
                    break;
                }
                report.mechanism.growth_resource_paid =
                    (repaired.growth_report
                         ? repaired.growth_report->paid_cost
                         : 0.0) +
                    (ood_repaired.growth_report
                         ? ood_repaired.growth_report->paid_cost
                         : 0.0);
                report.research.recovery_ticks = 1;
                const auto recovery = run_sequence(*recovery_phenotype, train);
                const auto ood_recovery = run_sequence(*ood_phenotype, ood);
                record_sequence(report, recovery);
                record_sequence(report, ood_recovery);
                report.research.recovery_score = recovery.score;
                report.research.ood_recovery_score = ood_recovery.score;
                break;
            }
        }

        report.mechanism.graph_revision_after =
            recovery_phenotype->runtime().revision().value;
        report.mechanism.cell_count_after =
            recovery_phenotype->runtime().plan()->cells().size();
        report.mechanism.edge_count_after =
            recovery_phenotype->runtime().plan()->edges().size();
        report.mechanism.structural_changed =
            report.mechanism.graph_revision_after !=
                report.mechanism.graph_revision_before ||
            report.mechanism.cell_count_after !=
                report.mechanism.cell_count_before ||
            report.mechanism.edge_count_after !=
                report.mechanism.edge_count_before;
        report.mechanism.extinct =
            recovery_phenotype->lifecycle().overall_state() ==
            LifecycleState::Extinct;
        report.research.resource_use =
            report.mechanism.growth_resource_paid +
            report.mechanism.forward_resource_paid +
            report.mechanism.lifecycle_resource_paid;
        if (report.research.conclusion.empty()) {
            const bool recovered =
                report.research.recovery_score >=
                report.research.baseline_score * 0.8;
            report.research.negative_result = !recovered;
            report.research.conclusion = recovered
                ? "capability recovery observed; compare controls and OOD separately"
                : "no capability recovery observed";
        }
        return report;
    }

private:
    struct SequenceResult {
        double score{0.0};
        double resource_paid{0.0};
        std::size_t steps{0};
        bool failed{false};
    };

    static void record_sequence(
        LifecycleCapabilityReport& report,
        const SequenceResult& sequence) {
        report.mechanism.forward_steps += sequence.steps;
        report.mechanism.forward_resource_paid += sequence.resource_paid;
        if (sequence.failed) {
            report.research.negative_result = true;
            if (report.research.conclusion.empty()) {
                report.research.conclusion =
                    "replica execution failed or became extinct; score is not a fallback";
            }
        }
    }

    static GermlineResult make_germline() {
        GraphDefinition graph{
            GraphIdentity{9400},
            GraphRevision{1},
            SemanticProfile::StrictCore,
            1,
            {{CellId{1}, CellType::SENSE_RAW_INPUT_0},
             {CellId{2}, CellType::OP_ABS}},
            {{EdgeId{10}, CellId{1}, OutputPort{0}, CellId{2}, InputPort{0},
              EdgeDelay::Immediate}}};
        InitialParameterSeeds seeds;
        for (const auto& cell : graph.cells) {
            const auto contract = contract_for(cell.type)->get();
            for (std::size_t slot = 0; slot < 2; ++slot) {
                ParameterValue value = UnusedParameter{};
                if (contract.parameters[slot].value_type ==
                    ParameterValueType::Continuous) {
                    value = ContinuousValue{1.0};
                }
                seeds.cell_parameters.push_back(
                    CellParameterSeed{
                        cell.id, static_cast<ParameterSlot>(slot), value});
            }
        }
        seeds.edge_weights.push_back(EdgeParameterSeed{EdgeId{10}, 1.0});
        return Germline::create(
            std::move(graph), std::move(seeds), "r6-causal-transfer-v1");
    }

    static OffspringSpec offspring_spec(
        const LifecycleTaskManifest& manifest,
        uint64_t id) {
        OffspringSpec spec;
        spec.organism_id = id;
        spec.rng_seed = manifest.master_seed + id;
        spec.lifecycle_config.apoptotic_resource = 0.0;
        spec.lifecycle_config.dormant_enter_resource = 0.0;
        spec.lifecycle_config.dormant_exit_resource = 1.0;
        spec.resource_config.transmission_scale = 1.0;
        spec.resource_config.activity_scale = 1.0;
        spec.resource_config.execution_costs.push_back(
            ResourceExecutionCost{CellType::OP_ABS, 0.01});
        spec.resource_compartments.push_back(
            ResourceCompartmentInitial{ResourceCompartmentId{1}, 0.0});
        spec.resource_cells = {
            {CellId{1}, ResourceCompartmentId{1},
             static_cast<double>(manifest.resource_budget),
             static_cast<double>(manifest.resource_budget), 0.0},
            {CellId{2}, ResourceCompartmentId{1},
             static_cast<double>(manifest.resource_budget),
             static_cast<double>(manifest.resource_budget), 0.0}};
        return spec;
    }

    static GrowthConfig growth_config() {
        GrowthConfig config;
        config.synapse_birth_cost = 0.1;
        config.initial_capacity = 0.0;
        config.initial_energy = 0.0;
        return config;
    }

    static std::vector<double> sequence(uint64_t seed, std::size_t count) {
        std::mt19937_64 rng(seed);
        std::vector<double> values;
        values.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            values.push_back(
                static_cast<double>(
                    static_cast<int>(rng() % 11) - 5) / 5.0);
        }
        return values;
    }

    static SequenceResult run_sequence(
        Phenotype& phenotype,
        std::span<const double> sequence) {
        double score = 0.0;
        double resource_paid = 0.0;
        std::size_t steps = 0;
        for (const double input : sequence) {
            const auto step = phenotype.step(
                std::span<const double>(&input, 1),
                std::span<const ResourceSupply>{});
            if (step.settlement.has_value()) {
                resource_paid += step.settlement->paid_cost;
            }
            ++steps;
            if (!step.ok() || !step.measurement.has_value()) {
                return SequenceResult{score, resource_paid, steps, true};
            }
            const auto found = std::find_if(
                step.measurement->cells.begin(),
                step.measurement->cells.end(),
                [](const auto& cell) { return cell.cell == CellId{2}; });
            if (found == step.measurement->cells.end()) {
                return SequenceResult{score, resource_paid, steps, true};
            }
            score += 1.0 - std::min(1.0, std::abs(found->output - input));
        }
        return SequenceResult{
            sequence.empty() ? 0.0 : score / sequence.size(),
            resource_paid,
            steps,
            false};
    }

    static bool damage(Phenotype& phenotype) {
        if (phenotype.runtime().parameters().size() <= 2) return false;
        const auto binding = phenotype.runtime().parameters()[2].binding;
        return phenotype.runtime().set_parameter(
            binding, ParameterValue{ContinuousValue{0.0}}).ok();
    }

    static LifecycleCapabilityReport failure_report(
        LifecycleCapabilityArm arm,
        const LifecycleTaskManifest& manifest,
        std::string reason) {
        LifecycleCapabilityReport report;
        report.arm = arm;
        report.train_seed = manifest.train_seed;
        report.ood_seed = manifest.ood_seed;
        report.research.negative_result = true;
        report.research.conclusion = std::move(reason);
        return report;
    }
};

}  // namespace kun::tasks::transfer
