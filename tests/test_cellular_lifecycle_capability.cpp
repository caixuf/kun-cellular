#include "tasks/transfer/cellular_lifecycle_tasks.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace kun::tasks::transfer;

namespace {

void test_seed_split_and_deterministic_manifest() {
    LifecycleTaskManifest first;
    first.master_seed = 1234;
    first.derive_seed_split();
    LifecycleTaskManifest second;
    second.master_seed = 1234;
    second.derive_seed_split();
    assert(first.train_seed == second.train_seed);
    assert(first.ood_seed == second.ood_seed);
    assert(first.train_seed != first.ood_seed);
}

void test_all_controls_report_separate_mechanism_and_research_evidence() {
    LifecycleTaskManifest manifest;
    manifest.master_seed = 5678;
    manifest.resource_budget = 100;
    manifest.train_ticks = 6;
    manifest.ood_ticks = 6;
    for (const auto arm : {
             LifecycleCapabilityArm::FixedGraph,
             LifecycleCapabilityArm::DevelopmentOnly,
             LifecycleCapabilityArm::HeredityOnly,
             LifecycleCapabilityArm::DualLayer,
             LifecycleCapabilityArm::RandomStructuralProposal}) {
        const auto report =
            CellularLifecycleCapabilityExperiment::run(arm, manifest);
        const auto repeated =
            CellularLifecycleCapabilityExperiment::run(arm, manifest);
        assert(report.train_seed != report.ood_seed);
        assert(report.mechanism.independent_replicas);
        assert(report.research.baseline_score == repeated.research.baseline_score);
        assert(report.research.damaged_score == repeated.research.damaged_score);
        assert(report.research.recovery_score == repeated.research.recovery_score);
        assert(report.research.ood_recovery_score ==
               repeated.research.ood_recovery_score);
        assert(report.research.conclusion.size() > 0);
        assert(std::isfinite(report.research.baseline_score));
        assert(std::isfinite(report.research.damaged_score));
        assert(std::isfinite(report.research.recovery_score));
        assert(std::isfinite(report.research.ood_recovery_score));
        assert(report.mechanism.forward_steps <=
               4 * (manifest.train_ticks + manifest.ood_ticks) + 2);
        assert(report.mechanism.structural_changed ||
               report.mechanism.identity_changed ||
               arm == LifecycleCapabilityArm::FixedGraph ||
               arm == LifecycleCapabilityArm::HeredityOnly);
    }
}

void test_structural_and_functional_change_are_distinct() {
    LifecycleTaskManifest manifest;
    manifest.master_seed = 9876;
    const auto fixed = CellularLifecycleCapabilityExperiment::run(
        LifecycleCapabilityArm::FixedGraph, manifest);
    const auto developed = CellularLifecycleCapabilityExperiment::run(
        LifecycleCapabilityArm::DevelopmentOnly, manifest);
    assert(fixed.mechanism.functional_changed);
    assert(!fixed.mechanism.structural_changed);
    assert(developed.mechanism.structural_changed ||
           developed.research.negative_result);
}

void test_negative_findings_are_preserved_not_fallback_success() {
    LifecycleTaskManifest manifest;
    manifest.master_seed = 42;
    manifest.resource_budget = 0;
    const auto report = CellularLifecycleCapabilityExperiment::run(
        LifecycleCapabilityArm::DevelopmentOnly, manifest);
    assert(report.research.negative_result);
    assert(report.research.conclusion.find("no capability") !=
           std::string::npos ||
           report.research.conclusion.find("rejected") != std::string::npos ||
           report.research.conclusion.find("extinct") != std::string::npos);
}

}  // namespace

int main() {
    test_seed_split_and_deterministic_manifest();
    test_all_controls_report_separate_mechanism_and_research_evidence();
    test_structural_and_functional_change_are_distinct();
    test_negative_findings_are_preserved_not_fallback_success();
    return 0;
}
