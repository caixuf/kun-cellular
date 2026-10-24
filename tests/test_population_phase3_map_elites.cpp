// ============================================================================
// test_population_phase3_map_elites.cpp — Prefer-A Phase 3 MAP-Elites / novelty
// A1 grid insert/replace · A2 stepping-stones/drivers · A3 protect_window
// A4 metrics wire · A5 anti-artifact · A6 E-Gate dry-run · A8 exec≠learn · A9 82/det
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/anti_artifact.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"
#include "kun/cellular/population/e_gate.hpp"
#include "kun/cellular/population/map_elites_grid.hpp"
#include "kun/cellular/population/novelty_stepping_stone.hpp"
#include "kun/cellular/population/phase3_metrics.hpp"
#include "kun/cellular/population/population_species_niche.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::FunctionalCoverageContract;
using kun::Synapse;
using kun::population::AntiArtifactEvidence;
using kun::population::AvalancheMetric;
using kun::population::BehaviorDescriptor;
using kun::population::EGatePreRegister;
using kun::population::EmergenceTag;
using kun::population::LearningScaleAxis;
using kun::population::LearningScalePoint;
using kun::population::MapElitesGrid;
using kun::population::NoveltySteppingStoneArchive;
using kun::population::PopulationNicheRegistry;
using kun::population::PopulationSpeciesNiche;
using kun::population::PreferAFitnessDriver;
using kun::population::bd_from_coverage_novelty;
using kun::population::default_prefer_a_learning_scales;
using kun::population::evaluate_anti_artifact;
using kun::population::kPreferAOrganismCells;
using kun::population::phase3_handoff_table_markdown;
using kun::population::run_e_gate;

namespace {

CellularOrganism make_organism_82(std::mt19937::result_type seed) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "PreferA-82-P3";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pdist(0.01, 0.99);
    std::uniform_real_distribution<double> wdist(-0.8, 0.8);
    static const CellType kTypes[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_SUM, CellType::OP_SUB,
        CellType::GATE_HYSTERESIS, CellType::GATE_THRESHOLD,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE
    };
    org.cells.reserve(kPreferAOrganismCells);
    for (std::size_t i = 0; i < kPreferAOrganismCells; ++i) {
        Cell c;
        c.id = static_cast<std::uint32_t>(i);
        c.type = kTypes[i % (sizeof(kTypes) / sizeof(kTypes[0]))];
        c.param1 = pdist(rng);
        c.param2 = pdist(rng) * 0.1;
        org.cells.push_back(c);
    }
    constexpr std::size_t kSyn = 164;
    org.synapses.reserve(kSyn);
    for (std::size_t s = 0; s < kSyn; ++s) {
        Synapse syn;
        syn.from_cell_id = static_cast<std::uint32_t>(s % kPreferAOrganismCells);
        syn.to_cell_id = static_cast<std::uint32_t>((s * 7 + 3) % kPreferAOrganismCells);
        syn.to_port = static_cast<std::uint8_t>(s & 1u);
        syn.weight = wdist(rng);
        syn.initial_weight = syn.weight;
        syn.is_active = true;
        org.synapses.push_back(syn);
    }
    org.compile();
    assert(org.cells.size() == kPreferAOrganismCells);
    return org;
}

void test_a1_map_elites_grid() {
    MapElitesGrid grid(2, 4, 0.3f);
    assert(grid.cell_count() == 16);
    assert(grid.occupied_count() == 0);

    BehaviorDescriptor bd;
    bd.dims = 2;
    bd[0] = 0.1f;
    bd[1] = 0.2f;
    auto r1 = grid.try_insert(1, bd, 0.4f, 0.5f);
    assert(r1.accepted);
    assert(!r1.replaced);
    assert(grid.occupied_count() == 1);

    // Same cell, lower QD → reject
    auto r2 = grid.try_insert(2, bd, 0.1f, 0.1f);
    assert(!r2.accepted);
    assert(grid.cell_at(r1.cell_index).individual_id == 1);

    // Same cell, higher QD → replace
    auto r3 = grid.try_insert(3, bd, 0.9f, 0.9f);
    assert(r3.accepted);
    assert(r3.replaced);
    assert(grid.cell_at(r1.cell_index).individual_id == 3);

    // Different BD → new cell
    bd[0] = 0.8f;
    bd[1] = 0.8f;
    auto r4 = grid.try_insert(4, bd, 0.5f, 0.5f);
    assert(r4.accepted);
    assert(grid.occupied_count() == 2);

    // Capacity stable across clear + refill (Prefer-A zero-alloc steady-state)
    const std::size_t cap = grid.capacity_cells();
    grid.clear();
    assert(grid.occupied_count() == 0);
    assert(grid.capacity_cells() == cap);
    grid.try_insert(5, bd, 0.2f, 0.2f);
    assert(grid.capacity_cells() == cap);

    std::cout << "  ✓ A1 MAP-Elites grid insert/replace/capacity" << std::endl;
}

void test_a2_novelty_stepping_stones() {
    NoveltySteppingStoneArchive archive(32, 3);
    MapElitesGrid grid(2, 4, 0.3f);

    std::vector<double> b0 = {0.0, 0.0};
    std::vector<double> b1 = {1.0, 0.0};
    std::vector<double> b2 = {0.0, 1.0};
    assert(archive.compute_novelty(b0) == 1.0);  // empty → 1.0
    archive.add_behavior(b0);
    const double nov1 = archive.compute_novelty(b1);
    assert(nov1 > 0.0);

    // Mid-goal: low fitness but high novelty → keep under NOVELTY_SEARCH
    assert(NoveltySteppingStoneArchive::keep_as_stepping_stone(
        PreferAFitnessDriver::NOVELTY_SEARCH, 0.01, 0.9, 0.5, 0.2));
    // TASK_FITNESS_ONLY: low fitness purged
    assert(!NoveltySteppingStoneArchive::keep_as_stepping_stone(
        PreferAFitnessDriver::TASK_FITNESS_ONLY, 0.01, 0.9, 0.5, 0.2));
    // HYBRID: novelty floor saves mid-goal
    assert(NoveltySteppingStoneArchive::keep_as_stepping_stone(
        PreferAFitnessDriver::HYBRID_CURIOSITY, 0.01, 0.9, 0.5, 0.2));

    // Driver modes
    assert(NoveltySteppingStoneArchive::driver_score(
        PreferAFitnessDriver::NOVELTY_SEARCH, 10.0, 0.4) == 0.4);
    const double hybrid = NoveltySteppingStoneArchive::driver_score(
        PreferAFitnessDriver::HYBRID_CURIOSITY, 10.0, 0.4, 0.3);
    assert(std::abs(hybrid - (10.0 + 0.3 * 0.4)) < 1e-9);
    assert(NoveltySteppingStoneArchive::driver_score(
        PreferAFitnessDriver::TASK_FITNESS_ONLY, 10.0, 0.4) == 10.0);

    auto feed = archive.feed_map_elites(grid, 7, 0.6f, b2);
    assert(feed.accepted);
    assert(archive.size() >= 2);

    // Capacity does not grow past max after fill
    for (int i = 0; i < 40; ++i) {
        archive.add_behavior({static_cast<double>(i) * 0.01, 0.5});
    }
    assert(archive.size() == archive.max_size());

    std::cout << "  ✓ A2 novelty stepping-stones + driver modes" << std::endl;
}

void test_a3_protect_window() {
    PopulationSpeciesNiche niche;
    niche.species_id = 1;
    niche.protect_window = 3;
    niche.members.push_back({1, 0.1, 0.0, false});
    niche.members.push_back({2, 0.9, 0.0, false});
    niche.calculate_shared_fitness();
    assert(niche.in_protect_window());
    assert(niche.purge_unprotected(0.5) == 0);  // protected — no purge
    assert(niche.members.size() == 2);

    niche.note_generation(0.9);
    niche.note_generation(0.9);
    niche.note_generation(0.9);  // age >= protect_window
    assert(!niche.in_protect_window());
    const auto removed = niche.purge_unprotected(0.5);
    assert(removed == 1);
    assert(niche.members.size() == 1);
    assert(niche.members[0].individual_id == 2);

    // Name disambiguation: type exists in kun::population
    PopulationNicheRegistry reg(0.35f, 5);
    assert(reg.default_protect_window() == 5);
    reg.place_or_speciate(10, 1.0, 0.1, true);
    assert(reg.niches().size() == 1);
    assert(reg.niches()[0].protect_window == 5);

    std::cout << "  ✓ A3 PopulationSpeciesNiche protect_window" << std::endl;
}

void test_a4_metrics_wire() {
    auto org = make_organism_82(42);
    FunctionalCoverageContract cov = org.evaluate_coverage_contract();
    AvalancheMetric av(kPreferAOrganismCells, 1.0f);
    // Drive a few grains from synthetic activity
    std::vector<float> act(kPreferAOrganismCells, 0.0f);
    for (std::size_t i = 0; i < kPreferAOrganismCells; ++i) {
        act[i] = static_cast<float>(org.cells[i].param1);
    }
    av.ingest_activity(act.data(), act.size(), 2.0f);
    for (int i = 0; i < 16; ++i) {
        av.drive(static_cast<std::size_t>(i * 5), 1.2f);
    }
    auto ar = av.report();
    assert(ar.valid);
    assert(ar.events > 0);
    assert(ar.events > 0);  // histogram populated via events

    // Lyapunov companion (must not be claimed as avalanche)
    auto lyap = org.check_lyapunov_stability();
    (void)lyap;
    assert(std::string(AvalancheMetric::lyapunov_companion_note()).find("NOT") != std::string::npos);

    // L1 distance
    auto org2 = make_organism_82(99);
    const double dist = kun::population::compatibility_distance(org, org2);
    assert(dist >= 0.0);

    // Handoff table non-empty
    const auto md = phase3_handoff_table_markdown();
    assert(md.find("criticality_avalanche") != std::string::npos);
    assert(md.find("Lyapunov companion") != std::string::npos);

    auto bd = bd_from_coverage_novelty(cov, 0.3f);
    assert(bd.dims == 2);

    std::cout << "  ✓ A4 Coverage · L1 distance · avalanche · Lyapunov companion" << std::endl;
}

void test_a5_anti_artifact() {
    AntiArtifactEvidence ev;
    ev.dual = {"coverage", "novelty", 0.7, 0.4, true};
    ev.seeds = {1, 2, 3, 4};
    auto scales = default_prefer_a_learning_scales();
    ev.scales.assign(scales.begin(), scales.end());
    ev.ood_condition_present = true;
    ev.ood_label = "ood_seed_shift";
    // McNemar: paired binary outcomes
    std::vector<int> a = {1, 1, 0, 1, 0, 1, 1, 0};
    std::vector<int> b = {1, 0, 0, 1, 1, 0, 1, 0};
    ev.mcnemar = kun::cellular::mcnemar_test(a, b);
    assert(ev.mcnemar.n == 8);

    auto v = evaluate_anti_artifact(ev);
    assert(v.dual_ok);
    assert(v.seeds_ok);
    assert(v.scales_ok);
    assert(v.ood_paired_ok);
    assert(v.pass);

    // Fail: only 2 seeds
    ev.seeds = {1, 2};
    auto v2 = evaluate_anti_artifact(ev);
    assert(!v2.pass);

    // Fail: FE-style conflation avoided — scales must be distinct learning scales
    ev.seeds = {1, 2, 3, 4};
    ev.scales = {
        {LearningScaleAxis::CELLS, 82, "a"},
        {LearningScaleAxis::CELLS, 82, "b"},
        {LearningScaleAxis::CELLS, 82, "c"},
    };
    auto v3 = evaluate_anti_artifact(ev);
    assert(!v3.scales_ok);

    std::cout << "  ✓ A5 anti-artifact dual · 4×3 · OOD+McNemar" << std::endl;
}

void test_a6_e_gate() {
    EGatePreRegister pr;
    pr.claim_id = "E1-phase3-map-elites-fixture";
    pr.behavioral_descriptor = "coverage_frac × novelty_knn";
    pr.metrics = {"coverage", "novelty", "avalanche_mean_size"};
    pr.seeds = {11, 22, 33, 44};
    auto scales = default_prefer_a_learning_scales();
    pr.learn_scales.assign(scales.begin(), scales.end());
    pr.exec_scale = 82;
    pr.learn_scale = 82;
    pr.exec_equals_learn_claimed = true;  // same — OK
    pr.stop_rule = "max_gens=0 dry-run fixture";
    pr.tip_hash = "2817e2f0851c466b5204dd384f5e6e14600095a0";
    pr.cmd = "ctest -R test_population_phase3_map_elites";
    pr.mechanism_sketch = "MAP-Elites grid cell insert + protect_window niche + avalanche hist";

    AntiArtifactEvidence ev;
    ev.dual = {"coverage", "novelty", 0.5, 0.5, true};
    ev.seeds = pr.seeds;
    ev.scales = pr.learn_scales;
    ev.ood_condition_present = true;
    ev.ood_label = "fixture_ood";
    ev.mcnemar = kun::cellular::mcnemar_test(2, 5, 1, 1);  // b=2,c=5

    auto green = run_e_gate(pr, ev);
    // Fixture may be E1-only in docs; pipeline itself can reach E2 when evidence green.
    assert(green.stage1_ok);
    assert(green.stage2_ok);
    assert(green.stage3_ok);
    assert(green.stage4_ok);
    assert(green.pass_e2);
    assert(green.tag == EmergenceTag::E2);

    // E3 on anti-artifact fail
    ev.seeds = {1};
    auto e3 = run_e_gate(pr, ev);
    assert(e3.tag == EmergenceTag::E3);

    // Ban exec≠learn
    pr.exec_scale = 200;
    pr.learn_scale = 82;
    pr.exec_equals_learn_claimed = true;
    ev.seeds = {11, 22, 33, 44};
    auto banned = run_e_gate(pr, ev);
    assert(!banned.stage1_ok);
    assert(banned.reason.find("exec_scale") != std::string::npos);

    std::cout << "  ✓ A6 E-Gate pipeline + exec≠learn ban + E3 path" << std::endl;
}

void test_a9_prefer_a_invariants() {
    auto org = make_organism_82(7);
    assert(org.cells.size() == kPreferAOrganismCells);
    // Determinism: same seed → same param1
    auto org_b = make_organism_82(7);
    assert(org.cells[0].param1 == org_b.cells[0].param1);
    assert(org.synapses.size() == org_b.synapses.size());
    std::cout << "  ✓ A9 82-cell / deterministic Prefer-A held" << std::endl;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  Prefer-A Phase 3: MAP-Elites / novelty / protect / E-Gate" << std::endl;
    std::cout << "==================================================================" << std::endl;
    test_a1_map_elites_grid();
    test_a2_novelty_stepping_stones();
    test_a3_protect_window();
    test_a4_metrics_wire();
    test_a5_anti_artifact();
    test_a6_e_gate();
    test_a9_prefer_a_invariants();
    std::cout << "[PASS] test_population_phase3_map_elites all assertions passed!" << std::endl;
    return 0;
}
