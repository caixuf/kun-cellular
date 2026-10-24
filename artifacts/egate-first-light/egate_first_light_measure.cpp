// E-Gate first-light measure dump — Prefer-A reuse only (no new ABI).
// Not a 4th genome ABI. Links header-only kun_population.
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/anti_artifact.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/e_gate.hpp"
#include "kun/cellular/population/map_elites_grid.hpp"
#include "kun/cellular/population/novelty_stepping_stone.hpp"
#include "kun/cellular/population/phase3_metrics.hpp"
#include "kun/cellular/population/population_species_niche.hpp"
#include "kun/cellular/statistical_evaluation.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace kun;
using namespace kun::population;

static CellularOrganism make_org(std::uint32_t seed) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "EGateFL";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pdist(0.01, 0.99);
    std::uniform_real_distribution<double> wdist(-0.8, 0.8);
    static const CellType kTypes[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_SUM, CellType::OP_SUB,
        CellType::GATE_HYSTERESIS, CellType::GATE_THRESHOLD,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE};
    org.cells.reserve(kPreferAOrganismCells);
    for (std::size_t i = 0; i < kPreferAOrganismCells; ++i) {
        Cell c;
        c.id = static_cast<std::uint32_t>(i);
        c.type = kTypes[i % 10];
        c.param1 = pdist(rng);
        c.param2 = pdist(rng) * 0.1;
        org.cells.push_back(c);
    }
    for (std::size_t s = 0; s < 164; ++s) {
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
    return org;
}

static double reach_of(const CellularOrganism& org) {
    auto cov = org.evaluate_coverage_contract();
    if (cov.reachable_cells <= 0) return 0.0;
    return std::min(1.0, static_cast<double>(cov.reachable_cells) / 82.0);
}

int main() {
    const std::vector<std::uint32_t> seeds = {11, 22, 33, 44};
    auto scales = default_prefer_a_learning_scales();
    MapElitesGrid grid(2, 8, 0.3f);
    NoveltySteppingStoneArchive archive(64, 3);
    PopulationNicheRegistry niches(0.35f, 5);
    std::vector<double> cov_vals, nov_vals, ood_vals;
    std::vector<int> bin_a, bin_b;

    std::printf("{\n");
    std::printf("  \"claim_id\": \"E1-egate-first-light\",\n");
    std::printf("  \"tip_hash\": \"bff1c509169c706eaa74d03640c9566100f54787\",\n");
    std::printf("  \"learn_scales\": [41, 82, 164],\n");
    std::printf("  \"learn_scale_primary\": 82,\n");
    std::printf("  \"exec_scale\": 82,\n");
    std::printf("  \"exec_equals_learn_claimed\": true,\n");
    std::printf("  \"note_scales\": \"Prefer-A organism builder fixed at 82 cells; 41/164 registered learn-scale points; live dump 82-only (adapter gap for multi-cell-count).\",\n");
    std::printf("  \"dual_metrics\": [\"coverage_frac\", \"novelty_knn\"],\n");
    std::printf("  \"companion_metrics\": [\"avalanche_mean_size\", \"lyapunov_max_loop_gain\"],\n");
    std::printf("  \"seeds\": [11, 22, 33, 44],\n");
    std::printf("  \"per_seed\": [\n");

    for (std::size_t i = 0; i < seeds.size(); ++i) {
        auto org = make_org(seeds[i]);
        const double reach = reach_of(org);
        std::vector<double> beh = {reach, static_cast<double>(seeds[i] % 17) / 17.0};
        const double nov = archive.compute_novelty(beh);
        archive.add_behavior(beh);
        auto cov = org.evaluate_coverage_contract();
        auto bd = bd_from_coverage_novelty(cov, static_cast<float>(nov));
        grid.try_insert(static_cast<std::uint32_t>(i + 1), bd, static_cast<float>(reach),
                        static_cast<float>(nov));
        niches.place_or_speciate(static_cast<std::uint32_t>(i + 1), reach, nov, true);

        AvalancheMetric av(kPreferAOrganismCells, 1.0f);
        std::vector<float> act(kPreferAOrganismCells, 0.0f);
        for (std::size_t c = 0; c < kPreferAOrganismCells; ++c) {
            act[c] = static_cast<float>(org.cells[c].param1);
        }
        av.ingest_activity(act.data(), act.size(), 2.0f);
        for (int g = 0; g < 16; ++g) {
            av.drive(static_cast<std::size_t>(g * 5), 1.2f);
        }
        auto ar = av.report();
        auto ly = org.check_lyapunov_stability();

        auto org_ood = make_org(seeds[i] + 1000u);
        const double reach_ood = reach_of(org_ood);
        const int pass_pri = reach >= 0.15 ? 1 : 0;
        const int pass_ood = reach_ood >= 0.15 ? 1 : 0;
        bin_a.push_back(pass_pri);
        bin_b.push_back(pass_ood);
        cov_vals.push_back(reach);
        nov_vals.push_back(nov);
        ood_vals.push_back(reach_ood);

        std::printf(
            "    {\"seed\": %u, \"coverage_frac\": %.6f, \"novelty_knn\": %.6f, "
            "\"avalanche_mean_size\": %.6f, \"avalanche_branching\": %.6f, "
            "\"lyapunov_max_loop_gain\": %.6f, \"ood_seed\": %u, "
            "\"ood_coverage_frac\": %.6f, \"pass_pri\": %d, \"pass_ood\": %d}%s\n",
            seeds[i], reach, nov, ar.mean_size, ar.branching_ratio, ly.max_loop_gain,
            seeds[i] + 1000u, reach_ood, pass_pri, pass_ood,
            (i + 1 < seeds.size()) ? "," : "");
    }

    auto mc = kun::cellular::mcnemar_test(bin_a, bin_b);
    auto paired = kun::cellular::analyze_paired_differences(cov_vals, ood_vals);

    AntiArtifactEvidence ev;
    ev.dual = {"coverage_frac", "novelty_knn",
               cov_vals.empty() ? 0.0 : cov_vals[0],
               nov_vals.empty() ? 0.0 : nov_vals[0], true};
    ev.seeds = seeds;
    ev.scales.assign(scales.begin(), scales.end());
    ev.ood_condition_present = true;
    ev.ood_label = "seed_shift_+1000";
    ev.mcnemar = mc;
    ev.paired = paired;
    ev.paired_used = true;
    auto aa = evaluate_anti_artifact(ev);

    EGatePreRegister pr;
    pr.claim_id = "E1-egate-first-light";
    pr.behavioral_descriptor = "coverage_frac x novelty_knn";
    pr.metrics = {"coverage_frac", "novelty_knn", "avalanche_mean_size"};
    pr.seeds = seeds;
    pr.learn_scales = ev.scales;
    pr.exec_scale = 82;
    pr.learn_scale = 82;
    pr.exec_equals_learn_claimed = true;
    pr.stop_rule =
        "max_gens=0 first-light measure; Prefer-A 82 live dump; scales registered 41/82/164";
    pr.tip_hash = "bff1c509169c706eaa74d03640c9566100f54787";
    pr.cmd =
        "ctest -R test_population_phase3_map_elites; "
        "g++ -std=c++20 -O2 -Iinclude artifacts/egate-first-light/egate_first_light_measure.cpp "
        "-o artifacts/egate-first-light/egate_first_light_measure && "
        "./artifacts/egate-first-light/egate_first_light_measure";
    pr.mechanism_sketch =
        "MapElitesGrid.try_insert BD(coverage,novelty); NoveltySteppingStoneArchive KNN feed; "
        "PopulationSpeciesNiche.protect_window; AvalancheMetric size/branching; "
        "Lyapunov companion NOT avalanche";
    auto eg = run_e_gate(pr, ev);

    std::printf("  ],\n");
    std::printf("  \"map_elites_coverage_ratio\": %.6f,\n", grid.coverage_ratio());
    std::printf("  \"map_elites_occupied\": %zu,\n", grid.occupied_count());
    std::printf("  \"niche_count\": %zu,\n", niches.niches().size());
    std::printf("  \"ood_label\": \"seed_shift_+1000\",\n");
    std::printf(
        "  \"mcnemar\": {\"n\": %llu, \"b\": %llu, \"c\": %llu, \"chi2\": %.6f, \"p\": %.6e},\n",
        static_cast<unsigned long long>(mc.n), static_cast<unsigned long long>(mc.b),
        static_cast<unsigned long long>(mc.c), mc.chi2, mc.p_value);
    std::printf("  \"paired_coverage\": {\"n\": %zu, \"mean_diff\": %.6f, \"p\": %.6e},\n",
                paired.n, paired.mean_diff, paired.p_value);
    std::printf(
        "  \"anti_artifact\": {\"pass\": %s, \"dual_ok\": %s, \"seeds_ok\": %s, "
        "\"scales_ok\": %s, \"ood_paired_ok\": %s, \"reason\": \"%s\"},\n",
        aa.pass ? "true" : "false", aa.dual_ok ? "true" : "false",
        aa.seeds_ok ? "true" : "false", aa.scales_ok ? "true" : "false",
        aa.ood_paired_ok ? "true" : "false", aa.reason.c_str());
    std::printf(
        "  \"e_gate\": {\"tag\": \"%s\", \"pass_e2\": %s, \"stage1_ok\": %s, "
        "\"stage2_ok\": %s, \"stage3_ok\": %s, \"stage4_ok\": %s, \"reason\": \"%s\"},\n",
        emergence_tag_name(eg.tag), eg.pass_e2 ? "true" : "false",
        eg.stage1_ok ? "true" : "false", eg.stage2_ok ? "true" : "false",
        eg.stage3_ok ? "true" : "false", eg.stage4_ok ? "true" : "false", eg.reason.c_str());
    std::printf("  \"honesty\": {\n");
    std::printf("    \"claimed_tag\": \"E1\",\n");
    std::printf("    \"pipeline_tag_if_green\": \"%s\",\n", emergence_tag_name(eg.tag));
    std::printf(
        "    \"why_not_e2\": \"Live organism dump fixed at Prefer-A 82 cells; 41/164 "
        "learn-scale points registered but not executed as distinct cell-count campaigns. "
        "OOD is seed-shift on same ABI. Honest E1.\",\n");
    std::printf("    \"forbidden\": [\"DouZero\", \"win-rate\", \"strength\", \"bare emerged\"]\n");
    std::printf("  }\n");
    std::printf("}\n");
    return (aa.pass && eg.stage1_ok) ? 0 : 1;
}
