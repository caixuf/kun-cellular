// E-Gate E3 openness measure — Prefer A controlled demotion (honest artifact label).
// Reuses E2 live adapters (41/82/164) + dual·≥3·≥4·OOD plumbing, then intentionally
// fails Prefer dual-independence so run_e_gate demotes stage2 → E3.
// NOT emergence success. tip_hash base: ab885f994fd2e6f4fbe257a33d7255368798b5d4
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
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace kun;
using namespace kun::population;

static constexpr const char* kTipHash =
    "ab885f994fd2e6f4fbe257a33d7255368798b5d4";
static constexpr const char* kClaimId = "E3-egate-openness";
static constexpr std::size_t kPrimaryLearnCells = 82;
// Controlled demotion lever: dual names present, but independence asserted false.
static constexpr bool kDemoteDualIndependence = false;

static CellularOrganism make_org(std::uint32_t seed, std::size_t cell_count) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "EGateE3";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> pdist(0.01, 0.99);
    std::uniform_real_distribution<double> wdist(-0.8, 0.8);
    static const CellType kTypes[] = {
        CellType::SENSE_RAW_INPUT_0, CellType::SENSE_RAW_INPUT_1,
        CellType::OP_EMA, CellType::OP_DIFF, CellType::OP_SUM, CellType::OP_SUB,
        CellType::GATE_HYSTERESIS, CellType::GATE_THRESHOLD,
        CellType::ACT_PRIMARY_POSITIVE, CellType::ACT_PRIMARY_NEGATIVE};
    org.cells.reserve(cell_count);
    for (std::size_t i = 0; i < cell_count; ++i) {
        Cell c;
        c.id = static_cast<std::uint32_t>(i);
        c.type = kTypes[i % 10];
        c.param1 = pdist(rng);
        c.param2 = pdist(rng) * 0.1;
        org.cells.push_back(c);
    }
    const std::size_t n_syn = cell_count * 2;
    org.synapses.reserve(n_syn);
    for (std::size_t s = 0; s < n_syn; ++s) {
        Synapse syn;
        syn.from_cell_id = static_cast<std::uint32_t>(s % cell_count);
        syn.to_cell_id = static_cast<std::uint32_t>((s * 7 + 3) % cell_count);
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
    const double denom = static_cast<double>(org.cells.size());
    if (cov.reachable_cells <= 0 || denom <= 0.0) return 0.0;
    return std::min(1.0, static_cast<double>(cov.reachable_cells) / denom);
}

struct ScaleCampaignResult {
    std::size_t cells{0};
    std::size_t avalanche_sites{0};
    bool avalanche_capped{false};
    std::vector<double> cov_vals;
    std::vector<double> nov_vals;
    std::vector<double> ood_vals;
    std::vector<int> bin_a;
    std::vector<int> bin_b;
    double map_coverage{0.0};
    std::size_t map_occupied{0};
    std::size_t niche_count{0};
    cellular::McNemarResult mcnemar{};
    cellular::PairedDifferenceResult paired{};
};

static ScaleCampaignResult run_campaign(std::size_t cells,
                                        const std::vector<std::uint32_t>& seeds) {
    ScaleCampaignResult out;
    out.cells = cells;
    out.avalanche_sites = std::min(cells, kAvalancheMaxSites);
    out.avalanche_capped = (cells > kAvalancheMaxSites);

    MapElitesGrid grid(2, 8, 0.3f);
    NoveltySteppingStoneArchive archive(64, 3);
    PopulationNicheRegistry niches(0.35f, 5);

    std::printf("    {\n");
    std::printf("      \"cells\": %zu,\n", cells);
    std::printf("      \"exec_scale\": %zu,\n", cells);
    std::printf("      \"learn_scale_campaign\": %zu,\n", cells);
    std::printf("      \"learn_scale_primary\": %zu,\n", kPrimaryLearnCells);
    std::printf("      \"exec_equals_learn_campaign\": true,\n");
    std::printf("      \"exec_equals_primary_learn\": %s,\n",
                (cells == kPrimaryLearnCells) ? "true" : "false");
    std::printf("      \"live_organism_cells\": %zu,\n", cells);
    std::printf("      \"synapse_count\": %zu,\n", cells * 2);
    std::printf("      \"avalanche_sites\": %zu,\n", out.avalanche_sites);
    std::printf("      \"avalanche_sites_capped_at_128\": %s,\n",
                out.avalanche_capped ? "true" : "false");
    std::printf("      \"per_seed\": [\n");

    for (std::size_t i = 0; i < seeds.size(); ++i) {
        auto org = make_org(seeds[i], cells);
        if (org.cells.size() != cells) {
            std::fprintf(stderr, "FATAL: make_org(%u,%zu) size=%zu\n", seeds[i], cells,
                         org.cells.size());
            std::exit(2);
        }
        const double reach = reach_of(org);
        std::vector<double> beh = {reach, static_cast<double>(seeds[i] % 17) / 17.0};
        const double nov = archive.compute_novelty(beh);
        archive.add_behavior(beh);
        auto cov = org.evaluate_coverage_contract();
        auto bd = bd_from_coverage_novelty(cov, static_cast<float>(nov));
        grid.try_insert(static_cast<std::uint32_t>(i + 1), bd, static_cast<float>(reach),
                        static_cast<float>(nov));
        niches.place_or_speciate(static_cast<std::uint32_t>(i + 1), reach, nov, true);

        AvalancheMetric av(out.avalanche_sites, 1.0f);
        std::vector<float> act(out.avalanche_sites, 0.0f);
        for (std::size_t c = 0; c < out.avalanche_sites; ++c) {
            act[c] = static_cast<float>(org.cells[c].param1);
        }
        av.ingest_activity(act.data(), act.size(), 2.0f);
        for (int g = 0; g < 16; ++g) {
            av.drive(static_cast<std::size_t>(g * 5), 1.2f);
        }
        auto ar = av.report();
        auto ly = org.check_lyapunov_stability();

        auto org_ood = make_org(seeds[i] + 1000u, cells);
        const double reach_ood = reach_of(org_ood);
        const int pass_pri = reach >= 0.15 ? 1 : 0;
        const int pass_ood = reach_ood >= 0.15 ? 1 : 0;
        out.bin_a.push_back(pass_pri);
        out.bin_b.push_back(pass_ood);
        out.cov_vals.push_back(reach);
        out.nov_vals.push_back(nov);
        out.ood_vals.push_back(reach_ood);

        std::printf(
            "        {\"seed\": %u, \"cells\": %zu, \"coverage_frac\": %.6f, "
            "\"novelty_knn\": %.6f, \"avalanche_mean_size\": %.6f, "
            "\"avalanche_branching\": %.6f, \"lyapunov_max_loop_gain\": %.6f, "
            "\"ood_seed\": %u, \"ood_coverage_frac\": %.6f, \"pass_pri\": %d, "
            "\"pass_ood\": %d}%s\n",
            seeds[i], org.cells.size(), reach, nov, ar.mean_size, ar.branching_ratio,
            ly.max_loop_gain, seeds[i] + 1000u, reach_ood, pass_pri, pass_ood,
            (i + 1 < seeds.size()) ? "," : "");
    }

    out.mcnemar = kun::cellular::mcnemar_test(out.bin_a, out.bin_b);
    out.paired = kun::cellular::analyze_paired_differences(out.cov_vals, out.ood_vals);
    out.map_coverage = grid.coverage_ratio();
    out.map_occupied = grid.occupied_count();
    out.niche_count = niches.niches().size();

    std::printf("      ],\n");
    std::printf("      \"map_elites_coverage_ratio\": %.6f,\n", out.map_coverage);
    std::printf("      \"map_elites_occupied\": %zu,\n", out.map_occupied);
    std::printf("      \"niche_count\": %zu,\n", out.niche_count);
    std::printf(
        "      \"mcnemar\": {\"n\": %llu, \"b\": %llu, \"c\": %llu, \"chi2\": %.6f, "
        "\"p\": %.6e},\n",
        static_cast<unsigned long long>(out.mcnemar.n),
        static_cast<unsigned long long>(out.mcnemar.b),
        static_cast<unsigned long long>(out.mcnemar.c), out.mcnemar.chi2,
        out.mcnemar.p_value);
    std::printf("      \"paired_coverage\": {\"n\": %zu, \"mean_diff\": %.6f, \"p\": %.6e}\n",
                out.paired.n, out.paired.mean_diff, out.paired.p_value);
    std::printf("    }");
    return out;
}

static const char* stage_name(EGateStage s) {
    switch (s) {
        case EGateStage::PRE_REGISTER: return "PRE_REGISTER";
        case EGateStage::ANTI_ARTIFACT: return "ANTI_ARTIFACT";
        case EGateStage::EXPLAINABLE: return "EXPLAINABLE";
        case EGateStage::REPRODUCIBLE: return "REPRODUCIBLE";
        default: return "UNKNOWN";
    }
}

int main() {
    const std::vector<std::uint32_t> seeds = {11, 22, 33, 44};
    const std::vector<std::size_t> live_scales = {41, 82, 164};
    auto scales = default_prefer_a_learning_scales();

    std::vector<ScaleCampaignResult> campaigns;
    campaigns.reserve(live_scales.size());

    std::printf("{\n");
    std::printf("  \"claim_id\": \"%s\",\n", kClaimId);
    std::printf("  \"tip_hash\": \"%s\",\n", kTipHash);
    std::printf("  \"pack_kind\": \"controlled_demotion\",\n");
    std::printf("  \"intended_demotion\": {\n");
    std::printf("    \"prefer_check\": \"dual_independence\",\n");
    std::printf("    \"lever\": \"AntiArtifactEvidence.dual.independent=false\",\n");
    std::printf("    \"expected_failed_stage\": 2,\n");
    std::printf("    \"expected_tag\": \"E3\",\n");
    std::printf("    \"note\": \"Vocabulary dual·>=3·>=4·OOD+paired still invoked; "
                "dual names+values present; independence intentionally false so "
                "evaluate_anti_artifact fails measurable Prefer check.\"\n");
    std::printf("  },\n");
    std::printf("  \"learn_scales\": [41, 82, 164],\n");
    std::printf("  \"learn_scales_live\": [41, 82, 164],\n");
    std::printf("  \"learn_scale_primary\": 82,\n");
    std::printf("  \"note_scales\": \"LIVE cell-count campaigns reuse E2 adapters at 41/82/164. "
                "Prefer-A ABI untouched. Plumbing invoked; demotion via dual_independence.\",\n");
    std::printf("  \"dual_metrics\": [\"coverage_frac\", \"novelty_knn\"],\n");
    std::printf("  \"dual_independent_asserted\": false,\n");
    std::printf("  \"companion_metrics\": [\"avalanche_mean_size\", \"lyapunov_max_loop_gain\"],\n");
    std::printf("  \"seeds\": [11, 22, 33, 44],\n");
    std::printf("  \"ood_label\": \"seed_shift_+1000\",\n");
    std::printf("  \"per_scale\": [\n");

    for (std::size_t si = 0; si < live_scales.size(); ++si) {
        auto camp = run_campaign(live_scales[si], seeds);
        campaigns.push_back(camp);
        if (si + 1 < live_scales.size()) std::printf(",\n");
        else std::printf("\n");
    }

    const auto& primary = campaigns[1];  // 82
    AntiArtifactEvidence ev;
    // Invoke dual vocabulary (names + values) but demote independence.
    ev.dual = {"coverage_frac", "novelty_knn",
               primary.cov_vals.empty() ? 0.0 : primary.cov_vals[0],
               primary.nov_vals.empty() ? 0.0 : primary.nov_vals[0],
               kDemoteDualIndependence};
    ev.seeds = seeds;
    ev.scales.assign(scales.begin(), scales.end());
    ev.ood_condition_present = true;
    ev.ood_label = "seed_shift_+1000";
    ev.mcnemar = primary.mcnemar;
    ev.paired = primary.paired;
    ev.paired_used = true;
    auto aa = evaluate_anti_artifact(ev);

    EGatePreRegister pr;
    pr.claim_id = kClaimId;
    pr.behavioral_descriptor = "coverage_frac x novelty_knn";
    pr.metrics = {"coverage_frac", "novelty_knn", "avalanche_mean_size"};
    pr.seeds = seeds;
    pr.learn_scales = ev.scales;
    pr.exec_scale = 82;
    pr.learn_scale = 82;
    pr.exec_equals_learn_claimed = true;
    pr.stop_rule =
        "max_gens=0 E3 controlled demotion; invoke live 41/82/164 + dual·>=3·>=4·OOD; "
        "fail Prefer dual_independence; Prefer-A ABI held";
    pr.tip_hash = kTipHash;
    pr.cmd =
        "g++ -std=c++20 -O2 -Iinclude "
        "artifacts/egate-e3/egate_e3_demotion_measure.cpp "
        "-o artifacts/egate-e3/egate_e3_demotion_measure && "
        "./artifacts/egate-e3/egate_e3_demotion_measure";
    // Mechanism present so demotion is stage2 (not stage3 empty).
    pr.mechanism_sketch =
        "Reuse E2 adapters make_org(seed,cell_count) live 41/82/164; MapElitesGrid + "
        "NoveltySteppingStone + PopulationSpeciesNiche; AvalancheMetric companion; "
        "Lyapunov companion NOT avalanche; CONTROLLED DEMOTION: dual.independent=false";
    auto eg = run_e_gate(pr, ev);

    bool live_41 = false, live_82 = false, live_164 = false;
    for (const auto& c : campaigns) {
        if (c.cells == 41) live_41 = true;
        if (c.cells == 82) live_82 = true;
        if (c.cells == 164) live_164 = true;
    }

    const int failed_stage_num = static_cast<int>(eg.failed_stage);

    std::printf("  ],\n");
    std::printf("  \"live_adapters_executed\": {\"cells_41\": %s, \"cells_82\": %s, "
                "\"cells_164\": %s},\n",
                live_41 ? "true" : "false", live_82 ? "true" : "false",
                live_164 ? "true" : "false");
    std::printf(
        "  \"anti_artifact\": {\"pass\": %s, \"dual_ok\": %s, \"seeds_ok\": %s, "
        "\"scales_ok\": %s, \"ood_paired_ok\": %s, \"reason\": \"%s\"},\n",
        aa.pass ? "true" : "false", aa.dual_ok ? "true" : "false",
        aa.seeds_ok ? "true" : "false", aa.scales_ok ? "true" : "false",
        aa.ood_paired_ok ? "true" : "false", aa.reason.c_str());
    std::printf(
        "  \"e_gate\": {\"tag\": \"%s\", \"pass_e2\": %s, \"failed_stage\": %d, "
        "\"failed_stage_name\": \"%s\", \"stage1_ok\": %s, \"stage2_ok\": %s, "
        "\"stage3_ok\": %s, \"stage4_ok\": %s, \"reason\": \"%s\"},\n",
        emergence_tag_name(eg.tag), eg.pass_e2 ? "true" : "false", failed_stage_num,
        stage_name(eg.failed_stage), eg.stage1_ok ? "true" : "false",
        eg.stage2_ok ? "true" : "false", eg.stage3_ok ? "true" : "false",
        eg.stage4_ok ? "true" : "false", eg.reason.c_str());
    std::printf("  \"honesty\": {\n");
    std::printf("    \"claimed_tag\": \"E3\",\n");
    std::printf("    \"pipeline_tag\": \"%s\",\n", emergence_tag_name(eg.tag));
    std::printf(
        "    \"why_e3\": \"Controlled demotion Prefer-A: dual·>=3·>=4·OOD+paired plumbing "
        "invoked on live 41/82/164; dual.independent=false → stage2 anti-artifact fail → "
        "E3. NOT emergence / almost-E2 / success.\",\n");
    std::printf("    \"prefer_check_failed\": \"dual_independence\",\n");
    std::printf("    \"failed_stage\": %d,\n", failed_stage_num);
    std::printf("    \"e1_e2_e3\": \"E1=#14 lit unchanged; E2=#16 @ ab885f99 lit unchanged; "
                "E3=this pack\",\n");
    std::printf("    \"forbidden\": [\"DouZero\", \"win-rate\", \"strength\", \"bare emerged\", "
                "\"E3-as-emergence\", \"Phase4\", \"ADAS reopen\", \"4th ABI\", "
                "\"retag E1/E2\", \"skip vocabulary then fiat E3\"]\n");
    std::printf("  }\n");
    std::printf("}\n");

    // Success of THIS measure = honest E3 demotion demonstrated (exit 0).
    const bool demotion_ok =
        (eg.tag == EmergenceTag::E3) && !eg.pass_e2 && !aa.pass && !aa.dual_ok &&
        (eg.failed_stage == EGateStage::ANTI_ARTIFACT) && eg.stage1_ok && live_41 &&
        live_82 && live_164 && aa.seeds_ok && aa.scales_ok && aa.ood_paired_ok;
    return demotion_ok ? 0 : 1;
}
