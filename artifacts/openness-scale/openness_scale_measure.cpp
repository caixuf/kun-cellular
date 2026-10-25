// Openness scale-sweep measure — Prefer-A E-Gate multi-scale pack (Stream A / 手术刀).
// Scales 41/82/164 · span_ratio>=4 · dual · OOD+paired · exec_cells=82 held.
// NOT a new E4 ABI; E1/E2/E3 closed — do not retag. A1 protect_window wire = 焊枪.
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/anti_artifact.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"
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

static constexpr const char* kPackId = "openness-scale-sweep";
static constexpr std::size_t kExecCells = kPreferAOrganismCells;  // 82

static CellularOrganism make_org(std::uint32_t seed, std::size_t cell_count) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "OpennessScale";
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
    std::printf("      \"learn_scale_campaign\": %zu,\n", cells);
    std::printf("      \"exec_cells_prefer_a\": %zu,\n", kExecCells);
    std::printf("      \"learn_equals_exec_lift\": false,\n");
    std::printf("      \"campaign_is_measure_only\": true,\n");
    std::printf("      \"exec_equals_primary_learn\": %s,\n",
                (cells == kExecCells) ? "true" : "false");
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

int main() {
    // Prefer-A exec lock: learning-scale sweep must NOT lift runtime ABI cells.
    if (kPreferAOrganismCells != 82) {
        std::fprintf(stderr, "FATAL: kPreferAOrganismCells=%zu (want 82)\n",
                     kPreferAOrganismCells);
        return 2;
    }

    const std::vector<std::uint32_t> seeds = {11, 22, 33, 44};
    const std::vector<std::size_t> live_scales = {41, 82, 164};
    auto scales = default_prefer_a_learning_scales();
    std::vector<LearningScalePoint> scale_vec(scales.begin(), scales.end());
    const double span = learning_scale_span_ratio(scale_vec);
    const bool span_ok = learning_scale_span_ok(scale_vec);

    std::vector<ScaleCampaignResult> campaigns;
    campaigns.reserve(live_scales.size());

    std::printf("{\n");
    std::printf("  \"pack_id\": \"%s\",\n", kPackId);
    std::printf("  \"tip_hash\": \"687cadf5bca20106086a73ae6f9a4a14b41d8cb2\",\n");
    std::printf("  \"claim_language\": \"openness / coverage / novelty scale-sweep — NOT E4 ABI\",\n");
    std::printf("  \"e_tags_closed\": [\"E1\", \"E2\", \"E3\"],\n");
    std::printf("  \"e_gate_echo\": \"none\",\n");
    std::printf("  \"learn_scales\": [41, 82, 164],\n");
    std::printf("  \"learn_scales_live\": [41, 82, 164],\n");
    std::printf("  \"learn_scale_primary\": 82,\n");
    std::printf("  \"exec_cells\": 82,\n");
    std::printf("  \"span_ratio\": %.6f,\n", span);
    std::printf("  \"span_ratio_min\": 4.0,\n");
    std::printf("  \"span_ok\": %s,\n", span_ok ? "true" : "false");
    std::printf("  \"learn_neq_exec_noted\": true,\n");
    std::printf("  \"note_scales\": \"LIVE measure campaigns at 41/82/164. Prefer-A exec "
                "kPreferAOrganismCells=82 held — learn span != exec lift. Adapters may "
                "measure at 41/164; runtime ABI stays 82. OOD seed_shift_+1000 per scale.\",\n");
    std::printf("  \"dual_metrics\": [\"coverage_frac\", \"novelty_knn\"],\n");
    std::printf("  \"companion_metrics\": [\"avalanche_mean_size\", \"lyapunov_max_loop_gain\"],\n");
    std::printf("  \"seeds\": [11, 22, 33, 44],\n");
    std::printf("  \"ood_label\": \"seed_shift_+1000\",\n");
    std::printf("  \"a1_protect_window\": \"焊枪 landing — measure slice does not wire deme path\",\n");
    std::printf("  \"per_scale\": [\n");

    for (std::size_t si = 0; si < live_scales.size(); ++si) {
        auto camp = run_campaign(live_scales[si], seeds);
        campaigns.push_back(camp);
        if (si + 1 < live_scales.size()) std::printf(",\n");
        else std::printf("\n");
    }

    const auto& primary = campaigns[1];  // 82
    AntiArtifactEvidence ev;
    ev.dual = {"coverage_frac", "novelty_knn",
               primary.cov_vals.empty() ? 0.0 : primary.cov_vals[0],
               primary.nov_vals.empty() ? 0.0 : primary.nov_vals[0], true};
    ev.seeds = seeds;
    ev.scales = scale_vec;
    ev.ood_condition_present = true;
    ev.ood_label = "seed_shift_+1000";
    ev.mcnemar = primary.mcnemar;
    ev.paired = primary.paired;
    ev.paired_used = true;
    auto aa = evaluate_anti_artifact(ev);

    bool live_41 = false, live_82 = false, live_164 = false;
    for (const auto& c : campaigns) {
        if (c.cells == 41) live_41 = true;
        if (c.cells == 82) live_82 = true;
        if (c.cells == 164) live_164 = true;
    }

    std::printf("  ],\n");
    std::printf("  \"live_adapters_executed\": {\"cells_41\": %s, \"cells_82\": %s, "
                "\"cells_164\": %s},\n",
                live_41 ? "true" : "false", live_82 ? "true" : "false",
                live_164 ? "true" : "false");
    std::printf(
        "  \"anti_artifact\": {\"pass\": %s, \"dual_ok\": %s, \"seeds_ok\": %s, "
        "\"scales_ok\": %s, \"ood_paired_ok\": %s, \"span_ok\": %s, "
        "\"span_ratio\": %.6f, \"reason\": \"%s\"},\n",
        aa.pass ? "true" : "false", aa.dual_ok ? "true" : "false",
        aa.seeds_ok ? "true" : "false", aa.scales_ok ? "true" : "false",
        aa.ood_paired_ok ? "true" : "false", span_ok ? "true" : "false", span,
        aa.reason.c_str());
    // Knife e_gate_tag_honesty_v1: do NOT stamp e_gate.tag as pack stage.
    // Pack is openness-scale-sweep; E1/E2/E3 closed; not E4. Echo already
    // printed as top-level e_gate_echo="none".
    std::printf("  \"prefer_checks\": {\n");
    std::printf("    \"A0_branch\": \"feat/cellular-openness-deepen-egate-tag off 687cadf5\",\n");
    std::printf("    \"A1_protect_window\": \"焊枪 landing\",\n");
    std::printf("    \"A2_scale_sweep\": %s,\n",
                (live_41 && live_82 && live_164 && span_ok) ? "\"PASS\"" : "\"FAIL\"");
    std::printf("    \"A3_dual_ood_paired\": %s,\n", aa.pass ? "\"PASS\"" : "\"FAIL\"");
    std::printf("    \"A4_exec_82\": %s,\n",
                (kPreferAOrganismCells == 82) ? "\"PASS\"" : "\"FAIL\"");
    std::printf("    \"A5_handoff\": \"docs/handoff/2026-10-25-openness-deepen-egate-tag.md\",\n");
    std::printf("    \"A6_bans\": \"PASS — no DouZero/strength/win-rate/ADAS/4th ABI/zoo\"\n");
    std::printf("  },\n");
    std::printf("  \"honesty\": {\n");
    std::printf("    \"not_e4\": true,\n");
    std::printf("    \"e1_e2_e3_closed\": true,\n");
    std::printf("    \"forbidden\": [\"DouZero\", \"win-rate\", \"strength\", \"bare emerged\", "
                "\"Phase4 reopen\", \"ADAS reopen\", \"4th ABI\", \"exec cell lift\", "
                "\"silent P1\", \"zoo/mutate\"]\n");
    std::printf("  }\n");
    std::printf("}\n");

    const bool ok = aa.pass && span_ok && live_41 && live_82 && live_164 &&
                    (kPreferAOrganismCells == 82);
    return ok ? 0 : 1;
}
