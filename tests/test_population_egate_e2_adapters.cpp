// ============================================================================
// test_population_egate_e2_adapters.cpp — Prefer-A E2 live multi-scale builders
// Proves make_org(seed, N) at N=41 and N=164 yields org.cells.size()==N;
// reach_of normalizes by organism cell count; primary ABI 82 untouched.
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/anti_artifact.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::Synapse;
using kun::population::AvalancheMetric;
using kun::population::default_prefer_a_learning_scales;
using kun::population::kAvalancheMaxSites;
using kun::population::kPreferAOrganismCells;

namespace {

CellularOrganism make_org(std::uint32_t seed, std::size_t cell_count) {
    CellularOrganism org;
    org.organism_id = seed;
    org.lineage_name = "EGateE2-unit";
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

double reach_of(const CellularOrganism& org) {
    auto cov = org.evaluate_coverage_contract();
    const double denom = static_cast<double>(org.cells.size());
    if (cov.reachable_cells <= 0 || denom <= 0.0) return 0.0;
    return std::min(1.0, static_cast<double>(cov.reachable_cells) / denom);
}

void test_builders_live_scales() {
    for (std::size_t n : {std::size_t{41}, std::size_t{82}, std::size_t{164}}) {
        auto org = make_org(11, n);
        assert(org.cells.size() == n);
        assert(org.synapses.size() == n * 2);
        const double r = reach_of(org);
        assert(r >= 0.0 && r <= 1.0);
        const std::size_t av_n = std::min(n, kAvalancheMaxSites);
        AvalancheMetric av(av_n, 1.0f);
        assert(av.n_sites() == av_n);
        std::vector<float> act(av_n, 0.5f);
        av.ingest_activity(act.data(), act.size(), 1.0f);
        av.drive(0, 1.2f);
        auto rep = av.report();
        (void)rep;
    }
    std::cout << "  ✓ builders at 41/82/164: cells.size()==N; reach normalized; avalanche sized\n";
}

void test_prefer_a_abi_untouched() {
    assert(kPreferAOrganismCells == 82);
    auto scales = default_prefer_a_learning_scales();
    assert(scales.size() == 3);
    assert(scales[0].value == 41);
    assert(scales[1].value == 82);
    assert(scales[2].value == 164);
    std::cout << "  ✓ Prefer-A ABI: kPreferAOrganismCells=82; scales registered 41/82/164\n";
}

}  // namespace

int main() {
    std::cout << "==================================================================\n";
    std::cout << "  Prefer-A E2: live multi-scale adapters (41/82/164)\n";
    std::cout << "==================================================================\n";
    test_builders_live_scales();
    test_prefer_a_abi_untouched();
    std::cout << "[PASS] test_population_egate_e2_adapters\n";
    return 0;
}
