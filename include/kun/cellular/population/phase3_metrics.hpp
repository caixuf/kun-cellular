#pragma once
// ============================================================================
// population/phase3_metrics.hpp — Prefer-A Phase 3 metrics bundle + handoff table
//
// Wired metrics:
//   1) FunctionalCoverageContract (organism coverage / BD / gate input)
//   2) L1 compatibility_distance (species + mate gates)
//   3) AvalancheReport (new SOC signal)
//   4) LyapunovReport companion (stability; NOT avalanche substitute)
// ============================================================================
#include "kun/cellular/cellular_genome.hpp"
#include "kun/cellular/population/avalanche_metric.hpp"
#include "kun/cellular/population/compatibility_distance.hpp"
#include "kun/cellular/population/map_elites_grid.hpp"

#include <sstream>
#include <string>

namespace kun {
namespace population {

struct Phase3MetricSnapshot {
    FunctionalCoverageContract coverage{};
    double compat_distance{0.0};
    AvalancheReport avalanche{};
    CellularOrganism::LyapunovReport lyapunov{};
    float map_elites_coverage_ratio{0.0f};
    float novelty{0.0f};
};

struct Phase3HandoffRow {
    const char* claim;
    const char* metrics;
    const char* e_gate_stage;
};

// Claim → metrics → E-Gate stage (A4 handoff table).
inline constexpr Phase3HandoffRow kPhase3HandoffTable[] = {
    {"grid_coverage_openness",
     "MapElitesGrid.coverage_ratio + FunctionalCoverageContract",
     "pre-register + anti-artifact"},
    {"novelty_stepping_stone",
     "NoveltySteppingStoneArchive.novelty + PreferAFitnessDriver",
     "anti-artifact + explainable"},
    {"species_protect_window",
     "PopulationSpeciesNiche.protect_window + compatibility_distance",
     "explainable"},
    {"criticality_avalanche",
     "AvalancheReport (size hist / branching) ; Lyapunov companion",
     "anti-artifact + explainable"},
    {"exec_vs_learn_scale",
     "EGatePreRegister.learn_scale + exec_scale (must differ if claimed)",
     "pre-register"},
};

inline constexpr std::size_t kPhase3HandoffTableN =
    sizeof(kPhase3HandoffTable) / sizeof(kPhase3HandoffTable[0]);

inline std::string phase3_handoff_table_markdown() {
    std::ostringstream oss;
    oss << "| Claim | Metrics | E-Gate stage |\n|---|---|---|\n";
    for (std::size_t i = 0; i < kPhase3HandoffTableN; ++i) {
        const auto& r = kPhase3HandoffTable[i];
        oss << "| " << r.claim << " | " << r.metrics << " | " << r.e_gate_stage << " |\n";
    }
    return oss.str();
}

// Extract Prefer-A BD defaults from coverage + novelty.
inline BehaviorDescriptor bd_from_coverage_novelty(const FunctionalCoverageContract& cov,
                                                   float novelty) {
    BehaviorDescriptor bd;
    bd.dims = 2;
    const float reach = (cov.reachable_cells > 0)
        ? std::min(1.0f, static_cast<float>(cov.reachable_cells) / 82.0f)
        : 0.0f;
    bd[0] = reach;
    bd[1] = std::clamp(novelty, 0.0f, 1.0f);
    return bd;
}

// Wire Lyapunov companion onto snapshot (caller supplies organism).
inline void attach_lyapunov_companion(Phase3MetricSnapshot& snap, const CellularOrganism& org) {
    snap.lyapunov = org.check_lyapunov_stability();
}

}  // namespace population
}  // namespace kun
