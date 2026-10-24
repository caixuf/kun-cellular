#pragma once
// ============================================================================
// population/compatibility_distance.hpp — Prefer-A compatibility / speciation distance
//
// Formula parity vs SpeciationMetric::compatibility_distance
// (morphogenetic_population.hpp:25–48):
//   distance = c1 * (disjoint / max_n) + c2 * avg_weight_diff
//   matching locus: aligned index i with equal op/type
//   disjoint_count = max_n - matching_count
//   avg_weight_diff = mean |weight_param| over matches; 1.0 if none
// Defaults: c1=1.0, c2=0.4 (identical to SpeciationMetric).
// Default speciation_threshold = 0.35 (identical to MorphogeneticEcosystem knob).
//
// Prefer-A adapters (no ReplicableGenome heap loci ABI):
//   CompactSoAGenome     — cell slots: op_types + gains as weight_param
//   CorticalMacroArray   — flatten columns in order (column-major cell loci)
//   CellularOrganism     — cell slots: CellType + param1 as weight_param
// Synapse topology is NOT a SpeciationMetric locus; shape gates live in cross_into.
// ============================================================================
#include "kun/cellular/cortical_column.hpp"
#include "kun/cellular/cellular_genome.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kun {
namespace population {

inline constexpr double kCompatC1 = 1.0;
inline constexpr double kCompatC2 = 0.4;
inline constexpr float  kDefaultSpeciationThreshold = 0.35f;

// Prefer-A DouDiZhu / A8 champion shape (cells conserved under sexual cross).
inline constexpr std::size_t kPreferAOrganismCells = 82;

namespace detail {

inline double neat_ish_distance(std::size_t n1, std::size_t n2,
                                std::size_t matching_count,
                                double weight_diff_sum,
                                double c1, double c2) {
    const std::size_t max_n = std::max(n1, n2);
    if (max_n == 0) return 0.0;
    const std::size_t disjoint_count = max_n - matching_count;
    const double avg_weight_diff =
        (matching_count > 0) ? (weight_diff_sum / static_cast<double>(matching_count)) : 1.0;
    return (c1 * static_cast<double>(disjoint_count) / static_cast<double>(max_n))
         + (c2 * avg_weight_diff);
}

}  // namespace detail

// CompactSoAGenome: cell-aligned loci (op_types / gains).
inline double compatibility_distance(const CompactSoAGenome& g1, const CompactSoAGenome& g2,
                                     double c1 = kCompatC1, double c2 = kCompatC2) {
    const std::size_t n1 = g1.num_cells;
    const std::size_t n2 = g2.num_cells;
    const std::size_t min_n = std::min(n1, n2);
    double weight_diff_sum = 0.0;
    std::size_t matching_count = 0;
    for (std::size_t i = 0; i < min_n; ++i) {
        if (g1.op_types[i] == g2.op_types[i]) {
            weight_diff_sum += std::abs(static_cast<double>(g1.gains[i]) -
                                        static_cast<double>(g2.gains[i]));
            ++matching_count;
        }
    }
    return detail::neat_ish_distance(n1, n2, matching_count, weight_diff_sum, c1, c2);
}

// CorticalMacroArray: flatten micro-column genomes in column order.
inline double compatibility_distance(const CorticalMacroArray& a, const CorticalMacroArray& b,
                                     double c1 = kCompatC1, double c2 = kCompatC2) {
    const auto& ac = a.columns();
    const auto& bc = b.columns();
    std::size_t n1 = 0, n2 = 0;
    for (const auto& col : ac) n1 += col.genome.num_cells;
    for (const auto& col : bc) n2 += col.genome.num_cells;
    if (std::max(n1, n2) == 0) return 0.0;

    double weight_diff_sum = 0.0;
    std::size_t matching_count = 0;
    // Align by walking both column ranges in lockstep cell order.
    std::size_t ai = 0, bi = 0, a_cell = 0, b_cell = 0;
    std::size_t compared = 0;
    const std::size_t min_n = std::min(n1, n2);
    while (compared < min_n && ai < ac.size() && bi < bc.size()) {
        const auto& ga = ac[ai].genome;
        const auto& gb = bc[bi].genome;
        if (a_cell >= ga.num_cells) { ++ai; a_cell = 0; continue; }
        if (b_cell >= gb.num_cells) { ++bi; b_cell = 0; continue; }
        if (ga.op_types[a_cell] == gb.op_types[b_cell]) {
            weight_diff_sum += std::abs(static_cast<double>(ga.gains[a_cell]) -
                                        static_cast<double>(gb.gains[b_cell]));
            ++matching_count;
        }
        ++a_cell;
        ++b_cell;
        ++compared;
    }
    return detail::neat_ish_distance(n1, n2, matching_count, weight_diff_sum, c1, c2);
}

// CellularOrganism: cell-aligned loci (CellType / param1).
inline double compatibility_distance(const CellularOrganism& a, const CellularOrganism& b,
                                     double c1 = kCompatC1, double c2 = kCompatC2) {
    const std::size_t n1 = a.cells.size();
    const std::size_t n2 = b.cells.size();
    const std::size_t min_n = std::min(n1, n2);
    double weight_diff_sum = 0.0;
    std::size_t matching_count = 0;
    for (std::size_t i = 0; i < min_n; ++i) {
        if (a.cells[i].type == b.cells[i].type) {
            weight_diff_sum += std::abs(a.cells[i].param1 - b.cells[i].param1);
            ++matching_count;
        }
    }
    return detail::neat_ish_distance(n1, n2, matching_count, weight_diff_sum, c1, c2);
}

// Gate helper: true iff distance < threshold (compatible for sexual mate).
template <typename Individual>
inline bool mates_compatible(const Individual& a, const Individual& b,
                             float threshold = kDefaultSpeciationThreshold) {
    if constexpr (requires(const Individual& x, const Individual& y) {
                      compatibility_distance(x, y);
                  }) {
        return compatibility_distance(a, b) < static_cast<double>(threshold);
    } else {
        return true;  // no Prefer-A distance → allow sexual attempt
    }
}

}  // namespace population
}  // namespace kun
