#pragma once
// ============================================================================
// population/map_elites_grid.hpp — Prefer-A L1 MAP-Elites descriptor grid
//
// Phase 3 §七: novelty/openness (NOT DouZero strength).
// L1 Prefer-A genomes only (CompactSoA / Cortical / 82-Organism). No 4th ABI.
//
// KNN NoveltyArchive (L0 heap MorphogeneticEvolutionEngine) is NOT this grid.
// This is a true MAP-Elites cell archive keyed by behavioral descriptor (BD) bins;
// elite per cell is replaced by QD objective (coverage/novelty hybrid).
//
// Prefer-A discipline: fixed-capacity grid (no unbounded heap growth on hot path),
// deterministic insert/replace, 82-cell / zero-alloc / det held for Prefer-A line.
// ============================================================================
#include "kun/cellular/population/compatibility_distance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kun {
namespace population {

inline constexpr std::size_t kMapElitesMaxBdDims = 4;
inline constexpr std::size_t kMapElitesMaxBinsPerDim = 16;
inline constexpr std::size_t kMapElitesMaxCells =
    kMapElitesMaxBinsPerDim * kMapElitesMaxBinsPerDim;  // 2D default capacity 256

// Behavioral descriptor: fixed-dim Prefer-A BD (coverage frac, novelty, etc.).
struct BehaviorDescriptor {
    std::array<float, kMapElitesMaxBdDims> v{};
    std::uint8_t dims{2};

    float operator[](std::size_t i) const { return v[i]; }
    float& operator[](std::size_t i) { return v[i]; }
};

// Elite payload stored in a grid cell (index into caller archive + QD score).
struct MapElitesElite {
    std::uint32_t individual_id{0};
    float qd_objective{0.0f};   // hybrid: coverage + novelty_weight * novelty
    float coverage{0.0f};
    float novelty{0.0f};
    BehaviorDescriptor bd{};
    bool occupied{false};
};

struct MapElitesInsertResult {
    bool accepted{false};
    bool replaced{false};
    std::size_t cell_index{0};
    float previous_qd{0.0f};
};

// Fixed-capacity MAP-Elites grid. Default 2D: bins_x × bins_y cells.
// BD axes are linearly mapped into [0,1] then quantized to bins.
class MapElitesGrid {
public:
    explicit MapElitesGrid(std::uint8_t dims = 2,
                           std::uint8_t bins_per_dim = 8,
                           float novelty_weight = 0.3f)
        : dims_(std::min<std::uint8_t>(dims, static_cast<std::uint8_t>(kMapElitesMaxBdDims))),
          bins_per_dim_(std::clamp<std::uint8_t>(bins_per_dim, 1,
                          static_cast<std::uint8_t>(kMapElitesMaxBinsPerDim))),
          novelty_weight_(novelty_weight) {
        cell_count_ = 1;
        for (std::uint8_t d = 0; d < dims_; ++d) {
            cell_count_ *= bins_per_dim_;
        }
        if (cell_count_ > kMapElitesMaxCells) {
            // Prefer-A: clamp to fixed max; reduce bins if needed.
            bins_per_dim_ = static_cast<std::uint8_t>(
                std::max(1.0, std::floor(std::pow(static_cast<double>(kMapElitesMaxCells),
                                                  1.0 / static_cast<double>(dims_)))));
            cell_count_ = 1;
            for (std::uint8_t d = 0; d < dims_; ++d) cell_count_ *= bins_per_dim_;
        }
        cells_.assign(cell_count_, MapElitesElite{});
        for (std::uint8_t d = 0; d < dims_; ++d) {
            bd_lo_[d] = 0.0f;
            bd_hi_[d] = 1.0f;
        }
    }

    void set_bd_bounds(std::uint8_t dim, float lo, float hi) {
        if (dim >= dims_) return;
        bd_lo_[dim] = lo;
        bd_hi_[dim] = (hi > lo) ? hi : (lo + 1e-6f);
    }

    std::size_t cell_count() const { return cell_count_; }
    std::uint8_t dims() const { return dims_; }
    std::uint8_t bins_per_dim() const { return bins_per_dim_; }
    float novelty_weight() const { return novelty_weight_; }

    std::size_t occupied_count() const {
        std::size_t n = 0;
        for (const auto& c : cells_) if (c.occupied) ++n;
        return n;
    }

    float coverage_ratio() const {
        if (cell_count_ == 0) return 0.0f;
        return static_cast<float>(occupied_count()) / static_cast<float>(cell_count_);
    }

    // Quantize BD → linear cell index. Out-of-range BD clamped to edge bins.
    std::size_t cell_index_of(const BehaviorDescriptor& bd) const {
        std::size_t idx = 0;
        std::size_t stride = 1;
        const std::uint8_t use_dims = std::min(bd.dims, dims_);
        for (std::uint8_t d = 0; d < use_dims; ++d) {
            const float span = bd_hi_[d] - bd_lo_[d];
            float t = (span > 0.0f) ? ((bd[d] - bd_lo_[d]) / span) : 0.0f;
            t = std::clamp(t, 0.0f, 0.999999f);
            const std::size_t bin = static_cast<std::size_t>(t * bins_per_dim_);
            idx += bin * stride;
            stride *= bins_per_dim_;
        }
        return std::min(idx, cell_count_ - 1);
    }

    static float qd_score(float coverage, float novelty, float novelty_weight) {
        return coverage + novelty_weight * novelty;
    }

    MapElitesInsertResult try_insert(std::uint32_t individual_id,
                                     const BehaviorDescriptor& bd,
                                     float coverage,
                                     float novelty) {
        MapElitesInsertResult r;
        r.cell_index = cell_index_of(bd);
        const float qd = qd_score(coverage, novelty, novelty_weight_);
        auto& cell = cells_[r.cell_index];
        if (!cell.occupied) {
            cell.occupied = true;
            cell.individual_id = individual_id;
            cell.qd_objective = qd;
            cell.coverage = coverage;
            cell.novelty = novelty;
            cell.bd = bd;
            cell.bd.dims = dims_;
            r.accepted = true;
            r.replaced = false;
            return r;
        }
        r.previous_qd = cell.qd_objective;
        if (qd > cell.qd_objective) {
            cell.individual_id = individual_id;
            cell.qd_objective = qd;
            cell.coverage = coverage;
            cell.novelty = novelty;
            cell.bd = bd;
            cell.bd.dims = dims_;
            r.accepted = true;
            r.replaced = true;
        }
        return r;
    }

    const MapElitesElite& cell_at(std::size_t i) const { return cells_[i]; }
    const std::vector<MapElitesElite>& cells() const { return cells_; }

    // Clear occupancy (capacity retained — Prefer-A zero-alloc steady-state).
    void clear() {
        for (auto& c : cells_) {
            c = MapElitesElite{};
        }
    }

    std::size_t capacity_cells() const { return cells_.capacity(); }

private:
    std::uint8_t dims_{2};
    std::uint8_t bins_per_dim_{8};
    float novelty_weight_{0.3f};
    std::size_t cell_count_{0};
    std::array<float, kMapElitesMaxBdDims> bd_lo_{};
    std::array<float, kMapElitesMaxBdDims> bd_hi_{};
    std::vector<MapElitesElite> cells_;
};

}  // namespace population
}  // namespace kun
