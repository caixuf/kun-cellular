#pragma once
// ============================================================================
// population/novelty_stepping_stone.hpp — Prefer-A novelty stepping-stones + driver modes
//
// Mid-goal individuals are NOT discarded solely for low task fitness.
// KNN novelty archive is kept/wrapped as a *feed* into MAP-Elites (not the grid itself).
// Driver modes mirror L0 FitnessDriverMode: TASK_FITNESS_ONLY / NOVELTY_SEARCH / HYBRID_CURIOSITY.
// ============================================================================
#include "kun/cellular/population/map_elites_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kun {
namespace population {

enum class PreferAFitnessDriver : std::uint8_t {
    TASK_FITNESS_ONLY = 0,
    NOVELTY_SEARCH = 1,
    HYBRID_CURIOSITY = 2
};

// Fixed-capacity KNN novelty archive (Prefer-A L1 wrap; may feed MAP-Elites BD novelty axis).
class NoveltySteppingStoneArchive {
public:
    explicit NoveltySteppingStoneArchive(std::size_t max_size = 64, std::size_t k = 5)
        : max_size_(std::max<std::size_t>(1, max_size)),
          k_neighbors_(std::max<std::size_t>(1, k)) {
        archive_.reserve(max_size_);
    }

    std::size_t size() const { return archive_.size(); }
    std::size_t capacity() const { return archive_.capacity(); }
    std::size_t max_size() const { return max_size_; }

    double compute_novelty(const std::vector<double>& behavior) const {
        if (archive_.empty() || behavior.empty()) return 1.0;
        std::vector<double> distances;
        distances.reserve(archive_.size());
        for (const auto& a : archive_) {
            double d2 = 0.0;
            const std::size_t dim = std::min(behavior.size(), a.size());
            for (std::size_t i = 0; i < dim; ++i) {
                const double diff = behavior[i] - a[i];
                d2 += diff * diff;
            }
            distances.push_back(std::sqrt(d2));
        }
        std::sort(distances.begin(), distances.end());
        const std::size_t kk = std::min(k_neighbors_, distances.size());
        double sum = 0.0;
        for (std::size_t i = 0; i < kk; ++i) sum += distances[i];
        return sum / static_cast<double>(kk);
    }

    void add_behavior(const std::vector<double>& b) {
        if (b.empty()) return;
        if (archive_.size() < max_size_) {
            archive_.push_back(b);
            return;
        }
        // Ring: drop oldest (capacity stable after fill).
        archive_.erase(archive_.begin());
        archive_.push_back(b);
    }

    // Combined driver score. Mid-goal: NOVELTY_SEARCH ignores task fitness for ranking.
    static double driver_score(PreferAFitnessDriver mode,
                               double task_fitness,
                               double novelty,
                               double novelty_weight = 0.3) {
        switch (mode) {
            case PreferAFitnessDriver::NOVELTY_SEARCH:
                return novelty;
            case PreferAFitnessDriver::HYBRID_CURIOSITY:
                return task_fitness + novelty_weight * novelty;
            case PreferAFitnessDriver::TASK_FITNESS_ONLY:
            default:
                return task_fitness;
        }
    }

    // Purge policy: under novelty / hybrid modes, do NOT purge solely on low task fitness.
    // Returns true if individual should be kept as a stepping-stone.
    static bool keep_as_stepping_stone(PreferAFitnessDriver mode,
                                       double task_fitness,
                                       double novelty,
                                       double fitness_floor,
                                       double novelty_floor) {
        if (mode == PreferAFitnessDriver::TASK_FITNESS_ONLY) {
            return task_fitness >= fitness_floor;
        }
        // Mid-goal: novelty above floor keeps individual even if task fitness is low.
        if (novelty >= novelty_floor) return true;
        return task_fitness >= fitness_floor;
    }

    // Feed MAP-Elites: insert using novelty from this archive + coverage as BD.
    MapElitesInsertResult feed_map_elites(MapElitesGrid& grid,
                                          std::uint32_t individual_id,
                                          float coverage,
                                          const std::vector<double>& behavior) {
        const float nov = static_cast<float>(compute_novelty(behavior));
        BehaviorDescriptor bd;
        bd.dims = 2;
        bd[0] = std::clamp(coverage, 0.0f, 1.0f);
        bd[1] = std::clamp(nov, 0.0f, 1.0f);  // novelty axis (caller may rescale bounds)
        auto r = grid.try_insert(individual_id, bd, coverage, nov);
        add_behavior(behavior);
        return r;
    }

private:
    std::size_t max_size_{64};
    std::size_t k_neighbors_{5};
    std::vector<std::vector<double>> archive_;
};

}  // namespace population
}  // namespace kun
