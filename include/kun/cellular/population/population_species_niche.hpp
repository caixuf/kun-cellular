#pragma once
// ============================================================================
// population/population_species_niche.hpp — Prefer-A species niche + protect_window
//
// NAME DISAMBIGUATION (Phase 3 landmine):
//   PopulationSpeciesNiche  = this L1 Prefer-A niche struct (NEAT-style protect)
//   kun::SpeciesNiche (ecosystem_biosphere.hpp) = biosphere enum PRODUCER/HERBIVORE/...
//   kun::SpeciesNiche (morphogenetic_population.hpp) = heap morphogenetic struct (L0-ish)
// Prefer-A code MUST use PopulationSpeciesNiche / namespace kun::population.
//
// Mechanism #4: protect_window generations protected from fitness purge.
// ============================================================================
#include "kun/cellular/population/compatibility_distance.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kun {
namespace population {

struct PopulationNicheMember {
    std::uint32_t individual_id{0};
    double fitness{0.0};
    double adjusted_fitness{0.0};
    bool protected_from_purge{false};
};

// Prefer-A L1 species niche (NOT biosphere enum SpeciesNiche).
struct PopulationSpeciesNiche {
    std::uint32_t species_id{0};
    std::uint32_t protect_window{5};          // generations protected from purge
    std::uint32_t generations_since_improve{0};
    std::uint32_t age_generations{0};
    double max_fitness{0.0};
    double representative_compat_anchor{0.0}; // optional distance anchor
    bool extinct{false};
    std::vector<PopulationNicheMember> members;

    bool in_protect_window() const {
        return age_generations < protect_window;
    }

    void calculate_shared_fitness() {
        if (members.empty()) return;
        const double sz = static_cast<double>(members.size());
        for (auto& m : members) {
            m.adjusted_fitness = m.fitness / sz;
            m.protected_from_purge = in_protect_window();
        }
    }

    void note_generation(double best_fitness_this_gen) {
        ++age_generations;
        if (best_fitness_this_gen > max_fitness) {
            max_fitness = best_fitness_this_gen;
            generations_since_improve = 0;
        } else {
            ++generations_since_improve;
        }
        calculate_shared_fitness();
    }

    // Purge members below fitness_floor UNLESS niche is inside protect_window
    // or member flagged protected. Returns number removed.
    std::size_t purge_unprotected(double fitness_floor) {
        if (in_protect_window()) {
            // Entire niche protected — mark members, remove none.
            for (auto& m : members) m.protected_from_purge = true;
            return 0;
        }
        const auto before = members.size();
        members.erase(std::remove_if(members.begin(), members.end(),
            [fitness_floor](const PopulationNicheMember& m) {
                if (m.protected_from_purge) return false;
                return m.fitness < fitness_floor;
            }), members.end());
        return before - members.size();
    }
};

// Simple Prefer-A niche registry (fixed small pop; no 4th genome ABI).
class PopulationNicheRegistry {
public:
    explicit PopulationNicheRegistry(float speciation_threshold = kDefaultSpeciationThreshold,
                                     std::uint32_t default_protect_window = 5)
        : speciation_threshold_(speciation_threshold),
          default_protect_window_(default_protect_window) {}

    PopulationSpeciesNiche& spawn_niche(std::uint32_t species_id) {
        PopulationSpeciesNiche n;
        n.species_id = species_id;
        n.protect_window = default_protect_window_;
        niches_.push_back(n);
        return niches_.back();
    }

    std::vector<PopulationSpeciesNiche>& niches() { return niches_; }
    const std::vector<PopulationSpeciesNiche>& niches() const { return niches_; }

    float speciation_threshold() const { return speciation_threshold_; }
    std::uint32_t default_protect_window() const { return default_protect_window_; }

    // Place individual into first niche whose representative distance < threshold,
    // else spawn. Distance is caller-supplied (Prefer-A adapters via compatibility_distance).
    void place_or_speciate(std::uint32_t individual_id, double fitness,
                           double distance_to_first_rep,
                           bool force_new_species) {
        if (!force_new_species && !niches_.empty() &&
            distance_to_first_rep < static_cast<double>(speciation_threshold_)) {
            PopulationNicheMember m;
            m.individual_id = individual_id;
            m.fitness = fitness;
            niches_.front().members.push_back(m);
            niches_.front().calculate_shared_fitness();
            return;
        }
        auto& n = spawn_niche(next_species_id_++);
        PopulationNicheMember m;
        m.individual_id = individual_id;
        m.fitness = fitness;
        n.members.push_back(m);
        n.calculate_shared_fitness();
    }

private:
    float speciation_threshold_{kDefaultSpeciationThreshold};
    std::uint32_t default_protect_window_{5};
    std::uint32_t next_species_id_{1};
    std::vector<PopulationSpeciesNiche> niches_;
};

}  // namespace population
}  // namespace kun
