// ============================================================================
// test_population_openness_protect_live.cpp — A1 SpeciesNiche protect_window
// LIVE deme / ecology Prefer-A path (not bare PopulationSpeciesNiche unit).
// Assert: freshly speciated niche below fitness_floor SURVIVES purge in window;
// after protect_window generations, purge_unprotected removes + deme consequence.
// ============================================================================
#include "kun/cellular/population/deme.hpp"
#include "kun/cellular/population/ecology_grid.hpp"

#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using kun::population::Deme;
using kun::population::EcologyGrid;
using kun::population::EvolutionParams;

namespace {

struct Ind {
    int tag = 0;
    void mutate(float, float, std::mt19937&) {}
};

void test_live_deme_protect_survives_then_purges() {
    constexpr size_t POP = 4;
    constexpr std::uint32_t PROTECT = 3;
    constexpr double FLOOR = 0.5;

    Deme<Ind> d(0, POP, 42);
    d.seed_population(POP, [](std::mt19937&) { return Ind{}; });
    d.individuals()[0].tag = 100;
    d.individuals()[1].tag = 101;
    d.individuals()[2].tag = 102;
    d.individuals()[3].tag = 999;  // weak new-species member
    d.set_fitness(0, 1.0);
    d.set_fitness(1, 0.9);
    d.set_fitness(2, 0.8);
    d.set_fitness(3, 0.1);  // below FLOOR

    auto& reg = d.niche_registry();
    reg.place_or_speciate(0, 1.0, 0.0, true);   // niche 0
    reg.place_or_speciate(1, 0.9, 0.0, false);
    reg.place_or_speciate(2, 0.8, 0.0, false);
    reg.place_or_speciate(3, 0.1, 1.0, true);    // niche 1 (new species)
    assert(reg.niches().size() == 2);
    reg.niches()[1].protect_window = PROTECT;
    assert(reg.niches()[1].age_generations == 0);
    assert(reg.niches()[1].in_protect_window());
    assert(reg.niches()[1].members.size() == 1);
    assert(reg.niches()[1].members[0].individual_id == 3);

    d.enable_niche_protect_live(true);

    // Inside window: niche_protect_step ages then purges; weak must survive.
    // After k steps age==k; protected while age < PROTECT.
    for (std::uint32_t step = 0; step + 1 < PROTECT; ++step) {
        const auto applied = d.niche_protect_step(FLOOR, /*elite_keep=*/1);
        assert(applied == 0);
        assert(reg.niches()[1].in_protect_window());
        assert(reg.niches()[1].members.size() == 1);
        assert(reg.niches()[1].members[0].individual_id == 3);
        assert(d.individuals()[3].tag == 999);
        assert(d.fitness()[3] == 0.1);
        assert(d.size() == POP);
    }
    assert(reg.niches()[1].age_generations + 1 == PROTECT);

    // Next step: age reaches PROTECT → unprotected purge + deme consequence.
    const auto applied = d.niche_protect_step(FLOOR, /*elite_keep=*/1);
    assert(applied == 1);
    assert(!reg.niches()[1].in_protect_window());
    for (const auto& m : reg.niches()[1].members) {
        assert(m.individual_id != 3);
    }
    assert(d.individuals()[3].tag == 100);   // elite clone
    assert(d.fitness()[3] <= -1e17);         // pending-eval
    assert(d.size() == POP);

    std::cout << "  ✓ live deme: survive purge in protect_window, then deme consequence"
              << std::endl;
}

void test_live_ecology_grid_protect_path() {
    constexpr size_t POP = 4;
    constexpr std::uint32_t PROTECT = 3;
    constexpr double FLOOR = 0.5;

    EcologyGrid<Ind> grid(1, POP, 7, {.parallel = false});
    grid.demes()[0].seed_population(POP, [](std::mt19937&) { return Ind{}; });
    auto& d = grid.demes()[0];
    d.individuals()[0].tag = 10;
    d.individuals()[1].tag = 11;
    d.individuals()[2].tag = 12;
    d.individuals()[3].tag = 77;
    d.set_fitness(0, 1.0);
    d.set_fitness(1, 0.95);
    d.set_fitness(2, 0.9);
    d.set_fitness(3, 0.05);

    auto& reg = d.niche_registry();
    reg.place_or_speciate(0, 1.0, 0.0, true);
    reg.place_or_speciate(1, 0.95, 0.0, false);
    reg.place_or_speciate(2, 0.9, 0.0, false);
    reg.place_or_speciate(3, 0.05, 1.0, true);
    reg.niches()[1].protect_window = PROTECT;

    grid.enable_niche_protect(FLOOR, /*elite_keep=*/1);
    assert(grid.niche_protect_enabled());
    assert(d.niche_protect_live());

    EvolutionParams params;
    params.elite_keep = 1;
    params.crossover_prob = 0.0;  // keep asexual for determinism
    params.migration_rate = 0.0;

    // Eval is identity on Ind (fitness already set); pending only if sentinel.
    auto eval = [](Ind& ind) -> double {
        // Keep tag-encoded fitness for non-pending: use tag as proxy only if needed.
        // Prefer: leave already-set fitness; for pending slots return 0.5.
        (void)ind;
        return 0.5;
    };

    // First step_generation: evaluate (no pending) → niche_protect_step → evolve.
    // During first PROTECT-1 steps, weak niche must still contain id 3 before evolve
    // ages it. We inspect niches after protect by calling protect in isolation for
    // the survive window, then one grid step past the window.

    // Survive window via deme path already covered; here prove EcologyGrid invokes it.
    // Age niches to PROTECT-1 without purge consequence, then one grid step.
    for (std::uint32_t s = 0; s + 1 < PROTECT; ++s) {
        assert(d.niche_protect_step(FLOOR, 1) == 0);
        assert(d.individuals()[3].tag == 77);
    }
    // Re-set fitness that evolve may have... we haven't evolved yet.
    // One EcologyGrid step: evaluate → niche_protect (age->PROTECT, purge) → evolve.
    // Re-place member fitness; slot 3 still tag 77 / fitness 0.05.
    d.set_fitness(3, 0.05);
    // Ensure niche still has member 3 (prior steps only aged; didn't remove).
    assert(reg.niches()[1].members.size() == 1);

    grid.step_generation(eval, params);

    // After protect inside step_generation, slot 3 should have been purged/replaced
    // before evolve. Evolve may further rewrite slots; pop size conserved.
    assert(d.size() == POP);
    // Niche member 3 must be gone (purge happened on live ecology path).
    for (const auto& m : reg.niches()[1].members) {
        assert(m.individual_id != 3);
    }
    // Tag 77 should no longer sit in slot 3 as original weak individual after purge
    // consequence (elite clone) and subsequent evolve — at minimum, not both
    // tag==77 AND fitness==0.05 as the protected-window survivor state.
    // Stronger check: niche no longer lists id 3 (above). Soft deme check:
    bool weak_still_unpurged = (d.individuals()[3].tag == 77 && d.fitness()[3] == 0.05);
    assert(!weak_still_unpurged);

    std::cout << "  ✓ live ecology_grid: step_generation runs niche_protect purge path"
              << std::endl;
}

}  // namespace

int main() {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  A1 SpeciesNiche protect_window LIVE deme/ecology path" << std::endl;
    std::cout << "==================================================================" << std::endl;

    test_live_deme_protect_survives_then_purges();
    test_live_ecology_grid_protect_path();

    std::cout << "[PASS] test_population_openness_protect_live all assertions passed!"
              << std::endl;
    return 0;
}
