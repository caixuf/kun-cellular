#include "kun/cellular/household_coverage.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <iostream>
#include <queue>
#include <string>
#include <stdexcept>
#include <vector>

using namespace kun;

void test_initial_household_coverage_state() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    env.reset(41);

    assert(env.is_cleanable(env.robot().x, env.robot().y));
    assert(env.coverage_ratio() == 0.0);
    assert(env.battery_ratio() == 1.0);
    assert(env.has_dock());
}

void test_reset_is_deterministic() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    bool first_layout[16][24]{};

    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 24; ++x) {
            first_layout[y][x] = env.is_cleanable(x, y);
        }
    }

    env.reset(41);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 24; ++x) {
            assert(env.is_cleanable(x, y) == first_layout[y][x]);
        }
    }
    assert(env.robot().x == 1);
    assert(env.robot().y == 1);
    assert(env.coverage_ratio() == 0.0);
    assert(env.battery_ratio() == 1.0);
}

void test_cleanable_cells_are_dock_reachable() {
    const int widths[] = {8, 12, 18, 26};
    const int heights[] = {8, 16, 32, 48};
    const uint32_t seeds[] = {0, 1, 24, 41};

    for (const int width : widths) {
        for (const int height : heights) {
            for (const uint32_t seed : seeds) {
                HouseholdCoverageEnvironment env(width, height, seed);
                const auto cell_index = [width](int x, int y) {
                    return static_cast<std::size_t>(y) *
                               static_cast<std::size_t>(width) +
                           static_cast<std::size_t>(x);
                };
                std::vector<bool> reachable(
                    static_cast<std::size_t>(width) *
                        static_cast<std::size_t>(height),
                    false);
                std::queue<std::pair<int, int>> pending;
                pending.emplace(env.dock_x(), env.dock_y());
                reachable[cell_index(env.dock_x(), env.dock_y())] = true;

                while (!pending.empty()) {
                    const auto [x, y] = pending.front();
                    pending.pop();
                    for (const auto& [dx, dy] :
                         {std::pair<int, int>{0, -1}, {1, 0}, {0, 1}, {-1, 0}}) {
                        const int next_x = x + dx;
                        const int next_y = y + dy;
                        if (next_x < 0 || next_y < 0 || next_x >= width ||
                            next_y >= height ||
                            !env.is_cleanable(next_x, next_y)) {
                            continue;
                        }
                        const auto next = cell_index(next_x, next_y);
                        if (reachable[next]) {
                            continue;
                        }
                        reachable[next] = true;
                        pending.emplace(next_x, next_y);
                    }
                }

                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        if (env.is_cleanable(x, y)) {
                            assert(reachable[cell_index(x, y)]);
                        }
                    }
                }
            }
        }
    }
}

void test_rejects_undersized_rooms() {
    bool width_rejected = false;
    try {
        HouseholdCoverageEnvironment env(7, 16, 41);
        (void)env;
    } catch (const std::invalid_argument&) {
        width_rejected = true;
    }
    assert(width_rejected);

    bool height_rejected = false;
    try {
        HouseholdCoverageEnvironment env(24, 7, 41);
        (void)env;
    } catch (const std::invalid_argument&) {
        height_rejected = true;
    }
    assert(height_rejected);
}

void test_rejects_oversized_rooms() {
    bool rejected = false;
    try {
        HouseholdCoverageEnvironment env(65536, 65536, 41);
        (void)env;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void test_marking_updates_coverage_once() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    const auto& dock = env.robot();

    assert(env.mark_cleaned(dock.x, dock.y));
    const double marked_coverage = env.coverage_ratio();
    assert(marked_coverage > 0.0);
    assert(!env.mark_cleaned(dock.x, dock.y));
    assert(env.coverage_ratio() == marked_coverage);
    assert(!env.mark_cleaned(0, 0));
    assert(!env.mark_cleaned(-1, 0));
}

void test_reset_clears_coverage() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    assert(env.mark_cleaned(env.robot().x, env.robot().y));
    assert(env.coverage_ratio() > 0.0);

    env.reset(41);
    assert(env.coverage_ratio() == 0.0);
}

void test_forward_cleans_destination_cell() {
    HouseholdCoverageEnvironment env(24, 16, 41);

    assert(env.heading() == HouseholdCoverageEnvironment::Heading::EAST);
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.robot_x() == 2);
    assert(env.robot_y() == 1);
    assert(env.coverage_ratio() > 0.0);
    assert(env.collision_count() == 0);
}

void test_dynamic_obstacle_blocks_forward_without_moving() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    const auto before = env.robot();
    const double battery_before = env.battery_ratio();

    env.inject_dynamic_obstacle_ahead();
    assert(!env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.robot_x() == before.x);
    assert(env.robot_y() == before.y);
    assert(env.coverage_ratio() == 0.0);
    assert(env.collision_count() == 1);
    assert(env.battery_ratio() < battery_before);
}

void test_interruption_preserves_state_until_resume() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    const auto before_interrupt = env.robot();
    const double coverage_before = env.coverage_ratio();
    const auto collisions_before = env.collision_count();
    const double battery_before = env.battery_ratio();

    env.interrupt_cleaning();
    assert(env.is_interrupted());
    assert(!env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.robot_x() == before_interrupt.x);
    assert(env.robot_y() == before_interrupt.y);
    assert(env.coverage_ratio() == coverage_before);
    assert(env.collision_count() == collisions_before);
    assert(env.battery_ratio() < battery_before);

    env.resume_cleaning();
    assert(!env.is_interrupted());
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.robot_x() == before_interrupt.x + 1);
    assert(env.robot_y() == before_interrupt.y);
    assert(env.coverage_ratio() > coverage_before);
    assert(env.collision_count() == collisions_before);
}

void test_every_action_consumes_battery() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    const double initial_battery = env.battery_ratio();

    assert(env.step(HouseholdCoverageEnvironment::Action::WAIT));
    assert(env.battery_ratio() < initial_battery);
    const double after_wait = env.battery_ratio();

    assert(env.step(HouseholdCoverageEnvironment::Action::TURN_LEFT));
    assert(env.battery_ratio() < after_wait);
    const double after_turn = env.battery_ratio();

    env.inject_dynamic_obstacle_ahead();
    assert(!env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.battery_ratio() < after_turn);
}

void test_return_to_dock_reaches_dock_one_step_at_a_time() {
    HouseholdCoverageEnvironment env(24, 16, 41);
    assert(env.at_dock());
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(env.step(HouseholdCoverageEnvironment::Action::FORWARD));
    assert(!env.at_dock());

    const auto collisions_before = env.collision_count();
    int return_actions = 0;
    while (!env.at_dock() && return_actions < 32) {
        assert(env.step(HouseholdCoverageEnvironment::Action::RETURN_TO_DOCK));
        ++return_actions;
    }

    assert(env.at_dock());
    assert(env.robot_x() == 1);
    assert(env.robot_y() == 1);
    assert(return_actions == 3);
    assert(env.collision_count() == collisions_before);
}

void test_baseline_is_deterministic_and_honest() {
    const auto first =
        HouseholdCoverageEvaluator::run_baseline(24, 16, 41, 2000);
    const auto second =
        HouseholdCoverageEvaluator::run_baseline(24, 16, 41, 2000);

    assert(first.coverage_ratio == second.coverage_ratio);
    assert(first.cleaned_cells == second.cleaned_cells);
    assert(first.cleanable_cells == second.cleanable_cells);
    assert(first.collisions == second.collisions);
    assert(first.energy_used == second.energy_used);
    assert(first.final_battery_ratio == second.final_battery_ratio);
    assert(first.returned_to_dock == second.returned_to_dock);
    assert(std::abs(first.energy_used -
                    (1.0 - first.final_battery_ratio)) < 1e-12);
    assert(first.coverage_ratio >= 0.70);
    assert(first.collisions == 0);
    assert(first.returned_to_dock);
    assert(first.cleaned_cells <= first.cleanable_cells);
    assert(first.cleanable_cells > 0);
    assert(first.coverage_ratio ==
           static_cast<double>(first.cleaned_cells) /
               static_cast<double>(first.cleanable_cells));
    static_assert(HouseholdCoverageReport::kSimulationOnly);
    assert(HouseholdCoverageReport::kSimulationOnly);

    const std::string json = first.to_json();
    assert(json.find("\"simulation_only\":true") != std::string::npos);
    assert(json.find("\"returned_to_dock\":true") != std::string::npos);
    assert(json.find("\"final_battery_ratio\":") != std::string::npos);
}

void test_baseline_rejects_non_positive_step_budget() {
    bool rejected = false;
    try {
        (void)HouseholdCoverageEvaluator::run_baseline(24, 16, 41, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

int main() {
    test_initial_household_coverage_state();
    test_reset_is_deterministic();
    test_cleanable_cells_are_dock_reachable();
    test_rejects_undersized_rooms();
    test_rejects_oversized_rooms();
    test_marking_updates_coverage_once();
    test_reset_clears_coverage();
    test_forward_cleans_destination_cell();
    test_dynamic_obstacle_blocks_forward_without_moving();
    test_interruption_preserves_state_until_resume();
    test_every_action_consumes_battery();
    test_return_to_dock_reaches_dock_one_step_at_a_time();
    test_baseline_is_deterministic_and_honest();
    test_baseline_rejects_non_positive_step_budget();

    std::cout << "Household coverage environment tests passed.\n";
    return 0;
}
