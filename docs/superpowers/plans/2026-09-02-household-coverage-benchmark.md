# Household Coverage Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, simulation-only household coverage benchmark for testing navigation, obstacle recovery, dock return, and interrupted cleaning continuity.

**Architecture:** Implement the environment as a header-only KunCellular task alongside the maze environment. It will use a seeded occupancy grid with static furniture and optional dynamic obstacles, keep an explicit cleaning coverage map and battery state, and expose metrics that can be evaluated against a deterministic lawnmower baseline. This is a benchmark, not a physical-robot safety or product-performance claim.

**Tech Stack:** C++20, header-only KunCellular, CTest, standard library.

---

### Task 1: Define deterministic household state and sensing

**Files:**
- Create: `include/kun/cellular/household_coverage.hpp`
- Test: `tests/test_flow_household_coverage.cpp`

- [ ] **Step 1: Write failing geometry and reset tests**

```cpp
HouseholdCoverageEnvironment env(24, 16, 41);
env.reset(41);
assert(env.is_cleanable(env.robot().x, env.robot().y));
assert(env.coverage_ratio() == 0.0);
assert(env.battery_ratio() == 1.0);
assert(env.has_dock());
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cmake --build build -j2 && ctest --test-dir build -R household_coverage --output-on-failure`

Expected: the test target cannot compile because `HouseholdCoverageEnvironment` does not exist.

- [ ] **Step 3: Implement deterministic room generation**

```cpp
class HouseholdCoverageEnvironment {
public:
    HouseholdCoverageEnvironment(int width, int height, uint32_t seed);
    void reset(uint32_t seed);
    bool is_cleanable(int x, int y) const;
    double coverage_ratio() const;
    double battery_ratio() const;
    bool has_dock() const { return true; }
};
```

Use a bounded grid, fixed dock location, seeded furniture placement, and a separate cleanable-cell count. Reject invalid dimensions smaller than `8 x 8`.

- [ ] **Step 4: Run the geometry tests**

Run: `cmake --build build -j2 && ctest --test-dir build -R household_coverage --output-on-failure`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/kun/cellular/household_coverage.hpp tests/test_flow_household_coverage.cpp
git commit -m "feat: add deterministic household coverage environment"
```

### Task 2: Add cleaning, collision, battery, and interruption behavior

**Files:**
- Modify: `include/kun/cellular/household_coverage.hpp`
- Modify: `tests/test_flow_household_coverage.cpp`

- [ ] **Step 1: Write failing behavior tests**

```cpp
env.step(HouseholdCoverageEnvironment::Action::FORWARD);
assert(env.cleaned_cells() > 0);
env.inject_dynamic_obstacle_ahead();
env.step(HouseholdCoverageEnvironment::Action::FORWARD);
assert(env.collision_count() == 1);
env.interrupt_cleaning();
assert(env.is_interrupted());
env.resume_cleaning();
assert(!env.is_interrupted());
```

- [ ] **Step 2: Run and verify the behavior tests fail**

Run: `cmake --build build -j2 && ctest --test-dir build -R household_coverage --output-on-failure`

Expected: compile failure because action, obstacle, and interruption APIs do not exist.

- [ ] **Step 3: Implement bounded actions and accounting**

```cpp
enum class Action { FORWARD, TURN_LEFT, TURN_RIGHT, RETURN_TO_DOCK, WAIT };
bool step(Action action);
void inject_dynamic_obstacle_ahead();
void interrupt_cleaning();
void resume_cleaning();
```

Each successful move marks its cleanable cell once. Every action drains battery deterministically; collisions leave the robot in place and increment a counter. An interruption freezes movement without erasing coverage or battery state.

- [ ] **Step 4: Run behavior tests**

Run: `cmake --build build -j2 && ctest --test-dir build -R household_coverage --output-on-failure`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add include/kun/cellular/household_coverage.hpp tests/test_flow_household_coverage.cpp
git commit -m "feat: model coverage interruptions and obstacle recovery"
```

### Task 3: Add a deterministic baseline and evaluation report

**Files:**
- Modify: `include/kun/cellular/household_coverage.hpp`
- Modify: `tests/test_flow_household_coverage.cpp`

- [ ] **Step 1: Write failing baseline tests**

```cpp
auto report = HouseholdCoverageEvaluator::run_baseline(24, 16, 41, 2000);
assert(report.coverage_ratio >= 0.70);
assert(report.collisions == 0);
assert(report.returned_to_dock);
assert(report.to_json().find("\"simulation_only\": true") != std::string::npos);
```

- [ ] **Step 2: Run and verify it fails**

Run: `cmake --build build -j2 && ctest --test-dir build -R household_coverage --output-on-failure`

Expected: compile failure because `HouseholdCoverageEvaluator` does not exist.

- [ ] **Step 3: Implement the baseline and report**

```cpp
struct HouseholdCoverageReport {
    double coverage_ratio;
    size_t cleaned_cells;
    size_t cleanable_cells;
    int collisions;
    double energy_used;
    bool returned_to_dock;
    bool simulation_only{true};
    std::string to_json() const;
};
```

The baseline must use deterministic wall-following or boustrophedon traversal, recover from blocked motion without collision, return to the dock before battery depletion, and serialize the explicit simulation-only boundary.

- [ ] **Step 4: Run the full KunCellular test suite**

Run: `cmake --build build -j2 && ctest --test-dir build --output-on-failure`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/kun/cellular/household_coverage.hpp tests/test_flow_household_coverage.cpp
git commit -m "feat: add household coverage benchmark baseline"
```
