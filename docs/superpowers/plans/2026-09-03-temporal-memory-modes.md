# Temporal Memory and Discrete Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic C++ evidence benchmark proving that real
`CellularOrganism::forward()` graphs can retain delayed input, implement
hysteretic discrete modes, and isolate episode state.

**Architecture:** Add one `tests/test_flow_temporal_memory_modes.cpp` executable
discovered by the existing CMake `tests/test_flow_*.cpp` glob. The executable
builds two small reference organisms from the existing public `Cell`,
`Synapse`, and `CellularOrganism` types, then evaluates them against
deterministic sequences with exact targets. The result is explicitly labeled
`REFERENCE_TOPOLOGY_EXPRESSIVITY`; it does not claim that evolution discovered
the graphs or that the system has general intelligence.

**Tech Stack:** C++20, existing header-only KunCellular core, CMake, CTest,
standard library only.

---

### Task 1: Add deterministic reference graphs and delayed recall evidence

**Files:**
- Create: `tests/test_flow_temporal_memory_modes.cpp`

- [x] **Step 1: Write the failing test skeleton**

Create the test file with the public core include, deterministic sequence
helpers, and a delayed-recall test entry point:

```cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "kun/cellular/cellular_genome.hpp"

using kun::Cell;
using kun::CellType;
using kun::CellularOrganism;
using kun::Synapse;

static CellularOrganism make_delay_organism(int delay_steps);
static std::vector<double> make_binary_sequence(uint32_t seed, size_t length);
static void test_delayed_recall();
static void require(bool condition, const char* message);

int main() {
    test_delayed_recall();
    return 0;
}
```

- [x] **Step 2: Configure and run to verify the new target is not available**

Run:

```bash
cmake -S . -B build
cmake --build build --target test_flow_temporal_memory_modes
```

Expected: the build reports that the target does not exist or that the
declared helper is not defined. This confirms the test is not accidentally
passing through an existing target.

- [x] **Step 3: Implement the reference graph and deterministic stream**

Use zero-initialized public structs so all runtime fields are known, set
`hebbian_rate` to zero for measurement, and compile before the first forward
step:

```cpp
static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(EXIT_FAILURE);
    }
}

static Cell make_cell(uint32_t id, CellType type, double param1 = 0.0,
                      double param2 = 0.0) {
    Cell cell{};
    cell.id = id;
    cell.type = type;
    cell.param1 = param1;
    cell.param2 = param2;
    return cell;
}

static Synapse make_synapse(uint32_t from, uint32_t to, uint8_t port = 0,
                            double weight = 1.0) {
    Synapse synapse{};
    synapse.from_cell_id = from;
    synapse.to_cell_id = to;
    synapse.to_port = port;
    synapse.weight = weight;
    synapse.initial_weight = weight;
    synapse.is_active = true;
    synapse.hebbian_rate = 0.0;
    synapse.hebbian_decay = 0.0;
    return synapse;
}

static CellularOrganism make_delay_organism(int delay_steps) {
    require(delay_steps >= 1 && delay_steps <= 16,
            "delay must be in the supported 1..16 range");
    CellularOrganism organism;
    organism.organism_id = static_cast<uint64_t>(100 + delay_steps);
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(
        make_cell(1, CellType::OP_DELAY_N,
                  static_cast<double>(delay_steps) / 16.0));
    organism.cells.push_back(
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "delay organism must compile");
    return organism;
}

static uint32_t next_state(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static std::vector<double> make_binary_sequence(uint32_t seed, size_t length) {
    uint32_t state = seed == 0 ? 1u : seed;
    std::vector<double> sequence;
    sequence.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        sequence.push_back((next_state(state) & 1u) != 0u ? 1.0 : 0.0);
    }
    return sequence;
}
```

For each delay in `{1, 4, 16}` and each seed `1..32`, reset state, run at
least 96 steps with `forward(input, false)`, compare
`positive_action` against `sequence[t-delay]` (or zero before history), and
run the same sequence again after reset. Require finite output, maximum error
at most `1e-12`, and identical traces between the two episodes:

```cpp
static void test_delayed_recall() {
    size_t cases = 0;
    for (int delay : {1, 4, 16}) {
        CellularOrganism organism = make_delay_organism(delay);
        for (uint32_t seed = 1; seed <= 32; ++seed) {
            const auto sequence = make_binary_sequence(seed, 96);
            std::vector<double> first_trace;
            first_trace.reserve(sequence.size());
            organism.reset_state(true);
            for (size_t t = 0; t < sequence.size(); ++t) {
                double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
                const auto actions = organism.forward(inputs, false);
                const double expected =
                    t >= static_cast<size_t>(delay)
                        ? sequence[t - static_cast<size_t>(delay)]
                        : 0.0;
                require(std::isfinite(actions.positive_action),
                        "delayed action output must be finite");
                require(std::abs(actions.positive_action - expected) <= 1e-12,
                        "delayed action must equal the k-step target");
                first_trace.push_back(actions.positive_action);
            }

            organism.reset_state(true);
            for (size_t t = 0; t < sequence.size(); ++t) {
                double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
                const auto actions = organism.forward(inputs, false);
                require(actions.positive_action == first_trace[t],
                        "reset must reproduce the delayed trace");
            }
            ++cases;
        }
    }
    std::cout << "DELAY_RECALL_CASES=" << cases << "\n";
    std::cout << "DELAY_RECALL=PASS\n";
}
```

- [x] **Step 4: Build and run the focused test**

Run:

```bash
cmake -S . -B build
cmake --build build --target test_flow_temporal_memory_modes
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_modes
```

Expected: this task's delayed recall cases pass and print
`DELAY_RECALL_CASES=96` and `DELAY_RECALL=PASS`.

- [x] **Step 5: Commit the delayed recall benchmark**

```bash
git add tests/test_flow_temporal_memory_modes.cpp
git commit -m "test: prove cellular delayed temporal recall"
```

### Task 2: Add hysteresis mode switching and structural isolation

**Files:**
- Modify: `tests/test_flow_temporal_memory_modes.cpp`

- [x] **Step 1: Add the hysteresis test before its implementation**

Add a test declaration and call it from `main`:

```cpp
static void test_hysteresis_modes();

int main() {
    test_delayed_recall();
    test_hysteresis_modes();
    return 0;
}
```

The initial build must fail because the helper and test body are not yet
defined:

```bash
cmake --build build --target test_flow_temporal_memory_modes
```

- [x] **Step 2: Implement a real hysteresis graph and a genome snapshot**

Construct `Sense_Input0 -> Gate_Hysteresis -> Act_PosAction`, with entry
threshold `+0.5` and exit threshold `-0.5`. Capture the genome-only JSON before
an episode and compare it after the episode; this excludes runtime state while
covering cell parameters, topology, and genetic weights:

```cpp
static CellularOrganism make_hysteresis_organism() {
    CellularOrganism organism;
    organism.organism_id = 200;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(
        make_cell(1, CellType::GATE_HYSTERESIS, 0.5, -0.5));
    organism.cells.push_back(
        make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "hysteresis organism must compile");
    return organism;
}

static void test_hysteresis_modes() {
    const std::vector<double> sequence = {
        0.0, 0.2, -0.2, 0.4,  0.75, 0.2, -0.2, 0.49,
        -0.49, -0.75, 0.1, -0.1, 0.49, -0.49, 0.8, 0.2,
        -0.2, 0.4, -0.8
    };
    const std::vector<bool> expected = {
        false, false, false, false, true,  true,  true,  true, true, false,
        false, false, false, false, true,  true,  true,  true, false
    };

    CellularOrganism organism = make_hysteresis_organism();
    const std::string genome_before = organism.export_genome_json();
    organism.reset_state(true);
    bool previous = false;
    size_t transitions = 0;
    size_t in_band_transitions = 0;
    for (size_t t = 0; t < sequence.size(); ++t) {
        double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
        const auto actions = organism.forward(inputs, false);
        const bool active = organism.cells[1].output_val > 0.0;
        require(std::isfinite(actions.positive_action),
                "hysteresis action output must be finite");
        require(active == expected[t], "hysteresis state mismatch");
        if (t != 0 && active != previous) {
            ++transitions;
            if (sequence[t] >= -0.5 && sequence[t] <= 0.5) {
                ++in_band_transitions;
            }
        }
        previous = active;
    }
    require(transitions == 4, "hysteresis must switch exactly four times");
    require(in_band_transitions == 0,
            "hysteresis must not switch inside the dead band");
    require(organism.export_genome_json() == genome_before,
            "hysteresis episode must not mutate the genome");

    organism.reset_state(true);
    double neutral_inputs[4] = {0.0, 0.0, 0.0, 0.0};
    organism.forward(neutral_inputs, false);
    require(organism.cells[1].output_val < 0.0,
            "reset must clear the hysteresis latch");

    std::cout << "HYSTERESIS_TRANSITIONS=" << transitions << "\n";
    std::cout << "HYSTERESIS_IN_BAND_TRANSITIONS=" << in_band_transitions
              << "\n";
    std::cout << "HYSTERESIS_MODES=PASS\n";
}
```

- [x] **Step 3: Repeat the hysteresis sequence over 32 deterministic offsets**

To ensure the result is not a single hand-picked trace, run this exact helper
for offsets from `-0.2` through `+0.2`. Only in-band values are changed and
they are clamped to `[-0.45, +0.45]`, so threshold-crossing events remain at
`0.75`, `-0.75`, `0.8`, and `-0.8`:

```cpp
static std::vector<double> hysteresis_sequence(double offset) {
    std::vector<double> base = {
        0.0, 0.2, -0.2, 0.4, 0.75, 0.2, -0.2, 0.49, -0.49, -0.75,
        0.1, -0.1, 0.49, -0.49, 0.8, 0.2, -0.2, 0.4, -0.8
    };
    for (double& value : base) {
        if (value >= -0.5 && value <= 0.5) {
            value = std::clamp(value + offset, -0.45, 0.45);
        }
    }
    return base;
}

static void run_hysteresis_case(double offset, size_t& transitions,
                                size_t& in_band_transitions) {
    CellularOrganism organism = make_hysteresis_organism();
    const std::string genome_before = organism.export_genome_json();
    organism.reset_state(true);
    bool expected_active = false;
    bool previous_active = false;
    const auto sequence = hysteresis_sequence(offset);
    for (size_t t = 0; t < sequence.size(); ++t) {
        if (sequence[t] > 0.5) expected_active = true;
        else if (sequence[t] < -0.5) expected_active = false;

        double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
        const auto actions = organism.forward(inputs, false);
        const bool actual_active = organism.cells[1].output_val > 0.0;
        require(std::isfinite(actions.positive_action),
                "hysteresis action output must be finite");
        require(actual_active == expected_active,
                "hysteresis state must follow Schmitt thresholds");
        if (t != 0 && actual_active != previous_active) {
            ++transitions;
            if (sequence[t] >= -0.5 && sequence[t] <= 0.5) {
                ++in_band_transitions;
            }
        }
        previous_active = actual_active;
    }
    require(organism.export_genome_json() == genome_before,
            "hysteresis episode must not mutate the genome");
}

static void test_hysteresis_modes() {
    size_t total_transitions = 0;
    size_t total_in_band_transitions = 0;
    for (size_t case_id = 0; case_id < 32; ++case_id) {
        const double offset = -0.2 + 0.4 *
            static_cast<double>(case_id) / 31.0;
        size_t transitions = 0;
        size_t in_band_transitions = 0;
        run_hysteresis_case(offset, transitions, in_band_transitions);
        require(transitions == 4, "each hysteresis case must switch four times");
        require(in_band_transitions == 0,
                "hysteresis must ignore in-band noise");
        total_transitions += transitions;
        total_in_band_transitions += in_band_transitions;
    }

    CellularOrganism reset_probe = make_hysteresis_organism();
    double active_inputs[4] = {0.8, 0.0, 0.0, 0.0};
    reset_probe.forward(active_inputs, false);
    reset_probe.reset_state(true);
    double neutral_inputs[4] = {0.0, 0.0, 0.0, 0.0};
    reset_probe.forward(neutral_inputs, false);
    require(reset_probe.cells[1].output_val < 0.0,
            "reset must clear the hysteresis latch");

    std::cout << "HYSTERESIS_CASES=32\n";
    std::cout << "HYSTERESIS_TOTAL_TRANSITIONS=" << total_transitions << "\n";
    std::cout << "HYSTERESIS_IN_BAND_TRANSITIONS="
              << total_in_band_transitions << "\n";
    std::cout << "HYSTERESIS_MODES=PASS\n";
}

- [x] **Step 4: Run the focused test**

Run:

```bash
cmake --build build --target test_flow_temporal_memory_modes
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_modes
```

Expected:

```text
DELAY_RECALL=PASS
HYSTERESIS_CASES=32
HYSTERESIS_TOTAL_TRANSITIONS=128
HYSTERESIS_IN_BAND_TRANSITIONS=0
HYSTERESIS_MODES=PASS
```

- [x] **Step 5: Commit mode and isolation evidence**

```bash
git add tests/test_flow_temporal_memory_modes.cpp
git commit -m "test: prove hysteretic modes and episode isolation"
```

### Task 3: Add explicit evidence labeling and regression validation

**Files:**
- Modify: `tests/test_flow_temporal_memory_modes.cpp`

- [x] **Step 1: Add machine-readable evidence labels**

Print a final summary only after all assertions pass:

```cpp
std::cout << "EVIDENCE_CLASS=REFERENCE_TOPOLOGY_EXPRESSIVITY\n";
std::cout << "EVOLUTION_DISCOVERY=NOT_TESTED\n";
std::cout << "GENERAL_INTELLIGENCE_CLAIM=NOT_MADE\n";
std::cout << "TEMPORAL_MEMORY_MODES=PASS\n";
```

Do not print a “champion”, “trained”, or profitability claim. Keep the
benchmark's scope limited to stateful expressivity and reset isolation.

- [x] **Step 2: Run the focused CTest target**

Run:

```bash
cmake -S . -B build
cmake --build build --target test_flow_temporal_memory_modes
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_modes
```

Expected: one test passes with 96 delayed-recall cases, 32 hysteresis
variants, 128 total expected transitions, zero illegal transitions, and the
explicit evidence labels.

- [x] **Step 3: Run the existing full regression suite**

Run:

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Observed: `test_flow_temporal_memory_modes` and the other 23 existing tests
pass. The pre-existing `test_binary_runtime_scale` remains the single failure
because `checkpoints/sdsc_mega_1million.bin` is absent; no file under
`include/kun/cellular/` was modified by this feature.

- [x] **Step 4: Commit the final evidence labeling**

```bash
git add tests/test_flow_temporal_memory_modes.cpp
git commit -m "test: label temporal memory evidence boundary"
```
