# Temporal Memory Evolutionary Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a C++ evidence executable that compares reference, disconnected, random, and evolved organisms on delayed recall and hysteresis, then reports an evolutionary-discovery verdict without claiming general intelligence.

**Architecture:** One new file `tests/test_flow_temporal_memory_evolution.cpp`, discovered by the existing CMake `tests/test_flow_*.cpp` glob. It uses public `CellularOrganism` and `MorphogeneticEvolutionEngine` APIs only. Fitness evaluation never enables Hebbian learning. CTest passes on protocol integrity; `EVOLUTION_DISCOVERY=PASS|FAIL` is a separate printed claim.

**Tech Stack:** C++20, header-only KunCellular core, CMake, CTest, standard library only.

**Hard constraints:**
- Do not modify any file under `include/kun/cellular/`.
- Do not modify ADAS checkpoints or ADAS trainers.
- Do not use `assert`; CMake adds `-DNDEBUG`. Use a `require()` helper that prints and `std::exit(EXIT_FAILURE)`.
- Do not fail CTest only because evolution did not discover DelayN/Hysteresis.
- Work in an isolated git worktree/branch. Commit only this feature's files.

**Spec:** `docs/superpowers/specs/2026-09-03-temporal-memory-evolution-design.md`

**Engine APIs to use:**
- `MorphogeneticEvolutionEngine(size_t pop, uint32_t seed, const EvolutionConstraintConfig& cfg)`
- `engine.population()` for scoring
- `engine.evolve_generation()` after scores are written into `org.fitness_score`
- `CellularOrganism::create_disconnected_embryo(id)`
- `CellularOrganism::create_minimal_random_graph(id, seed)`
- `org.forward(inputs, false)`, `org.reset_state(true)`, `org.export_genome_json()`, `org.compile()`

Default whitelist is already `FULL_24`. Set `seed_mode = DISCONNECTED_EMBRYO`.

---

### Task 1: Protocol helpers and control-group evidence

**Files:**
- Create: `tests/test_flow_temporal_memory_evolution.cpp`

- [x] **Step 1: Write a compiling skeleton that fails at link/runtime until helpers exist**

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
using kun::EvolutionConstraintConfig;
using kun::FitnessDriverMode;
using kun::MorphogeneticEvolutionEngine;
using kun::SeedInitMode;
using kun::SkeletonLockMode;
using kun::Synapse;
using kun::TypeWhitelistMode;

static void require(bool condition, const char* message);
static void test_protocol_controls();

int main() {
    test_protocol_controls();
    return 0;
}
```

- [x] **Step 2: Configure and confirm the target is picked up**

```bash
cmake -S . -B build
cmake --build build --target test_flow_temporal_memory_evolution
```

Expected: link error `undefined reference to test_protocol_controls()` or equivalent.

- [x] **Step 3: Implement helpers, reference graphs, fitness, and control checks**

Use this exact body after the includes and `using` aliases. Runtime checks must use `require`, not `assert`.

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

static CellularOrganism make_delay_reference() {
    CellularOrganism organism;
    organism.organism_id = 104;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(make_cell(1, CellType::OP_DELAY_N, 4.0 / 16.0));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "delay reference must compile");
    return organism;
}

static CellularOrganism make_hysteresis_reference() {
    CellularOrganism organism;
    organism.organism_id = 200;
    organism.cells.push_back(make_cell(0, CellType::SENSE_RAW_INPUT_0, 1.0));
    organism.cells.push_back(make_cell(1, CellType::GATE_HYSTERESIS, 0.5, -0.5));
    organism.cells.push_back(make_cell(2, CellType::ACT_PRIMARY_POSITIVE));
    organism.synapses.push_back(make_synapse(0, 1));
    organism.synapses.push_back(make_synapse(1, 2));
    require(organism.compile(), "hysteresis reference must compile");
    return organism;
}

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

static double delay_fitness(CellularOrganism& organism,
                            const std::vector<uint32_t>& seeds) {
    const int delay = 4;
    const size_t length = 48;
    double abs_error = 0.0;
    size_t count = 0;
    const std::string genome_before = organism.export_genome_json();
    for (uint32_t seed : seeds) {
        const auto sequence = make_binary_sequence(seed, length);
        organism.reset_state(true);
        for (size_t t = 0; t < sequence.size(); ++t) {
            double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
            const auto actions = organism.forward(inputs, false);
            require(std::isfinite(actions.positive_action),
                    "delay output must be finite");
            const double expected =
                t >= static_cast<size_t>(delay)
                    ? sequence[t - static_cast<size_t>(delay)]
                    : 0.0;
            abs_error += std::abs(actions.positive_action - expected);
            ++count;
        }
    }
    require(organism.export_genome_json() == genome_before,
            "fitness evaluation must not mutate the genome");
    const double mae = abs_error / static_cast<double>(count);
    return std::clamp(1.0 - mae, 0.0, 1.0);
}

static double hysteresis_fitness(CellularOrganism& organism,
                                 const std::vector<double>& offsets) {
    const std::string genome_before = organism.export_genome_json();
    double total = 0.0;
    for (double offset : offsets) {
        const auto sequence = hysteresis_sequence(offset);
        organism.reset_state(true);
        bool expected_active = false;
        bool previous_active = false;
        size_t transitions = 0;
        size_t in_band = 0;
        size_t mismatches = 0;
        for (size_t t = 0; t < sequence.size(); ++t) {
            if (sequence[t] > 0.5) expected_active = true;
            else if (sequence[t] < -0.5) expected_active = false;
            double inputs[4] = {sequence[t], 0.0, 0.0, 0.0};
            const auto actions = organism.forward(inputs, false);
            require(std::isfinite(actions.positive_action),
                    "hysteresis output must be finite");
            const bool actual_active = actions.positive_action > 0.0;
            if (actual_active != expected_active) ++mismatches;
            if (t != 0 && actual_active != previous_active) {
                ++transitions;
                if (sequence[t] >= -0.5 && sequence[t] <= 0.5) ++in_band;
            }
            previous_active = actual_active;
        }
        if (transitions == 4 && in_band == 0 && mismatches == 0) {
            total += 1.0;
        } else {
            const double in_band_rate =
                static_cast<double>(in_band) / static_cast<double>(sequence.size());
            const double mismatch_rate =
                static_cast<double>(mismatches) / static_cast<double>(sequence.size());
            total += std::max(0.0, 1.0 - 0.25 * std::abs(static_cast<double>(transitions) - 4.0)
                                       - 0.5 * in_band_rate - mismatch_rate);
        }
    }
    require(organism.export_genome_json() == genome_before,
            "hysteresis fitness must not mutate the genome");
    return total / static_cast<double>(offsets.size());
}

static std::vector<uint32_t> train_delay_seeds() {
    return {1, 2, 3, 4, 5, 6, 7, 8};
}

static std::vector<uint32_t> holdout_delay_seeds() {
    return {101, 102, 103, 104, 105, 106, 107, 108};
}

static std::vector<double> train_hysteresis_offsets() {
    std::vector<double> offsets;
    for (size_t i = 0; i < 16; ++i) {
        offsets.push_back(-0.2 + 0.4 * static_cast<double>(i) / 31.0);
    }
    return offsets;
}

static std::vector<double> holdout_hysteresis_offsets() {
    std::vector<double> offsets;
    for (size_t i = 16; i < 32; ++i) {
        offsets.push_back(-0.2 + 0.4 * static_cast<double>(i) / 31.0);
    }
    return offsets;
}

static void test_protocol_controls() {
    const auto train = train_delay_seeds();
    const auto holdout = holdout_delay_seeds();
    for (uint32_t s : train) {
        for (uint32_t h : holdout) require(s != h, "train/holdout seeds must be disjoint");
    }

    CellularOrganism delay_ref = make_delay_reference();
    const double delay_ref_holdout = delay_fitness(delay_ref, holdout);
    require(delay_ref_holdout >= 0.999, "reference delay holdout fitness too low");

    CellularOrganism embryo = CellularOrganism::create_disconnected_embryo(1);
    const double embryo_holdout = delay_fitness(embryo, holdout);
    require(embryo_holdout <= 0.55, "disconnected embryo delay fitness too high");

    double random_sum = 0.0;
    for (uint32_t i = 1; i <= 8; ++i) {
        CellularOrganism rnd = CellularOrganism::create_minimal_random_graph(i, 1000 + i);
        random_sum += delay_fitness(rnd, holdout);
    }
    const double random_mean = random_sum / 8.0;
    require(random_mean <= delay_ref_holdout - 0.20,
            "random graphs must trail the delay reference");

    CellularOrganism hyst_ref = make_hysteresis_reference();
    const double hyst_ref_holdout =
        hysteresis_fitness(hyst_ref, holdout_hysteresis_offsets());
    require(hyst_ref_holdout >= 0.999, "reference hysteresis holdout fitness too low");

    std::cout << "DELAY_REFERENCE_HOLDOUT=" << delay_ref_holdout << "\n";
    std::cout << "DELAY_EMBRYO_HOLDOUT=" << embryo_holdout << "\n";
    std::cout << "DELAY_RANDOM_HOLDOUT_MEAN=" << random_mean << "\n";
    std::cout << "HYSTERESIS_REFERENCE_HOLDOUT=" << hyst_ref_holdout << "\n";
    std::cout << "PROTOCOL_CONTROLS=PASS\n";
}
```

- [x] **Step 4: Build and run the focused test**

```bash
cmake --build build --target test_flow_temporal_memory_evolution
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_evolution
```

Expected: `PROTOCOL_CONTROLS=PASS`.

- [x] **Step 5: Commit**

```bash
git add tests/test_flow_temporal_memory_evolution.cpp
git commit -m "test: add temporal memory evolution controls"
```

Include the Copilot trailer:

```text
Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

---

### Task 2: Evolution loop and discovery verdict

**Files:**
- Modify: `tests/test_flow_temporal_memory_evolution.cpp`

- [x] **Step 1: Add `test_evolution_discovery()` declaration and call it from `main` after protocol controls**

```cpp
static void test_evolution_discovery();

int main() {
    test_protocol_controls();
    test_evolution_discovery();
    std::cout << "EVIDENCE_CLASS=EVOLUTIONARY_DISCOVERY\n";
    std::cout << "REFERENCE_TOPOLOGY_EXPRESSIVITY=ALREADY_PROVEN_SEPARATELY\n";
    std::cout << "GENERAL_INTELLIGENCE_CLAIM=NOT_MADE\n";
    return 0;
}
```

The first rebuild must fail until the function exists.

- [x] **Step 2: Implement the evolution experiment**

Use `get_population_mut()` or non-const `population()` to write `fitness_score`, then `evolve_generation()`. Champion is the organism with best **train** fitness; report **holdout** fitness. Do not enable Hebbian during scoring.

```cpp
static EvolutionConstraintConfig make_evolution_config() {
    EvolutionConstraintConfig cfg;
    cfg.seed_mode = SeedInitMode::DISCONNECTED_EMBRYO;
    cfg.type_whitelist = TypeWhitelistMode::FULL_24;
    cfg.skeleton_lock = SkeletonLockMode::LOCKED;
    cfg.fitness_driver = FitnessDriverMode::TASK_FITNESS_ONLY;
    cfg.enable_baldwin_crystallization = false;
    cfg.enable_mechanotransduction = false;
    cfg.max_cells_limit = 32;
    cfg.max_synapses_limit = 64;
    cfg.immigrant_rate = 0.10;
    return cfg;
}

static double evolve_task_holdout(
    uint32_t engine_seed,
    bool delay_task,
    size_t& champion_cells,
    size_t& champion_synapses) {
    MorphogeneticEvolutionEngine engine(16, engine_seed, make_evolution_config());
    const auto train_seeds = train_delay_seeds();
    const auto holdout_seeds = holdout_delay_seeds();
    const auto train_off = train_hysteresis_offsets();
    const auto holdout_off = holdout_hysteresis_offsets();

    CellularOrganism best;
    double best_train = -1.0;
    for (int gen = 1; gen <= 20; ++gen) {
        auto& pop = engine.population();
        for (auto& org : pop) {
            org.fitness_score = delay_task
                ? delay_fitness(org, train_seeds)
                : hysteresis_fitness(org, train_off);
            if (org.fitness_score > best_train) {
                best_train = org.fitness_score;
                best = org;
            }
        }
        if (gen < 20) engine.evolve_generation();
    }
    champion_cells = best.cells.size();
    champion_synapses = best.synapses.size();
    return delay_task ? delay_fitness(best, holdout_seeds)
                      : hysteresis_fitness(best, holdout_off);
}

static void test_evolution_discovery() {
    const std::vector<uint32_t> engine_seeds = {7, 11, 19};
    CellularOrganism embryo = CellularOrganism::create_disconnected_embryo(1);
    const double embryo_delay = delay_fitness(embryo, holdout_delay_seeds());
    double random_sum = 0.0;
    for (uint32_t i = 1; i <= 8; ++i) {
        CellularOrganism rnd = CellularOrganism::create_minimal_random_graph(i, 2000 + i);
        random_sum += delay_fitness(rnd, holdout_delay_seeds());
    }
    const double random_mean = random_sum / 8.0;

    size_t delay_wins = 0;
    size_t hyst_wins = 0;
    for (uint32_t seed : engine_seeds) {
        size_t cells = 0, syns = 0;
        const double delay_holdout =
            evolve_task_holdout(seed, true, cells, syns);
        std::cout << "EVOLVED_DELAY_SEED=" << seed
                  << " HOLDOUT=" << delay_holdout
                  << " CELLS=" << cells
                  << " SYNAPSES=" << syns << "\n";
        if (delay_holdout >= 0.90 &&
            delay_holdout >= embryo_delay + 0.40 &&
            delay_holdout >= random_mean + 0.25) {
            ++delay_wins;
        }

        const double hyst_holdout =
            evolve_task_holdout(seed + 100, false, cells, syns);
        std::cout << "EVOLVED_HYSTERESIS_SEED=" << seed
                  << " HOLDOUT=" << hyst_holdout
                  << " CELLS=" << cells
                  << " SYNAPSES=" << syns << "\n";
        if (hyst_holdout >= 0.80) ++hyst_wins;
    }

    const bool discovered = delay_wins >= 2 && hyst_wins >= 2;
    std::cout << "DELAY_EVOLUTION_WINS=" << delay_wins << "/3\n";
    std::cout << "HYSTERESIS_EVOLUTION_WINS=" << hyst_wins << "/3\n";
    std::cout << "EVOLUTION_DISCOVERY=" << (discovered ? "PASS" : "FAIL") << "\n";
}
```

Then print the evidence labels from `main` after both tests. CTest return code stays 0 if `require()` never fired.

- [x] **Step 3: Run focused then full CTest**

```bash
cmake --build build --target test_flow_temporal_memory_evolution
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_evolution
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Expected:
- `PROTOCOL_CONTROLS=PASS`
- `EVIDENCE_CLASS=EVOLUTIONARY_DISCOVERY`
- `EVOLUTION_DISCOVERY=PASS` or `FAIL`
- existing tests still pass
- if `test_binary_runtime_scale` fails on missing `checkpoints/sdsc_mega_1million.bin`, run `python3 tools/export_sdsc_binary.py` and rerun; that is not this feature's bug

- [x] **Step 4: Confirm protected files were not touched**

```bash
git diff --name-only
```

Must not include `include/kun/cellular/*` or ADAS checkpoint files.

- [x] **Step 5: Commit**

```bash
git add tests/test_flow_temporal_memory_evolution.cpp
git commit -m "test: report temporal memory evolutionary discovery"
```

Include the Copilot trailer.

---

### Task 3: Record the observed verdict in the plan

**Files:**
- Modify: `docs/superpowers/plans/2026-09-03-temporal-memory-evolution.md`

- [x] **Step 1: After the run, mark every checkbox `[x]`**
- [x] **Step 2: Append the exact printed labels under an Observed section**
- [x] **Step 3: Commit the plan status**

```bash
git add docs/superpowers/plans/2026-09-03-temporal-memory-evolution.md
git commit -m "docs: record temporal memory evolution verdict"
```

Do not rewrite a FAIL into PASS. If discovery failed, leave `EVOLUTION_DISCOVERY=FAIL` and stop. The next phase is a new spec, not a silent threshold change.


### Observed

```text
DELAY_REFERENCE_HOLDOUT=1
DELAY_EMBRYO_HOLDOUT=0.5
DELAY_RANDOM_HOLDOUT_MEAN=0.267059
HYSTERESIS_REFERENCE_HOLDOUT=1
PROTOCOL_CONTROLS=PASS
EVOLVED_DELAY_SEED=7 HOLDOUT=0.54723 CELLS=12 SYNAPSES=18
EVOLVED_HYSTERESIS_SEED=7 HOLDOUT=0 CELLS=8 SYNAPSES=0
EVOLVED_DELAY_SEED=11 HOLDOUT=0.519362 CELLS=10 SYNAPSES=12
EVOLVED_HYSTERESIS_SEED=11 HOLDOUT=0 CELLS=11 SYNAPSES=16
EVOLVED_DELAY_SEED=19 HOLDOUT=0.518351 CELLS=9 SYNAPSES=9
EVOLVED_HYSTERESIS_SEED=19 HOLDOUT=0 CELLS=8 SYNAPSES=0
DELAY_EVOLUTION_WINS=0/3
HYSTERESIS_EVOLUTION_WINS=0/3
EVOLUTION_DISCOVERY=FAIL
EVIDENCE_CLASS=EVOLUTIONARY_DISCOVERY
REFERENCE_TOPOLOGY_EXPRESSIVITY=ALREADY_PROVEN_SEPARATELY
GENERAL_INTELLIGENCE_CLAIM=NOT_MADE
```
