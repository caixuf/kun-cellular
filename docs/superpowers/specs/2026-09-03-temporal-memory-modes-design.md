# Temporal Memory and Discrete Modes Evidence Design

## Purpose

Establish the first independent evidence line for capabilities beyond the
existing embodied-control demonstrations:

1. Stateful temporal memory can preserve and recall an input after a known
   number of execution steps.
2. Hysteretic discrete modes remain stable under bounded in-band noise and
   switch only at the configured entry and exit thresholds.
3. Resetting an organism removes episode state without changing its topology
   or parameters.

The benchmark is an expressivity and runtime-state test. It is not evidence
that evolutionary search can discover the tested topology, nor evidence of
general intelligence.

## Constraints

- Do not modify `include/kun/cellular/` or any existing organism checkpoint.
- Drive the real `CellularOrganism::forward()` path; do not duplicate the
  organism's operators in Python or in a test-only neural-network
  implementation.
- Disable Hebbian updates during measurement so the test isolates deterministic
  state transitions rather than online parameter drift.
- Keep the benchmark deterministic and self-contained. It must not require
  external data, a GPU, or a network service.
- Preserve the existing CMake test discovery convention
  (`tests/test_flow_*.cpp`).

## Architecture

Add one C++ test executable:

```text
tests/test_flow_temporal_memory_modes.cpp
```

The test constructs small reference organisms through the public core data
structures:

```text
Sense_Input0 -> Op_DelayN -> Act_PosAction
Sense_Input0 -> Gate_Hysteresis -> Act_PosAction
```

The first graph uses the core cell's fixed 16-slot delay buffer. The
`Op_DelayN` parameter is selected to test delays of 1, 4, and 16 steps. The
second graph uses separate high/low thresholds and reads the gate cell's
actual output from the organism after each forward step.

No new CMake target is required: the existing `test_flow_*.cpp` glob will
discover the file after reconfiguration.

## Evidence Protocol

### Delayed recall

For each delay `k` in `{1, 4, 16}`, run 32 deterministic input sequences of
at least 96 steps. Each input is a reproducible binary pulse stream generated
by a local integer recurrence. At step `t`, the expected output is the input
from `t-k`, or zero before the history is populated.

For every sequence:

- reset the organism before the episode;
- call `forward(input, false)` for every step;
- compare the action output against the delayed target;
- record the maximum absolute error;
- repeat the same sequence after another reset and compare both traces.

### Hysteretic mode switching

Use entry threshold `+0.5` and exit threshold `-0.5`. Each of 32 deterministic
sequences contains:

- a value above `+0.5` to enter the active mode;
- alternating values inside `[-0.5, +0.5]` to exercise noise immunity;
- a value below `-0.5` to leave the active mode;
- another in-band noise segment and a second entry transition.

The test counts mode changes and any change while the input remains inside the
dead band. The gate must produce exactly the prescribed transition count and
zero illegal in-band transitions. A reset followed by a neutral input must
start in the inactive mode.

### Structural and episode isolation

Before and after each episode, capture cell types, parameters, synapse
endpoints, weights, and active flags. State changes are allowed only in runtime
fields. The benchmark also checks that a reset removes delayed values and the
hysteresis latch from the previous episode.

## Pass Criteria

The executable reports `PASS` only when all of the following hold:

- all 96 delayed-recall cases (32 sequences x 3 delay values) have maximum
  absolute error at or below `1e-12`;
- every repeated delayed-recall trace is identical after reset;
- all hysteresis sequences have zero in-band mode changes;
- all hysteresis sequences have exactly two entry/exit transitions;
- neutral input after reset starts inactive;
- no topology or genetic parameter changes are detected;
- all measured outputs are finite.

The report must label the result as
`REFERENCE_TOPOLOGY_EXPRESSIVITY`, not as an evolved-organism result.

## Validation

Run the focused test first:

```bash
cmake --build build --target test_flow_temporal_memory_modes
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_modes
```

Then run the existing full CTest suite to detect regressions. The benchmark
must not change the expected behavior of any existing test.

## Follow-up Boundary

Only after this reference-topology evidence passes should a separate phase add
an evolutionary trainer for the same tasks. That phase must compare the
evolved population against the reference topology, a disconnected embryo, and
random seeded organisms, with held-out sequences and multiple seeds. Its
success or failure must remain a separate claim from this expressivity test.
