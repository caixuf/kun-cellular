# Temporal Memory Evolutionary Discovery Evidence

## Purpose

Prove whether `MorphogeneticEvolutionEngine` can discover delayed recall and
hysteretic mode switching, after the reference-topology benchmark already
proved that the substrate can express those behaviors.

This is a separate claim from `REFERENCE_TOPOLOGY_EXPRESSIVITY`.

## Non-Goals

- Do not modify `include/kun/cellular/`.
- Do not modify ADAS checkpoints or ADAS trainers.
- Do not claim general intelligence, quantitative alpha, or that evolution
  learned an arbitrary algorithm.
- Do not fail CTest solely because evolution did not discover the graphs.

## Architecture

Add one C++ evidence executable:

`tests/test_flow_temporal_memory_evolution.cpp`

CMake already discovers `tests/test_flow_*.cpp`. The executable:

1. Reuses the same DelayN and Hysteresis reference graphs as the expressivity
   test.
2. Scores organisms with a deterministic fitness function, Hebbian learning
   off.
3. Compares four groups on disjoint train and holdout sequences:
   - reference topology
   - disconnected embryo
   - random minimal graphs
   - evolved population
4. Prints machine-readable labels. Evolution success or failure is a reported
   verdict, not mixed with the earlier expressivity result.

## Tasks

### Delayed recall

- Delay `k = 4` only. `param1 = 4.0 / 16.0` on `OP_DELAY_N`.
- Sequence length 48.
- Train seeds `1..8`, holdout seeds `101..108`. The two sets must be disjoint.
- At step `t`, expected `positive_action` is `sequence[t-4]` or `0` before
  history exists.
- Fitness = `1.0 - mean_absolute_error`, clipped to `[0, 1]`.

### Hysteresis modes

- Graph: `Sense_Input0 -> Gate_Hysteresis(+0.5, -0.5) -> Act_PosAction`.
- Use the same 19-step base pattern as the expressivity test, with 32 in-band
  offsets in `[-0.2, +0.2]` clamped to `[-0.45, +0.45]`.
- Train offsets: first 16. Holdout offsets: last 16.
- Fitness = `1.0` only when there are exactly four threshold transitions, zero
  in-band transitions, and the latch follows Schmitt rules. Otherwise:

```text
fitness = max(0, 1 - 0.25*|transitions-4| - 0.5*in_band_rate - mismatch_rate)
```

## Controls

| Group | Construction | Role |
|---|---|---|
| Reference | Hand-built DelayN / Hysteresis graphs | Upper bound |
| Disconnected | `SeedInitMode::DISCONNECTED_EMBRYO` | No hidden topology |
| Random | eight `MINIMAL_RANDOM_GRAPH` samples | Untrained baseline |
| Evolved | `MorphogeneticEvolutionEngine` | Discovery claim |

Evolution config:

- `seed_mode = DISCONNECTED_EMBRYO`
- `type_whitelist = FULL_24` so `OP_DELAY_N` and `GATE_HYSTERESIS` can appear
- `skeleton_lock = LOCKED`
- `fitness_driver = TASK_FITNESS_ONLY`
- `enable_baldwin_crystallization = false`
- `enable_mechanotransduction = false`
- `max_cells_limit = 32`
- `max_synapses_limit = 64`
- population 16, generations 20
- independent engine seeds `{7, 11, 19}`
- evaluate with `forward(inputs, false)` and `reset_state(true)` per episode

## Protocol Integrity (CTest must pass)

- Reference delayed-recall holdout fitness `>= 0.999`
- Disconnected delayed-recall holdout fitness `<= 0.55`
- Mean of eight random delayed-recall holdout scores
  `<= reference - 0.20`
- Train and holdout seed sets are disjoint
- Genome JSON is unchanged by fitness evaluation
- All outputs are finite
- Labels are printed even if discovery fails

## Discovery Verdict (reported, not a CTest hard fail)

`EVOLUTION_DISCOVERY=PASS` only if **both** hold:

1. Delayed recall: at least 2 of 3 evolution seeds produce a champion with
   holdout fitness `>= 0.90`, and that champion beats disconnected by `>= 0.40`
   and beats random mean by `>= 0.25`.
2. Hysteresis: at least 2 of 3 evolution seeds produce a champion with
   holdout fitness `>= 0.80`.

Otherwise print `EVOLUTION_DISCOVERY=FAIL` and keep CTest green if protocol
integrity holds.

Always print:

```text
EVIDENCE_CLASS=EVOLUTIONARY_DISCOVERY
REFERENCE_TOPOLOGY_EXPRESSIVITY=ALREADY_PROVEN_SEPARATELY
GENERAL_INTELLIGENCE_CLAIM=NOT_MADE
EVOLUTION_DISCOVERY=PASS|FAIL
```

## Validation

```bash
cmake -S . -B build
cmake --build build --target test_flow_temporal_memory_evolution
ctest --test-dir build --output-on-failure -R test_flow_temporal_memory_evolution
ctest --test-dir build --output-on-failure
```

If `test_binary_runtime_scale` fails because
`checkpoints/sdsc_mega_1million.bin` is missing, generate it with
`python3 tools/export_sdsc_binary.py` and rerun. Do not treat that as a
failure of this feature.
