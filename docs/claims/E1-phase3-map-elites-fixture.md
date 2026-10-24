# E-Gate pre-register fixture — E1-phase3-map-elites-fixture

**Tag:** **E1** (dry-run fixture — descriptive / pipeline smoke; not marketed as emergence)  
**Date:** 2026-10-25 Asia/Shanghai  
**Tip:** `2817e2f0851c466b5204dd384f5e6e14600095a0`  
**Branch:** `feat/cellular-phase3-map-elites`  
**Owner:** 焊枪 (Prefer A)

## Pre-register

| Field | Value |
|---|---|
| claim_id | `E1-phase3-map-elites-fixture` |
| behavioral_descriptor | `coverage_frac × novelty_knn` (2-D MAP-Elites BD) |
| metrics | `FunctionalCoverageContract`, `novelty` (KNN stepping-stone), `avalanche_mean_size` |
| seeds | `11, 22, 33, 44` (≥4) |
| learn_scales | cells **41 / 82 / 164** (learning axis — NOT FE `LIFEFORM_SCALES`) |
| learn_scale | **82** (Prefer-A primary) |
| exec_scale | **82** (demo/eval matches learn for this fixture) |
| exec_equals_learn_claimed | true only because both are 82; **ban** if exec≠learn yet claimed equal |
| stop_rule | `max_gens=0` dry-run — no evolution campaign |
| mechanism_sketch | MAP-Elites cell insert/replace + `PopulationSpeciesNiche.protect_window` + avalanche size histogram/branching; Lyapunov loop-gain is companion only |
| cmd | `ctest -R test_population_phase3_map_elites` |
| tip_hash | `2817e2f0851c466b5204dd384f5e6e14600095a0` |

## E1 / E2 / E3 discipline

| Tag | This fixture |
|---|---|
| **E1** | **YES** — dry-run / descriptive pipeline proof |
| **E2** | NOT claimed — would require live anti-artifact campaign beyond unit fixture |
| **E3** | N/A unless stage 2/3 fails on a real claim |

Forbidden: bare “emerged / life / AGI” without E-tag.

## Anti-artifact checklist (fixture path)

- [x] Dual metrics named (coverage + novelty)
- [x] ≥4 seeds listed
- [x] ≥3 distinct **learning** scales listed
- [x] OOD + McNemar path exercised in `test_population_phase3_map_elites` (synthetic)
- [x] exec_scale and learn_scale both recorded

## Ban: execution scale ≠ learning scale

Evaluation/demo size must not be marketed as the scale the agent **learned** at.  
200-cell openness probe (NON-BLOCK) may **execute** at 200 only if labeled `exec_scale=200` with Prefer-A `learn_scale=82` when different.

---

## Supersede note (E-Gate first-light · 2026-10-25)

**Status:** Fixture **superseded for tip_hash** by `docs/claims/E1-egate-first-light.md`.

| Field | Old (stale) | Current |
|---|---|---|
| tip_hash | `2817e2f0851c466b5204dd384f5e6e14600095a0` (Phase2 `#11`) | **`bff1c509169c706eaa74d03640c9566100f54787`** (main `#13`, contains Phase3 `#12`) |
| first-light claim | this dry-run fixture | `E1-egate-first-light` |
| role of this file | retained as Phase3 dry-run historical fixture | do **not** use `2817e2f0` as current E-Gate tip |

This file remains an **E1 dry-run** record of Phase3 landing. Current tip / Prefer-A first-light pack lives in `E1-egate-first-light.md` + `docs/handoff/2026-10-25-egate-first-light.md`.
