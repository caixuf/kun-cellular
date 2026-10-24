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
