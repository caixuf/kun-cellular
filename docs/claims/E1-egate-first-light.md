# E-Gate first-light — E1-egate-first-light

**Tag:** **E1** (honest first-light — stages lit + archive-repro; live multi-scale campaign thin)  
**Date:** 2026-10-25 Asia/Shanghai  
**Tip:** `bff1c509169c706eaa74d03640c9566100f54787`  
**Branch:** `feat/cellular-egate-first-light`  
**Owner:** 手术刀 (Prefer A measure-first)  
**Supersedes fixture tip for first-light:** `docs/claims/E1-phase3-map-elites-fixture.md` (stale tip `2817e2f0` → superseded note)

## Pre-register (stage 1)

| Field | Value |
|---|---|
| claim_id | `E1-egate-first-light` |
| behavioral_descriptor (BD) | `coverage_frac × novelty_knn` (2-D MAP-Elites BD via `bd_from_coverage_novelty`) |
| metric pack | dual: `coverage_frac` + `novelty_knn`; companions: `avalanche_mean_size`, `lyapunov_max_loop_gain` (companion **≠** avalanche) |
| seeds | `11, 22, 33, 44` (≥4 = `kAntiArtifactMinSeeds`) |
| learn_scales | cells **41 / 82 / 164** (`default_prefer_a_learning_scales`) |
| learn_scale | **82** (Prefer-A primary) |
| exec_scale | **82** |
| exec_equals_learn_claimed | true — both 82; **ban** conflation if they differed |
| stop_rule | `max_gens=0` first-light measure; Prefer-A 82 live dump; scales registered 41/82/164 |
| tip_hash | `bff1c509169c706eaa74d03640c9566100f54787` |
| cmd | see handoff / evidence JSON |
| mechanism_sketch | see Mechanism note below |

## Anti-artifact (stage 2)

| Check | Evidence |
|---|---|
| Dual metrics | `coverage_frac` + `novelty_knn` (independent) |
| ≥3 learn scales | 41 / 82 / 164 registered |
| ≥4 seeds | 11, 22, 33, 44 |
| OOD + paired/McNemar | OOD label `seed_shift_+1000`; McNemar + `analyze_paired_differences` via `anti_artifact.hpp` / `statistical_evaluation.hpp` |
| Evidence paths | `artifacts/egate-first-light/anti_artifact_evidence.json` · `artifacts/egate-first-light/phase3_unit_run.txt` · tool `artifacts/egate-first-light/egate_first_light_measure.cpp` · unit `ctest -R test_population_phase3_map_elites` |

**Honesty:** Harness greens on registered dual·scales·seeds·OOD+paired. Live Prefer-A organism dump is **82-cell only**; 41/164 are registered learn-scale points without distinct cell-count campaign adapters → **not E2**. No synthetic-only marketed as E2.

## Mechanism note (stage 3)

Bound to landed Phase3 operators only (no post-hoc story):

1. **MapElitesGrid** (`map_elites_grid.hpp`) — `try_insert` / replace by QD (`coverage + novelty_weight×novelty`); BD from `bd_from_coverage_novelty`.
2. **NoveltySteppingStoneArchive** (`novelty_stepping_stone.hpp`) — KNN novelty feeds grid; mid-goal keep under NOVELTY/HYBRID.
3. **PopulationSpeciesNiche.protect_window** (`population_species_niche.hpp`) — niche age gate before purge; L1 `compatibility_distance` mate/species context.
4. **Metric pack** — `FunctionalCoverageContract` + novelty dual; `AvalancheMetric` size/branching as criticality companion; **Lyapunov** (`check_lyapunov_stability`) is stability companion only — **not** avalanche substitute (`AvalancheMetric::lyapunov_companion_note`).

## Archive-reproducible (stage 4)

See `docs/handoff/2026-10-25-egate-first-light.md` — tip + cmd + seeds + this claim path.

## E1 / E2 / E3

| Tag | This claim |
|---|---|
| **E1** | **YES** — default honest first-light |
| **E2** | NOT claimed — live multi-scale (41/164 cell-count) adapters missing; OOD is seed-shift on same ABI |
| **E3** | N/A — stages documented; anti-artifact harness path greens without marketing E2 |

Forbidden: bare “emerged” / strength / DouZero / win-rate.

## Gap dossier (焊枪 escalate — parent only)

| Gap | Block? |
|---|---|
| Prefer-A organism builder fixed at 82 — no live cells=41/164 campaign adapter | Blocks honest **E2**; does **not** block E1 first-light |
| No archive dump/tip_hash plumbing break | N/A — tip recorded |
| No new metric beyond Phase3 headers required | N/A |

Phase4 **HOLD**. Win knives **STOP**. No 4th ABI.
