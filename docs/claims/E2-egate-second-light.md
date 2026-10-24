# E-Gate second-light — E2-egate-second-light

**Tag:** **E2** (live multi-scale anti-artifact on CELLS axis 41/82/164)  
**Date:** 2026-10-25 Asia/Shanghai  
**Tip:** `50431c48d935828cf4c34298363c38853de481e8`  
**Branch:** `feat/cellular-egate-e2-adapters`  
**Owner:** 焊枪 (Prefer A — campaign builders, no 4th ABI)  
**Does not retag E1:** `docs/claims/E1-egate-first-light.md` remains **E1** (#14 @ `d684b038`)

## Pre-register (stage 1)

| Field | Value |
|---|---|
| claim_id | `E2-egate-second-light` |
| behavioral_descriptor (BD) | `coverage_frac × novelty_knn` (2-D MAP-Elites BD via `bd_from_coverage_novelty`) |
| metric pack | dual: `coverage_frac` + `novelty_knn`; companions: `avalanche_mean_size`, `lyapunov_max_loop_gain` (companion **≠** avalanche) |
| seeds | `11, 22, 33, 44` (≥4) |
| learn_scales | **live** cells **41 / 82 / 164** (`default_prefer_a_learning_scales` + live campaign dumps) |
| learn_scale_primary | **82** (Prefer-A primary; ABI unchanged) |
| exec_scale (primary path) | **82** |
| exec≠learn honesty | per campaign: exec=learn=N at that cell count; vs primary: 41/164 have `exec_equals_primary_learn=false` |
| stop_rule | `max_gens=0` E2 measure; live cell-count campaigns 41/82/164; Prefer-A ABI held |
| tip_hash | `50431c48d935828cf4c34298363c38853de481e8` |
| cmd | see handoff / evidence JSON |
| mechanism_sketch | see Mechanism note below |

## Anti-artifact (stage 2)

| Check | Evidence |
|---|---|
| Dual metrics | `coverage_frac` + `novelty_knn` (independent) |
| ≥3 **live** learn scales | organisms built & measured at **41**, **82**, **164** (not register-only) |
| ≥4 seeds | 11, 22, 33, 44 |
| OOD + paired/McNemar | OOD label `seed_shift_+1000` **per scale**; McNemar + `analyze_paired_differences` |
| Evidence paths | `artifacts/egate-e2-adapters/anti_artifact_evidence.json` · tool `artifacts/egate-e2-adapters/egate_e2_adapters_measure.cpp` · unit `tests/test_population_egate_e2_adapters.cpp` |

**Honesty:** OOD alone ≠ multi-scale. Each scale has live organisms at that cell count. AvalancheMetric for 164 campaign capped at `kAvalancheMaxSites=128` (Prefer-A existing limit) — documented in JSON.

## Mechanism note (stage 3)

Bound to landed Phase3 operators + **new** adapter entry points:

1. **MapElitesGrid** — `try_insert` / BD from `bd_from_coverage_novelty`.
2. **NoveltySteppingStoneArchive** — KNN novelty feeds grid.
3. **PopulationSpeciesNiche.protect_window** — niche age gate; L1 compatibility context.
4. **Metric pack** — FunctionalCoverage + novelty dual; AvalancheMetric size/branching **sized per campaign**; Lyapunov companion **≠** avalanche.
5. **NEW adapters** — `make_org(seed, cell_count)` campaign builder + `egate_e2_adapters_measure` live dumps at 41/82/164. Does **not** change `kPreferAOrganismCells` / genome ABI / zero-alloc.

## Archive-reproducible (stage 4)

See `docs/handoff/2026-10-25-egate-e2-adapters.md`.

## E1 / E2 / E3

| Tag | This claim |
|---|---|
| **E1** | Prior `#14` — unchanged |
| **E2** | **YES** — live 41 **and** 164 (+82) cell-count campaigns · dual · ≥4 seeds · OOD+paired · mechanism bound |
| **E3** | N/A if stages green |

Forbidden: bare “emerged” / strength / DouZero / win-rate / Phase4 / ADAS reopen / 4th ABI / synthetic-only as E2.

## Prefer-A held

| Invariant | Status |
|---|---|
| `kPreferAOrganismCells` = 82 | **unchanged** |
| No 4th genome ABI | **held** |
| Multi-scale = campaign builders | **yes** |
| Phase4 / ADAS | **HOLD / PARK** |
