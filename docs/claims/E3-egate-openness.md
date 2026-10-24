# E-Gate openness — E3-egate-openness (honest demotion)

**Tag:** **E3** (artifact — failed stage 2 anti-artifact Prefer check)  
**Date:** 2026-10-25 Asia/Shanghai  
**Tip / tip_hash:** `ab885f994fd2e6f4fbe257a33d7255368798b5d4`  
**Branch:** `feat/cellular-egate-e3` off `ab885f99` only  
**Owner:** 手术刀 (Prefer A — controlled demotion measure-first)  
**Does not retag:** E1 `#14` remains **E1**; E2 `#16` @ `ab885f99` remains **E2**

## Intended demotion reason (lock)

| Field | Value |
|---|---|
| Prefer check failed | **dual_independence** (`AntiArtifactEvidence.dual.independent=false`) |
| failed_stage | **2** (`ANTI_ARTIFACT`) |
| Why not fiat | dual · ≥3 live scales · ≥4 seeds · OOD+paired **vocabulary still invoked** (reuse E2 live 41/82/164 adapters); fail is measurable via `evaluate_anti_artifact` → `run_e_gate` |
| Why not E2 / emergence | stage2 fail → tag **E3**; `pass_e2=false`. **NOT** emergence / almost-E2 / success |
| Why not E1 | code path on stage2/3 fail is **E3** (`e_gate.hpp`); do not soften |

## Pre-register (stage 1)

| Field | Value |
|---|---|
| claim_id | `E3-egate-openness` |
| behavioral_descriptor (BD) | `coverage_frac × novelty_knn` (2-D MAP-Elites BD via `bd_from_coverage_novelty`) |
| metric pack | dual names: `coverage_frac` + `novelty_knn` (independence **intentionally false**); companions: `avalanche_mean_size`, `lyapunov_max_loop_gain` (companion **≠** avalanche) |
| seeds | `11, 22, 33, 44` (≥4) |
| learn_scales | **live** cells **41 / 82 / 164** (reuse E2 adapters) |
| learn_scale_primary | **82** (Prefer-A primary; ABI unchanged) |
| exec_scale (primary path) | **82** |
| exec≠learn honesty | per campaign: exec=learn=N; vs primary: 41/164 have `exec_equals_primary_learn=false` |
| stop_rule | `max_gens=0` E3 controlled demotion; invoke live 41/82/164 + dual·≥3·≥4·OOD; fail Prefer dual_independence |
| tip_hash | `ab885f994fd2e6f4fbe257a33d7255368798b5d4` |
| cmd | see handoff / evidence JSON |
| mechanism_sketch | present (stage3 would pass if stage2 green) — demotion is **stage2** |

## Anti-artifact (stage 2) — intentional Prefer fail

| Check | Invoked? | Result |
|---|---|---|
| Dual metric **names** | yes (`coverage_frac`, `novelty_knn`) | present |
| Dual **independence** | yes (lever) | **FAIL** — `independent=false` |
| ≥3 **live** learn scales | yes 41/82/164 | pass (`scales_ok`) |
| ≥4 seeds | yes 11/22/33/44 | pass (`seeds_ok`) |
| OOD + paired/McNemar | yes `seed_shift_+1000` per scale | pass (`ood_paired_ok`) |
| Overall `evaluate_anti_artifact` | — | **pass=false** → `run_e_gate` **E3** |

Evidence: `artifacts/egate-e3/anti_artifact_evidence.json` · tool `artifacts/egate-e3/egate_e3_demotion_measure.cpp`

## Mechanism (stage 3)

Mechanism sketch **present** (reuse E2 adapters + Phase3 operators + demotion lever documented). Pack already **E3 from stage2** — stage3 not reached by `run_e_gate` after stage2 fail.

## Archive-reproducible (stage 4)

See `docs/handoff/2026-10-25-egate-e3.md`.

## E1 / E2 / E3 trichotomy

| Tag | Claim | Status |
|---|---|---|
| **E1** | `docs/claims/E1-egate-first-light.md` `#14` | **unchanged** |
| **E2** | `docs/claims/E2-egate-second-light.md` `#16` @ `ab885f99` | **unchanged** |
| **E3** | **this claim** | **lit** — honest demotion |

Forbidden: bare “emerged” / E3-as-emergence / strength / DouZero / win-rate / Phase4 / ADAS reopen / 4th ABI / retag E1/E2 / skip vocabulary then fiat E3.

## Prefer-A held

| Invariant | Status |
|---|---|
| `kPreferAOrganismCells` = 82 | **unchanged** |
| No 4th genome ABI | **held** |
| Vocabulary dual·≥3·≥4·OOD invoked | **yes** |
| Phase4 / ADAS | **HOLD / PARK** |
| 手术刀-only | **yes** (no 焊枪 — existing `run_e_gate` demotion path) |
