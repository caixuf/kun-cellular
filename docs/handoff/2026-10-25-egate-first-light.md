# Handoff — E-Gate first-light (Prefer A)

**Date:** 2026-10-25 Asia/Shanghai  
**Base tip:** `bff1c509` / `bff1c509169c706eaa74d03640c9566100f54787` (main `#13`; contains Phase3 `#12` @ `7b38b8e0`)  
**Branch:** `feat/cellular-egate-first-light`  
**Owners:** **手术刀** (measure / claim / archive) · **焊枪** standby (multi-scale adapter gap) · **颜料盒** none  
**Claim:** `docs/claims/E1-egate-first-light.md` · tag **E1**

## §七 product bet

Success = **E-Gate lit** on one claim pack (pre-register → anti-artifact → mechanism → archive-repro).  
**Not** win-rate / DouZero / PSRO / strength. Phase 4 2-ply **HOLD**. Win knives **STOP**.

## Four stages shipped

| Stage | Artifact |
|---|---|
| 1 Pre-register | `docs/claims/E1-egate-first-light.md` — tip_hash=`bff1c509…`, BD, metric pack, seeds, ≥3 learn scales, exec≠learn filled |
| 2 Anti-artifact | dual coverage+novelty · scales 41/82/164 · seeds 11/22/33/44 · OOD `seed_shift_+1000` + McNemar/paired — evidence JSON below |
| 3 Mechanism | claim Mechanism note — MapElitesGrid / stepping-stone / protect_window / metric pack (Lyapunov ≠ avalanche) |
| 4 Archive-repro | this handoff |

## Reproduce

```bash
# on caixuf WSL: /home/caixuf/code/kun-cellular @ tip bff1c509… / branch feat/cellular-egate-first-light
cd /home/caixuf/code/kun-cellular
git fetch origin && git checkout feat/cellular-egate-first-light
# unit path (Phase3 E-Gate / anti-artifact harness)
cmake --build build --target test_population_phase3_map_elites -j4
./build/test_population_phase3_map_elites
# or: ctest --test-dir build -R test_population_phase3_map_elites --output-on-failure
# measure dump → JSON
g++ -std=c++20 -O2 -Iinclude \
  artifacts/egate-first-light/egate_first_light_measure.cpp \
  -o artifacts/egate-first-light/egate_first_light_measure
./artifacts/egate-first-light/egate_first_light_measure \
  | tee artifacts/egate-first-light/anti_artifact_evidence.json
```

**Seeds:** `11, 22, 33, 44`  
**Learn scales (registered):** 41 / 82 / 164 cells · primary learn_scale=82 · exec_scale=82  
**Claim path:** `docs/claims/E1-egate-first-light.md`

## Evidence paths

| Path | Role |
|---|---|
| `artifacts/egate-first-light/anti_artifact_evidence.json` | dual · scales · seeds · OOD McNemar/paired · E-Gate verdict |
| `artifacts/egate-first-light/phase3_unit_run.txt` | `test_population_phase3_map_elites` PASS log |
| `artifacts/egate-first-light/egate_first_light_measure.cpp` | measure tool (header-only Prefer-A reuse) |

## Stale tip_hash (A7)

`docs/claims/E1-phase3-map-elites-fixture.md` tip `2817e2f0` **explicitly superseded** by this first-light pack @ `bff1c509`. Fixture retained as historical Phase3 dry-run; supersede appendix appended.

## E-tag honesty (A5)

| Tag | Verdict |
|---|---|
| **E1** | **Claimed** — stages documented + reproducible |
| **E2** | Not claimed — live cells≠82 adapters missing |
| Forbidden | DouZero / strength / win-rate / bare “emerged” |

## A0–A8

| ID | Verdict |
|---|---|
| A0 | PASS — branch off `bff1c509` clean fetch |
| A1 | PASS — pre-register tip_hash=`bff1c509…` |
| A2 | PASS — dual·≥3 scales·≥4 seeds·OOD+paired cited (JSON + unit) |
| A3 | PASS — mechanism note in claim |
| A4 | PASS — this handoff |
| A5 | PASS — E1 honest; exec≠learn filled; no DouZero |
| A6 | PASS — Phase4 HOLD; win STOP; 82/zero-alloc/det; no 4th ABI |
| A7 | PASS — `2817e2f0` superseded |
| A8 | PASS — 手术刀-only; 焊枪 gap dossier only (multi-scale adapter) — parent escalates |

## Gap → 焊枪 (do not unlock Phase4)

Live Prefer-A organism builder is 82-fixed; registering 41/164 without cell-count adapters blocks honest **E2**. First-light **E1** does not require that adapter. No new metric / no archive plumbing break.

## Park / STOP

Phase4 **HOLD** · win/DouZero **STOP** · no ADAS/zoo/mutate · no 4th ABI · do not touch `w1b-harden-repro`.
