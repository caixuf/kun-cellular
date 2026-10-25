# Handoff — Openness first-light (Prefer · 手术刀 measure)

**Date:** 2026-10-25 Asia/Shanghai  
**Measure tip:** `3d61327e` / `3d61327e97dc07476143d15719bd828e122c3a60` (main #21 ⊃ #20)  
**Prefer code tip:** `453ccb1a` / `453ccb1adb143ce51fa9a3b605a9aef80296d594` (#20 openness-scale)  
**Branch:** `feat/cellular-openness-first-light` off `3d61327e` only  
**Owner:** **手术刀** (measure / dossier) · **焊枪** (gap only)  
**Gate:** 找茬喵 · **Merge:** 扳手  
**Pack id:** `openness-scale-sweep` — **NOT** E4 ABI; E1/E2/E3 **closed** (do not retag)

**Invariant:** one tip exercise of landed #20 pack — **NOT strength, NOT E4, NOT ADAS.**

## Protect once (A1)

```bash
# caixuf WSL: /home/caixuf/code/kun-cellular @ feat/cellular-openness-first-light
# tip 3d61327e
cmake --build build --target test_population_openness_protect_live -j4
./build/test_population_openness_protect_live
echo exit:$?
```

| Check | Result |
|---|---|
| Test | `./build/test_population_openness_protect_live` **PASS** |
| Assertions | live deme: survive purge in protect_window, then deme consequence; live ecology_grid: step_generation niche_protect purge path |
| Exit | 0 |
| Measure tip | `3d61327e97dc07476143d15719bd828e122c3a60` |
| Not | protect_window retune · new speciation ABI · biosphere heap |

## Multi-scale pack once (A2/A3)

| Scale point | Axis | Role | Live? |
|---|---|---|---|
| **41** | CELLS | learn campaign (measure-only adapter) | YES |
| **82** | CELLS | learn primary **and** Prefer-A **exec** | YES |
| **164** | CELLS | learn campaign (measure-only adapter) | YES |

| Check | Value |
|---|---|
| Distinct scales | **3** (≥3) |
| Span ratio `max/min` | **164/41 = 4.0** (≥4×) |
| Dual metrics | `coverage_frac` · `novelty_knn` |
| Seeds (≥4) | `11, 22, 33, 44` |
| OOD label | `seed_shift_+1000` (McNemar + paired diffs) |
| Exec cells | **`kPreferAOrganismCells=82`** — **held** |
| Learn ≠ exec lift | YES — 41/164 measure-only; runtime ABI stays 82 |
| tip_hash (evidence/measure) | `3d61327e97dc07476143d15719bd828e122c3a60` |
| anti_artifact | pass=true · dual_ok · seeds_ok · scales_ok · ood_paired_ok · span_ok=true span_ratio=4.0 |
| scale_unit | `./build/test_population_openness_scale_sweep` **PASS** |

## Reproduce

```bash
cd /home/caixuf/code/kun-cellular
git fetch origin
git checkout feat/cellular-openness-first-light

# protect once
cmake --build build --target test_population_openness_protect_live -j4
./build/test_population_openness_protect_live

# scale unit
cmake --build build --target test_population_openness_scale_sweep -j4
./build/test_population_openness_scale_sweep

# fresh evidence at tip 3d61327e
g++ -std=c++20 -O2 -Iinclude \
  artifacts/openness-scale/openness_scale_measure.cpp \
  -o artifacts/openness-scale/openness_scale_measure
./artifacts/openness-scale/openness_scale_measure \
  | tee artifacts/openness-scale/anti_artifact_evidence.json
echo exit:$?
```

**Seeds:** `11, 22, 33, 44`  
**Learn scales (live):** 41 / 82 / 164 cells · primary learn=82 · **exec=82**  
**OOD:** `seed_shift_+1000` paired/McNemar

## Evidence paths

| Path | Role |
|---|---|
| `artifacts/openness-scale/anti_artifact_evidence.json` | **Re-run** at tip `3d61327e` |
| `artifacts/openness-scale/openness_scale_measure.cpp` | measure tool (tip_hash → `3d61327e…`) |
| `tests/test_population_openness_scale_sweep.cpp` | unit: ≥3 · ≥4× · exec 82 |
| `tests/test_population_openness_protect_live.cpp` | A1 protect live deme/ecology |
| this handoff | archive: protect + scale metrics · seeds · tip · cmd |

## Labels / claim language

| Allowed | Forbidden |
|---|---|
| openness / coverage / novelty · pack `openness-scale-sweep` | **strength** · **DouZero** · **win-rate** · bare "emerged" |
| E1/E2/E3 closed · this pack **not E4** | Retag E3-as-emergence · invent E4 ABI |

## E1 / E2 / E3 / this pack

| Tag / pack | Status |
|---|---|
| E1 #14 | **closed** — unchanged |
| E2 #16 | **closed** — unchanged |
| E3 #17 | **closed** — unchanged |
| **openness-scale-sweep** | first-light re-exercise — coverage/novelty language — **not E4** |

## Acceptance A0–A7 (self-check)

| ID | Check | Verdict |
|---|---|---|
| A0 | Branch off `3d61327e`; WSL caixuf only | **PASS** |
| A1 | Protect_window exercised once; metrics archived | **PASS** — protect_live PASS + this handoff |
| A2 | Multi-scale pack once: ≥3 scales · ≥4× span · dual · OOD+paired seeds | **PASS** — 41/82/164 · span=4.0 · evidence JSON |
| A3 | tip_hash + cmd + seeds in handoff; labels honest (no strength) | **PASS** |
| A4 | Exec 82 held; learn ≠ exec lift; champ untouched | **PASS** — kPreferAOrganismCells=82; champ md5=`7424dd…` |
| A5 | Cellular ADAS PARK · no silent P1 · no invented search wins · KAD deepen PARK | **PASS** |
| A6 | No DouZero/GPU/PSRO / 4th ABI / zoo-mutate / Phase4 deepen | **PASS** |
| A7 | Owner 手术刀 (+焊枪 gap); 找茬喵 gate | **PASS** — ready for 找茬喵 |

## Park / bans held

DouZero · GPU · PSRO · exec cell lift · 4th ABI · ADAS reopen · silent P1 · zoo/mutate · Phase4 deepen · Bot-box private clone · protect_window retune · silent drop scale/seed · KAD deepen until land · champ touch.

## Stream readiness

A0–A7 green on measure tip `3d61327e` — hand **找茬喵**.  
**Not** DONE until 找茬喵 PASS → 扳手 squash. No strength / DouZero / win-rate claim.
