# Handoff — Openness scale-sweep (Stream A · 手术刀 measure slice)

**Date:** 2026-10-25 Asia/Shanghai  
**Base tip:** `741c214c` / `741c214c8b6026616bbc8e92696034047348f9e3` (Flagship STATUS / Prefer A+B rails)  
**Branch tip:** `9a6a7c5d` / `9a6a7c5d42c4ef2cf541f9421fc28e763a2abfe5` (A1 protect on measure `862dfa96`) · PR #20  
**Branch:** `feat/cellular-openness-scale`  
**Owners:** **手术刀** (E-Gate multi-scale measure / span / handoff) · **焊枪** (A1 protect_window live deme/ecology wire)  
**Gate:** 找茬喵 · **Merge:** 扳手  
**Pack id:** `openness-scale-sweep` — **NOT** a new E4 ABI; E1/E2/E3 **closed** (do not retag)

## §七 product bet

Success = Prefer-A **scale-sweep** evidence: ≥3 learning-scale points · max/min span ≥4× · dual · OOD+paired · exec 82 held · learn span ≠ exec lift.  
**Not** DouZero / strength / win-rate / ADAS reopen / 4th ABI.

## Scale table (Prefer A2)

| Scale point | Axis | Role | Live measure? |
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
| Learn ≠ exec lift | YES — 41/164 are measure campaigns; runtime ABI stays 82 |

## Prefer checks A0–A6 (co-land tip `9a6a7c5d`)

| ID | Check | Verdict |
|---|---|---|
| A0 | Branch off `741c214c` | **PASS** (this branch tip ancestry) |
| A1 | New-species protect_window live deme wire | **PASS** — tip `9a6a7c5d` · deme + ecology_grid NicheRegistry wire · purge_all_unprotected · test_population_openness_protect_live PASS |
| A2 | Scale sweep ≥3 · span ≥4× | **PASS** — 41/82/164 · span=4.0 · `learning_scale_span_ok` |
| A3 | Dual + OOD+paired seeds | **PASS** — evidence JSON |
| A4 | Exec 82 / zero-alloc / det; learn ≠ exec lift | **PASS** — assert in measure + unit |
| A5 | This handoff with scale table | **PASS** |
| A6 | No DouZero / strength / win-rate / ADAS / silent P1 / 4th ABI / zoo | **PASS** |

## Collision / ownership

| Path | Owner |
|---|---|
| `deme.hpp` · `ecology_grid.hpp` · `population_species_niche.hpp` | **焊枪 — DO NOT TOUCH** |
| `tests/test_population_openness_protect_live.cpp` | **焊枪** |
| `anti_artifact.hpp` span helpers | **手术刀** |
| `artifacts/openness-scale/**` | **手术刀** |
| `tests/test_population_openness_scale_sweep.cpp` | **手术刀** |
| this handoff | **手术刀** |

## Reproduce

```bash
# on caixuf WSL: /home/caixuf/code/kun-cellular @ feat/cellular-openness-scale
cd /home/caixuf/code/kun-cellular
git fetch origin && git checkout feat/cellular-openness-scale

# unit: span>=4 · exec=82 · anti_artifact compat
cmake --build build --target test_population_openness_scale_sweep -j4
./build/test_population_openness_scale_sweep
# or: ctest --test-dir build -R test_population_openness_scale_sweep --output-on-failure

# measure dump → JSON (three live campaigns + span_ok)
g++ -std=c++20 -O2 -Iinclude \
  artifacts/openness-scale/openness_scale_measure.cpp \
  -o artifacts/openness-scale/openness_scale_measure
./artifacts/openness-scale/openness_scale_measure \
  | tee artifacts/openness-scale/anti_artifact_evidence.json
echo exit:$?
```

**Seeds:** `11, 22, 33, 44`  
**Learn scales (live):** 41 / 82 / 164 cells · primary learn=82 · **exec=82**  
**Span helper:** `learning_scale_span_ratio` / `learning_scale_span_ok` in `anti_artifact.hpp`

### A1 protect_window (焊枪)

```bash
cmake --build build --target test_population_openness_protect_live -j4
./build/test_population_openness_protect_live
```

## Evidence paths

| Path | Role |
|---|---|
| `artifacts/openness-scale/anti_artifact_evidence.json` | per-scale dumps · span_ratio · dual · OOD · Prefer checks |
| `artifacts/openness-scale/openness_scale_measure.cpp` | measure tool |
| `tests/test_population_openness_scale_sweep.cpp` | unit: ≥3 · ≥4× · exec 82 |
| `include/kun/cellular/population/anti_artifact.hpp` | `learning_scale_span_*` (E1–E3 compat) |
| `docs/claims/openness-scale-sweep.md` | thin pack claim (non-E-tag) |
| `tests/test_population_openness_protect_live.cpp` | A1 live deme/ecology protect (焊枪) |
| `include/kun/cellular/population/deme.hpp` · `ecology_grid.hpp` | A1 NicheRegistry wire (焊枪) |

## E1 / E2 / E3 / this pack

| Tag / pack | Status |
|---|---|
| E1 `#14` | **closed** — unchanged |
| E2 `#16` | **closed** — unchanged |
| E3 `#17` | **closed** — unchanged |
| **openness-scale-sweep** | this knife — coverage/novelty language · **not E4** |

## Stream A readiness

**A0–A6 green on tip `9a6a7c5d`** — measure (`862dfa96`) + A1 protect (`9a6a7c5d`) co-land.  
**Hand 找茬喵** for Stream A gate. **Not** DONE until 找茬喵 PASS → 扳手 squash.

## Park / bans held

DouZero · GPU · PSRO · exec cell lift · 4th ABI · ADAS reopen · silent P1 · zoo/mutate · CartPole revert · champ rewrite · Bot-box private clone.
