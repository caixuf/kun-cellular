# Handoff — Prefer Openness deepen² · `avalanche_164_cap_honesty_v1`

**Date:** 2026-10-25 Asia/Shanghai  
**Owner:** 手术刀 (measure)  
**Branch:** `feat/cellular-openness-deepen2` off tip **`bbcee73f`** / `bbcee73fa2f3e057d6e54787ae88be8bc151c330`  
**Knife LOCKED:** named residual `avalanche_164_cap_honesty_v1` (NOT uncap · NOT scale invent)  
**Rails:** `/workspace/contracts/kun-cellular-openness-deepen2-rails.md` (PASS)  
**REQ:** `/workspace/kun-cellular-docs/REQ-kun-cellular-openness-deepen2.md`  
**焊枪:** NOT needed — thin measure assert + cheap unit contract; no ABI invent / no uncap.

## Gap closed

Prior `#23` deepen evidence already printed per_scale[164]:

```json
"avalanche_sites": 128,
"avalanche_sites_capped_at_128": true
```

but `anti_artifact` verdict did **not** assert this — pass path could greenwash if the field were omitted. tip_hash was still stamped `687cadf5…` (stale vs deepen² tip).

**Fix (thin):**
- After campaigns in `artifacts/openness-scale/openness_scale_measure.cpp`, find scale `cells==164`; require `avalanche_capped==true` AND `avalanche_sites==kAvalancheMaxSites` (128).
- Fold into anti_artifact JSON: `avalanche_164_cap_ok`, `avalanche_164_sites`, `avalanche_164_capped_at_128`.
- `anti_artifact.pass` **FAILS** if scale-164 cap honesty missing/false (no greenwash); reason string updated.
- Stamp top-level `tip_hash` = `bbcee73fa2f3e057d6e54787ae88be8bc151c330`.
- Keep `e_gate_echo="none"` · pack_id `openness-scale-sweep` · `e_tags_closed` · `honesty.not_e4`.
- prefer_checks: A0=`feat/cellular-openness-deepen2 off bbcee73f`; A5=this handoff.
- Cheap unit contract in `tests/test_population_openness_scale_sweep.cpp`: `kAvalancheMaxSites==128` · 164→sites=128·capped=true (no evaluate_anti_artifact ABI churn).

## Explicit NO-uncap

- Cap remains **128** (`kAvalancheMaxSites` untouched).
- Scale set remains **41/82/164** — no invent 328.
- Companion metric honesty only; uncapping stays PARK / forbidden this Prefer.

## Re-run @ tip `bbcee73f` (once)

```bash
cd /home/caixuf/code/kun-cellular
cmake --build build --target test_population_openness_scale_sweep -j4
./build/test_population_openness_scale_sweep   # exit 0
g++ -std=c++20 -O2 -Iinclude \
  artifacts/openness-scale/openness_scale_measure.cpp \
  -o artifacts/openness-scale/openness_scale_measure
./artifacts/openness-scale/openness_scale_measure \
  | tee artifacts/openness-scale/anti_artifact_evidence.json
# MEASURE_EXIT:0
```

### Proof (evidence JSON)

| Field | Value |
|---|---|
| `pack_id` | `openness-scale-sweep` |
| `tip_hash` | `bbcee73fa2f3e057d6e54787ae88be8bc151c330` |
| `e_tags_closed` | `["E1","E2","E3"]` |
| `e_gate_echo` | `"none"` |
| `e_gate.tag` | **ABSENT** (no stage stamp; no E2) |
| `anti_artifact.pass` | `true` |
| `anti_artifact.avalanche_164_cap_ok` | `true` |
| `anti_artifact.avalanche_164_sites` | `128` |
| `anti_artifact.avalanche_164_capped_at_128` | `true` |
| `span_ratio` | `4.0` |
| `learn_scales_live` | `[41, 82, 164]` |
| `seeds` | `[11, 22, 33, 44]` · OOD `seed_shift_+1000` paired |
| `exec_cells` | `82` held |
| `honesty.not_e4` | `true` |
| strength / DouZero / win-rate as claims | **none** |
| unit exit | `0` (incl. avalanche_164_cap_honesty contract) |
| measure exit | `0` |

## Files touched

| Path | Role |
|---|---|
| `artifacts/openness-scale/openness_scale_measure.cpp` | Thin avalanche@164 cap honesty assert + tip_hash refresh |
| `artifacts/openness-scale/anti_artifact_evidence.json` | Re-run @ tip `bbcee73f` |
| `tests/test_population_openness_scale_sweep.cpp` | Cheap unit contract (no ABI churn) |
| `docs/handoff/2026-10-25-openness-deepen2-avalanche-cap.md` | This archive |

**Not touched:** protect retune · champ · ADAS · KAD deepen · invent search · kunautodrive · extra scale · avalanche uncapping · anti_artifact.hpp ABI · exec lift.

## PARK / bans held

Cellular ADAS reopen · silent P1 · invent search · invent-scenario · further KAD deepen · avalanche uncap · DouZero/GPU/PSRO · invent E4 · Bot-box clone · exec lift · champ touch · strength / win-rate.

## D2-0–D2-7 self-check

| ID | Check | Result |
|---|---|---|
| D2-0 | Branch off `bbcee73f`; WSL caixuf only | PASS — `feat/cellular-openness-deepen2` @ `bbcee73f…` |
| D2-1 | Named residual `avalanche_164_cap_honesty_v1` closed as anti-artifact check — cap labeled, **not** uncapped | PASS — `avalanche_164_cap_ok=true` · sites=128 · capped=true |
| D2-2 | Re-run: dual · scales 41/82/164 · OOD+paired · tip_hash `bbcee73f…` | PASS |
| D2-3 | Labels honest: coverage/novelty · pack `openness-scale-sweep` · e_gate_echo=none · no strength · not E4 | PASS |
| D2-4 | Exec 82 held; learn ≠ exec lift; champ untouched | PASS |
| D2-5 | Cellular ADAS PARK · silent P1 PARK · invent search PARK · invent-scenario PARK · further KAD deepen PARK | PASS |
| D2-6 | No DouZero/GPU/PSRO / 4th ABI / zoo-mutate / Phase4 deepen / extra scale invent / avalanche uncap | PASS |
| D2-7 | Owner 手术刀 (+焊枪 gap); 找茬喵 gate | PASS — 焊枪 not needed; hand to 找茬喵 (no DONE/squash declare) |

## Next

Gate-ready for **找茬喵**. Do **not** squash from this seat. Parent hands review.
