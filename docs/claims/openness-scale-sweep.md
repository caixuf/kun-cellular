# Openness scale-sweep pack (Stream A · non-E-tag)

**Pack id:** `openness-scale-sweep`  
**Date:** 2026-10-25 Asia/Shanghai  
**Branch:** `feat/cellular-openness-scale` off `741c214c`  
**Owner:** 手术刀 (measure / span / handoff) · A1 protect_window = 焊枪  
**Language:** openness / coverage / novelty — **NOT** strength / DouZero / win-rate  
**Not a 4th E-tag:** E1 / E2 / E3 remain closed; this is a Prefer-A scale-sweep evidence pack.

## Pre-register

| Field | Value |
|---|---|
| pack_id | `openness-scale-sweep` |
| BD | `coverage_frac × novelty_knn` |
| dual | `coverage_frac` + `novelty_knn` |
| seeds | `11, 22, 33, 44` |
| learn_scales | **live** 41 / 82 / 164 |
| span_ratio | **4.0** (164/41) via `learning_scale_span_ok` |
| learn_scale_primary | 82 |
| exec_scale | **82** (`kPreferAOrganismCells`) |
| learn ≠ exec lift | adapters measure 41/164; ABI exec stays 82 |
| stop_rule | max_gens=0 scale-sweep; span≥4; exec=82; E1–E3 closed |
| mechanism | MAP-Elites + novelty archive + avalanche sized per campaign + span helper |

## Anti-artifact

| Check | Evidence |
|---|---|
| Dual independent | coverage_frac · novelty_knn |
| ≥3 live scales | 41, 82, 164 |
| Span ≥4× | `learning_scale_span_ratio` == 4.0 |
| ≥4 seeds | 11–44 |
| OOD + paired/McNemar | `seed_shift_+1000` |
| Paths | `artifacts/openness-scale/anti_artifact_evidence.json` · measure cpp · unit test |

## Honesty

Forbidden: DouZero · win-rate · strength · bare “emerged” · Phase4 reopen · ADAS reopen · 4th ABI · silent exec lift · silent P1 · zoo/mutate.  
A1 protect_window live deme path is **焊枪** — not claimed by this pack.
