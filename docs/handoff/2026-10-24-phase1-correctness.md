# Handoff — Phase 1 Correctness lock

**Date:** 2026-10-24 Asia/Shanghai  
**Base:** `3aeb3eb` (`#9` P0 WASM)  
**Branch:** `feat/cellular-phase1-correctness`  
**Owners:** 焊枪 (CONTRACT v1 + CI gate) · 颜料盒 (FE inventory / residual cutover)

## A1 — FE dual-impl inventory (Design)

| Page | Local deal/play/legal/score/physics oracle? | Single-source / verdict |
|---|---|---|
| `frontend/doudizhu.html` | Rules via WASM wrappers only (`DouDiZhuWasm.*`); JS Fisher-Yates still builds display hands | **Done** `#9` — WASM rules. Dual-deal display residual = **NON-BLOCK** (rails) |
| `frontend/doudizhu_wasm_glue.js` | Thin emscripten calls only | OK — no local oracle |
| `frontend/maze.html` | No | **fetch-only** `/api/maze/*` — render + telemetry |
| `frontend/vehicle.html` | No rules/physics oracle; stadium projection is viz geometry; trail from backend only | **fetch-only** `/api/vehicle/*` — stadium flash **NON-BLOCK** |
| `frontend/slingshot.html` | No | **fetch-only** `/api/slingshot/*` |
| `frontend/locomotion.html` | No | **fetch-only** `/api/loco/*` |
| `frontend/ecosystem.html` | No | Demo habitat **viz-only** `/api/eco/*` |
| `frontend/immune.html` | No | Demo habitat **viz-only** `/api/immune/*` |
| `frontend/music.html` | Local WebAudio synth + optional SDSC bin load for viz; no deal/play/legal/score vs `tasks/**` | Demo habitat **viz-only** — leave |
| `frontend/cellular.html` + `frontend/cellular/*` | Worker `forward`/`stepPhysics` = organism **viz** sim (not eval-harness game rules) | Observatory viz — no competing game oracle; LIFEFORM_SCALES **NON-BLOCK** |
| `frontend/index.html` | Nav / shelf only | No oracle |

**CartPole / ADAS:** task exists under `tasks/**`; no dedicated FE page with a competing JS physics/rules oracle. Vehicle page is fetch-backed.

**Residual cutover this knife:** **none blocking.** No JS rules backup reintroduced. Dual-deal / stadium / LIFEFORM_SCALES / trick subset stay **NON-BLOCK** per rails.

## A2 — Default paths

Surfaced game with local legality = DouDiZhu only → WASM. Others fetch-only or demo viz.

## A3–A6 / A8–A10 — Weld (焊枪)

### CONTRACT v1
- Header: `tasks/transfer/doudizhu_rules_core.hpp` (`kContractId = doudizhu-rules-CONTRACT-v1`)
- Ops frozen: **deal** `deal_counts` / `assign_landlord` · **play** `apply_play` · **legal** `enumerate_candidates(_n)` · **score** `terminal_score` (+200/−200)
- Harness align: `doudizhu_eval_harness.hpp` uses `doudizhu_rules::terminal_score(success)` (4 sites); Prefer bounce required to reshape
- Drift lock: `tests/test_doudizhu_invariants.cpp` → `test_contract_v1_frozen` (+ I1–I3)

### Required CI (NOT continue-on-error)
| Job / step | Required? | Role |
|---|---|---|
| `build-and-test` → CTest (includes `test_doudizhu_invariants`) | yes | I1–I3 + CONTRACT v1 |
| `build-and-test` → **Assert DouDiZhu I1-I3 required gate** | yes | explicit fail if binary missing/red |
| `doudizhu-wasm-probe` | yes | 8k illegal+agree; fail if illegal_rate > 0.1% or disagree |
| `w1b-harden-repro` | no (`continue-on-error: true`) | evidence-only as today |

Other surfaced games: fetch-only / demo viz — **I* skip documented** (A1 table); no frozen local rule surface.

### Local proof (2026-10-24 Asia/Shanghai, caixuf WSL)

```
CI_PROOF: build-and-test + doudizhu-wasm-probe required; w1b-harden-repro non-blocking
CONTRACT_HEADER: tasks/transfer/doudizhu_rules_core.hpp CONTRACT v1 (+ harness align)
ILLEGAL_8K: seed=20261024 N=8000 deals=107 plies=8000 illegal=0 rate=0
AGREE: disagree_n=0 sum_mismatch=0 pass_p1=true pass_p2=true
I1_I3: PASS + CONTRACT v1 frozen
A8: 82-cell / zero-alloc / deterministic — unchanged from #9 WASM path (no organism path edit this knife)
A7: Phase 2/3/4 not started; hard bans held
```

```json
{"seed":20261024,"N_target":8000,"deals":107,"plies":8000,"illegal":0,"illegal_rate":0,"sum_mismatch":0,"disagree_n":0,"pass_p1":true,"pass_p2":true}
```

### YANLIAO_PAGES
None blocking. Dual-deal display residual on `doudizhu.html` remains NON-BLOCK (optional polish only — not this weld).

## NON-BLOCK (explicit)

LIFEFORM_SCALES · stadium flash · dual-deal · SOLO-PAIR-BOMB-ROCKET subset

## Park

Phase 2/3/4 · win knives STOP
