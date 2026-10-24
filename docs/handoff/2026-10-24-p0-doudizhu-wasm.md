# Handoff - Prefer #1 P0 DouDiZhu WASM single-source

**Date:** 2026-10-24 Asia/Shanghai
**Base:** `4f170246` (#8 evidence squash-merged; branch rebased)
**Branch:** `feat/cellular-p0-doudizhu-wasm`
**Owner:** hangqiang (functional cutover) · yanliao (polish TODO)

## Rails summary

- Kill JS rules in `frontend/doudizhu.html`; page = render + telemetry.
- Single source = `doudizhu_rules_core.hpp` aligned with `DouDiZhuCardGameTask` CandPlay / trick semantics (SOLO/PAIR/BOMB/ROCKET) + WASM exports.
- I1-I3 green **before** JS rules deleted from default path.
- 8k illegal probe <=0.1%; FE-harness agree; 82-cell / zero-alloc / deterministic held.
- Park #2/#3/#4; no DouZero/PSRO/ADAS P1/zoo/mutate/CartPole revert; no JS rules backup.

## Artifacts

| Path | Role |
|---|---|
| `tasks/transfer/doudizhu_rules_core.hpp` | Shared rules core (harness subset) |
| `tools/doudizhu_wasm_exports.cpp` | emscripten exports |
| `tools/build_doudizhu_wasm.sh` | WASM build |
| `frontend/wasm/doudizhu_rules.{js,wasm}` | Built artifact |
| `frontend/doudizhu_wasm_glue.js` | Thin JS->WASM only |
| `frontend/doudizhu.html` | Cutover: wrappers + AI via legal_actions |
| `tests/test_doudizhu_invariants.cpp` | I1-I3 |
| `tools/bench_doudizhu_wasm_probe.mjs` | 8k illegal + agree |

## Commands

```bash
cmake --build build --target test_doudizhu_invariants -j && ./build/test_doudizhu_invariants
bash tools/build_doudizhu_wasm.sh
node tools/bench_doudizhu_wasm_probe.mjs 20261024 8000
```

## Notes / TODO(yanliao)

- FE trick surface now matches harness (SOLO/PAIR/BOMB/ROCKET only).
- Telemetry HUD may need RESHIP polish mapping `dd_telemetry_tick` meters.
- Organism forward still via `/api/doudizhu/cortex` when server present; WASM tick is deterministic HUD stub (82 marker).

## Gate results (2026-10-24 Asia/Shanghai)

| Gate | Result |
|---|---|
| BASE | `4f170246` (#8 evidence squash-merged; branch rebased) |
| I1–I3 | PASS (`./build/test_doudizhu_invariants`) |
| EMCC | 6.0.13 → `frontend/wasm/doudizhu_rules.{js,wasm}` |
| ILLEGAL_8K | deals=107 plies=8000 illegal=0 rate=0 (<=0.1%) seed=20261024 |
| AGREE | disagree_n=0 pass_p1=true pass_p2=true |
| FE cutover | no JS rules fallback; harness subset SOLO/PAIR/BOMB/ROCKET |

```json
{"seed":20261024,"N_target":8000,"deals":107,"plies":8000,"illegal":0,"illegal_rate":0,"sum_mismatch":0,"disagree_n":0,"pass_p1":true,"pass_p2":true}
```
