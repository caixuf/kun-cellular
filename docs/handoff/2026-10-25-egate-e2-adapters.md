# Handoff — E-Gate E2 second-light (Prefer A adapters)

**Date:** 2026-10-25 Asia/Shanghai  
**Base tip:** `50431c48` / `50431c48d935828cf4c34298363c38853de481e8` (main `#15` ADAS A6; parent `#14` E1 @ `d684b038`)  
**Branch:** `feat/cellular-egate-e2-adapters`  
**Owners:** **焊枪** (live adapters + measure/CI) · **手术刀** not required (no ABI change) · **颜料盒** none  
**Claim:** `docs/claims/E2-egate-second-light.md` · tag **E2**

## E1 vs E2 label table

| Label | Knife | Tip / PR | Live cell-count | Tag |
|---|---|---|---|---|
| **E1** | first-light | `#14` @ `d684b038` (claim tip `bff1c509…`) | **82-only** live dump; 41/164 registered | **E1** |
| **E2** | second-light (this) | this knife off `50431c48…` | **live 41 / 82 / 164** | **E2** |

## §七 product bet

Success = honest **E2**: live multi-scale anti-artifact on CELLS axis.  
**Not** win-rate / DouZero / PSRO / strength / ADAS W/L. Phase4 **HOLD**. ADAS **PARK**.

## Four stages shipped

| Stage | Artifact |
|---|---|
| 1 Pre-register | `docs/claims/E2-egate-second-light.md` — tip_hash=`50431c48…`, BD, dual metrics, seeds≥4, live scales 41/82/164, primary 82 |
| 2 Anti-artifact | dual · live ≥3 scales · ≥4 seeds · OOD+paired — evidence JSON cites per-scale runs |
| 3 Mechanism | Phase3 operators + **new** `make_org(seed,cell_count)` / measure adapters |
| 4 Archive-repro | this handoff |

## Reproduce

```bash
# on caixuf WSL: /home/caixuf/code/kun-cellular @ tip 50431c48… / branch feat/cellular-egate-e2-adapters
cd /home/caixuf/code/kun-cellular
git fetch origin && git checkout feat/cellular-egate-e2-adapters

# unit: builders at 41/164 produce org.cells.size()==N
cmake --build build --target test_population_egate_e2_adapters -j4
./build/test_population_egate_e2_adapters
# or: ctest --test-dir build -R test_population_egate_e2_adapters --output-on-failure

# measure dump → JSON (three live campaigns)
g++ -std=c++20 -O2 -Iinclude \
  artifacts/egate-e2-adapters/egate_e2_adapters_measure.cpp \
  -o artifacts/egate-e2-adapters/egate_e2_adapters_measure
./artifacts/egate-e2-adapters/egate_e2_adapters_measure \
  | tee artifacts/egate-e2-adapters/anti_artifact_evidence.json
```

**Seeds:** `11, 22, 33, 44`  
**Learn scales (live):** 41 / 82 / 164 cells · primary learn_scale=82  
**Claim path:** `docs/claims/E2-egate-second-light.md`

## Evidence paths

| Path | Role |
|---|---|
| `artifacts/egate-e2-adapters/anti_artifact_evidence.json` | per-scale live dumps · dual · OOD McNemar/paired · E-Gate verdict |
| `artifacts/egate-e2-adapters/egate_e2_adapters_measure.cpp` | measure tool (header-only Prefer-A reuse; parameterized builders) |
| `tests/test_population_egate_e2_adapters.cpp` | thin unit: cells.size()==N at 41/82/164 |

## A0–A8

| ID | Verdict |
|---|---|
| A0 | PASS — branch off `50431c48` clean fetch |
| A1 | PASS — live orgs at 41 and 164 (+82) |
| A2 | PASS — anti-artifact green on live multi-scale pack |
| A3 | PASS — new E2 claim tip_hash=`50431c48…`; E1 unchanged |
| A4 | PASS — this handoff + E1 vs E2 table + archive-repro |
| A5 | PASS — no DouZero/strength; exec≠learn honest per campaign |
| A6 | PASS — Phase4 HOLD; ADAS PARK; no silent P1; no 4th ABI |
| A7 | PASS — OOD not sole multi-scale proof (live cell counts per scale) |
| A8 | PASS — 焊枪 lands; no 手术刀 ABI escalate |

## Park / STOP

Phase4 **HOLD** · ADAS **PARK** · win/DouZero **parked** · no silent P1 · no 4th ABI · no kunautodrive · no Lyapunov-as-avalanche.
