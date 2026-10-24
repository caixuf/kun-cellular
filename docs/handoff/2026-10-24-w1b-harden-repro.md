# W1b Harden Prefer A repro — HARDEN_FAIL (falsify STOP)

> **Asia/Shanghai 2026-10-24** · 手术刀 weld · Prefer A Harden repro  
> **Verdict: HARDEN_FAIL** — frozen champ clears F0 strict + F2 + T4, **misses F1 non-lose**.  
> **Falsify → STOP.** No ADAS P1 · no zoo/mutate · no Prefer A CartPole revert · no retrain · no PD retune.  
> Champ bin **untouched**. Tip base `b510848` lineage only.

## Banner

Multi-seed Pack S cold eval on **frozen** `checkpoints/cartpole_balance_champion.bin` (md5 `ea8e68269b24ed578070738a99441d83`). F0 registered Prefer A win **reproduced** (28 > 26). Holdout **F1** fails: cellular **33/40** < PD **34/40**. Prefer A Harden = **FAIL**. Report 扳手 → 北极星; Design standby only.

## Base / branch

| Item | Value |
|---|---|
| Tip base | `b510848d4a2d16bad473ae6df60c228910319c23` |
| Branch | `feat/cellular-w1b-harden-repro` |
| Champ path | `checkpoints/cartpole_balance_champion.bin` |
| Champ md5 | **`ea8e68269b24ed578070738a99441d83`** (A1 PASS — byte-frozen, no rewrite) |
| Named PD | Gate-6 `pd_expert`: `u=clamp(12θ+2.5θ̇+1.2x+1.0ẋ,-1,1)` + same ± split (untouched) |
| Cells / syns | 13 / 64 |

## Pack S protocol (byte-match W1b)

| Field | Lock |
|---|---|
| Params | `masspole=0.2`, `length=0.7`, `force_noise=12.0` |
| MAX_STEPS | 300 |
| Metric | Survival count (`steps >= MAX_STEPS`) |
| N | 40 independent seeds per family |

## CLI (regen)

```bash
g++ -O3 -march=native -std=c++20 -I include -I . \
  tools/bench_cartpole_vs_pd_harden.cpp \
  -o build/bench_cartpole_vs_pd_harden -pthread -lm

mkdir -p runs
./build/bench_cartpole_vs_pd_harden \
  --bin checkpoints/cartpole_balance_champion.bin \
  --out runs/cartpole_vs_pd_harden.json
```

Exit non-zero on A1/A4–A7 miss. JSON under `runs/` is gitignored.

## Results table (local self-test · Asia/Shanghai 2026-10-24)

| Slice | Params | Seeds | PD | Cellular | Gate | Status |
|---|---|---|---:|---:|---|---|
| T4 ID | default (`force_noise=0`) | `9000+i*17` N=20 ms=300 | 20/20 | **20/20** | ≥19/20 | **PASS** |
| T4 OOD | mp0.2 L0.7 noise=2 | `9100+i*19` N=20 ms=300 | 20/20 | **20/20** | ≥18/20 | **PASS** |
| **F0** Pack S | mp0.2 L0.7 noise=**12** | `9300+i*29` N=40 | **26/40** | **28/40** | `surv_c > surv_PD` strict | **PASS** |
| **F1** Pack S | mp0.2 L0.7 noise=**12** | `9600+i*41` N=40 | **34/40** | **33/40** | `surv_c ≥ surv_PD` | **FAIL** (33 < 34) |
| **F2** Pack S | mp0.2 L0.7 noise=**12** | `9700+i*43` N=40 | **34/40** | **35/40** | `surv_c ≥ surv_PD` | **PASS** |

Mean steps (informational): F0 PD 248.9 / cell 261.9 · F1 PD 273.4 / cell 272.6 · F2 PD 288.4 / cell 290.2.

## md5 assert

```
ea8e68269b24ed578070738a99441d83  checkpoints/cartpole_balance_champion.bin
```

Matches frozen lock. Harness + CI both fail closed on drift.

## Acceptance A1–A11 (self-check)

| ID | Check | Status |
|---|---|---|
| A1 | Champ md5 frozen; tip/champ untouched | **PASS** |
| A2 | Named PD Gate-6 formula unchanged | **PASS** (review) |
| A3 | Pack S physics copy mp0.2 L0.7 noise=12 ms=300 N=40 | **PASS** |
| A4 | F0 strict `surv_c > surv_PD` | **PASS** (28 > 26) |
| A5 | F1 `surv_c ≥ surv_PD` | **FAIL** (33 < 34) |
| A6 | F2 `surv_c ≥ surv_PD` | **PASS** (35 ≥ 34) |
| A7 | T4 ID≥19 OOD≥18 | **PASS** (20/20, 20/20) |
| A8 | CI fails closed on A1/A4–A7 | **PASS** (job `w1b-harden-repro`; local exit 1) |
| A9 | Handoff tables + CLI + md5 + tip `b510848` | **PASS** (this file) |
| A10 | Diff ⊆ touch list; hard bans held | **PASS** |
| A11 | Falsify STOP, no silent next Prefer | **PASS** — **STOP here** |

## Falsify STOP

Frozen champ **cannot** clear F0+F1+F2+T4 without bin rewrite. Prefer A Harden = **FAIL**.

**Do not:** unlock ADAS P1 · zoo/mutate · Prefer A CartPole revert · retrain · PD retune · seed-list swap.

Report → 扳手 → 北极星. Design standby only.

## CI

Job `w1b-harden-repro` in `.github/workflows/ci.yml`:

1. `md5sum` assert vs `ea8e68269b24ed578070738a99441d83`
2. `g++` build `tools/bench_cartpole_vs_pd_harden.cpp`
3. Run bench; non-zero exit fails the job (A1/A4–A7)

Existing `build-and-test` job unchanged.

## Touch list (this knife)

- `tools/bench_cartpole_vs_pd_harden.cpp` — **new** thin sibling (F0/F1/F2 + T4 + md5 + fail-closed)
- `.github/workflows/ci.yml` — job `w1b-harden-repro`
- `docs/handoff/2026-10-24-w1b-harden-repro.md` — this dossier
- `runs/cartpole_vs_pd_harden.json` — gitignored regen artifact

**Not touched:** champ.bin · `include/kun/cellular/**` · `train_cart_pole` retrain · ADAS/zoo · PD gains · original `bench_cartpole_vs_pd.cpp` (W1b F0-only harness kept)

## Forbidden moves (not used)

ADAS P1 · DomainZoo · mutate/apoptosis · PD retune · champ rewrite · seed-list swap · Prefer A CartPole revert · LIFEFORM_SCALES/stadium as blocker.
