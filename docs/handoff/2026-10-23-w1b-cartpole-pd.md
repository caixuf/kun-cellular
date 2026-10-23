# W1b CartPole vs Gate-6 named PD — Prefer A WIN

> **Asia/Shanghai 2026-10-23** · 焊枪 weld · Prefer A (task-layer retrain)  
> **Verdict: WIN** — `surv_cellular > surv_PD` on pack S (strict) **and** T4 floors held.  
> Lock `checkpoints/cartpole_balance_champion.bin` **not overwritten** (promote only post 找茬喵 + 扳手).

## Banner

Registered train budget (primary seed **20261023**) produced candidate that **strictly** beats frozen Gate-6 named PD on stress pack S while holding T4 ID/OOD floors. Optional retry seed 20261024 **not needed**.

## Base / branch

| Item | Value |
|---|---|
| Tip base | `2922599a833095f800135cbc5d6ce2c06ecc0a2b` |
| Branch | `feat/cellular-w1b-cartpole-pd` |
| Named PD | Gate-6 `pd_expert`: `u=clamp(12θ+2.5θ̇+1.2x+1.0ẋ,-1,1)` + same ± split |
| Start | **LOAD_CHAMP** `checkpoints/cartpole_balance_champion.bin` (13 cells / 49 syns) |
| pre_w1b backup | `checkpoints/cartpole_balance_champion_pre_w1b_20261023.bin` (md5 `8c71bcc33be2cdfaa39ad676a6ef916b`, byte-identical to lock) |
| Candidate | `checkpoints/cartpole_balance_champion_w1b_20261023.bin` (13 cells / 64 syns) |

## Registered budget (rails — exact CLI)

### Campaign 1 — primary (WIN)

```bash
g++ -O3 -march=native -std=c++20 -I include -I . \
  tools/train_cart_pole.cpp -o build/train_cart_pole -pthread -lm

./build/train_cart_pole \
  --seed 20261023 --pop 48 --gens 150 --max-steps 300 \
  --start checkpoints/cartpole_balance_champion.bin \
  --out checkpoints/cartpole_balance_champion_w1b_20261023.bin
```

- Wall ~32 s. Start mode documented: **LOAD_CHAMP** (not HANDCRAFTED).
- Fitness: ID (default, seeds 101–110) + mild OOD (`masspole=0.2 length=0.7 force_noise=2`, seeds 201–210) + **stress OOD pack S physics** (`force_noise=12`, seeds `9300+i*29` i=0..39). Weights 0.25/0.15/0.20 fitness + 0.40 stress success_rate.
- Train log: `runs/train_cartpole_w1b_20261023.log` (gitignored).

### Campaign 2 — optional retry

**Not run.** Primary already satisfies hard win. Budget not exhausted via failure.

## Shared cold eval harness

```bash
g++ -O3 -march=native -std=c++20 -I include -I . \
  tools/bench_cartpole_vs_pd.cpp -o build/bench_cartpole_vs_pd -pthread -lm

./build/bench_cartpole_vs_pd \
  --bin checkpoints/cartpole_balance_champion_w1b_20261023.bin \
  --out runs/cartpole_vs_pd_w1b_20261023.json
```

Regen JSON anytime (runs/ gitignored).

## Results table

| Slice | Params | Seeds | PD | Cellular (w1b_20261023) | Notes |
|---|---|---|---:|---:|---|
| T4 ID | default | `9000+i*17` N=20 ms=300 | 20/20 | **20/20** | floor ≥19 ✓ |
| T4 OOD | mp0.2 L0.7 noise=2 | `9100+i*19` N=20 ms=300 | 20/20 | **20/20** | floor ≥18 ✓ |
| **Pack S** | mp0.2 L0.7 noise=**12** | `9300+i*29` N=40 ms=300 | **26/40** | **28/40** | **strict gt ✓** |

Lock baseline (pre-train, same harness): Pack S cellular **20/40** vs PD **26/40** — Prefer A retrain closed the gap and overtook.

Soft 40/40 T4 alone ≠ win (A7); decisive metric is pack S.

## Acceptance A1–A10 (self-check)

| ID | Status |
|---|---|
| A1 Named PD byte-matches Gate-6 | Pass — copied `pd_expert` decode + clamp + ± split |
| A2 Shared harness T4+S both controllers | Pass — `tools/bench_cartpole_vs_pd.cpp` → JSON |
| A3 Pack S surv_cellular > surv_PD | Pass — **28 > 26** |
| A4 T4 floors | Pass — ID 20/20, OOD 20/20 |
| A5 Handoff lists actual train CLI | Pass — this file |
| A6 Ceiling | N/A (win) |
| A7 Tie/G6 shadow ≠ win | Observed — win is pack-S strict, not T4 tie |
| A8 Diff ⊆ touch list | Pass — no include/ADAS/zoo/paper/S2/PD retune |
| A9 Promote | **Not done**; pre_w1b backup present; lock bin unchanged |
| A10 Falsify STOP | N/A (win path) |

## Forbidden moves (not used)

ADAS P1 / tune_adas topo · DomainZoo · PD gain change · cherry seeds · paper/dist · S2 · mutate/apoptosis campaigns · TL/emily · promote without gate · silent W1c.

## Touch list

- `tools/train_cart_pole.cpp` — stress OOD in fitness; LOAD_CHAMP start; CLI budget locks
- `tools/bench_cartpole_vs_pd.cpp` — **new** shared harness
- `checkpoints/cartpole_balance_champion_w1b_20261023.bin`
- `checkpoints/cartpole_balance_champion_pre_w1b_20261023.bin`
- `docs/handoff/2026-10-23-w1b-cartpole-pd.md`
- `README.md` — CartPole claim row only
- `runs/cartpole_vs_pd_w1b_*.json` — gitignored; regen CLI above

## Next

找茬喵 A1–A10 → 扳手 squash merge → promote lock bin only after gate ACK.
