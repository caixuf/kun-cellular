# Handoff — E-D Best-of-N + verifier（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §2 E-D
> 结论：**ED FAIL**。1 步 oracle verifier 的 Best-of-N 只带来 +2pp（N=16），远低于 +10pp；
> 21×21 迷宫甚至负增益。搜索不是免费午餐，myopic verifier 不够。

---

## 1. 证据格式

```text
PHASE: E-D
COMMIT: 87309ab
COMMAND: KUN_COMMIT=87309ab ./build/bench_best_of_n --n 1,4,16,64 --episodes 100 --map 11 --steps 250 --policy-noise 0.2 --cand-noise 0.2 --out runs/best_of_n_20261123.json
ARTIFACT: runs/best_of_n_20261123.json
GATE: ED FAIL (N=16 vs N=1 +2pp < +10pp; compute 16x)
KEY NUMBERS: N=1 84.0% | N=4 85.0% | N=16 86.0% | N=64 87.0%
CONTROLS: 21×21 压力测试 N=1 20% → N=16 17.5%（负增益）
CONCLUSION: 1 步 oracle verifier 不足；best-of-N 需要更长前瞻或更好的 verifier
NEXT: S1（1M 适配器）或 E-D v2（H>1 / learned verifier）
```

---

## 2. 配置

| 项 | 值 |
|---|---|
| 策略 | `checkpoints/maze_navigation_champion.bin`（11 细胞） |
| 迷宫 | 11×11，250 步；policy noise σ=0.2 制造 headroom |
| 候选 | 围绕策略动作的高斯扰动，σ=0.2；**候选 0 = 原始 base action** |
| verifier | 真实迷宫 1 步前瞻（oracle），按**测地势场距离**打分 |
| N | 1 / 4 / 16 / 64 |
| episodes | 100 held-out seeds |
| compute multiplier | N（每步 N 次模型推演） |

关键修正（第一版踩过的坑）：

1. 候选集必须包含 base action，否则 N 越大越容易被噪声带偏；
2. verifier 不能用 task reward（含“探索新格 +6”，会奖励乱逛）；
3. verifier 用测地距离，不用欧氏距离（欧氏会选“看着近但撞墙”的动作）。

---

## 3. 结果（11×11）

| N | SR | 平均步数 | 墙钟 | compute |
|---:|---:|---:|---:|---:|
| 1 | 84.0% | 181.6 | 10.3 ms | 1× |
| 4 | 85.0% | 181.4 | 54.6 ms | 4× |
| **16** | **86.0%** | 179.3 | 181.1 ms | **16×** |
| 64 | 87.0% | 177.5 | 676.5 ms | 64× |

- N=16 相对 N=1：**+2.0pp**，远低于 +10pp；
- N=64 也只有 +3.0pp，且 compute 64×；
- 增益随 N 递减，像是“在已经很强的策略上做微调”，不是能力跃迁。

## 4. 21×21 压力测试（次要）

| N | SR |
|---:|---:|
| 1 | 20.0% |
| 4 | 20.0% |
| 16 | 17.5% |
| 64 | 25.0% |

- 基策略在 21×21 很弱（20%），1 步 verifier 无法补；
- N=16 甚至负增益，说明 **myopic 搜索会把策略带偏**；
- N=64 略回升，但 64× compute 换 +5pp，不划算。

---

## 5. 判定

| 判据 | 结果 |
|---|---|
| N=16 − N=1 ≥ 10pp | ❌ +2.0pp |
| compute ≤ 20× | ✅ 16× |
| **ED** | ❌ **FAIL** |
| E 标签 | E1（描述性负结果） |

**不是 E-D 的搜索机制完全无效**，而是：

- 11×11 上策略已接近天花板（84%），没有 +10pp 的空间；
- 21×21 上策略太弱，1 步梯度救不了长程导航；
- oracle 1 步 verifier 是“局部贪心”，会被死胡同骗。

要拿到 +10pp，需要 **更长前瞻（H>1）** 或 **学习到的 verifier**；两者都会把 compute 推过 20×，超出 E-D 的预注册预算。所以按纪律 **FAIL + 停**，不在这里偷偷加预算。

---

## 6. 可说的 / 不能说的

**可以说的**：
- 1 步 oracle best-of-N 在 11×11 迷宫上只有 +2pp；
- 21×21 上 myopic 搜索会负增益；
- 搜索的收益取决于 verifier 质量与策略 headroom，不是 N 越大越好。

**不能说的**：
- ❌ “Best-of-N 能提升控制能力”；
- ❌ “test-time search 是免费增益”；
- ❌ “E-D PASS”。

---

## 7. 成本

| 项 | 值 |
|---|---|
| 11×11 正式 | 100 episodes × (1+4+16+64) × 250 步模型推演 |
| 21×21 压力 | 40 episodes × 同上 × 400 步 |
| 本机墙钟 | 11×11 约 1 秒；21×21 约 2 秒 |
| 结论 | 算力极便宜；问题是 verifier 质量，不是算力 |

---

## 8. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_best_of_n
KUN_COMMIT=$(git rev-parse --short HEAD) \
  ./build/bench_best_of_n --n 1,4,16,64 --episodes 100 --map 11 --steps 250 \
  --policy-noise 0.2 --cand-noise 0.2 --out runs/best_of_n_20261123.json

ctest --test-dir build -R test_best_of_n --output-on-failure
```
