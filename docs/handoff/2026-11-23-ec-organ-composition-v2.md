# Handoff — E-C v2 冻结器官组合（4-way cue，2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §2 E-C v2
> 结论：**H3-v2 PASS**。4-way cue 下，冻结器官组合产生超加性能力；LOO 与单体对照都在 chance。
> 标签：**E2（窄口径）**——只对“该任务上的组合超加性”成立，不是通用组合智能。

---

## 1. 证据格式

```text
PHASE: E-C v2
COMMIT: fabcc56
COMMAND: KUN_COMMIT=fabcc56 ./build/bench_organ_composition_v2 --gens 120 --pop 24 --seeds 5 --eval-episodes 200 --out runs/organ_composition_v2_20261123.json
ARTIFACT: runs/organ_composition_v2_20261123.json
GATE: H3-v2 PASS
KEY NUMBERS: full 100% | best single 62.9% (single_event) | Δ=37.1pp | LOO 49.3% | monolith 59.9%
CONDITIONAL: pulse 子集 full 100% vs best single 26.8% (Δ=73.2pp, p=4.16e-112)
CONTROLS: LOO drop 50.7pp; monolith drop 40.1pp (p=3.87e-121); 5 seeds × 200 episodes
CONCLUSION: 四选一组合超加性成立（E2 窄口径）；不是通用组合智能
NEXT: E-D（best-of-N）或 S1（1M 适配器）
```

---

## 2. 为什么有 v2

v1 是二选一：

- `single_event` 靠“有 pulse 就猜边”天然拿 50%；
- best single = 50% + 50%×51.9% = 75.6%；
- full 已满分 100%，**+25pp 数学上不可达**（最大 24.4pp）。

v2 改成 **4-way cue**（左/右/上/下），chance=25%：

- best single 上限 ≈ 62.5%；
- full 100%；
- Δ 上限 37.5pp，+25pp 可达。

**v1 的 FAIL 保留，不覆盖、不改阈值。**

---

## 3. 任务与架构

### 任务：Conditional 4-Way Delayed Response

| 阶段 | tick | 观测 |
|---|---|---|
| 线索 | t=0 | cue ∈ {0,1,2,3} one-hot，只出现一拍 |
| 事件 | t=2..6 | 随机一拍 pulse（50% 无 pulse） |
| 响应 | t=10 | 有 pulse：argmax == cue 且 max action ≥0.2；无 pulse：max action <1e-6 |

### 器官（内部参数冻结）

| 器官 | 细胞 | 类型 | 作用 |
|---|---|---|---|
| delay | 4 | `OP_EMA` param1=0.05 | 每 cue 一个，10 拍后约 0.6 |
| event | 1 | `OP_INTEGRAL` | 锁存 pulse |
| gate | 4 | `OP_MULTIPLY` | gate_i = delay_i × event |
| readout | 4 | `ACT_CHANNEL` param2=0..3 | 四选一 |

只训练器官间/读出权重（12 个连接），器官内部参数冻结。
LOO/single 消融在每次 `apply_genes` 后强制置零并重新编译。

---

## 4. 结果（5 seeds × 200 held-out episodes）

| 条件 | 总体 SR | no-pulse | pulse |
|---|---:|---:|---:|
| **full** | **1.000** | **100%** | **100%** |
| loo_delay | 0.493 | 100% | 0% |
| loo_event | 0.493 | 100% | 0% |
| loo_gate | 0.493 | 100% | 0% |
| single_delay | 0.498 | 0% | ~99.6% |
| **single_event** | **0.629** | **100%** | **26.8%** |
| single_gate | 0.493 | 100% | 0% |
| monolith | 0.599 | ~60% | ~60% |

### 判据

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---:|---|
| full − best single | ≥25pp | **+37.1pp** | ✅ |
| full − min LOO | ≥20pp | **+50.7pp** | ✅ |
| full − monolith | ≥10pp | **+40.1pp** | ✅ |
| 配对显著性（vs best single） | p<0.05 | **4.16e-112** | ✅ |
| 配对显著性（vs monolith） | p<0.05 | **3.87e-121** | ✅ |

**H3-v2 PASS。**

---

## 5. 机制解读

- `single_delay`：能按 cue 回应，但不会抑制（no-pulse 0%）。
- `single_event`：会抑制，但不知道 cue，pulse 上只能 25% 猜（实测 26.8%）。
- `single_gate`：没有上游，只会 no-go。
- `loo_*`：断掉任一器官，只剩 no-go。
- `monolith`：同细胞数/同突触数，但打乱 op 类型 + 随机连接；能学到部分策略，但显著低于 full。
- **full 是唯一同时解决“何时回应”和“回哪边”的条件**。

这就是组合超加性：单独器官各自只能解决一半，合起来才完整。

---

## 6. 标签与边界

**可以说的（E2 窄口径）**：
- 在该 4-way 条件延迟任务上，冻结器官组合相对最佳单器官 +37.1pp；
- 相对同参数单体 +40.1pp；
- LOO 掉 50.7pp；
- 5 seeds、200 episodes、p<1e-111。

**不能说的**：
- ❌ “通用组合智能”；
- ❌ “任意任务都能组合”；
- ❌ “AGI/涌现”；
- ❌ 把 v2 的成功倒推成 v1 阈值正确（v1 仍 FAIL）。

---

## 7. 成本

| 项 | 值 |
|---|---|
| 训练 | 8 条件 × 5 seeds × 120 代 × 24 pop × 24 episodes |
| 评估 | 8 条件 × 5 seeds × 200 episodes |
| 本机墙钟 | 约 2~3 分钟（单线程 C++） |
| 参数 | 18 细胞；composition 只训练 12 个连接权重 |

---

## 8. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_organ_composition_v2
KUN_COMMIT=$(git rev-parse --short HEAD) \
  ./build/bench_organ_composition_v2 --gens 120 --pop 24 --seeds 5 --eval-episodes 200 \
  --out runs/organ_composition_v2_20261123.json

ctest --test-dir build -R test_organ_composition_v2 --output-on-failure
```
