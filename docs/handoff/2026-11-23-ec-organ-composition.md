# Handoff — E-C 冻结器官组合（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §2 E-C
> 结论：**H3 FAIL（差 0.6pp）**，但失败原因是**预注册阈值天花板**，不是组合无效。
> 组合在 pulse 子集上是 **100% vs 51.9%，Δ=48.1pp，p=7.07e-74**；按纪律标 E1，不写 E2。

---

## 1. 证据格式

```text
PHASE: E-C
COMMIT: 14da9cc
COMMAND: KUN_COMMIT=14da9cc ./build/bench_organ_composition --gens 120 --pop 24 --seeds 5 --eval-episodes 200 --out runs/organ_composition_20261123.json
ARTIFACT: runs/organ_composition_20261123.json
GATE: H3 FAIL (overall Δ=24.4pp < 25pp)
KEY NUMBERS: full 100% | best single 75.6% (single_event) | LOO 49.3% | monolith 50.7%
CONDITIONAL: pulse 子集 full 100% vs best single 51.9% (Δ=48.1pp, p=7.07e-74)
CONTROLS: LOO all 49.3%; monolith 50.7%; 5 seeds
CONCLUSION: 组合效应真实且大，但 overall +25pp 阈值在本任务二选一结构下不可达（E1）
NEXT: v2（4-way cue）重新预注册，或按合同 STOP
```

---

## 2. 任务：Conditional Delayed Response

| 阶段 | tick | 观测 |
|---|---|---|
| 线索 | t=0 | cue ∈ {left,right}，只出现一拍 |
| 事件 | t=2..6 | 随机一拍 pulse（50% 概率完全没有 pulse） |
| 响应 | t=10 | 有 pulse：选同侧门；无 pulse：no-go |

- 有 pulse 必须 **|diff| ≥ 0.2** 且符号正确；
- 无 pulse 必须 **|diff| < 1e-6**；
- 加幅度门禁是为了防止“1e-13 符号正确”的 epsilon 策略（第一版就抓到了这个漏洞）。

---

## 3. 四器官架构（冻结内部参数）

| 器官 | 细胞 | 类型 | 作用 |
|---|---|---|---|
| delay | D_left / D_right | `OP_EMA` param1=0.05 | 保持 cue；10 拍后约 0.6 |
| event | E | `OP_INTEGRAL` | 锁存 pulse |
| compare | C | `OP_SUB` | C = D_left − D_right |
| gate | G | `OP_MULTIPLY` | G = C × E |
| readout | Act+ / Act- | `ACT_PRIMARY_*` | G → 正/负动作 |

只训练器官间连接/读出权重（D→C、C→G、E→G、G→Act），
delay/event 的内部参数冻结。`LOO` 与 `single` 消融在每次 `apply_genes` 后强制置零并重新编译，
否则 ES 会把被消融的连接学回来（第一版踩过这个坑）。

对照：

- `full`：四器官全连；
- `loo_*`：分别切断 delay/event/compare/gate 输出；
- `single_*`：只把该器官输出接到读出（其余器官输出置零）；
- `monolith`：同细胞数/同突触数，但打乱 op 类型 + 随机连接，训练全部权重。

---

## 4. 结果（5 seeds × 200 held-out episodes）

### 4.1 总体

| 条件 | SR |
|---|---:|
| **full** | **1.000** |
| loo_delay | 0.493 |
| loo_event | 0.493 |
| loo_compare | 0.493 |
| loo_gate | 0.493 |
| single_delay | 0.507 |
| **single_event** | **0.756** |
| single_compare | 0.507 |
| single_gate | 0.493 |
| monolith | 0.507 |

### 4.2 拆分 pulse / no-pulse

| 条件 | no-pulse | pulse | 解读 |
|---|---:|---:|---|
| full | 100% | **100%** | 唯一两半都对的组合 |
| loo_* | 100% | 0% | 只会 no-go |
| single_delay | 0% | 100% | 只会按 cue 回应，不会抑制 |
| single_event | 100% | ~51.9% | 会抑制，但没 cue，只能猜边 |
| single_compare | 0% | 100% | 同 delay |
| single_gate | 100% | 0% | 同 no-go |
| monolith | 0% | 100% | 只会回应，不会抑制 |

**关键机制**：full 是唯一同时解决“何时回应（event/gate）”和“回哪边（delay/compare）”的条件。
其他条件都只解决一半。

---

## 5. 判定：H3 FAIL，但阈值有天花板问题

| 判据 | 结果 |
|---|---|
| 全组合 > 最佳单器官 **+25pp** | ❌ **+24.4pp**（100% − 75.6%），差 0.6pp |
| 留一掉 **≥20pp** | ✅ +50.7pp（100% − 49.3%） |
| 组合 > 同预算单体 **+10pp** | ✅ +49.3pp（100% − 50.7%） |

为什么 +25pp 不可达：

- 任务一半是 no-pulse，任何“会 no-go”的策略自动拿 50%；
- `single_event` 在 pulse 子集上靠 50% 猜边，pulse 准确率 51.9%；
- 于是 best single = 50% + 50%×51.9% = **75.6%**；
- full 已满分 100%，**最大可能 Δ = 24.4pp**，+25pp 在数学上不可达。

所以这是**预注册阈值设计错误**，不是组合失败。按纪律：

- **H3 = FAIL**，不写 PASS；
- **E 标签 = E1**，不写 E2；
- 不事后降低阈值。

### 可以说的

- 组合在 **pulse 子集**上：100% vs 51.9%，**Δ=48.1pp，p=7.07e-74**；
- LOO 和 monolith 都在 chance，说明不是“细胞多”或“参数多”；
- 组合效应是真实的，且机制清楚（何时回应 + 回哪边必须同时具备）。

### 不能说的

- ❌ “H3 PASS”；
- ❌ “组合涌现 E2”；
- ❌ “通用组合智能”。

---

## 6. 成本

| 项 | 值 |
|---|---|
| 训练 | 10 条件 × 5 seeds × 120 代 × 24 pop × 24 episodes |
| 评估 | 10 条件 × 5 seeds × 200 episodes |
| 本机墙钟 | 约 2~3 分钟（单线程 C++） |
| 参数 | 11 细胞；composition 只训练 7 个连接权重 |

---

## 7. v2 建议（重新预注册，不改 v1 阈值）

**问题**：二选一任务让 best single 天然有 50% 猜边，overall +25pp 不可达。

**v2**：4-way cue（左/右/上/下），4 个动作细胞，chance=25%。

- best single 上限 ≈ 50% + 50%×25% = 62.5%；
- full 100%；
- Δ 上限 = 37.5pp，+25pp 可达；
- 需要 4 个 delay 细胞 + 4 个 gate 细胞 + 4 个动作细胞；
- 其余协议（LOO、monolith、5 seeds、200 episodes、幅度门禁）不变。

**必须先写 v2 预注册，再跑**；不得看完 v1 数据后直接改阈值。

---

## 8. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_organ_composition
KUN_COMMIT=$(git rev-parse --short HEAD) \
  ./build/bench_organ_composition --gens 120 --pop 24 --seeds 5 --eval-episodes 200 \
  --out runs/organ_composition_20261123.json

ctest --test-dir build -R test_organ_composition --output-on-failure
```
