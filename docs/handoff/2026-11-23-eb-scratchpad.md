# Handoff — E-B 串行 scratchpad 因果验证（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §2 E-B
> 结论：**H2 PASS**。外部串行记忆是延迟回忆的**因果必要条件**；但不是“通用推理”，且 1 通道已足够。

---

## 1. 证据格式

```text
PHASE: E-B
COMMIT: 90f2a65
COMMAND: KUN_COMMIT=90f2a65 ./build/bench_scratchpad_cot --gens 160 --pop 24 --seeds 4 --eval-episodes 100 --out runs/scratchpad_cot_20261123.json
ARTIFACT: runs/scratchpad_cot_20261123.json
GATE: H2 PASS
KEY NUMBERS: D=16: none 0.550 | 8ch 1.000 | random 0.5275 | delta 0.450 | b=0 c=180 p=1.305e-54 | 4 seeds
CONTROLS: random-feedback 0.5275（chance）；no-scratchpad 0.550（chance）
CONCLUSION: 外部 scratchpad 是延迟回忆的因果必要条件（E2，窄口径）；不构成通用推理/CoT 能力
NEXT: E-C（器官组合）
```

---

## 2. 任务：Key-Door Memory（延迟线索 + 二选一门）

| 阶段 | tick | 观测 |
|---|---|---|
| 线索（钥匙出现） | t=0 | 左/右，只出现一拍 |
| 延迟（走向门） | t=1..D | 无线索，只有 bias |
| 响应（选门） | t=D+1 | response_phase=1，必须凭记忆选左/右 |

- 成功 = 响应期选对与线索同侧的门。
- 训练延迟 D=16；评估 D=2/8/16/32（D=16 是预注册判据，其余为探索）。
- 前馈策略在 t=D+1 看不到线索，**数学上不可能稳定答对**；只有跨拍状态能解。

---

## 3. 架构：无状态细胞 + 唯一外部反馈环

22 个细胞：

| 组 | 数量 | 类型 | 作用 |
|---|---:|---|---|
| 外部观测受体 | 4 | `SENSE_RAW_INPUT_0..3` | 线索左/右、响应期、bias |
| scratchpad 受体 | 8 | `SENSE_CHANNEL` param2=4..11 | 读上一拍记忆输出 |
| 记忆细胞 | 8 | `OP_SUM`（**无内部状态**） | 当前 obs + 反馈的线性组合 |
| 动作细胞 | 2 | `ACT_PRIMARY_POSITIVE/NEGATIVE` | 左/右门 |

关键设计：

- **没有 memory→memory 同拍突触**；
- **没有 EMA/积分等内部持留**；
- 唯一跨拍记忆是任务层 `prev_mem → inputs[4..11]` 的外部反馈环。

因此：

- `channels=0` 是**真前馈**（数学上失忆）；
- `channels>0` 才有跨拍记忆；
- `random_feedback` 同分布但无信息，是反伪影对照。

训练：固定拓扑 + (1+λ) ES，只调突触权重/细胞参数；每个条件同预算、同 seed。
（这是 E-B 的最小闭环，不是 BPTT/GRPO 的最终方案。）

---

## 4. 结果（4 seeds × 100 held-out episodes）

### D=16（预注册判据）

| 条件 | 平均 SR | 说明 |
|---|---:|---|
| no_scratchpad | 0.550 | 真前馈，chance |
| scratch_1ch | **1.000** | 1 通道足够 |
| scratch_2ch | **1.000** | — |
| scratch_4ch | **1.000** | — |
| scratch_8ch | **1.000** | 8 通道不更好 |
| scratch_8ch_random | 0.5275 | 随机反馈 = chance |

- Δ(8ch − none) = **0.450**；
- McNemar：b=0（none 对、full 错），c=180（none 错、full 对），**p=1.305e-54**；
- random-feedback 与 none 都在 chance，说明增益不是“多几个输入”或“额外参数”。

### 延迟扫描（探索，非门禁）

| 条件 | D=2 | D=8 | D=16 | D=32 |
|---|---:|---:|---:|---:|
| no_scratchpad | 0.550 | 0.550 | 0.550 | 0.550 |
| scratch_1ch | 0.480 | 0.745 | **1.000** | **1.000** |
| scratch_2ch | 0.620 | 0.745 | **1.000** | 0.765 |
| scratch_4ch | 0.605 | 0.495 | **1.000** | 0.590 |
| scratch_8ch | 0.380 | 0.590 | **1.000** | 0.375 |
| random | 0.5175 | 0.5375 | 0.5275 | 0.515 |

诚实解读：

1. **策略是延迟专用的**：只在 D=16 训练，D=2/8 没有学会；不能宣称“通用延迟记忆”。
2. **1 通道已足够**：二值线索只需 1 bit；8 通道不更好，甚至在 D=32 OOD 上更差。
3. **不是 CoT**：这是一步外部记忆环，不是多步串行推理；只能支撑“串行草稿纸是因果必要的”这一窄结论。

---

## 5. 判定

| 判据 | 结果 |
|---|---|
| H2：有/无 ΔSR ≥15pp | ✅ **+45pp** |
| 消融回落 ≥10pp | ✅ 回落 45pp |
| 随机反馈对照 | ✅ 0.5275（chance） |
| E-Gate：≥4 seeds | ✅ 4 seeds |
| 机制可解释 | ✅ 唯一外部反馈环；无内部持留 |
| 可复现 | ✅ commit + 命令 + JSON |
| E 标签 | **E2（窄口径）**：外部 scratchpad 对延迟回忆因果必要 |

**不是 E2 的部分**：通用推理、CoT、跨延迟泛化、AGI。这些都没有证据。

---

## 6. 成本

| 项 | 值 |
|---|---|
| 训练 | 4 seeds × 6 条件 × 160 代 × 24 pop × 16 episodes |
| 评估 | 4 seeds × 6 条件 × 4 delays × 100 episodes |
| 本机墙钟 | 约 1~2 分钟（单线程 C++） |
| 参数量 | ~22 细胞 / ~200 突触，ES 维度 ~250 |

---

## 7. 下一步：E-C

E-B 证明的是“外部记忆环有用”，不是“组合产生新能力”。按计划进入 E-C：

1. 4 个冻结器官：延迟、比较、门控、振荡；
2. 组合任务：需要 ≥2 个器官串联；
3. 四组对照：全组合 / 留一 / 单器官 / 同细胞数单体；
4. 判据：全组合 > 最佳单器官 +25pp，留一掉 ≥20pp，且 > 同预算单体 +10pp。

E-B 的 `OP_SUM + SENSE_CHANNEL` 可直接作为 E-C 的器官积木。

---

## 8. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_scratchpad_cot
KUN_COMMIT=$(git rev-parse --short HEAD) \
  ./build/bench_scratchpad_cot --gens 160 --pop 24 --seeds 4 --eval-episodes 100 \
  --out runs/scratchpad_cot_20261123.json

ctest --test-dir build -R test_scratchpad_cot --output-on-failure
```
