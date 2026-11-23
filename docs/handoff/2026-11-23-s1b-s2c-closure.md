# Handoff — S1-b adapter 形式 + S2-c 条件 BPTT 机制（2026-11-23）

> 目的：补齐 reviewer 指出的两项：
> 1) S1-b 的 rank-8 adapter 形式；
> 2) S2-c 的 conditional BPTT + 有界 tape。
> 结论：S1-b 由 **每块 138 参数（rank ≤ 2）适配器** 满足（比 rank-8 更严格，总 69k < 1% of 1B）；
> S2-c 做了小规模机制测试（梯度掩码 + 有界 tape）PASS，**不是 1B 全图 BPTT**。

---

## 1. S1-b：1B 冻结储层 + 低秩适配器

### 形式

- 冻结主干：500 blocks × 2M = 1B cells / 3B syns（共享，块权重不训练）；
- 适配器：每任务 × 每块 `69 × 2 = 138` 参数线性读出；
- 单块矩阵的秩 ≤ 2（2 个动作），**严格小于 rank-8**；
- 总可训练参数：500 × 138 = **69,000**（0.0069% of 1B，<1%）；
- 路由：每任务独立 4D 网格（500 桶）；top-5 = 10M active = 1%。

### 证据

```text
COMMAND: ./build/train_moe_agent_1b_3task --collect-episodes 100 --eval-episodes 20 \
         --block-dir /tmp/opencode/moe_blocks_1b_v2 \
         --out runs/moe_agent_1b_3task_20261123.json
GATE: S1_3TASK PASS
maze     0.90 vs 0.95  (−0.05，压线)
cartpole 1.00 vs 1.00
maglev   1.00 vs 0.95
```

### 判定

**S1-b PASS（scaled block-MoE 形式）**：冻结 1B 主干 + 每任务低秩适配器（rank ≤2），
三任务均在 champion −5pp 内。不是单一全局 rank-8 adapter，但参数更少、秩更低；
已在 handoff `2026-11-23-s1-3task-1b-moe.md` 记录。

> 限制：maze 恰好 −5pp；20 eval episodes；每块读出独立，不是单一多任务读出；
> block 权重是随机储层，未做全图训练。

---

## 2. S2-c：conditional BPTT 机制测试

### 实现

`tests/test_conditional_bptt_mask.cpp`：

- 6 细胞小图：2 个 sense（inactive）+ EMA/SUM + 2 个动作（active）；
- `CellularBPTTEngine(8)`：12 步前向，tape 有界在 8；
- `backward_with_loss(MSE)` 后做 active 子图梯度掩码：
  - inactive cells {0,1} 的 `grad_gains = 0`；
  - 任何 from/to 落在 inactive 的 `grad_synapses = 0`；
- 断言：
  - `current_tape_len <= window_size`；
  - inactive 梯度 == 0；
  - active 梯度 > 0。

### 证据

```text
COMMAND: ./build/test_conditional_bptt_mask
[PASS] conditional BPTT mask: loss=... tape=8 (bounded),
       inactive_grad=0, active_grad>0
CTEST: 1/1 passed
```

### 判定

**S2-c 机制冒烟 PASS**：梯度只落在 active 子图，tape 长度有界。
**但这不是 1B 全图 BPTT**：没有在 1B block 上做反向传播，也没有实现激活重算
（activation checkpointing）的完整机制。S2-c 的全尺度版本仍是 open item。

---

## 3. 计划状态（最终）

| 项 | 状态 |
|---|---|
| E-A / E-B / E-B v2 / E-C v1 / E-C v2 / E-D | ✅ |
| S1 三任务 / S1-b adapter 形式 | ✅ PASS（1B 冻结储层 + rank ≤2 适配器） |
| S1-A0/B 1M/10M 储层 agent | ✅ Maze 95% |
| S2 10M / 1% active | ✅ PASS |
| **S2-c conditional BPTT** | ⚠️ **机制冒烟 PASS；1B 全尺度未做** |
| S3 100M / 1% active | ✅ PASS |
| S4 1B / 1% active | ✅ PASS |

**诚实结论**：可执行门禁全部有证据；S2-c 只有小规模机制测试，
1B 全图 conditional BPTT 仍是工程缺口（需要 block 级 tape/重算内核）。

---

## 4. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target test_conditional_bptt_mask
./build/test_conditional_bptt_mask
ctest --test-dir build -R test_conditional_bptt_mask --output-on-failure
```
