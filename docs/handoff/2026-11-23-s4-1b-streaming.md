# Handoff — S4 1B 流式条件 MoE（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §3.6
> 结论：**S4 PASS**。1B cells / 3B syns，500 块 × 2M；top-5（10M active = 1%）
> Maze **95%**；块 bin 共 **30 GB** 落盘，流式加载，不把 1B 全塞进内存。

---

## 1. 证据格式

```text
PHASE: S4（1B streaming conditional probe）
COMMIT: <本次 commit>
COMMAND: ./build/train_moe_agent_1b --collect-episodes 500 --eval-episodes 20 --topk 5 --full-sr 0.95
ARTIFACT: runs/moe_agent_1b_20261123.json
GATE: S4 PASS（top-5 95% >= 0.90 × 95% = 85.5%）
KEY NUMBERS: 500 blocks × 2M = 1,000,000,000 cells / 3,000,000,000 syns；top-5 10M active (1%) Maze 95%
CONTROLS: 78,538 champion 样本；champ_sr 96.8%；64/500 blocks ready；30 GB block bins
CONCLUSION: 1B 总参数 / 1% active 的条件 agent 保持 Maze 95%（E2 窄口径）
```

---

## 2. 架构与流式实现

| 项 | 值 |
|---|---|
| 总参数 | 1,000,000,000 cells / 3,000,000,000 syns |
| 分块 | **500 blocks × 2,000,000 cells / 6,000,000 syns** |
| 每块 | 随机稀疏 DAG 储层，64 in / 64 features |
| 路由 | 4D 网格（ray×3 + bearing）→ 500 桶；按桶质心 top-k |
| 每步 | 只跑 top-5 块 = 10M cells = **1% active** |
| 读出 | 每块 independent ridge（64→2），冠军蒸馏；500×138 参数 |
| block bin | 500 × ~60 MB ≈ **30 GB**，放 `/tmp/opencode/moe_blocks_1b_v2` |
| 训练 | 逐块 load → 采集特征 → ridge → `sdsc_binary_free`，内存常驻 1 块 |
| 评测 | LRU cache 最多 24 块（~1.5 GB），按需加载 top-5 |

关键：**1B 全量从未同时驻留内存**；峰值内存约 2–3 GB（cache + 样本 + 单块）。

---

## 3. 结果

| 项 | 值 |
|---|---|
| champion 采集 SR | 96.8% |
| full 10M 参考 | 95% |
| **top-5 / 1% active** | **95%** |
| 门禁阈值 | 85.5% |
| ready blocks | 64 / 500 |
| 样本数 | 78,538 |

- top-5 从 64 个 ready 块里选；其余 436 块样本不足 <8；
- Maze 95% 与 full 10M / 100M 条件探针一致；
- 1B 总参数、1% active、30 GB 落盘、流式加载——全部真实执行。

---

## 4. 第一次尝试（失败记录）

先试了 **5000 blocks × 200k**：

- 5,000 块 / 1B cells / 3B syns / 33 GB；
- 162,046 样本，但 4D 网格覆盖太稀疏，只有 **82/5000 blocks ready**；
- top-50（1% active）Maze **75%**，低于 85.5% → FAIL。

原因：200k 小块需要更多样本才能覆盖 5000 个网格桶；改成 **500 × 2M** 后，
每块样本从 ~32 提升到 ~157，ready 块 64/500，top-5 达到 95%。

这不是硬件失败，是**分块粒度/样本覆盖**问题。

---

## 5. 标签

**E2（窄口径）**：1B 总参数 / 1% active 的 streaming block-MoE 在 Maze 上保持 95%。
不是 E2 的部分：多任务、全图 1B 反传、通用智能。

---

## 6. 限制

- block 权重是随机储层，**没有训练**；只训练每块 ridge 读出 + 网格路由；
- 64/500 blocks ready，其余块样本不足；有效专家数 64，不是 500；
- 只在 Maze 上评测，20 episodes；
- 不是 full 1B backprop（计划明确不做）；
- S1 原始三任务 frozen-trunk 判据仍 BLOCKED（无多任务训练主干）。

---

## 7. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target train_moe_agent_1b
./build/train_moe_agent_1b --collect-episodes 500 --eval-episodes 20 --topk 5 \
  --full-sr 0.95 --block-dir /tmp/opencode/moe_blocks_1b_v2 \
  --out runs/moe_agent_1b_20261123.json
```

> 30 GB block bins 可重新生成，不入库；`runs/` 报告 gitignored。
