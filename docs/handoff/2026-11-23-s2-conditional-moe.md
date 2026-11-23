# Handoff — S2 条件计算 MoE（10M 总参数 / 1% active，2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §3.4
> 结论：**S2 PASS**。10M 总参数拆成 100 个 100k 块；k-means 路由；
> **top-1（1% active）Maze 95% = 全量 10M 95%**；active 1/2/5/10% 全部 95%。

---

## 1. 证据格式

```text
PHASE: S2
COMMIT: <本次 commit>
COMMAND: ./build/train_moe_agent --collect-episodes 100 --eval-episodes 20 --full-sr 0.95
ARTIFACT: runs/moe_agent_20261123.json
GATE: S2 PASS（top1 95% >= 0.90 × full 95% = 85.5%）
KEY NUMBERS: 100 blocks × 100k = 10M cells / 30M syns；top-1 1% active Maze 95%；top-2/5/10 也是 95%
CONTROLS: 100/100 blocks ready；champion 采集 SR 93%；k-means 100 类路由
CONCLUSION: 1% active 的条件计算可以保持全量 10M 的 Maze 能力（E2 窄口径）
NEXT: S3 100M 条件训练探针
```

---

## 2. 架构

| 项 | 值 |
|---|---|
| 总参数 | 10,000,000 cells / 30,000,000 syns |
| 分块 | 100 blocks × 100,000 cells / 300,000 syns |
| 每块 | 随机稀疏 DAG 储层，64 in / 64 features |
| 路由 | k-means（100 类）on 4 维 maze 观测 |
| 每步 | 只跑 top-k 块；top-1 = 1% active |
| 读出 | 每块独立 ridge 回归（64→2），冠军蒸馏 |
| 评测 | 20 held-out episodes，250 steps |

关键实现：

- 100 个 block bin 由 `make_block(100k, 300k, seed)` 生成；
- 冠军采集 100 episodes → 15,601 个 `(obs, action)` 样本；
- k-means 100 类；每块 ~156 个样本；
- 每块 ridge 读出；100/100 块 ready；
- 评测时按最近的 **ready** 块路由；block 状态跨步保留。

---

## 3. 结果

| top-k | active % | Maze SR |
|---:|---:|---:|
| **1** | **1%** | **95%** |
| 2 | 2% | 95% |
| 5 | 5% | 95% |
| 10 | 10% | 95% |

- full 10M（S1-B，20 episodes）：**95%**；
- 冠军（100 episodes 采集）：93%；
- top-1 1% active：**95%**，与全量持平；
- 门禁阈值：`0.90 × 95% = 85.5%` → **PASS**。

---

## 4. 解读

1. **1% active 足够**：Maze 的观测只有 4 维，k-means 把状态空间切成 100 个局部区域；
   每块在自己的区域里用 64 维随机特征 + 线性读出就能复现冠军动作。

2. **active 曲线平坦**：1%→10% 都是 95%，说明该任务不需要更多 active 容量；
   这不证明“更大 active 无用”，只证明 **Maze 已饱和**。

3. **这是条件计算，不是全量前向**：  
   每步只调用 top-k 个 block 的 `sdsc_binary_forward`；其余 99% 块不执行。  
   但 block 权重是随机储层，不是训练出来的 10M 主干。

4. **不是 S3**：  
   没有 100M；没有跨任务；没有 full 100M 训练。

---

## 5. 标签

**E2（窄口径）**：10M 总参数 / 1% active 的 block-MoE 在 Maze 上保持全量能力。
不是 E2 的部分：通用条件计算、跨任务、100M、AGI。

---

## 6. 成本

| 项 | 值 |
|---|---|
| 100 block 生成 | 10M cells / 30M edges，秒级 |
| 冠军采集 | 100 episodes × 250 steps |
| 特征/读出 | 15,601 样本 × 100k block 前向 |
| 评测 | 4 个 top-k × 20 episodes × 250 steps |
| 本机墙钟 | 约 1–2 分钟（CPU block 前向） |

---

## 7. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target train_moe_agent
./build/train_moe_agent --collect-episodes 100 --eval-episodes 20 --full-sr 0.95 \
  --out runs/moe_agent_20261123.json
```

> block bin 在 `/tmp/opencode/moe_blocks`（100 × 3.3 MB），可重新生成，不入库。
