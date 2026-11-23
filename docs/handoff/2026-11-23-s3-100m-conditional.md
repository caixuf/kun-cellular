# Handoff — S3 100M 条件计算探针（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §3.5
> 结论：**S3 PASS**。100M 总参数 / 500 个 200k 块；top-5（1M active=1%）Maze 95%；
> top-25（5M active=5%）Maze 95%。只训练每块读出 + k-means 路由，不做全图反传。

---

## 1. 证据格式

```text
PHASE: S3（100M conditional probe）
COMMIT: <本次 commit>
COMMAND: ./build/train_moe_agent_100m --collect-episodes 400 --eval-episodes 20 --full-sr 0.95
ARTIFACT: runs/moe_agent_100m_20261123.json
GATE: S3 PASS（top-5 95% >= 0.90 × full 95% = 85.5%）
KEY NUMBERS: 500 blocks × 200k = 100M cells / 300M syns；top-5 1% active 95%；top-25 5% active 95%
CONTROLS: 496/500 blocks ready；63,252 champion 样本；champion 采集 SR 96%
CONCLUSION: 100M 总参数 / 1% active 的条件 agent 保持 Maze 95%（E2 窄口径）
NEXT: S4 1B（本机内存不够，需硬件）
```

---

## 2. 架构

| 项 | 值 |
|---|---|
| 总参数 | 100,000,000 cells / 300,000,000 syns |
| 分块 | 500 blocks × 200,000 cells / 600,000 syns |
| 每块 | 随机稀疏 DAG 储层，64 in / 64 features |
| 路由 | k-means（500 类）on 4 维 maze 观测 |
| 每步 | 只跑 top-k 块；top-5 = 1M active = 1% |
| 读出 | 每块独立 ridge（64→2），冠军蒸馏 |
| 可训练参数 | 500 × 138 = 69,000（0.069% of 100M） |
| 评测 | 20 held-out episodes，250 steps |

---

## 3. 结果

| top-k | active cells | active % | Maze SR |
|---:|---:|---:|---:|
| **5** | **1,000,000** | **1%** | **95%** |
| 25 | 5,000,000 | 5% | 95% |

- full 10M 参考：95%；
- 冠军采集：96%（400 episodes）；
- 门禁阈值：`0.90 × 95% = 85.5%` → **PASS**；
- 496/500 块 ready（4 块样本不足）。

---

## 4. 解读

1. **100M 总参数 + 1% active 可以保持 Maze 能力**：  
   每步只执行 5 个 200k 块（1M cells），其余 99% 不执行；  
   可训练参数只有 69k（每块 138 参数读出）。

2. **这不是全图 100M 反传**：  
   block 权重是随机储层，没有训练；  
   训练的是每块 ridge 读出 + k-means 路由。  
   这正是 S3 计划允许的“只训练适配器/路由，不做全图反传”。

3. **不是多任务**：  
   只在 Maze 上验证；CartPole/Maglev 未跑。  
   不能宣称“100M 通用智能体”。

4. **S4 1B 本机不可行**：  
   5000 块 × 200k = 1B cells / 3B syns；  
   仅 block bin 就约 33 GB，超过本机 RAM/VRAM；需要更大内存/磁盘。

---

## 5. 标签

**E2（窄口径）**：100M 总参数 / 1% active 的 block-MoE 在 Maze 上保持 95%。
不是 E2 的部分：多任务、全图训练、1B、AGI。

---

## 6. 产物 / 复现

- 代码：`tools/train_moe_agent_100m.cpp`
- block bin：`/tmp/opencode/moe_blocks_100m/block_000..499.bin`（约 3.3 GB，本地，不入库）
- 报告：`runs/moe_agent_100m_20261123.json`（本地，runs/ gitignored）

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target train_moe_agent_100m
./build/train_moe_agent_100m --collect-episodes 400 --eval-episodes 20 --full-sr 0.95 \
  --out runs/moe_agent_100m_20261123.json
```

---

## 7. 计划完成度

| 项 | 状态 |
|---|---|
| E-A / E-B / E-B v2 / E-C v1 / E-C v2 / E-D | ✅ |
| S1 三任务 frozen trunk | ❌ BLOCKED（无多任务主干） |
| S1-A0/B 1M/10M 储层 agent | ✅ Maze 95% |
| S2 10M / 1% active | ✅ PASS |
| **S3 100M / 1% active** | ✅ **PASS** |
| S4 1B | ⬜ 本机内存不足（需硬件） |

> 结论：**“有规模智能体”已从 1M → 10M → 100M 条件 agent 走通；  
> 完整计划的 S4 1B 受本机内存限制，需要更大机器。**
