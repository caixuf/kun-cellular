# Handoff — 10M 规模智能体（稀疏储层 + 蒸馏读出，2026-11-23）

> 目标：把 1M 储层 agent 推到 10M/100M，验证“规模”是否继续带来能力。
> 结论：**10M 稳定达到 Maze 95% = 冠军**；100M 不稳定（50%–80%），
> **规模不是越大越好**，随机储层需要密度/结构设计。

---

## 1. 证据格式

```text
PHASE: S1-B（10M reservoir agent）
COMMIT: <本次 commit>
COMMAND: LD_LIBRARY_PATH=<nvidia/cu13/lib> ./build/train_scaled_agent \
  --cells 10000000 --syns 30000000 --reservoir checkpoints/scaled_reservoir_10m.bin \
  --collect-episodes 20 --eval-episodes 20 --no-raw
ARTIFACT: checkpoints/scaled_agent_10m_maze_readout.json（tracked）
          checkpoints/scaled_reservoir_10m.bin（306 MB，本地，未入库）
          runs/scaled_agent_10m_20261123.json
GATE: SCALED PASS（Maze 95% vs 95%）
KEY NUMBERS: 10,000,000 cells / 30,000,000 syns；readout 138 params；Maze 95% vs 95%；CartPole 0%；Maglev 5%
CONTROLS: 1M agent Maze 95%；100M/100M 与 100M/300M 均不稳定
CONCLUSION: 10M 是当前随机储层的稳定规模点；100M 需要更密/更结构化的设计
NEXT: S2 条件计算 / 更密 100M / 非线性读出
```

---

## 2. 规模对比

| 规模 | 突触 | 特征探针 | Maze（agent/champion） | CartPole | Maglev |
|---|---:|---|---:|---:|---:|
| 1M | 3M | 非零 | **95% / 95%** | 70% / 100% | 0% / 100% |
| **10M** | **30M** | **非零** | **95% / 95%** | 0% / 100% | 5% / 100% |
| 100M | 100M | **全 0** | 30%–50% / 90% | 0% | 0% |
| 100M | 300M | 非零但极小（1e-5） | 50%–80% / 90% | 0% | 0% |

- 10M/30M 的 20-episode 评测：Maze **19/20 vs 19/20**；
- 100M/100M：1 边/细胞，特征为 0，信号死亡；
- 100M/300M：3 边/细胞，特征非零但幅度 1e-5，20 episodes 下 Maze 50%，不稳定；
- 1M/3M 的 CartPole 70%，但 10M/30M 的 CartPole 0%——**随机储层的任务适配随规模漂移**。

---

## 3. 产物

| 文件 | 说明 | 是否入库 |
|---|---|---|
| `checkpoints/scaled_reservoir_1m.bin` | 1M / 3M，31 MB | ✅ 已提交 |
| `checkpoints/scaled_agent_maze_readout.json` | 1M Maze 读出 138 参数 | ✅ 已提交 |
| `checkpoints/scaled_reservoir_10m.bin` | 10M / 30M，306 MB | ❌ 本地（太大） |
| `checkpoints/scaled_agent_10m_maze_readout.json` | 10M Maze 读出 138 参数 | ✅ 已提交 |
| `runs/scaled_agent_10m_20261123.json` | 10M 评测报告 | ❌ runs/ gitignored |
| `runs/scaled_agent_100m*.json` | 100M 负结果 | ❌ runs/ gitignored |

10M bin 可用命令重新生成（约 1 分钟）：

```bash
LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
  ./build/train_scaled_agent --cells 10000000 --syns 30000000 \
  --reservoir checkpoints/scaled_reservoir_10m.bin \
  --collect-episodes 20 --eval-episodes 20 --no-raw
```

---

## 4. 解读

1. **10M 是当前随机稀疏储层的稳定规模点**：  
   Maze 与 1M 一样达到冠军 95%，说明规模本身没有破坏能力。

2. **100M 不是“再堆 10 倍”**：  
   100M/100M 特征全 0；100M/300M 特征 1e-5，20 episodes 下 Maze 50%。  
   随机 DAG 的深度/密度导致信号衰减；需要分层/分块/残差结构，而不是均匀随机。

3. **CartPole 随规模漂移**：  
   1M 的 70% → 10M 的 0%，说明随机储层的任务适配不稳定；  
   Maze 的 95% 稳定，但不足以证明“多任务规模智能”。

4. **这不是 S2 条件计算**：  
   仍然是“全量 10M 前向 + 小读出”；S2 的 top-k active 路由尚未实现。

---

## 5. 标签

- **E2 窄口径**：10M 储层 + 138 参数读出，Maze 达到冠军水平（20 episodes）；
- **E1**：1M CartPole 70%；
- **E3**：100M 随机储层不稳定；
- **未完成**：S2 条件计算 / S3 100M 训练 / 多任务 S1 全通。

---

## 6. 下一步

1. **S2 条件计算**：10M 总参数 / 1% active，验证 active 扩展曲线；
2. **分层/分块 100M**：把 10M 稳定块复制 10 份 + 稀疏跨块连接，而不是均匀随机；
3. **非线性读出**：小 MLP 头，补 CartPole/Maglev；
4. **多任务读出**：一个读出吃三任务，接近 S1 的“单主干多任务”。

> 当前结论：**“有规模的智能体”已在 1M/10M 达成（Maze 冠军水平）；  
> “上亿稳定智能体”还没有，100M 需要结构设计，不是单纯堆细胞。**
