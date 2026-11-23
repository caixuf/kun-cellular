# Handoff — 1M 规模智能体（稀疏储层 + 蒸馏读出，2026-11-23）

> 目标：在“没有可用 1M 训练主干”的现实下，**新建**一个有功能的 1M 细胞稀疏储层，
> 用冠军蒸馏训练小读出，产出可部署的 1M 规模 agent。
> 结论：**Maze 95% vs 冠军 95%**（20 held-out episodes，储层-only）；
> CartPole 70% vs 100%；Maglev 0%。**规模里程碑达成，但不是 S1 三任务全通。**

---

## 1. 证据格式

```text
PHASE: S1-A0（1M reservoir agent）
COMMIT: <本次 commit>
COMMAND: LD_LIBRARY_PATH=<nvidia/cu13/lib> ./build/train_scaled_agent --collect-episodes 20 --eval-episodes 20 --no-raw
ARTIFACT: checkpoints/scaled_reservoir_1m.bin
          checkpoints/scaled_agent_maze_readout.json
          runs/scaled_agent_reservoir_only_20261123.json
GATE: SCALED PASS（Maze 达到冠军 −0pp；三任务 S1 仍 FAIL）
KEY NUMBERS: reservoir 1,000,000 cells / 3,000,000 syns；readout 138 params；Maze 95% vs 95%；CartPole 70% vs 100%；Maglev 0% vs 100%
CONTROLS: hybrid（加 4 维原始观测）Maze 95%、CartPole 70%；reservoir-only 同样 Maze 95%，说明不是靠原始观测
CONCLUSION: 1M 储层确实提供可用特征；Maze 达到冠军水平，CartPole 部分，Maglev 失败
NEXT: 多任务读出/非线性头，或 S2 条件计算
```

---

## 2. 为什么新建 1M 主干

仓库现有 1M/8k 资产都不是训练主干：

- `sdsc_mega_1million.bin`：压测拓扑，16 维输出全 0；
- `flagship_io_mix_n64_*`：IO 混合实验，4 维稀疏输入输出 0；
- ADAS 210 细胞：真训练，但只在 maglev 上 −4pp。

所以本工具用 `CompactSoAGenome` **新建**一个结构化稀疏 DAG 储层：

| 项 | 值 |
|---|---|
| 细胞 | 1,000,000 |
| 突触 | 3,000,000 |
| 输入 | 64（任务观测广播编码） |
| 特征输出 | 最后 64 个 feature 细胞 |
| 拓扑 | 每个 hidden 约 3 条入边，源索引 < 目标索引（DAG） |
| 算子 | DAMPER / INTEGRATE / DIFF / SUM / MULTIPLY / CLIP / ABS |
| checkpoint | `checkpoints/scaled_reservoir_1m.bin`（31 MB，v2） |

特征探针（输入 `{0.1,-0.2,0.3,-0.4}` 广播）非零：前 8 维约
`-0.018, 0.021, 0.00027, -3e-5, -7e-6, 9.3e-5, 2.3e-5, 0.083`。

---

## 3. 蒸馏读出

- 教师：现成冠军（maze / cartpole / maglev）；
- 采集：每任务 20 episodes 的冠军轨迹，记录 `(obs, action)`；
- 特征：64 维储层输出 + 4 维原始观测 + bias（混合），或仅 64 维储层（`--no-raw`）；
- 读出：ridge 回归，`(68+1) × 2 = 138` 参数；储层完全冻结；
- 部署：`checkpoints/scaled_agent_maze_readout.json`（mean/std/w + 指标）。

---

## 4. 结果（20 held-out episodes）

| 任务 | 冠军 | 1M agent（reservoir-only） | 1M agent（hybrid） |
|---|---:|---:|---:|
| **Maze** | **95%** | **95%** | **95%** |
| CartPole | 100% | 70% | 70% |
| Maglev | 100% | 0% | 0% |

- reservoir-only 与 hybrid 的 Maze 都是 95% → **不是靠 4 维原始观测**；
- CartPole 70%，低于冠军，但显著高于 chance；
- Maglev 失败（读出/特征不足以复现冠军的非线性控制）。

---

## 5. 判定

| 判据 | 结果 |
|---|---|
| 至少一个任务 ≥ 冠军 −5pp | ✅ Maze：95% vs 95% |
| 三任务 S1 判据 | ❌ CartPole −30pp / Maglev −100pp |
| 规模 | ✅ 1,000,000 cells / 3,000,000 syns |
| 可部署产物 | ✅ reservoir bin + readout JSON |

**标签**：
- **E2 窄口径**：1M 储层 + 138 参数读出，在 Maze 上达到冠军水平；
- **E1**：CartPole 70%；
- **E3**：Maglev 0%（当前读出/特征不成立）。

这是 **S1-A0 规模里程碑**，不是 S1 三任务全通，也不是 S2/S3。

---

## 6. 成本

| 项 | 值 |
|---|---|
| 储层生成 | 1M cells / 3M edges，秒级 |
| 特征采集 | 3 任务 × 20 episodes × ~250 steps × ~12 ms ≈ 3 min |
| 读出训练 | ridge 闭式，毫秒级 |
| 闭环评测 | 3 任务 × 20 episodes × ~250 steps × ~12 ms ≈ 3 min |
| 读出参数 | 138（<0.02% 储层） |

---

## 7. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target train_scaled_agent

LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
  ./build/train_scaled_agent --collect-episodes 20 --eval-episodes 20 --no-raw \
  --out runs/scaled_agent_reservoir_only_20261123.json
```

> 读出权重：`checkpoints/scaled_agent_maze_readout.json`；储层：`checkpoints/scaled_reservoir_1m.bin`。

---

## 8. 限制与下一步

**限制**：
- 储层是随机稀疏 DAG，不是多任务预训练主干；
- 读出是线性 ridge，无法表达 Maglev 的非线性控制；
- 只在 20 episodes 上评测，统计噪声约 ±11pp；
- 不是 S2/S3 的条件计算，也没有 10M/100M。

**下一步（按收益排序）**：
1. **非线性读出**（小 MLP / rank-8 + tanh）→ 看 CartPole/Maglev 能否补上；
2. **多任务联合读出**（一个读出吃三任务）→ 接近 S1 的“单主干多任务”；
3. **S2 条件计算**：1M 总参数 / 1% active，验证 active 扩展曲线；
4. 若以上成立，再谈 10M/100M。
