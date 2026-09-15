# 主结果：接线承重，有丝分裂不承重 (2026-09-15)

> 这是当前唯一接近口号、且有对照实验的定理。不是非冯机器证明。

## 命题

在现有契约宽度上，**可演化数据流图的控制能力主要在突触拓扑，不在细胞增殖。**

## 证据

**A. ADAS 拓扑消融**（`tests/test_adas_topology_ablation.py`）

- L3：同度数重连 34.1×，E/I 翻转 10.8×，重连后 0/8 完赛
- 体育场：23.9× / 22.0×

**B. ZooCartPole 形态发生旋钮**（`checkpoints/morph_operator_ablation_cartpole_20260915.json`）

| 臂 | ID SR | 裁定 |
|---|---:|---|
| full | 1.00 | 基线 |
| no_rewire | 0.43 | **LOAD_BEARING** |
| param_only | 0.77 | **LOAD_BEARING** |
| no_mitosis | 0.97 | NEGATIVE |
| no_baldwin | 1.00 | NEGATIVE |

§5.9 四启发式默认仍未进入 `evolve_generation`。凋亡关不掉。不要写成四算子完备消融。

## 边界

- 迷宫 96/100 依赖任务层测地脚手架，不是图自己长出地图。
- DomainZoo 冻结 12/12 是泄漏评测；卫生后隔离 **10/12**。
- 量化 0.22 未复现。
- `AutonomousReplicatorOrganism::spawn_offspring` 是 C++ 基因组拷贝，不是图内构造；图内协议见 `von_neumann_constructor.hpp`（描述带，非 28 原语通用构造器）。
