# 形态发生旋钮消融（线路 B，2026-09-15）

> 任务层 only。不改 `include/kun/cellular/`。不覆盖 `checkpoints/zoo_cartpole.bin`。  
> 工具：`tools/ablate_morph_operators.cpp`  
> 产物：`checkpoints/morph_operator_ablation_cartpole_20260915.json`

## 先划界（避免冒领 §5.9）

论文 §5.9 四个启发式里，**默认 `evolve_generation` 不会调用**：共生宏细胞、冻存器官库、Chicxulub 大灭绝。环路增益筛查在 `mutate()` 末尾每次都跑，本实验未关。

默认 `mutate()` 实际开着的：fast 参数漂移、medium 加边、slow 有丝分裂、**硬编码 5% 凋亡（无配置项，关不掉）**、鲍德温固化、力敏/随机分裂开关。

## 预注册

- 任务：`ZooCartPole`，POP=32，GENS=120，与 `train_domain_zoo` 同协议
- 种子：1, 2, 3
- 指标：holdout ID SR 均值；`full − arm ≥ 0.10` → 该旋钮 **LOAD_BEARING**，否则 **NEGATIVE**
- 凋亡：不测、不宣称已消融

## 结果（ID SR 均值）

| 臂 | ID SR | vs full | 裁定 | 备注 |
|---|---:|---:|---|---|
| full | **1.00** | — | 基线 | 3/3 M1 PASS |
| no_mitosis | 0.97 | 0.03 | **NEGATIVE** | 细胞更少仍能过 |
| no_rewire | 0.43 | **0.57** | **LOAD_BEARING** | 3/3 M1 FAIL |
| param_only | 0.77 | **0.23** | **LOAD_BEARING** | 几乎不长突触 |
| no_baldwin | 1.00 | 0.00 | **NEGATIVE** | 本尺度无贡献 |
| no_mechano | 0.90 | 0.10 | **NEGATIVE（ID）** | OOD 全塌（0/0.0/0.4），3/3 M1 FAIL |

**一句话**：在倒立摆这个契约宽度上，**加边/重连承重**；有丝分裂和鲍德温不承重；力敏关掉后 ID 勉强、OOD 不行。这不是 ADAS 上的四算子完备消融，也不是 §5.9 那四个启发式的开关表。

## 复现

```bash
cmake --build build --target ablate_morph_operators -j
./build/ablate_morph_operators --out runs/morph_operator_ablation_cartpole_20260915.json
```
