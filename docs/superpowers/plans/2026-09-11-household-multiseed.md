# 预注册：T2 室内覆盖（household）多种子复证 — 2026-09-11

> 状态：**T1 通过后启动前落盘**。任务层第二枪；L0 零改；负结果如实。
> 目标：把已有 `train_household_coverage_tripartite` + `household_coverage_champion.bin` 从「单次证书」升为 **多种子可复现任务门禁**，对齐论文具身面，不扩 FieldCML 接线机制。

---

## 一、前置

| 项 | 值 |
|----|-----|
| 环境 | `tasks/robotics/household_coverage.hpp` |
| 训练器 | `tools/train_household_coverage_tripartite.cpp` |
| 现有冠军 | `checkpoints/household_coverage_champion.bin`（11 细胞 / 10 突触；cert 2026-09-10） |
| 既有实现计划 | `docs/superpowers/plans/2026-09-02-household-coverage-benchmark.md`（工程清单；本文件管**科学门禁**） |

---

## 二、协议（冻结）

| 项 | 值 |
|----|-----|
| 评测 | 训练器内嵌 OOD/多种子评测路径（以源码为准；开跑前打印并冻结命令行） |
| 演化种子 | **3**：20260910, 20260911, 20260912（若训练器仅支持单种子，则固定评测种子族 3 组 holdout） |
| 对照 | 确定性割草机基线（若环境已暴露）；无则仅报绝对覆盖率 |

开跑前一步：`./build/train_household_coverage_tripartite --help` 或读 main，把**实际命令**写入本节「复现命令」子条（不得开跑后改门）。

### 复现命令（开跑时填写）

```
（待填）
```

---

## 三、冻结判据

| 编号 | 判据 |
|------|------|
| H1 | 三种子（或三组 holdout）**平均覆盖率 ≥ 既有单次宣称的 90% 相对水平**——开跑前从一次 dry-run 打印「当前冠军冷评覆盖率 C0」，冻结为 `mean ≥ C0 − 0.05` |
| H2 | 回充/回桩成功率：三组均值 ≥ max(C0_dock − 0.05, 0.50)（C0_dock 由 dry-run 冻结） |
| H3 | 动态障碍碰撞率：三组均值 ≤ C0_crash + 0.05（越低越好） |
| H4 | 冠军仍可导出 SDSC-BIN + cert；Lyapunov 声明不劣于现有 CERTIFIED |

**阈值冻结声明**：C0* 必须在 dry-run 后、正式多种子前写入本节；之后禁止改。

### 冻结的 C0（dry-run 后填）

| 量 | 值 |
|----|-----|
| C0 覆盖率 | （待填） |
| C0_dock | （待填） |
| C0_crash | （待填） |

---

## 四、结果（回填）

（待填）
