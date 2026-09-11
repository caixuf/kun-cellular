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

```bash
cmake --build build --target train_household_coverage_tripartite -j
./build/train_household_coverage_tripartite --eval-only
# 多种子重训（若冷评不过）：./build/train_household_coverage_tripartite
```

---

## 三、冻结判据

| 编号 | 判据 |
|------|------|
| H1 | 冷评或重训后三组 holdout **平均覆盖率 ≥ max(C0, 0.70) − 0.05**；若 C0<0.20，则战役升级为「修复至 ≥0.70 覆盖」而非「多种子巩固」 |
| H2 | 安全回充率 ≥ 0.50（C0_dock=0 时用绝对门槛） |
| H3 | OOD 动态避障自愈率 ≥ 0.50（C0 失效时用绝对门槛） |
| H4 | 可导出 SDSC-BIN + cert；Lyapunov CERTIFIED |

**阈值冻结声明**：下表 C0 来自 2026-09-11 `--eval-only` dry-run；之后禁止改写 C0。

### 冻结的 C0（dry-run 2026-09-11）

| 量 | 值 |
|----|-----|
| C0 覆盖率（ID 24×16，50 种子） | **0.070** |
| C0 合规通过率 ≥70% | **0/50** |
| C0_dock 安全回充率 | **0/50** |
| C0_crash 碰撞总次数 | 58950（评测器打印「安全零事故」——口径待核对，可能为接触计数非失败） |
| C0 OOD 覆盖率 | **0.062** |
| C0 OOD 动态避障自愈率 | **0/50** |
| BFS 教师覆盖率（同协议） | **1.000** |

**战役定性修正**：现有 `household_coverage_champion.bin` **冷评失效**（远低于论文/前端叙事）。T2 主目标改为 **修复或重训至 H1–H3 绝对门槛**，不得把失效冠军多种子「平均一下」当通过。

---

## 四、结果（回填）

（待填）
