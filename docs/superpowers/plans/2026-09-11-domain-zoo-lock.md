# 预注册：T1 DomainZoo 确定性复跑锁档 — 2026-09-11

> 状态：**T1 已通过（12/12 复跑锁档）**。启动前落盘。任务层战役第一枪；不改 L0；不调 M1 阈值。
> 背景：仓内 `domain_zoo_report.json` 已标 `gate: true`（12/12），README 曾长期停留在 9/12 勘误。本文件冻结「再跑一遍」的通过标准，防止文档与产物再次漂移。

---

## 协议（冻结）

| 项 | 值 |
|----|-----|
| 命令 | `./build/train_domain_zoo`（或 cmake 目标同名） |
| 种子/超参 | 以 `tools/train_domain_zoo.cpp` 内嵌确定性配置为准（不改代码凑过门） |
| 产物 | `checkpoints/zoo_*.bin` + `checkpoints/domain_zoo_report.json` |

## 判据

| 编号 | 判据 |
|------|------|
| Z1 | 报告 `gate == true` 且 12/12 域 `gate` 均为 true |
| Z2 | 每个域存在对应 `checkpoints/zoo_<name>.bin` 且可被运行时加载 |
| Z3 | 若任一域 FAIL：如实改回文档口径，**禁止**改植物/阈值后不注明 |

## 结果（回填）

| 项 | 值 |
|----|-----|
| 日期 | 2026-09-11 |
| 命令 | `OMP_NUM_THREADS=6 ./build/train_domain_zoo` |
| 总计 | **12/12** PASS，总耗时 24.1s |
| 日志 | `runs/domain_zoo_lock_20260911.log`（gitignore） |

| 门 | 结果 |
|----|------|
| Z1 | ✅ `gate: true`，12 域全 PASS |
| Z2 | ✅ 12× `checkpoints/zoo_*.bin` 重写落盘 |
| Z3 | n/a（无 FAIL） |

**结论**：DomainZoo 地基锁档通过；任务层战役进入 **T2 室内覆盖**。
