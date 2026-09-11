# 预注册：ADAS L3 增益重调 → 多种子复锁 — 2026-09-11

> 状态：**完成；L1–L4 全 ✅**。tuned 已升格为现仓 `adas_cortex_champion.bin`。
> 方法：拓扑冻结，sep-CMA-ES 80 iter（`tools/tune_adas_gains.py --seed 20260911`）。

---

## 协议（已执行）

| 项 | 值 |
|----|-----|
| 输入 | `adas_cortex_champion.bin`（pre-L3 备份：`…_pre_l3_20260911.bin`） |
| 调参 | iters=80 pop=24 seed=20260911 eval_seeds=2 |
| 复评 | `runs/adas_champion_vs_stanley_seeds1-10_tuned_20260911.json` |

## 门禁

| 编号 | 判据 | 结果 |
|------|------|------|
| L1 | straight_cruise ≤ 0.045 | ✅ **0.0312** |
| L2 | 胜场 ≥ 6 | ✅ **9W / 7L** |
| L3 | val_highway ≤ 0.35 | ✅ **0.1715**（且胜 Stanley） |
| L4 | 完赛 | ✅ |

相对 T3 未调参复现（5W/11L，straight 0.055，val_highway 0.53）：直道 ×1.75、高速留出 ×3.09 回收。

## 说明

- 战绩组合与历史 7W9L 快照**不完全同构**（现赢 `gentle_s`/`follow`/`val_highway`，微负 `s_curve`）。
- 不强求复刻历史表；现仓口径以本文件 + README 新表为准。
