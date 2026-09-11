# 预注册：T3 ADAS 冠军 vs Stanley 多种子评估锁档 — 2026-09-11

> 状态：**已跑完；相对论文/README 冻结表为负结果（A2–A4 ❌）；A1 ✅。脚本与现仓冠军复现表已落盘。**
> 口径诚实：**评估噪声多种子**（同一 bin × seeds 1..10），不是独立训练方差。

---

## 协议（冻结）

| 项 | 值 |
|----|-----|
| 冠军 | `checkpoints/adas_cortex_champion.bin` |
| 基线 | 工业 Stanley（与 `test_adas_cortex_contract` 同构） |
| 种子 | **1..10** |
| 命令 | `python3 tools/bench_adas_vs_stanley_seeds.py --baseline runs/adas_champion_vs_stanley_seeds1-10.json --out runs/adas_…_repro_….json` |
| 历史表（论文/README） | `runs/adas_champion_vs_stanley_seeds1-10.json`（**勿被复跑覆盖**；现为 7W/9L 归档） |

## 判据与结果

| 编号 | 判据 | 结果 |
|------|------|------|
| A1 | 16 场景×10 种子完赛 | ✅ |
| A2 | 胜场 ∈ {6,7,8} | ❌ **5W / 11L** |
| A3 | straight_cruise CTE ∈ [0.028, 0.045] | ❌ **0.0547** |
| A4 | vs 历史表 \|Δchamp\| ≤ 0.02 m | ❌ max **0.287**（`val_highway`） |

### 现仓冠军复现摘要（2026-09-11）

| 场景 | 历史表 champ | 现仓复现 champ | Stanley |
|------|-------------|----------------|---------|
| straight_cruise | 0.0338 | **0.0547** | 0.0286 |
| s_curve | 0.1124 | 0.1058 | 0.1394 |
| val_highway | 0.2430 | **0.5299** | 0.2076 |
| 战绩 | 7W/9L | **5W/11L** | — |

产物：`runs/adas_champion_vs_stanley_seeds1-10_repro_20260911.json`  
日志：`runs/adas_multiseed_lock_20260911.log`

## 阶段结论

1. **复现脚本可用**（`tools/bench_adas_vs_stanley_seeds.py`），评估链路未崩。
2. **随仓 `adas_cortex_champion.bin` 已与论文/README 引用的 10-seed 表脱节**（bin 内嵌 report 亦为 straight≈0.053 / val_highway≈0.51，与复现一致；历史表更像 L3 调优后的旧快照）。
3. **不得**用现仓 bin 继续宣称 7W9L / 3.38 cm 直道；文档须分流：**历史表** vs **现仓复现表**。
4. 下一步（另开预注册，非改本阈值）：对现仓 bin 重跑 L3 `tune_adas_gains.py` 或从可追溯 commit 恢复当时权重，再锁档。
