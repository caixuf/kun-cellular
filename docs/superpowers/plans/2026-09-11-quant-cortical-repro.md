# 预注册：T5 量化皮层阵列 OOS 复现锁 — 2026-09-11

> 状态：**已结案（负结果）**。论文主口径 `train_multi_asset_cortical_array --seed 0`。
> 纪律：任务层 only；不改底座；**未覆盖**正式 `quant_cortical_array_champion.bin`。

---

## 协议

```bash
OMP_NUM_THREADS=6 ./build/train_multi_asset_cortical_array --seed 0 \
  --out checkpoints/quant_cortical_array_champion_repro_20260911.bin \
  --report runs/quant_cortical_array_oos_repro_20260911.json
```

论文锚点：OOS 夏普 **0.22** / 收益 **+13.55%** / 回撤 **20.11%** / 卡玛 **0.67**（2593 日）。

## 判据（冻结）

| 编号 | 判据 |
|------|------|
| Q1 | 命令跑通；≥40 品种；打印选择集 + OOS 四元组 |
| Q2 | OOS 年化夏普 **> 0** |
| Q3 | \|ΔSharpe\|≤0.10 且 \|ΔRet\|≤8pp 且 \|ΔMDD\|≤8pp；日期延长可标 DRIFT |

## 结果（回填 2026-09-11）

| 编号 | 结果 | 证据 |
|------|------|------|
| Q1 | ✅ | 43 品种；train 1970 / val 740 / test **2593**（2016-01-04…2026-09-01） |
| Q2 | ❌ | OOS 夏普 **-0.03**，收益 **-15.35%**，回撤 **32.19%**，卡玛 **-0.48** |
| Q3 | ❌ | 相对论文大幅偏离（非软 DRIFT 可解释范围） |

选择集（同跑）：夏普 **2.16** / 收益 **+85.3%** / 回撤 **5.6%** → **严重过拟合选择集，OOS 翻车**。

对照：论文宣称选择集夏普 0.76 / OOS 0.22；本仓 `CorticalMacroArray` **无 load_checkpoint**，无法对现有正式 bin 做同口径冷评，只能种子复训。复训不复现论文 OOS。

**结论**：T5 **负结果**——不得继续宣称「单命令可复现夏普 0.22」。正式冠军 bin 保留不动；sidecar 仅作失败证据。

日志：`runs/quant_cortical_array_repro_20260911.log`  
报告：`runs/quant_cortical_array_oos_repro_20260911.json`
