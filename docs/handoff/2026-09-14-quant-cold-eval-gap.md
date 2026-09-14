# 量化冷评缺口（2026-09-14，已落地）

> 任务层 only；不改 `include/kun/cellular/`；不覆盖正式 bin；不改冻结 OOS JSON。

## 五行摘要

1. **已补齐**：`tools/quant_array_common.hpp::load_cortical_array_from_sdsc_bin` 从 SDSC-BIN v2 还原 43 柱 + 宏轴突。
2. **入口**：`./build/eval_cortical_array_bin [--bin PATH] [--dry-load] [--report PATH]`。
3. **Loader 对账**：sidecar 复训档冷评 val 2.16 / OOS **-0.04**，对齐冻结 JSON（-0.03），差值来自存盘 u8/64 增益量化。
4. **正式 champion.bin 首次冷评**：val 夏普 **0.74**，OOS 夏普 **+0.14** / +6.3% / 回撤 25.6%（2016–2026）。**不是**论文锚点 0.22。
5. **纪律**：`--seed 0` 复训仍锁 FAIL；不得把 0.14 写成 0.22 已复现；Gate 3 对锚点仍 FAIL。

## 复现

```bash
cmake --build build --target eval_cortical_array_bin -j
python3 tests/test_quant_cortical_array_cold_eval.py
./build/eval_cortical_array_bin --bin checkpoints/quant_cortical_array_champion.bin \
  --report checkpoints/quant_cortical_array_champion_cold_eval_20260914.json
```

## 数字

| 对象 | val 夏普 | OOS 夏普 | OOS 收益 | OOS 回撤 |
|---|---:|---:|---:|---:|
| 论文锚点 | 0.76（旧宣称） | **0.22** | +13.55% | 20.11% |
| `--seed 0` 复训（冻结） | 2.16 | **-0.03** | -15.35% | 32.19% |
| 同上 sidecar 冷评（u8） | 2.16 | **-0.04** | -16.17% | 32.26% |
| **正式 champion.bin 冷评** | 0.74 | **+0.14** | +6.32% | 25.55% |

## 实现边界

- 增益按存盘 `param1_u8/64` 还原，不是训练内存 float。
- 底座 `CorticalMacroArray` 仍无 `load_checkpoint`；加载器在任务层。
- 观测台量化页仍只渲染拓扑，不跑这笔回测。
