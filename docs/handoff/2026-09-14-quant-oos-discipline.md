# 量化 OOS 纪律门槛（稳定性战役 #5，2026-09-14）

> 不复训。把已有负结果冻成 CI 门禁，并勘误论文仍在说的「夏普 0.22 可复现」。

## 锁档证据

`checkpoints/quant_cortical_array_oos_repro_20260911.json`

| 集 | 夏普 | 收益 | 回撤 |
|---|---:|---:|---:|
| 选择集 val | **2.16** | +85.3% | 5.6% |
| OOS 2016–2026 | **-0.03** | -15.35% | 32.19% |
| 论文锚点 | 0.22 | +13.55% | 20.11% |

**Q2 FAIL（复训）。** 过拟合签名成立。正式 `quant_cortical_array_champion.bin` **未覆盖、不得改写本 JSON**。

## 正式 bin 冷评（另一份报告，2026-09-14）

任务层 `eval_cortical_array_bin` 首次加载正式档：OOS 夏普 **+0.14** / +6.3% / 回撤 25.6%。sidecar 复训档冷评仍为 **-0.04**，与上表对齐。证据：`checkpoints/quant_cortical_array_champion_cold_eval_20260914.json`。论文锚点 0.22 **仍未复现**。

## 工程门禁

```bash
python3 tests/test_quant_oos_discipline.py
```

若有人把冻结 JSON 改成正夏普来「过 CI」，测试会要求换**新报告文件**并改测试本身——禁止偷改数字。

## 文档

- 中英论文摘要 / 表 3 / §6.12：当前口径 = **OOS FAIL**，0.22 未复现
- Manifest 原本已诚实，未改宣称
