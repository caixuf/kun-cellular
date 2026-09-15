# 全屋覆盖 Gate 5 / Gate 6 实现卡 (2026-09-15)

> 落地：`tests/test_flow_household_gate5_replay.cpp`、`tests/test_flow_household_gate6_shadow.cpp`。任务层 only；不改底座。锁档 `checkpoints/household_coverage_champion.bin`。

`tests/test_flow_household_coverage.cpp` 只测环境契约与 `run_baseline`，不加载冠军 bin。

## Gate 5

- ID 户型 24×16，种子 `1000`，步数上限 1200，**不**注入动态障碍
- 两次独立 `load` + `compile` + `reset_state(false)`
- 实测：`steps=811 max_dxy=0 max_dhead=0 max_dcov=0 max_dbat=0 max_dact=0`

## Gate 6

- 20 个 ID 种子 `1000..1019`，24×16，1200 步
- 影子基线：`HouseholdCoverageEvaluator::run_baseline`（最近未扫格 BFS，握全图）
- 冠军成功定义与任务层一致：覆盖 ≥ 0.70 **且** 回桩
- **禁止**用成功计数对撞教师（信息集不同；迷宫 G6 的 2/20 只适用于同信息测地）

实测：

```
GATE6_HOUSEHOLD expert=20/20 champ=17/20 mean_cov e/c=1/0.926891 champ_col=0 mean_|dneg|=0.240482
```

未达 0.70 的三局均已回桩：1002=0.689、1005=0.568、1018=0.482。OOD 动态障碍自愈仍走既有 G3，不写入本 G6。
