# CartPole Gate 5 / Gate 6 实现卡 (2026-09-15)

> 落地：`tests/test_flow_cartpole_gate5_replay.cpp`、`tests/test_flow_cartpole_gate6_shadow.cpp`。任务层 only；不改底座。锁档 `checkpoints/cartpole_balance_champion.bin`。

仓库里没有 `tests/test_flow_cartpole_balance.cpp`；ID/OOD 冷评入口是 `tools/bench_easy_task_regression.cpp`。

## Gate 5

- 种子 `9017`（ID 族 `9000+i*17`），`max_steps=300`，`force_noise=0`
- 两次独立 `load` + `compile` + `reset_state(false)`
- 实测：`steps=300 max_dobs=0 max_dact=0`

OOD（`masspole=0.2 / length=0.7 / force_noise=2.0`）含高斯推力，**不能**当 G5 回放工况。

## Gate 6

- 20 个 ID 种子 `9000+i*17`，300 步
- 影子基线：观测互逆后的 PD（`u = 12θ + 2.5θ̇ + 1.2x + ẋ`）
- 实测：`expert=20/20 champ=20/20 mean_|dF|=0.45495`（阈值 < 0.80）
- 冠军与专家不必轨迹重合；mean\|ΔF\|≈0.45 说明冠军推力偏抖，只过「不狂抖」门，不是丝滑 LQR。
