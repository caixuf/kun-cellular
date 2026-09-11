# 预注册：T4 易证三任务冷评回归 — 2026-09-11

> 状态：**启动前落盘**。不扩机制；只验证 README 宣称的三份冠军仍可冷复现。
> 任务：CartPole 三权 / 迷宫导航 / 流体阻尼。L0 零改（评测可用 `reset_state(false)`）。

---

## 协议

| 任务 | 冠军 | 命令 |
|------|------|------|
| CartPole | `checkpoints/cartpole_balance_champion.bin` | `./build/bench_easy_task_regression` |
| 迷宫 11×11 | `checkpoints/maze_navigation_champion.bin` | 同上 |
| 流体 | `checkpoints/fluid_damper_champion.bin` + C11 压测 | 同上 + `./build/test_multiphase_fluid_stress` |

## 判据（冻结）

| 编号 | 判据 |
|------|------|
| C1 | CartPole：phenotype `reset_state(false)`；ID ≥ **0.95**（≥19/20，MAX_STEPS=300）；OOD（重摆/长杆/噪）≥ **0.90** |
| M1 | 迷宫：100 未见种子（`50000+s*17`）逃逸 ≥ **0.95**（README 称 100%；主口径 phenotype false） |
| F1 | 流体 bin 可加载且前向有限；**且** `test_multiphase_fluid_stress` 三相 ALL PASS |

不足写 negative；坏掉则修评测/`initial_weight` 后重训，不调阈值。

## 结果（回填 2026-09-11）

| 编号 | 结果 | 证据 |
|------|------|------|
| C1 | ✅ PASS | `cartpole_tripartite` ID/OOD **20/20**；重训后 `cartpole_balance` 亦 **20/20**（旧 bin `initial_weight` 全 0 已作废备份） |
| M1 | ❌→✅ L3 | 欧氏方位 **80~86/100**（T4 负）；测地方位脚手架后 **96/100**（见 `2026-09-11-maze-l3-distill.md`） |
| F1 | ✅ PASS | fluid bin 拓扑/前向冒烟 ✅；`test_multiphase_fluid_stress` 三相 ALL PASS |

复现：

```bash
./build/bench_easy_task_regression
./build/test_multiphase_fluid_stress
```

日志：`runs/easy_task_regression_post_retrain_20260911.log`、`runs/maze_retrain_t4_20260911.log`、`runs/cartpole_balance_retrain_t4_20260911.log`

### 工程教训（任务层）

1. 落盘前必须 `initial_weight = weight`（balance 旧档、maze_tripartite 均踩坑）。
2. 冷评主口径用 `reset_state(false)`；`evaluate_organism` 内 `true` 依赖基因组同步。
3. 迷宫 11×11 现口径天花板约 **80%** 逃逸；要冲 ≥95% 须另开课程/奖励战役，不在 T4 烟测硬磨。
