# 预注册：T2 室内覆盖（household）多种子复证 — 2026-09-11

> 状态：**已完成。H1–H3 ✅**（修复训练器后重训锁档）。
> 目标：把已有 household 冠军从失效冷评修复为可复现任务门禁；L0 零改。

---

## 一、根因（诊断）

| 缺陷 | 后果 |
|------|------|
| `run_household_episode` 调用 `reset_state(true)` | 每回合把权重打回 `initial_weight` |
| 变异只改 `weight`、不同步 `initial_weight` | 演化变异被系统性抹掉 |
| 旧冠军 bin 为 **v2** 且类型/突触残缺 | 冷载入无有效效应器输出 → 零动作默认直行 → ~7% 覆盖 |
| `step_continuous` 双零动作默认 FORWARD | 放大「僵尸直行」 |

修复：`reset_state(false)`；变异同步 `initial_weight`；落盘前同步；双零 → WAIT；重训并 v4 落盘。

---

## 二、协议与复现

```bash
cmake --build build --target train_household_coverage_tripartite -j
./build/train_household_coverage_tripartite          # 重训+锁档评测
./build/train_household_coverage_tripartite --eval-only
```

### 冻结的 C0（失效基线，dry-run 2026-09-11）

| 量 | 值 |
|----|-----|
| C0 覆盖率 | **0.070** |
| C0_dock | **0/50** |
| C0 OOD 自愈 | **0/50** |

### 绝对门槛（C0 失效后启用）

| 编号 | 判据 | 结果 |
|------|------|------|
| H1 | ID 50 种子平均覆盖率 ≥ 0.70 | ✅ **0.877** |
| H2 | 安全回充率 ≥ 0.50 | ✅ **50/50** |
| H3 | OOD 动态避障自愈率 ≥ 0.50 | ✅ **50/50**（OOD 覆盖 0.827） |
| H4 | SDSC-BIN v4 可冷载入 | ✅ 11 细胞 / 11 突触 |

---

## 三、结果（回填）

| 项 | 值 |
|----|-----|
| 冠军 | `checkpoints/household_coverage_champion.bin`（v4） |
| 训练 | POP=32 GENS=60 seed=20260911；耗时 ~1.1s |
| ID 合规 ≥70% | 40/50（80%） |
| 碰撞 | ID/OOD 评测协议下 **0** |
| 教师 BFS | 100%（上界参照，非击败门） |
| 日志 | `runs/household_retrain_20260911.log` / `household_eval_final.log` |

**结论**：T2 通过。论文/README 若仍暗示「失效旧 bin 的 100% 自愈」须以本冷评口径为准；当前可复现为 **ID 覆盖 ~88%、OOD ~83%、回充/自愈 100%**（50 种子）。
