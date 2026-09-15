# 稳定性战役（2026-09-14）

> 目标：对外可宣称「稳定」= 每生境 **锁档 bin + 门禁证据 + OOD/消融 + 一键复现**。  
> 纪律：不改 `include/kun/cellular/` 底座；失败如实记；禁止冒领。

## 顺序（一个一个做）

| # | 项 | Owner | 状态 |
|---|---|---|---|
| 1 | ADAS 学生闭环 ≥ 导师可默认展示 | Cursor 主线 | **✅ 达标**：student 体育场 p95 1.235→**0.145 m**；Stanley **10W/6L**；`adas_cortex_champion_stadium.bin`（不覆盖 L3）；观测台默认 student |
| 2 | 六道门禁 × 生境打勾矩阵（证据路径） | agy 并行 | agy 已交 |
| 3 | DomainZoo 9/12 vs 12/12 统一 + 三失败域锁 FAIL | agy 取证已交 | **✅ 统一为 12/12**：2026-09-14 实测复跑 12/12 全 PASS；三失败域不再锁 FAIL（中间窗已修复）；见 `2026-09-14-domainzoo-repro.md` |
| 4 | 拓扑消融（重连 / E-I） | Cursor 主线 | **✅ PASS**：L3 重连 34.1× / E-I 10.8×；stadium 23.9× / 22.0×；见 `2026-09-14-topology-ablation.md` |
| 5 | 量化 OOS 纪律门槛 | Cursor 主线 | **✅ 复训锁 FAIL**（-0.03）。正式 bin 冷评已补：OOS **+0.14** ≠ 锚点 0.22；见 `2026-09-14-quant-cold-eval-gap.md` |
| 6 | 观测台一键 clone→cmake→8833 | Cursor 主线 | **✅** `bash tools/run_observatory.sh` → `http://localhost:8833/` |
| A | 口径对齐 + CI 挂门禁 | Cursor + agy | **✅** README/观测台拆开 L3 与体育场 |
| C | 迷宫 Gate 5 回放 | Cursor + agy | **✅** `tests/test_flow_maze_gate5_replay.cpp`；G6 已补，见下行 |
| C2 | 迷宫 Gate 6 影子 | Cursor | **✅** `tests/test_flow_maze_gate6_shadow.cpp`：专家 20/20、冠军 20/20、mean\|Δneg\|=0.306 |
| C3 | CartPole Gate 5/6 | Cursor | **✅** G5 300 步差分全 0；G6 PD 与冠军 20/20，mean\|ΔF\|=0.455。OOD 噪声不可回放。 |
| C4 | household Gate 5/6 | Cursor | **✅** G5 种子 1000 共 811 步差分全 0；G6 教师 20/20、冠军 17/20、均覆盖 0.927、零碰撞。 |
| B | 形态发生旋钮 On/Off | Cursor + agy | **✅ 部分**：ZooCartPole 加边承重；有丝分裂/鲍德温 NEGATIVE；凋亡仍关不掉。见 `2026-09-15-morph-operator-ablation.md` |

## #1 成功判据（预注册）

1. `checkpoints/adas_cortex_champion.bin`（或明确标为实验档的后继）在 `VAL_SCENARIOS` 上 **全 ok**；
2. 相对 Stanley：验证集 **净胜率 ≥ 锁档叙事（约 9W/7L 量级）或明确写下新数字**；
3. 观测台默认 `student` 模式跑体育场公路：**CTE p95 < 0.5 m**，无反复离轨重置；
4. 产物：`runs/adas_student_vs_teacher_baseline_YYYYMMDD.json` + 可选 `live_exp` 不覆盖 L3。

## #2 给 agy 的任务卡（可直接 `--print`）

扫描仓库，产出 `docs/handoff/2026-09-14-gate-matrix.md`：

- 行：ADAS / maze / doudizhu / cartpole / DomainZoo / household / quant（有则列）
- 列：Gate1…Gate6 + 锁档路径 + 测试文件路径 + PASS/FAIL/未测
- 只引用现有 `tests/test_gate*.py`、`test_adas_cortex_parity.py`、manifest、STATUS_BOARD；不得编造 PASS
- 末尾列「最小补测命令」清单

## 协作约定

- Cursor：改控制/训练/观测台与 #1 数字。
- agy：只读扫描 + 写矩阵 markdown；需要跑短测可跑，长训留给 Cursor。
- 两边都写回本文件「状态」列。
