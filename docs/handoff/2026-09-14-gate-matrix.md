# 六道门禁 × 生境实证打勾矩阵 (2026-09-14)

> **宪条纪律**：
> 1. 不修改 `include/kun/cellular/` 底座；
> 2. 绝对不编造 PASS；未找到证据一律如实标注「未测」，已知失败标注「FAIL」；
> 3. 证据仅引用仓库四项法定凭据：
>    - `tests/test_gate*.py` (`tests/test_gate5_gate6_replay_shadow.py`)
>    - `tests/test_adas_cortex_parity.py`
>    - `models/business_lifeforms/manifest.json` (Lifeforms Manifest)
>    - `docs/STATUS_BOARD.md` (底座双榜与历史战役记录)

---

## 一、 六道实证门禁体系定义

依据 `docs/ARCHITECTURE_DISCIPLINE.md` 及 `docs/morphogenetic_cellular_evolution_paper.zh.md`：
- **Gate 1 (基线探针)**：环境真实性与基线可解性探针 (Baseline Probe / Environment Health)
- **Gate 2 (选择收敛)**：代际演化与真实选择压力收敛 (Selection Pressure & Variance Convergence)
- **Gate 3 (OOD 盲测)**：合规物理样本外盲测与保留数据集泛化 (Holdout & OOD Generalization)
- **Gate 4 (纯 C 零 GC)**：纯 C 原生确定性零 GC、零堆分配与时延/位级对账 (Deterministic Zero-GC & C11 Parity)
- **Gate 5 (管线回放)**：生产级管线时序感知流离线全真回放 (Production Pipeline Offline Replay)
- **Gate 6 (影子对账)**：影子模式全工况差分对账与控制平滑度验证 (Shadow Mode Differential Audit)

---

## 二、 六道门禁 × 生境打勾矩阵 (Gate Matrix)

| 生境 (Habitat) | 锁档路径 (Checkpoint) | 核心测试/验证文件 | Gate 1<br>(基线探针) | Gate 2<br>(选择收敛) | Gate 3<br>(OOD盲测) | Gate 4<br>(纯C零GC) | Gate 5<br>(管线回放) | Gate 6<br>(影子对账) | 综合评级 / 状态速记 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ADAS**<br>(具身驾驶) | `checkpoints/adas_cortex_champion.bin`<br>*(210 细胞 / 659 突触)* | `tests/test_gate5_gate6_replay_shadow.py`<br>`tests/test_adas_cortex_parity.py`<br>`tests/test_adas_topology_ablation.py` | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **6/6 全闭环** + 战役 #4 拓扑消融 PASS（重连 34× / E-I 11×）。 |
| **maze**<br>(空间迷宫) | `checkpoints/maze_navigation_champion.bin`<br>*(11 细胞 / 15 突触)* | `tests/test_flow_maze_navigation.cpp`<br>`tests/test_flow_maze_gate5_replay.cpp` | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **未测** | **5/6**。测地冷评 96/100；G5 同种子两次独立回放位级重合；G6 未做。 |
| **doudizhu**<br>(斗地主) | `checkpoints/doudizhu_cand_scorer.bin`<br>*(82 细胞 / 425 突触)* | `tests/test_flow_doudizhu_card_game.cpp`<br>`tools/p9_runner.cpp` | **PASS** | **PASS** | **PASS** | **未测**<br>*(部分等价)* | **未测**<br>*(明确未做)* | **未测**<br>*(明确未做)* | **3/6**。使命胜率 57.0% 持平教师；位级等价已通但零堆分配未重测；STATUS_BOARD 明文严禁宣称 G5/G6。 |
| **cartpole**<br>(倒立摆) | `checkpoints/cartpole_balance_champion.bin`<br>*(13 细胞 / 49 突触)* | `tests/test_flow_cartpole_balance.cpp` | **PASS** | **PASS** | **PASS** | **PASS** | **未测** | **未测** | **4/6**。T4 易任务重训修复 initial_weight 同步；ID/OOD 冷评 20/20 满分；BIBO 零漂移证书；缺 G5/G6。 |
| **DomainZoo**<br>(动力学12域) | `checkpoints/domain_zoo_report.json`<br>`checkpoints/zoo_*.bin` *(12域)* | `tools/train_domain_zoo.cpp` | **PASS**<br>*(现仓12/12)* | **PASS** | **PASS** | **未测** | **未测** | **未测** | **3/6**。2026-09-14 复跑锁档 **12/12**（战役 #3）；历史 9/12 仅为中间窗，已勘误；无单独 C11/回放。 |
| **household**<br>(全屋覆盖) | `checkpoints/household_coverage_champion.bin`<br>*(11 细胞 / 11 突触)* | `tests/test_flow_household_coverage.cpp` | **PASS** | **PASS** | **PASS** | **PASS** | **未测** | **未测** | **4/6**。T2 战役根除 reset_state 覆盖权重缺陷；ID 88.2%，OOD 大户型 83.1%，回充 100%；BIBO 证书；缺 G5/G6。 |
| **quant**<br>(量化皮层) | `checkpoints/quant_cortical_array_champion.bin`<br>*(1032 细胞 / 1634 突触)* | `tools/train_multi_asset_cortical_array.cpp`<br>`tools/eval_cortical_array_bin.cpp`<br>`tests/test_quant_oos_discipline.py` | **PASS**<br>*(稳态基线)* | **PASS**<br>*(复训过拟合)* | **FAIL**<br>*(锚点 0.22 未复现)* | **未测**<br>*(C11 未测)* | **未测** | **未测** | **1/6 (G3 对锚点仍 FAIL)**。`--seed 0` 复训 OOS -0.03 冻结；正式 bin 任务层冷评 OOS **+0.14**（不是 0.22）。已撤实盘宣称。 |

### 补充生境清单 (Manifest 登记其余生命体)

| 生境 / 生命体 | 锁档路径 | 核心测试文件 | G1 | G2 | G3 | G4 | G5 | G6 | 证据说明 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **fluid_damper** (流体阻尼) | `checkpoints/fluid_damper_champion.bin` | `tests/test_flow_sota_benchmark.cpp` | PASS | PASS | PASS | PASS | 未测 | 未测 | Lyapunov 最大增益 ρ=0.021，响应 4.2ms，振动衰减提升 4.8x。 |
| **locomotion_gait** (多足步态) | `checkpoints/locomotion_gait_champion.bin` | `tests/test_demo_runtime_parity.py` | PASS | PASS | PASS | 未测 | 未测 | 未测 | CellularOrganism 真前向驱动肌肉；Train/ID/OOD 均 100%。 |
| **slingshot_nav** (三体引力弹弓) | `checkpoints/slingshot_nav_champion.bin` | `tests/test_demo_runtime_parity.py` | PASS | PASS | PASS | 未测 | 未测 | 未测 | 混沌引力场导航；Train 100% / ID 80% / OOD 50% (G=1.6)。 |
| **music_composer** (声学音高) | `checkpoints/music_composer_cortex.bin` | `tests/test_music_composer_cortex.py` | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 | 玩具演示体；仅测 BIN 头解析与清单，明确不构成生成能力宣称。 |

---

## 三、 细项证据核验溯源

### 1. ADAS (具身智能驾驶)
- **Gate 1 (基线探针)**：`PASS`。`models/business_lifeforms/manifest.json` 记录 16 工况多种子 vs 经典 Stanley 控制器基准对撞检验；Lyapunov 稳态最大环路增益检验通过。
- **Gate 2 (选择收敛)**：`PASS`。Manifest 记录经 sep-CMA-ES 增益微调收敛，直道 CTE 降至 3.1cm，高速 CTE 17cm。
- **Gate 3 (OOD 盲测)**：`PASS`。`tests/test_gate5_gate6_replay_shadow.py` 实测留出验证集 4/4 场景（`VAL_SCENARIOS`: val_s_curve, val_curve, val_highway, val_stop_go）全步数跑满通过；Manifest 记录 seeds 1-10 盲测 9W/7L (56.3% 净胜率)。
- **Gate 4 (纯 C 零 GC)**：`PASS`。`tests/test_adas_cortex_parity.py` 经 pytest 实测 PASS（400 帧 C11 与 Python 差分 $\max |\Delta| < 10^{-5}$，18 原语全覆盖）；Manifest 签署 50,000 步位级零漂移形式化证书 (`adas_cortex_champion.bin.cert.json`)。
- **Gate 5 (管线回放)**：`PASS`。`tests/test_gate5_gate6_replay_shadow.py::run_gate5_pipeline_offline_replay()` 实跑通过：480 步离线回放，两次独立回放轨迹最大差分 CTE = 0.00e+00 m，转向 = 0.00e+00 rad，平均 CTE 31.29 cm (< 40.0 cm)。
- **Gate 6 (影子对账)**：`PASS`。`tests/test_gate5_gate6_replay_shadow.py::run_gate6_shadow_mode_differential_audit()` 实跑通过：4 留出工况平均转向抖动 4.87 mrad/step (< 10.0 mrad 车规要求)，平均纵向速度跟踪偏差 0.39 m/s。

### 2. maze (空间迷宫导航)
- **Gate 1 (基线探针)**：`PASS`。Manifest 与 wrap 报告记录纯欧氏直线方位探针在死胡同反向梯度失效 (仅 80–86/100)，任务层引入测地方位脚手架 (BFS 势场) 恢复基线可解性。
- **Gate 2 (选择收敛)**：`PASS`。Manifest 记录 11 细胞微柱在测地先验下稳定演化，权重与初始权重严格同步。
- **Gate 3 (OOD 盲测)**：`PASS`。Manifest 记录未参与选择的盲测迷宫中 250 步 96/100，400 步 100/100；`STATUS_BOARD.md` 记录 T4 L3-M1/M2/M3 ✅。
- **Gate 4 (纯 C 零 GC)**：`PASS`。`maze_navigation_champion.bin.cert.json` 签署 BIBO 稳态与 50,000 步位级零漂移证书。
- **Gate 5 (管线回放)**：`PASS`。`tests/test_flow_maze_gate5_replay.cpp`：同一 `maze_navigation_champion.bin`、同一未见种子，两次独立加载回放，位姿与动作最大差分 < 1e-6。
- **Gate 6 (影子对账)**：`未测`。无专家控制器双轨影子脚本。

### 3. doudizhu (斗地主博弈打分)
- **Gate 1 (基线探针)**：`PASS`。`STATUS_BOARD.md` 记录 P0 教师对照验证环境健康可解；完整规则引擎与启发式对手基准正常。
- **Gate 2 (选择收敛)**：`PASS`。`STATUS_BOARD.md` 记录 M2 对手池课程跃迁产物（57.0% 学习收敛至教师水平 58.3%，McNemar $p=0.779$ 持平）；自发超越教师未成立如实标注。
- **Gate 3 (OOD 盲测)**：`PASS`。`STATUS_BOARD.md` 记录跨局 holdout 600 决策正确率 89.33%~90.33%，使命口径 2000 局胜率 57.0% (Wilson95 下界 54.8% > 50%)；Manifest 撤销旧 82.5% 简化环境宣称。
- **Gate 4 (纯 C 零 GC)**：`未测 (部分等价)`。`STATUS_BOARD.md` P9 支持矩阵记录：“✅ C 路径前向与新核心双执行器位级等价；❌ legacy 零堆分配语义未重测，历史单步 411.8μs Zero-GC 宣称未在新核心复测”。
- **Gate 5 & 6 (管线回放 / 影子对账)**：`未测 (明确未做)`。`STATUS_BOARD.md` 第 80 行铁律明确写下：“❌ Replay/Shadow 门禁（未做，不得宣称）”。

### 4. cartpole (倒立摆姿态平衡)
- **Gate 1 (基线探针)**：`PASS`。刚体动力学摆杆姿态基线可解，空白受精卵与启发式探针通过。
- **Gate 2 (选择收敛)**：`PASS`。Manifest 记录 T4 重训并同步 `initial_weight`，根除权重抹掉缺陷。
- **Gate 3 (OOD 盲测)**：`PASS`。Manifest 记录【T4 易任务锁档】ID 与 OOD 冷评均为 20/20 满分；`STATUS_BOARD.md` T4 C1 ✅。
- **Gate 4 (纯 C 零 GC)**：`PASS`。`cartpole_balance_champion.bin.cert.json` 签署 BIBO 稳态与 50,000 步位级零漂移证书。
- **Gate 5 & 6 (管线回放 / 影子对账)**：`未测`。未针对倒立摆开发管线离线回放或影子对账测试。

### 5. DomainZoo (物理动力学控制动物园 12 域)
- **Gate 1 (基线探针 / M1 Gate)**：`PASS` *(现仓 12/12，战役 #3 已统一)*。`STATUS_BOARD.md` Line 101 记录“T1 ✅ 12/12”；2026-09-14 `./build/train_domain_zoo` 复跑 12/12；论文旧稿 9/12 仅为 2026-09-09 中间窗，已在双语论文勘误。
- **Gate 2 (选择收敛)**：`PASS`。`domain_zoo_report.json` 记录 12 域 train_sr 达 0.9 ~ 1.0。
- **Gate 3 (OOD 盲测)**：`PASS`。`domain_zoo_report.json` 记录 12 域 ood_sr 达 0.6 ~ 1.0 (均值 > 0.8)。
- **Gate 4 (纯 C 零 GC)**：`未测`。未见针对 12 域各紧凑 `.bin` 的独立纯 C 零 GC 形式化测试。
- **Gate 5 & 6 (管线回放 / 影子对账)**：`未测`。四证据源中无记录。

### 6. household (全屋覆盖清洁机器人)
- **Gate 1 (基线探针)**：`PASS`。有限空间遍历与避障环境基线探针可解。
- **Gate 2 (选择收敛)**：`PASS`。Manifest 记录 T2 战役修复 `reset_state(true)` 抹除权重的缺陷，同步 `initial_weight` 后 50 种子基准收敛；`STATUS_BOARD.md` T2 ✅。
- **Gate 3 (OOD 盲测)**：`PASS`。Manifest 记录 ID 覆盖率 88.2%，OOD 异构大户型覆盖率 83.1%，安全回充率 100%，障碍愈合率 100%。
- **Gate 4 (纯 C 零 GC)**：`PASS`。`household_coverage_champion.bin.cert.json` 签署 BIBO 零漂移证书。
- **Gate 5 & 6 (管线回放 / 影子对账)**：`未测`。四证据源中无记录。

### 7. quant (多资产量化皮层阵列 / 三方稳态微柱)
- **Gate 1 (基线探针)**：`PASS` *(仅三方稳态基线)*。Manifest 记录 9 细胞 `quant_tripartite_champion.bin` Lyapunov 最大增益 $\rho=0.019 < 1.0$，BIBO 严格收敛。
- **Gate 2 (选择收敛)**：`PASS` *(但严重过拟合)*。Manifest 记录 1032 细胞阵列在 43 商品期货训练集上夏普达 2.16。
- **Gate 3 (OOD 盲测)**：`FAIL` *(对论文锚点 0.22)*。`--seed 0` 复训 OOS -0.03 冻结。2026-09-14 正式 `champion.bin` 任务层冷评 OOS 夏普 **+0.14** / +6.3% / 回撤 25.6%（`eval_cortical_array_bin`），**不得**写成 0.22 已复现，也不得改冻结 JSON。
- **Gate 4 (纯 C 零 GC)**：`未测`。任务层 loader 已补；C11 位级对账仍未做。
- **Gate 5 & 6 (管线回放 / 影子对账)**：`未测`。Manifest 虽在 `quant_tripartite_champion` 条目标注 `test_gate5_gate6_replay_shadow.py`，但实际该脚本仅有 ADAS 回放与影子对账逻辑，未执行 quant 实测。

---

## 四、 最小补测命令清单 (Minimal Test Suite)

为在本地快速验证或为未通生境补齐门禁凭证，提供如下最小复现/补测命令清单：

### 1. 快速回归与门禁对账 (Pytest 短测)
```bash
# ADAS C11 导出体逐帧对账 (Gate 4 实测)
pytest tests/test_adas_cortex_parity.py -v

# ADAS 门禁 5 (离线回放) 与门禁 6 (影子模式差分对账) 实弹测试
python3 tests/test_gate5_gate6_replay_shadow.py

# ADAS 契约与运行时符号导出验证
pytest tests/test_adas_cortex_contract.py tests/test_c_runtime_backend_parity.py -v
```

### 2. 易证三任务冷评回归 (CartPole / Maze / Fluid)
```bash
# 验证 CartPole ID/OOD (20/20)、Maze 测地脚手架冷评 (96/100) 及 Fluid 拓扑
./build/bench_easy_task_regression

# 流体全动力学应力测试
./build/test_multiphase_fluid_stress
```

### 3. DomainZoo 物理动力学 12 域复测
```bash
# 运行 12 经典控制域演化训练与 M1 门禁判定
./build/train_domain_zoo
```

### 4. 斗地主打分器基准与天梯评测
```bash
# 2000 局天梯胜率评测
./build/p9_runner 2000 checkpoints/doudizhu_cand_scorer.bin

# 配对教师对照显著性检验 (McNemar)
./build/p9_paired 2000 checkpoints/doudizhu_cand_scorer.bin checkpoints/doudizhu_cand_scorer.bin 3100000 teacher
```

### 5. 全屋清洁覆盖机器人回归
```bash
# 验证覆盖率 (ID 88.2% / OOD 83.1%) 及 initial_weight 同步
./build/test_flow_household_coverage
```

### 6. 量化多资产皮层阵列复现 (负例断言)
```bash
# 复现 --seed 0 选择集过拟合与 OOS 盲测夏普 -0.03 负例
OMP_NUM_THREADS=6 ./build/train_multi_asset_cortical_array --seed 0 --report runs/quant_repro.json

# 正式 champion.bin 任务层冷评（不覆盖 bin，不改冻结 JSON）
./build/eval_cortical_array_bin --bin checkpoints/quant_cortical_array_champion.bin --dry-load
python3 tests/test_quant_cortical_array_cold_eval.py
```
