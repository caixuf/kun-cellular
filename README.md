# KunCellular: 软件定义硅基细胞演化计算框架

[![CI](https://github.com/caixuf/kun-cellular/actions/workflows/ci.yml/badge.svg)](https://github.com/caixuf/kun-cellular/actions)
![C++20](https://img.shields.io/badge/C++-20-blue.svg)
![C11](https://img.shields.io/badge/C-11-555555.svg)
![Zero-GC](https://img.shields.io/badge/Memory-Zero--GC-emerald.svg)
![Latency](https://img.shields.io/badge/Latency-19.06ns-cyan.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 基于冯·诺依曼自复制自动机理论、26 种算存一体原子动力学原语与 Kahn 拓扑排序编译器的**软件定义硅基细胞计算框架（Software-Defined Silicon Cellular Computer, SDSCC）**。
>
> 核心价值：~11,000 行纯 C/C++ 底座，零动态堆分配（Zero-GC），确定性实时推演，支持在小规模控制任务上通过演化算法得到可运行智能体。

---

## ⚠️ 诚实声明（Honest Disclosure）

本仓库经过系统性自查审计（2026-09）。下方"能力矩阵"仅列出有真实检查点、真实基准数据支撑的结果。以下内容已从本 README 删除：

- **10 个"业务生命体"**（`runs/ten_business_organisms_catalog.json`）：量子密码卫士、德州扑克、蛋白质折叠、聚变控制等——所有 10 个检查点文件均不存在，系纯目录条目。
- **空壳大模型**：`runs/real_billion_champion.pt`（1.6 KB）/ `runs/cellular_language_model_1b.pt`（1.5 KB）解包验证仅含元数据，非真实参数；真正的十亿参数模型需 ≥4 GB。
- **斗地主"OOD 胜率 82.5%"**：训练/测试环境为简化合成仿真（无叫分、无合法牌型判定、对手为随机数发生器），该数字对真实三人斗地主无参考价值。
- **量化收益宣称**：合成回测环境，无实盘验证，已全部移除。
- **ADAS 基准择优引用**：完整 16 场景数据为 **9 胜 / 7 负**，原 README 仅列赢的场景。
- **音乐皮层**：权重 = 随机初始化 + 高斯变异若干代，属声光玩具演示，不构成音乐生成能力宣称。

---

## 一、真实工程资产

### 1. C/C++ 核心底座（~11,000 行，可复现）

| 模块 | 文件 | 说明 |
| :--- | :--- | :--- |
| **26 原语动力学库** | `include/kun/cellular/cellular_genome.hpp` | 26 种算存一体细胞原语（OSCILLATOR、EMA、INTEGRATE、GATE_HYSTERESIS 等）、李雅普诺夫 BIBO 稳定性判定器 |
| **CSR 稀疏运行时** | `include/kun/cellular/sdsc_binary_runtime.h` | 纯 C11 SDSC-BIN v2 零拷贝 mmap 二进制运行时 |
| **基础皮层** | `include/kun/cellular/sdsc_cortex.h` | 纯 C11 零 GC 基础控制皮层单头文件 |
| **Apex 皮层** | `include/kun/cellular/sdsc_apex_cortex.h` | 5 微柱复合机动皮层，实测 19.06 ns/step |
| **演化引擎** | `include/kun/cellular/island_evolution_grid.hpp` | 8 岛屿拓扑网格迁移演化 |
| **生态圈** | `include/kun/cellular/ecosystem_biosphere.hpp` | 多相生态圈与食物网 |

### 2. 硬实时微架构（可复现基准）

```bash
./build/test_c11_apex_maneuver
# 1,000,000 iterations: ~19 ns/step, ~52 M-Inf/s, 0 malloc / 0 free
```

| 指标 | 实测值 |
| :--- | :--- |
| 单步推理时延 | **19.06 ns** |
| 峰值吞吐 | **52.47 M-Inf/s** |
| 堆内存申请 | **严格 0 字节** |
| P99 时延抖动 | **179 ns** |

---

## 二、真实演化实证矩阵（有检查点 + 有基准数据）

以下所有条目均有对应 `.bin` 检查点文件和可复现的测试/基准数据。

| 任务 | 规模（细胞 / 突触） | 检查点 | 实测结果 |
| :--- | :--- | :--- | :--- |
| **CartPole 平衡** | 12 细胞 / 21 突触 | `checkpoints/cartpole_balance_champion.bin` (1.5 KB) | Train SR=100%，ID-Holdout SR=100%，OOD SR=100% |
| **空间迷宫自主脱困** | 13 细胞 / 18 突触 | `checkpoints/maze_navigation_champion.bin` (700 B) | 100 轮随机迷宫 100% 成功逃逸，0 死锁 |
| **流体阻尼控制** | 40 细胞 / 145 突触 | `checkpoints/fluid_damper_champion.bin` (5.0 KB) | Aero / Hydro / Vacuum 三态 3000 步极限扰动 100% 收敛 |
| **ADAS 循迹皮层** | 210 细胞 / 630 突触 | `checkpoints/adas_cortex_champion.bin` (56 KB) | 详见下方完整基准表（9W / 7L） |
| **12 任务控制动物园** | 8~18 细胞 / 9~47 突触 | `checkpoints/zoo_*.bin` | 12/12 门禁通过，训练耗时 1.4~4.6 秒/任务 |

### ADAS vs Stanley 完整基准（16 场景，10-seed 平均）

> 数据来源：`runs/adas_champion_vs_stanley_seeds1-10.json`（完整 10-seed，未过滤）

| 场景 | 数据集 | Champion 均值 | Stanley 均值 | 结果 |
| :--- | :---: | :---: | :---: | :---: |
| straight_cruise | train | 0.0338 | 0.0286 | ✅ 胜 |
| gentle_s | train | 0.0687 | 0.0640 | ✅ 胜 |
| s_curve | train | 0.1124 | 0.1394 | ❌ 负 |
| s_curve_mid | train | 0.2712 | 0.2173 | ✅ 胜 |
| s_curve_hard | train | 0.2664 | 0.2148 | ✅ 胜 |
| curve_easy | train | 0.1840 | 0.2209 | ❌ 负 |
| tight_curve | train | 0.1309 | 0.1661 | ❌ 负 |
| tight_curve_max | train | 0.1556 | 0.1559 | ❌ 负（微弱） |
| stop_go | train | 0.0725 | 0.0519 | ✅ 胜 |
| follow | train | 0.1119 | 0.1061 | ✅ 胜 |
| highway | train | 0.1323 | 0.1409 | ❌ 负 |
| ramp_merge | train | 0.1269 | 0.1672 | ❌ 负 |
| **val_s_curve** | **val** | **0.3430** | **0.2585** | ✅ **胜** |
| **val_curve** | **val** | **0.1707** | **0.2030** | ❌ **负** |
| **val_highway** | **val** | **0.2430** | **0.2076** | ✅ **胜** |
| **val_stop_go** | **val** | **0.0762** | **0.0568** | ✅ **胜** |
| **汇总** | | | | **9 胜 / 7 负（16 场景）** |

> 说明：Champion 在平顺 S 形弯道中段和跟随启停场景具备优势，在宽幅易弯、高速公路和匝道汇入场景中输给 Stanley。这是一个有部分竞争力但尚未全面超越的小规模控制皮层。

---

## 三、理论体系：26 原语动力学演化框架

系统通过 **Kahn 拓扑排序编译器**将有向图展平为无循环计算序列，并使用以下四个演化机制驱动拓扑搜索：

| 演化机制 | 实现 | 数学保证 |
| :--- | :--- | :--- |
| **超细胞共生微柱** | `SymbioticMacroCell` | 接口绝缘，消除高维搜索爆炸 |
| **器官冷冻库** | `OrganFrozenBank` | 跨任务借用已验证微柱，加速冷启动 |
| **李雅普诺夫 BIBO 稳定性** | Tarjan 环检测 + 谱半径约束 `ρ < 1.0` | 杜绝数值发散与自激振荡 |
| **大灭绝算子** | 连续停滞后抹杀前 80% 头部拓扑 | 激发边缘变异，防止早熟收敛 |

---

## 四、前端交互沙盒

平台提供浏览器可访问的实时仿真沙盒（监听端口 `8833`）：

- **斗地主对战 (`frontend/doudizhu.html`)**：标准三人斗地主规则界面，支持人机对战与 AI 全自动观战。AI 出牌逻辑为启发式规则，**非真实神经网络推演**。
- **3D 细胞观测台 (`frontend/cellular.html`)**：WebGL 实时渲染细胞微柱网络，LOD 动态调度。
- **自动驾驶仿真 (`frontend/vehicle.html`)**：210 细胞 ADAS 皮层闭环仿真可视化。
- **迷宫脱困 (`frontend/maze.html`)**：13 细胞代理实时避障演示。
- **硅基天籁钢琴 (`frontend/music.html`)**：WebAudio 物理建模钢琴音色 + SDSC-BIN v2 权重浏览器内推演（**玩具级声光演示**；权重 = 随机初始化 + 高斯变异，不构成"音乐智能"宣称）。
- **生物圈生态、引力弹射、百足虫步态、免疫猎杀**等演示沙盒。

---

## 五、仓库架构

```
kun-cellular/
├── include/kun/cellular/          # C/C++ 核心底座（~11,000 行）
│   ├── cellular_genome.hpp        # 26 原语 + BIBO 稳定性判定器
│   ├── sdsc_binary_runtime.h      # SDSC-BIN v2 零拷贝 mmap 运行时
│   ├── sdsc_cortex.h              # 基础自动驾驶皮层（Zero-GC）
│   ├── sdsc_apex_cortex.h         # 5 微柱 Apex 复合皮层（19.06 ns）
│   ├── island_evolution_grid.hpp  # 8 岛屿拓扑演化
│   └── ecosystem_biosphere.hpp    # 多相生态圈
├── frontend/                      # 浏览器交互沙盒
├── checkpoints/                   # 真实演化产物（SDSC-BIN v2 二进制检查点）
├── tools/                         # 演化工具链与后端网关
├── tests/                         # 34 组回归测试（ctest 34/34 PASS）
└── runs/                          # 基准数据（含完整 ADAS 10-seed 结果）
    ├── adas_champion_vs_stanley_seeds1-10.json  # 完整基准（9W / 7L）
    └── domain_zoo_report.json                   # 12 任务控制动物园
```

---

## 六、构建与复现

```bash
# 编译与测试
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
# 34/34 Test suites passed

# 纳秒级推理基准
./build/test_c11_apex_maneuver

# 流体压测
./build/test_multiphase_fluid_stress

# 启动沙盒服务
python3 tools/cellular_live_backend.py --port 8833
# http://localhost:8833/
```

---

## 七、项目定位

**KunCellular 是什么：**
- 约 11,000 行 C/C++ 演化计算底座，26 种动力学原语构成的通用算存一体框架
- 在玩具级控制任务（CartPole、迷宫、流体阻尼、定速巡航）上，演化算法可在数秒内得到小规模（8~40 细胞）的可用控制器
- ADAS 皮层在 16 场景中击败 Stanley 基准 9 次（输 7 次），部分场景具备竞争力
- 适合研究"极小神经元数量下非冯·诺依曼动力学"的教学/实验平台

**KunCellular 不是什么：**
- 不是经过验证的百万/十亿参数世界模型（相关 `.pt` 文件为空壳）
- 不是可量化交易的金融系统（合成回测，无实盘验证）
- 不是真实斗地主 AI（训练环境为简化合成仿真，非完整三人牌局）
- 不是量子密码/蛋白质折叠/聚变控制等"业务生命体"（检查点全部不存在）

---

## 八、许可证

本项目采用 MIT 许可证。
