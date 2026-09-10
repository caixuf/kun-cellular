# 交接：T0「百万细胞演化出可用有机体」— Stage A（结构化发育接线）2026-09-10

## 0. 一句话现状
**机制成立、假设否证**：结构化接线**证明**消除坍缩（活性比 1.0000），但**本实现远劣于同规模随机接线**（H4 反向）；并发现 **Oja 塑性是掩盖真实信号的强混淆变量**——去掉后结构化臂反而能超越持续性基线。

## 1. 主题与目标
- **T0**：让百万细胞级有机体在 `FieldCML2DTask`（512² 体素时空混沌场）上**学到有用预测**（不坍缩 + 超越持续性基线），而非像上一轮自坍缩（330K→43、1M→9）。
- **两道墙**：① 发育接线（← 本阶段）② eval 批量 rollout（→GPU，后置）。
- **节奏**：先机制（64K–256K）后放大（1M）。**L0 零修改**；改动全落 L1(`include/kun/cellular/population/`) 与 L2(`tools/`)。
- 预注册：`docs/superpowers/plans/2026-09-10-structured-developmental-wiring.md`（阈值冻结，结果已回填）。

## 2. 交付物（全部未提交，工作区）
| 文件 | 层级 | 说明 |
|---|---|---|
| `include/kun/cellular/population/structured_wiring.hpp`（新） | L1 | `ColumnLatticeSpec`/`ReadoutAnchor`/`build_columnar_structured_wiring`/`active_cell_count`/`randomize_internal_synapse_targets` |
| `include/kun/cellular/population/individual_traits.hpp`（改） | L1 | 新增 `MutationTrait` seam（默认透传 `Individual::mutate`） |
| `include/kun/cellular/population/deme.hpp`（改） | L1 | `evolve_generation` 经 `MutationTrait` 调 mutate |
| `tests/test_population_structured_wiring.cpp`（新） | L1 | H1/H2 + 局部性 + 确定性（已过）|
| `tools/train_flagship_wired.cpp`（新） | L2 | 对照 trainer（S / S⁻ᵃ / R′；训练塑性开关）|
| `CMakeLists.txt`（改） | 构建 | 注册 `train_flagship_wired` |
| `docs/superpowers/plans/2026-09-10-structured-developmental-wiring.md`（新） | 文档 | 预注册 + 结果 |

## 3. 关键结果
**机制（结构口径）**：H1 ✅ 活性比 **1.0000**（n=128 → 81,924 细胞；n=64 → 8,196）；H2 ✅ DAG+同种子逐字段一致；局部性 ✅。

**训练对照（n=64, G=16 → 8,196 细胞, POP=12, 80 代, MS=200，**注意规模低于预注册 64K**）**：

| 臂 | 训练塑性 | 训练最佳 | 未见(id) | 未见(ood体制) |
|---|---|---|---|---|
| S 结构化 | 开（预注册口径） | 0.0187 | 0.0045 | 0.0047 |
| S⁻ᵃ 去锚定 | 开 | 0.0213 | 0.0044 | 0.0045 |
| **R′ 随机同规模** | 开 | **0.6706** | **0.2237** | **0.1861** |
| S 结构化 | **关**（post-hoc） | 0.3904 | **0.1226** | 0.0991 |
| R′ 随机同规模 | 关（post-hoc, gen40） | 0.7576 | — | — |
| P 持续性基线 | — | 0.0547 | | |

## 4. 关键发现（务必读）
1. **H1 机制成立**：结构化分层 DAG（层单调 + 局部 + 每非终端细胞出边必达效应器）**可证明**零坍缩——旧随机接线的坍缩是结构性缺陷，此结论可靠。
2. **H4 反向（核心否证）**：同细胞数/同突触数下，**随机拓扑碾压结构化**（0.2237 vs 0.0045，~50×；关塑性下 0.7576 vs 0.3904）。「空间结构化接线更优」**未被支持**。
3. **Oja 塑性是强混淆**：同一 S 冠军，塑关评 0.1226 / 塑开评 0.0030。塑开训练把选择压力引向「抗塑性破坏」而非「预测准」。**去掉塑性后 H3 反而成立**（+0.068 ≥ +0.02，post-hoc）。
4. **两个可疑根因（未证）**：① **幅度衰减**——内部细胞 `param1=0.1` 逐层相乘 + `w_readout=0.005` 过小 → 效应器输出趋零（训练最佳 0.0187 与 q=exp(-0.5/0.15)≈0.036 吻合）；② **缺递归**——结构化为严格 DAG，而 R′ 打乱端点后由 `compile()` 环保护补齐产生 `is_recurrent` 单拍反馈，随机臂意外获得递归记忆。

## 5. 未解决难题 / 下一步（按优先级）
1. **定位「结构化为何输」**：先做 ① 幅度校准（`param1`→1.0、增大 `w_readout`、或加直接受体→效应器残差），② 递归必要性（给结构化加入层间反馈边）——分别单独消融。**这是决定 T0 走「结构化皮层」还是「随机+递归」路线的关键实验。**
2. **H7 未过**：S 的 ood体制/训练 ≈ 0.25 < 0.70，同尺寸异动力学泛化差。
3. **规模放大**：本阶段仅 8,196 细胞（低于预注册 64K）；需在 n=128（G=64,b=2 → 81,920）/ n=256 重跑 S 与 R′（CPU 已可实现，但每代耗时随细胞数增长，需酌情减种子/步数或上 GPU）。
4. **GPU eval（第二道墙，未启动）**：eval = 前向×步数 是任务口径主成本；设施盘点见 `docs/superpowers/plans/2026-09-10-l1-parallel-evolution-engine.md` 与 GPU 报告（RTX 5060 8GB/CUDA13；`sdsc_cuda_runtime.hpp` 单个体 1M 前向 0.103ms；需新建 L1 GPU BatchEvaluator + 稀疏批量内核）。

## 6. 如何复现
```bash
cmake -S . -B build && cmake --build build --target train_flagship_wired test_population_structured_wiring -j
./build/test_population_structured_wiring                 # H1/H2/局部性
# 用法: ./train_flagship_wired [N] [GENS] [ARM] [G] [train_plast]
OMP_NUM_THREADS=12 ./build/train_flagship_wired 64 80 structured 16 1   # S，塑开(预注册口径)
OMP_NUM_THREADS=12 ./build/train_flagship_wired 64 80 structured 16 0   # S，塑关(post-hoc)
OMP_NUM_THREADS=12 ./build/train_flagship_wired 64 80 random     16 0   # R′
OMP_NUM_THREADS=12 ./build/train_flagship_wired 64 80 noanchor   16 0   # S⁻ᵃ
./build/train_flagship_voxel 256 60 60                    # 旧随机接线对照臂 R
```
约束：`N` 必须是 `G` 的倍数（场边长 = G·b）；G=16 快(~4-7s/代)、G=64 慢。

## 7. 纪律 / 边界 / 陷阱
- **L0 零修改**（`include/kun/cellular/` 非 population 部分）。结构化接线只消费 L0 公开 API。
- **Stage A 冻结拓扑**：`MutationTrait<CellularOrganism>` 特化只扰动权重/param1（同步改 `weight` 与 `initial_weight`——`reset_state(true)` 每 episode 从 `initial_weight` 复原）。**切勿调用 L0 `evolve_generation()`**（含结构变异+凋亡，会毁掉拓扑）。
- **别重走已证伪的路**：`develop_to_scale` 随机链 + `wire_global_bridge` 随机桥（无任务梯度、被活性集/凋亡清零）。
- **预注册纪律**：阈值已冻结；H3 的 post-hoc ✅ 不等于预注册口径通过，两者必须分开报告。
- 陷阱：并行/线程相关 benchmark 需多次采样且勿并发跑（WSL2 离群，见 memory `feedback_benchmark_discipline`）。
