# 原语普查与分类学对账 (S1, 2026-09-28)

> 只读普查，零底座改动，零跑批。工具：`tools/primitive_census.cpp`（新增）
> 产物：`runs/primitive_census_20260928.json`
> 复现：`cmake --build build --target primitive_census -j && ./build/primitive_census`

## 0. 为什么要做这一步

论文 Limitations 第 1 条：*"primitive-set necessity beyond the 6 used by the champion is not yet demonstrated"*。
本步不训练、不消融，只回答前置问题：**已入库的冠军制品到底用了哪些原语**。这是任何必要性宣称的地基。

## 1. 分类学对账：26 / 28 / 30

| 计数 | 归属 | 权威来源 |
|---|---|---|
| **28** | 执行原语 `sdsc_primitives` (ids 0-27) | `ops.yaml` v2.0.0 → `generated_ops.hpp:350` `std::array<PrimitiveMeta,28>` |
| **30** | 基因组 `CellType` 序列化码 | `generated_ops.hpp:60-91`；`is_valid_cell_type_code` 全 30 类；`test_cellular_core_contract.cpp:15` 断言 `all.size()==30` |
| 26 | — | **陈旧**。`ARCHITECTURE_DISCIPLINE.md:48` 与 `ADAS_SCALE_BENCHMARK_REPORT.md:14` 写 26，无代码依据 |

**裁决：28 是执行原语数，30 是基因组类型数，两者是不同空间。文档里的 26 应改为 28。**

## 2. 核心发现一：op 字节有两个命名空间，且 40/73 制品的原语身份不可恢复

`.bin` 制品的首个 op 字节**按版本属于不同命名空间**：

| 版本 | 文件数 | op 字节语义 | CellType 身份 |
|---|---:|---|---|
| v2 | **40** | **legacy SDSC opcode (0-27)** | **不可从制品恢复** |
| v3 | 2 | CellType 序列化码 | 权威 |
| v4 | 31 | CellType 序列化码 | 权威 |

载入器 `load_checkpoint_bin`（`cellular_genome.hpp:2670-2672`）已正确区分（`version>=3` 走序列化码，否则走 `sdsc_opcode_to_cell_type`），但对 v2 而言这是**有损再解释**——legacy opcode 26 无法区分 `Op_DelayN / Op_Oscillator / Op_Quadratic / Act_Channel / Pred_Sense0 / Pred_Sense1`（塌缩见 §3）。

**后果（必须写进任何原语必要性结论）**：凡 op 字节落在**塌缩集 {0, 23, 26}** 的 v2 制品，其 CellType 身份不可恢复（`adas_cortex_champion.bin`、`quant_cortical_array_champion.bin`、`fluid_damper_champion.bin` 等）。工具已按 `celltype_histogram_authoritative` 布尔位标记，聚合统计只采信 v3+。

精度补充（不可过度外推）：v2 的再解释是**flags 辅助**的，能消解受体/效应器层面的歧义（`sdsc_opcode_to_cell_type` 用 flags=1/2 区分），因此**当制品的 op 字节不落在塌缩集内时，重建是忠实的**——见 §5.3 的正面验证。不可恢复是 **op-26 六义性**造成的，不是所有 v2 一律报废。

**反例警告**：若不做版本/塌缩区分，会得出"`Gate_Inhibit` 被 18 个冠军使用"的错误结论——那 18 个全是 v2 制品的有损再解释，是假阳性。

## 3. 核心发现二：执行层塌缩是**既有设计**，非缺陷（本步为确认，非发现）

`cell_type_to_sdsc_opcode`（`generated_ops.hpp:230-264`）是多对一：

| opcode | 原语 | 塌缩的 CellType |
|---|---|---|
| 0 | SDSC_OP_SENSE_0 | Sense_Input0, Sense_Channel |
| 23 | SDSC_OP_ACT_RESET | Act_DefReset, Act_ImmuneLock |
| 26 | SDSC_OP_PASSTHRU | **6 类**：Op_DelayN, Op_Oscillator, Op_Quadratic, Act_Channel, Pred_Sense0, Pred_Sense1 |

该塌缩**已被冻结固化**：`tests/test_cellular_core_kernel.cpp:58-79` 以 `frozen_legacy_mapping` 逐条编码，`test_cellular_core_contract.cpp:27` 断言 `mapped_opcode == cell_type_to_sdsc_eval_op(...)`。

**训练/演化执行不走这条塌缩路径**，必须澄清以免误判：`dispatch_cell_forward`（`generated_ops.hpp:411`）传的是 `static_cast<uint8_t>(c.type)`，内核 `sdsc_cell_kernel_step_common`（`sdsc_cell_kernel.h:729`）按 `SDSC_CELL_*` 常量分派——该枚举镜像 CellType 码空间（`SDSC_CELL_OP_QUADRATIC = 20` ≡ `CellType::OP_QUADRATIC`）。**故 30 类在演化前向中各有独立语义。**塌缩只影响 legacy 契约表与 v2 制品往返。

**另：5 个执行原语从任何 CellType 都不可达**——`SDSC_OP_INTEGRATE(5)`、`AMPLIFY(6)`、`INVERT(7)`、`CLIP(9)`、`FATIGUE(25)`。普查确认这 5 个在原语直方图中从未出现，与映射分析一致。即"28 原语"中实际可由基因组表达的是 **至多 23 个 opcode**。

## 4. 核心发现三：ADAS L3 锁档制品不含原语异质性

原语同质制品（全部细胞共用**同一** op 字节）：

```
adas_cortex_champion.bin                    v2  210 cells  op=4   (flags: 192/12/6)
adas_cortex_champion_stadium.bin            v2  210 cells  op=4
adas_cortex_champion_tuned.bin              v2  210 cells  op=4
adas_cortex_champion_pre_l3_20260911.bin    v2  210 cells  op=4
adas_cortex_champion_live_exp.bin           v2  210 cells  op=4
```

**ADAS L3 锁档（论文与 README 的头号控制器）的制品里只有一个 uniform op + flags（192 自由 / 12 受体 / 6 效应器，合计 210）。** 其原语构成不在制品里，而在 `tools/train_adas_cortex.py` 的构造代码中——这与论文 §6 第 5 条自述吻合（`paper:396`：*"the ADAS 18-type histogram is the initialisation prior of `train_adas_cortex.py`"*）。

### 4.1 与论文 §3 第 2 条的正面冲突（本步最尖锐的一处）

`paper:154` 如此描述同一个文件：

> *"In the 192-hidden-cell ADAS driving cortex (`adas_cortex_champion.bin`), **all 18 primitives appear with near-uniform frequency (6~19 cells each)**. … Hence, this 18-type diversity is an initialization sampling artifact rather than evolved synergy"*

即论文称该制品含 **18 类原语**、每类 6~19 个细胞。而普查直读该制品：**210 个细胞的 op 字节全部为 4，无任何异质性**（192 flags=0 与论文的"192 hidden"吻合）。

**结论（可验证的硬事实）**：论文所描述的 18 类原语多样性**无法从随仓制品 `adas_cortex_champion.bin` 中恢复**。论文的 18 类来自训练期有机体（其自己指认的溯源点是 `train_adas_cortex.py:190`），而不是序列化产物；v2 序列化把逐细胞原语身份压成了单一 op。

注意两者给的原因不同，须分开陈述：
- 论文说"18 类是随机播种的**初始化采样假象**，选择压力尚未重塑"（这是关于**训练期**的判断）；
- 普查说"制品**根本没有**18 类，只有一个 op"（这是关于**产物**的事实）。

**前者是科学的自我批评，后者是产物的信息丢失。混为一谈会错判。**

### 4.2 对任何"ADAS 用了 N 类原语"宣称的约束

该制品的原语构成**只能**以"训练器先验"身份引用，且必须同时标注：
1. 制品本身不可验证（uniform op）；
2. 论文自述该先验是随机播种假象、未经选择压力重塑；
3. 若要得到可验证的 ADAS 原语表，**必须重训为 v4 或另存 CellType 级元数据**——这属于 S5 的前置工作。

## 5. 任务级冠军实际使用了什么（v3+ 权威制品，cells ≤ 200，共 25 个）

并集 **24 / 30** 类。频次（全 25 个制品累计）：

| CellType | 次数 | | CellType | 次数 |
|---|---:|---|---|---:|
| Sense_Channel | 56 | | Op_Integral | 11 |
| Op_EMA | 53 | | Sense_Input2/3 | 7 / 7 |
| Op_Sum | 38 | | Act_DefReset | 3 |
| Op_Sub | 26 | | Act_Channel | 2 |
| Sense_Input0 | 24 | | Gate_Threshold | 2 |
| Gate_Hysteresis | 23 | | Op_Multiply | 2 |
| Act_PosAction | 23 | | Op_Ratio | 1 |
| Act_NegAction | 23 | | **Op_Oscillator** | **1** |
| Gate_Deadzone | 21 | | Gate_And | 1 |
| Act_ImmuneLock | 17 | | Assoc_Hub | 1 |
| Op_Diff | 15 | | Sense_Input1 | 13 |
| Op_Abs | 13 | | | |

**任务级从未出现的 6 类**：`Gate_Inhibit`、`Gate_MinMax`、`Op_DelayN`、`Op_Quadratic`、`Pred_Sense0`、`Pred_Sense1`。

**在全部 73 个制品中从未出现（不分版本）**：`Pred_Sense0`、`Pred_Sense1`、`Op_DelayN` —— 这三个原语**从未被任何演化或构造使用过**。

### 5.1 对论文 §7.4 的直接冲击

论文 §7.4 预注册了 CPG 步态任务，理由是"用来检验 `OSCILLATOR` 原语（在 6 原语量化冠军中缺失）"。
但已入库的 `locomotion_gait_champion.bin`（v4, 11 细胞）构成为：

```
2 Op_EMA | 2 Op_Integral | 1 Sense_Input0 | 1 Sense_Input1 | 1 Op_Sub
1 Gate_Hysteresis | 1 Act_PosAction | 1 Act_NegAction | 1 Act_ImmuneLock
```

**一个 Oscillator 都没有。** 全任务级语料里 `Op_Oscillator` 只出现 1 次（`maze_tripartite_champion.bin`）。
→ §7.4 的 CPG 任务**并没有**验证到它声称要验证的原语，该预设前提与制品不符。

### 5.2 结构化接线 flagship 的原语构成由生成器决定，不由演化决定

`flagship_io_mix_*` / `flagship_io_evo_*`（8196 与 32772 细胞）构成是严格分层的：

```
4096 Sense_Channel | 512×8 各一层的原语 (EMA/Diff/Integral/Sum/Multiply/Abs/Quadratic/Threshold) | 4 效应器
```

每层是**单一原语的等量重复**——即 `structured_wiring.hpp` 的层单调构造写死了原语集。**这些制品的"13 类原语"不能作为"演化选择了 13 类原语"的证据。**

### 5.3 正面验证：论文 Limitations 第 1 条的锚点是可复现的 ✅

论文 Limitations 第 1 条表述为 *"the **quant champion** uses **6 of 28** primitives"*（`paper:396`）。其真实出处是 `paper §5.3.1` 的**单微柱玩具模型**（`FuturesQuantTask`，14 细胞 / 12 突触，Rebar），**不是** `quant_cortical_array_champion.bin`（1032 细胞）——这一点常被误读，务必分清。

普查直读 `checkpoints/quant_futures_champion.bin`（v2，14 细胞，22 突触）：

| 论文 §5.3.1 宣称 | 普查实测 | 一致 |
|---|---|---|
| `EMA/INTEGRATE` × 3 | `Op_EMA` × 3（`Op_Integral` × 0） | ✅ 计数一致（论文把 EMA 与 INTEGRATE 并为一个类别标签，计数无误） |
| `SUM` × 2 | `Op_Sum` × 2 | ✅ |
| `SUB` × 1 | `Op_Sub` × 1 | ✅ |
| `HYSTERESIS` × 1 | `Gate_Hysteresis` × 1 | ✅ |
| `DIFF` × 1 | `Op_Diff` × 1 | ✅ |
| `DEADZONE` × 1 | `Gate_Deadzone` × 1 | ✅ |

即 **6 个计算原语类 / 9 个计算细胞**，另有 2 受体 + 3 效应器 = 14 细胞。

**这是本步唯一的正面结论，且它很重要**：论文 Limitations 第 1 条不是空口自陈，而是**可从随仓制品逐项复现的**。这为该条 Limitations 提供了可审计的地基，也验证了本工具的重建保真度（该制品 op 字节未落入塌缩集，故 v2 重建忠实）。

注意口径边界：这只验证了"**量化玩具冠军用 6 类**"，**没有**验证"28 类里其余 22 类不必要"。后者仍需 S5 的消融。

## 6. 对必要性战役（H1）的直接含义

1. **可用的证据基础比预想窄**：CellType 级必要性只有 33 个权威制品可用；40 个 v2 制品（含 ADAS 与量化主冠军）出局。
2. **ADAS 与量化的原语构成不可从制品获得**，只能从训练器先验引用——任何"ADAS 用了 N 类原语"的说法都必须标注这是先验而非演化产物。
3. **必要性问题的真正形状**变了：不是"28 类都用到了吗"，而是
   - 5 个 opcode 从基因组**不可达**（§3）；
   - 6 个 CellType 在任务级**从未出现**（§5），其中 3 个在任何制品中都从未出现；
   - 任务级实际收敛到 **24 类**，且高度集中于 `EMA / Sum / Sub / Hysteresis / Deadzone / Diff / Integral`——与 §5.3 已验证的"6 类"锚点同一量级。
4. **白名单受限演化（S5 口径 B）是最干净的路径**：既然 6 类从未出现、7 类核心占了绝大多数，直接问"砍掉某类后演化还能不能收敛"比图内敲除更有信息量。

## 7. 本步**不能**说什么

- 不能说"28 原语里 5 个是死的"——它们在 legacy opcode 空间与 C11 运行时（`sdsc_binary_runtime.h:206` 走 `sdsc_primitive_eval`）中有效，只是**基因组无法产生它们**。
- 不能说"6 类原语执行时退化为 PASSTHRU"——训练前向按 CellType 分派，语义完整（§3）。
- 不能说"ADAS 用了 18 类原语"——制品是 uniform op，18 类出自训练器先验。
- 本步是**横截面普查**，不是消融；"未出现"≠"不必要"。
- 频次是跨**异构任务**的累计，不可解释为"重要性排序"。

## 8. 顺带发现的缺陷（另立提交，本次不动）

`tools/cellular_live_backend.py:59-122` 的 `read_sdsc_binary` 硬编码 4 字节细胞记录（`:76` `f.read(num_cells*4)`），遇 v4 制品（10 字节记录）会**静默错读**。本工具刻意绕开它，改用 C++ 权威载入路径。该 Python 侧缺陷影响前端展示口径，建议独立修复。

## 9. 复现与产物

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target primitive_census -j
./build/primitive_census                       # 默认扫 checkpoints/, 写 runs/primitive_census_<date>.json
./build/primitive_census --dir checkpoints --out runs/primitive_census_20260928.json
```

JSON 结构：`taxonomy`（分类学与命名空间）、`opcode_collisions`、`champions[]`（逐制品：`raw_op_histogram` / `celltype_histogram` / `celltype_histogram_authoritative` / `op_namespace`）、`aggregate`（含 `homogeneous_primitive_artifacts` 与从未观测清单）。
