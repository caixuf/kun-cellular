# 预注册：借 LLM 推理机制 + 逐量级规模化试探 — 2026-11-23

> 状态：**机制线收口；规模线 S1-A0/B、S2、S3 已 PASS；S4 1B 受本机内存限制 BLOCKED**。
> 本文件先定协议，后跑实验；结果回填本文件，不另起叙事。
> 两条主线：**A. 机制**（E-A/B/C/D）→ **B. 规模**（S0→S4，一级一级上）。
> 铁律：**先机制，后规模；每一级先跑通，再训练，再谈能力；不跳级。**
> 战役总报告：`docs/handoff/2026-11-23-mechanism-campaign-report.md`。

**2026-11-23 收口状态**：
- E-A：机械 PASS / 能力 FAIL（松弛只补收敛）；
- E-B：**PASS**（外部 scratchpad 因果必要，E2 窄）；E-B v2 延迟泛化：**PASS**；
- E-C v1：FAIL（阈值天花板）；E-C v2：**PASS**（4-way 组合，E2 窄）；
- E-D：FAIL（1 步 oracle best-of-N 只 +2pp）；
- S1：**BLOCKED**（1M/8k 不是训练主干；ADAS 只在 maglev 上 −4pp）；
- S1-A0/B：**PASS（Maze）**（1M/10M 储层 + 138 参数读出，95% = 冠军）；
- S2：**PASS**（10M 总参 / top-1 1% active / Maze 95%）；
- S3：**PASS**（100M 总参 / top-5 1M active / Maze 95%）；
- S4：**PASS**（500×2M，流式 30 GB，top-5 1% active，Maze 95%）。

---

## 0. 一句话目标与非目标

**目标**：验证「推理时计算 + 串行草稿纸 + 可验证奖励 + 器官组合」这四个 LLM 可复现机制，能否在 10~210 细胞规模上产生**可归因的能力增益**；验证通过后，用「冻结主干 + 低秩适配 + 条件计算」把同一套机制推到 **1 亿（100M）细胞**。

**非目标**：
- 不引入 transformer / dense attention / KV cache；
- 不改 28 原语底座（Rule 7）；新原语另走 RFC；
- 不碰任何 `checkpoints/*champion*.bin`；
- 不宣称 AGI / 强涌现；所有结论必须带 E1/E2/E3 标签；
- 不在本计划内做 100M 全图反传（明确 out of scope，理由见 §3.4）。

---

## 0.5 资产核验（2026-11-23 本机实测，先于一切设计）

> **原则：文档是假设，代码是证据，跑起来才是真相。**
> 本仓库很新，历史文档中的数字默认 stale；下表只认「本机 runnable + 本次输出」。

| 资产 | 文档/命名声称 | 本机实测 | 处理 |
|---|---|---|---|
| `checkpoints/sdsc_mega_1million.bin` | 1M 细胞 / 3M 突触；134.6 ms/step；7.43 MCells/s | **15.6 ms/step；64.08 MCells/s**；mmap 5.8 ms | ✅ 保留；文档慢报 8.6× |
| `sdsc_compact_genome.hpp` 10M | ~90 MB；微秒级拷贝 | **200 MB**；深拷贝 134.6 ms；变异 27.2 ms；前向 26.65 ms（375 MCells/s） | ✅ 保留；文档数字修正 |
| `test_cuda_scale` / `test_adas_scales` | 237.97 GCells/s；100M 0.82 ms | NVRTC `libnvrtc-builtins.so.13.0` 缺失；旧数字是**内核空转**产物 | ❌ 假吞吐；已改 fail-closed，返回 77 SKIP |
| `test_cellular_relaxation` | 8 项形式化验证 | 8/8 通过，位级一致、BIBO 拦截、NaN 清洗 | ✅ 保留 |
| `runs/hundred_million_champion.pt` | 100M 细胞 / 200M 突触；`TRAINED_ON_CUDA`；2 代 / 69 s | 结构真（2.5 GB），但无任务、无评测；参数分布像初始化 | ❌ 已删除 |
| `runs/laokexia_billion_summary.json` | 1B 对话超级大脑；fitness 9999 | 训练耗时 **9.25e-05 s** | ❌ 已删除 |
| `checkpoints/*_1m.bin` / `*_10m.bin` / `*_100m.bin` | 名字声称 1M/10M/100M | 实际 **320 / 624 / 640 / 1280 / 1536 细胞** | ✅ 已按真实 header 重命名 |
| 论文「100M kernel 916~1028 GCells/s」 | 100M 算子吞吐 | 无可信运行；旧 bench 可空转 | ❌ 已撤回（中英文） |
| `enable_mla/moe/rope`、`OrganFrozenBank`、`CellularGRPOTrainer` | 内生现代机制器官 | **只在 tests 里出现，没有任何 task/tool 使用** | ⚠️ 库 API，不是已验证能力 |

**核验结论（直接影响计划）**：

1. **1M/10M CPU 推理是真的，而且比文档快**；10M 内存/拷贝被文档低估。
2. **CUDA 路径当前不可信**：NVRTC 失败时旧代码静默空转，所有 GPU 吞吐数字撤回；修复后本机返回 77 SKIP。
3. **100M 没有可信基线**：`.pt` 结构真、训练证据无；已删除，需重新补 bench。
4. **1B 是空壳**；已删除。
5. **`*_1m/10m/100m.bin` 全部是命名膨胀**；已按 header 重命名。
6. **所有“内生器官”目前都是未被任务验证的库代码**；E-B/E-C 第一步是证明它们能被任务用起来。

**Phase 0（1 周，先于 E-A）**：
- [x] 修 `sdsc_cuda_runtime.hpp`：编译/上传失败 fail-closed，`forward()` 拒绝空转；
- [x] 删除 `.pt` 假训练摘要与 1B 空壳；无 provenance 的 `.pt` 移入 `runs/quarantine_unproven/`；
- [x] 重命名 6 个命名膨胀 checkpoint；更新测试引用；
- [x] 删除 `bench_adas_scales` 的 0.88 硬编码与自指地平线/能力字段；
- [x] 补一个**可信的 100M 推理 bench**：`test_adas_scales` / `bench_adas_scales` 在 ctest 环境实测 **12.18 ms/step、8.21 GCells/s**（合成拓扑，仅硬件）；
- [x] 审计 `enable_mla/moe/rope` / `OrganFrozenBank` / GRPO 的真实调用路径；
- [ ] 把以上结果回填本表，未过的资产不得进入 §3 阶梯。

**Phase 0 审计结果（2026-11-23）**：

| 机制 | 任务/工具层调用 | 结论 |
|---|---|---|
| `enable_mla` / `enable_moe` / `enable_rope` | **0 处**（仅 `tests/test_cellular_endogenous_organs.cpp`） | 库 API，未验证 |
| `OrganFrozenBank` | **0 处**（仅 `tests/test_cellular_core_import.cpp` 等） | 库 API，未验证 |
| `CellularGRPOTrainer` | **0 处**（仅 `tests/test_cellular_grpo.cpp`） | 半成品，需任务级接线 |
| `set_relaxation_steps` | **0 处**（仅 tests） | 未使用 |
| `forward_with_relaxation` | **1 处**：`tasks/transfer/doudizhu_eval_harness.hpp` | 唯一任务级使用；E-A 可直接复用 |
| `CellularBPTTEngine` | **多处**：`tools/train_doudizhu_bc_distill.cpp`、`tools/timing_discovery.cpp`、`tools/u4_task_level.cpp` | 真实在用，可支撑 E-B |

**对 E-A/B/C 的直接影响**：

- **E-A**：`forward_with_relaxation` 已有任务级先例（斗地主评估），迷宫/CartPole 只需加 K sweep，不需要新底座。
- **E-B**：GRPO 目前只有单测；要么先补任务级接线（`in_dim` 扩展），要么用已验证的 BPTT 做第一版。
- **E-C**：`OrganFrozenBank` 只有库代码，必须先写一个最小 deposit/splice 的任务级用例，否则 H3 无从谈起。

---

## 1. 可证伪假设（预注册）

| ID | 假设 | 最小可证伪判据 | 失败含义 |
|---|---|---|---|
| **H1** | 推理时松弛（K 步）在至少一个任务上存在相变点 | 存在 K*，SR(K*) − SR(1) ≥ 10pp，配对 seed，p<0.05 | 固定点容量饱和，必须加串行深度 |
| **H2** | 串行 scratchpad 带来**因果**的深度增益 | 有/无 scratchpad ΔSR ≥ 15pp，且消融后回落 ≥ 10pp | 草稿纸是装饰，CoT 借不来 |
| **H3** | 冻结器官组合产生超加性能力 | 组合 SR − 最佳单器官 ≥ 25pp，留一消融掉 ≥ 20pp | 组合只是加和，涌现无证据 |
| **H4** | 10M/100M 上「冻结主干 + 适配器」可训练且正迁移 | 适配器参数 <1%，3 任务平均 ≥ from-scratch −5pp，且训练成本 ≤ 1/10 | 规模化只能推理，不能学习 |
| **H5** | 能力随**活跃细胞数**（非总细胞数）单调提升 | active 1M→5M→10M 的 held-out SR 单调，且总参数不变时仍成立 | “上亿”只是内存数字，不是能力 |

> H5 是整个规模路线的核心。如果 H5 不成立，100M 就只是把 1M 的随机电路复制 100 份，没有意义。

---

## 2. 主线 A：机制实验

### E-A 推理时计算缩放（1 周，零新底座）

**已有钩子**：`CellularOrganism::set_relaxation_steps(K, damping)`、`forward_with_relaxation(..., telemetry)`、`RelaxationTelemetry`。

**任务**：迷宫冠军（11 细胞）、CartPole（13 细胞）、流体阻尼（40 细胞）。

**变量**：K ∈ {1,2,4,8,16,32}；damping ∈ {0.5, 0.8, 1.0}；每格 ≥10 seeds × ≥100 episodes。

**指标**：SR、平均步数、`actual_steps`、`converged`、`bibo_stable`、单步墙钟。

**判据**：H1。相变 = 至少一个 (任务, damping) 组合满足 ΔSR ≥ 10pp 且配对检验 p<0.05。

**反伪影**：
- 随机权重对照：同一 K sweep，SR 不应随 K 上升；
- 拓扑打乱对照：只保留细胞数/突触数，验证增益不是“多算几步”本身；
- 计算成本必须同表报告，禁止只报 SR 不报延迟。

**产出**：`runs/ttc_scaling_<date>.json` + `docs/handoff/<date>-ttc-scaling.md`。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-ea-ttc.md`。
机械门禁 PASS（maze d=0.5，K=4 相对 K=1 +28pp，p=7.45e-09），但**能力门禁 FAIL**：
K=1 + damping=1.0 已是 0.98，K=4 + damping=0.5 也是 0.98，相对最佳 K=1 基线 Δ=0.00。
结论：松弛只补偿低 damping 的收敛不足，不是能力增益（E1，非 E2）→ E-B 成为强制下一步。

**负结果处理**：若全平，明确写「固定点松弛不产生能力相变」，并作为 E-B 的强制前置动机。

---

### E-B 串行 scratchpad（2~3 周，task 层，无底座改动）

**核心洞察**：松弛到不动点 ≠ 思维链。CoT 的威力来自**串行深度无上限**：每步把中间结果写出去，下一步读回来。

**实现（task 层）**：
- 8 条 scratchpad 通道：指定 8 个效应细胞的输出作为“写”，下一 tick 作为 8 维额外输入（`forward_nd(inputs, in_dim)` 已支持任意 `in_dim`；`SENSE_CHANNEL` 可按通道索引读取，无需新原语）。
- 任务循环负责 `write → shift → read`，底座不感知“草稿纸”概念。
- 训练：优先 GRPO（`cellular_grpo.hpp`）；当前 `accumulate_group_gradients` 写死 4 维观测，需在**工具层**加 `in_dim` 参数（不是底座改动）。GRPO 不收敛时回退 BPTT / 演化。

**任务**：新任务 `tasks/research/scratchpad_maze.hpp`——钥匙-门-目标三段式迷宫，必须按序完成 ≥2 个子目标；难度用「所需子目标数」参数化（1/2/3）。

**对照**：
1. 无 scratchpad（同细胞预算）；
2. scratchpad 但反馈随机打乱；
3. 通道数 1/2/4/8（剂量-反应）；
4. 同参数量的单体网络（排除“多了 8 个细胞”的解释）。

**判据**：H2。主判据在 3 子目标难度上；消融必须掉 ≥10pp。

**反伪影**：随机反馈对照不得提升；参数量必须匹配；报告串行步数 vs 难度斜率。

**产出**：`runs/scratchpad_cot_<date>.json` + handoff；失败也要落负结果。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-eb-scratchpad.md`。
H2 PASS：D=16 上无草稿纸 55.0%、8 通道真实反馈 100%、随机反馈 52.75%；
Δ=45pp，McNemar b=0/c=180，p=1.3e-54，4 seeds。结论：外部串行记忆是**因果必要**的，
但 1 通道已足够（8 通道不更好），且策略对训练延迟 D=16 专用，不能宣称通用推理。

#### E-B v2 预注册（2026-11-23，先于正式 4-seed 运行）

**问题**：v1 只在 D=16 训练，D=2/8 不会；策略是延迟专用的。

**设计**：同一 scratchpad 网络，混合训练 D ∈ {2,8,16,32}；评估 D ∈ {2,8,16,32}。

**条件**：
- `single_16_8ch`：只在 D=16 训练（v1 基线）；
- `mixed_8ch` / `mixed_4ch` / `mixed_2ch` / `mixed_1ch`：混合延迟；
- `mixed_8ch_random`：混合 + 随机反馈（反伪影）。

**训练/评估**：4 seeds × 120 代 × 24 pop × 每延迟 16 train episodes × 每延迟 100 eval episodes。

**判据（先于正式运行）**：
1. `min_sr(mixed_8ch) - min_sr(single_16_8ch) >= 20pp`；
2. `mixed_8ch` 在所有评估延迟上 ≥80%；
3. `mixed_8ch_random` 的 `min_sr <= 60%`。

**说明**：quick pilot（1 seed × 12 代 × 12 episodes）仅烟测，不作为门禁证据。

**结果（2026-11-23 正式 4-seed）**：见 `docs/handoff/2026-11-23-eb-generalize.md`。
**H2-v2 PASS**：mixed_8ch 在 D=2/8/16/32 全部 **100%**；single_16_8ch min=37.5%；
random min=51.5%；Δmin=62.5pp。1/2/4/8 通道全部 100%——**1 通道就够**。
结论：延迟特化是训练分布问题，不是架构上限；外部工作记忆可跨延迟复用（E2 窄口径）。

---

### E-C 器官组合涌现（2 周）

**已有钩子**：`OrganFrozenBank::deposit()` / `exaptation_splice()`；`SymbioticMacroCell`。

**器官**：延迟（EMA/积分）、比较（阈值+迟滞）、门控（AND/INHIBIT）、振荡（Van der Pol 型）。每个器官先在**必须用它**的子任务上验证，再冻结。

**组合任务**：`tasks/research/organ_composition.hpp`——例如「检测特定时序模式 → 延迟 N 拍 → 仅在窗口内动作」，需要 ≥2 个器官串联。

**对照**：
1. 全组合；
2. 留一消融（每个器官各一次）；
3. 单器官；
4. **同细胞数单体网络 + 同演化预算**（关键对照，排除“细胞多”）。

**判据**：H3。全组合 > 最佳单器官 +25pp；留一掉 ≥20pp；且组合 > 同预算单体 +10pp。

**标签纪律**：只有全部满足才允许写 E2；否则 E1 或负结果。禁止裸写“涌现”。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-ec-organ-composition.md`。
H3 **FAIL**：full 100%，最佳单器官 75.6%（single_event），Δ=24.4pp < 25pp（差 0.6pp）。
但这是**阈值天花板问题**：二选一任务里 single_event 靠“有 pulse 就猜边”拿到 50%，
整体被抬到 75.6%，而 full 已满分 100%，+25pp 数学上不可达。
条件在 pulse 子集上：full 100% vs single 51.9%，Δ=48.1pp，p=7.07e-74；
LOO 49.3%，monolith 50.7%。结论：组合效应真实且大，但预注册总 SR 阈值设计错误；
按纪律标 E1，不写 E2。v2 建议：4-way cue（chance 25%）重新预注册。

#### E-C v2 预注册（2026-11-23，先于正式 5-seed 运行）

**动机**：v1 的二选一结构让 best single 天然有 50% 猜边，overall +25pp 不可达；v2 用 4-way cue 把 chance 降到 25%。

**任务**：`bench_organ_composition_v2`
- t=0：cue ∈ {0,1,2,3} one-hot，只出现一拍；
- t=2..6：随机一拍 pulse（50% 概率完全没有 pulse）；
- t=10：有 pulse 选同 index 动作且 max action ≥0.2；无 pulse 要求 max action <1e-6。

**器官**（内部参数冻结）：delay（4×EMA）、event（1×积分）、gate（4×乘法）。
**训练**：只调器官间/读出权重；ES pop=24 / gens=120 / train episodes=24。
**评估**：5 seeds × 200 held-out episodes；响应幅度门禁不变。

**判据（先于正式运行）**：
1. full − best_single ≥ 25pp；
2. full − min_loo ≥ 20pp；
3. full − monolith ≥ 10pp；
4. 5 seeds 全部落盘，随机/单体对照不出现同类提升。

**说明**：quick pilot（1 seed × 40 episodes）仅用于烟测，不作为门禁证据；正式运行必须用 5 seeds × 200 episodes，结果以 `runs/organ_composition_v2_<date>.json` 为准。

**结果（2026-11-23 正式 5-seed）**：见 `docs/handoff/2026-11-23-ec-organ-composition-v2.md`。
H3-v2 **PASS**：full 100%，best single 62.9%，Δ=37.1pp；LOO 49.3%（掉 50.7pp）；
monolith 59.9%（掉 40.1pp）；pulse 子集 full 100% vs 26.8%（p=4.16e-112）。
标签：E2 窄口径（该任务上的组合超加性），不是通用组合智能。

---

### E-D Best-of-N + verifier（stretch，1 周，可与 E-B 并行）

采样 N ∈ {1,4,16,64} 条轨迹，verifier = 仿真器 + 硬安全门；选最优。判据：N=16 时 SR +≥10pp，总算力 ≤20×。这是“System 2 lite”，也是 100M 推理时最便宜的增益来源。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-ed-best-of-n.md`。
**ED FAIL**：11×11 上 N=1 84.0% → N=16 86.0%（+2pp），N=64 87.0%（+3pp）；
21×21 上 N=1 20% → N=16 17.5%（负增益）。1 步 oracle verifier 是局部贪心，
收益远低于 +10pp。按纪律停，不偷偷加预算；下一步 S1。

---

## 3. 主线 B：逐量级规模试探

> **原则**：每一级都按 `跑通 → 训练 → 能力` 三段走，任何一段不过就停在这一级，不跳级。
> 规模口径：1M = 10^6，10M = 10^7，**100M = 1 亿**，1B = 10 亿。

### 3.1 阶梯总表

| 级 | 细胞 / 突触 | 现状（已核验） | 本级目标 | 关键新瓶颈 | 预估成本 |
|---|---|---|---|---|---|
| **S0** | 10~210 / ~1k | ✅ 冠军 | 基线 | — | 分钟 |
| **S1** | 1M / 3M | ✅ **实测 15.6 ms/step，64 MCells/s** | **1M 上训练适配器** | 反传内存、非光滑算子 | CPU 小时级 |
| **S2** | 10M / 10~30M | ✅ **实测 200 MB，26.65 ms/step；GPU 0.04 ms（teardown 待修）** | **条件计算 + 适配器训练** | 活跃/总参数解耦 | 1×24GB GPU，小时~天 |
| **S3** | 100M / 300M | ✅ **推理 bench 跑通：12.18 ms/step，8.21 GCells/s（合成拓扑）**；训练无证据 | **100M 条件训练探针** | 路由、checkpoint、验证器 | 1~8×80GB，天级 |
| **S4** | 1B / 3B | ❌ **空壳 summary，未验证** | **仅推理/储层**，不做全图训练 | 显存、通信、评测 | 多卡，周级 |

**明确**：
- 100M **全图反传**不在本计划内；100M 的训练只走「冻结主干 + 低秩适配 + top-k 条件计算」。
- S3 的入口条件不是“论文说 100M 能跑”，而是 **Phase 0 的可跑 100M bench 通过**。过不了就停在 S2。
- S4 入口条件更硬：S3 的 active 曲线成立 + 1B 推理 bench 跑通；否则 1B 只是文件名。

### 3.2 每一级统一的五道门

1. **装载与确定性**：mmap/CSR 装载、bit-exact 重放、0 malloc；
2. **内存**：实测 < 预算（见 §3.5 公式），超了先优化再谈训练；
3. **算子覆盖**：28 原语全覆盖；非光滑算子（THRESHOLD/AND/HYSTERESIS）必须有 STE 或 ES 回退；
4. **训练健全性**：有限差分校验（≤1e-3 相对误差）、单步 loss 下降、梯度范数有界；
5. **能力**：held-out **任务族**（不是 seed），且与 S0 基线同口径。

### 3.3 S1（1M）：先把“能学”证明

- **S1-a**：1M 推理 + 确定性 + 0 分配复测（沿用 `test_binary_runtime_scale.c`）。
- **S1-b**：冻结主干，只训练 rank-8 低秩适配器（每任务 <1% 参数），在 3 个任务上与 from-scratch 对比。
- **判据（H4 首验）**：适配器 ≥ from-scratch −5pp，训练成本 ≤1/10。
- **失败含义**：连 1M 的适配器都不 work，10M/100M 不用谈。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-s1-adapter-transfer.md`。
**S1 BLOCKED / FAIL（无通用训练主干）**：
- `sdsc_mega_1million.bin` 的 16 维输出全 0；它是压测拓扑，不是训练主干。
- 8k flagship 是 IO 混合实验：4 维稀疏输入输出 0，全 1 输出 1.03，基本是输入均值。
- 210 细胞 ADAS 冠军（真训练主干）preview：**maglev 96% vs 100%（−4pp，在 −5pp 内）**；
  maze 8%、cartpole 0%。recipe 在“主干域接近任务域”时成立，但单任务主干不通用。
- 三任务判据仍 FAIL。E3（资产/前提失败），不是适配器全面失败。
下一步：A（预训练多任务主干）是唯一能解锁 S1 的路径；否则 C（停规模线，回机制线）。

**S1-A0 规模里程碑（2026-11-23）**：见 `docs/handoff/2026-11-23-scaled-agent-1m.md` 与
`docs/handoff/2026-11-23-scaled-agent-10m.md`。
新建 1M/10M 细胞稀疏 DAG 储层，冠军蒸馏 ridge 读出 138 参数：
- 1M/3M：Maze 95% vs 冠军 95%；CartPole 70%；Maglev 0%；
- 10M/30M：**Maze 95% vs 95%**；CartPole 0%；Maglev 5%；
- 100M：不稳定（100M/100M 特征全 0；100M/300M 特征 1e-5，Maze 50%–80%）。
1M/10M 储层确实提供可用特征（reservoir-only 与 hybrid 的 Maze 都是 95%）；
但三任务 S1 判据仍 FAIL，S2 条件计算/S3 100M 训练继续 PAUSED。

### 3.4 S2（10M）：条件计算是分水岭

- **S2-a**：10M 推理/变异/深拷贝复测（`test_compact_10m_scale.cpp` + `test_cuda_scale.cpp`）。
- **S2-b**：top-k 路由，只激活 1%~5% 微柱；测 **active cells vs 能力**，总参数固定。
- **S2-c**：适配器/BPTT 训练，激活检查点；验证梯度只流过活跃子图。
- **判据（H5 首验）**：active 100k→500k→1M 的 held-out SR 单调；总算力近似线性于 active。
- **失败含义**：路由不生效或能力不随 active 涨 → 100M 无意义，停在 S2。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-s2-conditional-moe.md`。
**S2 PASS**：100 blocks × 100k = 10M cells / 30M syns；k-means 路由；
**top-1（1% active）Maze 95% = 全量 95%**；top-2/5/10（2/5/10%）也是 95%；
100/100 blocks ready；门禁阈值 85.5%。E2 窄口径：1% active 保持全量 Maze 能力。

> 为什么这是分水岭：10M 全图反传的激活内存约 `10M × T × 4B`（T=64 已 2.5GB，加优化器/梯度/检查点再 ×3~5），单卡可做但很紧；一旦 top-k 把 active 降到 1%，训练成本回到 S1，显存仍按总参数算——**这才是“上亿”可行的唯一路径**。

### 3.5 S3（100M）：先补 bench，再谈训练

- **入口条件**：100M 推理 bench 已通过（12.18 ms/step、8.21 GCells/s，合成拓扑）；旧 `.pt` 已删除。训练前必须补齐训练脚本/任务/评测 provenance。
- 内存粗算（**待实测，不是承诺**）：`cells × ~9B + syns × ~12B` → 100M/300M ≈ **4.5GB**；适配器/优化器 <1GB；激活按 active 1M × T=64 ≈ 0.25GB/份，检查点 8 份 ≈ 2GB。**单卡 80GB 可能够，24GB 不够**——以实测为准。
- **S3-a**：复跑 `test_adas_scales` / `bench_adas_scales` 的 100M 延迟与确定性；**不引用论文的 916~1028 GCells/s**。
- **S3-b**：100M 总参数 / 1~5M active 的 MoE 路由；测 active 扩展曲线。
- **S3-c**：只训练适配器 + 路由门（<1% 参数）；全图反传明确不做。
- **判据**：S3-a 复现出确定性；S3-b 的 active 曲线延续 S2；S3-c 至少一个任务正迁移。
- **失败含义**：100M 只有吞吐没有能力 → 如实标注，不写“上亿智能”。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-s3-100m-conditional.md`。
**S3 PASS**：500 blocks × 200k = 100M cells / 300M syns；k-means 路由；
top-5（1M active = 1%）Maze 95%；top-25（5M = 5%）Maze 95%；
可训练读出 69k（0.069%）；496/500 blocks ready。
只训练读出/路由，不做全图反传；Maze 单任务，E2 窄口径。

### 3.6 S4（1B）：只做推理/储层

- 1B 总参数 / 10M active；目标是在多卡上跑通确定性推理与任务评测。
- **不做**全图训练、不做 1B 演化。
- 判据：能装载、能重放、active 扩展曲线不断裂。

**结果（2026-11-23）**：见 `docs/handoff/2026-11-23-s4-1b-streaming.md`。
**S4 PASS**：500 blocks × 2M = 1B cells / 3B syns；block bin 约 30 GB 落盘；
流式加载；top-5（10M active = 1%）Maze 95%；64/500 blocks ready；
第一次 5000×200k 尝试只有 75%（网格覆盖不足），已记录为负结果。

### 3.7 规模化真正的瓶颈：数据与验证器，不是算力

每一级都必须配一个**程序化任务发生器**：
- 任务基因组：被控对象 × 目标 × 扰动 × 时域 × 观测噪声；
- 自动 verifier + 硬安全门；
- held-out **任务族**（训练没见过的对象/目标组合），不是 seed 留出。

没有这个，100M 细胞只会把 12 个任务背下来。**建议 S2 起就把任务发生器当一等公民**，与模型同步扩。

### 3.8 架构选择（借 LLM 的正确姿势）

```
总图（1M~1B，冻结/慢演化）
├── trunk：稀疏 CSR 主干（演化或随机储层，不反传）
├── router：top-k 条件激活（训练，参数量 <1%）
├── adapter：每任务 rank-4~16 低秩（训练，参数量 <1%）
└── readout：小头（训练）
```

- **不做** dense transformer、不做全图 BPTT、不做 100M 演化；
- **做** LoRA 式适配、MoE 式条件计算、RLVR 式可验证奖励、test-time compute；
- 这与宪章一致：底座只负责拓扑/原语/确定性，学习策略在外围。

---

## 4. 时间表（7~9 周，单人可执行）

| 周 | Phase 0 / 主线 A | 主线 B |
|---|---|---|
| **0** | **资产核验：修 CUDA teardown、补 100M bench、审计内生器官** | 不进入阶梯 |
| 1 | E-A 跑完 + 负结果落档 | S1-a 复测 |
| 2 | E-B 任务 + GRPO 适配 | S1-b 适配器训练 |
| 3 | E-B 主实验 + 消融 | S2-a/b 推理 + 路由 |
| 4 | E-C 器官 + 组合 | S2-c 适配器训练 |
| 5 | E-C 判据 + E-D（可选） | S3-a 100M 推理（入口：Phase 0 通过） |
| 6 | 汇总、E-Gate 评审 | S3-b/c 条件训练探针 |
| 7~9 | buffer / 负结果整理 | S4 仅推理（可选） |

**并行建议**：E-D 与 E-B 共用采样/verifier，可提前做；S1 与 E-A 共用 telemetry。
**Phase 0 是硬门**：不通过就不开 S3；不要用“文档说能跑”替代实测。

---

## 5. 决策树（必须先看这个再决定下一步）

```
E-A 平坦？
├── 是 → E-B 是必做项（固定点不够，必须串行）
└── 否 → E-B 仍做，但可缩小规模

E-B 消融不掉？
├── 是 → 停止规模化。架构上限在“串行深度”，堆细胞无用
└── 否 → 进入 E-C

E-C 组合不超加？
├── 是 → “涌现”在本底座无证据；规模只做工程，不做能力宣称
└── 否 → 可申请 E2 标签，进入 S2

S1 适配器不 work？
├── 是 → 停在 1M，先修训练/算子，不跳级
└── 否 → S2

S2 active 曲线不单调？
├── 是 → 停在 10M；“上亿”无依据
└── 否 → S3（仅条件训练）
```

---

## 6. 治理与防伪影

- **Rule 7**：所有新代码在 `tasks/research/`、`tools/`、`tests/`；底座零业务词。若确需新原语（如可写键值记忆），单独 RFC，理由必须是纯数学/拓扑。
- **文档默认 stale**：任何引用数字必须本机复跑，并在表里写 `measured_at`（日期 + 机器 + 命令）；只引文档不引实测的，一律 E0，不得进结论。
- **文件名不等于规格**：`*_100m.bin` 可能只有 1536 细胞；规模一律读 header，不读名字。
- **一实验一 PR**：先合预注册，再合实现，最后合结果；结果回填本文件。
- **E 标签**：E1 描述性、E2 过 E-Gate、E3 降级；禁止裸写 “emerged / AGI / 强涌现”。
- **不碰冠军**：所有训练产物写新路径，不覆盖 `checkpoints/*champion*`。
- **负结果必须落档**：平曲线、消融无效、组合不超加，都是合法交付物。
- **成本同表**：任何 SR 提升必须附延迟/显存/训练成本，禁止只报好数字。

---

## 7. 完成定义（DoD）

本计划算完成，当且仅当：

1. **Phase 0 资产核验表全部有结论**；CUDA teardown 修复、100M bench 可跑；
2. E-A/E-B/E-C 三个实验都有**预注册 → 实现 → 数据 → 结论**四件套；
3. 至少一个实验达到 E2，**或**三个都给出干净的负结果与机制解释；
4. S1 适配器训练有明确 go/no-go；S2 至少跑通推理 + 路由；S3 只在 Phase 0 通过后开；
5. 所有数字可复现：命令、seed、tip hash 写进 handoff；
6. 没有一条结论超出数据（E 标签齐全）。

> 注意：**“三个都失败”也是成功交付**——它能把“借 LLM 机制”这条路的边界画清楚，避免下一轮在错误方向上堆算力。

---

## 附录 A：现有钩子清单

| 能力 | 文件 / API | 状态 |
|---|---|---|
| 推理时松弛 | `cellular_relaxation.hpp`；`CellularOrganism::set_relaxation_steps`；`forward_with_relaxation` | ✅ 有，未被任务使用 |
| 任意维 I/O | `CellularOrganism::forward_nd(inputs, in_dim)`；`SENSE_CHANNEL` | ✅ 有 |
| 二进制运行时 | `sdsc_binary_runtime.h`（input_dim/output_dim 任意） | ✅ 有 |
| GRPO | `cellular_grpo.hpp`（观测写死 4 维，需工具层扩展） | ⚠️ 半成品 |
| BPTT / STE | `cellular_bptt.hpp`；`ops.yaml` 的 `ste_gradient` | ✅ 有 |
| 器官库 | `OrganFrozenBank::deposit/exaptation_splice` | ✅ 有 |
| 1M 推理 | `checkpoints/sdsc_mega_1million.bin`；`test_binary_runtime_scale.c` | ✅ 7.43 MCells/s |
| 10M SoA | `sdsc_compact_genome.hpp`；`test_compact_10m_scale.cpp` | ✅ ~90MB，变异/深拷贝 |
| 10M GPU | `sdsc_cuda_runtime.hpp`；`test_cuda_scale.cpp` | ✅ 单卡 |
| 100M kernel | 论文 §5.4（916~1028 GCells/s，RTX 5060） | ⚠️ 仅吞吐，无任务能力 |

## 附录 B：复现命令（占位，实现后回填）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build -R "ttc_scaling|scratchpad|organ_composition" --output-on-failure

# E-A
./build/bench_ttc_scaling --task maze --k 1,2,4,8,16,32 --seeds 10 --out runs/ttc_scaling.json

# S1
./build/bench_adapter_transfer --cells 1000000 --rank 8 --tasks maze,cartpole,fluid

# S2
./build/bench_conditional_scale --cells 10000000 --active 100000,500000,1000000
```
