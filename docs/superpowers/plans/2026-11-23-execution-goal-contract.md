# 执行 Goal 合同 — 2026-11-23

> 配套计划：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md`
> 目的：把「完整完成计划」拆成**可验收、可停止、可交接**的阶段 goal，防止 agent 在一个长上下文里跳过门禁、发明指标、或把负结果洗成正结果。

---

## 0. 为什么不能只写一个 goal

一个巨型 goal 会触发三种失败：

1. **上下文腐烂**：跑到 E-C 时，E-A 的预注册判据已经退化成残缺摘要；
2. **错误固化**：一旦 agent 把“这个提升看起来像涌现”写进上下文，后面所有推理都会围着它转；
3. **门禁被绕过**：为了“完成任务”，agent 会不自觉地调阈值、换 seed、删负结果。

所以用三层：

```
主 Goal（只调度，不执行实验）
  └── 阶段 Goal × N（一次会话只做一个，可独立验收）
        └── Handoff（落盘；会话结束前必须写）
```

**铁律：上一阶段没有 `GATE: PASS` 或 `GATE: FAIL (已落档)`，不得开下一阶段。**

---

## 1. 主 Goal（每次新会话开头粘贴）

```text
你是本计划的执行总控。唯一目标：让
docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md
按预注册门禁完整落地。

你的职责不是“做实验”，而是：
1) 每次只激活一个阶段 Goal（G1/G2/G3/S1/S2/S3），不得并行跳阶段；
2) 验收上一阶段的产物：commit、命令、JSON、gate 结论，四者缺一不可；
3) 上一阶段未过门禁，不得开下一阶段；失败也必须写负结果并落盘；
4) 每阶段收口写 docs/handoff/<date>-<phase>.md，包含：目标、命令、数据、gate、下一步；
5) 不发明指标、不调预注册阈值、不删负结果、不把 SKIP 当 PASS；
6) 不修改 checkpoints/*champion*，不新增业务词到底座，不改 28 原语。

当前进度只看仓库，不看聊天记忆。第一步：读计划文件的 §0.5 资产核验与 §2/§3，
报告「当前应该激活哪个阶段」，然后停下等我确认。
```

---

## 2. 阶段 Goal 卡

每张卡按七槽写：目标 / 约束 / 输入 / 输出 / 验收 / 证据 / 禁区。

### G1 — E-A 推理时计算缩放

| 槽 | 内容 |
|---|---|
| **目标** | 验证 `relaxation K` 是否在迷宫/CartPole/流体上产生能力相变（H1） |
| **约束** | 只做 E-A；不改底座；不碰冠军 bin；测试集与训练集分离 |
| **输入** | 计划 §2 E-A；`cellular_relaxation.hpp`；`set_relaxation_steps`；三个冠军 bin |
| **输出** | `tools/bench_ttc_scaling.cpp`；`runs/ttc_scaling_<date>.json`；`docs/handoff/<date>-ea-ttc.md` |
| **验收** | `ctest -R ttc_scaling` 通过；JSON 含 K/damping/seeds/SR/成本；随机权重与打乱拓扑两个对照齐全 |
| **证据** | 每个 (任务,K,damping) 的 SR、配对检验 p、单步延迟；不得只报 SR |
| **禁区** | 不许看到结果后改 K 集合或阈值；不许只挑显著的任务；不许把 SKIP 写 PASS |

**停止规则**：若所有组合 ΔSR < 10pp，写 `GATE: FAIL`，明确“固定点松弛不产生相变”，然后停，不自动进 E-B。

### G2 — E-B 串行 scratchpad

| 槽 | 内容 |
|---|---|
| **目标** | 验证串行草稿纸是否带来**因果**深度增益（H2） |
| **约束** | task 层实现；优先 BPTT；GRPO 只在 `in_dim` 扩展完成后用；不新增原语 |
| **输入** | 计划 §2 E-B；`forward_nd`；`SENSE_CHANNEL`；`CellularBPTTEngine`；钥匙-门任务设计 |
| **输出** | `tasks/research/scratchpad_maze.hpp`；`tools/bench_scratchpad_cot.cpp`；`runs/scratchpad_cot_<date>.json`；handoff |
| **验收** | 有/无 scratchpad 对照；随机反馈对照；参数量匹配；消融后 SR 下降 ≥10pp |
| **证据** | 主难度 SR、消融 Δ、串行步数 vs 难度斜率、训练成本 |
| **禁区** | 不许只报最好难度；不许用测试 seed 调参；不许把随机反馈也提升解释成“正则化” |

**停止规则**：若消融不降，写 `GATE: FAIL`，结论“串行深度不是瓶颈”，停。

### G3 — E-C 器官组合

| 槽 | 内容 |
|---|---|
| **目标** | 验证冻结器官组合是否超加性（H3） |
| **约束** | 先用最小 `OrganFrozenBank` 任务级用例；同细胞数单体对照必须有 |
| **输入** | 计划 §2 E-C；`OrganFrozenBank`；4 个子技能器官 |
| **输出** | `tasks/research/organ_composition.hpp`；`tools/bench_organ_composition.cpp`；JSON；handoff |
| **验收** | 全组合 > 最佳单器官 +25pp；留一掉 ≥20pp；组合 > 同预算单体 +10pp |
| **证据** | 四个对照（全组合/留一/单器官/单体）全表；每项 ≥5 seeds |
| **禁区** | 不许只报全组合；不许把“多几个细胞”当组合收益；不满足全部条件不得写 E2 |

### S1 — 1M 适配器

| 槽 | 内容 |
|---|---|
| **目标** | 验证冻结主干 + rank-8 适配器可训练、可迁移（H4） |
| **约束** | 不改主干权重；适配器 <1% 参数；与 from-scratch 同预算对比 |
| **输入** | 1M bin；`CellularBPTTEngine`；三个任务 |
| **输出** | `tools/bench_adapter_transfer.cpp`；JSON；handoff |
| **验收** | 适配器 ≥ from-scratch −5pp；训练成本 ≤1/10；梯度有限差分校验 ≤1e-3 |
| **证据** | 三任务 SR、训练时长、参数量、梯度校验 |
| **禁区** | 不许只报最好任务；不许把“没训练”说成“零样本” |

### S2 — 10M 条件计算

| 槽 | 内容 |
|---|---|
| **目标** | 验证 active cells 扩展曲线（H5） |
| **约束** | 总参数固定；只激活 1%~5%；CPU/GPU 均可，必须报显存 |
| **输入** | `CompactSoAGenome`；10M 基因组；top-k 路由 |
| **输出** | `tools/bench_conditional_scale.cpp`；JSON；handoff |
| **验收** | active 100k→500k→1M 的 held-out SR 单调；总算力近似线性 |
| **证据** | active/总参数/延迟/显存/SR 五列全表 |
| **禁区** | 不许用总参数冒充计算量；不许只报峰值 |

### S3 — 100M 条件训练探针

| 槽 | 内容 |
|---|---|
| **目标** | 只验证 100M 总参数 / 1~5M active 的适配器训练是否可行 |
| **约束** | 不做全图反传；不训练主干；不做 100M 演化 |
| **输入** | Phase 0 的 100M bench（12.18 ms / 8.21 GCells/s）；S2 的 top-k 路由 |
| **输出** | JSON；handoff；明确 go/no-go |
| **验收** | active 曲线延续 S2；至少一个任务正迁移；显存/延迟在预算内 |
| **证据** | 同上 + 训练稳定性（loss 不炸、BIBO 不越界） |
| **禁区** | 不许引用 916~1028 GCells/s；不许把合成拓扑写成训练产物 |

---

## 3. 门禁矩阵（主 Goal 只认这个）

| 阶段 | PASS 条件 | FAIL 后动作 |
|---|---|---|
| G0 | 105/105 ctest；资产核验表有结论；CUDA fail-closed | 修到绿再进 G1 |
| G1 | 至少一个组合 ΔSR ≥10pp，p<0.05，两对照不提升 | 落负结果，停 |
| G2 | 有/无 ΔSR ≥15pp，消融回落 ≥10pp | 落负结果，停 |
| G3 | 全组合 > 单器官 +25pp，留一 ≥20pp，> 单体 +10pp | 落负结果，停 |
| S1 | 适配器 ≥ from-scratch −5pp，成本 ≤1/10 | 停 1M，不跳 S2 |
| S2 | active 曲线单调 | 停 10M，不跳 S3 |
| S3 | active 曲线延续 + 正迁移 | 停 100M，不写“上亿智能” |

> **SKIP 不算 PASS。** GPU 用例返回 77 时，必须写“未验证”，不能写“通过”。

---

## 4. 反作弊 / 防伪影（写进每个阶段 goal 的禁区）

1. **预注册先于数据**：先提交计划/阈值，再看结果；看后改阈值 → 直接标 E3。
2. **所有 SR 必须附成本**：延迟、显存、训练时长，缺一不可。
3. **负结果必须落盘**：平曲线、消融无效、组合不超加，都是合法交付。
4. **禁止挑任务/挑 seed**：预注册哪些任务就报哪些；不允许“这个任务不算”。
5. **禁止把自指指标当能力**：预测地平线、觉醒度、保真度必须有 ground-truth 对账。
6. **禁止命名膨胀**：规模读 header，不读文件名；CI 会拦。
7. **禁止静默失败**：任何 compile/upload/assert 失败必须 fail-closed，不许空转。
8. **禁止碰冠军**：所有产物写新路径；`checkpoints/*champion*` 只读。
9. **禁止底座业务化**：28 原语不改；新原语走 RFC；底座零业务词。
10. **E 标签**：E1 描述 / E2 过 E-Gate / E3 降级；禁止裸写 “emerged / AGI”。

---

## 5. 每阶段证据格式（handoff 必须包含）

```text
PHASE: E-A
COMMIT: <hash>
COMMAND: <exact command>
ARTIFACT: runs/ttc_scaling_20261123.json
GATE: PASS | FAIL | SKIP
KEY NUMBERS: <SR / p / cost / seeds>
CONTROLS: <random-weight / shuffled-topology / ablation>
CONCLUSION: <one sentence, E-tag>
NEXT: <E-B | STOP | ask>
```

没有 `COMMIT` 和 `COMMAND` 的结论，主 Goal 不接受。

---

## 6. 上下文边界（怎么开会话 / 怎么派子 agent）

- **一个阶段一个会话**。G1 结束就收口，不在同一会话开 G2。
- **主 agent 自己持有**：门禁判据、最终结论、是否进下一阶段。
- **可派子 agent**：只读审计、批量跑 seed、批量抽数；回传只给事实+命令+输出片段，不给全局结论。
- **不可拆**：E-B 的消融因果判断、E-C 的“是否超加”判定、S2 的 active 曲线解释——这些需要看见同一批事实。
- 每阶段结束必须写 handoff；聊天里的结论不算数。

---

## 7. 什么时候停 / 什么时候问人

| 情况 | 动作 |
|---|---|
| 门禁 FAIL | 写负结果，停，不自动进下一阶段 |
| 连续 2 天无进展 | 写 blocker（命令+报错+已试方案），问人 |
| 结果“好得不像真的” | 先跑 ≥4 seeds + OOD + 对照；未复现不得写 E2 |
| 想改阈值/换 seed/删任务 | 停；这是作弊信号，先问人 |
| 环境缺依赖（NVRTC/GPU） | 标 SKIP/BLOCKED，写复现命令，不写 PASS |

---

## 8. 启动模板（直接复制）

### 启动 G1

```text
/goal G1-EA
读 docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md 的 §2 E-A、§3.2、§6。
只做 E-A，不做 E-B/C。
先复述：预注册判据、K/damping 网格、seed 数、两个反伪影对照。
然后实现 tools/bench_ttc_scaling.cpp，跑完写 runs/ttc_scaling_<date>.json 与 handoff。
结束时只回：COMMIT / COMMAND / ARTIFACT / GATE / KEY NUMBERS / NEXT。
```

### 启动 G2

```text
/goal G2-EB
读计划 §2 E-B 与 G1 的 handoff。
只做 E-B：scratchpad 任务 + 有/无/随机反馈/参数量匹配四组对照。
优先 BPTT；GRPO 只在 in_dim 扩展完成后用。
结束时回：COMMIT / COMMAND / ARTIFACT / GATE / ABLATION / NEXT。
```

### 启动 G3 / S1 / S2 / S3

按 §2 对应卡片填七槽；每次只激活一个。

---

## 9. 完成定义（总）

计划算完成，当且仅当：

1. G0 资产核验表全有结论；
2. G1/G2/G3 各有 `预注册 → 实现 → 数据 → 结论` 四件套；
3. 至少一个达到 E2，或三个都给出干净负结果与机制解释；
4. S1 有明确 go/no-go；S2 跑通 active 曲线；S3 有 go/no-go；
5. 所有 handoff 含 commit + command + JSON；
6. 没有一条结论超出数据（E 标签齐全）。

> **三个都失败也是成功交付**——它把“借 LLM 机制”的边界画清楚，避免下一轮在错误方向堆算力。
