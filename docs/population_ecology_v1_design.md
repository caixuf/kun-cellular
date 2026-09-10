# 生命体种群生态 — 系统层设计文档 (Population Ecology, System Layer)

> 状态: 设计评审稿 v2 (待用户裁决)
> 定位裁决 (用户): 种群生态能力归属**核心层之上的系统层**, 不是训练器脚本
> 病根诊断: 现行训练为"单冠军制"; 且底座已有的生态框架 (IslandEvolutionGrid 等)
> 各绑死特定玩具个体类型, 任务层从未接线 — 系统层"存在但从未被使用"。

---

## 〇、审计发现: 底座生态框架现状 (为什么不能直接用)

| 既有框架 | 能力 | 致命耦合 | 使用现状 |
|---|---|---|---|
| `island_evolution_grid.hpp` | 8 岛 deme、红皇后对抗扰动 (AdversarialStressProfile 四级)、迁移计数 | `step_island()` 硬编码 `org.forward(inputs[4])` + 力场物理 + 正负动作收益公式 — 绑死 4 输入原始生命体 | 仅 2 个 flow 测试 |
| `morphogenetic_population.hpp` | 物种化指标、SpeciesNiche、PureCellularIndividual | 个体类型绑死 cellular_genome | 仅自身测试 |
| `multispecies_ecology.hpp` | 物种公会 (SpeciesGuild)、EcologicalOrganism、世界遥测 | — | **零引用 (完全孤儿)** |
| `digital_pathogen_ecosystem.hpp` | 病原体/免疫生态 | 绑定自有生命周期 | 仅自身测试 |
| `ecosystem_biosphere.hpp` | 生物圈、量子辐射场 | 耦合辐射场 | 仅 2 个测试 |
| `tripartite_learning.hpp` | 三权分立学习栈 (训练器实际在用的唯一框架) | 无杂交/无岛屿/无生态位 | 4 个训练器 |

**结论**: 系统层的"魂"已在底座里 (岛屿/对抗/生态位/物种化概念齐全), 但 (a) 类型耦合使其不可复用,
(b) 缺失关键机制 (通用杂交算子、组合级评估、池持久化), (c) 任务层从未接线。
既有框架保留原测试不动 (历史资产), 新系统层吸收其概念重新泛型化。

---

## 一、三层架构定案

```
┌─────────────────────────────────────────────────────────────┐
│ L2 任务层: tasks/ + tools/trainers                           │
│   领域适应度公式(权重常数)、评估委托、市场/牌局数据、业务名词    │
├─────────────────────────────────────────────────────────────┤
│ L1 系统层 (新建): include/kun/population/   ← 本设计主体       │
│   领域无关: 岛屿/迁移/杂交/捕食者课程/生态位/池持久化/LOO/相关性 │
│   依赖方向: L1 → L0 单向。禁止依赖 tasks/。禁止业务名词。       │
├─────────────────────────────────────────────────────────────┤
│ L0 核心底座 (神圣不可修改): include/kun/cellular/              │
│   动力学原语、图编译、前向推演、SDSC-BIN、既有生态框架原样保留    │
└─────────────────────────────────────────────────────────────┘
```

**L1 宪章 (系统层纪律, 与 L0 铁律衔接)**:
1. L0 零修改 — L1 只消费 L0 公开 API (`CellularOrganism`, `CorticalMacroArray::columns()` 可变引用等)
2. L1 不含任何业务名词 (无 PnL/胜率/市场/牌局; 只有 returns/fitness/individual/pool)
3. L1 全部确定性: 无隐藏全局状态, 一切随机源显式传入 (std::mt19937), 同种子位级可复现
4. L1 头文件仅式 (INTERFACE 库 `kun_population`), 与 `kun_cellular` 同风格
5. L1 每个机制必须有独立单测; L0 的 78 项测试保持全绿 (回归证明)
6. L1 前向路径零介入 — 训练离线使用, 运行时 (19ns 实时链路) 不感知 L1 存在

---

## 二、L1 API 草案 (include/kun/population/)

```
population/
├── individual_traits.hpp      # 个体概念 (C++20 concepts): copyable + mutate(rng) + serialize/deserialize
│                              # L1 为 CellularOrganism 与 CorticalMacroArray 提供 crossover 特化 (仅引用 L0)
├── deme.hpp                   # Deme<Individual>: 精英保留、锦标赛、变异代际
├── ecology_grid.hpp           # EcologyGrid<Individual>: N deme + 迁移率 + 列级/突触级杂交 (泛型化吸收 IslandEvolutionGrid 概念)
├── adversarial_course.hpp     # 捕食者课程 (泛型化 AdversarialStressProfile): 任务层注册"压力算子", L1 负责调度
├── pool_ecology.hpp           # 组合级评估原语: LOO 边际贡献、成员相关矩阵、体制分位划分、寄生惩罚
│                              #   输入: 任务层注入的逐日 returns 矩阵 + 冻结的权重常数
│                              #   输出: PoolMetrics (纯数学, 无业务语义)
├── pool_manifest.hpp          # 池持久化: 成员 bin 路径 + 生态位标签 + 指标 (JSON, 索引式; 不改 SDSC-BIN)
└── population_eval.hpp        # 评估委托接口: PopulationEval<Individual> → ReturnsMatrix (任务层实现)
```

**关键泛型边界**:
- `crossover` 走 traits 特化: L1 定义 `crossover_with(a, b, rng)` 概念; `CorticalMacroArray` 特化 = 列级整列重组
  (43 列逐列取父 A/B, 长程轴突重连, 杂交后 compile 冒烟); `CellularOrganism` 特化 = 突触级均匀重组
  (v1 只需前者, 后者留给斗地主 M2)
- `pool_ecology` 只做数学 (LOO/相关/分位), 权重常数 (夏普 2.0/卡玛 1.0/回撤 2.0/寄生阈 0.90) 由 L2 注入并冻结
- `adversarial_course` 只做调度 (何时、以何种强度调用), 压力算子本体由 L2 注册

---

## 三、L2 量化首战适配 (首个客户端)

| 件 | 归属 | 内容 |
|---|---|---|
| `tools/train_quant_population_ecology.cpp` (新) | L2 | EcologyGrid< CorticalMacroArray > 实例化: 8 deme × 16 个体 × 200 代, 迁移 6%, 列级杂交; 评估委托 = 43 品种逐日信号回放; 池 K=5 等权 |
| 适应度公式 | L2 冻结 | 池级: Sharpe×2.0 + Calmar×1.0 − MDD×2.0; 个体级: LOO 贡献 − 寄生惩罚 (corr>0.90) − 体制空缺惩罚 |
| 对照组 | 既有 | 单冠军训练器原样保留, 同协议对账 |
| 池产物 | L1 格式 | `checkpoints/quant_ecology_pool.json` + K 个标准 SDSC-BIN 成员 |

---

## 四、预注册协议 (不变, 沿用 v1 草案第四节)

- val (2013~2015) 选择, test (2016~) 一次性盲报; 常数冻结; val 附月度 block-bootstrap 95% CI;
- 诚实条款: 夏普<0 成员不入池 / val 池回撤>15% 判失败回退 / 8 deme 全量如实报告 / 池必须含 ≥2 个体制标签不同成员。

---

## 五、测试矩阵

| 级 | 测试 | 不变量 |
|---|---|---|
| L1 | test_population_individual_traits | crossover 后 compile 通过、无 NaN、个体数守恒 |
| L1 | test_population_ecology_grid | deme 规模守恒、精英不被迁移覆盖、同种子位级复现 |
| L1 | test_population_pool_ecology | LOO 方向性、寄生惩罚方向性 (相同个体→惩罚生效)、体制分位正确 |
| L1 | test_population_pool_manifest | 池存取往返逐字节对账 |
| L2 | test_quant_population_ecology | 量化端到端冒烟: 2 deme × 3 代出池、加载融合一致 |
| 回归 | 既有 78 项 | L0 零修改证明 |

---

## 六、治理与前瞻

- **governance 记录**: `multispecies_ecology.hpp` 孤儿状态记入 STATUS_BOARD 待办 (接线或标记 deprecated, 本设计不处理它)
- **迁移路径**: 量化 (首战) → 斗地主 M2 对手池 (捕食/被食协同进化, 复用 ecology_grid + adversarial_course) → ADAS 域随机化课程 → 通用
- **v2+**: L1 与 GermlineLibraryStore (tasks/transfer, SQLite) 对接做跨战役种质借用 — 届时评估是否上浮 germline 至 L1
- **回退**: L1 为纯增量 (新目录/新库/新测试), 任何失败可整体删除回退, 零污染
