# 拓扑消融证据库存（2026-09-14）

> agy 扫描任务退出码 0 但未落盘；本文件按仓库实扫补写。不改 `include/kun/cellular/`。

## 五句话

1. **只有 ADAS** 有同度数重连 + E/I 翻转门禁，覆盖 L3 与 stadium 两把锁档。
2. 迷宫/斗地主/量化没有同协议对照；论文「敲除承重」目前没有对应测试文件。
3. U4 / flow 消融测的是生长预算、种子模式、骨架锁，不是拓扑因果。
4. `create_minimal_random_graph` 只作时延记忆对照，不是 ADAS 重连。
5. 四个形态发生算子的 On/Off 消融论文已声明未做。

## 覆盖 ADAS 锁档

| 路径 | 测什么 | ADAS 锁档 |
|---|---|---|
| `tools/ablate_topology_prior.py` | 同度数重连（Maslov–Sneppen）+ 全突触符号翻转 | **是**：`adas_cortex_champion.bin` / `adas_cortex_champion_stadium.bin` |
| `tests/test_adas_topology_ablation.py` | CI：`--rewires 2`，stdout 须含 `overall=PASS` | **是**（L3） |
| `docs/handoff/2026-09-14-topology-ablation.md` | 战役 #4 结果：重连 34.1× / 23.9×，E/I 10.8× / 22.0× | 报告 |
| `runs/adas_topology_ablation_20260914.json` | 数值产物 | 证据 |

## 名称像消融、协议不是拓扑因果

| 路径 | 实际测什么 | ADAS |
|---|---|---|
| `tools/u4_ablation.cpp` + `docs/superpowers/plans/2026-09-08-u4-ablation-manifest.md` | 斗地主知识模块：固定拓扑 vs 发育分裂 vs 同预算随机分裂 | 否 |
| `tests/test_flow_constraint_ablation.cpp` | 迷宫：骨架锁 / 白名单 / 种子模式 / 新颖性权重 | 否 |
| `tests/test_flow_seed_ablation.cpp` | 迷宫：种子初始化模式定量实验 | 否 |
| `tools/train_maze_tripartite.cpp --ablate-evo` | 关掉演化臂 | 否 |
| `tests/test_flow_temporal_memory_evolution.cpp` | 时延参考图 vs `create_minimal_random_graph` | 否 |
| `tests/test_cellular_core_migration.cpp` / `test_cellular_graph_edit.cpp` | 编译器/图编辑后边重连正确性 | 否（底座单元） |
| `tools/train_flagship_wired.cpp` | 训练期 IO 轴突重连计数，不是事后敲除 | 否 |

## 缺口

- 迷宫 / 斗地主：无同度数重连、无 E/I 翻转。
- 细胞敲除（knockout deficit）：论文有叙述，仓库无 `knockout` 测试命中。
- 四算子 On/Off：论文 §6 已撤回表 5b，明确未跑。
