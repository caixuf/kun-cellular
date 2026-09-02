# KunCellular: 软件定义硅基细胞计算机与形态发生演化底座

[![CI](https://github.com/caixuf/kun-cellular/actions/workflows/ci.yml/badge.svg)](https://github.com/caixuf/kun-cellular/actions)
![C++20](https://img.shields.io/badge/C++-20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 基于冯·诺依曼自复制自动机理论、三维力敏形态发生（Morphogenesis）动力学、算存一体局部状态与 Kahn 拓扑排序零 GC 编译器的**软件定义硅基细胞计算机（Software-Defined Silicon Cellular Computer, SDSCC）**。

---

## 核心理论体系与学术成果

本项目的完整数学推导、离散力场方程、形式化安全证明与实证评估已收录于核心论文：
* 中文完整论文：[docs/morphogenetic_cellular_evolution_paper.zh.md](docs/morphogenetic_cellular_evolution_paper.zh.md)
* 英文学术论文：[docs/morphogenetic_cellular_evolution_paper.md](docs/morphogenetic_cellular_evolution_paper.md)
* 演化路线图规范：[docs/2026-09-01-quantitative-cellular-evolution-roadmap.md](docs/2026-09-01-quantitative-cellular-evolution-roadmap.md)

### 1. 计算范式对照：传统深度学习 vs 软件定义硅基细胞计算机

| 对比维度 | 传统人工神经网络 (ANN / Transformer) | 软件定义硅基细胞计算机 (SDSCC) |
| :--- | :--- | :--- |
| **计算基元** | 均质张量乘加（$\mathbf{W}\mathbf{x} + \mathbf{b}$）+ 统一静态激活函数 | 24 种具备显式物理动力学语义的异构计算细胞（含微积分、迟滞门控、极限环振荡器等） |
| **状态与存储** | 外部隐藏状态张量，运算器与存储器物理分离 | 原生算存一体，每个细胞私有持久内部状态累积电位 $s_i$ 与 FIFO 缓冲 |
| **网络拓扑** | 静态规则矩阵、分层前馈或全连接自注意力 | 三维空间自组织动态 DAG/循环图，力场自发聚类出功能皮层微柱 |
| **优化机制** | 全局梯度反向传播 (BP)，依赖链式法则全局同步 | 受控形态发生（有丝分裂/凋亡）+ 微观 Oja 局部塑性 + 鲍德温代际固化 |
| **时延与确定性** | 动态解释器开销、GC 停顿、毫秒级时延抖动 | Kahn 拓扑线性化编译、扁平连续数组布局、**24.1 ns 确定性硬实时、零 GC** |
| **因果可解释性** | 连续稠密黑箱分布表示，单神经元无法独立逻辑证伪 | 显式因果通路、WL 图同构指纹、精确图编辑距离与敲除性能承重断言 |

### 2. 四阶段人工生命大满贯实测指标

| 演化阶段 | 核心机制 | 验证重点 | 实测验收指标 |
| :--- | :--- | :--- | :--- |
| **阶段一：动态自组织稳态** | 空间哈希力场自组织与零 GC 拓扑编译 | 胚胎自下而上折叠成脑，拓扑线性化连续内存执行 | **100 万细胞推理耗时 20.61 ms**，空间哈希物理 $O(N)$ 线性扩展，全拓扑零内存碎片 |
| **阶段二：自主复制与遗传** | 能量驱动去中心化繁衍与代际遗传 | 储能盈余触发分裂，隔室空间容量硬阻滞与饥荒停育 | **40 ticks 繁衍至 40 个体 (最高 Gen 3)**，谱系哈希分支覆盖率 100% |
| **阶段三：红皇后协同对抗** | 宿主-数字病原体对抗演化与 HGT 扩增 | 抗原漂移突变、获得性免疫记忆与群体免疫屏障 | **群体免疫覆盖率 26.7%**，病毒介导基因水平转移 (HGT) 成功率 100% |
| **阶段四：开放式具身沙盒** | 感知-动作-位移-摄食闭环热力学因果链 | 无监督长周期生存稳态，自发涌现全新信号拓扑 | **300 ticks 无监督盲行存活率 100%**，自发涌现 4 条新功能信号通路 |

---

## 性能与规模光谱实测矩阵

* **微观确定性**：单步推理耗时 **24.1 ns**，P99 延迟 **179.0 ns**，最坏极限时延 **35.9 us**（远低于车规 10ms 限额），单步前向推理内存分配严格为 **0 字节**。
* **宏观扩展性**：
  * 百万细胞脑 (1M)：单步推理耗时 12.90 ms，显存占用仅 568.4 MB。
  * 亿级张量脑 (100M)：CUDA 流式计算吞吐达 1,114.4 MCells/s，显存占用受控于 4.38 GB。
* **鲍德温效应 (Baldwin Effect)**：后天突触塑性权重代际无损固化率 100%，有效抵抗外部环境分布漂移。

---

## 目录结构

```
kun-cellular/
|-- include/kun/cellular/      # C++20 Header-Only 核心算法库
|   |-- cellular_genome.hpp    # 24 种算存一体细胞、空间力场与 Kahn 零 GC 编译器
|   |-- island_evolution_grid.hpp # 岛屿网格迁移与多目标演化
|   |-- ecosystem_biosphere.hpp# 宏观生态圈、食物网与香农多样性
|   |-- digital_pathogen_ecosystem.hpp # 数字病原体、免疫抗体与 HGT 机制
|   |-- open_ended_embodiment.hpp # 开放式具身物理沙盒
|   `-- ...
|-- tests/                     # 21 组严苛回归与极限压测套件 (CTest 100% PASS)
|-- tools/                     # GPU 张量训练器、CUDA 演化内核与 Web 交互中枢
|-- docs/                      # 核心学术论文 (zh/en) 与四阶段 ALife 规范
`-- skills/                    # 专业 Agent Skills 知识库 (形态发生/病原体对抗/具身演化)
```

---

## 快速构建与验证

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
ctest --output-on-failure
```

---

## 许可证
本项目采用 MIT 许可证。
