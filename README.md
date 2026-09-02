# KunCellular: 形态发生计算图与人工生命演化底座

[![CI](https://github.com/caixuf/kun-cellular/actions/workflows/ci.yml/badge.svg)](https://github.com/caixuf/kun-cellular/actions)
![C++20](https://img.shields.io/badge/C++-20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 基于胚胎形态发生 (Morphogenesis)、空间哈希力场自组织、鲍德温效应 (Baldwin Effect) 基因固化与开放式具身演化的通用人工生命 (Artificial Life) 与形态发生计算图基础底座。

---

## 核心理论与学术成果

本项目的理论推导、数学证明与实证评估已收录于核心论文：
* 中文完整论文：[docs/morphogenetic_cellular_evolution_paper.zh.md](docs/morphogenetic_cellular_evolution_paper.zh.md)
* 英文学术论文：[docs/morphogenetic_cellular_evolution_paper.md](docs/morphogenetic_cellular_evolution_paper.md)
* 演化路线图规范：[docs/2026-09-01-quantitative-cellular-evolution-roadmap.md](docs/2026-09-01-quantitative-cellular-evolution-roadmap.md)

### 四阶段人工生命大满贯实测指标

| 演化阶段 | 核心机制 | 验证重点 | 实测验收指标 |
| :--- | :--- | :--- | :--- |
| **阶段一：动态自组织稳态** | 空间哈希力场自组织与零 GC 拓扑编译 | 胚胎自下而上折叠成脑，拓扑线性化连续内存执行 | **100 万细胞推理耗时 20.61 ms**，空间哈希物理 $O(N)$ 线性扩展，全拓扑零内存碎片 |
| **阶段二：自主复制与遗传** | 能量驱动去中心化繁衍与代际遗传 | 储能盈余触发分裂，隔室空间容量硬阻滞与饥荒停育 | **40 ticks 繁衍至 40 个体 (最高 Gen 3)**，谱系哈希分支覆盖率 100% |
| **阶段三：红皇后协同对抗** | 宿主-数字病原体对抗演化与 HGT 扩增 | 抗原漂移突变、获得性免疫记忆与群体免疫屏障 | **群体免疫覆盖率 26.7%**，病毒介导基因水平转移 (HGT) 成功率 100% |
| **阶段四：开放式具身沙盒** | 感知-动作-位移-摄食闭环热力学因果链 | 无监督长周期生存稳态，自发涌现全新信号拓扑 | **300 ticks 无监督盲行存活率 100%**，自发涌现 4 条新功能信号通路 |

---

## 性能与规模光谱实测矩阵

* **微观确定性**：单次前向传导中位数延迟 152.0 ns，P99 延迟 179.0 ns，最坏延迟 35.9 us（远低于硬实时限额）。
* **宏观扩展性**：
  * 蜜蜂脑（1,000,000 细胞）：单步推理耗时 12.90 ms，吞吐率 48.5 Ticks/s。
  * 亿级张量脑（100,000,000 细胞）：支持 CUDA 流式并行加速，单脑显存占用受控于 3.3 GB。
* **鲍德温效应 (Baldwin Effect)**：后天突触慢速塑性权重在代际繁衍中实现 100% 无损固化，有效抵抗环境分布漂移。

---

## 目录结构

```
kun-cellular/
|-- include/kun/cellular/      # C++20 Header-Only 核心算法库
|   |-- cellular_genome.hpp    # 元胞基因组、空间力场与 Kahn 零 GC 编译器
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
