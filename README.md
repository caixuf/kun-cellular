# Kun-Cellular (Kun 形态发生计算图与人工生命底座)

[![CI](https://github.com/caixuf/kun-cellular/actions/workflows/ci.yml/badge.svg)](https://github.com/caixuf/kun-cellular/actions)
![C++20](https://img.shields.io/badge/C++-20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> 基于胚胎形态发生（Morphogenesis）、空间哈希力场自组织、鲍德温效应（Baldwin Effect）基因固化与开放式具身演化的通用人工生命（Artificial Life）与计算图基础底座。

---

##  核心特性

1. **零 GC 拓扑线性化编译 (Zero-GC Topology)**：
   - 胚胎通过力场自组织分化为数万至数百万细胞的有向图谱，通过 Kahn 算法线性化为平铺连续内存数组，消除动态内存分配，实现纳秒级（~150ns）至毫秒级确定性时延。
2. **鲍德温效应 (Baldwin Effect Crystallization)**：
   - 后天突触慢速塑性权重在代际繁衍中部分固化为下一代先天遗传基线，实现“以变化对抗分布漂移”。
3. **宏观生态圈与红皇后协同演化 (Ecosystem & Red Queen)**：
   - 生产者、消费者、顶级掠食者、清算派在生境季相轮替下动态自平衡；
   - 数字病原体抗原漂移与宿主群体免疫对抗，驱动基因水平转移（HGT）。
4. **开放式具身沙盒 (Open-Ended Embodiment)**：
   - 感知-身体动作-物理位移-能量反馈的闭环热力学因果链，无监督长周期动态自维持。

---

##  目录结构

```
kun-cellular/
├── include/kun/cellular/      # C++20 Header-Only 核心算法库
│   ├── cellular_genome.hpp    # 元胞基因组与空间力场推演
│   ├── island_evolution_grid.hpp # 岛屿网格迁移演化
│   ├── ecosystem_biosphere.hpp# 宏观生态圈与食物网流动
│   ├── digital_pathogen_ecosystem.hpp # 数字病原体与免疫系统
│   ├── open_ended_embodiment.hpp # 开放式具身沙盒
│   └── ...
├── tests/                     # 20+ 单元测试与极端场景大考
├── tools/                     # GPU 张量训练器、CUDA 演化内核与 Web 交互中枢
├── docs/                      # 科学论文 (zh/en) 与演化规格定义
└── skills/                    # 专业 Agent Skills 知识库
```

---

##  快速开始

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

---

##  引用与协议
本项目采用 MIT 许可证。
