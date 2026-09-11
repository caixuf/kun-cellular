# 战役总结：FieldCML 收口 → 任务层 T1–T6（2026-09-11）

> 用户裁决：核心机制收口后转入任务；顺序代理拟定。本文件为整场战役一页纸结论。

---

## 总表

| 序 | 战役 | 结果 | 一句话 |
|----|------|------|--------|
| 核心 | FieldCML / io_mix | ✅ 冻结 | 发育默认：结构化内部 + 随机 IO；io_evo / RRAC 负 |
| T0/T1 | DomainZoo | ✅ | 12/12 锁档；README 旧 9/12 已勘误 |
| T2 | Household | ✅ | 冷评塌缩根因=`reset_state(true)`+未同步 `initial_weight`；修复后 ID~88% |
| T3+L3 | ADAS | ⚠️→✅ | 历史 7W9L 漂；L3 增益重调 → **9W/7L** |
| T4 | 易证三任务 | ✅（迷宫经 L3） | CartPole/流体绿；迷宫欧氏~80–86% 负 → 测地方位脚手架 **96/100** |
| T5 | 量化皮层阵列 | ❌ 负 | `--seed 0` 复训选择集过拟合（夏普 2.16），OOS **−0.03**；论文 0.22 未复现 |
| T6 | 斗地主 | ✅ 锁档 | 57.0%@2000，vs 教师 p=0.13 **持平**；未再训、未宣称超越 |

---

## 分项要点

### 核心机制（非任务）
- 可用配方：`io_mix` + `train_plast=0`
- 不要再磨：结构化 DAG 必胜、广谱 IO 逐边演化、未预注册的 64K 放大

### T2 Household — 工程铁律
- 落盘/变异后必须 `initial_weight = weight`
- 无塑性评测优先 `reset_state(false)`
- 同类坑已在迷宫 tripartite / CartPole balance 旧档复现

### T4→迷宫 L3 — 任务可解性
- 「简单迷宫」实为连续控制 + 局部传感；欧氏方位在死胡同给错梯度
- 任务层 `set_use_geodesic_bearing(true)` 解锁 96%@250 / 100%@400（不动底座）

### T5 量化 — 训练手法，非底座
- 失败是选择集过拟合 → OOS 翻车，不是细胞原语算错
- `CorticalMacroArray` 缺 load，无法对正式 bin 冷评；复训 ≠ 论文冠军
- 不得恢复已撤实盘收益宣称

### T6 斗地主 — 评测锁，非再开 RL
- 现仓 md5 `9513b89d…`；57.0% / Wilson 54.8% / 过牌 47.7%
- vs 教师 58.3%，McNemar p=0.128 → **持平**；超越仍未成立

---

## 复现入口（精简）

```bash
./build/train_domain_zoo
./build/bench_easy_task_regression
./build/test_multiphase_fluid_stress
OMP_NUM_THREADS=6 ./build/train_multi_asset_cortical_array --seed 0 --report runs/quant_repro.json  # 预期负
./build/p9_runner 2000 checkpoints/doudizhu_cand_scorer.bin
./build/p9_paired 2000 checkpoints/doudizhu_cand_scorer.bin checkpoints/doudizhu_cand_scorer.bin 3100000 teacher
```

---

## 下一步（建议，非本文件承诺）

1. 量化若再攻：任务层选种/走步/阵列 load 冷评 — **勿动 L0**
2. 斗地主若再攻：须新预注册「动作空间/更强对手」；禁止无协议长线自博弈
3. 文档：论文若仍写 DomainZoo 9/12、迷宫 100% 欧氏、量化可复现 0.22 — 按本战役结果勘误
