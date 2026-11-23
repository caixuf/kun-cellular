# Handoff — E-B v2 延迟泛化（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §2 E-B v2
> 结论：**H2-v2 PASS**。混合延迟训练让**同一个 scratchpad 网络**覆盖 D=2/8/16/32；
> 单延迟训练在非训练延迟上只有 37.5%–59%；1 通道已足够。

---

## 1. 证据格式

```text
PHASE: E-B v2
COMMIT: 4484d54
COMMAND: KUN_COMMIT=4484d54 ./build/bench_scratchpad_generalize --gens 120 --pop 24 --seeds 4 --eval-episodes 100 --out runs/scratchpad_generalize_20261123.json
ARTIFACT: runs/scratchpad_generalize_20261123.json
GATE: H2-v2 PASS
KEY NUMBERS: mixed_8ch min=100%（D=2/8/16/32 全 100%）| single_16 min=37.5% | random min=51.5% | delta_min=62.5pp
CONTROLS: mixed_8ch_random D=2/8/16/32 = 51.5/53.3/52.5/55.0%；4 seeds
CONCLUSION: 混合延迟训练得到延迟通用的外部工作记忆（E2 窄口径）；1 通道足够
NEXT: 规模智能体 / E-C 时序组合
```

---

## 2. 设计

- 任务同 E-B v1：延迟线索 + 二选一门；线索只出现 1 拍；
- 架构同 v1：8 个无状态 `OP_SUM` 记忆细胞 + 唯一外部反馈环；无内部持留；
- 训练延迟集合：
  - `single_16_8ch`：{16}（v1 基线）
  - `mixed_8ch` / `mixed_4ch` / `mixed_2ch` / `mixed_1ch`：{2,8,16,32}
  - `mixed_8ch_random`：{2,8,16,32} + 随机反馈
- 4 seeds × 120 代 × 24 pop × 每延迟 16 train episodes × 每延迟 100 eval episodes。

---

## 3. 结果

### 3.1 各条件 × 延迟（4 seeds 平均）

| 条件 | D=2 | D=8 | D=16 | D=32 | min |
|---|---:|---:|---:|---:|---:|
| single_16_8ch | 0.380 | 0.590 | **1.000** | 0.375 | 0.375 |
| **mixed_8ch** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| mixed_4ch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_2ch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_1ch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mixed_8ch_random | 0.515 | 0.533 | 0.525 | 0.550 | 0.515 |

### 3.2 判据

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---:|---|
| mixed − single（min SR） | ≥20pp | **+62.5pp** | ✅ |
| mixed_8ch 所有延迟 | ≥80% | **100%** | ✅ |
| random min SR | ≤60% | **51.5%** | ✅ |

**H2-v2 PASS。**

---

## 4. 解读

1. **延迟特化是训练分布问题，不是架构上限**：  
   v1 只在 D=16 训练，D=2/8 不会；v2 混合训练后同一个网络覆盖全部延迟。

2. **1 通道就够**：  
   二值线索只需要 1 bit；8/4/2/1 通道全部 100%。  
   这说明瓶颈不是 scratchpad 容量，而是**训练分布是否覆盖延迟**。

3. **随机反馈仍在 chance**：  
   51.5%–55.0%，与“无信息反馈”一致；增益不是“多几个输入”。

4. **这不是通用推理**：  
   任务仍是二值线索 + 固定响应；只证明“外部工作记忆可以跨延迟复用”，不证明 CoT/多步推理。

---

## 5. 标签

**E2（窄口径）**：混合延迟训练 → 延迟通用的外部工作记忆。
不是 E2 的部分：通用推理、CoT、跨任务组合、AGI。

---

## 6. 成本

| 项 | 值 |
|---|---|
| 训练 | 6 条件 × 4 seeds × 120 代 × 24 pop × 4 delays × 16 episodes |
| 评估 | 6 条件 × 4 seeds × 4 delays × 100 episodes |
| 本机墙钟 | 约 3~5 分钟（单线程 C++） |
| 参数 | 22 细胞 / ~200 突触；ES ~250 维 |

---

## 7. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_scratchpad_generalize
KUN_COMMIT=$(git rev-parse --short HEAD) \
  ./build/bench_scratchpad_generalize --gens 120 --pop 24 --seeds 4 --eval-episodes 100 \
  --out runs/scratchpad_generalize_20261123.json

ctest --test-dir build -R test_scratchpad_generalize --output-on-failure
```
