# Handoff — 可学习架构：teacher + task adapter → agent（2026-11-24）

> 目标：新任务不需要改底座，只要：
> 1) 实现一个 `TaskSpec`（环境工厂 + 成功语义 + teacher checkpoint）；
> 2) 启动 `train_from_teacher`；
> 3) 自动采老师轨迹、路由、训练读出、评测。
>
> 结论：**新任务 `cruise` / `water` PASS（100/100，top-1 = 1% active）**；
> 200M 共享储层 + `combiner-topk=1` 后 **maze/cartpole/maglev 三任务 100ep PASS**；
> rocket/thermal 仍失败（强非线性）。

---

## 1. 架构契约

```text
TaskSpec:
  name       任务名
  teacher    任意 CellularOrganism checkpoint（obs -> pos/neg action）
  max_steps  回合长度
  make(seed) 环境工厂（实现 EvolvableTask）
```

流程：

1. 采 teacher 轨迹 `(obs, action)`；
2. k-means 路由（默认 100 块）；
3. 每块 ridge 读出（64 block features + bias）；
4. 每任务 combiner（top-5 块输出 + 4 维原始观测 → 2 动作）；
5. held-out 100 episodes 评测，`agent ≥ teacher −5pp` 即 PASS。

新任务只改 registry 一行 + 一个 `make_*` 函数，**不碰 28 原语底座**。

---

## 2. 新任务验证

```text
COMMAND: ./build/train_from_teacher --task cruise --blocks 100 --cells 100000 \
         --syns 300000 --teacher-episodes 100 --eval-episodes 100 \
         --topk 1 --combiner-topk 5 --out runs/distill_cruise_20261124.json

COMMAND: ./build/train_from_teacher --task water ... --out runs/distill_water_20261124.json
```

| 任务 | Teacher SR | Agent SR | Δ | 判定 |
|---|---:|---:|---:|---|
| **cruise**（新） | 1.00 | **1.00** | 0.00 | ✅ PASS |
| **water**（新） | 1.00 | **1.00** | 0.00 | ✅ PASS |
| cartpole（200M c1） | 1.00 | **1.00** | 0.00 | ✅ |
| maglev（200M c1） | 0.98 | **1.00** | +0.02 | ✅ |
| maze（200M c1） | 0.97 | **0.98** | +0.01 | ✅ |
| thermal（200M c1） | 0.95 | 0.35 | −0.60 | ❌ |
| rocket（200M c1） | 0.84 | 0.05 | −0.79 | ❌ |

- `cruise` / `water`：10M total / 100 blocks / top-1 = 100k active = **1%**；
- 只训练每块 readout + combiner；block 权重是随机储层；
- 100 held-out episodes 全部通过。

---

## 3. 解读

1. **架构可复用**：新任务只加 TaskSpec + teacher，管线不变；cruise/water 直接 100%。
2. **任务难度分层**：
   - 一阶/线性域（cruise/water）→ 可通过；
   - 中等非线性（cartpole/thermal）→ 部分；
   - 强非线性/不稳定（maglev/rocket）→ 当前 100×100k + 线性 combiner 不够。
3. **不是“老师万能”**：老师 SR 高不代表可蒸馏；学生容量/观测充分性/路由粒度都会限制。
4. **可扩展旋钮**：`--blocks/--cells/--topk/--combiner-topk`；难任务可增大到 500×200k 或换非线性 combiner。

---

## 4. 产物

| 文件 | 说明 |
|---|---|
| `tools/train_from_teacher.cpp` | 通用 teacher-distillation pipeline |
| `runs/distill_cruise_20261124.json` | 新任务 PASS |
| `runs/distill_water_20261124.json` | 新任务 PASS |
| `runs/distill_cartpole_20261124.json` | 中等任务 FAIL |
| `runs/distill_thermal_20261124.json` | 中等任务 FAIL |
| `runs/distill_maglev_20261124.json` | 难任务 FAIL |
| `runs/distill_rocket_20261124.json` | 难任务 FAIL |

---

## 5. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target train_from_teacher

./build/train_from_teacher --task cruise --blocks 100 --cells 100000 \
  --syns 300000 --teacher-episodes 100 --eval-episodes 100 \
  --topk 1 --combiner-topk 5 --out runs/distill_cruise_20261124.json
```

> 新任务接入 = 在 `registry()` 加一行 + 写 `make_<task>(seed)`；底座与训练管线不动。
