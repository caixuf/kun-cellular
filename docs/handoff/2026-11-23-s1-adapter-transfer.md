# Handoff — S1 冻结 1M 主干 + rank-8 适配器（2026-11-23）

> 预注册：`docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md` §3.3
> 结论：**S1 BLOCKED / FAIL**。`checkpoints/sdsc_mega_1million.bin` 的 16 维输出**恒为 0**；
> 它不是训练过的特征主干，而是压测拓扑。适配器在零特征上不可能学习。
> 这不是“适配器 recipe 失败”，是**资产不成立**。

---

## 1. 证据格式

```text
PHASE: S1
COMMIT: <S1 代码 commit>
COMMAND: LD_LIBRARY_PATH=<nvidia/cu13/lib> ./build/bench_adapter_transfer --quick
ARTIFACT: runs/adapter_transfer_quick.json（正式 JSON 未生成：零特征下恒为 0）
GATE: S1 BLOCKED/FAIL
KEY NUMBERS: 1M probe 0；8k 稀疏输入 0；ADAS 210c preview: maze 8% / cartpole 0% / maglev 96%（冠军 96/100/100）
CONTROLS: 同 seed 冠军正常；ADAS 主干在 maglev 上 −4pp（在 −5pp 内），maze/cartpole 大幅失败
CONCLUSION: 1M/8k 不是训练主干；ADAS 单任务主干只在 maglev 上验证了 recipe，不通用
NEXT: 多任务预训练主干 / 或停规模线，回到机制线
```

---

## 2. 做了什么

`tools/bench_adapter_transfer.cpp`：

1. `sdsc_binary_load()` mmap 读取 `sdsc_mega_1million.bin`（1,000,000 cells / 3,000,000 syns，64 in / 16 out）；
2. 转成 `CompactSoAGenome`，上传 `SdscCUDAGraph`；
3. 主干完全冻结；
4. 适配器：16 维主干输出 → rank-8 → 2 维动作，**154 参数**（约为 1M 细胞的 0.015%）；
5. ES 只调适配器；在 maze / cartpole / maglev 上与 from-scratch 冠军对比。

---

## 3. 结果

### 3.1 主干特征探针

```text
trunk probe out: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

- 输入 `{0.1, -0.2, 0.3, -0.4}`（其余 60 维补 0）后，16 个输出全 0。
- 这不是本次转换的 bug：官方 `test_binary_runtime_scale` 也只打印 `out[0]=0.0000`，
  它验证的是“无 NaN + 零分配 + 吞吐”，**不验证输出非零**。

### 3.2 8k flagship S1-preview

为排除“只是 1M 太大/转换错”，又用 8k flagship 做了 preview：

| 项 | 值 |
|---|---|
| 主干 | `checkpoints/flagship_io_mix_n64_s20260910.bin` |
| 规模 | 8,196 cells / 35,848 syns；4096 in / 4 out |
| 后端 | CPU（GPU 输入注入忽略 receptor 标志，另行记录） |
| 适配器 | 4 维特征 → rank-8 → 2 动作，58 参数 |

探针结果：

```text
4 维稀疏输入 (0.1,-0.2,0.3,-0.4)，其余 0  ->  acts (0, 0, 0, 0)
4096 维全 1                             ->  acts (1.033, 0, -0.047, 0)
4096 维其余 0.1                         ->  acts (0.103, 0, -0.0047, 0)
```

结论：这个 8k 主干基本是 **输入均值/IO 混合**，不是任务特征提取器。
把 4 维观测塞进前 4 个通道，输出就是 0；适配器拿不到任何信息。

顺带修了一个底座 bug：`sdsc_binary_load` 的版本检查漏了 **v4**，而所有 flagship 都是 v4。
（`include/kun/cellular/sdsc_binary_runtime.h`）

### 3.3 Quick 评测（1M，1 seed，10 episodes）

| 任务 | Adapter | From-scratch 冠军 | 结论 |
|---|---:|---:|---|
| maze | 0% | 80%（10 episodes 小样本；全量约 98%） | adapter 无信息 |
| cartpole | 0% | 100% | adapter 无信息 |
| maglev | 0% | 100% | adapter 无信息 |

- 冠军正常，说明评测循环没有整体坏掉；
- adapter 在三个任务上全 0，与“特征恒为 0”一致。

### 3.4 210-cell ADAS trained trunk preview（选项 D）

用真正训练过的 `checkpoints/adas_cortex_champion.bin`（210 cells / 659 syns，6 in / 2 out）
做冻结主干；直接读 6 个执行器细胞的 `output_val` 作为特征（`ActionOutputs` 只暴露 4 个字段，
读不到 `ACT_CHANNEL`）。适配器：6 维 → rank-8 → 2 动作，42 参数。

正式：2 seeds × 120 代 × 24 pop × 4 train episodes × 100 eval episodes。

| 任务 | Adapter | 冠军 | Δ | 判定 |
|---|---:|---:|---:|---|
| maze | 8% | 96% | −88pp | ❌ |
| cartpole | 0% | 100% | −100pp | ❌ |
| **maglev** | **96%** | **100%** | **−4pp** | ✅ 在 −5pp 内 |

- **Maglev 上 recipe 成立**：冻结 ADAS 主干 + 42 参数适配器，达到冠军的 96%；
- Maze / cartpole 不成立：ADAS 特征不含这两个任务需要的信息；
- 这是“**单任务主干不通用**”，不是“adapter 完全无效”。

训练墙钟：maze ~12.6s/seed，cartpole ~0.6s/seed，maglev ~4.0s/seed。
（maze 慢是因为 250 步 × 更多 episode；不是主干慢。）

### 3.3 训练成本（quick）

| 任务 | 墙钟 |
|---|---:|
| maze | ~4.0 s |
| cartpole | ~0.43 s |
| maglev | ~0.29 s |

（正式 3-seed 未跑：零特征下结果恒为 0，跑 3 次只会得到 3 个 0，不构成额外证据。）

---

## 4. 根因

`sdsc_mega_1million.bin` 的定位是**规模压测拓扑**：

- 1M 细胞 / 3M 突触，用来测 mmap、CSR、零 GC、吞吐；
- 它的 op 分布是随机/合成的（census: `legacy_sdsc_opcode`，均匀分布）；
- 前 64 个细胞不是 sense 受体，输入注入语义不成立；
- 输出恒 0，官方测试也接受这个结果（只查 NaN）。

`flagship_io_mix_n64_s20260910.bin` 的定位是 **IO 混合实验**：

- 4096 输入 / 4 输出，输出近似“所有输入的均值”；
- 4 维稀疏输入 → 0；全 1 → 1.03；其余 0.1 → 0.103；
- 它不是任务特征提取器，适配器同样拿不到信息。

所以：

- **不能**把 S1 的失败解释成“冻结主干 + 适配器没用”；
- 正确结论是：**S1 的前提（存在一个可用的训练主干）不成立**；
- 这也解释了为什么 Phase 0 的资产核验必须放在规模线之前。

---

## 5. 判定

| 判据 | 结果 |
|---|---|
| adapter ≥ champion − 5pp（三任务） | ❌ maze 8% / cartpole 0% / maglev 96% |
| 适配器参数 <1% 主干 | ✅ 42 / 210（20%，但绝对量小；仍不是 1M 规模） |
| **S1** | ❌ **BLOCKED/FAIL（无通用训练主干）** |

**E 标签**：E3（资产/前提失败），不是“适配器 recipe 全面失败”。
Maglev 的单任务证据是 E1：recipe 在“主干域接近任务域”时成立。

---

## 6. 下一步选择

### 选项 B：S1-preview on 8k flagship（已做，也失败）

- 用 `flagship_io_mix_n64_s20260910.bin`（8,196 cells）等已有旗舰主干；
- 结果：8k 是 IO 混合实验，4 维稀疏输入输出 0；recipe 仍无法验证。

### 选项 D：用真正的训练主干做 preview（已做）

- `checkpoints/adas_cortex_champion.bin`（210 cells / 659 syns，6 in / 2 out）是**真训练过的**车道保持皮层；
- 结果：**maglev 96% vs 100%（−4pp，在 −5pp 内）**；maze 8%、cartpole 0%；
- 结论：recipe 在“主干域接近任务域”时成立，但单任务主干**不通用**；
- 这排除了“adapter 完全无效”，但没有解锁 S1 的 3 任务判据。

### 选项 A：先预训练多任务主干

- 用任务分布预训练主干（自监督下一状态 / 多任务控制）；
- 这是数天~数周的工程，不是一次 benchmark；
- 这是唯一能让 S1 真正可测的路径；单任务主干已经被 D 证伪为“不通用”。

### 选项 C：停规模线

- 回到 E-A/B/C 的机制线，先把“串行草稿纸 + 组合”做深；
- 规模线等有多任务训练主干再开。

**建议**：B/D 已做，A 是唯一能解锁 S1 的路径但成本高；若不想投入，选 C。
不要再用 1M/8k 压测拓扑冒充训练主干，也不要拿单任务 ADAS 主干硬凑三任务。

---

## 7. 复现

```bash
cd /home/caixuf/code/kun-cellular
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target bench_adapter_transfer

LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
  ./build/bench_adapter_transfer --quick

ctest --test-dir build -R test_adapter_transfer --output-on-failure
```
