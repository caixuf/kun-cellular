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
KEY NUMBERS: trunk probe out[0..15] = 0（全部）；adapter 0% vs champion maze 80% / cartpole 100% / maglev 100%
CONTROLS: 同 seed 的 from-scratch 冠军正常（cartpole/maglev 100%），排除评测代码整体坏掉
CONCLUSION: 1M bin 是压测拓扑，无可用特征；S1 需先有训练过的 1M 主干
NEXT: 预训练 1M 主干 / 或用 8k flagship 主干做 S1-preview / 或停规模线
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

### 3.2 Quick 评测（1 seed，10 episodes）

| 任务 | Adapter | From-scratch 冠军 | 结论 |
|---|---:|---:|---|
| maze | 0% | 80%（10 episodes 小样本；全量约 98%） | adapter 无信息 |
| cartpole | 0% | 100% | adapter 无信息 |
| maglev | 0% | 100% | adapter 无信息 |

- 冠军正常，说明评测循环没有整体坏掉；
- adapter 在三个任务上全 0，与“特征恒为 0”一致。

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

所以：

- **不能**把 S1 的失败解释成“冻结主干 + 适配器没用”；
- 正确结论是：**S1 的前提（存在一个可用的 1M 特征主干）不成立**；
- 这也解释了为什么 Phase 0 的资产核验必须放在规模线之前。

---

## 5. 判定

| 判据 | 结果 |
|---|---|
| adapter ≥ champion − 5pp（三任务） | ❌ 全 0 vs 80/100/100 |
| 适配器参数 <1% 主干 | ✅ 154 / 1,000,000 |
| **S1** | ❌ **BLOCKED/FAIL（资产不成立）** |

**E 标签**：E3（资产/前提失败，不是能力结论）。

---

## 6. 下一步选择

### 选项 A：先预训练 1M 主干

- 用任务分布预训练 1M 细胞主干（自监督下一状态 / 多任务控制）；
- 这是数天~数周的工程，不是一次 benchmark；
- 训练完成后，S1 才真正可测。

### 选项 B：S1-preview on 8k flagship

- 用 `flagship_io_mix_n64_s20260910.bin`（8,196 cells）等已有旗舰主干；
- 先验证“冻结主干 + rank-8 适配器”的 recipe 是否成立；
- 如果 8k 都不成立，1M 不用谈；如果 8k 成立，再投入 1M 预训练。

### 选项 C：停规模线

- 回到 E-A/B/C 的机制线，先把“串行草稿纸 + 组合”做深；
- 规模线等有训练主干再开。

**建议顺序**：B → A → C。先用 8k 验证 recipe，成本最低；不要直接投入 1M 预训练。

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
