# KunCellular 规模硬件压测报告（非能力报告）

> **2026-11-23 清理重写**。本文件此前名为“ADAS 尺度物理能力阶跃与实测对账报告”，其中 1M/10M/100M 的“智驾能力/预测地平线/波前保真度”均不成立，已撤回。
> 本文件现在只记录**硬件延迟、显存、吞吐**；不记录、也不推导任何 ADAS 能力结论。

---

## 1. 为什么重写

旧报告的三类问题：

| 问题 | 具体证据 | 处理 |
|---|---|---|
| **保真度硬编码** | `tools/bench_adas_scales.cpp` 在相关范数接近零时直接返回 `0.88f`（88%），不是测量值 | 已删除；无数据即 0 |
| **预测地平线自指** | `evaluate_effectors` 只累加网络自身输出幅值，不与任何 ground-truth 未来轨迹对账；随机网络也能“预测 8 秒” | 字段更名为 `self_reported_horizon_proxy_seconds`；能力门禁已删除 |
| **合成拓扑冒充能力** | 1M/10M/100M 基因组均为手工构造/合成栖息地，未训练、未演化 | 报告与测试统一改称“硬件压测”，不再声称 ADAS 能力 |

同时，旧报告引用的 CUDA 吞吐数字来自可能未实际执行的内核（NVRTC 编译失败后 `forward()` 静默空转）。2026-11-23 已把 CUDA 运行时改为 **fail-closed**：编译/上传失败时 `upload()` 返回 `false`，`forward()` 拒绝执行。

---

## 2. 本机可复现的硬件数字（CPU 路径）

| 规模 | 资产 | 实测 | 命令 |
|---|---|---|---|
| 1M 细胞 / 3M 突触 | `checkpoints/sdsc_mega_1million.bin` | mmap 5.8 ms；**15.6 ms/step；64.08 MCells/s**；0 malloc | `./build/test_binary_runtime_scale` |
| 10M 细胞 / 10M 突触 | `CompactSoAGenome`（SoA） | **200 MB**；深拷贝 134.6 ms；变异 27.2 ms；**26.65 ms/step（375 MCells/s）** | `./build/test_compact_10m_scale` |

> 这两组数字是 2026-11-23 本机实测，可直接引用。注意：1M 的旧文档写的是 134.6 ms / 7.43 MCells/s，已过时 8.6×。

---

## 3. CUDA 路径：可用，但必须带 NVRTC builtins 路径

直接执行 `./build/test_adas_scales` 时，若 `LD_LIBRARY_PATH` 未包含
`~/.local/lib/python3.12/site-packages/nvidia/cu13/lib`，NVRTC 编译返回 code=7，
测试返回 **77 (SKIP)**。这是正确的 fail-closed 行为。

在 CTest 或显式设置路径后，CUDA kernel 真实执行。2026-11-23 实测
（合成拓扑，仅硬件延迟，不代表能力）：

| 规模 | 拓扑 | 单步延迟 | 全流水吞吐 |
|---|---|---:|---:|
| 1M 细胞 / 1M 突触 | 16 In / 8 Out，合成 | **0.087 ms** | **11.46 GCells/s** |
| 10M 细胞 / 10M 突触 | 256 In / 64 Out，合成 | **1.272 ms** | **7.86 GCells/s** |
| 100M 细胞 / 100M 突触 | 1024 In / 128 Out，合成 | **12.184 ms** | **8.21 GCells/s** |

复现：

```bash
cd build
ctest -V -R "^test_adas_scales$"
# 或
LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" \
  ./bench_adas_scales
```

### 旧数字为什么撤回

- 旧的 **237.97 GCells/s（10M）** 来自一次 NVRTC 失败后的空转 forward，不是真实吞吐；
- 旧的 **916.3 ~ 1,027.9 GCells/s（100M “pure kernel”）** 比本机全流水实测高约 **100×**，在当前树中无可信复现路径；
- 旧报告里的 **6.74 GCells/s / 14.838 ms（100M 全流水）** 与本次 8.21 GCells/s / 12.18 ms 大致同量级，可作为历史参考，但仍以本次实测为准。

**结论：可以引用 1M/10M/100M 的“硬件延迟与全流水吞吐”；不得引用 916/237 GCells/s 的“纯内核吞吐”，也不得把任何延迟数字解释为 ADAS 能力。**

---

## 4. 当前能说什么 / 不能说什么

**能说：**
- 1M 细胞的纯 C11 mmap 运行时在本机 15.6 ms/step、64 MCells/s、零堆分配；
- 10M 细胞的 SoA 基因组 200 MB、26.65 ms/step；
- CUDA 全流水在合成拓扑上：1M 0.087 ms / 11.46 GCells/s，10M 1.272 ms / 7.86 GCells/s，100M 12.18 ms / 8.21 GCells/s；
- 28 原语 + CSR + Kahn 拓扑的 CPU/GPU 路径是真实可跑的。

**不能说：**
- ❌ “100M 细胞世界模型 / 5~8 秒预测地平线”；
- ❌ “1M 爆胎 300ms 收敛 / 侧偏 0.238m”；
- ❌ “10M 波前保真度 12.2% 达标”；
- ❌ “CUDA 916~1028 GCells/s（pure kernel）”或“237 GCells/s（10M）”；
- ❌ 任何把合成拓扑当作训练产物、把硬件延迟当作 ADAS 能力的表述。

---

## 5. 相关清理

- `checkpoints/*_1m.bin` / `*_10m.bin` / `*_100m.bin` 命名膨胀 → 已按真实 header 重命名为 `*_320c` / `*_624c` / `*_640c` / `*_1280c` / `*_1536c`；
- `runs/hundred_million_champion.pt` + 假 `TRAINED_ON_CUDA` 摘要 → 已删除；
- `runs/laokexia_billion_*` → 已删除；
- `runs/deprecated_hollow_artifacts/` → 已删除；
- 其他无 provenance 的 `.pt` → 移入 `runs/quarantine_unproven/`，服务端不再加载。

详见 `docs/handoff/2026-11-23-fake-artifact-cleanup.md`。
