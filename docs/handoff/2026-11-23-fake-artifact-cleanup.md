# Handoff — 虚假/膨胀资产清理（2026-11-23）

> 范围：`runs/` 本地假产物、`checkpoints/` 命名膨胀、ADAS 规模假指标、CUDA 静默空转。
> 原则：**能证明为假的删除；结构真但无 provenance 的隔离；命名膨胀按 header 重命名；假指标连同门禁一起删。**

---

## 1. 删除（可证明为假 / 空壳）

| 路径 | 原因 |
|---|---|
| `runs/hundred_million_summary.json` | 声称 `TRAINED_ON_CUDA`，实际仅 2 代 / 69 s，无任务、无评测 |
| `runs/hundred_million_champion.pt` | 2.5 GB，结构为 100M 细胞 / 200M 突触，但无训练 provenance；被服务端当作“已训练大脑”加载 |
| `runs/laokexia_billion_summary.json` | 训练耗时 9.25e-05 s、fitness 9999、1B 对话超级大脑；明显空壳 |
| `runs/deprecated_hollow_artifacts/` | 已标注 hollow 的 1B / real_billion / language_model_1b `.pt` 与假业务目录 |

## 2. 隔离（结构真、无 provenance，不删）

移动到 `runs/quarantine_unproven/`，服务端不再加载：

| 文件 | 结构 |
|---|---|
| `adas_million_champion.pt` | 1M 细胞 / 2M 突触，fitness 1011.7，无评测 |
| `cellular_language_model_10m.pt` | 224×384 微型 transformer，无训练 loss/评测 |
| `mathematician_ten_million_champion.pt` | 10M 细胞 / 20M 突触，fitness 587.97，无评测 |
| `pure_neural_arithmetic_10m.pt` | 10M 细胞，无评测 |
| `quant_million_brain_evolved_champion.pt` | 1M 细胞，无评测 |
| `quant_million_brain_seed42.pt` | 1M 细胞，seed 初始化，无评测 |

> 这些文件可能包含真实实验数据，因此不删除；但在补齐「任务 + 评测 + seed + 命令」之前，不得进入任何能力叙事，也不得被前端/后端当作冠军加载。

## 3. 重命名（命名膨胀 → 真实 header）

| 旧名 | 新名 | 真实细胞数 |
|---|---|---:|
| `checkpoints/adas_transient_1m.bin` | `checkpoints/adas_transient_320c.bin` | 320 |
| `checkpoints/adas_occupancy_10m.bin` | `checkpoints/adas_occupancy_624c.bin` | 624 |
| `checkpoints/adas_world_model_100m.bin` | `checkpoints/adas_world_model_1536c.bin` | 1536 |
| `checkpoints/quant_cross_asset_10m.bin` | `checkpoints/quant_cross_asset_640c.bin` | 640 |
| `checkpoints/quant_market_making_1m.bin` | `checkpoints/quant_market_making_320c.bin` | 320 |
| `checkpoints/quant_world_model_100m.bin` | `checkpoints/quant_world_model_1280c.bin` | 1280 |

引用更新：`tests/test_checkpoint_roundtrip.cpp`、`tools/cellular_live_backend.py`。

## 4. 代码修复

### 4.1 CUDA 运行时 fail-closed

`include/kun/cellular/sdsc_cuda_runtime.hpp`：

- NVRTC 编译失败 → `compile_ok_ = false`，不再继续 `cuModuleLoadDataEx`；
- `upload()` 编译失败 / 分配失败 / 拷贝失败 → 返回 `false`；
- `forward()` 在 `ready_ == false` 时拒绝执行，不再静默空转；
- 析构释放 primary context，修掉 teardown `invalid resource handle`；
- 宏记录 thread-local 最近错误，不再只打印。

后果：编译失败 → `upload()` 返回 `false`，`forward()` 拒绝执行；直接运行 `./build/test_adas_scales` 时若未带 NVRTC builtins 路径，返回 **77 (SKIP)**，不是 PASS。CMake 已设置 `SKIP_RETURN_CODE 77`。设置 `LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/nvidia/cu13/lib` 后，kernel 真实执行。

### 4.2 ADAS 规模假指标

`tools/bench_adas_scales.cpp`：

- 删除 `: 0.88f` 保真度硬编码；
- 删除“预测地平线 / 盲区觉醒度 / 智驾本质能力”字段与汇总行；
- 改名 `bench_1m_reflex` → `bench_1m_stress` 等；标题改为“硬件吞吐压测（非能力报告）”；
- upload 失败直接 `exit(2)`，不再打印假延迟。

`tests/test_adas_scales.cpp`：

- 删除 1M `peak_lat_dev < 0.5` 断言（手工接线 + 合成栖息地，不是能力）；
- 删除 100M `prediction_horizon >= 4.0` 断言（自指指标）；
- upload 失败返回 77 SKIP。

`tasks/adas/adas_world_model_4d.hpp`：

- `prediction_horizon_seconds` → `self_reported_horizon_proxy_seconds`；
- 注释明确：所有字段是网络自报 proxy，不与 ground-truth 对账。

### 4.3 后端去假加载

`tools/cellular_live_backend.py`：

- 删除 `hundred_million_champion.pt` / `quant_million_brain_evolved_champion.pt` 专用加载路径；
- 删除按 oid 名字猜规模（`if "100m" in oid: return 100000000` 等），未知返回 0；
- `.pt` 只认 `runs/{oid}.pt`，隔离目录不再被扫描。

## 5. 文档修复

- `README.md` 诚实声明新增 4 条（100M 假训练、1B 空壳、命名膨胀、ADAS 假指标）；
- `docs/ADAS_SCALE_BENCHMARK_REPORT.md` 整篇重写为“硬件压测（非能力报告）”；
- `docs/morphogenetic_cellular_evolution_paper.md` / `.zh.md`：撤回 100M GPU 吞吐与能力表；
- `docs/superpowers/plans/2026-11-23-borrow-llm-reasoning-and-scale.md`：Phase 0 资产核验表更新。

## 6. 本机复现结果（2026-11-23）

```text
./build/test_binary_runtime_scale
  1M: 15.605 ms/step, 64.08 MCells/s, mmap 5.786 ms

./build/test_compact_10m_scale
  10M: 200 MB, 26.65 ms/step, deep copy 134.6 ms, mutate 27.2 ms

./build/test_cellular_relaxation
  8/8 PASS

./build/test_checkpoint_roundtrip
  ALL PASS（含重命名后的 320c/624c/1536c）

./build/test_adas_scales
  直接运行：NVRTC code=7 → 77 SKIP
  ctest 环境（LD_LIBRARY_PATH 已设置）：
    1M   0.084 ms/step, 11.46 GCells/s
    10M  1.349 ms/step,  7.86 GCells/s
    100M 11.63 ms/step,  8.21 GCells/s
  （合成拓扑，仅硬件延迟；能力未评估）

./build/test_cuda_scale
  ctest 环境：10M 1.34 ms/step, 7.48 GCells/s
  旧的 237.97 GCells/s 是空转 forward 产物，已撤回
```

## 7. 后续硬门

1. 直接运行 CUDA 用例必须带 NVRTC builtins 路径；否则返回 77 SKIP，**不得当作 PASS**；
2. 100M **推理/延迟**已有合成 bench（12.18 ms / 8.21 GCells/s）；100M **训练**仍无证据，进 S3 前必须补训练 provenance；
3. 任何新 checkpoint 命名必须带真实细胞数（CI `check_checkpoint_honesty.py` 会拦）；
4. 任何“预测/能力”指标必须与 ground-truth 对账，禁止自指。
