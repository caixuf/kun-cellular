# DomainZoo 评测隔离冷评 (2026-09-15)

> 测试：`tests/test_flow_domain_zoo_eval_isolation.cpp`  
> 产物：`checkpoints/domain_zoo_isolation_audit_20260915.json`  
> **不改底座、不覆盖 `zoo_*.bin`、不改冻结 `domain_zoo_report.json`。**

## 污染机制（任务层可见，底座未动）

`ZooTask::reset()` / `reset_physics()` **不清 `o_[]`**。`evaluate_organism` 在第一拍 `forward` 之前读 `current_observation()`，因此从第 2 个种子起，控制器吃到上一回合终态观测。这不是细胞膜孔道问题的全部，但是冷评能钉死的任务层泄漏。

## 三条协议

| 协议 | 做法 | cartpole ID SR |
|---|---|---|
| JSON 冻结 | `train_domain_zoo` 内存冠军当场评 | **0.9** |
| leaky | 磁盘 bin + `evaluate_organism`（共享环境） | **0.5** |
| clone | 每回合新环境 + 从未 forward 的模板拷贝 | **0.1** |
| fresh | 每回合 `load+compile` + 新环境 | **0.1** |

clone 与 fresh 位级一致（0.1）。JSON 0.9 **不是**当前磁盘 bin 的可复现数。

## M1（同一阈值：train≥0.70，ID≥0.60 且比≥0.70，OOD≥0.50 且比≥0.50）

- leaky：**11/12**（cartpole 0.5/0.5/0.5 已 FAIL）
- clone：**10/12**（cartpole；ballbeam 隔离 OOD **0.4**）

不得把冻结 JSON 的 12/12 改口成「隔离后仍 12/12」。也不得为填表重训覆盖 bin。

## 复现

```bash
./build/test_flow_domain_zoo_eval_isolation
```
