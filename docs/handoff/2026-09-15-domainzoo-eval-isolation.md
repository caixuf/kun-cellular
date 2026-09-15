# DomainZoo 评测隔离冷评 (2026-09-15)

> 测试：`tests/test_flow_domain_zoo_eval_isolation.cpp`  
> 卫生：`ZooTask::reset` 清 `o_[]`；`reset_state` 还原 `kDefaultMembranePores`；隔离 harness 对齐 OOD 步数并把 clone 接到 `evaluate_organism`。  
> **不覆盖 `zoo_*.bin`，不改冻结 `domain_zoo_report.json`。**

## 锁定数字

| 锚点 | JSON 冻结 | 隔离 leaky / clone / fresh |
|---|---|---|
| cartpole ID SR | 0.9 | **0.1 / 0.1 / 0.1** |
| cartpole OOD SR | （泄漏评测） | **0.1 / 0.1** |
| ballbeam OOD SR | （泄漏评测） | **0.3 / 0.3** |
| M1 | 12/12 | **10/12**（cartpole；ballbeam OOD 0.3） |

12 域 leaky 与 clone 行对齐。旧口径「cartpole OOD leaky 0.3 vs clone 0.1 / ballbeam OOD 0.4」来自 OOD 环境 `max_steps_=300` 却按 600 步循环计分，以及手写 clone 循环与 `evaluate_organism` 不一致。膜孔道复位本身不改变控制 SR（`dispatch_cell_forward` 不读孔道），但仍必须做：回合间代谢门控不得残留。

## 复现

```bash
./build/test_flow_domain_zoo_eval_isolation
```
