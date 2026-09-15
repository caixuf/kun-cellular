# DomainZoo 评测隔离冷评 (2026-09-15，o_[] 卫生后)

> 测试：`tests/test_flow_domain_zoo_eval_isolation.cpp`  
> 卫生：`tasks/control/domain_zoo.hpp` `ZooTask::reset` 清零 `o_[]`  
> **不覆盖 `zoo_*.bin`，不改冻结 `domain_zoo_report.json`。**

## 卫生效果

| cartpole ID SR | 卫生前 | 卫生后 |
|---|---|---|
| JSON 冻结 | 0.9 | 0.9（未改） |
| leaky `evaluate_organism` | 0.5 | **0.1** |
| clone / fresh | 0.1 | **0.1** |

ID 上跨回合脏观测已堵住。M1：leaky 与 clone 均为 **10/12**（cartpole；ballbeam OOD 0.4）。卫生前 leaky 曾是 11/12——那多出来的 1 格是脏观测送的。

残留：cartpole OOD leaky 0.3 vs clone 0.1。这是同一有机体被 `evaluate_organism` 跨回合复用，`reset_state` 不清 `membrane_pores`（底座，本回合未改）。

## 复现

```bash
./build/test_flow_domain_zoo_eval_isolation
```
