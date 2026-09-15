# DomainZoo Gate 5 / Gate 6 (2026-09-15)

> G5 落地：`tests/test_flow_domain_zoo_gate5_replay.cpp` **PASS**。  
> G6 落地：`tests/test_flow_domain_zoo_gate6_shadow.cpp` **FAIL**（负例锁档，ctest 绿 ≠ 过关）。  
> 不改底座；不覆盖 `zoo_*.bin`；不改 `domain_zoo_report.json`。

## Gate 5

- 12 域，`ood=1.0`，ID 种子 `201`，maglev 600 步 / 其余 300 步
- `reset_state(true)` 对齐 `ZooTask::evaluate_organism`
- 实测 12/12：`max_dobs=0 max_dact=0`
- cartpole 本种子 16 步、rocket 52 步即越包络，仍位级重合

## Gate 6（ID 种子 201–210）

硬阈本拟 12 域 mean\|ΔF\| < 0.80 **且** 冠军生存 ≥7/10。实测达不到，锁 FAIL：

| 域 | PD 专家 | 冠军生存 | mean\|ΔF\| |
|---|---|---|---|
| cartpole | 10/10 | **1/10** | 0.634 |
| ballbeam | **0/10** | 10/10 | 0.133 |
| maglev | 10/10 | 10/10 | 0.215 |
| rocket_hover | 10/10 | 8/10 | **2.386** |
| cruise | 10/10 | 10/10 | 0.296 |
| thermal | 10/10 | 10/10 | **3.068** |
| water_tank | 10/10 | 10/10 | 0.012 |
| dc_motor | 10/10 | 10/10 | 0.016 |
| vibration | 10/10 | 10/10 | 0.443 |
| servo | 10/10 | 10/10 | **1.946** |
| boiler | 10/10 | 10/10 | 0.303 |
| bicycle | 10/10 | 10/10 | **2.228** |

汇总：jitter **8/12**，champ≥7 **11/12**，PD≥7 **11/12**。

`domain_zoo_report.json` 写 cartpole `id_sr=0.9`。`evaluate_organism` 跨回合复用有机体可报到 5/10；**fresh-load 每回合独立 compile 是 1/10**。JSON 0.9 不是本测试口径，不得勘误成「已复现」。
