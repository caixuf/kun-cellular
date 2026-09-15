# 细胞生命体战役结案 (2026-09-15)

> 稳定性战役 **冻结**。不再为填表扩生境、不重训 Zoo 回填 12/12、不改 `mutate` / Cell 热路径。  
> 工程主线转到兄弟仓：[flowserve](https://github.com/caixuf/flowserve)（推理）、[flowtrain](https://github.com/caixuf/flowtrain)（训练切分）。

## 冻结时能说的

- 可演化数据流图是有效小控制器：ADAS / maze / cartpole / household **门禁 6/6**（迷宫靠测地脚手架，household G6 禁止与全图教师对撞成功计数）。
- 接线承重、有丝分裂不承重（ADAS 拓扑消融 + ZooCartPole 旋钮）。§5.9 四启发式默认仍未进 `evolve_generation`。
- 图是确定性前向机：同 bin 同种子 G5 回放位级重合。
- 描述带 + 构造器开关的最小协议（切断/脏带失败）。不是 28 原语通用构造器，不是 C++ 拷贝。
- DomainZoo：**冻结 JSON 12/12 是泄漏评测**；`o_[]` 卫生后隔离 **10/12**。G6 抖动 8/12 **FAIL**。

## 冻结时不能说的

- 非冯机器、车规、夏普 0.22、JSON 隔离仍 12/12、迷宫自己长出地图、C++ 拷贝=自复制、§5.9 四算子完备消融。

## 本战役明确不做（WON'T）

| 项 | 原因 |
|---|---|
| `reset_state` 清 `membrane_pores` | 底座热路径，需全域回归 |
| 凋亡从 `mutate()` 解耦 | 同上 |
| 重训 Zoo 填回 12/12 | 粉饰隔离负例 |
| 斗地主 G5/G6、量化 G4/G5/G6 | 未测就是未测 |
| 改论文摘要 / 回改 `dist/` | 避免大段漂移；发行包属历史 |

## 观测台

`bash tools/run_observatory.sh` → 默认体育场学生档（10W/6L，p95 0.145 m）。L3 锁档 9W/7L 是另一把 bin。

## 复现入口

`docs/handoff/2026-09-14-gate-matrix.md` 第四节命令清单 + `./build/test_flow_von_neumann_constructor` + `./build/test_flow_domain_zoo_eval_isolation`。
