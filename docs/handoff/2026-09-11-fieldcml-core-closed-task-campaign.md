# 交接：FieldCML 核心收口 → 任务层战役（2026-09-11）

> 用户裁决：核心机制文档更新后，转入具体任务；任务顺序由代理拟定。
> 纪律：L0 底座免疫；业务/任务只在 `tasks/` + `tools/`；负结果如实；不回头雕结构化 DAG / 广谱 IO 重连。

---

## 1. 核心机制结论（FieldCML / 路线 A，已冻结）

| 命题 | 状态 | 证据 |
|------|------|------|
| 塑关主口径下可用发育默认 | ✅ | **`io_mix`**：结构化内部 + 受体∪效应器联合随机；3 种子 mean un_id≈**0.300**（M1–M4 ✅） |
| 结构化局部接线 > 随机 | ❌ 反向 | H4 否证；structured 长跑 un_id≈0.123 |
| RRAC（保 IO 内部随机） | ❌ | ≪ R′ / io_mix |
| 代际逐边 IO 轴突重连 `io_evo` | ❌ | μ_E≈0.131；E1–E3 负（结构搜索发生但未进盆地） |
| 中等规模自动增益 | ❌ | n=128 S1✅ S2❌；不得宣称 64K 已通 |
| 塑性 | 混淆变量 | 主协议 `train_plast=0` |

**默认配方**：FieldCML 旗舰用 `./build/train_flagship_wired … io_mix … 0 <seed>`。  
预注册：`docs/superpowers/plans/2026-09-11-io-mix-developmental-default.md`、`…-evolvable-io-axon-rewiring.md`（负结果）。

**科学对齐**：接受「随机 IO 投影先验」为合法解（ESN/ELM 同构）；可演化接线若再攻，须另开 **间接编码/超参** 预注册，禁止复读本阶段负结果。

---

## 2. 任务层战役顺序（冻结优先级）

| 序 | 战役 | 为何现在 | 出口 |
|----|------|----------|------|
| **T0** | 文档对齐 | README 仍写 DomainZoo 9/12，实仓已 12/12（`8ea0c61`） | README/STATUS/本交接一致 |
| **T1** | DomainZoo 确定性复跑锁档 | 最低成本确认「任务层地基」未漂 | `domain_zoo_report.json` 与 bin 位级/门禁一致；12/12 或如实回退 |
| **T2** | 室内覆盖（household）多种子复证 | 计划已有、易证空间任务、补论文「具身」面 | 预注册多种子门禁；过则升 README，不过写负 |
| **T3** | ADAS 多种子统计加固 | 论文自承单种子；7W9L 需置信区间 | seeds 统计表 + 与 Stanley 配对更新 |
| **T4** | 迷宫/CartPole/流体 回归烟测 | 防文档漂移；不扩机制 | CI 或一键脚本绿 |
| **T5** | 量化多资产复现锁 | 论文主审计面；易被环境漂 | 单命令复现夏普/回撤口径 |
| **T6** | 斗地主长线（后置） | 机制榜≠牌力；在线 RL 已收官持平教师 | 另开战役，不占 T1–T5 带宽 |

**明确不做（本阶段）**：FieldCML 64K 放大、GPU eval 主线、广谱 `io_evo` 变体撒网、为业务改 L0。

---

## 3. 复现入口（核心）

```bash
# FieldCML 默认发育
OMP_NUM_THREADS=6 ./build/train_flagship_wired 64 120 io_mix 16 0 20260910

# DomainZoo
./build/train_domain_zoo
```

---

## 4. 状态机

- [x] 核心收口文档（本文件 + STATUS_BOARD 增补 + README 动物园勘误）
- [x] T1 DomainZoo 复跑锁档（2026-09-11，12/12）
- [x] T2 household 修复重训（H1–H3 ✅；根因=塑性重置抹掉演化）
- [x] T3 ADAS 多种子锁档（脚本 ✅；未调参相对历史表负）
- [x] ADAS L3 重调复锁（L1–L4 ✅ → 现仓 9W/7L）
- [x] T4 易证回归烟测（CartPole/迷宫/流体）— C1✅ F1✅；**M1 负**（迷宫 80/100，非 100%）
- [ ] T5–T6 按序
