# 预注册：T6 斗地主评测锁档（非再训）— 2026-09-11

> 状态：**已结案 ✅**。只做 eval 锁档；未开长线 RL。

---

## 协议

```bash
md5sum checkpoints/doudizhu_cand_scorer.bin
./build/p9_runner 2000 checkpoints/doudizhu_cand_scorer.bin
./build/p9_paired 2000 checkpoints/doudizhu_cand_scorer.bin checkpoints/doudizhu_cand_scorer.bin 3100000 teacher
```

## 判据（冻结）

| 编号 | 判据 |
|------|------|
| D0 | 记录现仓 bin md5 |
| D1 | `p9_runner 2000`：胜率 ≥0.54；记 Wilson95 下界 |
| D2 | vs 教师 McNemar **p ≥ 0.05** → 持平；不得改阈宣称超越 |
| D3 | 过牌率约 48%±8pp |

## 结果（回填 2026-09-11）

| 编号 | 结果 | 证据 |
|------|------|------|
| D0 | ✅ | md5 **`9513b89df43a3c1eece405faf340c7b3`**（≠ 旧文档 `3c0f`；点估计仍对齐 57%） |
| D1 | ✅ | **57.0%**（1140/2000），Wilson95 下界 **54.8%**；地主 59.7% / 农民 56.9% |
| D2 | ✅ 持平教师 | 模型 57.0% vs 教师 58.3%；McNemar χ²=2.315 **p=0.1281**（不显著） |
| D3 | ✅ | 过牌率 **47.7%** |

**结论**：现仓候选评分器冷评锁档成立；**逼近/持平教师成立，超越未成立**（与 2026-09-09 在线 RL 收官一致）。未再训。

日志：`runs/doudizhu_t6_p9_runner_2000_20260911.log`、`runs/doudizhu_t6_p9_paired_teacher_20260911.log`
