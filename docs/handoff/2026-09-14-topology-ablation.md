# ADAS 拓扑先验消融（稳定性战役 #4，2026-09-14）

> 预注册：`tools/ablate_topology_prior.py`  
> 产物：`runs/adas_topology_ablation_20260914.json`  
> CI：`tests/test_adas_topology_ablation.py`

## 判据

锁档能力若来自拓扑（而非偶然权重共振），则：
- **A** 同度数随机重连（Maslov–Sneppen，保持入/出度，禁止指向感受器）cost 比 ≥ 1.3 或完赛率掉到 0
- **B** 全突触 E/I 符号翻转同样显著变差

## 结果（noise_seed=21，训练 12 + 验证 4 场景）

| bin | lock cost | rewire mean (8) | rewire ok | A 比 | E/I cost | B 比 | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| `adas_cortex_champion.bin` | 34.89 | 1191.4 | 0/8 | **34.1×** | 375.6 | **10.8×** | PASS |
| `adas_cortex_champion_stadium.bin` | 52.02 | 1245.1 | 0/8 | **23.9×** | 1143.0 | **22.0×** | PASS |

**overall = PASS。** 重连后全部未能跑满场景；符号翻转同样完赛失败。这支持「拓扑结构本身携带控制能力」，而非同度数随机图碰巧能开。

## 复现

```bash
python3 tools/ablate_topology_prior.py --rewires 8
python3 tests/test_adas_topology_ablation.py
```

## 边界（诚实）

- 只覆盖 ADAS 皮层器官；迷宫/斗地主未做同协议。
- E/I 翻转是全局取负，不是按细胞类型翻转神经递质。
- 不改底座；消融发生在任务层 Python 器官副本。
