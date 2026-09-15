# 冯·诺依曼描述带构造 — 最小协议 (2026-09-15)

> 落地：`include/kun/cellular/von_neumann_constructor.hpp`  
> 测试：`tests/test_flow_von_neumann_constructor.cpp`

## 证的是什么

亲本写出**基因型磁带**，构造器打开时从磁带组装女儿图；切断构造器或空磁带必须失败。女儿不含亲本运行时态。同观测下第一拍动作与亲本一致。

磁带只编码细胞类型/参数/坐标与突触拓扑权重；不含膜孔道、`output_val`、赫布率。实测 `tape_bytes=1656` 对 13 细胞冠军图。

## 明确不证什么

- 不是 28 原语实现的通用构造器
- 不是 C++ `CellularOrganism` 拷贝构造（测试里宿主拷贝会带走 `output_val`，磁带构造不会）
- 不是 `autonomous_replicator.hpp` 那套能量阈值分裂（宿主侧 `ReplicableGenome` 拷贝）

## 实测

```
VN_CTOR parent_cells=13 tape_bytes=1656 child_match=1 ablate=1 empty=1 magic=1 trunc=1
```
