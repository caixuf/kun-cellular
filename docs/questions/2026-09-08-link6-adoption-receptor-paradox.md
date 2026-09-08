# 请教：环 6 任务级 adoption 的 receptor require 悖论

## 现象

`adopt_at_cold_boundary` 的 require（knowledge_adoption.hpp:82-84）:

```cpp
require(target_plan.cells()[link->source_index].type == CellType::SENSE_RAW_INPUT_0 &&
        std::get<ContinuousValue>(*target.runtime().parameter(
            target_plan.cells()[link->source_index].id, ParameterSlot::Param1)).value == 1.0,
        "motif boundary requires an unscaled scalar receptor");
```

外部复算（同构表达式、同 runtime、同边 990）全部通过：

```
[复算] edge990 src_index=56 id=90 type=0 (RAW枚举=0) param1=1.000000
```

但调用仍抛 "motif boundary requires an unscaled scalar receptor"。

## 已排除

- type: 细胞 90 确为 SENSE_RAW_INPUT_0（枚举 0）
- gain: parameter(cell90, Param1) = 1.000000（种子显式设 1.0）
- coupling: 边 990 权重 0.3 ≠ 0
- delay: Immediate
- link 查找: req.target_edge = 990（hook 搜索限定 s==90，唯一边）
- expected_tick: 构造时同步读取

## 疑点

`RuntimeState::parameter(CellId, ParameterSlot)` 是否在 adoption 上下文中
读取与外部 `runtime().parameter()` 不同的参数视图（如编译期 binding 快照
vs runtime live 值）？或 `target` 引用在 require 时序上存在别名/重绑定？

## 临时处置

环 6 机制级闭合成立（tests/test_the_chain.cpp 环 6 全绿——StrictCore
微世界借阅），任务级借阅（tools/u4_link6.cpp）挂起于此问题，诊断代码留档。

## 附带契约问题（请示）

冠军图（业务宽契约）的感受器全部是 SENSE_CHANNEL，而 adoption 的 motif
源契约写死 SENSE_RAW_INPUT_0。是否应扩展 adoption 的 motif 源契约支持
SENSE_CHANNEL（gain 语义对齐），否则所有宽契约业务图都必须附加人工 RAW
受体才能参与知识借阅。
