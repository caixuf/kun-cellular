# 请教：环 6 任务级 adoption 的 receptor require 悖论

## 状态：已解决（大佬 plan 方案 B，2026-09-08）

大佬定位（plan.md Findings #1）：**悖论是我方驱动代码的混乱**——
`u4_link6.cpp` 的诊断块检查硬编码 EdgeId 990（源=CellId 90），而
`req.target_edge` 赋的是另一个循环（源=CellId 0）选出的边。
"外部打印的谓词不是作用于请求的谓词"。底座无执行/存储 bug。

## 已实施的修复（方案 B：显式宽宿主绑定）

1. `AdoptionRequest` 追加 opt-in `AdoptionHostBinding{input_count, source_channel}`：
   模块输入槽 0 映射宿主 source_channel；不改 ModuleContract/工件/摘要/证据/schema。
2. 无 host_binding 时保留 RAW0-only 标量路径（现有测试不变）。
3. 有 host_binding 时：宿主 SENSE_CHANNEL（typed Param2 通道匹配）或 RAW0
   均可作为挂点源；Param1 精确 ==1.0（类型化检查，无 unchecked 解引用）。
4. preflight/commit 都接收完整宿主帧（56 维 scorer 帧，禁止单值帧）。
5. driver（tools/u4_link6.cpp）重写：删除人工 CellId 90/EdgeId 990；挂点 =
   真实 unit-gain SENSE_CHANNEL 通道 0 的 typed 契约边；宿主是新建年轻个体
   （感受器 gain 回到构建器出生设计值 1.0——bin 里的 2.1162 是父代历史训练
   live 痕迹，非本个体出生状态，不属于"改写训练增益"）；控制分支（同 tick
   同帧无借入）防 tick 伪装成知识增益；IO 隔离（库文件已存在则拒绝）；
   record_adoption 记录在案。
6. 无正 delta 断言：实测 Δ=+0.00pp 如实呈现（单通道直通提取器对完整
   82 细胞打分器的候选排序无可测影响——衍生宿主、非无关谱系、非盲测声明，
   均按 plan 的诚实标签执行）。

## 遗留契约问题（采纳 plan 边界，转为待办）

- 宽宿主借入的性能增益尚未成立（本模块语义太弱——单通道直通）。
  有任务意义的模块（多特征交互通路）需要 Task 6 的"任意子图 adoption"，
  按大佬边界明确不做，留作独立提案。
- "无关谱系借入"需独立出生模板的宿主实验，当前借入方为衍生宿主。

## 原始观察（存档）

`adopt_at_cold_boundary` 的 require（knowledge_adoption.hpp:82-84）外部同构
复算通过但调用仍抛 "motif boundary requires an unscaled scalar receptor"。
根因见上（Findings #1），非底座缺陷。
