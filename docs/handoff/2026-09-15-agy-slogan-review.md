# 口号路线审稿短评 (2026-09-15)

> 审稿人角色：口号路线审稿人，不是再开一条战役。  
> 审阅对象：Cursor 刚落地的三条未 commit 改动（描述带构造协议、主定理接线承重卡、DomainZoo 评测隔离与卫生）。

---

## 1. 这三条分别证到了什么、没证到什么

### ① 冯·诺依曼描述带构造（`von_neumann_constructor.hpp`）
- **证到了什么**：证到了纯净基因型（细胞拓扑与突触权重）序列化为字节带后，可通过构造器协议装配出首拍控制行为与亲本一致的新有机体；消融构造器开关或清空磁带均严格拒构；实测重构女儿对象无亲本膜孔道与电位残留（`output_val == 0`）。
- **没证到什么**：没证到 28 原语实现的图内通用构造器（自复制依赖 C++ 宿主解析器驱动装配）；没证到图内自主读带与带复制闭环；没证到跨代际遗传变异的自维持自繁殖系统。

### ② 主定理卡：接线承重，有丝分裂不承重（`main-theorem-topology.md`）
- **证到了什么**：证到了在现有契约宽度（ADAS 控制与倒立摆任务）上，数据流图的控制能力本质依赖突触拓扑与重连加边（ADAS 同度数重连退化 34.1×，CartPole 关重连 ID SR 跌至 0.43）；证到了在微尺度任务上有丝分裂（0.97）与鲍德温固化（1.00）对控制性能不构成主要承重。
- **没证到什么**：没证到全套形态算子的完备消融（硬编码 5% 凋亡关不掉，§5.9 四个启发式默认仍未进入 `evolve_generation`）；没证到该定型结论在宽输入生境或高阶时空图中的通用性。

### ③ DomainZoo 评测隔离与环境卫生（`domain_zoo.hpp` + `test_flow_domain_zoo_eval_isolation.cpp`）
- **证到了什么**：证到了 `ZooTask::reset` 清零 `o_[]` 能够拔除跨回合、跨种子的首帧脏观测污染；证到了消除评测污染后，cartpole ID 成功率在 leaky、clone、fresh 三种评测路径下均咬死在真实的 0.1；锁定了隔离状态下 12 域动力学控制的真实基线为 10/12。
- **没证到什么**：没证到底座 `reset_state` 能够清理膜孔道脏态（`membrane_pores` 未清导致复用个体在 cartpole OOD 上仍有 leaky 0.3 vs clone 0.1 的残留漂移）；没证到历史冻结的 12 域模型具有全域隔离泛化能力（12/12 不可复现）。

---

## 2. 冒领审查：红线合规判定

**审查结论：全项守住红线，未发生冒领。**

- **描述带表述审查**：代码注释、测试用例与交付文档均明文声明「这不是 28 原语实现的通用构造器，也不是 C++ 拷贝构造，不是 autonomous_replicator 的能量阈值分裂」。测试专门构造了 C++ 拷贝保留运行态而磁带构造置零运行态的对照断言，定性严守在「描述带+构造器开关的最小协议」。
- **DomainZoo 12/12 审查**：未篡改历史冻结的 `domain_zoo_report.json`，未覆盖现存 `zoo_*.bin`；文档与门禁矩阵明确标示「冻结 12/12 是泄漏评测；评测卫生后隔离 10/12」；测试脚本末尾设置了硬断言 `assert(clone_m1 < 12 && "isolated M1 must not be claimed 12/12")`，杜绝宣传回填。
- **口径禁区核对**：交付文档明确撤回非冯机器、车规、量化夏普 0.22 宣称；明确声明迷宫 96/100 依赖测地脚手架而非自主长出地图；明确声明 §5.9 四算子未消融。符合口径铁律。

---

## 3. 审稿人想法分类（严控每类最多 3 条）

### 【采纳】便宜、可证伪、本回合就能做
1. **在 `test_flow_domain_zoo_eval_isolation.cpp` 尾部补齐 ballbeam OOD 0.4 的法定断言**：当前末尾只显式断言了 cartpole ID 0.1 和总数 10/12；既然审计认定挂科域为 cartpole ID 与 ballbeam OOD，补上一行 `assert(ballbeam_clone_ood == 0.4)` 可彻底固化 10/12 负例证据链。
2. **为 `von_neumann_constructor.hpp` 补写截断/脏魔数磁带拒绝的负向用例**：在测试中追加 3 行代码，验证篡改 magic 或截断 header/payload 时返回 `bad_tape_header` 或 `truncated_tape`，完善协议自防伪闭环。
3. **将两项新 flow 测试并入本地串行自动化回归集合**：确保 CMakeLists.txt 自动发现的 `test_flow_von_neumann_constructor` 与 `test_flow_domain_zoo_eval_isolation` 纳入日常 pre-commit 快速检查。

### 【推迟】有牙齿但要改 evolve_generation / 膜孔道 / FPGA
1. **底座 `reset_state()` 彻底清理 `membrane_pores`**：能彻底消灭 cartpole OOD leaky 0.3 vs clone 0.1 残余差异，但改动直击 `include/kun/cellular/` 底座热路径，需要安排全域回归，暂不宜动。
2. **将 `mutate()` 中硬编码的 5% 凋亡率解耦为演化可控参数**：有助于把形态发生消融推进至「五算子完备」，但触碰核心演化热路径，推迟至后续战役。
3. **基于 28 原语实现图内指令解码与微拓扑生长的通用构造器**：通往真正图内自复制的唯一正道，但依赖图内状态机体系与执行流重构，工程代价巨大。

### 【拒绝】包装、堆生境 6/6、重训 Zoo 填回 12/12、观测台炫技
1. **重新调优重训 DomainZoo 挂科域强行凑回 12/12**：拒绝粉饰历史检查点。10/12 是评测隔离后的客观真实现状，强训只是用过拟合掩盖模型泛化瓶颈。
2. **将数据流图序列化包装为「冯·诺依曼通用自复制机/非冯生物计算机」**：拒绝概念包装。当前实现本质是「可演化数据流图控制器的基因型外置最小装配协议」，过度阐释即是学术欺诈。
3. **开发三维细胞图谱分裂回放观测大屏**：拒绝一切不产生可证伪判据的前端可视化炫技。

---

## 4. 下一刀唯一推荐

**推荐项目：锁死 DomainZoo 隔离挂科双锚点（ballbeam OOD 0.4 与 cartpole ID 0.1）的硬化门禁**

- **对照实验**：使用未复用干净环境（`eval_clone`），对比已存固定 bin 在泄漏历史报告（JSON: ballbeam OOD 0.7, cartpole ID 0.9）与环境隔离卫生后（leaky=clone: ballbeam OOD 0.4, cartpole ID 0.1）的表现差异。
- **成功判据**：
  - `test_flow_domain_zoo_eval_isolation.cpp` 中单点硬断言 `cartpole_clone_id == 0.1` 与 `ballbeam_clone_ood == 0.4` 同时成立；
  - 12 域综合合格数断言严格满足 `clone_m1 == 10 && leaky_m1 == 10`；
  - 零内存泄漏，执行耗时 < 100ms。
- **失败判据**：出现 `clone_m1 > 10`（说明隔离失效或污染未清），或 `ballbeam_clone_ood > 0.4`（评测边界漂移）。
- **严格不许改动**：
  - 不许修改 `include/kun/cellular/` 下任何热路径文件（Cell / mutate / compile / reset_state）；
  - 不许重写或覆盖 `checkpoints/domain_zoo_report.json` 与任何 `checkpoints/zoo_*.bin`；
  - 不许在未隔离的共享环境下提取法定结论。

---

## Cursor 筛后（2026-09-15）

- **收下**：isolation 补 `ballbeam_clone_ood==0.4`；构造器补脏 magic / 截断磁带负例。
- **不收**：<100ms 时限（隔离测试实测约 3s，不是门禁）；另做 pre-commit 发现层（CMake `GLOB tests/test_*.cpp` 已自动挂上）。
- **仍推迟**：`reset_state` 清膜孔道、凋亡解耦、28 原语图内构造器。
