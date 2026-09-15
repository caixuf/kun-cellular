# 对外口径残余不一致扫描清单 (2026-09-15)

> **宪条纪律**：只读扫描 + 产出 handoff；不修改 `include/kun/cellular/` 底座；不长训。  
> **法定定性基准**：
> 1. **L3 锁档** `checkpoints/adas_cortex_champion.bin`：210 细胞 / 659 突触，16 场景多种子（seeds 1-10）对 Stanley **9W / 7L**（净胜率 56.3%），直道 CTE 3.12 cm，高速 CTE 17.15 cm；
> 2. **观测台默认** `checkpoints/adas_cortex_champion_stadium.bin`：体育场公路适配学生档，叙事 **10W / 6L**（叙事可读分 62.5），p95 CTE **0.145 m**（不覆盖 L3 锁档）；
> 3. **历史 7W / 9L**：仅作为 2026-09-04 审计快照（对应旧 630 突触未调参版），严禁作为当前生产/对外宣传基准。

---

## 一、 残余不一致汇总表

| 序号 | 所属模块 / 文件 | 行号 | 现状定性 | 违例/失准旧句 | 拟对齐方案 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `frontend/vehicle.html` | 259 | 混淆沙盒与 L3 | `<!-- 1. SDSCC 210-细胞 ADAS L3 锁档：AdasCortexOrgan 真前向闭环 -->` | 改为明确标明为「体育场学生档（默认）」与「L3 锁档」分立 |
| 2 | `frontend/vehicle.html` | 262 | 静态默认标题混淆 | `<span id="cortex-panel-title">[SDSCC_ADAS_L3_210]</span>` | 静态初始占位改为 `[SDSCC_ADAS_STADIUM_210]`，与后端默认载入保持一致 |
| 3 | `frontend/vehicle.html` | 263 | 静态徽章口径混淆 | `<span style="color:var(--emerald-glow);" id="stat-gen-badge">L3 · 真闭环 · 9W/7L</span>` | 静态初始占位改为 `stadium适配 · vs Stanley 10W/6L · 观测台学生默认` |
| 4 | `frontend/vehicle.html` | 273 | 静态胜率混淆 | `<div class="stat-val amber" id="stat-fit">56.3</div>` | 静态默认初始值改为 `62.5`（对应 10W/6L 叙事分，避免初次渲染闪烁 56.3%） |
| 5 | `frontend/vehicle.html` | 383 | 释义卡片混淆 | `车辆控制由离散动力学原语构建的 DAG 拓扑网络接管（L3 锁档 210 细胞）。...` | 明确标注「当前沙盒默认挂载体育场学生档（adas_cortex_champion_stadium.bin，10W/6L）；基准评测 L3 锁档（9W/7L）为独立检查点」 |
| 6 | `frontend/vehicle.html` | 563-564 | 动态回退兜底丢失 | `} else if (nCells === 210) { titleEl.innerText = "[SDSCC_ADAS_L3_210]"; }` | 当未带 stadium 关键字时，若为体育场默认应区分标记，不宜武断判定为 L3 |
| 7 | `frontend/index.html` | 213 | 导航卡混淆沙盒与 L3 | `搭载 210-细胞微柱皮层冠军模型（L3 增益重调·16 场景对 Stanley 9W/7L）。基于阿克曼微分几何运动学 ODE...` | 标明「沙盒演示默认加载体育场学生档（10W/6L，p95 CTE 0.145m）；L3 锁档（16工况 9W/7L）为留出基准档」 |
| 8 | `frontend/index.html` | 217 | 入口性能指标混淆 | `<div>* 实证性能: 直道巡航 CTE 3.12cm（Stanley 2.86cm），16 工况 10-seed 平均 9 胜 7 负</div>` | 补充说明此为 L3 锁档实证基准，观测台现场为体育场环形自适应巡线 |
| 9 | `frontend/cellular.html` | 1579 | 突触数遗留快照 | `<b>神经规模:</b> <span id="active-org-scale"...>210 细胞 / 630 突触 (宏观 210 规模)</span>` | 静态占位 630 突触为历史快照，L3 锁档实际为 659 突触，建议修正 |
| 10 | `README.md` | 137-140 | 目录树丢失 L3 锁档 | `checkpoints/` 下仅列出 `adas_cortex_champion_stadium.bin`，遗漏 `adas_cortex_champion.bin` | 在目录树中补齐两把锁档的分立列出与职责注释 |
| 11 | `docs/morphogenetic_cellular_evolution_paper.md` | 18 | 摘要缺沙盒边界说明 | 摘要条目 2 仅提及 `checkpoints/adas_cortex_champion.bin` 与 9W/7L，未声明观测台默认体育场学生档 | 在 Update 补充说明：Web 交互观测台默认搭载针对环形赛道平滑适配的 student 档（10W/6L），与 16 工况离线基准 L3 锁档（9W/7L）正交解耦 |
| 12 | `docs/morphogenetic_cellular_evolution_paper.zh.md` | 18 | 中文摘要缺沙盒边界 | 中文摘要条目 2 同样未提及观测台默认体育场档与 L3 锁档的分工边界 | 同步补充边界声明，杜绝读者进入前端后因看到 10W/6L 与论文 9W/7L 产生对账困惑 |
| 13 | `dist/kun-cellular-v1.3.1-linux-x86_64/README.md` | 108 | 发行包 README 遗留旧口径 | `- **自动驾驶仿真 (frontend/vehicle.html)**：搭载 210-细胞微柱皮层冠军模型（C11 / Python 真实前向推理）。` | 历史发行包中未同步更新观测台默认 student 与 L3 锁档的分流说明 |

---

## 二、 逐项详细诊断与证据链

### 1. `frontend/vehicle.html`：观测台前端代码与静态 DOM 严重混淆

#### [Line 259 & 261-264] 静态面板标题与徽章
- **文件与行号**：`frontend/vehicle.html:259-264`
- **旧代码**：
  ```html
  <!-- 1. SDSCC 210-细胞 ADAS L3 锁档：AdasCortexOrgan 真前向闭环 -->
  <div class="viewport-card">
      <div class="panel-header">
          <span id="cortex-panel-title">[SDSCC_ADAS_L3_210]</span>
          <span style="color:var(--emerald-glow);" id="stat-gen-badge">L3 · 真闭环 · 9W/7L</span>
      </div>
  ```
- **冲突根因**：`vehicle.html` 运行的环境是 `[STADIUM_CIRCUIT]`（体育场公路），后端 `tools/cellular_live_backend.py:475-476` 默认优选加载 `adas_cortex_champion_stadium.bin`。但 HTML 静态结构全写死了 `L3` 与 `9W/7L`。虽然 JS fetch 后会用 `champion_claim` 覆写，但未连接后端或弱网首帧瞬间暴露出 L3 宣称。

#### [Line 272-274] 静态胜率数值
- **文件与行号**：`frontend/vehicle.html:272-274`
- **旧代码**：
  ```html
  <div class="stat-label">VS_STANLEY_WIN%</div>
  <div class="stat-val amber" id="stat-fit">56.3</div>
  ```
- **冲突根因**：`56.3` 是 L3 锁档在 16 场景多种子下的胜率（9/16 = 56.25%）。体育场学生档的叙事可读分是 `62.5`（10W/6L，见 `tools/cellular_live_backend.py:531`）。

#### [Line 383] 释义文本
- **文件与行号**：`frontend/vehicle.html:383`
- **旧代码**：
  ```html
  车辆控制由离散动力学原语构建的 DAG 拓扑网络接管（L3 锁档 210 细胞）。受体层感知 CTE/航向/曲率，隐层为演化定型拓扑，效应层输出转向与加速。
  ```
- **冲突根因**：把当前页面运行的模型直接解释为「L3 锁档 210 细胞」，完全抹杀了体育场适配学生档的存在。

---

### 2. `frontend/index.html`：平台门户直接把车辆沙盒宣称为 L3 锁档

#### [Line 211-221]
- **文件与行号**：`frontend/index.html:211-221`
- **旧代码**：
  ```html
  <div class="card-title">真实阿克曼智能驾驶车辆控制</div>
  <div class="card-desc">
    搭载 210-细胞微柱皮层冠军模型（L3 增益重调·16 场景对 Stanley 9W/7L）。基于阿克曼微分几何运动学 ODE，解算横向偏差（CTE）与道路曲率进行自适应巡线（部分直道/弯道场景领先，部分落后）。
  </div>
  <div class="card-meta">
    <div>* 运动学核: 轴距差分阿克曼双轮微分模型</div>
    <div>* 实证性能: 直道巡航 CTE 3.12cm（Stanley 2.86cm），16 工况 10-seed 平均 9 胜 7 负</div>
    <div>* 独立展示: 真实沥青双向环形赛道 · 实时遥测仪表盘与动力学追踪</div>
  </div>
  <a href="vehicle.html" class="btn btn-secondary" ...>进入车辆控制沙盒 (单独展示)</a>
  ```
- **冲突根因**：用户在 `index.html` 看到的是「L3 增益重调 · 16 场景 9 胜 7 负 · 直道 CTE 3.12cm」，点击按钮跳转到 `vehicle.html` 后，跑的却是体育场环形赛道（实际运行 `adas_cortex_champion_stadium.bin`，10W/6L，p95 CTE 0.145m）。这是典型的将实验基准档与演示沙盒学生档混为一谈。

---

### 3. `README.md`：目录结构树单边遗漏

#### [Line 137-144]
- **文件与行号**：`README.md:137-144`
- **旧代码**：
  ```markdown
  ├── checkpoints/                   # 真实演化产物（SDSC-BIN v2 二进制检查点）
  │   ├── domain_zoo_report.json     # 12 任务控制动物园报告
  │   └── adas_cortex_champion_stadium.bin  # 观测台默认学生档（非 L3）
  ```
- **冲突根因**：虽然 `README.md:56-57` 与 `README.md:68` 已经非常清晰地对齐了两把锁档的口径，但在第五节「仓库架构」的目录树中，`checkpoints/` 下只写了 `adas_cortex_champion_stadium.bin`，把真正的基准核心锁档 `adas_cortex_champion.bin` 漏掉了，导致查阅目录树的读者会误以为仓库里只有体育场档。

---

### 4. 论文摘要 (`docs/morphogenetic_cellular_evolution_paper*.md`)

#### [English: Line 18 / Chinese: Line 18]
- **文件与行号**：
  - `docs/morphogenetic_cellular_evolution_paper.md:18`
  - `docs/morphogenetic_cellular_evolution_paper.zh.md:18`
- **旧文本摘录**：
  - EN: `2. Lane-keeping cortex, shipped champion checkpoints/adas_cortex_champion.bin (210 cells, 630 synapses) [E1]: ... overall 16-scenario tally is 7 wins / 9 losses (Table 2b); ... Update (2026-09-11): after L3 sep-CMA-ES retuning the shipped champion is 210 cells / 659 synapses with a 16-scenario 9W / 7L tally (runs/adas_champion_vs_stanley_seeds1-10_tuned_20260911.json); the tables in this section keep the 2026-09-04 audit snapshot (630 synapses / 7W-9L), and the README carries the current figures.`
  - ZH: `2. 车道保持皮层，随仓冠军 checkpoints/adas_cortex_champion.bin（210 细胞，630 突触）[E1]：... 16 场景总战绩为 7 胜 9 负（表 2b）... 口径更新（2026-09-11）：随仓冠军经 L3 sep-CMA-ES 重调后为 210 细胞 / 659 突触、16 场景 9 胜 7 负... 本节表格保留 2026-09-04 审计快照（630 突触 / 7 胜 9 负），当前口径以 README 为准。`
- **残余脱节**：
  1. 结构化摘要标题处的括号仍写着 `(210 cells, 630 synapses)`，虽然文末写了 Update 为 659 突触 / 9W7L，但前置标题仍沿袭旧快照；
  2. 论文完全未对随仓的前端演示沙盒做边界切分说明。读者执行 `bash tools/run_observatory.sh` 打开浏览器看到的 `vehicle.html` 默认呈现 `10W/6L` 与体育场环道（`adas_cortex_champion_stadium.bin`），若论文摘要不作声明，读者会误以为 10W/6L 是对 9W/7L 的篡改或不一致。

---

## 三、 遗留修补执行清单 (Checklist for Next Commit)

- [x] `frontend/vehicle.html` 静态标题/徽章/info-box（首帧不再写死 9W/7L）
- [x] `frontend/index.html` 入口卡拆开沙盒与 L3
- [x] `README.md` 目录树同时列出两把锁档
- [ ] 论文摘要补一句观测台边界（未改，避免大段论文漂移）
- [ ] `dist/` 发行包 README 属历史包，不回改

> **2026-09-15**：上两项列入结案 WON'T。战役冻结，见 `2026-09-15-cellular-closeout.md`。
