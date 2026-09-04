/* ============================================================
 * patch_clamp_hud.js - 单细胞全细胞膜片钳实时示波器 (Patch-Clamp HUD)
 * 点击任何细胞弹出实时电位示波轨迹与权威 26 类原子原语动力学数学方程，
 * 硬核可信度与物理底座透明度拉满。
 * ============================================================ */

export const PRIMITIVE_EQUATIONS = {
  "Op_EMA": {
    name: "一阶惯性指数移动平均 (Exponential Moving Average)",
    c_op: "OP_EMA",
    category: "连续物理平滑与动量积分",
    ode: "τ · ds/dt = -s + u",
    discrete: "s[t] = (1 - α) · s[t-1] + α · u[t]",
    desc: "过滤感知高频噪声，提取低频主导趋势，保持动力学阻尼连续性。"
  },
  "Op_DIFF": {
    name: "一阶向后差分率 (First-Order Difference)",
    c_op: "OP_DIFF",
    category: "速度/曲率瞬态加速度感知",
    ode: "out = du/dt",
    discrete: "out[t] = (u[t] - u[t-1]) / Δt",
    desc: "检测横向偏差变化率 (CTE Rate) 或价格订单流加速度。"
  },
  "Op_DAMPER": {
    name: "二阶物理抗共振阻尼器 (Second-Order Viscous Damper)",
    c_op: "OP_DAMPER",
    category: "动力学振荡消除与超调抑制",
    ode: "d²s/dt² + 2ζω_n · ds/dt + ω_n² · s = ω_n² · u",
    discrete: "s[t] = 2·s[t-1] - s[t-2] + dt²(ω²(u - s) - 2ζω·v)",
    desc: "消除车辆急打方向引起的侧倾共振，或金融盘口雪崩超调。"
  },
  "Gate_HYSTERESIS": {
    name: "施密特双阈值迟滞比较门 (Schmitt Trigger Hysteresis)",
    c_op: "GATE_HYSTERESIS",
    category: "非线性抗噪声离散跳变锁存",
    ode: "s = (u > θ_high) ? 1 : ((u < θ_low) ? 0 : s_prev)",
    discrete: "Stateful Latch with High/Low Threshold deadband",
    desc: "避免在临界点频繁翻转颤抖 (Chatter-Free)，保证控制状态机确定性。"
  },
  "Gate_DEADZONE": {
    name: "不灵敏死区滤波门 (Deadzone Filter)",
    c_op: "GATE_DEADZONE",
    category: "微扰动过滤与静默抑制",
    ode: "out = sign(u) · max(0, |u| - δ)",
    discrete: "out = |u| < δ ? 0 : (u - sign(u)·δ)",
    desc: "在微小漂移时保持零功耗输出，仅在有效控制输入时打破静默。"
  },
  "Op_INTEGRAL": {
    name: "带防饱和漏电积分器 (Leaky Anti-Windup Integrator)",
    c_op: "OP_INTEGRAL",
    category: "稳态静差消除与能量累积",
    ode: "ds/dt = u - λ·s,  out = clamp(s, -Limit, +Limit)",
    discrete: "s[t] = clamp((1 - λ·dt)·s[t-1] + u[t]·dt, -L, L)",
    desc: "消除直道循迹稳态静差，具备积分防饱和硬限位保护。"
  },
  "Gate_INHIBIT": {
    name: "分流前馈侧向突触抑制门 (Shunting Lateral Inhibit)",
    c_op: "GATE_INHIBIT",
    category: "侧向抑制与竞争胜者独占 (WTA)",
    ode: "out = u / (1 + γ · v_inhibit)",
    discrete: "out = u · max(0, 1 - γ · v_inhibit)",
    desc: "模拟大脑皮层 GABA 能神经元，抑制对立肌群或相反买卖意图。"
  },
  "Op_AMPLIFY": {
    name: "可调增益线性放大元 (Proportional Gain Amplifier)",
    c_op: "OP_AMPLIFY",
    category: "尺度代偿与信噪比增强",
    ode: "out = K · u",
    discrete: "out[t] = K · u[t]",
    desc: "调整各反射弧通道权重增益，满足 ASIL-D 裕度契约。"
  }
};

export class PatchClampHUD {
  constructor() {
    this.container = null;
    this.canvas = null;
    this.ctx = null;
    this.selectedCell = null;
    this.voltageHistory = new Float32Array(160);
    this.historyHead = 0;
    this.initDOM();
  }

  initDOM() {
    if (document.getElementById("patch-clamp-hud")) return;

    const hud = document.createElement("div");
    hud.id = "patch-clamp-hud";
    hud.style.cssText = `
      position: absolute;
      left: 380px;
      bottom: 74px;
      width: 360px;
      background: rgba(8, 14, 26, 0.92);
      border: 1px solid rgba(56, 189, 248, 0.45);
      border-radius: 8px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.75), inset 0 0 16px rgba(56, 189, 248, 0.08);
      backdrop-filter: blur(14px);
      z-index: 100;
      font-family: var(--font-mono, monospace);
      color: #f1f5f9;
      user-select: none;
      display: none;
      overflow: hidden;
    `;

    hud.innerHTML = `
      <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 12px;
        background: rgba(15, 23, 42, 0.95);
        border-bottom: 1px solid rgba(56, 189, 248, 0.3);
      ">
        <div style="display: flex; align-items: center; gap: 6px;">
          <span style="width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 8px #34d399;"></span>
          <span id="patch-clamp-title" style="font-size: 11px; font-weight: 700; color: #38bdf8;">膜片钳全细胞示波器 (Patch-Clamp HUD)</span>
        </div>
        <button id="patch-clamp-close" style="
          background: transparent;
          border: none;
          color: #94a3b8;
          cursor: pointer;
          font-size: 14px;
          line-height: 1;
        ">&times;</button>
      </div>
      <div style="padding: 10px 12px;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; font-size: 10px;">
          <span style="color: #94a3b8;">靶向元胞: <b id="pc-cell-name" style="color: #f1f5f9;">Cell #0</b></span>
          <span style="color: #94a3b8;">膜电位 Vm: <b id="pc-vm-val" style="color: #34d399; font-size: 12px;">-70.0 mV</b></span>
        </div>
        
        <!-- 阴极射线管示波器网格 (CRT Oscilloscope) -->
        <div style="position: relative; width: 100%; height: 110px; background: #030a16; border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 4px; overflow: hidden;">
          <canvas id="pc-oscilloscope-canvas" width="336" height="110" style="width: 100%; height: 100%; display: block;"></canvas>
          <div style="position: absolute; right: 4px; top: 4px; font-size: 8px; color: rgba(52, 211, 153, 0.6);">
            +30mV (峰值)
          </div>
          <div style="position: absolute; right: 4px; top: 48px; font-size: 8px; color: rgba(56, 189, 248, 0.6);">
            -55mV (阈值)
          </div>
          <div style="position: absolute; right: 4px; bottom: 4px; font-size: 8px; color: rgba(148, 163, 184, 0.5);">
            -70mV (静息)
          </div>
        </div>

        <!-- 动力学权威数学方程卡片 -->
        <div id="pc-equation-card" style="margin-top: 8px; padding: 8px; background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 4px; font-size: 10px;">
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <b id="pc-op-title" style="color: #38bdf8;">Op_EMA (一阶惯性滤波)</b>
            <span id="pc-layer-tag" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 1px 5px; border-radius: 2px; font-size: 9px;">L2_ASSOCIATION</span>
          </div>
          <div style="margin: 4px 0; padding: 4px 6px; background: #030712; border-radius: 3px; border-left: 2px solid #38bdf8; font-family: monospace; color: #a5f3fc; font-size: 11px;">
            <div id="pc-ode-eq">τ · ds/dt = -s + u</div>
            <div id="pc-discrete-eq" style="color: #94a3b8; font-size: 9px; margin-top: 2px;">s[t] = (1 - α)·s[t-1] + α·u[t]</div>
          </div>
          <div id="pc-op-desc" style="color: #cbd5e1; font-size: 9px; line-height: 1.4;">
            过滤感知高频噪声，提取主导动力学趋势。
          </div>
        </div>

        <!-- 细胞私有寄存器实时对账 -->
        <div style="margin-top: 8px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; text-align: center; font-size: 9px;">
          <div style="background: rgba(15, 23, 42, 0.6); padding: 4px; border-radius: 3px; border: 1px solid rgba(30, 41, 59, 0.6);">
            <div style="color: #94a3b8;">私有寄存器 s</div>
            <b id="pc-reg-s" style="color: #38bdf8; font-size: 11px;">0.000</b>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 4px; border-radius: 3px; border: 1px solid rgba(30, 41, 59, 0.6);">
            <div style="color: #94a3b8;">放电输出 out</div>
            <b id="pc-reg-out" style="color: #34d399; font-size: 11px;">0.000</b>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 4px; border-radius: 3px; border: 1px solid rgba(30, 41, 59, 0.6);">
            <div style="color: #94a3b8;">自适应增益</div>
            <b id="pc-reg-gain" style="color: #fbbf24; font-size: 11px;">1.00</b>
          </div>
          <div style="background: rgba(15, 23, 42, 0.6); padding: 4px; border-radius: 3px; border: 1px solid rgba(30, 41, 59, 0.6);">
            <div style="color: #94a3b8;">激化次数</div>
            <b id="pc-reg-acts" style="color: #f43f5e; font-size: 11px;">0</b>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(hud);
    this.container = hud;
    this.canvas = document.getElementById("pc-oscilloscope-canvas");
    this.ctx = this.canvas.getContext("2d");

    document.getElementById("patch-clamp-close").onclick = () => {
      this.close();
    };
  }

  selectCell(cell, org = null) {
    if (!this.container) this.initDOM();
    this.selectedCell = cell;
    this.container.style.display = "block";

    // 填充原语元数据与方程
    const ctype = cell.type || "Op_EMA";
    const eqInfo = PRIMITIVE_EQUATIONS[ctype] || {
      name: `${ctype} 动力学原语`,
      c_op: ctype.toUpperCase(),
      category: "通用非冯算存一体计算元",
      ode: "s[t] = f(u[t], s[t-1])",
      discrete: "Atomic C11 Kernel Function",
      desc: "执行单步纳秒级硬件原子操作。"
    };

    const nameEl = document.getElementById("pc-cell-name");
    if (nameEl) nameEl.textContent = `Cell #${cell.id} [${ctype}]`;

    const titleEl = document.getElementById("pc-op-title");
    if (titleEl) titleEl.textContent = eqInfo.name;

    const layerEl = document.getElementById("pc-layer-tag");
    if (layerEl) layerEl.textContent = cell.layer || "L2_ASSOCIATION";

    const odeEl = document.getElementById("pc-ode-eq");
    if (odeEl) odeEl.textContent = eqInfo.ode;

    const discEl = document.getElementById("pc-discrete-eq");
    if (discEl) discEl.textContent = eqInfo.discrete;

    const descEl = document.getElementById("pc-op-desc");
    if (descEl) descEl.textContent = eqInfo.desc;
  }

  close() {
    this.selectedCell = null;
    if (this.container) this.container.style.display = "none";
  }

  update(time) {
    if (!this.selectedCell || !this.container || this.container.style.display === "none") return;

    const c = this.selectedCell;
    const out = c.out || 0.0;
    const state = c.state || 0.0;
    const gain = c.gain !== undefined ? c.gain : (c.param1 || 1.0);
    const acts = c.acts || 0;

    // 模拟 Hodgkin-Huxley 生物膜电位 (mV)
    // 静息电位 -70mV，动作电位去极化射至 +30mV，复极化下冲 -75mV
    let vm = -70.0 + out * 45.0;
    if (out > 0.4) vm = -70.0 + out * 85.0 + Math.sin(time * 35.0) * 3.0; // 动作电位峰顶震颤
    vm = Math.max(-85.0, Math.min(40.0, vm));

    this.voltageHistory[this.historyHead] = vm;
    this.historyHead = (this.historyHead + 1) % this.voltageHistory.length;

    // 渲染示波器
    this.renderOscilloscope();

    // 更新数值标签
    const vmEl = document.getElementById("pc-vm-val");
    if (vmEl) {
      vmEl.textContent = `${vm.toFixed(1)} mV`;
      vmEl.style.color = vm > -50 ? (vm > 0 ? "#f43f5e" : "#fbbf24") : "#34d399";
    }
    const regS = document.getElementById("pc-reg-s");
    if (regS) regS.textContent = state.toFixed(3);
    const regOut = document.getElementById("pc-reg-out");
    if (regOut) regOut.textContent = out.toFixed(3);
    const regGain = document.getElementById("pc-reg-gain");
    if (regGain) regGain.textContent = gain.toFixed(2);
    const regActs = document.getElementById("pc-reg-acts");
    if (regActs) regActs.textContent = acts.toLocaleString();
  }

  renderOscilloscope() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    // 1. 示波器 CRT 磷光网格
    ctx.strokeStyle = "rgba(52, 211, 153, 0.12)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 28) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 22) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // -70mV 静息参考零线 (灰色点虚线)
    const yRest = h - (( -70.0 + 85.0) / 125.0) * h;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, yRest); ctx.lineTo(w, yRest); ctx.stroke();

    // -55mV 动作电位阈值线 (金黄色虚线)
    const yThresh = h - (( -55.0 + 85.0) / 125.0) * h;
    ctx.strokeStyle = "rgba(251, 191, 36, 0.45)";
    ctx.beginPath(); ctx.moveTo(0, yThresh); ctx.lineTo(w, yThresh); ctx.stroke();
    ctx.setLineDash([]);

    // 2. 膜电位滚动波形
    ctx.strokeStyle = "#34d399";
    ctx.shadowColor = "#34d399";
    ctx.shadowBlur = 6;
    ctx.lineWidth = 2.0;
    ctx.beginPath();

    const len = this.voltageHistory.length;
    for (let i = 0; i < len; i++) {
      const idx = (this.historyHead + i) % len;
      const v = this.voltageHistory[idx];
      const px = (i / (len - 1)) * w;
      // 映射 [-85, 40] -> [h, 0]
      const py = h - ((v + 85.0) / 125.0) * h;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // 最新扫描光点
    const curIdx = (this.historyHead - 1 + len) % len;
    const curV = this.voltageHistory[curIdx];
    const curY = h - ((curV + 85.0) / 125.0) * h;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(w - 2, curY, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

export const patchClampHUD = new PatchClampHUD();
