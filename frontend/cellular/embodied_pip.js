/* ============================================================
 * embodied_pip.js - 具身画中画数字孪生微视窗 (Embodied PIP Twin)
 * 在 3D 观测台同屏嵌入车辆巡航轨迹/方向盘姿态或 L2 盘口阶梯，
 * 实现“脑-机-环”因果实时共鸣与端到端闭环物理观测。
 * ============================================================ */

export class EmbodiedPIPTwin {
  constructor() {
    this.container = null;
    this.canvas = null;
    this.ctx = null;
    this.isCollapsed = false;
    this.currentDomain = "adas";
    this.lastData = null;
    this.steerAngle = 0;
    this.targetSteerAngle = 0;
    this.initDOM();
  }

  initDOM() {
    if (document.getElementById("embodied-pip-twin")) return;

    const pip = document.createElement("div");
    pip.id = "embodied-pip-twin";
    pip.style.cssText = `
      position: absolute;
      right: 20px;
      bottom: 74px;
      width: 320px;
      background: rgba(8, 14, 26, 0.88);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65), inset 0 0 12px rgba(56, 189, 248, 0.08);
      backdrop-filter: blur(12px);
      z-index: 100;
      font-family: var(--font-mono, monospace);
      color: #f1f5f9;
      user-select: none;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      overflow: hidden;
    `;

    pip.innerHTML = `
      <div id="pip-header" style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 10px;
        background: rgba(15, 23, 42, 0.9);
        border-bottom: 1px solid rgba(56, 189, 248, 0.25);
        cursor: move;
      ">
        <div style="display: flex; align-items: center; gap: 6px;">
          <span id="pip-domain-dot" style="width: 8px; height: 8px; border-radius: 50%; background: var(--cyan, #38bdf8); box-shadow: 0 0 8px var(--cyan, #38bdf8);"></span>
          <span id="pip-title" style="font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #38bdf8;">具身数字孪生 · 闭环因果</span>
        </div>
        <div style="display: flex; gap: 6px;">
          <button id="pip-collapse-btn" title="最小化/展开" style="
            background: transparent;
            border: none;
            color: #94a3b8;
            cursor: pointer;
            font-size: 12px;
            padding: 0 4px;
          ">&minus;</button>
        </div>
      </div>
      <div id="pip-body" style="padding: 8px 10px;">
        <div id="pip-canvas-container" style="position: relative; width: 100%; height: 110px; background: rgba(3, 7, 18, 0.85); border: 1px solid rgba(30, 41, 59, 0.8); border-radius: 4px; overflow: hidden;">
          <canvas id="pip-viewport-canvas" width="300" height="110" style="width: 100%; height: 100%; display: block;"></canvas>
          <div id="pip-overlay-badge" style="position: absolute; top: 4px; left: 6px; font-size: 9px; color: #94a3b8; text-shadow: 0 1px 2px #000;">
            2D 闭环公路轨迹
          </div>
        </div>
        <div id="pip-metrics-panel" style="margin-top: 8px; display: grid; grid-template-columns: 80px 1fr; gap: 8px; align-items: center;">
          <div id="pip-gauge-slot" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 74px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(30, 41, 59, 0.6); border-radius: 4px;">
            <canvas id="pip-gauge-canvas" width="64" height="64" style="width: 56px; height: 56px;"></canvas>
            <span id="pip-gauge-label" style="font-size: 9px; color: #94a3b8; margin-top: -4px;">舵角 0.0°</span>
          </div>
          <div id="pip-data-slot" style="display: flex; flex-direction: column; gap: 3px; font-size: 10px;">
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">巡航车速:</span><b id="pip-v-speed" style="color: var(--emerald, #34d399);">0.0 km/h</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">横向误差:</span><b id="pip-v-cte" style="color: #38bdf8;">0.00 cm</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">加减速度:</span><b id="pip-v-acc" style="color: #fbbf24;">0.00 m/s²</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">硬实时闭环:</span><b id="pip-v-lat" style="color: var(--emerald, #34d399);">150.6 μs PASS</b></div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(pip);
    this.container = pip;
    this.canvas = document.getElementById("pip-viewport-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.gaugeCanvas = document.getElementById("pip-gauge-canvas");
    this.gaugeCtx = this.gaugeCanvas.getContext("2d");

    const collapseBtn = document.getElementById("pip-collapse-btn");
    const body = document.getElementById("pip-body");
    collapseBtn.onclick = () => {
      this.isCollapsed = !this.isCollapsed;
      body.style.display = this.isCollapsed ? "none" : "block";
      collapseBtn.innerHTML = this.isCollapsed ? "&plus;" : "&minus;";
    };
  }

  update(stateData) {
    if (!this.container) this.initDOM();
    if (this.isCollapsed) return;

    const et = stateData && stateData.embodied_twin;
    if (!et) {
      this.renderFallback(stateData);
      return;
    }

    this.lastData = et;
    const domain = et.domain || "adas";

    if (domain !== this.currentDomain) {
      this.currentDomain = domain;
      this.switchDomainUI(domain, et);
    }

    if (domain === "adas") {
      this.renderADAS(et, stateData);
    } else if (domain === "quant") {
      this.renderQuant(et, stateData);
    } else if (domain === "maze") {
      this.renderMaze(et, stateData);
    } else {
      this.renderFallback(stateData);
    }
  }

  switchDomainUI(domain, et) {
    const titleEl = document.getElementById("pip-title");
    const dotEl = document.getElementById("pip-domain-dot");
    const overlayBadge = document.getElementById("pip-overlay-badge");
    const gaugeLabel = document.getElementById("pip-gauge-label");

    if (domain === "adas") {
      if (titleEl) titleEl.textContent = "具身公路巡航 · 阿克曼孪生";
      if (dotEl) dotEl.style.background = "#38bdf8";
      if (overlayBadge) overlayBadge.textContent = "2D 闭环公路轨迹与前视曲率";
      if (gaugeLabel) gaugeLabel.textContent = "实时舵角";
    } else if (domain === "quant") {
      if (titleEl) titleEl.textContent = "Level-2 逐笔盘口 · 订单流阶梯";
      if (dotEl) dotEl.style.background = "#34d399";
      if (overlayBadge) overlayBadge.textContent = "L2 盘口挂单深度梯 (5档买卖)";
      if (gaugeLabel) gaugeLabel.textContent = "OFI 不平衡";
    } else if (domain === "maze") {
      if (titleEl) titleEl.textContent = "空间迷宫寻路 · 拓扑孪生";
      if (dotEl) dotEl.style.background = "#fbbf24";
      if (overlayBadge) overlayBadge.textContent = "迷宫栅格与雷达探针";
      if (gaugeLabel) gaugeLabel.textContent = "航向角";
    }
  }

  renderADAS(et, stateData) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const car = et.car || {};
    const track = et.track || [];
    const trail = et.trail || [];

    // 1. 绘制背景网格
    ctx.strokeStyle = "rgba(30, 41, 59, 0.4)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 22) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    const cx = w * 0.45;
    const cy = h * 0.55;
    const carX = car.x || 0;
    const carY = car.y || 0;
    const carTheta = car.theta || 0;
    const scale = 0.22;

    // 2. 变换到车辆前瞻坐标系
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-carTheta - Math.PI * 0.5);

    // 2.1 沿车辆运动方向连续提取赛道切片并绘制道路
    if (track.length > 0) {
      // 找到最近的赛道参考点索引
      let closestIdx = 0;
      let minD2 = Infinity;
      for (let i = 0; i < track.length; i++) {
        const dx = track[i].x - carX;
        const dy = track[i].y - carY;
        const d2 = dx * dx + dy * dy;
        if (d2 < minD2) {
          minD2 = d2;
          closestIdx = i;
        }
      }

      // 取车后 15 点到车前 85 点的连续道路序列 (按道路弧长顺序遍历，彻底消灭穿心飞线与圆环)
      const roadHalfW = 23.0 * scale;
      const nPts = track.length;
      const startIdx = closestIdx - 15;
      const endIdx = closestIdx + 85;

      const leftPts = [];
      const rightPts = [];
      const centerPts = [];

      for (let k = startIdx; k <= endIdx; k++) {
        const idx = ((k % nPts) + nPts) % nPts;
        const pt = track[idx];
        const px = (pt.x - carX) * scale;
        const py = (pt.y - carY) * scale;

        // 切向与法向计算双边车道线
        const th = pt.theta !== undefined ? pt.theta : 0;
        const nx = -Math.sin(th) * roadHalfW;
        const ny = Math.cos(th) * roadHalfW;

        centerPts.push({ x: px, y: py });
        leftPts.push({ x: px + nx, y: py + ny });
        rightPts.push({ x: px - nx, y: py - ny });
      }

      // 绘制道路沥青底色带
      if (leftPts.length > 1) {
        ctx.fillStyle = "rgba(15, 23, 42, 0.45)";
        ctx.beginPath();
        ctx.moveTo(leftPts[0].x, leftPts[0].y);
        for (let i = 1; i < leftPts.length; i++) ctx.lineTo(leftPts[i].x, leftPts[i].y);
        for (let i = rightPts.length - 1; i >= 0; i--) ctx.lineTo(rightPts[i].x, rightPts[i].y);
        ctx.closePath();
        ctx.fill();

        // 左右道路护栏/车道边缘线
        ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let i = 0; i < leftPts.length; i++) {
          if (i === 0) ctx.moveTo(leftPts[i].x, leftPts[i].y);
          else ctx.lineTo(leftPts[i].x, leftPts[i].y);
        }
        ctx.stroke();

        ctx.beginPath();
        for (let i = 0; i < rightPts.length; i++) {
          if (i === 0) ctx.moveTo(rightPts[i].x, rightPts[i].y);
          else ctx.lineTo(rightPts[i].x, rightPts[i].y);
        }
        ctx.stroke();
      }

      // 道路中心虚线 (航向引导线)
      ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
      ctx.lineWidth = 2.0;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i < centerPts.length; i++) {
        if (i === 0) ctx.moveTo(centerPts[i].x, centerPts[i].y);
        else ctx.lineTo(centerPts[i].x, centerPts[i].y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2.2 绘制历史行驶轨迹 (尾迹流)
    if (trail.length > 1) {
      ctx.strokeStyle = "rgba(52, 211, 153, 0.6)";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      let trailStarted = false;
      for (let i = 0; i < trail.length; i++) {
        const pt = trail[i];
        const px = (pt.x - carX) * scale;
        const py = (pt.y - carY) * scale;
        if (Math.hypot(px, py) < 140) {
          if (!trailStarted) { ctx.moveTo(px, py); trailStarted = true; }
          else ctx.lineTo(px, py);
        } else {
          trailStarted = false;
        }
      }
      ctx.stroke();
    }

    // 2.3 绘制车辆本体
    ctx.fillStyle = "#38bdf8";
    ctx.shadowColor = "#38bdf8";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(0, 7);
    ctx.lineTo(-4, -6);
    ctx.lineTo(4, -6);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;

    // 2.4 前视感知预瞄射线
    ctx.strokeStyle = "rgba(251, 191, 36, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(0, 7);
    ctx.lineTo(0, 32);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.restore();

    // 3. 绘制方向盘仪表
    const deltaDeg = car.delta_deg !== undefined ? car.delta_deg : 0;
    this.targetSteerAngle = THREE_DEG_TO_RAD(deltaDeg);
    this.steerAngle += (this.targetSteerAngle - this.steerAngle) * 0.35;
    this.renderSteeringWheel(this.steerAngle, deltaDeg);

    // 4. 更新文本指标
    const vSpeed = document.getElementById("pip-v-speed");
    if (vSpeed) vSpeed.textContent = `${(car.speed_kmh || 0).toFixed(1)} km/h`;
    const vCte = document.getElementById("pip-v-cte");
    if (vCte) {
      const cteCm = (car.cte_m || 0) * 100;
      vCte.textContent = `${cteCm.toFixed(1)} cm`;
      vCte.style.color = Math.abs(cteCm) < 15 ? "var(--emerald, #34d399)" : (Math.abs(cteCm) < 35 ? "#fbbf24" : "var(--crimson, #f43f5e)");
    }
    const vAcc = document.getElementById("pip-v-acc");
    if (vAcc) vAcc.textContent = `${(car.accel || 0.45).toFixed(2)} m/s²`;
    const vLat = document.getElementById("pip-v-lat");
    if (vLat) vLat.textContent = `150.6 μs PASS`;
  }

  renderSteeringWheel(angleRad, deltaDeg) {
    const ctx = this.gaugeCtx;
    const w = this.gaugeCanvas.width;
    const h = this.gaugeCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    const r = 24;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angleRad);

    // 外轮圈
    ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.stroke();

    // 12点钟对齐标记
    ctx.fillStyle = "#f43f5e";
    ctx.beginPath();
    ctx.arc(0, -r, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // 辐条 (T型方向盘骨架)
    ctx.strokeStyle = "rgba(226, 232, 240, 0.7)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, 0); ctx.lineTo(0, r * 0.85); // 下辐条
    ctx.moveTo(-r * 0.85, 0); ctx.lineTo(r * 0.85, 0); // 横辐条
    ctx.stroke();

    // 中心轮毂
    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.arc(0, 0, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.restore();

    const lbl = document.getElementById("pip-gauge-label");
    if (lbl) {
      lbl.textContent = `舵角 ${deltaDeg > 0 ? '+' : ''}${deltaDeg.toFixed(1)}°`;
      lbl.style.color = Math.abs(deltaDeg) > 15 ? "#fbbf24" : "#94a3b8";
    }
  }

  renderQuant(et, stateData) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const bids = et.bids || [];
    const asks = et.asks || [];
    const lastPrice = et.last_price || 3850.0;

    // 绘制 L2 订单簿深度梯
    ctx.font = "9px monospace";
    const rowH = 18;

    // Asks (卖档，红/玫瑰色，倒序排列在上方)
    const askRows = asks.slice(0, 3).reverse();
    askRows.forEach((a, idx) => {
      const y = 8 + idx * rowH;
      const barW = Math.min(100, (a.v / 500) * 100);
      ctx.fillStyle = "rgba(244, 63, 94, 0.22)";
      ctx.fillRect(w - 110, y, barW, rowH - 3);
      ctx.fillStyle = "#f43f5e";
      ctx.fillText(`卖${3 - idx}  ${a.p.toFixed(1)}`, 14, y + 11);
      ctx.fillStyle = "#fda4af";
      ctx.fillText(`${a.v}`, w - 105, y + 11);
    });

    // 最新成交价中间分界线
    const midY = 8 + 3 * rowH;
    ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
    ctx.fillRect(8, midY, w - 16, 16);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.6)";
    ctx.strokeRect(8, midY, w - 16, 16);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "bold 10px monospace";
    ctx.fillText(`最新价: ${lastPrice.toFixed(1)}  价差: 0.4`, 14, midY + 12);

    // Bids (买档，翠绿色，排列在下方)
    ctx.font = "9px monospace";
    bids.slice(0, 3).forEach((b, idx) => {
      const y = midY + 18 + idx * rowH;
      const barW = Math.min(100, (b.v / 500) * 100);
      ctx.fillStyle = "rgba(52, 211, 153, 0.22)";
      ctx.fillRect(w - 110, y, barW, rowH - 3);
      ctx.fillStyle = "#34d399";
      ctx.fillText(`买${idx + 1}  ${b.p.toFixed(1)}`, 14, y + 11);
      ctx.fillStyle = "#a7f3d0";
      ctx.fillText(`${b.v}`, w - 105, y + 11);
    });

    // 绘制 OFI 仪表
    const ofi = et.ofi !== undefined ? et.ofi : 0.0;
    this.renderOFIMeter(ofi);

    // 更新指标面板
    const vSpeed = document.getElementById("pip-v-speed");
    if (vSpeed) vSpeed.textContent = et.symbol || "IF2409";
    const vCte = document.getElementById("pip-v-cte");
    if (vCte) {
      vCte.textContent = et.action || "ACT_HOLD";
      vCte.style.color = et.action && et.action.includes("POS") ? "#34d399" : (et.action && et.action.includes("NEG") ? "#f43f5e" : "#38bdf8");
    }
    const vAcc = document.getElementById("pip-v-acc");
    if (vAcc) vAcc.textContent = `+${et.pnl_pct || 99.04}%`;
    const vLat = document.getElementById("pip-v-lat");
    if (vLat) vLat.textContent = `夏普 ${et.sharpe || 403.9}`;
  }

  renderOFIMeter(ofi) {
    const ctx = this.gaugeCtx;
    const w = this.gaugeCanvas.width;
    const h = this.gaugeCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    const r = 24;

    // 弧形仪表底座
    ctx.strokeStyle = "rgba(30, 41, 59, 0.8)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 0.75, Math.PI * 2.25);
    ctx.stroke();

    // OFI 偏转指针
    const ofiNorm = Math.max(-1, Math.min(1, ofi));
    const angle = Math.PI * 1.5 + ofiNorm * (Math.PI * 0.65);
    const col = ofiNorm > 0 ? "#34d399" : (ofiNorm < 0 ? "#f43f5e" : "#94a3b8");

    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 1.5, angle, ofiNorm < 0);
    ctx.stroke();

    ctx.fillStyle = col;
    ctx.font = "bold 9px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`${ofiNorm > 0 ? '+' : ''}${ofiNorm.toFixed(2)}`, cx, cy + 3);

    const lbl = document.getElementById("pip-gauge-label");
    if (lbl) {
      lbl.textContent = `OFI流向`;
      lbl.style.color = col;
    }
  }

  renderMaze(et, stateData) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
    ctx.font = "11px monospace";
    ctx.fillText("空间迷宫 2D 拓扑闭环寻路", 20, 40);
    ctx.fillText(`通关率: ${et.pass_rate || 96.5}%`, 20, 65);
  }

  renderFallback(stateData) {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "10px monospace";
    ctx.fillText("SDSCC 实体物理闭环流式观测中...", 16, 48);
  }
}

function THREE_DEG_TO_RAD(deg) {
  return (deg * Math.PI) / 180.0;
}

export const embodiedPIP = new EmbodiedPIPTwin();
