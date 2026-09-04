/**
 * SDSCC Cosmic Web Worker (Off-Main-Thread Compute Engine)
 * 
 * 职责：
 * 1. 彻底解耦 UI 主线程，承担所有几何空间坐标换算、LOD 视锥剔除与拓扑解压
 * 2. 基于宿主硬件算力常数（如 RTX 5060 8GB / 1.78×10^8 细胞）推导物理宇宙视界与真实体积占比
 * 3. 产出 TypedArray (Float32Array) 并通过 Transferable 对象零拷贝交付给渲染流水线
 */

// 默认硬件算力常数 (NVIDIA GeForce RTX 5060 8GB)
const HARDWARE_SPECS = {
  gpuName: "NVIDIA GeForce RTX 5060 Laptop GPU",
  vramMB: 8192,
  cellFootprintBytes: 48, // CompactSoAGenome 单细胞内存开销
  maxCapacityCells: Math.floor((8192 * 1024 * 1024) / 48), // ~178,956,970 细胞
  throughputGCells: 8.65, // 实测 86.5 亿细胞/秒
  cosmicBoundSize: 1000.0 // 宇宙立方体边长 (-500 到 +500)
};

self.onmessage = function (e) {
  const { type, payload } = e.data;

  switch (type) {
    case "SET_HARDWARE":
      if (payload) {
        if (payload.vramMB) HARDWARE_SPECS.vramMB = payload.vramMB;
        if (payload.gpuName) HARDWARE_SPECS.gpuName = payload.gpuName;
        if (payload.throughputGCells) HARDWARE_SPECS.throughputGCells = payload.throughputGCells;
        HARDWARE_SPECS.maxCapacityCells = Math.floor((HARDWARE_SPECS.vramMB * 1024 * 1024) / HARDWARE_SPECS.cellFootprintBytes);
      }
      self.postMessage({
        type: "HARDWARE_UPDATED",
        payload: { ...HARDWARE_SPECS }
      });
      break;

    case "PROCESS_ORGANISM":
      processOrganismTopology(payload);
      break;

    case "PROCESS_TELEMETRY_FRAME":
      processTelemetryFrame(payload);
      break;

    default:
      console.warn("[CosmicWorker] Unknown message type:", type);
  }
};

/**
 * 将生命体拓扑数据解析、映射至真实硬件算力宇宙空间
 */
function processOrganismTopology(data) {
  const startTime = performance.now();
  const cells = data.cells || [];
  const synapses = data.synapses || [];
  const nominalScale = data.cells_scale || cells.length || 1;
  const actualCellsCount = cells.length;

  // 1. 计算硬件宇宙相对空间指标
  const maxCapacity = HARDWARE_SPECS.maxCapacityCells;
  // 体积分数比 (Volumetric Occupancy Ratio)
  const volumeRatio = Math.min(1.0, nominalScale / maxCapacity);
  // 相对空间半径: R_org = (L/2) * cbrt(N / N_max)
  const halfSize = HARDWARE_SPECS.cosmicBoundSize * 0.5;
  const trueRadius = Math.max(2.5, halfSize * Math.cbrt(volumeRatio));

  // 2. 细胞坐标与属性解压与映射
  const numCells = actualCellsCount;
  const positions = new Float32Array(numCells * 3);
  const colors = new Float32Array(numCells * 3);
  const sizes = new Float32Array(numCells);

  // 计算原始包围盒以进行空间正则化
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < numCells; i++) {
    const c = cells[i];
    const x = c.x || 0, y = c.y || 0, z = c.z || 0;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  const rawSpanX = Math.max(1e-4, maxX - minX);
  const rawSpanY = Math.max(1e-4, maxY - minY);
  const rawSpanZ = Math.max(1e-4, maxZ - minZ);
  const maxRawSpan = Math.max(rawSpanX, Math.max(rawSpanY, rawSpanZ));

  // 映射到生命体在算力宇宙中的真实物理尺寸
  for (let i = 0; i < numCells; i++) {
    const c = cells[i];
    const rawX = c.x || 0;
    const rawY = c.y || 0;
    const rawZ = c.z || 0;

    // 归一化后乘以真实物理半径
    const normX = ((rawX - (minX + maxX) * 0.5) / maxRawSpan);
    const normY = ((rawY - (minY + maxY) * 0.5) / maxRawSpan);
    const normZ = ((rawZ - (minZ + maxZ) * 0.5) / maxRawSpan);

    const idx3 = i * 3;
    positions[idx3]     = normX * trueRadius * 1.8;
    positions[idx3 + 1] = normY * trueRadius * 1.8;
    positions[idx3 + 2] = normZ * trueRadius * 1.8;

    // 颜色编码 (根据层级与动力学类型)
    const layer = c.layer || "L2_ASSOCIATION";
    let r = 0.2, g = 0.8, b = 0.9;
    if (layer === "L1_SENSORY" || (c.type && String(c.type).startsWith("Sense"))) {
      r = 0.13; g = 0.83; b = 0.93; // 传感器天蓝 #22d3ee
    } else if (layer === "L3_MOTOR" || (c.type && String(c.type).startsWith("Act"))) {
      r = 0.96; g = 0.25; b = 0.37; // 效应器绯红 #f43f5e
    } else {
      r = 0.20; g = 0.83; b = 0.60; // 联络核翡翠绿 #34d399
    }

    colors[idx3]     = r;
    colors[idx3 + 1] = g;
    colors[idx3 + 2] = b;

    // 点尺寸：小尺度生命体单点视觉稍微增强，大尺度生命体细密微粒
    sizes[i] = nominalScale > 10000000 ? 1.5 : (nominalScale > 100000 ? 2.5 : 4.5);
  }

  // 3. 突触坐标解压 (仅对显示的部分突触)
  const maxSynDisplay = Math.min(synapses.length, 50000);
  const synPositions = new Float32Array(maxSynDisplay * 6);
  const synColors = new Float32Array(maxSynDisplay * 6);
  let synCount = 0;

  // 建立快速 ID -> 坐标索引表
  const idToCoord = new Map();
  for (let i = 0; i < numCells; i++) {
    const cid = cells[i].id !== undefined ? cells[i].id : i;
    idToCoord.set(cid, i);
  }

  for (let s = 0; s < synapses.length && synCount < maxSynDisplay; s++) {
    const syn = synapses[s];
    const u = idToCoord.get(syn.from);
    const v = idToCoord.get(syn.to);
    if (u !== undefined && v !== undefined) {
      const u3 = u * 3;
      const v3 = v * 3;
      const s6 = synCount * 6;

      synPositions[s6]     = positions[u3];
      synPositions[s6 + 1] = positions[u3 + 1];
      synPositions[s6 + 2] = positions[u3 + 2];
      synPositions[s6 + 3] = positions[v3];
      synPositions[s6 + 4] = positions[v3 + 1];
      synPositions[s6 + 5] = positions[v3 + 2];

      const w = Math.min(1.0, Math.max(0.1, Math.abs(syn.weight || 0.5)));
      for (let k = 0; k < 6; k += 3) {
        synColors[s6 + k]     = 0.3 * w;
        synColors[s6 + k + 1] = 0.7 * w;
        synColors[s6 + k + 2] = 0.9 * w;
      }
      synCount++;
    }
  }

  const parseDurationMs = performance.now() - startTime;

  // 4. 返回 Transferable 零拷贝包
  const responsePayload = {
    nominalScale,
    actualCellsCount,
    synapsesCount: synapses.length,
    volumeRatio,
    volumeRatioPercent: (volumeRatio * 100).toFixed(6),
    trueRadius,
    cosmicBoundSize: HARDWARE_SPECS.cosmicBoundSize,
    maxCapacityCells: HARDWARE_SPECS.maxCapacityCells,
    gpuName: HARDWARE_SPECS.gpuName,
    vramMB: HARDWARE_SPECS.vramMB,
    parseDurationMs: parseDurationMs.toFixed(2),
    numCells,
    synCount,
    positions,
    colors,
    sizes,
    synPositions: synPositions.subarray(0, synCount * 6),
    synColors: synColors.subarray(0, synCount * 6)
  };

  self.postMessage(
    {
      type: "ORGANISM_PROCESSED",
      payload: responsePayload
    },
    [
      positions.buffer,
      colors.buffer,
      sizes.buffer,
      synPositions.buffer,
      synColors.buffer
    ]
  );
}

/**
 * 遥测帧高频流解算 (脱离主线程执行相图、能量谱计算)
 */
function processTelemetryFrame(rawFrame) {
  const states = rawFrame.states || [];
  const len = states.length;
  if (len === 0) return;

  let energySum = 0;
  let maxAbs = 0;

  for (let i = 0; i < len; i++) {
    const val = states[i];
    const sq = val * val;
    energySum += sq;
    if (Math.abs(val) > maxAbs) maxAbs = Math.abs(val);
  }

  const meanEnergy = energySum / len;
  const lyapunovEst = Math.log(Math.max(1e-5, maxAbs)) * 0.1;

  self.postMessage({
    type: "TELEMETRY_PROCESSED",
    payload: {
      step: rawFrame.step || 0,
      timestamp: performance.now(),
      meanEnergy,
      maxAbs,
      lyapunovEst,
      rawSlice: states.slice(0, 16)
    }
  });
}
