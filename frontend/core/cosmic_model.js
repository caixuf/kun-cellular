/**
 * SDSCC Cosmic Model (Data Layer / Single Source of Truth)
 * 
 * 纯数据状态模型，不引用任何 DOM，不依赖 Three.js。
 * 负责状态存储、遥测历史环形缓冲、属性计算与响应式事件订阅。
 */

export class CosmicModel {
  constructor() {
    // 宿主硬件算力常数与宇宙界限
    this.hardware = {
      gpuName: "NVIDIA GeForce RTX 5060 Laptop GPU",
      vramMB: 8192,
      maxCapacityCells: 178956970, // 8GB / 48B
      throughputGCells: 8.65,
      cosmicBoundSize: 1000.0
    };

    // 当前活跃生命体状态
    this.organism = {
      id: "none",
      name: "未加载生命体",
      domain: "未知生境",
      nominalScale: 0,
      actualCellsCount: 0,
      synapsesCount: 0,
      volumeRatio: 0,
      volumeRatioPercent: "0.000000",
      trueRadius: 10.0,
      geometry: null // 存放 Worker 移交的 TypedArray 几何包
    };

    // 遥测环形缓冲区 (固定容量，零垃圾回收)
    this.telemetryHistoryCapacity = 120;
    this.telemetryHistory = {
      steps: new Float32Array(this.telemetryHistoryCapacity),
      energies: new Float32Array(this.telemetryHistoryCapacity),
      lyapunovs: new Float32Array(this.telemetryHistoryCapacity),
      count: 0,
      writeIndex: 0
    };

    // 观测台全局运行状态
    this.status = {
      fps: 60,
      renderFrameTimeMs: 16.6,
      cameraMode: "cosmic", // "cosmic" (全宇宙大景) 或 "focus" (生命体局部聚焦)
      cosmicZoomLevel: 1.0,  // 对数视距标尺
      isPlaying: true
    };

    // 事件监听器注册表
    this._listeners = new Map();
  }

  // 事件订阅机制
  on(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this._listeners.has(event)) {
      this._listeners.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this._listeners.has(event)) {
      for (const cb of this._listeners.get(event)) {
        try {
          cb(data);
        } catch (err) {
          console.error(`[CosmicModel] Error in listener for ${event}:`, err);
        }
      }
    }
  }

  // 更新硬件参数
  updateHardware(specs) {
    Object.assign(this.hardware, specs);
    this.emit("HARDWARE_CHANGED", this.hardware);
  }

  // 更新生命体数据（由 Worker 计算完成后的结果写入）
  setOrganismPayload(meta, workerPayload) {
    this.organism.id = meta.id || this.organism.id;
    this.organism.name = meta.name || this.organism.name;
    this.organism.domain = meta.domain || this.organism.domain;
    this.organism.nominalScale = workerPayload.nominalScale;
    this.organism.actualCellsCount = workerPayload.actualCellsCount;
    this.organism.synapsesCount = workerPayload.synapsesCount;
    this.organism.volumeRatio = workerPayload.volumeRatio;
    this.organism.volumeRatioPercent = workerPayload.volumeRatioPercent;
    this.organism.trueRadius = workerPayload.trueRadius;
    this.organism.geometry = workerPayload;

    this.emit("ORGANISM_LOADED", this.organism);
  }

  // 写入遥测帧
  pushTelemetry(telemetry) {
    const th = this.telemetryHistory;
    const idx = th.writeIndex;

    th.steps[idx] = telemetry.step;
    th.energies[idx] = telemetry.meanEnergy;
    th.lyapunovs[idx] = telemetry.lyapunovEst;

    th.writeIndex = (idx + 1) % this.telemetryHistoryCapacity;
    if (th.count < this.telemetryHistoryCapacity) th.count++;

    this.emit("TELEMETRY_UPDATED", telemetry);
  }

  // 更新运行状态
  updateStatus(patch) {
    Object.assign(this.status, patch);
    this.emit("STATUS_CHANGED", this.status);
  }
}
