/**
 * SDSCC Cosmic Dispatcher (Event Hub & Pipeline Orchestrator)
 * 
 * 核心调度控制器：
 * 1. 负责实例化与管理 Web Worker，将所有数据反序列化与几何计算调度至后台；
 * 2. 衔接 Model 状态与 View 渲染流水线，维持严格单向数据流与 60fps 丝滑渲染；
 * 3. 驱动 API 遥测轮询与用户交互事件分发。
 */

export class CosmicDispatcher {
  constructor(model) {
    this.model = model;
    this.view = null;
    this.worker = null;

    this.rafId = null;
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.fpsTimer = performance.now();

    // 后端遥测轮询定时器
    this.telemetryInterval = null;
    this.isPolling = false;
  }

  /**
   * 初始化调度中枢：启动 Worker、装配 View、启动主渲染泵
   */
  async init(containerElement) {
    // 1. 实例化后台计算 Worker
    this._initWorker();

    // 2. 动态导入 View 并装配
    const { CosmicView } = await import("./cosmic_view.js");
    this.view = new CosmicView(containerElement, this.model, this);

    // 3. 启动主线程 60fps 渲染循环 (Zero Heavy Task on Main Thread)
    this._startRenderPump();

    // 4. 发起硬件规格同步
    this.worker.postMessage({
      type: "SET_HARDWARE",
      payload: {
        gpuName: this.model.hardware.gpuName,
        vramMB: this.model.hardware.vramMB,
        throughputGCells: this.model.hardware.throughputGCells
      }
    });

    // 5. 启动遥测流式监听 (40Hz 轮询，计算移交 Worker)
    this._startTelemetryPolling();

    console.log("[CosmicDispatcher] SDSCC MVD Engine initialized successfully.");
  }

  _initWorker() {
    this.worker = new Worker("./workers/cosmic_worker.js");

    this.worker.onmessage = (e) => {
      const { type, payload } = e.data;

      switch (type) {
        case "ORGANISM_PROCESSED":
          // Worker 已经零拷贝生成 TypedArray，安全写入 Model
          this.model.setOrganismPayload(this._pendingMeta || {}, payload);
          this._pendingMeta = null;
          break;

        case "TELEMETRY_PROCESSED":
          this.model.pushTelemetry(payload);
          break;

        case "HARDWARE_UPDATED":
          this.model.updateHardware(payload);
          break;

        default:
          console.warn("[CosmicDispatcher] Unknown worker message:", type);
      }
    };

    this.worker.onerror = (err) => {
      console.error("[CosmicDispatcher] Worker error:", err);
    };
  }

  /**
   * 调度生命体加载：彻底不在主线程做 JSON 或几何计算，全部抛给 Worker！
   */
  loadOrganism(meta, rawData) {
    this._pendingMeta = meta;

    // 将原始数据直接交由后台 Worker 解析计算
    this.worker.postMessage({
      type: "PROCESS_ORGANISM",
      payload: rawData
    });
  }

  /**
   * 启动遥测流拉取与解算
   */
  _startTelemetryPolling() {
    if (this.telemetryInterval) clearInterval(this.telemetryInterval);

    this.telemetryInterval = setInterval(async () => {
      if (!this.model.status.isPlaying) return;

      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        if (res.ok) {
          const rawFrame = await res.json();
          // 将高频遥测帧投递给 Worker 处理相空间解算
          this.worker.postMessage({
            type: "PROCESS_TELEMETRY_FRAME",
            payload: rawFrame
          });
        }
      } catch (err) {
        // 静默处理轮询异常（例如后端离线）
      }
    }, 25); // 40Hz
  }

  /**
   * 主线程渲染泵：纯 GPU 硬件管线，毫秒级开销
   */
  _startRenderPump() {
    const loop = (now) => {
      this.rafId = requestAnimationFrame(loop);

      const deltaMs = now - this.lastFrameTime;
      this.lastFrameTime = now;

      // 帧率统计
      this.frameCount++;
      if (now - this.fpsTimer >= 1000) {
        const currentFps = Math.round((this.frameCount * 1000) / (now - this.fpsTimer));
        this.model.updateStatus({
          fps: currentFps,
          renderFrameTimeMs: parseFloat(deltaMs.toFixed(2))
        });
        this.frameCount = 0;
        this.fpsTimer = now;

        // 更新顶部 FPS 标签
        const fpsEl = document.getElementById("hud-fps-val");
        if (fpsEl) fpsEl.innerText = `${currentFps} FPS`;
      }

      // 执行纯粹的 WebGL 渲染
      if (this.view) {
        this.view.renderLoop();
      }
    };

    this.rafId = requestAnimationFrame(loop);
  }

  // 用户交互指令分发
  setCameraMode(mode) {
    this.model.updateStatus({ cameraMode: mode });
  }

  togglePlayPause() {
    const cur = this.model.status.isPlaying;
    this.model.updateStatus({ isPlaying: !cur });
    return !cur;
  }
}
