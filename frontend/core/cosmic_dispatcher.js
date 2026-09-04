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

    // 5. 启动遥测流式监听 (优先 WebSocket，降级为非阻塞防拥塞 HTTP 轮询)
    this._startTelemetryStream();

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

  _startTelemetryStream() {
    this._initWebSocketTelemetry();
  }

  _initWebSocketTelemetry() {
    try {
      const loc = window.location;
      const wsProtocol = loc.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${loc.host}/ws`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("[CosmicDispatcher] WebSocket telemetry stream connected.");
      };

      this.ws.onmessage = (e) => {
        if (!this.model.status.isPlaying) return;
        try {
          const rawFrame = JSON.parse(e.data);
          this.worker.postMessage({
            type: "PROCESS_TELEMETRY_FRAME",
            payload: rawFrame
          });
        } catch (err) {}
      };

      this.ws.onerror = () => {
        this.ws = null;
        this._startHttpPollingFallback();
      };

      this.ws.onclose = () => {
        this.ws = null;
        this._startHttpPollingFallback();
      };
    } catch (e) {
      this._startHttpPollingFallback();
    }
  }

  _startHttpPollingFallback() {
    if (this.telemetryTimeout) clearTimeout(this.telemetryTimeout);

    const poll = async () => {
      if (!this.model.status.isPlaying) {
        this.telemetryTimeout = setTimeout(poll, 250);
        return;
      }
      try {
        const res = await fetch("/api/state", { cache: "no-store" });
        if (res.ok) {
          const rawFrame = await res.json();
          this.worker.postMessage({
            type: "PROCESS_TELEMETRY_FRAME",
            payload: rawFrame
          });
        }
      } catch (err) {}
      // 严格串行防拥塞：前一个请求彻底返回后，延迟 150ms (约6.6Hz) 再发起下一个
      this.telemetryTimeout = setTimeout(poll, 150);
    };

    this.telemetryTimeout = setTimeout(poll, 150);
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

  setScenarioMode(mode) {
    if (this.view) {
      this.view.setScenarioMode(mode);
    }
  }

  togglePlayPause() {
    const cur = this.model.status.isPlaying;
    this.model.updateStatus({ isPlaying: !cur });
    return !cur;
  }
}
