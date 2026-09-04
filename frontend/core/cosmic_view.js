/**
 * SDSCC Cosmic View (Presentation & Three.js Holographic Engine)
 * 
 * 纯视图层：负责 Three.js WebGL 场景渲染与 HUD DOM 的轻量级声明式更新。
 * 不包含任何繁重计算逻辑，消费 Model 提供的 Transferable TypedArray 缓冲区。
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class CosmicView {
  constructor(containerElement, model, dispatcher) {
    this.container = containerElement;
    this.model = model;
    this.dispatcher = dispatcher;

    // Three.js 核心对象
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // 场景视觉组件
    this.cosmicCage = null;      // 硬件算力宇宙边界框
    this.starfield = null;       // 深空星尘
    this.cellPoints = null;      // 细胞微粒集群 (BufferGeometry)
    this.synapseLines = null;    // 突触光流纤维 (LineSegments)
    this.organismAura = null;    // 生命体物理引力范围光环 (SphereWireframe)
    this.voxelWorldBox = null;   // 100M+ 4D 全息体素边界

    // 相机运镜平滑目标
    this.targetCameraPos = new THREE.Vector3(0, 650, 1350);
    this.targetControlsTarget = new THREE.Vector3(0, 0, 0);

    // 示波器与相图 Canvas 引用
    this.phaseCanvas = null;
    this.phaseCtx = null;
    this.energyCanvas = null;
    this.energyCtx = null;

    // 初始化视口与事件监听
    this._initThreeScene();
    this._bindModelEvents();
  }

  _initThreeScene() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    // 1. Scene & Fog
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x02040a, 0.00025);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 50000);
    this.camera.position.copy(this.targetCameraPos);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 15000;
    this.controls.minDistance = 2.0;

    // 5. 构筑【硬件算力宇宙边界线框】(Cosmic Bounding Wireframe)
    this._buildCosmicHorizonCage();

    // 6. 构筑深空微弱背景星尘
    this._buildCosmicStarfield();

    // 7. 构筑环境光源
    const ambLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambLight);

    // 8. 窗口大小自适应
    window.addEventListener("resize", () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  /**
   * 构筑神圣的【硬件算力宇宙视界】(Cosmic Horizon)
   * 立方体大小为 1000x1000x1000，代表 RTX 5060 的 8GB 显存容纳上限
   */
  _buildCosmicHorizonCage() {
    const size = this.model.hardware.cosmicBoundSize;
    const half = size * 0.5;

    const group = new THREE.Group();

    // 金黄色高质感晶莹边框
    const geom = new THREE.BoxGeometry(size, size, size);
    const edges = new THREE.EdgesGeometry(geom);
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.35,
      linewidth: 1
    });
    const wireframe = new THREE.LineSegments(edges, edgeMat);
    group.add(wireframe);

    // 底部基底辅助参考平面网格
    const grid = new THREE.GridHelper(size, 20, 0x1e293b, 0x0f172a);
    grid.position.y = -half;
    group.add(grid);

    // 宇宙八个顶角的激光定位信标
    const cornerGeom = new THREE.BoxGeometry(12, 12, 12);
    const cornerMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true });
    const corners = [
      [-half, -half, -half], [half, -half, -half],
      [-half, half, -half],  [half, half, -half],
      [-half, -half, half],  [half, -half, half],
      [-half, half, half],   [half, half, half]
    ];
    corners.forEach(pos => {
      const m = new THREE.Mesh(cornerGeom, cornerMat);
      m.position.set(...pos);
      group.add(m);
    });

    this.cosmicCage = group;
    this.scene.add(group);
  }

  _buildCosmicStarfield() {
    const starCount = 3000;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const i3 = i * 3;
      const r = 2000 + Math.random() * 8000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);

      positions[i3]     = r * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = r * Math.cos(phi);

      colors[i3]     = 0.4 + Math.random() * 0.3;
      colors[i3 + 1] = 0.6 + Math.random() * 0.4;
      colors[i3 + 2] = 0.8 + Math.random() * 0.2;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 3.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.45,
      sizeAttenuation: true
    });

    this.starfield = new THREE.Points(geom, mat);
    this.scene.add(this.starfield);
  }

  /**
   * 绑定 Model 状态变更，执行响应式渲染更新
   */
  _bindModelEvents() {
    this.model.on("ORGANISM_LOADED", (org) => {
      this._renderOrganismGeometry(org);
      this._updateHUDDetails(org);
      this._adjustCameraToOrganism(org);
    });

    this.model.on("TELEMETRY_UPDATED", (telemetry) => {
      this._renderTelemetryPlots(telemetry);
    });

    this.model.on("STATUS_CHANGED", (status) => {
      if (status.cameraMode === "cosmic") {
        this.targetCameraPos.set(0, 650, 1350);
        this.targetControlsTarget.set(0, 0, 0);
      } else if (status.cameraMode === "focus") {
        this._adjustCameraToOrganism(this.model.organism);
      }
    });
  }

  /**
   * 依据 Worker 交付的 TypedArray 零拷贝渲染细胞与突触
   */
  _renderOrganismGeometry(org) {
    const geo = org.geometry;
    if (!geo) return;

    // 1. 清理旧网格
    if (this.cellPoints) {
      this.scene.remove(this.cellPoints);
      this.cellPoints.geometry.dispose();
      this.cellPoints.material.dispose();
      this.cellPoints = null;
    }
    if (this.synapseLines) {
      this.scene.remove(this.synapseLines);
      this.synapseLines.geometry.dispose();
      this.synapseLines.material.dispose();
      this.synapseLines = null;
    }
    if (this.organismAura) {
      this.scene.remove(this.organismAura);
      this.organismAura.geometry.dispose();
      this.organismAura.material.dispose();
      this.organismAura = null;
    }
    if (this.voxelWorldBox) {
      this.scene.remove(this.voxelWorldBox);
      this.voxelWorldBox.geometry.dispose();
      this.voxelWorldBox.material.dispose();
      this.voxelWorldBox = null;
    }

    // 2. 构造细胞粒子群 BufferGeometry
    const cellGeom = new THREE.BufferGeometry();
    cellGeom.setAttribute("position", new THREE.BufferAttribute(geo.positions, 3));
    cellGeom.setAttribute("color", new THREE.BufferAttribute(geo.colors, 3));

    const pointSize = org.nominalScale > 10000000 ? 1.8 : (org.nominalScale > 100000 ? 3.0 : 5.0);
    const cellMat = new THREE.PointsMaterial({
      size: pointSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    });
    this.cellPoints = new THREE.Points(cellGeom, cellMat);
    this.scene.add(this.cellPoints);

    // 3. 构造突触光流纤维 LineSegments
    if (geo.synPositions && geo.synPositions.length > 0) {
      const synGeom = new THREE.BufferGeometry();
      synGeom.setAttribute("position", new THREE.BufferAttribute(geo.synPositions, 3));
      synGeom.setAttribute("color", new THREE.BufferAttribute(geo.synColors, 3));

      const synMat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.4,
        blending: THREE.AdditiveBlending
      });
      this.synapseLines = new THREE.LineSegments(synGeom, synMat);
      this.scene.add(this.synapseLines);
    }

    // 4. 构造生命体在宇宙中的真实物理包络光环 (Aura)
    const auraGeom = new THREE.SphereGeometry(geo.trueRadius * 1.05, 24, 16);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.12
    });
    this.organismAura = new THREE.Mesh(auraGeom, auraMat);
    this.scene.add(this.organismAura);

    // 5. 若为一亿细胞 (100M+) 全息 4D 世界模型，添加宏伟体素连续场半透明体
    if (org.nominalScale >= 50000000) {
      const boxSize = geo.trueRadius * 1.8;
      const boxGeom = new THREE.BoxGeometry(boxSize, boxSize, boxSize);
      const boxMat = new THREE.MeshBasicMaterial({
        color: 0xa855f7,
        wireframe: true,
        transparent: true,
        opacity: 0.25
      });
      this.voxelWorldBox = new THREE.Mesh(boxGeom, boxMat);
      this.scene.add(this.voxelWorldBox);
    }
  }

  /**
   * 平滑相机运镜：依据真实尺度平滑推进或拉远
   */
  _adjustCameraToOrganism(org) {
    if (!org || !org.geometry) return;
    const r = org.geometry.trueRadius;

    // 视距根据真实物理半径动态自适应 (微尘推进至近处，巨型星云推远)
    const viewDist = Math.max(12.0, r * 2.8);
    this.targetCameraPos.set(0, viewDist * 0.45, viewDist * 1.15);
    this.targetControlsTarget.set(0, 0, 0);
  }

  /**
   * 更新屏幕 HUD 元件
   */
  _updateHUDDetails(org) {
    // 更新顶部宇宙占比徽章
    const volumeBadge = document.getElementById("hud-volume-badge");
    if (volumeBadge) {
      volumeBadge.innerHTML = `
        <span style="color:#94a3b8;">硬件宇宙体积占比:</span> 
        <strong style="color:#38bdf8; font-size:14px;">${org.volumeRatioPercent}%</strong>
        <span style="color:#64748b; font-size:11px;">(物理半径 R=${org.trueRadius.toFixed(1)}m)</span>
      `;
    }

    // 更新生命体详情名牌
    const orgTitle = document.getElementById("hud-org-title");
    if (orgTitle) {
      orgTitle.innerText = org.name;
    }
    const orgMeta = document.getElementById("hud-org-meta");
    if (orgMeta) {
      orgMeta.innerHTML = `
        <div>标称尺度: <strong style="color:#f1f5f9;">${org.nominalScale.toLocaleString()}</strong> 细胞</div>
        <div>实际解析: <strong style="color:#34d399;">${org.actualCellsCount.toLocaleString()}</strong> 细胞 / <strong style="color:#a855f7;">${org.synapsesCount.toLocaleString()}</strong> 突触</div>
        <div>生境分类: <span style="color:#fbbf24;">${org.domain}</span></div>
      `;
    }
  }

  /**
   * 遥测示波器与相空间轨道实时渲染 (原生 Canvas 极速绘制)
   */
  _renderTelemetryPlots(telemetry) {
    const th = this.model.telemetryHistory;
    const count = th.count;
    if (count < 2) return;

    // 1. 相平面轨道绘制 (Phase Portrait: X=Energy, Y=Lyapunov)
    if (!this.phaseCanvas) {
      this.phaseCanvas = document.getElementById("canvas-phase-portrait");
      if (this.phaseCanvas) this.phaseCtx = this.phaseCanvas.getContext("2d");
    }
    if (this.phaseCtx) {
      const ctx = this.phaseCtx;
      const w = this.phaseCanvas.width;
      const h = this.phaseCanvas.height;

      ctx.fillStyle = "rgba(10, 16, 28, 0.25)";
      ctx.fillRect(0, 0, w, h);

      ctx.beginPath();
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.5;

      const halfW = w * 0.5;
      const halfH = h * 0.5;

      for (let i = 0; i < count; i++) {
        const readIdx = (th.writeIndex - count + i + th.telemetryHistoryCapacity) % th.telemetryHistoryCapacity;
        const e = th.energies[readIdx];
        const l = th.lyapunovs[readIdx];

        const px = halfW + (e - 1.0) * (w * 0.35);
        const py = halfH - l * (h * 0.35);

        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // 最新相空间点高亮光标
      const lastE = telemetry.meanEnergy;
      const lastL = telemetry.lyapunovEst;
      const curX = halfW + (lastE - 1.0) * (w * 0.35);
      const curY = halfH - lastL * (h * 0.35);
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(curX, curY, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * 主渲染时钟驱动 (与 requestAnimationFrame 严格同步)
   */
  renderLoop() {
    // 平滑运镜缓动 (Lerp Camera & Target)
    this.camera.position.lerp(this.targetCameraPos, 0.05);
    this.controls.target.lerp(this.targetControlsTarget, 0.05);
    this.controls.update();

    // 旋转背景星尘与宇宙微弱自转
    if (this.starfield) {
      this.starfield.rotation.y += 0.00015;
    }
    if (this.cellPoints && this.model.status.isPlaying) {
      this.cellPoints.rotation.y += 0.001;
      if (this.synapseLines) this.synapseLines.rotation.y += 0.001;
      if (this.voxelWorldBox) this.voxelWorldBox.rotation.y += 0.0005;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
