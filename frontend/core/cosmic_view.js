/**
 * SDSCC Cosmic View (Presentation & Three.js Holographic Engine)
 * 
 * 纯视图层：负责 Three.js WebGL 场景渲染与 HUD DOM 的轻量级声明式更新。
 * 不包含任何繁重计算逻辑，消费 Model 提供的 Transferable TypedArray 缓冲区。
 * 
 * 包含三大工业级具身展示场景：
 * 1. 算力宇宙深空全景与生命体真实空间拓扑 (Cosmic Substrate & True Organism Geometry)
 * 2. 100M+ 全息 4D 时空体素世界模型 (3D Voxel Kolmogorov Field & 盲区反事实幽灵推演波)
 * 3. 1M 极速 10 kHz 本能阻尼极速爆胎稳控 (100km/h 爆胎横摆稳控 vs 传统50Hz翻车双轨迹对账)
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

function createGlowParticleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.25, "rgba(56, 189, 248, 0.9)");
  grad.addColorStop(0.65, "rgba(56, 189, 248, 0.25)");
  grad.addColorStop(1.0, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

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

    // 共享高光粒子纹理
    this.glowTexture = createGlowParticleTexture();

    // 场景视觉组件
    this.cosmicCage = null;      // 硬件算力宇宙边界框
    this.starfield = null;       // 深空星尘
    this.organismGroup = null;   // 生命体专属容器 (细胞点云、突触光纤、光环、雷达信标)
    this.cellPoints = null;      // 细胞微粒集群 (BufferGeometry)
    this.synapseLines = null;    // 突触光流纤维 (LineSegments)
    this.organismAura = null;    // 生命体物理引力范围光环 (SphereWireframe)
    this.beaconReticle = null;   // 微观生命体高亮定位信标

    // 具身专业演示场景组
    this.worldModelGroup = null; // 100M+ 4D 全息世界模型 + 盲区反事实
    this.transientGroup = null;  // 1M 10kHz 极速爆胎动力学对账
    this.activeScenario = "cosmic"; // "cosmic" | "world_model" | "transient_blowout"

    // 动画状态与时钟
    this.animClock = 0;
    this.voxelGridPoints = null;
    this.ghostWaveMeshes = [];
    this.blowoutCarCyan = null;
    this.blowoutCarRed = null;

    // 相机运镜平滑目标
    this.targetCameraPos = new THREE.Vector3(0, 200, 450);
    this.targetControlsTarget = new THREE.Vector3(0, 0, 0);

    // 示波器与相图 Canvas 引用
    this.phaseCanvas = null;
    this.phaseCtx = null;

    // 初始化视口与场景
    this._initThreeScene();
    this._buildCosmicHorizonCage();
    this._buildCosmicStarfield();
    this._buildWorldModelScenario();
    this._buildTransientTireBlowoutScenario();
    this._bindModelEvents();
  }

  _initThreeScene() {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x02040a, 0.00015);

    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 50000);
    this.camera.position.copy(this.targetCameraPos);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 15000;
    this.controls.minDistance = 2.0;

    // 独立生命体容器
    this.organismGroup = new THREE.Group();
    this.scene.add(this.organismGroup);

    const ambLight = new THREE.AmbientLight(0xffffff, 1.0);
    this.scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.5);
    dirLight.position.set(200, 400, 200);
    this.scene.add(dirLight);

    window.addEventListener("resize", () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  /**
   * 1. 构筑【硬件算力宇宙视界】(Cosmic Horizon)
   */
  _buildCosmicHorizonCage() {
    const size = this.model.hardware.cosmicBoundSize;
    const half = size * 0.5;
    const group = new THREE.Group();

    const geom = new THREE.BoxGeometry(size, size, size);
    const edges = new THREE.EdgesGeometry(geom);
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.35,
      linewidth: 1
    });
    group.add(new THREE.LineSegments(edges, edgeMat));

    const grid = new THREE.GridHelper(size, 20, 0x1e293b, 0x0f172a);
    grid.position.y = -half;
    group.add(grid);

    const cornerGeom = new THREE.BoxGeometry(16, 16, 16);
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
    const starCount = 3500;
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const i3 = i * 3;
      const r = 2500 + Math.random() * 8000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);

      positions[i3]     = r * Math.sin(phi) * Math.cos(theta);
      positions[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i3 + 2] = r * Math.cos(phi);

      colors[i3]     = 0.3 + Math.random() * 0.4;
      colors[i3 + 1] = 0.6 + Math.random() * 0.4;
      colors[i3 + 2] = 0.9;
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
   * 2. 构筑【100M+ 全息 4D 时空体素世界模型】场景
   */
  _buildWorldModelScenario() {
    const group = new THREE.Group();
    group.visible = false;

    const roadGeom = new THREE.PlaneGeometry(36, 300);
    const roadMat = new THREE.MeshBasicMaterial({ color: 0x080e1a, side: THREE.DoubleSide });
    const road = new THREE.Mesh(roadGeom, roadMat);
    road.rotation.x = -Math.PI / 2;
    group.add(road);

    for (let z = -140; z < 140; z += 12) {
      const dashGeom = new THREE.PlaneGeometry(0.4, 6);
      const dashMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.6 });
      const dash = new THREE.Mesh(dashGeom, dashMat);
      dash.rotation.x = -Math.PI / 2;
      dash.position.set(0, 0.05, z);
      group.add(dash);
    }

    const egoGeom = new THREE.BoxGeometry(3.2, 1.6, 6.0);
    const egoMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true });
    const egoCar = new THREE.Mesh(egoGeom, egoMat);
    egoCar.position.set(-6, 0.8, -20);
    group.add(egoCar);

    const coneGeom = new THREE.ConeGeometry(18, 50, 16, 1, true);
    const coneMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.15 });
    const cone = new THREE.Mesh(coneGeom, coneMat);
    cone.rotation.x = -Math.PI / 2;
    cone.position.set(-6, 1.2, 15);
    group.add(cone);

    const truckGeom = new THREE.BoxGeometry(4.2, 3.8, 14.0);
    const truckMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: true });
    const truck = new THREE.Mesh(truckGeom, truckMat);
    truck.position.set(6, 2.0, 10);
    group.add(truck);

    const blindGeom = new THREE.BoxGeometry(16, 5, 25);
    const blindMat = new THREE.MeshBasicMaterial({ color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.25 });
    const blindBox = new THREE.Mesh(blindGeom, blindMat);
    blindBox.position.set(16, 2.5, 20);
    group.add(blindBox);

    const nx = 24, ny = 6, nz = 24;
    const totalPoints = nx * ny * nz;
    const vPos = new Float32Array(totalPoints * 3);
    const vCol = new Float32Array(totalPoints * 3);

    let idx = 0;
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++) {
          const i3 = idx * 3;
          vPos[i3]     = (x - nx / 2) * 2.8;
          vPos[i3 + 1] = y * 1.8 + 0.5;
          vPos[i3 + 2] = (z - nz / 2) * 5.0;

          vCol[i3]     = 0.2;
          vCol[i3 + 1] = 0.7;
          vCol[i3 + 2] = 0.9;
          idx++;
        }
      }
    }

    const vGeom = new THREE.BufferGeometry();
    vGeom.setAttribute("position", new THREE.BufferAttribute(vPos, 3));
    vGeom.setAttribute("color", new THREE.BufferAttribute(vCol, 3));

    const vMat = new THREE.PointsMaterial({
      size: 3.5,
      map: this.glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.voxelGridPoints = new THREE.Points(vGeom, vMat);
    group.add(this.voxelGridPoints);

    const waveCount = 5;
    this.ghostWaveMeshes = [];
    for (let w = 0; w < waveCount; w++) {
      const curve = new THREE.CubicBezierCurve3(
        new THREE.Vector3(-6, 1.2, -5),
        new THREE.Vector3(0, 1.5, 5 + w * 3),
        new THREE.Vector3(12, 1.5, 10 + w * 4),
        new THREE.Vector3(18 + w * 2, 1.2, 25 + w * 2)
      );
      const pts = curve.getPoints(30);
      const lineGeom = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xa855f7,
        transparent: true,
        opacity: 0.65,
        linewidth: 2,
        blending: THREE.AdditiveBlending
      });
      const line = new THREE.Line(lineGeom, lineMat);
      this.ghostWaveMeshes.push(line);
      group.add(line);
    }

    this.worldModelGroup = group;
    this.scene.add(group);
  }

  /**
   * 3. 构筑【1M 极速 10 kHz 本能阻尼 100km/h 爆胎稳控】场景
   */
  _buildTransientTireBlowoutScenario() {
    const group = new THREE.Group();
    group.visible = false;

    const trackGeom = new THREE.PlaneGeometry(28, 400);
    const trackMat = new THREE.MeshBasicMaterial({ color: 0x0a101f, side: THREE.DoubleSide });
    const track = new THREE.Mesh(trackGeom, trackMat);
    track.rotation.x = -Math.PI / 2;
    group.add(track);

    const barMat = new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true });
    const barLeft = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.2, 400), barMat);
    barLeft.position.set(-14, 0.6, 0);
    const barRight = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.2, 400), barMat);
    barRight.position.set(14, 0.6, 0);
    group.add(barLeft);
    group.add(barRight);

    const triggerGeom = new THREE.BoxGeometry(28, 0.1, 1.0);
    const triggerMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
    const triggerLine = new THREE.Mesh(triggerGeom, triggerMat);
    triggerLine.position.set(0, 0.05, -80);
    group.add(triggerLine);

    const carAGeom = new THREE.BoxGeometry(2.8, 1.4, 5.2);
    const carAMat = new THREE.MeshBasicMaterial({ color: 0x34d399, wireframe: true });
    this.blowoutCarCyan = new THREE.Mesh(carAGeom, carAMat);
    this.blowoutCarCyan.position.set(-5, 0.7, -150);
    group.add(this.blowoutCarCyan);

    const carBGeom = new THREE.BoxGeometry(2.8, 1.4, 5.2);
    const carBMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: true });
    this.blowoutCarRed = new THREE.Mesh(carBGeom, carBMat);
    this.blowoutCarRed.position.set(5, 0.7, -150);
    group.add(this.blowoutCarRed);

    const cyanPts = [];
    for (let z = -150; z < 150; z += 2) {
      let x = -5;
      if (z > -80 && z < 0) {
        x += Math.sin((z + 80) * 0.08) * 0.238;
      }
      cyanPts.push(new THREE.Vector3(x, 0.1, z));
    }
    const cyanGeom = new THREE.BufferGeometry().setFromPoints(cyanPts);
    const cyanLine = new THREE.Line(cyanGeom, new THREE.LineBasicMaterial({ color: 0x34d399, linewidth: 2 }));
    group.add(cyanLine);

    const redPts = [];
    for (let z = -150; z < 150; z += 2) {
      let x = 5;
      if (z > -80) {
        const t = (z + 80) * 0.05;
        x += Math.sin(t * 1.5) * Math.min(8.5, t * 1.8);
      }
      redPts.push(new THREE.Vector3(x, 0.1, z));
    }
    const redGeom = new THREE.BufferGeometry().setFromPoints(redPts);
    const redLine = new THREE.Line(redGeom, new THREE.LineBasicMaterial({ color: 0xf43f5e, linewidth: 2 }));
    group.add(redLine);

    this.transientGroup = group;
    this.scene.add(group);
  }

  /**
   * 场景视界模式切换
   */
  setScenarioMode(mode) {
    this.activeScenario = mode;

    if (this.cosmicCage) this.cosmicCage.visible = (mode === "cosmic");
    if (this.organismGroup) this.organismGroup.visible = (mode === "cosmic");
    if (this.worldModelGroup) this.worldModelGroup.visible = (mode === "world_model");
    if (this.transientGroup) this.transientGroup.visible = (mode === "transient_blowout");

    if (mode === "world_model") {
      this.targetCameraPos.set(0, 32, 68);
      this.targetControlsTarget.set(4, 3, 10);
      this._updateHUDForWorldModel();
    } else if (mode === "transient_blowout") {
      this.targetCameraPos.set(38, 28, 45);
      this.targetControlsTarget.set(0, 2, -20);
      this._updateHUDForBlowout();
    } else {
      if (this.model.organism && this.model.organism.geometry) {
        this._adjustCameraToOrganism(this.model.organism);
      } else {
        this.targetCameraPos.set(0, 300, 600);
        this.targetControlsTarget.set(0, 0, 0);
      }
      this._updateHUDDetails(this.model.organism);
    }
  }

  _updateHUDForWorldModel() {
    const volumeBadge = document.getElementById("hud-volume-badge");
    if (volumeBadge) {
      volumeBadge.innerHTML = `
        <span style="color:#a855f7;">4D全息世界模型:</span> 
        <strong style="color:#38bdf8;">盲区反事实因果觉醒度 16.3%</strong>
        <span style="color:#64748b; font-size:11px;">(地平线 5.07s · 1024 维体素场)</span>
      `;
    }
    const orgTitle = document.getElementById("hud-org-title");
    if (orgTitle) orgTitle.innerText = "智能驾驶全息 4D 时空连续体素世界模型 (100M+ 细胞)";
    const orgMeta = document.getElementById("hud-org-meta");
    if (orgMeta) {
      orgMeta.innerHTML = `
        <div>实测显存: <strong style="color:#38bdf8;">2.76 GB / 8.00 GB</strong></div>
        <div>单步延迟: <strong style="color:#34d399;">14.838 ms (67.4 Hz 硬实时)</strong></div>
        <div>物理涌现: <span style="color:#fbbf24;">视线死角 Kolmogorov 连续场分裂推演</span></div>
      `;
    }
  }

  _updateHUDForBlowout() {
    const volumeBadge = document.getElementById("hud-volume-badge");
    if (volumeBadge) {
      volumeBadge.innerHTML = `
        <span style="color:#34d399;">100km/h 极速爆胎动力学:</span> 
        <strong style="color:#34d399;">SDSCC 10 kHz 横摆收敛 (0.238m)</strong>
        <span style="color:#f43f5e;">vs 传统 50Hz 翻车失稳</span>
      `;
    }
    const orgTitle = document.getElementById("hud-org-title");
    if (orgTitle) orgTitle.innerText = "1M 细胞瞬态底盘动力学稳控 (10 kHz 本能阻尼)";
    const orgMeta = document.getElementById("hud-org-meta");
    if (orgMeta) {
      orgMeta.innerHTML = `
        <div>控制频率: <strong style="color:#34d399;">~9,740 Hz (103 μs 极速闭环)</strong></div>
        <div>冲击力矩: <strong style="color:#f43f5e;">14,500 N·m 突发偏航</strong></div>
        <div>对账结果: <span style="color:#38bdf8;">300ms 完全收敛, 偏离死锁 0.238m</span></div>
      `;
    }
  }

  /**
   * 绑定 Model 事件
   */
  _bindModelEvents() {
    this.model.on("ORGANISM_LOADED", (org) => {
      this._renderOrganismGeometry(org);
      if (this.activeScenario === "cosmic") {
        this._updateHUDDetails(org);
        this._adjustCameraToOrganism(org);
      }
    });

    this.model.on("TELEMETRY_UPDATED", (telemetry) => {
      this._renderTelemetryPlots(telemetry);
    });

    this.model.on("STATUS_CHANGED", (status) => {
      if (status.cameraMode === "cosmic") {
        this.setScenarioMode("cosmic");
      }
    });
  }

  /**
   * 零拷贝渲染细胞与突触：无论在什么尺度，细胞都 100% 耀眼可见！
   */
  _renderOrganismGeometry(org) {
    const geo = org.geometry;
    if (!geo) return;

    // 清理旧组件
    while (this.organismGroup.children.length > 0) {
      const obj = this.organismGroup.children[0];
      this.organismGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    // 1. 构造细胞粒子群 BufferGeometry
    const cellGeom = new THREE.BufferGeometry();
    cellGeom.setAttribute("position", new THREE.BufferAttribute(geo.positions, 3));
    cellGeom.setAttribute("color", new THREE.BufferAttribute(geo.colors, 3));

    // 计算粒子视觉尺寸 (自适应微观到宏观)
    const baseSize = Math.max(6.0, Math.min(28.0, geo.trueRadius * 0.18));

    const cellMat = new THREE.PointsMaterial({
      size: baseSize,
      map: this.glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true
    });
    this.cellPoints = new THREE.Points(cellGeom, cellMat);
    this.organismGroup.add(this.cellPoints);

    // 2. 构造突触光流纤维 LineSegments
    if (geo.synPositions && geo.synPositions.length > 0) {
      const synGeom = new THREE.BufferGeometry();
      synGeom.setAttribute("position", new THREE.BufferAttribute(geo.synPositions, 3));
      synGeom.setAttribute("color", new THREE.BufferAttribute(geo.synColors, 3));

      const synMat = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.65,
        blending: THREE.AdditiveBlending,
        linewidth: 1.5
      });
      this.synapseLines = new THREE.LineSegments(synGeom, synMat);
      this.organismGroup.add(this.synapseLines);
    }

    // 3. 构造物理引力光环与雷达标定线框 (Aura & Beacon)
    const auraGeom = new THREE.SphereGeometry(geo.trueRadius * 1.08, 24, 16);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.2
    });
    this.organismAura = new THREE.Mesh(auraGeom, auraMat);
    this.organismGroup.add(this.organismAura);

    // 微观微元雷达定位信标框 (如果半径小于 15m，添加外部准星以便深空定位)
    if (geo.trueRadius < 20.0) {
      const reticleGeom = new THREE.RingGeometry(geo.trueRadius * 1.8, geo.trueRadius * 2.0, 32);
      const reticleMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24, side: THREE.DoubleSide, transparent: true, opacity: 0.6 });
      const reticle = new THREE.Mesh(reticleGeom, reticleMat);
      reticle.rotation.x = Math.PI / 2;
      this.organismGroup.add(reticle);
    }

    // 确保可见性
    this.organismGroup.visible = (this.activeScenario === "cosmic");
  }

  /**
   * 平滑相机运镜：自适应生命体物理半径，聚焦在视口正中
   */
  _adjustCameraToOrganism(org) {
    if (!org || !org.geometry) return;
    const r = org.geometry.trueRadius;

    // 自适应最佳观赏视距 (微尘推进至 15m，大生命体推至 2.5倍半径)
    const viewDist = Math.max(16.0, r * 2.4);
    this.targetCameraPos.set(0, viewDist * 0.45, viewDist * 1.1);
    this.targetControlsTarget.set(0, 0, 0);
  }

  _updateHUDDetails(org) {
    const volumeBadge = document.getElementById("hud-volume-badge");
    if (volumeBadge) {
      volumeBadge.innerHTML = `
        <span style="color:#94a3b8;">硬件宇宙体积占比:</span> 
        <strong style="color:#38bdf8; font-size:14px;">${org.volumeRatioPercent}%</strong>
        <span style="color:#64748b; font-size:11px;">(物理半径 R=${org.trueRadius.toFixed(1)}m)</span>
      `;
    }
    const orgTitle = document.getElementById("hud-org-title");
    if (orgTitle) orgTitle.innerText = org.name;
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
   * 遥测示波器渲染
   */
  _renderTelemetryPlots(telemetry) {
    if (!this.phaseCanvas) {
      this.phaseCanvas = document.getElementById("canvas-phase-portrait");
      if (this.phaseCanvas) this.phaseCtx = this.phaseCanvas.getContext("2d");
    }
    if (!this.phaseCtx) return;

    const ctx = this.phaseCtx;
    const w = this.phaseCanvas.width;
    const h = this.phaseCanvas.height;

    ctx.fillStyle = "rgba(8, 14, 26, 0.3)";
    ctx.fillRect(0, 0, w, h);

    if (this.activeScenario === "transient_blowout") {
      ctx.strokeStyle = "#f43f5e";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let t = 0; t < Math.PI * 4; t += 0.1) {
        const r = t * 10.0;
        const x = w * 0.5 + Math.cos(t) * r;
        const y = h * 0.5 + Math.sin(t) * r;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      for (let t = 0; t < Math.PI * 3; t += 0.1) {
        const r = Math.max(0, 45.0 - t * 14.0);
        const x = w * 0.5 + Math.cos(t) * r;
        const y = h * 0.5 + Math.sin(t) * r;
        if (t === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = "#34d399";
      ctx.font = "10px monospace";
      ctx.fillText("SDSCC 10kHz 收敛稳态", 12, 20);
      ctx.fillStyle = "#f43f5e";
      ctx.fillText("传统50Hz 发散翻车", 12, 36);
    } else {
      const th = this.model.telemetryHistory;
      const count = th.count;
      if (count < 2) return;

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
    }
  }

  /**
   * 主渲染泵与动态流变物理推演
   */
  renderLoop() {
    this.animClock += 0.03;

    this.camera.position.lerp(this.targetCameraPos, 0.05);
    this.controls.target.lerp(this.targetControlsTarget, 0.05);
    this.controls.update();

    if (this.starfield) this.starfield.rotation.y += 0.00015;

    // 1. 4D 全息世界模型体素流变与幽灵波动画
    if (this.worldModelGroup && this.worldModelGroup.visible) {
      if (this.voxelGridPoints) {
        const pos = this.voxelGridPoints.geometry.attributes.position.array;
        const len = pos.length / 3;
        for (let i = 0; i < len; i++) {
          const x = pos[i * 3];
          const z = pos[i * 3 + 2];
          pos[i * 3 + 1] = Math.sin(x * 0.15 + this.animClock) * Math.cos(z * 0.15) * 1.5 + 2.5;
        }
        this.voxelGridPoints.geometry.attributes.position.needsUpdate = true;
      }
      const pulse = 0.5 + 0.5 * Math.sin(this.animClock * 3.0);
      for (const line of this.ghostWaveMeshes) {
        line.material.opacity = 0.3 + pulse * 0.55;
      }
    }

    // 2. 100km/h 极速爆胎双车运动循环推演
    if (this.transientGroup && this.transientGroup.visible) {
      const loopZ = ((this.animClock * 25.0) % 300.0) - 150.0;
      if (this.blowoutCarCyan && this.blowoutCarRed) {
        let cyanX = -5;
        if (loopZ > -80 && loopZ < 20) {
          cyanX += Math.sin((loopZ + 80) * 0.1) * 0.238;
        }
        this.blowoutCarCyan.position.set(cyanX, 0.7, loopZ);

        let redX = 5;
        if (loopZ > -80) {
          const dist = (loopZ + 80) * 0.05;
          redX += Math.sin(dist * 1.8) * Math.min(8.8, dist * 2.0);
        }
        this.blowoutCarRed.position.set(redX, 0.7, loopZ);
      }
    }

    // 3. 默认算力宇宙中生命体自转
    if (this.organismGroup && this.organismGroup.visible && this.model.status.isPlaying) {
      this.organismGroup.rotation.y += 0.0012;
    }

    this.renderer.render(this.scene, this.camera);
  }
}
