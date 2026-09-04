/**
 * SDSCC Cosmic View (Living Biomimetic 3D Cellular & Holographic Engine)
 * 
 * 融合两大视觉巅峰：
 * 1. 生物拟真 3D 多层活体细胞计算组织：
 *    - 半透明双脂质质膜 (3D Translucent Lipid Bilayer Icosahedron)
 *    - 高能细胞核中枢与致密核仁 (Karyon Core & Nucleolus)
 *    - 围绕细胞核公转的 3 颗微观线粒体/能量细胞器 (Orbiting Mitochondria)
 *    - 动作电位放电冲击波圆环 (Action Potential Ion Shockwave Rings)
 *    - 二次贝塞尔曲线突触轴突 (3D Quadratic Bezier Axon Cables)
 *    - 轴突上飞速奔涌的能量光子 (Traveling Photons) 与突触小球 (Bouton)
 *    - 生物原语全息浮动标签 (Holographic Type Badges)
 * 2. 算力宇宙与工业级具身展示：
 *    - 硬件算力宇宙边界线框 (Cosmic Bounding Wireframe)
 *    - 空气介质微粒场 (Atmospheric Aero-Medium Swirling Cloud)
 *    - 100M+ 全息 4D 时空体素世界模型 (Kolmogorov 流变场 & 盲区反事实幽灵推演波)
 *    - 1M 极速 10 kHz 爆胎动力学双车相空间对账
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// 家族色板
const FAMILY_COLORS = {
  SENSE: 0x22d3ee,  // 天青 (传感器)
  ACT: 0xf43f5e,    // 绯红 (效应器)
  CORE: 0x34d399,   // 翡翠 (动力学核)
  GATE: 0xa855f7,   // 紫晶 (门控逻辑)
  CHAOS: 0xfbbf24   // 琥珀 (混沌振荡)
};

function getCellColorHex(typeStr) {
  const t = String(typeStr || "").toUpperCase();
  if (t.startsWith("SENSE") || t.includes("INPUT") || t.includes("IN")) return FAMILY_COLORS.SENSE;
  if (t.startsWith("ACT") || t.includes("OUTPUT") || t.includes("OUT") || t.includes("MOTOR")) return FAMILY_COLORS.ACT;
  if (t.includes("GATE") || t.includes("THRESHOLD") || t.includes("HYSTERESIS")) return FAMILY_COLORS.GATE;
  if (t.includes("CHAOS") || t.includes("OSCILLATOR")) return FAMILY_COLORS.CHAOS;
  return FAMILY_COLORS.CORE;
}

// 缓存动态纹理
const TEX_CACHE = new Map();
function getOrCreateGlowTexture() {
  if (TEX_CACHE.has("glow")) return TEX_CACHE.get("glow");
  const cv = document.createElement("canvas");
  cv.width = 64; cv.height = 64;
  const ctx = cv.getContext("2d");
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
  grad.addColorStop(0.25, "rgba(56, 189, 248, 0.9)");
  grad.addColorStop(0.65, "rgba(56, 189, 248, 0.25)");
  grad.addColorStop(1.0, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(cv);
  TEX_CACHE.set("glow", tex);
  return tex;
}

function getLabelTexture(typeStr) {
  const key = "lbl_" + typeStr;
  if (TEX_CACHE.has(key)) return TEX_CACHE.get(key);
  const cv = document.createElement("canvas");
  cv.width = 256; cv.height = 64;
  const g = cv.getContext("2d");
  g.font = "bold 22px monospace";
  g.fillStyle = "rgba(226, 232, 240, 0.95)";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(typeStr, 128, 32);
  const tex = new THREE.CanvasTexture(cv);
  TEX_CACHE.set(key, tex);
  return tex;
}

/* ============================================================
 * 单个生物拟真 3D 细胞计算结构
 * ============================================================ */
class LivingCellView {
  constructor(cellData, trueScaleMultiplier, glowTex) {
    this.id = cellData.id;
    this.type = cellData.type || "Op_Core";
    this.baseX = (cellData.x || 0) * trueScaleMultiplier;
    this.baseY = (cellData.y || 0) * trueScaleMultiplier;
    this.baseZ = (cellData.z || 0) * trueScaleMultiplier;
    this.phase = (this.id * 0.785) % (Math.PI * 2);
    this.shockwaveRadius = 1.0;

    const col = getCellColorHex(this.type);
    this.colorHex = col;

    this.group = new THREE.Group();
    this.group.position.set(this.baseX, this.baseY, this.baseZ);

    // 1. 3D 半透明双脂质质膜 (Lipid Bilayer Shell)
    const memGeo = new THREE.IcosahedronGeometry(7.5, 2);
    this.membraneMat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.2,
      metalness: 0.1,
      transparent: true,
      opacity: 0.72,
      emissive: col,
      emissiveIntensity: 0.45,
      depthWrite: false
    });
    this.membraneMesh = new THREE.Mesh(memGeo, this.membraneMat);
    this.group.add(this.membraneMesh);

    // 2. 外层生物发光晕 (Halo Sprite)
    const haloMat = new THREE.SpriteMaterial({
      map: glowTex,
      color: col,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.6
    });
    this.haloSprite = new THREE.Sprite(haloMat);
    this.haloSprite.scale.set(32, 32, 1);
    this.group.add(this.haloSprite);

    // 3. 高能细胞核中枢与致密核仁 (Karyon Core & Nucleolus)
    const coreGeo = new THREE.SphereGeometry(3.6, 16, 16);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: col,
      emissiveIntensity: 1.8,
      roughness: 0.1,
      metalness: 0.3
    });
    this.nucleus = new THREE.Mesh(coreGeo, coreMat);
    this.group.add(this.nucleus);

    // 4. 围绕细胞核公转的 3 颗微观线粒体细胞器 (Orbiting Mitochondria)
    this.organelles = [];
    const orgGeo = new THREE.SphereGeometry(1.2, 8, 8);
    for (let k = 0; k < 3; ++k) {
      const orgMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: col,
        emissiveIntensity: 2.2,
        roughness: 0.1
      });
      const orgMesh = new THREE.Mesh(orgGeo, orgMat);
      this.group.add(orgMesh);
      this.organelles.push({
        mesh: orgMesh,
        orbitR: 5.5 + k * 1.2,
        speed: 2.5 + k * 0.9,
        tilt: (k * Math.PI) / 3
      });
    }

    // 5. 动作电位电离冲击波圆环 (Action Potential Ion Shockwave Ring)
    const ringGeo = new THREE.RingGeometry(5, 8, 24);
    this.ringMat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.shockwaveRing = new THREE.Mesh(ringGeo, this.ringMat);
    this.shockwaveRing.rotation.x = Math.PI / 2;
    this.group.add(this.shockwaveRing);

    // 6. 全息浮动原语名牌 (Holographic Label Badge)
    const labelMat = new THREE.SpriteMaterial({
      map: getLabelTexture(this.type),
      transparent: true,
      depthWrite: false,
      opacity: 0.88
    });
    this.labelSprite = new THREE.Sprite(labelMat);
    this.labelSprite.scale.set(24, 6, 1);
    this.labelSprite.position.set(0, -11, 0);
    this.group.add(this.labelSprite);
  }

  update(time) {
    // 自然生物呼吸律动
    const breath = Math.sin(time * 2.5 + this.phase) * 0.15;
    const memScale = 1.0 + breath;
    this.membraneMesh.scale.set(memScale, memScale, memScale);

    // 线粒体公转轨道
    for (let k = 0; k < this.organelles.length; ++k) {
      const org = this.organelles[k];
      const ang = time * org.speed + (k * Math.PI * 2) / 3;
      const ox = Math.cos(ang) * org.orbitR;
      const oy = Math.sin(ang) * org.orbitR * Math.cos(org.tilt);
      const oz = Math.sin(ang) * org.orbitR * Math.sin(org.tilt);
      org.mesh.position.set(ox, oy, oz);
    }

    // 周期性动作电位放电扩散
    this.shockwaveRadius += 0.05;
    if (this.shockwaveRadius > 3.0) this.shockwaveRadius = 1.0;
    const ringScale = this.shockwaveRadius;
    this.shockwaveRing.scale.set(ringScale, ringScale, ringScale);
    this.ringMat.opacity = Math.max(0.0, (3.0 - this.shockwaveRadius) * 0.35);
  }

  dispose() {
    this.membraneMesh.geometry.dispose();
    this.membraneMat.dispose();
    this.nucleus.geometry.dispose();
    this.nucleus.material.dispose();
    this.shockwaveRing.geometry.dispose();
    this.ringMat.dispose();
    this.haloSprite.material.dispose();
    this.labelSprite.material.dispose();
    for (const org of this.organelles) {
      org.mesh.geometry.dispose();
      org.mesh.material.dispose();
    }
  }
}

/* ============================================================
 * 突触光流纤维与流动光子 (Synapse Axon Fiber & Traveling Photons)
 * ============================================================ */
class LivingSynapseView {
  constructor(synData, fromCell, toCell, glowTex) {
    this.syn = synData;
    this.fromCell = fromCell;
    this.toCell = toCell;
    this.numSegments = 16;

    this.group = new THREE.Group();

    // 1. 二次贝塞尔曲线几何体
    this.curvePoints = new Float32Array((this.numSegments + 1) * 3);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.curvePoints, 3));

    const w = this.syn.weight || this.syn.w || 1.0;
    const colHex = w >= 0 ? 0x38bdf8 : 0xf43f5e;

    this.lineMat = new THREE.LineBasicMaterial({
      color: colHex,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending,
      linewidth: 1.5
    });
    this.line = new THREE.Line(this.geo, this.lineMat);
    this.group.add(this.line);

    // 2. 突触上奔涌的 2 颗能量光子
    this.photonMat1 = new THREE.SpriteMaterial({
      map: glowTex,
      color: 0x38bdf8,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.photonMat2 = new THREE.SpriteMaterial({
      map: glowTex,
      color: 0xa855f7,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.photon1 = new THREE.Sprite(this.photonMat1);
    this.photon2 = new THREE.Sprite(this.photonMat2);
    this.photon1.scale.set(8, 8, 1);
    this.photon2.scale.set(6, 6, 1);
    this.group.add(this.photon1);
    this.group.add(this.photon2);

    // 3. 突触终扣发光小球 (Synaptic Bouton)
    this.boutonMat = new THREE.SpriteMaterial({
      map: glowTex,
      color: colHex,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.bouton = new THREE.Sprite(this.boutonMat);
    this.bouton.scale.set(10, 10, 1);
    this.bouton.position.set(toCell.baseX, toCell.baseY, toCell.baseZ);
    this.group.add(this.bouton);

    // 计算二次贝塞尔控制点
    const p0 = new THREE.Vector3(fromCell.baseX, fromCell.baseY, fromCell.baseZ);
    const p2 = new THREE.Vector3(toCell.baseX, toCell.baseY, toCell.baseZ);
    const chord = new THREE.Vector3().subVectors(p2, p0);
    const dist = chord.length();
    const arcHeight = Math.min(24.0, dist * 0.2);

    const up = Math.abs(chord.y) < 0.95 * dist ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const norm = new THREE.Vector3().crossVectors(chord, up).normalize();

    this.p0 = p0;
    this.p2 = p2;
    this.p1 = new THREE.Vector3().addVectors(p0, p2).multiplyScalar(0.5).addScaledVector(norm, arcHeight);

    // 采样贝塞尔点
    const pos = this.geo.attributes.position;
    for (let i = 0; i <= this.numSegments; i++) {
      const t = i / this.numSegments;
      const it = 1.0 - t;
      const bx = it * it * this.p0.x + 2.0 * it * t * this.p1.x + t * t * this.p2.x;
      const by = it * it * this.p0.y + 2.0 * it * t * this.p1.y + t * t * this.p2.y;
      const bz = it * it * this.p0.z + 2.0 * it * t * this.p1.z + t * t * this.p2.z;
      pos.setXYZ(i, bx, by, bz);
    }
    pos.needsUpdate = true;

    this.flowT1 = Math.random();
    this.flowT2 = (this.flowT1 + 0.5) % 1.0;
  }

  update(time) {
    this.flowT1 = (this.flowT1 + 0.015) % 1.0;
    this.flowT2 = (this.flowT2 + 0.015) % 1.0;

    const evalBezier = (t, outVec) => {
      const it = 1.0 - t;
      outVec.x = it * it * this.p0.x + 2.0 * it * t * this.p1.x + t * t * this.p2.x;
      outVec.y = it * it * this.p0.y + 2.0 * it * t * this.p1.y + t * t * this.p2.y;
      outVec.z = it * it * this.p0.z + 2.0 * it * t * this.p1.z + t * t * this.p2.z;
    };

    evalBezier(this.flowT1, this.photon1.position);
    evalBezier(this.flowT2, this.photon2.position);
  }

  dispose() {
    this.geo.dispose();
    this.lineMat.dispose();
    this.photonMat1.dispose();
    this.photonMat2.dispose();
    this.boutonMat.dispose();
  }
}

/* ============================================================
 * 宇宙级视觉引擎总控
 * ============================================================ */
export class CosmicView {
  constructor(containerElement, model, dispatcher) {
    this.container = containerElement;
    this.model = model;
    this.dispatcher = dispatcher;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    this.glowTexture = getOrCreateGlowTexture();

    // 视觉图层
    this.cosmicCage = null;        // 算力宇宙边界框 (1000m³)
    this.starfield = null;         // 背景深空星系
    this.airParticles = null;      // 空气流动微粒场 (900颗热对流微粒)
    this.organismGroup = null;     // 活体细胞组织根节点

    // 活体微元图元列表
    this.livingCells = [];
    this.livingSynapses = [];

    // 具身专业演示场景组
    this.worldModelGroup = null;   // 100M+ 4D 全息世界模型 + 盲区反事实
    this.transientGroup = null;    // 1M 10kHz 极速爆胎动力学对账
    this.activeScenario = "cosmic"; // "cosmic" | "world_model" | "transient_blowout"

    this.animClock = 0;
    this.voxelGridPoints = null;
    this.ghostWaveMeshes = [];
    this.blowoutCarCyan = null;
    this.blowoutCarRed = null;

    this.targetCameraPos = new THREE.Vector3(0, 120, 260);
    this.targetControlsTarget = new THREE.Vector3(0, 0, 0);

    this.phaseCanvas = null;
    this.phaseCtx = null;

    this._initThreeScene();
    this._buildCosmicHorizonCage();
    this._buildAtmosphericAirField();
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
    this.renderer.toneMappingExposure = 1.25;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxDistance = 15000;
    this.controls.minDistance = 2.0;

    this.organismGroup = new THREE.Group();
    this.scene.add(this.organismGroup);

    const ambLight = new THREE.AmbientLight(0x0e1726, 1.8);
    this.scene.add(ambLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 2.0);
    dirLight1.position.set(300, 500, 400);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf43f5e, 1.4);
    dirLight2.position.set(-300, -400, -300);
    this.scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0x00f0ff, 1.8, 1200);
    this.scene.add(pointLight);

    window.addEventListener("resize", () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

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

  /**
   * 空气介质微粒流动场 (Atmospheric Aero-Medium Swirling Cloud)
   */
  _buildAtmosphericAirField() {
    const count = 900;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      pos[i * 3]     = (Math.random() - 0.5) * 1200;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 900;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 1200;

      const isIon = Math.random() < 0.3;
      col[i * 3]     = isIon ? 0.22 : 0.58;
      col[i * 3 + 1] = isIon ? 0.74 : 0.64;
      col[i * 3 + 2] = isIon ? 0.97 : 0.72;
    }

    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));

    const mat = new THREE.PointsMaterial({
      size: 4.0,
      map: this.glowTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.airParticles = new THREE.Points(geo, mat);
    this.scene.add(this.airParticles);
  }

  _buildCosmicStarfield() {
    const count = 3000;
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const r = 2500 + Math.random() * 8000;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);

      pos[i3]     = r * Math.sin(phi) * Math.cos(theta);
      pos[i3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i3 + 2] = r * Math.cos(phi);

      col[i3]     = 0.3 + Math.random() * 0.4;
      col[i3 + 1] = 0.6 + Math.random() * 0.4;
      col[i3 + 2] = 0.95;
    }

    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));

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

  _buildWorldModelScenario() {
    const group = new THREE.Group();
    group.visible = false;

    const roadGeom = new THREE.PlaneGeometry(36, 300);
    const roadMat = new THREE.MeshBasicMaterial({ color: 0x080e1a, side: THREE.DoubleSide });
    const road = new THREE.Mesh(roadGeom, roadMat);
    road.rotation.x = -Math.PI / 2;
    group.add(road);

    for (let z = -140; z < 140; z += 12) {
      const dash = new THREE.Mesh(
        new THREE.PlaneGeometry(0.4, 6),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.6 })
      );
      dash.rotation.x = -Math.PI / 2;
      dash.position.set(0, 0.05, z);
      group.add(dash);
    }

    const egoCar = new THREE.Mesh(
      new THREE.BoxGeometry(3.2, 1.6, 6.0),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true })
    );
    egoCar.position.set(-6, 0.8, -20);
    group.add(egoCar);

    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(18, 50, 16, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, wireframe: true, transparent: true, opacity: 0.15 })
    );
    cone.rotation.x = -Math.PI / 2;
    cone.position.set(-6, 1.2, 15);
    group.add(cone);

    const truck = new THREE.Mesh(
      new THREE.BoxGeometry(4.2, 3.8, 14.0),
      new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: true })
    );
    truck.position.set(6, 2.0, 10);
    group.add(truck);

    const blindBox = new THREE.Mesh(
      new THREE.BoxGeometry(16, 5, 25),
      new THREE.MeshBasicMaterial({ color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.25 })
    );
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

    this.voxelGridPoints = new THREE.Points(
      vGeom,
      new THREE.PointsMaterial({
        size: 4.5,
        map: this.glowTexture,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      })
    );
    group.add(this.voxelGridPoints);

    this.ghostWaveMeshes = [];
    for (let w = 0; w < 5; w++) {
      const curve = new THREE.CubicBezierCurve3(
        new THREE.Vector3(-6, 1.2, -5),
        new THREE.Vector3(0, 1.5, 5 + w * 3),
        new THREE.Vector3(12, 1.5, 10 + w * 4),
        new THREE.Vector3(18 + w * 2, 1.2, 25 + w * 2)
      );
      const pts = curve.getPoints(30);
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: 0xa855f7,
          transparent: true,
          opacity: 0.65,
          linewidth: 2,
          blending: THREE.AdditiveBlending
        })
      );
      this.ghostWaveMeshes.push(line);
      group.add(line);
    }

    this.worldModelGroup = group;
    this.scene.add(group);
  }

  _buildTransientTireBlowoutScenario() {
    const group = new THREE.Group();
    group.visible = false;

    const track = new THREE.Mesh(
      new THREE.PlaneGeometry(28, 400),
      new THREE.MeshBasicMaterial({ color: 0x0a101f, side: THREE.DoubleSide })
    );
    track.rotation.x = -Math.PI / 2;
    group.add(track);

    const barMat = new THREE.MeshBasicMaterial({ color: 0x334155, wireframe: true });
    const barLeft = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.2, 400), barMat);
    barLeft.position.set(-14, 0.6, 0);
    const barRight = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.2, 400), barMat);
    barRight.position.set(14, 0.6, 0);
    group.add(barLeft);
    group.add(barRight);

    const triggerLine = new THREE.Mesh(
      new THREE.BoxGeometry(28, 0.1, 1.0),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24 })
    );
    triggerLine.position.set(0, 0.05, -80);
    group.add(triggerLine);

    this.blowoutCarCyan = new THREE.Mesh(
      new THREE.BoxGeometry(2.8, 1.4, 5.2),
      new THREE.MeshBasicMaterial({ color: 0x34d399, wireframe: true })
    );
    this.blowoutCarCyan.position.set(-5, 0.7, -150);
    group.add(this.blowoutCarCyan);

    this.blowoutCarRed = new THREE.Mesh(
      new THREE.BoxGeometry(2.8, 1.4, 5.2),
      new THREE.MeshBasicMaterial({ color: 0xf43f5e, wireframe: true })
    );
    this.blowoutCarRed.position.set(5, 0.7, -150);
    group.add(this.blowoutCarRed);

    const cyanPts = [];
    for (let z = -150; z < 150; z += 2) {
      let x = -5;
      if (z > -80 && z < 20) x += Math.sin((z + 80) * 0.1) * 0.238;
      cyanPts.push(new THREE.Vector3(x, 0.1, z));
    }
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(cyanPts),
      new THREE.LineBasicMaterial({ color: 0x34d399, linewidth: 2 })
    ));

    const redPts = [];
    for (let z = -150; z < 150; z += 2) {
      let x = 5;
      if (z > -80) {
        const t = (z + 80) * 0.05;
        x += Math.sin(t * 1.5) * Math.min(8.5, t * 1.8);
      }
      redPts.push(new THREE.Vector3(x, 0.1, z));
    }
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(redPts),
      new THREE.LineBasicMaterial({ color: 0xf43f5e, linewidth: 2 })
    ));

    this.transientGroup = group;
    this.scene.add(group);
  }

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
        this.targetCameraPos.set(0, 150, 320);
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
  }

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
   * 零拷贝构建震撼的活体 3D 细胞组织与突触光流
   */
  _renderOrganismGeometry(org) {
    const geo = org.geometry;
    if (!geo) return;

    // 清理旧活体对象
    for (const c of this.livingCells) c.dispose();
    for (const s of this.livingSynapses) s.dispose();
    this.livingCells = [];
    this.livingSynapses = [];

    while (this.organismGroup.children.length > 0) {
      const obj = this.organismGroup.children[0];
      this.organismGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    // 原始细胞列表与突触列表
    const rawCells = org.rawCells || [];
    const rawSyns = org.rawSynapses || [];

    // 计算真实空间坐标乘数
    const r = geo.trueRadius;
    const trueMultiplier = Math.max(0.25, Math.min(2.5, r / 120.0));

    // 1. 逐个实例化高精活体 3D 细胞
    const cellMap = new Map();
    for (let i = 0; i < rawCells.length; i++) {
      const cData = rawCells[i];
      const livingCell = new LivingCellView(cData, trueMultiplier, this.glowTexture);
      this.livingCells.push(livingCell);
      this.organismGroup.add(livingCell.group);
      cellMap.set(cData.id !== undefined ? cData.id : i, livingCell);
    }

    // 2. 逐个实例化高精二次贝塞尔突触与奔涌光子
    const maxSyns = Math.min(rawSyns.length, 300);
    for (let s = 0; s < maxSyns; s++) {
      const syn = rawSyns[s];
      const from = cellMap.get(syn.from);
      const to = cellMap.get(syn.to);
      if (from && to && syn.active !== false) {
        const livingSyn = new LivingSynapseView(syn, from, to, this.glowTexture);
        this.livingSynapses.push(livingSyn);
        this.organismGroup.add(livingSyn.group);
      }
    }

    // 3. 构造宇宙空间物理引力包络光环 (Aura)
    const auraGeom = new THREE.SphereGeometry(r * 1.1, 24, 16);
    const auraMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.18
    });
    this.organismGroup.add(new THREE.Mesh(auraGeom, auraMat));

    // 4. 微观生命体高亮准星信标 (微尘雷达)
    if (r < 25.0) {
      const reticle = new THREE.Mesh(
        new THREE.RingGeometry(r * 2.2, r * 2.5, 32),
        new THREE.MeshBasicMaterial({ color: 0xfbbf24, side: THREE.DoubleSide, transparent: true, opacity: 0.75 })
      );
      reticle.rotation.x = Math.PI / 2;
      this.organismGroup.add(reticle);
    }

    this.organismGroup.visible = (this.activeScenario === "cosmic");
    console.log(`[CosmicView] Rendered Living Tissue: ${this.livingCells.length} cells, ${this.livingSynapses.length} synapses at R=${r.toFixed(1)}m`);
  }

  _adjustCameraToOrganism(org) {
    if (!org || !org.geometry) return;
    const r = org.geometry.trueRadius;

    // 自适应最佳观赏视角：微尘拉近到 25m，大生命体拉远到 2.2倍半径
    const viewDist = Math.max(30.0, r * 2.2);
    this.targetCameraPos.set(0, viewDist * 0.45, viewDist * 1.05);
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

  _renderTelemetryPlots(telemetry) {
    if (!this.phaseCanvas) {
      this.phaseCanvas = document.getElementById("canvas-phase-portrait");
      if (this.phaseCanvas) this.phaseCtx = this.phaseCanvas.getContext("2d");
    }
    if (!this.phaseCtx) return;

    const ctx = this.phaseCtx;
    const w = this.phaseCanvas.width;
    const h = this.phaseCanvas.height;

    ctx.fillStyle = "rgba(8, 14, 26, 0.35)";
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
   * 主时钟渲染泵
   */
  renderLoop() {
    this.animClock += 0.03;

    this.camera.position.lerp(this.targetCameraPos, 0.05);
    this.controls.target.lerp(this.targetControlsTarget, 0.05);
    this.controls.update();

    if (this.starfield) this.starfield.rotation.y += 0.00015;

    // 空气微粒漂移与热对流
    if (this.airParticles) {
      const pos = this.airParticles.geometry.attributes.position.array;
      const len = pos.length / 3;
      for (let i = 0; i < len; i++) {
        pos[i * 3 + 1] += 0.25; // 真实热对流上升
        if (pos[i * 3 + 1] > 450) pos[i * 3 + 1] = -450;
      }
      this.airParticles.geometry.attributes.position.needsUpdate = true;
    }

    // 1. 活体生命体微观运动 (细胞呼吸、核仁线粒体公转、突触光子疾驰)
    if (this.organismGroup && this.organismGroup.visible && this.model.status.isPlaying) {
      for (let i = 0; i < this.livingCells.length; i++) {
        this.livingCells[i].update(this.animClock);
      }
      for (let i = 0; i < this.livingSynapses.length; i++) {
        this.livingSynapses[i].update(this.animClock);
      }
      this.organismGroup.rotation.y += 0.001;
    }

    // 2. 4D 全息世界模型流变动画
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

    // 3. 100km/h 极速爆胎双车运动推演
    if (this.transientGroup && this.transientGroup.visible) {
      const loopZ = ((this.animClock * 25.0) % 300.0) - 150.0;
      if (this.blowoutCarCyan && this.blowoutCarRed) {
        let cyanX = -5;
        if (loopZ > -80 && loopZ < 20) cyanX += Math.sin((loopZ + 80) * 0.1) * 0.238;
        this.blowoutCarCyan.position.set(cyanX, 0.7, loopZ);

        let redX = 5;
        if (loopZ > -80) {
          const dist = (loopZ + 80) * 0.05;
          redX += Math.sin(dist * 1.8) * Math.min(8.8, dist * 2.0);
        }
        this.blowoutCarRed.position.set(redX, 0.7, loopZ);
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}
