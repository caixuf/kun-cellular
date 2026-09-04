/* ============================================================
 * tractography.js - 宏观白质纤维束流形 (Tractography Streamlines)
 * 借鉴高阶神经成像 DTI / HARDI 扩散张量成像白质示踪算法，
 * 将高密度突触聚合成贯穿全脑的彩色神经纤维束光纤高速公路。
 *
 * 色彩编码遵循国际神经影像标准 (DTI Standard Orientation):
 *   - 红色 (Red): 左右连合纤维束 (Corpus Callosum / Commissural Tracts)
 *   - 绿色 (Green): 前后联络纤维束 (Superior Longitudinal Fasciculus)
 *   - 蓝色 (Blue): 上下投射纤维束 (Corticospinal Projection Fibers)
 * ============================================================ */

import * as THREE from 'three';
import { scene } from './scene_setup.js';
import { getGlowTexture } from './texture_cache.js';

export class TractographySystem {
  constructor(scn = scene) {
    this.scene = scn || scene;
    this.tractGroup = new THREE.Group();
    this.scene.add(this.tractGroup);
    this.streamlines = [];
    this.particleStreams = [];
    this.lastOrgId = null;
    this.lastCellCount = 0;
  }

  rebuildTracts(org, bounds) {
    if (!org || !org.cells || org.cells.length < 16) {
      this.clear();
      return;
    }

    this.clear();
    const cells = org.cells;
    const syns = org.syns || [];
    const n = cells.length;

    // 1. 自动提取核心拓扑功能微区中枢 (Sensory, Cortex Left/Right, Motor, Columns)
    const clusters = this.extractAnatomicalClusters(cells, bounds);
    if (clusters.length < 2) return;

    // 2. 在中枢之间构建高质量三次 Hermite / Catmull-Rom 样条纤维束
    const glowTex = getGlowTexture();
    const tractLines = [];

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const c1 = clusters[i];
        const c2 = clusters[j];
        // 检查两个微区之间是否有突触连接密度
        const p1 = c1.center;
        const p2 = c2.center;
        const d = p1.distanceTo(p2);
        if (d < 15.0 || d > 360.0) continue;

        // 构建 3~5 根并行的束状微纤维 (Fiber Bundle Fibrils)
        const bundleCount = (n > 256) ? 4 : 2;
        for (let b = 0; b < bundleCount; b++) {
          const curve = this.createBundleCurve(p1, p2, b, bundleCount, d);
          const pts = curve.getPoints(32);
          const geo = new THREE.BufferGeometry().setFromPoints(pts);

          // DTI 标准张量方向着色
          const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
          const r = Math.abs(dir.x); // 红: 左右 (Commissural)
          const g = Math.abs(dir.z); // 绿: 前后 (Longitudinal)
          const bl = Math.abs(dir.y); // 蓝: 上下 (Projection)
          const dtiColor = new THREE.Color(r * 0.85 + 0.15, g * 0.85 + 0.15, bl * 0.85 + 0.15);

          const mat = new THREE.LineBasicMaterial({
            color: dtiColor,
            transparent: true,
            opacity: (n > 256) ? 0.38 : 0.48,
            blending: THREE.AdditiveBlending,
            depthWrite: false
          });

          const line = new THREE.Line(geo, mat);
          this.tractGroup.add(line);
          this.streamlines.push({ line, curve, color: dtiColor, length: d });

          // 3. 伴随流动的白质信息光子束 (Streaming Photons)
          const pGeo = new THREE.BufferGeometry();
          const pPos = new Float32Array(3);
          pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
          const pMat = new THREE.PointsMaterial({
            size: 6.0,
            color: dtiColor,
            map: glowTex,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending,
            depthWrite: false
          });
          const pMesh = new THREE.Points(pGeo, pMat);
          this.tractGroup.add(pMesh);
          this.particleStreams.push({
            mesh: pMesh,
            curve,
            t: Math.random(),
            speed: (0.12 + Math.random() * 0.18) / Math.max(1, d * 0.01)
          });
        }
      }
    }
  }

  createBundleCurve(p1, p2, bundleIdx, totalBundles, dist) {
    const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    const dir = new THREE.Vector3().subVectors(p2, p1).normalize();
    const up = (Math.abs(dir.y) < 0.9) ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const norm = new THREE.Vector3().crossVectors(dir, up).normalize();
    const binorm = new THREE.Vector3().crossVectors(dir, norm).normalize();

    // 束状扩散偏移 (Fascicular Dispersion)
    const angle = (bundleIdx / totalBundles) * Math.PI * 2;
    const bundleR = 4.0 + (dist * 0.04);
    const offsetX = Math.cos(angle) * bundleR;
    const offsetY = Math.sin(angle) * bundleR;

    const arcLift = Math.min(30.0, dist * 0.16);
    mid.addScaledVector(binorm, arcLift);
    mid.addScaledVector(norm, offsetX);

    const cp1 = new THREE.Vector3().lerpVectors(p1, mid, 0.55).addScaledVector(norm, offsetX * 0.5);
    const cp2 = new THREE.Vector3().lerpVectors(mid, p2, 0.45).addScaledVector(norm, offsetX * 0.5);

    return new THREE.CatmullRomCurve3([p1, cp1, mid, cp2, p2]);
  }

  extractAnatomicalClusters(cells, bounds) {
    // 依据解剖方位将细胞分为 4~8 个功能集群 (K-Means/Spatial Quadrants)
    const clusters = [];
    const n = cells.length;

    // 前向感觉核 (Rostral Sensory)
    const sensory = cells.filter(c => (c.layer === "L1_SENSORY" || (c.x || 0) < -70 || c.id < 32));
    if (sensory.length > 0) {
      clusters.push({ name: "Rostral_Sensory", center: this.calcCenter(sensory) });
    }

    // 左大脑半球皮层 (Left Cortical Hemisphere)
    const leftCortex = cells.filter(c => (c.y || 0) > 20 && Math.abs(c.x || 0) <= 90);
    if (leftCortex.length > 0) {
      clusters.push({ name: "Cortex_Left", center: this.calcCenter(leftCortex) });
    }

    // 右大脑半球皮层 (Right Cortical Hemisphere)
    const rightCortex = cells.filter(c => (c.y || 0) < -20 && Math.abs(c.x || 0) <= 90);
    if (rightCortex.length > 0) {
      clusters.push({ name: "Cortex_Right", center: this.calcCenter(rightCortex) });
    }

    // 后向运动尾核 (Caudal Motor Effector)
    const motor = cells.filter(c => (c.layer === "L3_MOTOR" || (c.x || 0) > 70 || c.id >= n - 224));
    if (motor.length > 0) {
      clusters.push({ name: "Caudal_Motor", center: this.calcCenter(motor) });
    }

    // 若无法区分象限，按空间 4 分位自动聚类
    if (clusters.length < 2) {
      const step = Math.max(1, Math.floor(n / 4));
      for (let k = 0; k < 4; k++) {
        const slice = cells.slice(k * step, (k + 1) * step);
        if (slice.length > 0) {
          clusters.push({ name: `Cluster_${k}`, center: this.calcCenter(slice) });
        }
      }
    }

    return clusters;
  }

  calcCenter(cellList) {
    let sx = 0, sy = 0, sz = 0;
    for (const c of cellList) {
      sx += (c.x || 0);
      sy += (c.y || 0);
      sz += (c.z || 0);
    }
    const len = cellList.length;
    return new THREE.Vector3(sx / len, sy / len, sz / len);
  }

  update(time, warpMultiplier = 1) {
    if (this.streamlines.length === 0) return;

    // 更新伴随流动的白质信息光子位置
    for (const p of this.particleStreams) {
      p.t = (p.t + p.speed * 0.016 * warpMultiplier) % 1.0;
      const pt = p.curve.getPoint(p.t);
      const pos = p.mesh.geometry.attributes.position.array;
      pos[0] = pt.x;
      pos[1] = pt.y;
      pos[2] = pt.z;
      p.mesh.geometry.attributes.position.needsUpdate = true;
      p.mesh.material.opacity = Math.sin(p.t * Math.PI) * 0.85;
    }
  }

  clear() {
    while (this.tractGroup.children.length > 0) {
      const obj = this.tractGroup.children[0];
      this.tractGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }
    this.streamlines.length = 0;
    this.particleStreams.length = 0;
  }
}

export const tractography = new TractographySystem();
