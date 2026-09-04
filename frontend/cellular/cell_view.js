/* ============================================================
 * cell_view.js - 生物拟真 3D 实体多层细胞计算结构与相空间显微透视
 * 包含：双脂质质膜、微管蛋白细胞骨架、16阶环形时滞轮盘、相空间极限环吸引子、核仁与线粒体
 * ============================================================ */
import * as THREE from 'three';
import { FAMILY, FAMILY_COLOR } from './config.js';
import { getGlowTexture, getLabelTexture } from './texture_cache.js';
import { getCellWorldRadius } from './spatial_bounds.js';

export class CellView {
  constructor(cell, scene, org) {
    this.cell = cell;
    this.scene = scene;
    this.org = org;
    this.targetX = cell.x || 0;
    this.targetY = cell.y || 0;
    this.targetZ = cell.z || 0;
    this.curX = this.targetX;
    this.curY = this.targetY;
    this.curZ = this.targetZ;
    this.phase = (cell.id * 0.785) % (Math.PI * 2);
    this.shockwaveRadius = 1.0;

    const fam = FAMILY(cell.type);
    const col = FAMILY_COLOR[fam] || 0x38bdf8;
    const glow = getGlowTexture();

    // 1. 3D 半透明双脂质细胞质膜 (Lipid Bilayer Shell)
    const membraneGeo = new THREE.IcosahedronGeometry(13, 3);
    const membraneMat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.35,
      metalness: 0.12,
      transparent: true,
      opacity: 0.48,
      emissive: col,
      emissiveIntensity: 0.08,
      depthWrite: false
    });
    this.membraneMesh = new THREE.Mesh(membraneGeo, membraneMat);

    // 1.5 细胞微管蛋白骨架晶格 (Cytoskeleton Microtubule Lattice)
    const cytoGeo = new THREE.IcosahedronGeometry(12.5, 2);
    const cytoMat = new THREE.MeshBasicMaterial({
      color: col,
      wireframe: true,
      transparent: true,
      opacity: 0.14,
      depthWrite: false
    });
    this.cytoMesh = new THREE.Mesh(cytoGeo, cytoMat);

    // 2. 外层生物发光晕
    const haloMat = new THREE.SpriteMaterial({
      map: glow,
      color: col,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.16
    });
    this.membrane = new THREE.Sprite(haloMat);
    this.membrane.scale.set(24, 24, 1);

    // 3. 细胞核中枢与致密核仁 (26类动力学原语内核)
    const coreGeo = new THREE.SphereGeometry(5.8, 20, 20);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: col,
      emissiveIntensity: 0.50,
      roughness: 0.2,
      metalness: 0.1
    });
    this.nucleus = new THREE.Mesh(coreGeo, coreMat);

    // 3.5 16阶环形时滞数据轮盘 (16-step Ring Delay Buffer Carousel)
    const delayPoints = new Float32Array(16 * 3);
    const delayRadius = 8.4;
    for (let i = 0; i < 16; i++) {
      const th = (i / 16) * Math.PI * 2;
      delayPoints[i * 3]     = Math.cos(th) * delayRadius;
      delayPoints[i * 3 + 1] = Math.sin(th) * delayRadius;
      delayPoints[i * 3 + 2] = 0;
    }
    const delayGeo = new THREE.BufferGeometry();
    delayGeo.setAttribute('position', new THREE.BufferAttribute(delayPoints, 3));
    const delayMat = new THREE.PointsMaterial({
      size: 2.0,
      map: glow,
      color: col,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.delayRing = new THREE.Points(delayGeo, delayMat);

    // 3.8 动力学相空间极限环吸引子 (Phase-Space Limit Cycle Ribbon)
    const ATTR_SEGS = 36;
    const attrPoints = new Float32Array((ATTR_SEGS + 1) * 3);
    for (let i = 0; i <= ATTR_SEGS; i++) {
      const t = (i / ATTR_SEGS) * Math.PI * 2;
      const a = 6.2;
      const denom = 1.0 + Math.sin(t) * Math.sin(t);
      attrPoints[i * 3]     = (a * Math.cos(t)) / denom;
      attrPoints[i * 3 + 1] = (a * Math.sin(t) * Math.cos(t)) / denom;
      attrPoints[i * 3 + 2] = Math.sin(t * 2.0) * 2.0;
    }
    const attrGeo = new THREE.BufferGeometry();
    attrGeo.setAttribute('position', new THREE.BufferAttribute(attrPoints, 3));
    const attrMat = new THREE.LineBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.32,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.attrRibbon = new THREE.Line(attrGeo, attrMat);

    // 4. 围绕细胞核公转的 3 颗线粒体/能量细胞器
    this.organelles = [];
    const orgGeo = new THREE.SphereGeometry(1.6, 12, 12);
    for (let k = 0; k < 3; ++k) {
      const orgMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: col,
        emissiveIntensity: 0.55,
        roughness: 0.2
      });
      const orgMesh = new THREE.Mesh(orgGeo, orgMat);
      this.organelles.push({
        mesh: orgMesh,
        orbitR: 9.8 + k * 1.5,
        speed: 2.2 + k * 0.8,
        tilt: (k * Math.PI) / 3
      });
    }

    // 5. 动作电位电离冲击波圆环
    const ringGeo = new THREE.RingGeometry(9, 13, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.shockwaveRing = new THREE.Mesh(ringGeo, ringMat);
    this.shockwaveRing.rotation.x = Math.PI / 2;

    // 6. 生物原语全息浮动标签
    const lm = new THREE.SpriteMaterial({ map: getLabelTexture(cell.type), transparent: true, depthWrite: false });
    this.labelMat = lm;
    this.label = new THREE.Sprite(lm);
    this.label.scale.set(52, 13, 1);

    this.group = new THREE.Group();
    this.group.add(this.membraneMesh);
    this.group.add(this.cytoMesh);
    this.group.add(this.membrane);
    this.group.add(this.nucleus);
    this.group.add(this.delayRing);
    this.group.add(this.attrRibbon);
    this.group.add(this.shockwaveRing);
    for (const o of this.organelles) {
      this.group.add(o.mesh);
    }
    this.group.add(this.label);

    const rBase = getCellWorldRadius(this.org);
    const sf = Math.max(0.12, Math.min(1.0, rBase / 13.0));
    this.group.scale.set(sf, sf, sf);

    if (this.scene) {
      this.scene.add(this.group);
    }
  }

  updateCell(cell, org = null) {
    this.cell = cell;
    if (org) this.org = org;
    this.targetX = cell.x;
    this.targetY = cell.y;
    this.targetZ = cell.z || 0;
    const rBase = getCellWorldRadius(this.org);
    const sf = Math.max(0.12, Math.min(1.0, rBase / 13.0));
    this.group.scale.set(sf, sf, sf);

    const fam = FAMILY(cell.type);
    const col = FAMILY_COLOR[fam] || 0x38bdf8;
    if (this.membraneMesh && this.membraneMesh.material) {
      this.membraneMesh.material.color.setHex(col);
      this.membraneMesh.material.emissive.setHex(col);
    }
    if (this.cytoMesh && this.cytoMesh.material) {
      this.cytoMesh.material.color.setHex(col);
    }
    if (this.labelMat) {
      this.labelMat.map = getLabelTexture(cell.type);
    }
  }

  update(time, warpMultiplier = 1) {
    const c = this.cell;
    this.curX += (this.targetX - this.curX) * 0.25;
    this.curY += (this.targetY - this.curY) * 0.25;
    this.curZ += (this.targetZ - this.curZ) * 0.25;
    this.group.position.set(this.curX, this.curY, this.curZ);
    this.label.position.set(0, -18, 0);

    const breath = Math.sin(time * 2.2 + this.phase) * 0.08;
    const actIntensity = Math.min(1.5, Math.abs(c.out || 0) + (c.glow || 0));

    // 1. 3D 质膜呼吸形变与发光
    const memScale = (1.0 + breath) * (1.0 + actIntensity * 0.18);
    this.membraneMesh.scale.set(memScale, memScale, memScale);
    this.membraneMesh.rotation.y += 0.006;
    this.membraneMesh.rotation.x += 0.003;
    this.membraneMesh.material.emissiveIntensity = 0.06 + actIntensity * 0.30;
    this.membraneMesh.material.opacity = 0.45 + Math.min(0.20, actIntensity * 0.15);

    // 1.5 细胞微管骨架随动慢旋
    this.cytoMesh.scale.set(memScale, memScale, memScale);
    this.cytoMesh.rotation.y += 0.004;
    this.cytoMesh.rotation.z += 0.002;
    this.cytoMesh.material.opacity = 0.10 + actIntensity * 0.15;

    // 2. 最外层氛围光晕
    const haloScale = (22 + breath * 4) * (1.0 + actIntensity * 0.20);
    this.membrane.scale.set(haloScale, haloScale, 1);
    this.membrane.material.opacity = 0.10 + actIntensity * 0.20;

    // 3. 细胞核高能电位激化
    const nScale = 1.0 + actIntensity * 0.20;
    this.nucleus.scale.set(nScale, nScale, nScale);
    this.nucleus.material.emissiveIntensity = 0.45 + actIntensity * 0.65;

    // 3.5 16阶环形时滞数据轮盘旋转自旋
    this.delayRing.rotation.z += (0.012 + actIntensity * 0.03) * warpMultiplier;
    this.delayRing.rotation.x = Math.sin(time * 0.5 + this.phase) * 0.25;
    this.delayRing.material.opacity = 0.25 + actIntensity * 0.35;
    this.delayRing.material.size = 1.8 + actIntensity * 1.2;

    // 3.8 相空间极限环双纽吸引子实时翻滚
    this.attrRibbon.rotation.x += 0.010 * warpMultiplier;
    this.attrRibbon.rotation.y += 0.016 * warpMultiplier;
    const ribScale = 1.0 + Math.sin(time * 3.0 + this.phase) * 0.12 + Math.min(0.3, Math.abs(c.state || 0) * 0.15);
    this.attrRibbon.scale.set(ribScale, ribScale, ribScale);
    this.attrRibbon.material.opacity = 0.22 + actIntensity * 0.45;

    // 4. 线粒体能量颗粒公转
    for (let k = 0; k < this.organelles.length; ++k) {
      const o = this.organelles[k];
      const ang = time * o.speed + this.phase + (k * Math.PI * 2) / 3;
      const ox = Math.cos(ang) * o.orbitR;
      const oy = Math.sin(ang) * o.orbitR * Math.cos(o.tilt);
      const oz = Math.sin(ang) * o.orbitR * Math.sin(o.tilt);
      o.mesh.position.set(ox, oy, oz);
      o.mesh.material.emissiveIntensity = 0.45 + Math.sin(time * 3.0 + k) * 0.20 + actIntensity * 0.35;
    }

    // 5. 动作电位放电冲击波环扩散
    if (Math.abs(c.out || 0) > 0.45) {
      this.shockwaveRadius += 0.06;
      if (this.shockwaveRadius > 2.6) this.shockwaveRadius = 1.0;
      const ringScale = this.shockwaveRadius;
      this.shockwaveRing.scale.set(ringScale, ringScale, ringScale);
      this.shockwaveRing.material.opacity = Math.max(0.0, (2.6 - this.shockwaveRadius) * 0.25);
      this.shockwaveRing.visible = true;
    } else {
      this.shockwaveRadius = 1.0;
      this.shockwaveRing.visible = false;
    }
  }

  dispose() {
    if (this.scene) this.scene.remove(this.group);
    if (this.labelMat) this.labelMat.dispose();
    if (this.membraneMesh.geometry) this.membraneMesh.geometry.dispose();
    if (this.membraneMesh.material) this.membraneMesh.material.dispose();
    if (this.cytoMesh.geometry) this.cytoMesh.geometry.dispose();
    if (this.cytoMesh.material) this.cytoMesh.material.dispose();
    if (this.nucleus.geometry) this.nucleus.geometry.dispose();
    if (this.nucleus.material) this.nucleus.material.dispose();
    if (this.delayRing.geometry) this.delayRing.geometry.dispose();
    if (this.delayRing.material) this.delayRing.material.dispose();
    if (this.attrRibbon.geometry) this.attrRibbon.geometry.dispose();
    if (this.attrRibbon.material) this.attrRibbon.material.dispose();
    if (this.shockwaveRing.geometry) this.shockwaveRing.geometry.dispose();
    if (this.shockwaveRing.material) this.shockwaveRing.material.dispose();
    for (const o of this.organelles) {
      if (o.mesh.geometry) o.mesh.geometry.dispose();
      if (o.mesh.material) o.mesh.material.dispose();
    }
  }
}
