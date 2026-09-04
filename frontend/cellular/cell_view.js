/* ============================================================
 * cell_view.js - 生物拟真 3D 实体多层细胞计算结构
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

    // 1. 3D 半透明双脂质细胞质膜
    const membraneGeo = new THREE.IcosahedronGeometry(13, 3);
    const membraneMat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.35,
      metalness: 0.12,
      transparent: true,
      opacity: 0.52,
      emissive: col,
      emissiveIntensity: 0.08,
      depthWrite: false
    });
    this.membraneMesh = new THREE.Mesh(membraneGeo, membraneMat);

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

    // 3. 细胞核中枢与致密核仁
    const coreGeo = new THREE.SphereGeometry(6.0, 20, 20);
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: col,
      emissiveIntensity: 0.50,
      roughness: 0.2,
      metalness: 0.1
    });
    this.nucleus = new THREE.Mesh(coreGeo, coreMat);

    // 4. 围绕细胞核公转的 3 颗线粒体/能量细胞器
    this.organelles = [];
    const orgGeo = new THREE.SphereGeometry(1.8, 12, 12);
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
        orbitR: 9.5 + k * 1.5,
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
    this.group.add(this.membrane);
    this.group.add(this.nucleus);
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

    // 2. 最外层氛围光晕
    const haloScale = (22 + breath * 4) * (1.0 + actIntensity * 0.20);
    this.membrane.scale.set(haloScale, haloScale, 1);
    this.membrane.material.opacity = 0.10 + actIntensity * 0.20;

    // 3. 细胞核高能电位激化
    const nScale = 1.0 + actIntensity * 0.20;
    this.nucleus.scale.set(nScale, nScale, nScale);
    this.nucleus.material.emissiveIntensity = 0.45 + actIntensity * 0.65;

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
    if (this.nucleus.geometry) this.nucleus.geometry.dispose();
    if (this.nucleus.material) this.nucleus.material.dispose();
    if (this.shockwaveRing.geometry) this.shockwaveRing.geometry.dispose();
    if (this.shockwaveRing.material) this.shockwaveRing.material.dispose();
    for (const o of this.organelles) {
      if (o.mesh.geometry) o.mesh.geometry.dispose();
      if (o.mesh.material) o.mesh.material.dispose();
    }
  }
}
