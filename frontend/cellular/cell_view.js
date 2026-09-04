/* ============================================================
 * cell_view.js - 生物拟真 3D 实体多层细胞计算结构与微观代谢通道
 * 包含：
 * 1. 磷脂双分子层多层膜结构 (外层亲水质膜 Outer Leaflet + 内层胞质支持膜 Inner Leaflet)
 * 2. 8向跨膜通道蛋白孔道 (Transmembrane Porins & Gated Channels，营养/离子/病毒进出通道)
 * 3. 动态穿膜新陈代谢粒子交换流 (Metabolic Nutrient & Ion Particle Flux)
 * 4. 微管蛋白细胞骨架、16阶环形时滞轮盘、相空间极限环吸引子、核仁与线粒体
 * ============================================================ */
import * as THREE from 'three';
import { FAMILY, FAMILY_COLOR } from './config.js';
import { getGlowTexture, getLabelTexture } from './texture_cache.js';
import { getCellWorldRadius } from './spatial_bounds.js';

export const PORE_DIRS = [
  new THREE.Vector3( 1,  1,  1).normalize(),
  new THREE.Vector3( 1,  1, -1).normalize(),
  new THREE.Vector3( 1, -1,  1).normalize(),
  new THREE.Vector3( 1, -1, -1).normalize(),
  new THREE.Vector3(-1,  1,  1).normalize(),
  new THREE.Vector3(-1,  1, -1).normalize(),
  new THREE.Vector3(-1, -1,  1).normalize(),
  new THREE.Vector3(-1, -1, -1).normalize()
];

function buildPoresGeometry() {
  const geos = [];
  const upVec = new THREE.Vector3(0, 1, 0);
  const zVec = new THREE.Vector3(0, 0, 1);

  for (const dir of PORE_DIRS) {
    // 1. 外层受体漏斗环口 (Outer receptor funnel rim at r=13.3)
    const rimGeo = new THREE.RingGeometry(0.85, 2.1, 10);
    const qRim = new THREE.Quaternion().setFromUnitVectors(zVec, dir);
    rimGeo.applyQuaternion(qRim);
    rimGeo.translate(dir.x * 13.3, dir.y * 13.3, dir.z * 13.3);
    geos.push(rimGeo.toNonIndexed());

    // 2. 跨膜通道蛋白中空柱 (Transmembrane channel tube from r=10.5 to r=13.3)
    const tubeGeo = new THREE.CylinderGeometry(1.25, 1.25, 3.2, 10, 1, true);
    const qTube = new THREE.Quaternion().setFromUnitVectors(upVec, dir);
    tubeGeo.applyQuaternion(qTube);
    tubeGeo.translate(dir.x * 11.9, dir.y * 11.9, dir.z * 11.9);
    geos.push(tubeGeo.toNonIndexed());

    // 3. 内侧胞质门控环口 (Inner cytosolic gate rim at r=10.5)
    const innerRimGeo = new THREE.RingGeometry(0.7, 1.8, 10);
    const qInner = new THREE.Quaternion().setFromUnitVectors(zVec, dir.clone().negate());
    innerRimGeo.applyQuaternion(qInner);
    innerRimGeo.translate(dir.x * 10.5, dir.y * 10.5, dir.z * 10.5);
    geos.push(innerRimGeo.toNonIndexed());
  }

  let totalVerts = 0;
  for (const g of geos) totalVerts += g.attributes.position.count;
  const posArr = new Float32Array(totalVerts * 3);
  const normArr = new Float32Array(totalVerts * 3);
  let offset = 0;
  for (const g of geos) {
    posArr.set(g.attributes.position.array, offset * 3);
    if (g.attributes.normal) normArr.set(g.attributes.normal.array, offset * 3);
    offset += g.attributes.position.count;
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));
  return merged;
}

let _sharedGeos = null;
function getSharedGeos() {
  if (!_sharedGeos) {
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

    const ATTR_SEGS = 24;
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

    _sharedGeos = {
      outerMembrane: new THREE.IcosahedronGeometry(13.0, 2),
      innerMembrane: new THREE.IcosahedronGeometry(11.2, 2),
      membrane: new THREE.IcosahedronGeometry(13.0, 2),
      cyto: new THREE.IcosahedronGeometry(10.8, 1),
      pores: buildPoresGeometry(),
      nucleus: new THREE.SphereGeometry(5.8, 14, 14),
      delay: delayGeo,
      attr: attrGeo,
      org: new THREE.SphereGeometry(1.6, 8, 8),
      shock: new THREE.RingGeometry(9, 13, 24)
    };
  }
  return _sharedGeos;
}

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
    const geos = getSharedGeos();

    // 1. 3D 外层磷脂双分子亲水质膜 (Outer Lipid Leaflet Membrane)
    const outerMat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.25,
      metalness: 0.15,
      transparent: true,
      opacity: 0.38,
      emissive: col,
      emissiveIntensity: 0.08,
      depthWrite: false,
      side: THREE.FrontSide
    });
    this.outerMembraneMesh = new THREE.Mesh(geos.outerMembrane, outerMat);
    this.membraneMesh = this.outerMembraneMesh;

    // 1.2 3D 内层胞质支持膜 (Inner Leaflet / Cortex Membrane)
    // 外膜与内膜之间形成间质腔 (Periplasmic Cavity)，显微镜下呈现真实双层膜厚度
    const innerMat = new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.40,
      metalness: 0.08,
      transparent: true,
      opacity: 0.22,
      emissive: col,
      emissiveIntensity: 0.05,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    this.innerMembraneMesh = new THREE.Mesh(geos.innerMembrane, innerMat);

    // 1.3 8向跨膜蛋白离子通道孔道 (Transmembrane Porins & Gated Ion Channels)
    // 贯穿内外双层膜，为营养/离子与病毒/抗原提供真实物理出入通道
    const poreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.25,
      metalness: 0.40,
      emissive: col,
      emissiveIntensity: 0.35,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.poresMesh = new THREE.Mesh(geos.pores, poreMat);

    // 1.4 穿膜新陈代谢粒子交换流 (Metabolic Nutrient & Ion Particle Flux)
    // 24颗动态代谢粒子沿着8个穿膜孔道流动 (外部营养吸入，胞内核仁/线粒体代谢排泄)
    const METABOLIC_COUNT = 24;
    const metaPos = new Float32Array(METABOLIC_COUNT * 3);
    const metaGeo = new THREE.BufferGeometry();
    metaGeo.setAttribute('position', new THREE.BufferAttribute(metaPos, 3));
    const metaMat = new THREE.PointsMaterial({
      size: 2.4,
      map: glow,
      color: 0x67e8f9,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.metabolicGeo = metaGeo;
    this.metabolicMat = metaMat;
    this.metabolicPoints = new THREE.Points(metaGeo, metaMat);

    // 1.5 细胞微管蛋白骨架晶格 (Cytoskeleton Microtubule Lattice)
    const cytoMat = new THREE.MeshBasicMaterial({
      color: col,
      wireframe: true,
      transparent: true,
      opacity: 0.14,
      depthWrite: false
    });
    this.cytoMesh = new THREE.Mesh(geos.cyto, cytoMat);

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
    const coreMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: col,
      emissiveIntensity: 0.50,
      roughness: 0.2,
      metalness: 0.1
    });
    this.nucleus = new THREE.Mesh(geos.nucleus, coreMat);

    // 3.5 16阶环形时滞数据轮盘 (16-step Ring Delay Buffer Carousel)
    const delayMat = new THREE.PointsMaterial({
      size: 2.0,
      map: glow,
      color: col,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.delayRing = new THREE.Points(geos.delay, delayMat);

    // 3.8 动力学相空间极限环吸引子 (Phase-Space Limit Cycle Ribbon)
    const attrMat = new THREE.LineBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.32,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.attrRibbon = new THREE.Line(geos.attr, attrMat);

    // 4. 围绕细胞核公转的 3 颗线粒体/能量细胞器
    this.organelles = [];
    for (let k = 0; k < 3; ++k) {
      const orgMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: col,
        emissiveIntensity: 0.55,
        roughness: 0.2
      });
      const orgMesh = new THREE.Mesh(geos.org, orgMat);
      this.organelles.push({
        mesh: orgMesh,
        orbitR: 9.8 + k * 1.5,
        speed: 2.2 + k * 0.8,
        tilt: (k * Math.PI) / 3
      });
    }

    // 5. 动作电位电离冲击波圆环
    const ringMat = new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: 0.0,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.shockwaveRing = new THREE.Mesh(geos.shock, ringMat);
    this.shockwaveRing.rotation.x = Math.PI / 2;

    // 6. 生物原语全息浮动标签
    const lm = new THREE.SpriteMaterial({ map: getLabelTexture(cell.type), transparent: true, depthWrite: false });
    this.labelMat = lm;
    this.label = new THREE.Sprite(lm);
    this.label.scale.set(52, 13, 1);

    this.group = new THREE.Group();
    this.group.add(this.outerMembraneMesh);
    this.group.add(this.innerMembraneMesh);
    this.group.add(this.poresMesh);
    this.group.add(this.metabolicPoints);
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
    const sf = rBase / 13.0;
    this.group.scale.set(sf, sf, sf);

    if (this.scene) {
      this.scene.add(this.group);
    }
  }

  updateCell(cell, org = null) {
    this.cell = cell;
    if (org) this.org = org;
    this.targetX = cell.x || 0;
    this.targetY = cell.y || 0;
    this.targetZ = cell.z || 0;
    this.curX = this.targetX;
    this.curY = this.targetY;
    this.curZ = this.targetZ;
    this.group.position.set(this.curX, this.curY, this.curZ);
    const rBase = getCellWorldRadius(this.org);
    const sf = rBase / 13.0;
    this.group.scale.set(sf, sf, sf);

    const fam = FAMILY(cell.type);
    const col = FAMILY_COLOR[fam] || 0x38bdf8;
    if (this.outerMembraneMesh && this.outerMembraneMesh.material) {
      this.outerMembraneMesh.material.color.setHex(col);
      this.outerMembraneMesh.material.emissive.setHex(col);
    }
    if (this.innerMembraneMesh && this.innerMembraneMesh.material) {
      this.innerMembraneMesh.material.color.setHex(col);
      this.innerMembraneMesh.material.emissive.setHex(col);
    }
    if (this.poresMesh && this.poresMesh.material) {
      this.poresMesh.material.emissive.setHex(col);
    }
    if (this.cytoMesh && this.cytoMesh.material) {
      this.cytoMesh.material.color.setHex(col);
    }
    if (this.nucleus && this.nucleus.material) {
      this.nucleus.material.emissive.setHex(col);
    }
    if (this.membrane && this.membrane.material) {
      this.membrane.material.color.setHex(col);
    }
    if (this.delayRing && this.delayRing.material) {
      this.delayRing.material.color.setHex(col);
    }
    if (this.attrRibbon && this.attrRibbon.material) {
      this.attrRibbon.material.color.setHex(col);
    }
    if (this.organelles) {
      for (const o of this.organelles) {
        if (o.mesh && o.mesh.material) o.mesh.material.emissive.setHex(col);
      }
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
    const actIntensity = Math.min(2.0, Math.abs(c.out || 0) + (c.glow || 0));

    // 因果脉冲去极化击穿闪光检测 (Depolarization Flash Trigger)
    if (this.lastOut === undefined) this.lastOut = c.out || 0;
    if (this.flashIntensity === undefined) this.flashIntensity = 0.0;
    const deltaOut = Math.abs((c.out || 0) - this.lastOut);
    this.lastOut = c.out || 0;
    if (deltaOut > 0.12 || Math.abs(c.out || 0) > 0.45) {
      this.flashIntensity = Math.min(2.8, this.flashIntensity + deltaOut * 2.5 + (Math.abs(c.out || 0) > 0.6 ? 0.8 : 0.2));
    }
    this.flashIntensity *= 0.90;

    // 1. 3D 外层质膜呼吸形变与电位击穿发光
    const memScale = (1.0 + breath) * (1.0 + actIntensity * 0.18 + this.flashIntensity * 0.08);
    this.outerMembraneMesh.scale.set(memScale, memScale, memScale);
    this.outerMembraneMesh.rotation.y += 0.006 + this.flashIntensity * 0.02;
    this.outerMembraneMesh.rotation.x += 0.003;
    this.outerMembraneMesh.material.emissiveIntensity = 0.06 + actIntensity * 0.35 + this.flashIntensity * 0.85;
    this.outerMembraneMesh.material.opacity = 0.38 + Math.min(0.35, actIntensity * 0.18 + this.flashIntensity * 0.30);

    // 1.2 内层膜同步呼吸 (保持在膜间质腔内侧)
    if (this.innerMembraneMesh && this.innerMembraneMesh.visible) {
      const innerScale = memScale * 0.96;
      this.innerMembraneMesh.scale.set(innerScale, innerScale, innerScale);
      this.innerMembraneMesh.rotation.y -= 0.004;
      this.innerMembraneMesh.material.opacity = 0.20 + Math.min(0.15, actIntensity * 0.12);
    }

    // 1.3 跨膜通道孔开合脉动与离子发光
    if (this.poresMesh && this.poresMesh.visible && this.poresMesh.material) {
      this.poresMesh.material.emissiveIntensity = 0.30 + Math.min(1.8, actIntensity * 1.2);
      const poreBreath = 1.0 + Math.sin(time * 3.2 + this.phase) * 0.04;
      this.poresMesh.scale.set(poreBreath, poreBreath, poreBreath);
    }

    // 1.4 动态更新穿膜新陈代谢粒子流 (Metabolic Nutrient & Ion Flux)
    if (this.metabolicPoints && this.metabolicPoints.visible) {
      const pos = this.metabolicGeo.attributes.position.array;
      for (let p = 0; p < 24; p++) {
        const poreIdx = Math.floor(p / 3);
        const type = p % 3;
        const d = PORE_DIRS[poreIdx];
        let r = 0;
        if (type === 0) {
          // 外部环境营养底质/ATP离子穿孔吸入 (18.5 -> 6.0)
          const prog = ((time * 1.6 + poreIdx * 0.25 + this.phase) % 1.0);
          r = 18.5 - prog * 12.5;
        } else if (type === 1) {
          // 膜通道内部离子门控振荡 (10.0 <-> 14.0)
          r = 12.0 + Math.sin(time * 4.0 + poreIdx * 1.1 + this.phase) * 2.0;
        } else {
          // 代谢废物/动作电位电荷穿孔向外泵出 (6.0 -> 18.5)
          const prog = ((time * 1.4 + poreIdx * 0.35 + this.phase) % 1.0);
          r = 6.0 + prog * 12.5;
        }
        pos[p * 3]     = d.x * r;
        pos[p * 3 + 1] = d.y * r;
        pos[p * 3 + 2] = d.z * r;
      }
      this.metabolicGeo.attributes.position.needsUpdate = true;
    }

    // 1.5 细胞微管骨架随动慢旋
    if (this.cytoMesh && this.cytoMesh.visible) {
      this.cytoMesh.scale.set(memScale, memScale, memScale);
      this.cytoMesh.rotation.y += 0.004;
      this.cytoMesh.rotation.z += 0.002;
      this.cytoMesh.material.opacity = 0.10 + actIntensity * 0.15;
    }

    // 2. 最外层氛围光晕
    const haloScale = (22 + breath * 4) * (1.0 + actIntensity * 0.20);
    this.membrane.scale.set(haloScale, haloScale, 1);
    this.membrane.material.opacity = 0.10 + actIntensity * 0.20;

    // 3. 细胞核高能电位激化
    const nScale = 1.0 + actIntensity * 0.20;
    this.nucleus.scale.set(nScale, nScale, nScale);
    this.nucleus.material.emissiveIntensity = 0.45 + actIntensity * 0.65;

    // 3.5 16阶环形时滞数据轮盘旋转自旋
    if (this.delayRing && this.delayRing.visible) {
      this.delayRing.rotation.z += (0.012 + actIntensity * 0.03) * warpMultiplier;
      this.delayRing.rotation.x = Math.sin(time * 0.5 + this.phase) * 0.25;
      this.delayRing.material.opacity = 0.25 + actIntensity * 0.35;
      this.delayRing.material.size = 1.8 + actIntensity * 1.2;
    }

    // 3.8 相空间极限环双纽吸引子实时翻滚
    if (this.attrRibbon && this.attrRibbon.visible) {
      this.attrRibbon.rotation.x += 0.010 * warpMultiplier;
      this.attrRibbon.rotation.y += 0.016 * warpMultiplier;
      const ribScale = 1.0 + Math.sin(time * 3.0 + this.phase) * 0.12 + Math.min(0.3, Math.abs(c.state || 0) * 0.15);
      this.attrRibbon.scale.set(ribScale, ribScale, ribScale);
      this.attrRibbon.material.opacity = 0.22 + actIntensity * 0.45;
    }

    // 4. 线粒体能量颗粒公转
    if (this.organelles && this.organelles.length > 0 && this.organelles[0].mesh && this.organelles[0].mesh.visible) {
      for (let k = 0; k < this.organelles.length; ++k) {
        const o = this.organelles[k];
        const ang = time * o.speed + this.phase + (k * Math.PI * 2) / 3;
        const ox = Math.cos(ang) * o.orbitR;
        const oy = Math.sin(ang) * o.orbitR * Math.cos(o.tilt);
        const oz = Math.sin(ang) * o.orbitR * Math.sin(o.tilt);
        o.mesh.position.set(ox, oy, oz);
        o.mesh.material.emissiveIntensity = 0.45 + Math.sin(time * 3.0 + k) * 0.20 + actIntensity * 0.35;
      }
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
    if (this.outerMembraneMesh && this.outerMembraneMesh.material) this.outerMembraneMesh.material.dispose();
    if (this.innerMembraneMesh && this.innerMembraneMesh.material) this.innerMembraneMesh.material.dispose();
    if (this.poresMesh && this.poresMesh.material) this.poresMesh.material.dispose();
    if (this.metabolicGeo) this.metabolicGeo.dispose();
    if (this.metabolicMat) this.metabolicMat.dispose();
    if (this.cytoMesh && this.cytoMesh.material) this.cytoMesh.material.dispose();
    if (this.nucleus && this.nucleus.material) this.nucleus.material.dispose();
    if (this.delayRing && this.delayRing.material) this.delayRing.material.dispose();
    if (this.attrRibbon && this.attrRibbon.material) this.attrRibbon.material.dispose();
    if (this.shockwaveRing && this.shockwaveRing.material) this.shockwaveRing.material.dispose();
    for (const o of this.organelles) {
      if (o.mesh && o.mesh.material) o.mesh.material.dispose();
    }
  }
}
