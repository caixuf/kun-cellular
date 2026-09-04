/* ============================================================
 * synapse_view.js - 突触连接与动作电位光子流动 (二次贝塞尔曲线与终扣)
 * ============================================================ */
import * as THREE from 'three';
import { getGlowTexture } from './texture_cache.js';

const _tmpP0 = new THREE.Vector3();
const _tmpP1 = new THREE.Vector3();
const _tmpP2 = new THREE.Vector3();
const _tmpChord = new THREE.Vector3();
const _tmpUp = new THREE.Vector3(0, 1, 0);
const _tmpNorm = new THREE.Vector3();

export class SynapseView {
  constructor(syn, org, scene = null) {
    this.syn = syn;
    this.org = org;
    this.scene = scene;
    this.numSegments = 16;

    this.curvePoints = new Float32Array((this.numSegments + 1) * 3);
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute("position", new THREE.BufferAttribute(this.curvePoints, 3));

    this.lineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.35,
      blending: THREE.NormalBlending
    });
    this.line = new THREE.Line(this.geo, this.lineMat);

    const glow = getGlowTexture();
    this.photonMat1 = new THREE.SpriteMaterial({ map: glow, color: 0x38bdf8, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.photonMat2 = new THREE.SpriteMaterial({ map: glow, color: 0xa855f7, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.photon1 = new THREE.Sprite(this.photonMat1);
    this.photon2 = new THREE.Sprite(this.photonMat2);
    this.photon1.scale.set(8, 8, 1);
    this.photon2.scale.set(6, 6, 1);

    this.boutonMat = new THREE.SpriteMaterial({ map: glow, color: 0x38bdf8, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    this.bouton = new THREE.Sprite(this.boutonMat);
    this.bouton.scale.set(10, 10, 1);

    this.group = new THREE.Group();
    this.group.add(this.line);
    this.group.add(this.photon1);
    this.group.add(this.photon2);
    this.group.add(this.bouton);

    if (this.scene) {
      this.scene.add(this.group);
    }

    this.flowT1 = Math.random();
    this.flowT2 = (this.flowT1 + 0.5) % 1.0;
  }

  updateSyn(syn, org) {
    this.syn = syn;
    this.org = org;
  }

  update(time, warpMultiplier = 1) {
    if (!this.org.cellMap) this.org.cellMap = new Map(this.org.cells.map(c => [c.id, c]));
    const a = this.org.cellMap.get(this.syn.from);
    const b = this.org.cellMap.get(this.syn.to);
    if (!a || !b || !this.syn.active) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    _tmpP0.set(a.x, a.y, a.z || 0);
    _tmpP2.set(b.x, b.y, b.z || 0);
    _tmpChord.subVectors(_tmpP2, _tmpP0);
    const dist = _tmpChord.length();
    const arcHeight = Math.min(25, dist * 0.18);

    if (Math.abs(_tmpChord.y) < 0.95 * dist) _tmpUp.set(0, 1, 0);
    else _tmpUp.set(1, 0, 0);
    _tmpNorm.crossVectors(_tmpChord, _tmpUp).normalize();

    _tmpP1.addVectors(_tmpP0, _tmpP2).multiplyScalar(0.5);
    _tmpP1.addScaledVector(_tmpNorm, arcHeight * ((this.syn.port === 0) ? 1 : -1));

    const pos = this.geo.attributes.position;
    for (let i = 0; i <= this.numSegments; i++) {
      const t = i / this.numSegments;
      const it = 1.0 - t;
      const bx = it * it * _tmpP0.x + 2.0 * it * t * _tmpP1.x + t * t * _tmpP2.x;
      const by = it * it * _tmpP0.y + 2.0 * it * t * _tmpP1.y + t * t * _tmpP2.y;
      const bz = it * it * _tmpP0.z + 2.0 * it * t * _tmpP1.z + t * t * _tmpP2.z;
      pos.setXYZ(i, bx, by, bz);
    }
    pos.needsUpdate = true;

    const w = this.syn.w !== undefined ? this.syn.w : (this.syn.weight !== undefined ? this.syn.weight : 1.0);
    const hot = Math.abs(w) > 0.6;
    this.lineMat.color.setHex(w >= 0 ? (hot ? 0x38bdf8 : 0x0284c7) : (hot ? 0xf43f5e : 0xbe123c));
    this.lineMat.opacity = 0.25 + Math.min(0.35, Math.abs(w) * 0.25);

    const speed = (0.012 + Math.min(0.04, Math.abs(w) * 0.025)) * warpMultiplier;
    this.flowT1 = (this.flowT1 + speed) % 1.0;
    this.flowT2 = (this.flowT2 + speed * 1.1) % 1.0;

    // 囊泡 1 采样
    {
      const t = this.flowT1, it = 1.0 - t;
      this.photon1.position.set(
        it * it * _tmpP0.x + 2.0 * it * t * _tmpP1.x + t * t * _tmpP2.x,
        it * it * _tmpP0.y + 2.0 * it * t * _tmpP1.y + t * t * _tmpP2.y,
        it * it * _tmpP0.z + 2.0 * it * t * _tmpP1.z + t * t * _tmpP2.z
      );
      const pScale1 = (6 + Math.sin(t * Math.PI) * 4) * (0.8 + Math.abs(w) * 0.3);
      this.photon1.scale.set(pScale1, pScale1, 1);
      this.photon1.material.opacity = 0.28 + Math.sin(t * Math.PI) * 0.5;
    }

    // 囊泡 2 采样
    {
      const t = this.flowT2, it = 1.0 - t;
      this.photon2.position.set(
        it * it * _tmpP0.x + 2.0 * it * t * _tmpP1.x + t * t * _tmpP2.x,
        it * it * _tmpP0.y + 2.0 * it * t * _tmpP1.y + t * t * _tmpP2.y,
        it * it * _tmpP0.z + 2.0 * it * t * _tmpP1.z + t * t * _tmpP2.z
      );
      const pScale2 = (5 + Math.sin(t * Math.PI) * 3) * (0.8 + Math.abs(w) * 0.3);
      this.photon2.scale.set(pScale2, pScale2, 1);
      this.photon2.material.opacity = 0.22 + Math.sin(t * Math.PI) * 0.45;
    }

    // 突触终端小体
    this.bouton.position.copy(_tmpP2);
    const boutonFlash = (1.0 - Math.min(1.0, this.flowT1)) * (a.glow || 0.2);
    const bScale = 7 + boutonFlash * 7;
    this.bouton.scale.set(bScale, bScale, 1);
    this.bouton.material.opacity = 0.25 + boutonFlash * 0.45;
    this.bouton.material.color.setHex(w >= 0 ? 0x38bdf8 : 0xf43f5e);
  }

  dispose() {
    if (this.scene) this.scene.remove(this.group);
    if (this.geo) this.geo.dispose();
    if (this.lineMat) this.lineMat.dispose();
    if (this.photonMat1) this.photonMat1.dispose();
    if (this.photonMat2) this.photonMat2.dispose();
    if (this.boutonMat) this.boutonMat.dispose();
  }
}
