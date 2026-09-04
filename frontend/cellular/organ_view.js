/* ============================================================
 * organ_view.js - 3D 生物器官形态学外包膜与共生微柱全息透视系统
 * (3D Organ Morphological Envelopes & Biome Fields)
 * 揭示超细胞微柱 (Symbiotic Macro-Cells) 与跨物种冷冻器官 (Organ Frozen Bank)
 * 在三维空间中的宏观聚合组织边界、器官间神经索与全息铭牌
 * ============================================================ */
import * as THREE from 'three';
import { camState } from './camera_controller.js';
import { log } from './network_sync.js';
import { getCellWorldRadius } from './spatial_bounds.js';

let _organRootGroup = null;
const _organViewsMap = new Map();
let _isOrganVisible = true;

const _sharedEnvelopeGeo = new THREE.SphereGeometry(1.0, 24, 16);
const _sharedRingGeo = new THREE.TorusGeometry(1.0, 0.022, 8, 48);

const ORGAN_BADGE_CACHE = new Map();

function getOrganBadgeTexture(label, count, colorHex) {
  const key = `${label}_${count}_${colorHex}`;
  if (ORGAN_BADGE_CACHE.has(key)) return ORGAN_BADGE_CACHE.get(key);

  const cv = document.createElement('canvas');
  cv.width = 384;
  cv.height = 96;
  const ctx = cv.getContext('2d');

  // 半透明深色生化玻璃底板
  ctx.fillStyle = 'rgba(6, 12, 24, 0.82)';
  ctx.strokeStyle = colorHex || '#38bdf8';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(6, 6, 372, 84, 12);
  ctx.fill();
  ctx.stroke();

  // 左侧发光微柱信号灯
  ctx.fillStyle = colorHex || '#38bdf8';
  ctx.beginPath();
  ctx.arc(28, 48, 8, 0, Math.PI * 2);
  ctx.fill();

  // 标题：微柱器官名称
  ctx.font = 'bold 24px "JetBrains Mono", monospace';
  ctx.fillStyle = colorHex || '#38bdf8';
  ctx.textAlign = 'left';
  ctx.fillText(`[ORGAN] ${label.toUpperCase()}`, 48, 42);

  // 副标题：包含元胞数与生物功能
  ctx.font = '16px "JetBrains Mono", monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(`${count} CELLS · SYMBIOTIC COLUMN`, 48, 70);

  const tex = new THREE.CanvasTexture(cv);
  ORGAN_BADGE_CACHE.set(key, tex);
  return tex;
}

export function initOrganSystem(scene) {
  if (!_organRootGroup) {
    _organRootGroup = new THREE.Group();
    _organRootGroup.name = 'OrganRootGroup';
    scene.add(_organRootGroup);
  }
  return _organRootGroup;
}

export function toggleOrganVisibility(visible = null) {
  if (visible === null) _isOrganVisible = !_isOrganVisible;
  else _isOrganVisible = visible;
  if (_organRootGroup) _organRootGroup.visible = _isOrganVisible;
  return _isOrganVisible;
}

export function extractOrganList(org) {
  if (org && org.symbiotic_macro_cells && org.symbiotic_macro_cells.length > 0) {
    return org.symbiotic_macro_cells.map(mc => ({
      id: mc.id || mc.macro_id,
      label: mc.label || `MacroColumn_${mc.id}`,
      cellIds: mc.internal_cell_ids || mc.cells || [],
      color: mc.color || '#38bdf8'
    }));
  }

  // 自动自组织功能分区 (当未收到服务端显式超细胞微柱时，按生物原语层自组织聚合)
  if (!org || !org.cells || org.cells.length === 0) return [];

  const sensory = [];
  const association = [];
  const motor = [];

  for (const c of org.cells) {
    const t = (c.type || '').toUpperCase();
    if (t.startsWith('SENSE') || t.startsWith('REC') || (c.layer && c.layer.includes('SENSORY'))) {
      sensory.push(c.id);
    } else if (t.startsWith('ACT') || t.startsWith('MOTOR') || (c.layer && c.layer.includes('MOTOR'))) {
      motor.push(c.id);
    } else {
      association.push(c.id);
    }
  }

  const list = [];
  if (sensory.length > 0) {
    list.push({ id: 1, label: 'SensoryColumn', cellIds: sensory, color: '#22d3ee' });
  }
  if (association.length > 0) {
    list.push({ id: 2, label: 'AssociationCortex', cellIds: association, color: '#34d399' });
  }
  if (motor.length > 0) {
    list.push({ id: 3, label: 'MotorEffectorCore', cellIds: motor, color: '#f43f5e' });
  }
  return list;
}

class OrganView {
  constructor(organDef, parentGroup) {
    this.id = organDef.id;
    this.label = organDef.label;
    this.cellIds = new Set(organDef.cellIds);
    this.colorHex = organDef.color || '#38bdf8';
    this.color = new THREE.Color(this.colorHex);

    this.group = new THREE.Group();
    parentGroup.add(this.group);

    this.center = new THREE.Vector3();
    this.radius = new THREE.Vector3(20, 20, 20);
    this.pulseTime = 0;
    this.highlightPulse = 0;

    // 1. 半透明生物组织外包膜 (Translucent Organ Morphological Envelope)
    this.envelopeMat = new THREE.MeshStandardMaterial({
      color: this.color,
      emissive: this.color,
      emissiveIntensity: 0.10,
      roughness: 0.18,
      metalness: 0.08,
      transparent: true,
      opacity: 0.13,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.envelopeMesh = new THREE.Mesh(_sharedEnvelopeGeo, this.envelopeMat);
    this.group.add(this.envelopeMesh);

    // 2. 赤道全息生物共振环 (Equatorial Holographic Lattice Ring)
    this.ringMat = new THREE.MeshBasicMaterial({
      color: this.color,
      transparent: true,
      opacity: 0.32,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.ringMesh = new THREE.Mesh(_sharedRingGeo, this.ringMat);
    this.ringMesh.rotation.x = Math.PI / 2;
    this.group.add(this.ringMesh);

    // 3. 3D 全息浮动生物器官铭牌 (3D Spatial Hologram Title)
    const badgeTex = getOrganBadgeTexture(this.label, this.cellIds.size, this.colorHex);
    this.badgeMat = new THREE.SpriteMaterial({
      map: badgeTex,
      transparent: true,
      depthWrite: false,
      opacity: 0.92
    });
    this.badge = new THREE.Sprite(this.badgeMat);
    this.badge.scale.set(64, 16, 1);
    this.group.add(this.badge);
  }

  update(org, t, rBase) {
    if (!org || !org.cells || this.cellIds.size === 0) return;

    let sx = 0, sy = 0, sz = 0, cnt = 0;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const c of org.cells) {
      if (this.cellIds.has(c.id)) {
        const x = c.x || 0, y = c.y || 0, z = c.z || 0;
        sx += x; sy += y; sz += z; cnt++;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
    }

    if (cnt === 0) return;

    this.center.set(sx / cnt, sy / cnt, sz / cnt);
    this.group.position.copy(this.center);

    // 组织包络膜的自适应物理外包边界
    const pad = Math.max(16.0, rBase * 2.8);
    const rx = Math.max(pad, (maxX - minX) * 0.5 + pad);
    const ry = Math.max(pad, (maxY - minY) * 0.5 + pad);
    const rz = Math.max(pad, (maxZ - minZ) * 0.5 + pad);
    this.radius.set(rx, ry, rz);

    // 有机代谢呼吸节奏 (Organic Metabolic Breathing)
    const breath = 1.0 + Math.sin(t * 1.6 + this.id * 0.8) * 0.035;
    this.envelopeMesh.scale.set(rx * breath, ry * breath, rz * breath);
    this.ringMesh.scale.set(rx * 1.03 * breath, rz * 1.03 * breath, 1.0);

    // 全息浮动铭牌悬浮在器官正上方
    this.badge.position.set(0, ry + 15, 0);

    // 高亮脉冲衰减
    if (this.highlightPulse > 0) {
      this.highlightPulse -= 0.02;
      if (this.highlightPulse < 0) this.highlightPulse = 0;
      this.envelopeMat.opacity = 0.13 + this.highlightPulse * 0.35;
      this.envelopeMat.emissiveIntensity = 0.10 + this.highlightPulse * 1.5;
    }
  }

  pulse() {
    this.highlightPulse = 1.0;
  }

  dispose(parentGroup) {
    if (parentGroup) parentGroup.remove(this.group);
    if (this.envelopeMat) this.envelopeMat.dispose();
    if (this.ringMat) this.ringMat.dispose();
    if (this.badgeMat) this.badgeMat.dispose();
  }
}

export function updateOrganSystem(scene, org, t, isMacroView) {
  initOrganSystem(scene);
  if (!_isOrganVisible || !org || !org.cells) {
    if (_organRootGroup) _organRootGroup.visible = false;
    return;
  }
  _organRootGroup.visible = true;

  const organList = extractOrganList(org);
  const rBase = getCellWorldRadius(org);

  const currentIds = new Set();
  for (const def of organList) {
    currentIds.add(def.id);
    let view = _organViewsMap.get(def.id);
    if (!view) {
      view = new OrganView(def, _organRootGroup);
      _organViewsMap.set(def.id, view);
    } else {
      // 保持属性最新
      view.label = def.label;
      view.cellIds = new Set(def.cellIds);
    }
    view.update(org, t, rBase);
  }

  // 清理不存在的器官
  for (const [id, view] of _organViewsMap) {
    if (!currentIds.has(id)) {
      view.dispose(_organRootGroup);
      _organViewsMap.delete(id);
    }
  }
}

export function focusOrgan(organIdOrLabel) {
  for (const [id, view] of _organViewsMap) {
    if (id === organIdOrLabel || view.label === organIdOrLabel || String(id) === String(organIdOrLabel)) {
      view.pulse();
      camState.targetLookAt.copy(view.center);
      const maxDim = Math.max(view.radius.x, view.radius.y, view.radius.z);
      camState.targetCamR = Math.max(160, maxDim * 2.8);
      camState.isCamTransitioning = true;
      log(`[ORGAN_FOCUS] 已精准对焦器官微柱【${view.label}】(${view.cellIds.size} 胞)！`, true);
      return true;
    }
  }
  return false;
}
