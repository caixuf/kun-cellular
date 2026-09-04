/* ============================================================
 * lod_system.js - 空间哈希、自适应点云流形与像素级 LOD 裁剪调度
 * ============================================================ */
import * as THREE from 'three';
import { FAMILY, FAMILY_COLOR } from './config.js';
import { getGlowTexture } from './texture_cache.js';
import { getCellWorldRadius } from './spatial_bounds.js';
import { CellView } from './cell_view.js';
import { SynapseView } from './synapse_view.js';

export const MIN_CELL_PIXELS = 26.0;
export const MAX_SOLID_CELLS = 300;

export const cellViewPool = [];
export const synViewPool = [];
export const cellViewsMap = new Map();
export const synViewsMap = new Map();
export const views = { cells: [], syns: [] };

export let cellSpatialHash = new Map();
export let cellSynAdj = new Map();
export let synKeyMap = new Map();
export let lodPointsMesh = null;

let lodPositions = null;
let lodColors = null;

const MAX_MACRO_SYNS = 120;
let cachedMajorSynKeys = null;
let lastMajorSynOrgId = null;
let lastMajorSynCount = 0;

export function getMajorSynapseKeys(org, bounds) {
  if (!org || !org.syns || org.syns.length === 0) return [];
  const curId = org.lastOrganismId || (bounds && bounds.organismId) || '';
  if (cachedMajorSynKeys && lastMajorSynOrgId === curId && lastMajorSynCount === org.syns.length) {
    return cachedMajorSynKeys;
  }

  const activeSyns = org.syns.filter(s => s.active !== false);
  const candidates = activeSyns.length > 0 ? activeSyns : org.syns;

  const sorted = candidates.slice().sort((a, b) => {
    const wa = Math.abs(a.w !== undefined ? a.w : (a.weight !== undefined ? a.weight : 1.0));
    const wb = Math.abs(b.w !== undefined ? b.w : (b.weight !== undefined ? b.weight : 1.0));
    return wb - wa;
  });

  const top = sorted.slice(0, MAX_MACRO_SYNS);
  cachedMajorSynKeys = top.map(s => `${s.from}->${s.to}:${s.port || 0}`);
  lastMajorSynOrgId = curId;
  lastMajorSynCount = org.syns.length;
  return cachedMajorSynKeys;
}

export function computeAdaptiveLodCount(bounds, org) {
  const cellScale = (bounds && bounds.cellScale) || ((org && org.cells) ? org.cells.length : 1);
  if (cellScale >= 100000000) return 90000;
  if (cellScale >= 10000000) return 60000;
  if (cellScale >= 1000000) return 40000;
  if (cellScale >= 1000) return 15000;
  const nCells = (org && org.cells) ? org.cells.length : 1;
  const nSyns = (org && org.syns) ? org.syns.length : 0;
  return Math.min(30000, Math.max(1500, Math.round(nCells * 110 + nSyns * 45)));
}

export function initLODCloud(scene, org, bounds, count = null) {
  if (count === null || count === undefined) {
    count = computeAdaptiveLodCount(bounds, org);
  }
  if (lodPointsMesh) {
    scene.remove(lodPointsMesh);
    if (lodPointsMesh.geometry) lodPointsMesh.geometry.dispose();
    if (lodPointsMesh.material) lodPointsMesh.material.dispose();
  }
  const geo = new THREE.BufferGeometry();
  lodPositions = new Float32Array(count * 3);
  lodColors = new Float32Array(count * 3);

  const nCells = (org && org.cells && org.cells.length) ? org.cells.length : 1;
  const synList = (org && org.syns && org.syns.length) ? org.syns : [];
  const cellsMap = new Map();
  if (org && org.cells) {
    for (const c of org.cells) cellsMap.set(c.id, c);
  }

  const spreadScale = Math.min(2.5, Math.max(0.12, (bounds ? bounds.radius : 180.0) / 180.0));

  for (let i = 0; i < count; ++i) {
    let px = 0, py = 0, pz = 0;
    let colHex = 0x38bdf8;

    const isAxon = (synList.length > 0) && (Math.random() < 0.65);

    if (isAxon) {
      const syn = synList[Math.floor(Math.random() * synList.length)];
      const cFrom = cellsMap.get(syn.from) || org.cells[0];
      const cTo = cellsMap.get(syn.to) || org.cells[Math.min(1, nCells - 1)];

      const t = Math.random();
      const midX = (cFrom.x + cTo.x) * 0.5 + (Math.sin(syn.from * 3.1 + syn.to * 1.7)) * 12.0 * spreadScale;
      const midY = (cFrom.y + cTo.y) * 0.5 + (Math.cos(syn.from * 2.3 + syn.to * 4.1)) * 12.0 * spreadScale;
      const midZ = ((cFrom.z || 0) + (cTo.z || 0)) * 0.5 + (Math.sin(syn.from * 1.9 + syn.to * 2.8)) * 12.0 * spreadScale;

      const oneMinusT = 1.0 - t;
      px = oneMinusT * oneMinusT * cFrom.x + 2.0 * oneMinusT * t * midX + t * t * cTo.x + (Math.random() - 0.5) * 5.0 * spreadScale;
      py = oneMinusT * oneMinusT * cFrom.y + 2.0 * oneMinusT * t * midY + t * t * cTo.y + (Math.random() - 0.5) * 5.0 * spreadScale;
      pz = oneMinusT * oneMinusT * (cFrom.z || 0) + 2.0 * oneMinusT * t * midZ + t * t * (cTo.z || 0) + (Math.random() - 0.5) * 5.0 * spreadScale;

      const fam = FAMILY(cTo.type);
      colHex = FAMILY_COLOR[fam] || 0x38bdf8;
    } else {
      const c = org.cells[Math.floor(Math.random() * nCells)];
      const u = Math.random();
      const phi = Math.acos(1 - 2 * Math.random());
      const theta = Math.random() * Math.PI * 2;
      const r = (1.5 + Math.pow(u, 0.4) * (18.0 + (c.glow || 0) * 8.0)) * spreadScale;

      px = c.x + r * Math.sin(phi) * Math.cos(theta);
      py = c.y + r * Math.sin(phi) * Math.sin(theta);
      pz = (c.z || 0) + r * Math.cos(phi);

      const fam = FAMILY(c.type);
      colHex = FAMILY_COLOR[fam] || 0x38bdf8;
    }

    lodPositions[i * 3]     = px;
    lodPositions[i * 3 + 1] = py;
    lodPositions[i * 3 + 2] = pz;

    const baseCol = new THREE.Color(colHex);
    lodColors[i * 3]     = baseCol.r;
    lodColors[i * 3 + 1] = baseCol.g;
    lodColors[i * 3 + 2] = baseCol.b;
  }

  geo.setAttribute("position", new THREE.BufferAttribute(lodPositions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(lodColors, 3));

  const mat = new THREE.PointsMaterial({
    size: 2.2 * spreadScale,
    vertexColors: true,
    map: getGlowTexture(),
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true
  });

  lodPointsMesh = new THREE.Points(geo, mat);
  scene.add(lodPointsMesh);
}

export function buildFullPointCloud(scene, org, bounds) {
  const n = org.cells.length;
  if (n === 0) return;
  const cellScale = (bounds && bounds.cellScale) || n;

  if (cellScale >= 200 || n >= 80) {
    initLODCloud(scene, org, bounds, computeAdaptiveLodCount(bounds, org));
    return;
  }

  if (lodPointsMesh) {
    scene.remove(lodPointsMesh);
    if (lodPointsMesh.geometry) lodPointsMesh.geometry.dispose();
    if (lodPointsMesh.material) lodPointsMesh.material.dispose();
    lodPointsMesh = null;
  }
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = org.cells[i];
    pos[i * 3]     = c.x || 0;
    pos[i * 3 + 1] = c.y || 0;
    pos[i * 3 + 2] = c.z || 0;
    const hex = FAMILY_COLOR[FAMILY(c.type)] || 0x38bdf8;
    col[i * 3]     = ((hex >> 16) & 255) / 255;
    col[i * 3 + 1] = ((hex >> 8) & 255) / 255;
    col[i * 3 + 2] = (hex & 255) / 255;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const mat = new THREE.PointsMaterial({
    size: 4.2,
    vertexColors: true,
    map: getGlowTexture(),
    transparent: true,
    opacity: 0.75,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    sizeAttenuation: true
  });
  lodPointsMesh = new THREE.Points(geo, mat);
  lodPointsMesh.frustumCulled = false;
  scene.add(lodPointsMesh);
}

export function buildCellSpatialHash(org, bounds) {
  cellSpatialHash = new Map();
  cellSynAdj = new Map();
  synKeyMap = new Map();
  const B = Math.max(20, ((bounds && bounds.radius) || 180) / 12);
  for (const c of org.cells) {
    const bx = Math.floor((c.x || 0) / B), by = Math.floor((c.y || 0) / B), bz = Math.floor((c.z || 0) / B);
    const key = bx + ',' + by + ',' + bz;
    let arr = cellSpatialHash.get(key);
    if (!arr) { arr = []; cellSpatialHash.set(key, arr); }
    arr.push(c);
  }
  for (const s of org.syns) {
    const key = `${s.from}->${s.to}:${s.port || 0}`;
    synKeyMap.set(key, s);
    let outList = cellSynAdj.get(s.from);
    if (!outList) { outList = []; cellSynAdj.set(s.from, outList); }
    outList.push(key);
    let inList = cellSynAdj.get(s.to);
    if (!inList) { inList = []; cellSynAdj.set(s.to, inList); }
    inList.push(key);
  }
}

export function rebuildViews(scene, org, bounds) {
  for (const v of cellViewsMap.values()) { v.group.visible = false; cellViewPool.push(v); }
  cellViewsMap.clear();
  for (const v of synViewsMap.values()) { v.group.visible = false; synViewPool.push(v); }
  synViewsMap.clear();

  buildFullPointCloud(scene, org, bounds);
  buildCellSpatialHash(org, bounds);

  cachedMajorSynKeys = null;
  org.cellMap = new Map(org.cells.map(c => [c.id, c]));
  views.cells = [];
  views.syns = [];
}

const _lodCamPos = new THREE.Vector3();
const _lodCellPos = new THREE.Vector3();
const _lodSphere = new THREE.Sphere();
const _lodCandidates = [];

export function updateDetailLOD(scene, camera, frustum, org, bounds) {
  const n = org.cells.length;
  if (n === 0) return;
  _lodCamPos.copy(camera.position);

  const cellRadius = getCellWorldRadius(org);
  const projScale = window.innerHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5));
  const solidMaxDist = (2.0 * cellRadius * projScale) / MIN_CELL_PIXELS;
  const orgR = (bounds && bounds.radius) || 180;
  const camToCtr = _lodCamPos.distanceTo(bounds.center);

  // 1. 实体细胞筛选
  _lodCandidates.length = 0;
  if (camToCtr <= solidMaxDist + orgR) {
    for (const arr of cellSpatialHash.values()) {
      for (const c of arr) {
        _lodCellPos.set(c.x || 0, c.y || 0, c.z || 0);
        const d = _lodCamPos.distanceTo(_lodCellPos);
        if (d > solidMaxDist) continue;
        _lodSphere.center.copy(_lodCellPos);
        _lodSphere.radius = 52.0;
        if (!frustum.intersectsSphere(_lodSphere)) continue;
        _lodCandidates.push({ id: c.id, d });
      }
    }
    _lodCandidates.sort((a, b) => a.d - b.d);
    if (_lodCandidates.length > MAX_SOLID_CELLS) _lodCandidates.length = MAX_SOLID_CELLS;
  }

  const wantIds = new Set();
  for (const cd of _lodCandidates) wantIds.add(cd.id);

  for (const [id, v] of cellViewsMap) {
    if (!wantIds.has(id)) { v.group.visible = false; cellViewPool.push(v); cellViewsMap.delete(id); }
  }
  for (const cd of _lodCandidates) {
    let v = cellViewsMap.get(cd.id);
    if (!v) {
      const c = org.cellMap.get(cd.id);
      v = cellViewPool.length ? cellViewPool.pop() : new CellView(c, scene, org);
      v.updateCell(c, org);
      cellViewsMap.set(cd.id, v);
    }
    v.group.visible = true;
  }
  views.cells = Array.from(cellViewsMap.values());

  // 2. 突触 LOD：关联突触 + 点云模式下全脑活跃大型突触
  const wantSyn = new Set();

  if (wantIds.size > 0) {
    for (const id of wantIds) {
      const adj = cellSynAdj.get(id);
      if (adj) {
        for (const key of adj) {
          wantSyn.add(key);
          if (wantSyn.size >= 120) break;
        }
      }
      if (wantSyn.size >= 120) break;
    }
  }

  const majorKeys = getMajorSynapseKeys(org, bounds);
  for (const key of majorKeys) {
    wantSyn.add(key);
    if (wantSyn.size >= 140) break;
  }

  for (const [key, v] of synViewsMap) {
    if (!wantSyn.has(key)) { v.group.visible = false; synViewPool.push(v); synViewsMap.delete(key); }
  }
  for (const key of wantSyn) {
    if (!synViewsMap.has(key)) {
      const syn = synKeyMap.get(key);
      if (!syn) continue;
      let v = synViewPool.length ? synViewPool.pop() : new SynapseView(syn, org, scene);
      v.updateSyn(syn, org);
      synViewsMap.set(key, v);
    }
    const v = synViewsMap.get(key);
    if (v) v.group.visible = true;
  }
  views.syns = Array.from(synViewsMap.values());
}
