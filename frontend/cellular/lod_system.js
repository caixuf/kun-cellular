/* ============================================================
 * lod_system.js - 空间哈希、自适应点云流形与像素级 LOD 裁剪调度
 * ============================================================ */
import * as THREE from 'three';
import { scene, camera } from './scene_setup.js';
import { org as defaultOrg } from './organism_model.js';
import { currentOrganismBounds } from './spatial_bounds.js';
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

export function getMajorSynapseKeys(org = defaultOrg, bounds = currentOrganismBounds) {
  const o = org || defaultOrg;
  const b = bounds || currentOrganismBounds;
  if (!o || !o.syns || o.syns.length === 0) return [];
  const curId = o.lastOrganismId || (b && b.organismId) || '';
  if (cachedMajorSynKeys && lastMajorSynOrgId === curId && lastMajorSynCount === o.syns.length) {
    return cachedMajorSynKeys;
  }

  const activeSyns = o.syns.filter(s => s.active !== false);
  const candidates = activeSyns.length > 0 ? activeSyns : o.syns;

  const sorted = candidates.slice().sort((a, b) => {
    const wa = Math.abs(a.w !== undefined ? a.w : (a.weight !== undefined ? a.weight : 1.0));
    const wb = Math.abs(b.w !== undefined ? b.w : (b.weight !== undefined ? b.weight : 1.0));
    return wb - wa;
  });

  const top = sorted.slice(0, MAX_MACRO_SYNS);
  cachedMajorSynKeys = top.map(s => `${s.from}->${s.to}:${s.port || 0}`);
  lastMajorSynOrgId = curId;
  lastMajorSynCount = o.syns.length;
  return cachedMajorSynKeys;
}

export function computeAdaptiveLodCount(bounds = currentOrganismBounds, org = defaultOrg) {
  const b = bounds || currentOrganismBounds;
  const o = org || defaultOrg;
  const cellScale = (b && b.cellScale) || ((o && o.cells) ? o.cells.length : 1);
  if (cellScale >= 100000000) return 90000;
  if (cellScale >= 10000000) return 60000;
  if (cellScale >= 1000000) return 40000;
  if (cellScale >= 1000) return 15000;
  const nCells = (o && o.cells) ? o.cells.length : 1;
  const nSyns = (o && o.syns) ? o.syns.length : 0;
  return Math.min(30000, Math.max(1500, Math.round(nCells * 110 + nSyns * 45)));
}

export function initLODCloud(s = scene, o = defaultOrg, b = currentOrganismBounds, count = null) {
  const scn = s || scene;
  const orgObj = o || defaultOrg;
  const bnds = b || currentOrganismBounds;

  if (count === null || count === undefined) {
    count = computeAdaptiveLodCount(bnds, orgObj);
  }
  if (lodPointsMesh) {
    scn.remove(lodPointsMesh);
    if (lodPointsMesh.geometry) lodPointsMesh.geometry.dispose();
    if (lodPointsMesh.material) lodPointsMesh.material.dispose();
  }
  const geo = new THREE.BufferGeometry();
  lodPositions = new Float32Array(count * 3);
  lodColors = new Float32Array(count * 3);

  const nCells = (orgObj && orgObj.cells && orgObj.cells.length) ? orgObj.cells.length : 1;
  const synList = (orgObj && orgObj.syns && orgObj.syns.length) ? orgObj.syns : [];
  const cellsMap = new Map();
  if (orgObj && orgObj.cells) {
    for (const c of orgObj.cells) cellsMap.set(c.id, c);
  }

  const spreadScale = Math.min(2.5, Math.max(0.12, (bnds ? bnds.radius : 180.0) / 180.0));

  for (let i = 0; i < count; ++i) {
    let px = 0, py = 0, pz = 0;
    let colHex = 0x38bdf8;

    const isAxon = (synList.length > 0) && (Math.random() < 0.65);

    if (isAxon) {
      const syn = synList[Math.floor(Math.random() * synList.length)];
      const cFrom = cellsMap.get(syn.from) || orgObj.cells[0];
      const cTo = cellsMap.get(syn.to) || orgObj.cells[Math.min(1, nCells - 1)];

      const t = Math.random();
      const fx = cFrom ? (cFrom.x || 0) : 0;
      const fy = cFrom ? (cFrom.y || 0) : 0;
      const fz = cFrom ? (cFrom.z || 0) : 0;
      const tx = cTo ? (cTo.x || 0) : 0;
      const ty = cTo ? (cTo.y || 0) : 0;
      const tz = cTo ? (cTo.z || 0) : 0;

      const sag = Math.sin(t * Math.PI) * (12.0 * spreadScale);
      px = fx + (tx - fx) * t + (Math.random() - 0.5) * (6.0 * spreadScale);
      py = fy + (ty - fy) * t + sag + (Math.random() - 0.5) * (6.0 * spreadScale);
      pz = fz + (tz - fz) * t + (Math.random() - 0.5) * (6.0 * spreadScale);

      const w = (syn && syn.w !== undefined) ? syn.w : 1.0;
      colHex = w >= 0 ? 0x38bdf8 : 0xf43f5e;
    } else {
      const c = (orgObj && orgObj.cells && orgObj.cells.length) ? orgObj.cells[Math.floor(Math.random() * orgObj.cells.length)] : null;
      const cx = c ? (c.x || 0) : 0;
      const cy = c ? (c.y || 0) : 0;
      const cz = c ? (c.z || 0) : 0;

      const r = (Math.random() * 18.0 + 2.0) * spreadScale;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      px = cx + r * Math.sin(phi) * Math.cos(theta);
      py = cy + r * Math.sin(phi) * Math.sin(theta);
      pz = cz + r * Math.cos(phi);

      const fam = c ? FAMILY(c.type) : 'metabolic';
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
  scn.add(lodPointsMesh);
}

export function buildFullPointCloud(s = scene, o = defaultOrg, b = currentOrganismBounds) {
  const scn = s || scene;
  const orgObj = o || defaultOrg;
  const bnds = b || currentOrganismBounds;

  if (!orgObj || !orgObj.cells) return;
  const n = orgObj.cells.length;
  if (n === 0) return;
  const cellScale = (bnds && bnds.cellScale) || n;

  if (cellScale >= 200 || n >= 80) {
    initLODCloud(scn, orgObj, bnds, computeAdaptiveLodCount(bnds, orgObj));
    return;
  }

  if (lodPointsMesh) {
    scn.remove(lodPointsMesh);
    if (lodPointsMesh.geometry) lodPointsMesh.geometry.dispose();
    if (lodPointsMesh.material) lodPointsMesh.material.dispose();
    lodPointsMesh = null;
  }
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = orgObj.cells[i];
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
  scn.add(lodPointsMesh);
}

export function buildCellSpatialHash(o = defaultOrg, b = currentOrganismBounds) {
  const orgObj = o || defaultOrg;
  const bnds = b || currentOrganismBounds;
  if (!orgObj || !orgObj.cells) return;

  cellSpatialHash = new Map();
  cellSynAdj = new Map();
  synKeyMap = new Map();
  const B = Math.max(20, ((bnds && bnds.radius) || 180) / 12);
  for (const c of orgObj.cells) {
    const bx = Math.floor((c.x || 0) / B), by = Math.floor((c.y || 0) / B), bz = Math.floor((c.z || 0) / B);
    const key = bx + ',' + by + ',' + bz;
    let arr = cellSpatialHash.get(key);
    if (!arr) { arr = []; cellSpatialHash.set(key, arr); }
    arr.push(c);
  }
  if (orgObj.syns) {
    for (const s of orgObj.syns) {
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
}

export function rebuildViews(s = scene, o = defaultOrg, b = currentOrganismBounds) {
  const scn = s || scene;
  const orgObj = o || defaultOrg;
  const bnds = b || currentOrganismBounds;

  for (const v of cellViewsMap.values()) { v.group.visible = false; cellViewPool.push(v); }
  cellViewsMap.clear();
  for (const v of synViewsMap.values()) { v.group.visible = false; synViewPool.push(v); }
  synViewsMap.clear();

  buildFullPointCloud(scn, orgObj, bnds);
  buildCellSpatialHash(orgObj, bnds);

  cachedMajorSynKeys = null;
  if (orgObj && orgObj.cells) {
    orgObj.cellMap = new Map(orgObj.cells.map(c => [c.id, c]));
  }
  views.cells = [];
  views.syns = [];
}

const _lodCamPos = new THREE.Vector3();
const _lodCellPos = new THREE.Vector3();
const _lodSphere = new THREE.Sphere();
const _lodCandidates = [];

export function updateDetailLOD(arg1, arg2, arg3, arg4, arg5) {
  let frustum = arg1;
  let scn = scene, cam = camera, orgObj = defaultOrg, bnds = currentOrganismBounds;

  if (arg1 && arg1.isScene) {
    scn = arg1;
    cam = arg2;
    frustum = arg3;
    orgObj = arg4 || defaultOrg;
    bnds = arg5 || currentOrganismBounds;
  } else {
    frustum = arg1;
    if (arg2 && arg2.isScene) scn = arg2;
    if (arg3 && arg3.isCamera) cam = arg3;
    if (arg4 && arg4.cells) orgObj = arg4;
    if (arg5) bnds = arg5;
  }

  if (!orgObj || !orgObj.cells) return;
  const n = orgObj.cells.length;
  if (n === 0) return;
  if (!cam || !frustum) return;

  _lodCamPos.copy(cam.position);

  const cellRadius = getCellWorldRadius(orgObj);
  const projScale = window.innerHeight / (2 * Math.tan(THREE.MathUtils.degToRad(cam.fov) * 0.5));
  const solidMaxDist = (2.0 * cellRadius * projScale) / MIN_CELL_PIXELS;
  const orgR = (bnds && bnds.radius) || 180;
  const camToCtr = _lodCamPos.distanceTo(bnds.center);

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
      const c = (orgObj.cellMap && orgObj.cellMap.get(cd.id)) || orgObj.cells.find(x => x.id === cd.id);
      if (!c) continue;
      v = cellViewPool.length ? cellViewPool.pop() : new CellView(c, scn, orgObj);
      v.updateCell(c, orgObj);
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

  const majorKeys = getMajorSynapseKeys(orgObj, bnds);
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
      let v = synViewPool.length ? synViewPool.pop() : new SynapseView(syn, orgObj, scn);
      v.updateSyn(syn, orgObj);
      synViewsMap.set(key, v);
    }
    const v = synViewsMap.get(key);
    if (v) v.group.visible = true;
  }
  views.syns = Array.from(synViewsMap.values());
}
