/* ============================================================
 * spatial_bounds.js - 三维拓扑空间包围盒、尺度映射与全景视距计算
 * ============================================================ */
import { HARDWARE_COSMIC_SPECS, LIFEFORM_SCALES } from './config.js';

export const currentOrganismBounds = {
  center: { x: 0, y: 0, z: 0, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
  radius: 180,
  min: { x: -180, y: -180, z: -180, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
  max: { x: 180, y: 180, z: 180, set(x, y, z) { this.x = x; this.y = y; this.z = z; } },
  macroDist: 540,
  microDist: 180,
  cellScale: 12,
  volumeRatio: 0.000001,
  trueRadius: 14.0
};

export function getOrganismCellScale(data, org, currentSelectedOrgId) {
  if (data && data.cells_scale && Number(data.cells_scale) > 0) return Number(data.cells_scale);
  if (data && data.organism_id && LIFEFORM_SCALES[data.organism_id]) {
    return LIFEFORM_SCALES[data.organism_id];
  }
  if (currentSelectedOrgId && LIFEFORM_SCALES[currentSelectedOrgId]) {
    return LIFEFORM_SCALES[currentSelectedOrgId];
  }
  if (org && org.lastOrganismId && LIFEFORM_SCALES[org.lastOrganismId]) {
    return LIFEFORM_SCALES[org.lastOrganismId];
  }
  if (currentOrganismBounds && currentOrganismBounds.cellScale > 0) {
    return currentOrganismBounds.cellScale;
  }
  if (data && data.macro_cells && data.macro_cells > 100) return Number(data.macro_cells);
  return (org && org.cells && org.cells.length > 0) ? org.cells.length : 12;
}

export function updateOrganismBounds(org, data = null, cellViewsMap = null, currentSelectedOrgId = null) {
  if (!org || !org.cells || org.cells.length === 0) return;

  const cellScale = getOrganismCellScale(data, org, currentSelectedOrgId);
  const V_ratio = cellScale / HARDWARE_COSMIC_SPECS.maxCellCapacity;
  const R_raw = (HARDWARE_COSMIC_SPECS.universeEdge * 0.5) * Math.cbrt(V_ratio);
  const R_min = 14.0;
  const R_max = 440.0;
  const R_true = Math.max(R_min, Math.min(R_max, R_raw));

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  for (const c of org.cells) {
    if (c._rawX === undefined) {
      c._rawX = (c.x !== undefined) ? c.x : 0;
      c._rawY = (c.y !== undefined) ? c.y : 0;
      c._rawZ = (c.z !== undefined) ? c.z : 0;
    }
    const x = c._rawX, y = c._rawY, z = c._rawZ;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const dx = Math.max(1.0, maxX - minX);
  const dy = Math.max(1.0, maxY - minY);
  const dz = Math.max(1.0, maxZ - minZ);
  const rawRad = Math.max(10.0, Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5);

  const targetVisRadius = Math.max(160.0, Math.min(240.0, rawRad));
  const scale = targetVisRadius / rawRad;

  for (const c of org.cells) {
    c.x = (c._rawX - cx) * scale;
    c.y = (c._rawY - cy) * scale;
    c.z = (c._rawZ - cz) * scale;
    if (cellViewsMap) {
      const v = cellViewsMap.get(c.id);
      if (v) {
        v.targetX = c.x;
        v.targetY = c.y;
        v.targetZ = c.z;
      }
    }
  }

  currentOrganismBounds.center.set(0, 0, 0);
  currentOrganismBounds.radius = targetVisRadius;
  currentOrganismBounds.min.set(-targetVisRadius, -targetVisRadius, -targetVisRadius);
  currentOrganismBounds.max.set(targetVisRadius, targetVisRadius, targetVisRadius);
  currentOrganismBounds.macroDist = Math.max(480, targetVisRadius * 2.6);
  currentOrganismBounds.microDist = Math.max(140, targetVisRadius * 0.7);
  currentOrganismBounds.cellScale = cellScale;
  currentOrganismBounds.volumeRatio = V_ratio;
  currentOrganismBounds.trueRadius = R_true;

  const radEl = document.getElementById("st-universe-rad");
  if (radEl) radEl.textContent = `${R_true.toFixed(1)} m`;
  const volEl = document.getElementById("st-universe-vol");
  if (volEl) {
    const pct = V_ratio * 100;
    volEl.textContent = pct >= 0.01 ? `${pct.toFixed(2)}%` : `${pct.toFixed(6)}%`;
  }
}

export function getCellWorldRadius(org) {
  const R = (currentOrganismBounds && currentOrganismBounds.radius) || 180;
  const n = (org && org.cells && org.cells.length) ? org.cells.length : 12;
  const cellScale = (currentOrganismBounds && currentOrganismBounds.cellScale) || n;
  if (cellScale >= 10000000) return 2.0;
  if (cellScale >= 1000000) return 2.6;
  return Math.max(2.8, Math.min(6.5, (R / Math.cbrt(n)) * 0.42));
}
