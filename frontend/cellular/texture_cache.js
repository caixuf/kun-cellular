/* ============================================================
 * texture_cache.js - 全局发光粒子纹理与全息标签画布复用池
 * ============================================================ */
import * as THREE from 'three';

let cachedGlowTex = null;

export function getGlowTexture() {
  if (cachedGlowTex) return cachedGlowTex;
  const cv = document.createElement('canvas');
  cv.width = 64;
  cv.height = 64;
  const g = cv.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 1, 32, 32, 30);
  grad.addColorStop(0, 'rgba(255,255,255,1.0)');
  grad.addColorStop(0.30, 'rgba(255,255,255,0.75)');
  grad.addColorStop(0.65, 'rgba(255,255,255,0.22)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  cachedGlowTex = new THREE.CanvasTexture(cv);
  return cachedGlowTex;
}

const LABEL_TEX_CACHE = new Map();

export function getLabelTexture(typeStr) {
  if (LABEL_TEX_CACHE.has(typeStr)) return LABEL_TEX_CACHE.get(typeStr);
  const cv = document.createElement('canvas');
  cv.width = 256;
  cv.height = 64;
  const g = cv.getContext('2d');
  g.font = '600 24px monospace';
  g.fillStyle = 'rgba(226,232,240,0.95)';
  g.textAlign = 'center';
  g.fillText(typeStr, 128, 40);
  const tex = new THREE.CanvasTexture(cv);
  LABEL_TEX_CACHE.set(typeStr, tex);
  return tex;
}
