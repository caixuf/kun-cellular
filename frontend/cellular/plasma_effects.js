/* ============================================================
 * plasma_effects.js - 空气介质流体相变、分形闪电与等离子击穿放电
 * ============================================================ */
import * as THREE from 'three';
import { scene, camera, renderer, ambientLight, lightningGroup, airParticleCloud } from './scene_setup.js';
import { playIonizationSpark, playChicxulubAtmosphericThunder } from './audio_system.js';
import { log } from './network_sync.js';

export let currentFluidPhase = 'aero'; // 'aero' | 'hydro' | 'vacuum'
window.currentFluidPhase = currentFluidPhase;

export function setFluidPhase(phase) {
  currentFluidPhase = phase;
  window.currentFluidPhase = phase;
  document.querySelectorAll('#btn-phase-aero, #btn-phase-hydro, #btn-phase-vacuum').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`btn-phase-${phase}`);
  if (btn) btn.classList.add('active');

  const badge = document.getElementById('st-fluid-phase-badge');
  const nameEl = document.getElementById('st-fluid-phase-name');
  const densEl = document.getElementById('st-fluid-density');
  const bdEl = document.getElementById('st-fluid-breakdown');
  const viscEl = document.getElementById('st-fluid-viscosity');
  const headVal = document.getElementById('fluid-field-val');

  if (phase === 'aero') {
    if (badge) { badge.textContent = 'AERO // 气相'; badge.style.color = 'var(--cyan)'; }
    if (nameEl) nameEl.textContent = '气相分子气溶胶';
    if (densEl) densEl.textContent = '1.225 kg/m³';
    if (bdEl) bdEl.textContent = '3.0 kV/mm';
    if (viscEl) viscEl.textContent = '0.018 mPa·s';
    if (headVal) headVal.textContent = 'AERO 3.0 kV/mm';

    if (scene && scene.fog) scene.fog.color.setHex(0x04070d);
    if (renderer) renderer.setClearColor(0x04070d);

    if (airParticleCloud && airParticleCloud.geometry) {
      airParticleCloud.visible = true;
      const colArr = airParticleCloud.geometry.attributes.color.array;
      for (let i = 0; i < colArr.length / 3; i++) {
        const isIon = Math.random() < 0.25;
        const c = isIon ? new THREE.Color(0x38bdf8) : new THREE.Color(0x94a3b8);
        colArr[i * 3] = c.r; colArr[i * 3 + 1] = c.g; colArr[i * 3 + 2] = c.b;
      }
      airParticleCloud.geometry.attributes.color.needsUpdate = true;
    }
    log('[FLUID] 已切换为【连续气相分子介质】(空气动力学对流，击穿场强 3.0 kV/mm)', true);
  } else if (phase === 'hydro') {
    if (badge) { badge.textContent = 'HYDRO // 水相'; badge.style.color = '#38bdf8'; }
    if (nameEl) nameEl.textContent = '原始水生物圈分子汤';
    if (densEl) densEl.textContent = '1000.0 kg/m³';
    if (bdEl) bdEl.textContent = '0.15 kV/mm (高电导率)';
    if (viscEl) viscEl.textContent = '1.002 mPa·s (水力粘度)';
    if (headVal) headVal.textContent = 'HYDRO 0.15 kV/mm';

    if (scene && scene.fog) scene.fog.color.setHex(0x021627);
    if (renderer) renderer.setClearColor(0x021627);

    if (airParticleCloud && airParticleCloud.geometry) {
      airParticleCloud.visible = true;
      const colArr = airParticleCloud.geometry.attributes.color.array;
      for (let i = 0; i < colArr.length / 3; i++) {
        const isBio = Math.random() < 0.45;
        const c = isBio ? new THREE.Color(0x10b981) : new THREE.Color(0x00f0ff);
        colArr[i * 3] = c.r; colArr[i * 3 + 1] = c.g; colArr[i * 3 + 2] = c.b;
      }
      airParticleCloud.geometry.attributes.color.needsUpdate = true;
    }
    log('[FLUID] 已切换为【连续液相水生物圈】(原始细胞外水汤，高电导率离子流动，声速 1500m/s)', true);
  } else if (phase === 'vacuum') {
    if (badge) { badge.textContent = 'VACUUM // 真空'; badge.style.color = 'var(--rose)'; }
    if (nameEl) nameEl.textContent = '绝对深空虚空';
    if (densEl) densEl.textContent = '0.000 kg/m³';
    if (bdEl) bdEl.textContent = '∞ (绝缘极限)';
    if (viscEl) viscEl.textContent = '0.000 mPa·s';
    if (headVal) headVal.textContent = 'VACUUM 0 kg/m³';

    if (scene && scene.fog) scene.fog.color.setHex(0x010204);
    if (renderer) renderer.setClearColor(0x010204);

    if (airParticleCloud) airParticleCloud.visible = false;
    log('[FLUID] 已切换为【深空真空临界态】(分子密度为零，绝对声学静默)', true);
  }
}

export function generateFractalLightningPoints(start, end, displacement, iterations) {
  let segments = [{ start: start.clone(), end: end.clone() }];

  for (let iter = 0; iter < iterations; iter++) {
    const newSegs = [];
    const scale = displacement * Math.pow(0.55, iter);
    for (const seg of segments) {
      const mid = seg.start.clone().add(seg.end).multiplyScalar(0.5);
      mid.x += (Math.random() - 0.5) * scale;
      mid.y += (Math.random() - 0.5) * scale;
      mid.z += (Math.random() - 0.5) * scale;
      newSegs.push({ start: seg.start, end: mid });
      newSegs.push({ start: mid, end: seg.end });
    }
    segments = newSegs;
  }

  const points = [];
  if (segments.length > 0) points.push(segments[0].start);
  for (const seg of segments) points.push(seg.end);
  return points;
}

export function spawnDielectricBreakdownArc(startVec, endVec, colorHex = 0x38bdf8, intensity = 1.0) {
  if (!lightningGroup) return;
  const pts = generateFractalLightningPoints(startVec, endVec, 34 * intensity, 4);
  if (pts.length < 2) return;

  const arcObj = new THREE.Group();

  const curve = new THREE.CatmullRomCurve3(pts);
  const coreGeo = new THREE.TubeGeometry(curve, Math.max(16, pts.length * 2), 1.3 * intensity, 4, false);
  const coreMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.95
  });
  const coreMesh = new THREE.Mesh(coreGeo, coreMat);
  arcObj.add(coreMesh);

  const auraGeo = new THREE.TubeGeometry(curve, Math.max(16, pts.length * 2), 4.2 * intensity, 4, false);
  const auraMat = new THREE.MeshBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending
  });
  const auraMesh = new THREE.Mesh(auraGeo, auraMat);
  arcObj.add(auraMesh);

  const sparkGeo = new THREE.SphereGeometry(2.8 * intensity, 6, 6);
  const sparkMat = new THREE.MeshBasicMaterial({ color: 0xffffff, blending: THREE.AdditiveBlending });
  const s1 = new THREE.Mesh(sparkGeo, sparkMat); s1.position.copy(startVec);
  const s2 = new THREE.Mesh(sparkGeo, sparkMat); s2.position.copy(endVec);
  arcObj.add(s1); arcObj.add(s2);

  lightningGroup.add(arcObj);

  const startT = performance.now();
  const duration = 360;
  function stepArc() {
    const elapsed = performance.now() - startT;
    if (elapsed < duration) {
      const p = 1.0 - elapsed / duration;
      const flicker = (Math.random() * 0.4 + 0.6);
      coreMat.opacity = p * 0.95 * flicker;
      auraMat.opacity = p * 0.85 * flicker;
      requestAnimationFrame(stepArc);
    } else {
      lightningGroup.remove(arcObj);
      coreGeo.dispose();
      coreMat.dispose();
      auraGeo.dispose();
      auraMat.dispose();
      sparkGeo.dispose();
      sparkMat.dispose();
    }
  }
  stepArc();
}

export function triggerExtinctionLightningBurst(views) {
  playChicxulubAtmosphericThunder();
  if (!views || !views.cells || views.cells.length === 0) return;
  const targetPoints = views.cells.map(v => new THREE.Vector3(v.cell.x, v.cell.y, v.cell.z));
  if (targetPoints.length === 0) return;

  const boltCount = 12;
  for (let i = 0; i < boltCount; i++) {
    setTimeout(() => {
      const skyStart = new THREE.Vector3(
        (Math.random() - 0.5) * 700,
        450 + Math.random() * 180,
        (Math.random() - 0.5) * 700
      );
      const target = targetPoints[Math.floor(Math.random() * targetPoints.length)].clone();
      target.x += (Math.random() - 0.5) * 50;
      target.y += (Math.random() - 0.5) * 50;
      target.z += (Math.random() - 0.5) * 50;

      const colors = [0x38bdf8, 0xa855f7, 0xf43f5e, 0xffffff];
      spawnDielectricBreakdownArc(skyStart, target, colors[i % colors.length], 2.4);
      playIonizationSpark(0.85);
    }, i * 35);
  }
}

export function triggerExtinctionVisualShock(views) {
  const shockEl = document.createElement("div");
  shockEl.className = "screen-shockwave";
  document.body.appendChild(shockEl);
  setTimeout(() => shockEl.remove(), 800);

  triggerExtinctionLightningBurst(views);

  if (ambientLight && camera) {
    const origAmb = ambientLight.intensity;
    ambientLight.intensity = Math.min(5.0, origAmb * 3.5);
    let shakeCount = 20;
    const origPos = camera.position.clone();
    function doShake() {
      if (shakeCount > 0) {
        camera.position.x += (Math.random() - 0.5) * 14 * (shakeCount / 20);
        camera.position.y += (Math.random() - 0.5) * 14 * (shakeCount / 20);
        camera.position.z += (Math.random() - 0.5) * 14 * (shakeCount / 20);
        ambientLight.intensity = origAmb + (shakeCount / 20) * 2.5;
        shakeCount--;
        requestAnimationFrame(doShake);
      } else {
        camera.position.copy(origPos);
        ambientLight.intensity = origAmb;
      }
    }
    doShake();
  }
}
