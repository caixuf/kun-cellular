/* ============================================================
 * camera_controller.js - 动力学相机控制器 (动量Lerp, 光标推进, 双击显微, 全景复位)
 * ============================================================ */
import * as THREE from 'three';

export const camState = {
  camTheta: 0,
  camPhi: Math.PI / 2,
  camR: 540,
  targetTheta: 0,
  targetPhi: Math.PI / 2,
  targetCamR: 540,
  targetLookAt: new THREE.Vector3(0, 0, 0),
  currentLookAt: new THREE.Vector3(0, 0, 0),
  isCamTransitioning: false,
  autoOrbitEnabled: false,
  shakeImpulse: 0
};

export function clampCamDistance(r, bounds) {
  const macro = (bounds && bounds.macroDist) || 540;
  return Math.max(15, Math.min(Math.max(3200, macro * 4), r));
}

export function cameraShake(strength = 4.0) {
  camState.shakeImpulse = Math.max(camState.shakeImpulse, strength);
}

export function setCameraDistance(r, bounds) {
  camState.targetCamR = clampCamDistance(r, bounds);
  camState.camR = camState.targetCamR;
  camState.isCamTransitioning = false;
}

export function setCameraTarget(v) {
  camState.targetLookAt.copy(v);
  camState.currentLookAt.copy(v);
}

export function focusOnCell(cellId, org, distance = 25) {
  const c = (org.cellMap && org.cellMap.get(cellId)) || (org.cells && org.cells[0]);
  if (!c) return;
  camState.targetLookAt.set(c.x || 0, c.y || 0, c.z || 0);
  camState.currentLookAt.copy(camState.targetLookAt);
  camState.targetCamR = distance;
  camState.camR = distance;
  camState.camTheta = 0;
  camState.targetTheta = 0;
  camState.camPhi = Math.PI / 2;
  camState.targetPhi = Math.PI / 2;
  camState.isCamTransitioning = false;
}

export function setCameraPreset(preset, bounds) {
  const dist = (bounds && bounds.macroDist) || 540;
  camState.targetCamR = dist;
  camState.isCamTransitioning = true;

  if (preset === 'front') {
    camState.targetTheta = 0;
    camState.targetPhi = Math.PI / 2;
  } else if (preset === 'top') {
    camState.targetTheta = 0;
    camState.targetPhi = 0.05;
  } else if (preset === 'side') {
    camState.targetTheta = Math.PI / 2;
    camState.targetPhi = Math.PI / 2;
  } else if (preset === 'dramatic') {
    camState.targetTheta = 0.785;
    camState.targetPhi = 1.1;
    camState.targetCamR = Math.max(140, dist * 0.65);
  } else if (preset === 'reset') {
    camState.targetLookAt.set(0, 0, 0);
    camState.targetTheta = 0;
    camState.targetPhi = Math.PI / 2;
  }
}

export function toggleAutoOrbit() {
  camState.autoOrbitEnabled = !camState.autoOrbitEnabled;
  const btn = document.getElementById('btn-auto-orbit');
  if (btn) btn.classList.toggle('active', camState.autoOrbitEnabled);
}

export function initCameraController(renderer, camera, getOrgFn, getViewsFn, getBoundsFn, logFn) {
  const raycaster = new THREE.Raycaster();
  const mouseNDC = new THREE.Vector2();
  const focalPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const _camForward = new THREE.Vector3();
  const _zoomHit = new THREE.Vector3();

  let dragging = false;
  let panDragging = false;
  let lx = 0, ly = 0;

  function panCameraByPixels(dx, dy) {
    const panScale = Math.max(0.05, camState.targetCamR / 480);
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    camera.matrix.extractBasis(right, up, new THREE.Vector3());
    camState.targetLookAt.addScaledVector(right, -dx * panScale);
    camState.targetLookAt.addScaledVector(up, dy * panScale);
  }

  renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());

  renderer.domElement.addEventListener('pointerdown', e => {
    lx = e.clientX; ly = e.clientY;
    const wantPan = e.button === 2 || e.button === 1 || e.altKey || e.ctrlKey;
    if (wantPan) {
      panDragging = true;
    } else if (e.button === 0) {
      dragging = true;
    }
    renderer.domElement.setPointerCapture?.(e.pointerId);
  });

  window.addEventListener('pointerup', () => { dragging = false; panDragging = false; });

  window.addEventListener('pointermove', e => {
    mouseNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouseNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
    const dx = e.clientX - lx;
    const dy = e.clientY - ly;
    if (panDragging) {
      panCameraByPixels(dx, dy);
      lx = e.clientX; ly = e.clientY;
    } else if (dragging) {
      camState.targetTheta -= dx * 0.0045;
      camState.targetPhi = Math.max(0.12, Math.min(Math.PI - 0.12, camState.targetPhi - dy * 0.0045));
      lx = e.clientX; ly = e.clientY;
    }
  });

  window.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA'].includes((e.target && e.target.tagName) || '')) return;
    const step = Math.max(8, camState.targetCamR * 0.045);
    if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') panCameraByPixels(step, 0);
    if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') panCameraByPixels(-step, 0);
    if (e.key === 'w' || e.key === 'W' || e.key === 'ArrowUp') panCameraByPixels(0, step);
    if (e.key === 's' || e.key === 'S' || e.key === 'ArrowDown') panCameraByPixels(0, -step);
    if (e.key === 'q' || e.key === 'Q') camState.targetLookAt.y += step * 0.6;
    if (e.key === 'e' || e.key === 'E') camState.targetLookAt.y -= step * 0.6;
  });

  renderer.domElement.addEventListener('wheel', e => {
    e.preventDefault();
    mouseNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouseNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
    const zoomFactor = Math.pow(1.0018, e.deltaY);
    const oldCamR = Math.max(1e-3, camState.targetCamR);
    const bounds = getBoundsFn();
    camState.targetCamR = clampCamDistance(oldCamR * zoomFactor, bounds);

    camera.getWorldDirection(_camForward);
    focalPlane.normal.copy(_camForward).negate();
    focalPlane.constant = -camState.currentLookAt.dot(focalPlane.normal);
    raycaster.setFromCamera(mouseNDC, camera);
    if (raycaster.ray.intersectPlane(focalPlane, _zoomHit)) {
      camState.targetLookAt.lerp(_zoomHit, 1.0 - camState.targetCamR / oldCamR);
    }
  }, { passive: false });

  renderer.domElement.addEventListener('dblclick', e => {
    mouseNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouseNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouseNDC, camera);

    const views = getViewsFn();
    const bounds = getBoundsFn();
    const clickableObjects = [];
    for (const v of views.cells) {
      if (v.membraneMesh) { v.membraneMesh.userData.cellView = v; clickableObjects.push(v.membraneMesh); }
      if (v.nucleus) { v.nucleus.userData.cellView = v; clickableObjects.push(v.nucleus); }
      if (v.membrane) { v.membrane.userData.cellView = v; clickableObjects.push(v.membrane); }
    }
    const hits = raycaster.intersectObjects(clickableObjects, false);

    if (hits.length > 0) {
      const v = hits[0].object.userData.cellView;
      if (v && v.cell) {
        camState.targetLookAt.set(v.cell.x, v.cell.y, v.cell.z || 0);
        const neighborR = Math.max(90, Math.min(220, (bounds.radius || 180) * 0.55));
        camState.targetCamR = neighborR;
        camState.isCamTransitioning = true;
        if (logFn) logFn(`[街区对焦] 推进至元胞 #${v.cell.id} (${v.cell.type})，邻域组织保持原尺度同屏可见`, true);
      }
    } else {
      camState.targetLookAt.copy(bounds.center);
      camState.targetCamR = bounds.macroDist || 540;
      camState.targetPhi = Math.PI / 2;
      camState.targetTheta = 0;
      camState.isCamTransitioning = true;
      if (logFn) logFn(`[全景复位] 镜头回到生命体中心全景`, true);
    }
  });
}

export function updateCamera(camera, dt) {
  if (camState.autoOrbitEnabled) {
    camState.targetTheta += 0.0035;
  }

  const lerpFactor = camState.isCamTransitioning ? 0.08 : 0.14;
  camState.camTheta += (camState.targetTheta - camState.camTheta) * lerpFactor;
  camState.camPhi += (camState.targetPhi - camState.camPhi) * lerpFactor;
  camState.camR += (camState.targetCamR - camState.camR) * lerpFactor;
  camState.currentLookAt.lerp(camState.targetLookAt, lerpFactor);

  if (Math.abs(camState.targetCamR - camState.camR) < 1.0 &&
      Math.abs(camState.targetTheta - camState.camTheta) < 0.001 &&
      Math.abs(camState.targetPhi - camState.camPhi) < 0.001) {
    camState.isCamTransitioning = false;
  }

  let shakeX = 0, shakeY = 0, shakeZ = 0;
  if (camState.shakeImpulse > 0.01) {
    shakeX = (Math.random() - 0.5) * camState.shakeImpulse;
    shakeY = (Math.random() - 0.5) * camState.shakeImpulse;
    shakeZ = (Math.random() - 0.5) * camState.shakeImpulse;
    camState.shakeImpulse *= Math.exp(-dt * 4.5);
  }

  const r = camState.camR;
  camera.position.x = camState.currentLookAt.x + r * Math.sin(camState.camPhi) * Math.sin(camState.camTheta) + shakeX;
  camera.position.y = camState.currentLookAt.y + r * Math.cos(camState.camPhi) + shakeY;
  camera.position.z = camState.currentLookAt.z + r * Math.sin(camState.camPhi) * Math.cos(camState.camTheta) + shakeZ;
  camera.lookAt(camState.currentLookAt);
}
