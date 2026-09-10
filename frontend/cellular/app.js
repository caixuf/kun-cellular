/* ============================================================
 * app.js - SDSCC 全息细胞观测台主调度器与装配入口
 * ============================================================ */
import * as THREE from 'three';
import { FAMILY, FAMILY_COLOR } from './config.js';
import { org, compile } from './organism_model.js';
import { currentOrganismBounds, updateOrganismBounds } from './spatial_bounds.js';
import { scene, camera, renderer, cellPointLight, airParticleCloud, updateAirParticles } from './scene_setup.js';
import { initPostprocessing, setVisualBloomMode, renderScene, resizePostprocessing } from './postprocessing.js';
import { camState, updateCamera, setCameraDistance, setCameraTarget, focusOnCell, setCameraPreset, toggleAutoOrbit, initCameraController, cameraShake } from './camera_controller.js';
import { views, lodPointsMesh, rebuildViews, updateDetailLOD, setActivePresentationMode } from './lod_system.js';
import { frameBus } from './frame_bus.js';
import { hudSet, hudFlush } from './hud_bus.js';
import {
  startSimEngine, setSimPaused, setSimMode, pushOrgToWorker, setSimMarketHints,
  simCommand, simWorkerActive, simFallbackMain, simMainThreadTick, simMainThreadMarketAcc
} from './sim_client.js';
import { initBioAudio, toggleBioAcoustics, playIonizationSpark, playChicxulubAtmosphericThunder } from './audio_system.js';
import { triggerGlobalLifeEvent, playLifeEpicStory, triggerManualDischargeBurst, togglePlasmaStorm, triggerChicxulubExtinction, triggerOrganSplice, triggerLyapunovEnforce } from './life_events.js';
import { serverOnline, wsConnected, clientWarpMultiplier, lastPrice, realPrice, log, fetchRealPrice, setWarp, setStress, pollIslands, pollBiosphere, bioViews, bioLayerVisible, radPlane, radRays, radUniforms, radVisible, connectWebSocket, syncBackendState, sendBackendCommand, toggleBioLayer, toggleRadLayer } from './network_sync.js';
import { openDocReader, closeDocReader, escapeHtml } from './document_reader.js';
import { toggleDialogueDeck, sendQuickPrompt, sendDialogueMsg } from './dialogue_system.js';
import { startAutoTour, showTourStep, nextTourStep, prevTourStep, endAutoTour, initTooltipEngine, TOUR_STAGES } from './tour_system.js';
import { currentSelectedOrgId, currentHighlightedBookId, currentRenderMode, currentLOD, ORGAN_DESCRIPTIONS, onOrganSelectionChange, onRowClick, toggleTreeNode, toggleDock, openLibraryDrawer, toggleHabitatMenu, selectOrganism, highlightBookSubcircuit, loadPreset, switchLOD, setRenderMode, pollLibrary } from './organism_library.js';
import { currentFluidPhase, setFluidPhase, generateFractalLightningPoints, spawnDielectricBreakdownArc, triggerExtinctionLightningBurst, triggerExtinctionVisualShock } from './plasma_effects.js';
import { initOrganSystem, updateOrganSystem, toggleOrganVisibility, focusOrgan } from './organ_view.js';
import { patchClampHUD } from './patch_clamp_hud.js';
import { tractography } from './tractography.js';
import { updateManifoldSystem } from './manifold_system.js';

// 1. 初始化后处理通道与镜头控制
initPostprocessing(renderer, scene, camera);
initCameraController(renderer, camera, () => org, () => views, () => currentOrganismBounds, log);

// 2. 初始构建全景流形与微观视图（默认：实体+星云 + 舒适科研，仪器可选）
compile(org);
updateOrganismBounds(null, org);
setActivePresentationMode('symbiosis');
rebuildViews(scene, org, currentOrganismBounds);
frameBus.publish(org);
setVisualBloomMode('scientific', renderer, null);

// 3. 启动 Worker Model 线程（失败则主线程降级）
startSimEngine({ log });

let paused = false;
let lastServerOnline = null;
const clock = new THREE.Clock();
let frameCount = 0;
let lastFpsTime = performance.now();

const _projScreenMatrix = new THREE.Matrix4();
const _frustum = new THREE.Frustum();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  frameCount++;
  if (now - lastFpsTime >= 500) {
    const fps = (frameCount * 1000) / (now - lastFpsTime);
    hudSet('st-fps', fps.toFixed(1));
    frameCount = 0;
    lastFpsTime = now;
  }

  const dt = Math.min(clock.getDelta(), 0.05);
  const instrument = currentRenderMode === 'instrument';

  // 后端在线时 Worker 切 mirror，避免与权威状态双写
  if (lastServerOnline !== serverOnline) {
    lastServerOnline = serverOnline;
    setSimMode(serverOnline ? 'mirror' : 'owner');
    if (serverOnline) pushOrgToWorker();
  }

  // ── MODEL：Worker 异步产出帧；仅降级时主线程步进 ──
  if (simWorkerActive && !simFallbackMain) {
    setSimMarketHints({ realPrice, lastPrice });
  } else if (!paused && !serverOnline) {
    const doMarket = simMainThreadMarketAcc(dt);
    simMainThreadTick(dt, { paused: false, doMarket, lastPrice, realPrice });
  } else if (!simWorkerActive || simFallbackMain) {
    frameBus.publish(org);
  }

  const snap = frameBus.latest() || frameBus.publish(org);

  // ── VIEW（只消费快照 + 相机）──
  updateCamera(dt, camera);

  _projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  _frustum.setFromProjectionMatrix(_projScreenMatrix);

  const closeLook = camState.camR < Math.max(160, (currentOrganismBounds.microDist || 220) * 0.85);
  const totalCellCount = snap.cellCount;
  const macroScale = (currentOrganismBounds && currentOrganismBounds.cellScale) || totalCellCount;
  const isLargeScale = (macroScale >= 100000) || (totalCellCount > 3000);
  const isDiscrete = !isLargeScale;

  let showPointCloud = true;
  if (instrument || currentRenderMode === "puremesh") {
    showPointCloud = false;
  } else if (currentRenderMode === "lod") {
    showPointCloud = isLargeScale || (views.cells.length === 0) || !closeLook;
  } else { // "symbiosis"
    showPointCloud = true;
  }

  if (lodPointsMesh && lodPointsMesh.material) {
    const hasSolid = views.cells && views.cells.length > 0;
    if (isLargeScale || instrument) {
      lodPointsMesh.visible = instrument ? false : (isLargeScale ? false : showPointCloud);
    } else {
      lodPointsMesh.visible = showPointCloud;
      lodPointsMesh.material.opacity = hasSolid ? (closeLook ? 0.25 : (currentRenderMode === "lod" ? 0.45 : 0.65)) : 0.90;
      lodPointsMesh.material.size = hasSolid ? ((currentRenderMode === "lod") ? 1.6 : 1.4) : 2.0;
    }
  }

  const manifoldOpacity = !isLargeScale ? 0.0 : (views.cells && views.cells.length > 0 ? (closeLook ? 0.35 : 0.65) : 0.88);
  updateManifoldSystem(now * 0.001, instrument ? 0.0 : manifoldOpacity, !instrument && showPointCloud && isLargeScale);

  updateDetailLOD(_frustum, scene, camera, org, currentOrganismBounds, currentRenderMode);
  updateOrganSystem(scene, org, now * 0.001, !instrument && !closeLook);

  let visibleMicroCount = 0;
  const isDenseCells = views.cells.length > 50;
  const showMicroOrganelles = !instrument && (!isDenseCells || closeLook);

  for (const v of views.cells) {
    visibleMicroCount++;
    v.group.visible = true;
    const cellSnap = snap.cellById.get(v.cell && v.cell.id);
    if (cellSnap) v.applySnapshot(cellSnap);

    if (!instrument) {
      if (v.delayRing) v.delayRing.visible = showMicroOrganelles;
      if (v.attrRibbon) v.attrRibbon.visible = showMicroOrganelles;
      if (v.metabolicPoints) v.metabolicPoints.visible = closeLook;
      if (v.innerMembraneMesh) v.innerMembraneMesh.visible = showMicroOrganelles;
      if (v.poresMesh) v.poresMesh.visible = showMicroOrganelles;
      if (v.cytoMesh) v.cytoMesh.visible = showMicroOrganelles;
      if (v.organelles) {
        for (const o of v.organelles) {
          if (o.mesh) o.mesh.visible = showMicroOrganelles;
        }
      }
    }

    v.update(now * 0.001, clientWarpMultiplier);

    if (v.label) {
      const showLabel = instrument
        ? (closeLook ? visibleMicroCount <= 24 : visibleMicroCount <= 12)
        : (closeLook ? (visibleMicroCount <= 16) : (!isDenseCells && visibleMicroCount <= 8));
      v.label.visible = showLabel;
      if (v.label.material) v.label.material.opacity = showLabel ? 0.90 : 0;
    }
  }

  const isDenseSyn = views.syns.length > 60;
  for (const v of views.syns) {
    v.group.visible = true;
    v.update(now * 0.001, clientWarpMultiplier);
    if (instrument) continue;
    if (isDenseSyn && !closeLook) {
      v.lineMat.opacity = 0.12;
      v.photon1.material.opacity = 0.32;
      v.photon2.material.opacity = 0.32;
      if (v.bouton && v.bouton.material) v.bouton.material.opacity = 0.22;
    } else if (isDenseSyn && closeLook) {
      v.lineMat.opacity = 0.35;
      v.photon1.material.opacity = 0.60;
      v.photon2.material.opacity = 0.60;
      if (v.bouton && v.bouton.material) v.bouton.material.opacity = 0.45;
    } else {
      v.lineMat.opacity = closeLook ? 0.65 : 0.32;
      v.photon1.material.opacity = closeLook ? 0.85 : 0.50;
      v.photon2.material.opacity = closeLook ? 0.85 : 0.50;
      if (v.bouton && v.bouton.material) v.bouton.material.opacity = closeLook ? 0.75 : 0.40;
    }
  }

  if (!instrument) {
    patchClampHUD.update(now * 0.001);
    tractography.update(now * 0.001, clientWarpMultiplier);
  }

  const ptCount = (lodPointsMesh && lodPointsMesh.geometry && lodPointsMesh.geometry.attributes.position) ? lodPointsMesh.geometry.attributes.position.count : totalCellCount;

  // ── UI（集中刷 DOM，类 Qt GUI 线程）──
  if (isDiscrete) {
    hudSet('st-real-cells', `${visibleMicroCount}/${totalCellCount} 实体全量 (100% 显微实化)`);
    hudSet('st-pipe', `实体 ${visibleMicroCount}/${totalCellCount} (100% 全量实化)`);
    hudSet('vital-scale-sub', instrument
      ? `${snap.cellCount} 细胞 · ${snap.synCount} 突触 · rev ${snap.revision}`
      : `${visibleMicroCount}/${totalCellCount} 实体全量晶化 · 30,000 星云`);
  } else if (visibleMicroCount > 0) {
    hudSet('st-real-cells', `${visibleMicroCount} 实体视锥局部实化 / ${ptCount.toLocaleString()} 点云流形`);
    hudSet('st-pipe', `实体 ${visibleMicroCount} / 点云 ${ptCount.toLocaleString()} · 像素LOD实化`);
    hudSet('vital-scale-sub', `${visibleMicroCount} 实体视锥实化 / ${ptCount.toLocaleString()} 点云`);
  } else {
    hudSet('st-real-cells', `${ptCount.toLocaleString()} 点云流形 (宏观亚像素，真实未放大)`);
    hudSet('st-pipe', `实体 ${visibleMicroCount} / 点云 ${ptCount.toLocaleString()} · 像素LOD实化`);
    hudSet('vital-scale-sub', `全视界 ${ptCount.toLocaleString()} 动力学流形点云 (LOD)`);
  }
  hudSet('vital-scale', macroScale.toLocaleString() + ' 细胞');
  hudSet('st-cam-r', `${Math.round(camState.camR)} 单位`);
  hudSet('st-focal', `${visibleMicroCount}/${totalCellCount} 细胞可见 · 世界尺度未改`);
  hudSet('st-cells', String(snap.cellCount));
  hudSet('st-syn', String(snap.synCount));

  if (bioLayerVisible) {
    const t = clock.elapsedTime;
    for (const view of bioViews.values()) {
      const agent = view.lastAgent;
      if (agent) view.update(agent, t);
    }
  }
  if (radVisible) {
    radUniforms.uTime.value = clock.elapsedTime;
    for (const [id, v] of radRays) {
      if (!v.step(dt)) { v.dispose(); radRays.delete(id); }
    }
  }

  if (!instrument && airParticleCloud && airParticleCloud.geometry && airParticleCloud.visible) {
    updateAirParticles(dt);
  } else if (instrument && airParticleCloud) {
    airParticleCloud.visible = false;
  }

  if (!instrument) {
    const dischargeProb = window.plasmaStormActive ? 0.22 : 0.003;
    if (views && views.cells && views.cells.length > 2 && Math.random() < dischargeProb) {
      const c1 = views.cells[Math.floor(Math.random() * views.cells.length)];
      const c2 = views.cells[Math.floor(Math.random() * views.cells.length)];
      if (c1 !== c2) {
        const v1 = new THREE.Vector3(c1.cell.x, c1.cell.y, c1.cell.z);
        const v2 = new THREE.Vector3(c2.cell.x, c2.cell.y, c2.cell.z);
        const dist = v1.distanceTo(v2);
        if (dist > 20 && dist < 260) {
          const colors = [0x38bdf8, 0xa855f7, 0x00f0ff, 0xfbbf24];
          const col = colors[Math.floor(Math.random() * colors.length)];
          spawnDielectricBreakdownArc(v1, v2, col, window.plasmaStormActive ? 0.9 : 0.45);
          playIonizationSpark(window.plasmaStormActive ? 0.35 : 0.12);
        }
      }
    }
  }

  hudFlush();
  renderScene(dt, renderer, scene, camera);
}

// 绑定所有全局 HTML 事件处理器
window.setWarp = setWarp;
window.setVisualBloomMode = (mode) => {
  // 「舒适科研 / 柔和微光」期望旧渲染观感：若仍卡在仪器模式，一并切回实体+星云
  if ((mode === 'scientific' || mode === 'cinematic') && currentRenderMode === 'instrument') {
    setRenderMode('symbiosis');
    if (airParticleCloud) airParticleCloud.visible = true;
  }
  setVisualBloomMode(mode, renderer, log);
};
window.setStress = setStress;
window.playLifeEpicStory = () => playLifeEpicStory(views, currentOrganismBounds, log, (w, s) => sendBackendCommand('extinction', { wipeout_ratio: w, shock_scale: s }), () => sendBackendCommand('splice'));
window.triggerGlobalLifeEvent = (type) => triggerGlobalLifeEvent(type, views, currentOrganismBounds, log, (w, s) => sendBackendCommand('extinction', { wipeout_ratio: w, shock_scale: s }), () => sendBackendCommand('splice'));
window.triggerLifeEvent = window.triggerGlobalLifeEvent;
window.openLibraryDrawer = openLibraryDrawer;
window.toggleHabitatMenu = toggleHabitatMenu;
window.toggleBioAcoustics = () => toggleBioAcoustics(log);
window.toggleDock = toggleDock;
window.triggerLyapunovEnforce = async () => { triggerLyapunovEnforce(log); await sendBackendCommand('lyapunov_enforce', { max_gain: 0.95 }); };
window.setFluidPhase = setFluidPhase;
window.triggerManualDischargeBurst = () => { log("[DISCHARGE] 局部强电场击穿空气介质！高压离子电弧正在神经微柱间立体放电", true); triggerManualDischargeBurst(views, currentOrganismBounds, log); };
window.togglePlasmaStorm = () => togglePlasmaStorm(views, currentOrganismBounds, log);
window.triggerChicxulubExtinction = async () => { triggerExtinctionVisualShock(views); triggerChicxulubExtinction(views, currentOrganismBounds, log, (w, s) => sendBackendCommand('extinction', { wipeout_ratio: w, shock_scale: s })); };
window.triggerOrganSplice = async () => { const sel = document.getElementById("sel-frozen-organ"); const organName = sel ? sel.value : "schmitt_damping_column"; log(`[VAULT] 正在从冷冻库借用剪裁器官【${organName}】并接入中枢网络...`, true); triggerOrganSplice(views, currentOrganismBounds, log, null, () => sendBackendCommand('splice', { name: organName })); };
window.setCameraPreset = (mode) => setCameraPreset(mode, currentOrganismBounds);
window.toggleAutoOrbit = toggleAutoOrbit;
window.setRenderMode = (mode) => {
  setRenderMode(mode);
  if (mode === 'instrument') {
    setVisualBloomMode('off', renderer, null);
    if (airParticleCloud) airParticleCloud.visible = false;
  } else {
    if (airParticleCloud) airParticleCloud.visible = true;
    setVisualBloomMode('scientific', renderer, null);
  }
};
window.switchLOD = switchLOD;
window.loadPreset = loadPreset;
window.startAutoTour = () => startAutoTour(views, currentOrganismBounds);
window.showTourStep = (step) => showTourStep(step, views, currentOrganismBounds);
window.nextTourStep = () => nextTourStep(views, currentOrganismBounds);
window.prevTourStep = () => prevTourStep(views, currentOrganismBounds);
window.endAutoTour = () => endAutoTour(views, currentOrganismBounds);
window.toggleDialogueDeck = toggleDialogueDeck;
window.sendQuickPrompt = (text) => sendQuickPrompt(text, views, FAMILY, FAMILY_COLOR, log);
window.sendDialogueMsg = () => sendDialogueMsg(views, FAMILY, FAMILY_COLOR, log);
window.openDocReader = openDocReader;
window.closeDocReader = closeDocReader;
window.selectOrganism = selectOrganism;
window.highlightBookSubcircuit = highlightBookSubcircuit;
window.onRowClick = onRowClick;
window.toggleTreeNode = toggleTreeNode;
window.onOrganSelectionChange = onOrganSelectionChange;
window.cameraShake = cameraShake;
window.toggleOrganVisibility = toggleOrganVisibility;
window.focusOrgan = focusOrgan;
window.focusOnCell = (id, dist = 25) => focusOnCell(id, org, dist);
window.camState = camState;
window.log = log;

// 绑定底座微观操作按钮
const _bind = (id, fn) => { const el = document.getElementById(id); if (el) el.onclick = fn; };
_bind('b-mito', () => { simCommand('mitosis'); });
_bind('b-rewire', () => { simCommand('rewire'); });
_bind('b-apop', () => { simCommand('apoptosis'); });
_bind('b-bio', e => {
  const vis = toggleBioLayer();
  e.target.textContent = vis ? ' 生态圈' : ' 生态圈(关)';
});
_bind('b-rad', e => {
  const vis = toggleRadLayer();
  e.target.textContent = vis ? '[RAD] 辐射场' : '[RAD] 辐射场(关)';
});
_bind('b-pause', e => {
  paused = !paused;
  setSimPaused(paused);
  e.target.textContent = paused ? '[RESUME] 继续' : '[PAUSE] 暂停';
});
_bind('b-reset', () => {
  simCommand('reset');
  log('[RESET] 重置为种子形态生物 Genesis-0', true);
});

// 定时任务调度
fetchRealPrice();
setInterval(fetchRealPrice, 8000);
pollIslands();
setInterval(pollIslands, 1000);
pollBiosphere();
setInterval(pollBiosphere, 2000);
pollLibrary();
setInterval(pollLibrary, 5000);
setInterval(syncBackendState, 100);
connectWebSocket();

setInterval(() => {
  if (!paused && !serverOnline) simCommand('mitosis');
}, 14000);
setInterval(() => {
  if (!paused && !serverOnline && org.cells.length > 16) simCommand('apoptosis');
}, 42000);

// 事件监听器
window.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDocReader(); });
window.addEventListener('click', (e) => {
  const container = document.querySelector('.habitat-dropdown-container');
  const m = document.getElementById('habitat-menu');
  if (m && container && !container.contains(e.target)) {
    m.style.display = 'none';
  }
});
window.addEventListener("pointerdown", () => { initBioAudio(); }, { once: true });
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  resizePostprocessing(window.innerWidth, window.innerHeight);
});

// 初始化新手 Smart Tooltips 引擎
initTooltipEngine();

log('细胞观测台已启动 — 默认实体+星云 · 仪器模式可选 · 拖拽旋转 · 滚轮缩放', true);
setRenderMode('symbiosis');

// 启动动画渲染循环（GUI 线程）
animate();
