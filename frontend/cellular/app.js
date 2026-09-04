/* ============================================================
 * app.js - SDSCC 全息细胞观测台主调度器与装配入口
 * ============================================================ */
import * as THREE from 'three';
import { T, FAMILY, FAMILY_COLOR, MUT_CANDIDATES } from './config.js';
import { org, compile, forward, stepPhysics, mitosis, rewire, apoptosis, seedOrganism } from './organism_model.js';
import { currentOrganismBounds, updateOrganismBounds } from './spatial_bounds.js';
import { scene, camera, renderer, cellPointLight, airParticleCloud, updateAirParticles } from './scene_setup.js';
import { initPostprocessing, setVisualBloomMode, renderScene, resizePostprocessing } from './postprocessing.js';
import { camState, updateCamera, setCameraDistance, setCameraTarget, focusOnCell, setCameraPreset, toggleAutoOrbit, initCameraController, cameraShake } from './camera_controller.js';
import { views, lodPointsMesh, rebuildViews, updateDetailLOD } from './lod_system.js';
import { initBioAudio, toggleBioAcoustics, playIonizationSpark, playChicxulubAtmosphericThunder } from './audio_system.js';
import { triggerGlobalLifeEvent, playLifeEpicStory, triggerManualDischargeBurst, togglePlasmaStorm, triggerChicxulubExtinction, triggerOrganSplice, triggerLyapunovEnforce } from './life_events.js';
import { serverOnline, wsConnected, clientWarpMultiplier, lastPrice, realPrice, totalActs, log, fetchRealPrice, marketTick, setWarp, setStress, pollIslands, pollBiosphere, bioViews, bioLayerVisible, radPlane, radRays, radUniforms, radVisible, connectWebSocket, syncBackendState, sendBackendCommand, toggleBioLayer, toggleRadLayer } from './network_sync.js';
import { openDocReader, closeDocReader, escapeHtml } from './document_reader.js';
import { toggleDialogueDeck, sendQuickPrompt, sendDialogueMsg } from './dialogue_system.js';
import { startAutoTour, showTourStep, nextTourStep, prevTourStep, endAutoTour, initTooltipEngine, TOUR_STAGES } from './tour_system.js';
import { currentSelectedOrgId, currentHighlightedBookId, currentRenderMode, currentLOD, ORGAN_DESCRIPTIONS, onOrganSelectionChange, onRowClick, toggleTreeNode, toggleDock, openLibraryDrawer, toggleHabitatMenu, selectOrganism, highlightBookSubcircuit, loadPreset, switchLOD, setRenderMode, pollLibrary } from './organism_library.js';
import { currentFluidPhase, setFluidPhase, generateFractalLightningPoints, spawnDielectricBreakdownArc, triggerExtinctionLightningBurst, triggerExtinctionVisualShock } from './plasma_effects.js';
import { initOrganSystem, updateOrganSystem, toggleOrganVisibility, focusOrgan } from './organ_view.js';

// 1. 初始化后处理通道与镜头控制
initPostprocessing(renderer, scene, camera);
initCameraController(renderer, camera, () => org, () => views, () => currentOrganismBounds, log);

// 2. 初始构建全景流形与微观视图
compile(org);
updateOrganismBounds(null, org);
rebuildViews(scene, org, currentOrganismBounds);

let paused = false;
let tickTimer = 0;
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
    const fpsEl = document.getElementById('st-fps');
    if (fpsEl) fpsEl.textContent = fps.toFixed(1);
    frameCount = 0;
    lastFpsTime = now;
  }

  const dt = Math.min(clock.getDelta(), 0.05);
  if (!paused) {
    tickTimer += dt;
    if (tickTimer >= 0.6) {
      tickTimer = 0;
      marketTick();
    }
    stepPhysics(org);
  }

  // 1. 相机动力学更新与阻尼插值
  updateCamera(dt, camera);

  // 2. 视锥裁剪准备
  _projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  _frustum.setFromProjectionMatrix(_projScreenMatrix);

  const closeLook = camState.camR < Math.max(160, (currentOrganismBounds.microDist || 220) * 0.85);
  const totalCellCount = org && org.cells ? org.cells.length : 0;
  const macroScale = (currentOrganismBounds && currentOrganismBounds.cellScale) || totalCellCount;
  const isLargeScale = (macroScale >= 1000) || (totalCellCount > 256);
  const isDiscrete = !isLargeScale;

  let showPointCloud = true;
  if (currentRenderMode === "puremesh") {
    showPointCloud = false;
  } else if (currentRenderMode === "lod") {
    showPointCloud = isLargeScale || !closeLook;
  } else { // "symbiosis"
    showPointCloud = true;
  }

  if (lodPointsMesh && lodPointsMesh.material) {
    lodPointsMesh.visible = showPointCloud;
    if (isLargeScale) {
      lodPointsMesh.material.opacity = closeLook ? 0.35 : 0.85;
      lodPointsMesh.material.size = closeLook ? 3.0 : 4.2;
    } else {
      lodPointsMesh.material.opacity = closeLook ? 0.18 : (currentRenderMode === "lod" ? 0.70 : 0.45);
      lodPointsMesh.material.size = (currentRenderMode === "lod") ? 3.6 : 2.6;
    }
  }

  // 3. 动态屏幕像素视锥实化 LOD
  updateDetailLOD(_frustum, scene, camera, org, currentOrganismBounds, currentRenderMode);

  // 3.2 动态更新 3D 生物器官外包膜与共生微柱全息透视
  updateOrganSystem(scene, org, now * 0.001, !closeLook);

  let visibleMicroCount = 0;
  const isDenseCells = views.cells.length > 50;
  const showMicroOrganelles = !isDenseCells || closeLook;

  for (const v of views.cells) {
    visibleMicroCount++;
    v.group.visible = true;

    // 宏观远景下隐藏繁琐的大圆环/色带/线粒体/穿膜代谢流/内膜孔道，避免几何与CPU过载；特写近视距时才展示微观超精细内部结构
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

    v.update(now * 0.001, clientWarpMultiplier);

    // 标签显示：避免几十个巨大文字漂浮遮挡画面。只在近距特写或少量细胞下针对前排受体显示
    if (v.label) {
      const showLabel = closeLook ? (visibleMicroCount <= 16) : (!isDenseCells && visibleMicroCount <= 8);
      v.label.visible = showLabel;
      if (v.label.material) v.label.material.opacity = showLabel ? 0.90 : 0;
    }
  }

  const isDenseSyn = views.syns.length > 60;
  for (const v of views.syns) {
    v.group.visible = true;
    v.update(now * 0.001, clientWarpMultiplier);
    // 大规模与密集突触网络下线条与光子轻盈微透（隐隐若现），避免强光刺眼糊屏
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

  const ptCount = (lodPointsMesh && lodPointsMesh.geometry && lodPointsMesh.geometry.attributes.position) ? lodPointsMesh.geometry.attributes.position.count : totalCellCount;
  const realCellsEl = document.getElementById("st-real-cells");
  if (realCellsEl) {
    if (isDiscrete) {
      realCellsEl.textContent = `${visibleMicroCount}/${totalCellCount} 实体全量 (100% 显微实化)`;
    } else if (visibleMicroCount > 0) {
      realCellsEl.textContent = `${visibleMicroCount} 实体视锥局部实化 / ${ptCount.toLocaleString()} 点云流形`;
    } else {
      realCellsEl.textContent = `${ptCount.toLocaleString()} 点云流形 (宏观亚像素，真实未放大)`;
    }
  }
  const elScale = document.getElementById("st-pipe");
  if (elScale) elScale.textContent = isDiscrete ? `实体 ${visibleMicroCount}/${totalCellCount} (100% 全量实化)` : `实体 ${visibleMicroCount} / 点云 ${ptCount.toLocaleString()} · 像素LOD实化`;

  const vitalScaleEl = document.getElementById("vital-scale");
  const vitalScaleSubEl = document.getElementById("vital-scale-sub");
  if (vitalScaleEl) {
    vitalScaleEl.textContent = macroScale.toLocaleString() + ' 细胞';
  }
  if (vitalScaleSubEl) {
    if (isDiscrete) {
      vitalScaleSubEl.textContent = `${visibleMicroCount}/${totalCellCount} 实体全量晶化 · 30,000 星云`;
    } else if (visibleMicroCount > 0) {
      vitalScaleSubEl.textContent = `${visibleMicroCount} 实体视锥实化 / ${ptCount.toLocaleString()} 点云`;
    } else {
      vitalScaleSubEl.textContent = `全视界 ${ptCount.toLocaleString()} 动力学流形点云 (LOD)`;
    }
  }

  const elCamR = document.getElementById("st-cam-r");
  if (elCamR) elCamR.textContent = `${Math.round(camState.camR)} 单位`;

  const elFocal = document.getElementById("st-focal");
  if (elFocal) {
    elFocal.textContent = `${visibleMicroCount}/${totalCellCount} 细胞可见 · 世界尺度未改`;
  }

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

  // 流体微粒模拟
  if (airParticleCloud && airParticleCloud.geometry && airParticleCloud.visible) {
    updateAirParticles(dt);
  }

  // 离子电弧放电
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

  // 最终渲染
  renderScene(dt, renderer, scene, camera);
}

// 绑定所有全局 HTML 事件处理器
window.setWarp = setWarp;
window.setVisualBloomMode = (mode) => setVisualBloomMode(mode, renderer, log);
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
window.setRenderMode = setRenderMode;
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
_bind('b-mito', () => { mitosis(org); rebuildViews(scene, org, currentOrganismBounds); });
_bind('b-rewire', () => { rewire(org); rebuildViews(scene, org, currentOrganismBounds); });
_bind('b-apop', () => { apoptosis(org); rebuildViews(scene, org, currentOrganismBounds); });
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
  e.target.textContent = paused ? '[RESUME] 继续' : '[PAUSE] 暂停';
});
_bind('b-reset', () => {
  const seeded = seedOrganism();
  org.cells = seeded.cells;
  org.syns = seeded.syns;
  compile(org);
  rebuildViews(scene, org, currentOrganismBounds);
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

setInterval(() => { if (!paused && !serverOnline) { mitosis(org); rebuildViews(scene, org, currentOrganismBounds); } }, 14000);
setInterval(() => { if (!paused && !serverOnline && org.cells.length > 16) { apoptosis(org); rebuildViews(scene, org, currentOrganismBounds); } }, 42000);

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

log('形态发生细胞全息观测台已启动 — 拖拽旋转 · 滚轮缩放', true);

// 启动动画渲染循环
animate();
