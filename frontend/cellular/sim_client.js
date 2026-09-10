/* ============================================================
 * sim_client.js - 主线程仿真代理（类 Qt 跨线程 QObject 代理）
 *
 * - Worker 持有权威 Model（离线 owner 模式）
 * - 主线程 org 为 View 镜像：帧同步 out/glow，拓扑变更时整表替换
 * - Worker 不可用时降级到主线程同路径（仍走 frameBus）
 * ============================================================ */
import { org, compile, forward, stepPhysics, mitosis, rewire, apoptosis, seedOrganism } from './organism_model.js';
import { frameBus } from './frame_bus.js';
import { hudSet } from './hud_bus.js';
import { updateOrganismBounds, currentOrganismBounds } from './spatial_bounds.js';
import { rebuildViews } from './lod_system.js';
import { scene } from './scene_setup.js';

export let simWorkerActive = false;
export let simFallbackMain = false;
export let simLastActions = { buy: 0, sell: 0, reset: 0, immune: false };
export let simTotalActs = 0;

/** @type {Worker|null} */
let worker = null;
let onLog = null;
let marketAccMain = 0;

function applyMirrorFromSnap(snap) {
  if (!snap || !org.cells) return;
  if (!org.cellMap || org.cellMap.size !== org.cells.length) {
    org.cellMap = new Map(org.cells.map((c) => [c.id, c]));
  }
  for (let i = 0; i < snap.cells.length; i++) {
    const sc = snap.cells[i];
    const c = org.cellMap.get(sc.id);
    if (!c) continue;
    c.out = sc.out;
    c.glow = sc.glow;
    c.state = sc.state;
    c.acts = sc.acts;
    c.x = sc.x;
    c.y = sc.y;
    c.z = sc.z;
  }
  if (snap.syns && org.syns && snap.syns.length === org.syns.length) {
    for (let i = 0; i < snap.syns.length; i++) {
      org.syns[i].photon = snap.syns[i].photon;
      org.syns[i].active = snap.syns[i].active;
      org.syns[i].w = snap.syns[i].w;
    }
  }
  org.generation = snap.generation;
  org.phySteps = snap.phySteps;
}

function applyTopology(data) {
  if (!data || !data.cells) return;
  org.generation = data.generation || 0;
  org.phySteps = data.phySteps || 0;
  org.lastOrganismId = data.lastOrganismId || org.lastOrganismId;
  org.lastFingerprint = data.lastFingerprint || null;
  org.cells = data.cells;
  org.syns = data.syns || [];
  compile(org);
  updateOrganismBounds(null, org);
  rebuildViews(scene, org, currentOrganismBounds);
  frameBus.publish(org);
}

function handleWorkerMessage(ev) {
  const msg = ev.data || {};
  if (msg.type === 'ready') {
    simWorkerActive = true;
    if (msg.org) applyTopology(msg.org);
    if (onLog) onLog('[SIM] Worker Model 线程就绪（类 Qt QThread）', true);
    return;
  }
  if (msg.type === 'frame') {
    const snap = frameBus.accept(msg.snap);
    if (snap) applyMirrorFromSnap(snap);
    if (msg.actions) simLastActions = msg.actions;
    if (msg.totalActs !== undefined) simTotalActs = msg.totalActs;
    if (msg.lastPrice !== undefined) {
      hudSet('px', Number(msg.lastPrice).toFixed(1));
    }
    hudSet('v-buy', (simLastActions.buy || 0).toFixed(2));
    hudSet('v-sell', (simLastActions.sell || 0).toFixed(2));
    hudSet('v-immune', simLastActions.immune ? '熔断!' : '—');
    hudSet('st-act', String(simTotalActs));
    hudSet('st-gen', String(msg.generation != null ? msg.generation : org.generation));
    hudSet('st-phys', String(msg.phySteps != null ? msg.phySteps : org.phySteps));
    return;
  }
  if (msg.type === 'topology') {
    applyTopology(msg.org);
    if (onLog && msg.reason) onLog(`[SIM] 拓扑变更: ${msg.reason}`, true);
  }
}

/**
 * 启动仿真线程。失败则主线程降级。
 * @param {{ log?: Function }} [opts]
 */
export function startSimEngine(opts = {}) {
  onLog = opts.log || null;
  if (typeof Worker === 'undefined') {
    simFallbackMain = true;
    if (onLog) onLog('[SIM] 无 Worker，主线程降级推演', true);
    return false;
  }
  try {
    worker = new Worker(new URL('./sim_worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (err) => {
      simWorkerActive = false;
      simFallbackMain = true;
      if (onLog) onLog(`[SIM] Worker 失败，降级主线程: ${err.message || err}`, true);
      stopSimEngine();
    };
    worker.postMessage({ type: 'start', mode: 'owner' });
    return true;
  } catch (e) {
    simFallbackMain = true;
    worker = null;
    if (onLog) onLog(`[SIM] Worker 创建失败，降级主线程: ${e.message || e}`, true);
    return false;
  }
}

export function stopSimEngine() {
  if (worker) {
    try { worker.postMessage({ type: 'stop' }); } catch (_) { /* ignore */ }
    try { worker.terminate(); } catch (_) { /* ignore */ }
  }
  worker = null;
  simWorkerActive = false;
}

export function setSimPaused(paused) {
  if (worker && simWorkerActive) {
    worker.postMessage({ type: 'setPaused', paused: !!paused });
  }
}

/** 后端在线 → mirror；离线 → owner */
export function setSimMode(mode) {
  if (worker && simWorkerActive) {
    worker.postMessage({ type: 'setMode', mode: mode === 'mirror' ? 'mirror' : 'owner' });
  }
}

export function pushOrgToWorker() {
  if (!worker || !simWorkerActive) return;
  worker.postMessage({
    type: 'replace',
    org: {
      generation: org.generation,
      phySteps: org.phySteps,
      lastOrganismId: org.lastOrganismId,
      lastFingerprint: org.lastFingerprint,
      cells: org.cells,
      syns: org.syns
    }
  });
}

export function setSimMarketHints({ realPrice, lastPrice } = {}) {
  if (worker && simWorkerActive) {
    worker.postMessage({ type: 'setMarket', realPrice, lastPrice });
  }
}

export function simCommand(command) {
  if (worker && simWorkerActive && !simFallbackMain) {
    worker.postMessage({ type: 'command', command });
    return true;
  }
  // 主线程降级路径
  let ok = false;
  if (command === 'mitosis') ok = !!mitosis(org);
  else if (command === 'rewire') ok = !!rewire(org);
  else if (command === 'apoptosis') { apoptosis(org); ok = true; }
  else if (command === 'reset') {
    const seeded = seedOrganism();
    org.cells = seeded.cells;
    org.syns = seeded.syns;
    org.generation = seeded.generation || 0;
    org.phySteps = 0;
    compile(org);
    ok = true;
  }
  if (ok) {
    updateOrganismBounds(null, org);
    rebuildViews(scene, org, currentOrganismBounds);
    frameBus.publish(org);
  }
  return ok;
}

/**
 * 主线程降级时的 Model 步进（Worker 活跃时 no-op）。
 * @param {number} dt
 * @param {{ paused?: boolean, doMarket?: boolean, lastPrice?: number, realPrice?: number|null }} opts
 */
export function simMainThreadTick(dt, opts = {}) {
  if (simWorkerActive && !simFallbackMain) return null;
  if (opts.paused) return frameBus.publish(org);
  stepPhysics(org, dt);
  if (opts.doMarket) {
    let price = opts.lastPrice != null ? opts.lastPrice : 3620;
    const rp = opts.realPrice;
    if (rp != null) price += (rp - price) * 0.15 + (Math.random() - 0.5) * 1.2;
    else price += (Math.random() - 0.5) * 2.4;
    const actions = forward(org, [price, 4000 + Math.random() * 3000, 0.8 + Math.random() * 2.5, Math.random() * 2 - 1]);
    simLastActions = actions;
    simTotalActs += org.cells.filter((c) => Math.abs(c.out) > 1e-6).length;
    return { price, actions, snap: frameBus.publish(org) };
  }
  return { snap: frameBus.publish(org) };
}

export function simMainThreadMarketAcc(dt) {
  if (simWorkerActive && !simFallbackMain) return false;
  marketAccMain += dt;
  if (marketAccMain >= 0.6) {
    marketAccMain = 0;
    return true;
  }
  return false;
}
