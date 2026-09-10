/* ============================================================
 * sim_worker.js - 仿真 Model 线程（类 Qt QThread / Worker 对象）
 *
 * 主线程 = GUI：Three.js + DOM
 * 本线程 = Model：forward / stepPhysics / 拓扑变异
 * 通信：postMessage 帧快照（queued connection）
 * ============================================================ */
import {
  org,
  compile,
  forward,
  stepPhysics,
  mitosis,
  rewire,
  apoptosis,
  seedOrganism
} from './organism_model.js';
import { captureFrameSnapshot } from './frame_bus.js';

/** @type {'owner'|'mirror'} owner=离线自主推演；mirror=后端权威，本线程只持副本不推演 */
let mode = 'owner';
let paused = false;
let timerId = null;
let revision = 0;
let marketAcc = 0;
let lastPrice = 3620;
let realPrice = null;
let totalActs = 0;
let lastActions = { buy: 0, sell: 0, reset: 0, immune: false };
const TICK_MS = 16;

function serializeOrg(o) {
  return {
    generation: o.generation || 0,
    phySteps: o.phySteps || 0,
    lastOrganismId: o.lastOrganismId || null,
    lastFingerprint: o.lastFingerprint || null,
    cells: (o.cells || []).map((c) => ({
      id: c.id,
      type: c.type,
      param1: c.param1,
      param2: c.param2,
      state: c.state || 0,
      prev: c.prev || 0,
      latch: !!c.latch,
      out: c.out || 0,
      acts: c.acts || 0,
      x: c.x || 0,
      y: c.y || 0,
      z: c.z || 0,
      _rawX: c._rawX,
      _rawY: c._rawY,
      _rawZ: c._rawZ,
      glow: c.glow || 0,
      aux: c.aux,
      buf: c.buf,
      didx: c.didx
    })),
    syns: (o.syns || []).map((s) => ({
      from: s.from,
      to: s.to,
      port: s.port || 0,
      w: s.w !== undefined ? s.w : 1.0,
      active: s.active !== false,
      photon: s.photon !== undefined ? s.photon : -1,
      rest: s.rest
    }))
  };
}

function applySerialized(data) {
  if (!data || !data.cells) return;
  org.generation = data.generation || 0;
  org.phySteps = data.phySteps || 0;
  org.lastOrganismId = data.lastOrganismId || org.lastOrganismId;
  org.lastFingerprint = data.lastFingerprint || null;
  org.cells = data.cells;
  org.syns = data.syns || [];
  compile(org);
}

function emitFrame() {
  revision += 1;
  const snap = captureFrameSnapshot(org, {
    revision,
    t: performance.now() * 0.001
  });
  // Map 可 structured clone；主线程 accept 会再规范化
  self.postMessage({
    type: 'frame',
    snap,
    actions: lastActions,
    totalActs,
    lastPrice,
    generation: org.generation,
    phySteps: org.phySteps
  });
}

function emitTopology(reason) {
  self.postMessage({
    type: 'topology',
    reason: reason || 'mutate',
    org: serializeOrg(org)
  });
  emitFrame();
}

function runMarketStep() {
  if (realPrice != null) {
    lastPrice += (realPrice - lastPrice) * 0.15 + (Math.random() - 0.5) * 1.2;
  } else {
    lastPrice += (Math.random() - 0.5) * 2.4;
  }
  const vol = 4000 + Math.random() * 3000;
  const spread = 0.8 + Math.random() * 2.5;
  const imb = Math.random() * 2 - 1;
  lastActions = forward(org, [lastPrice, vol, spread, imb]);
  totalActs += org.cells.filter((c) => Math.abs(c.out) > 1e-6).length;
  for (const s of org.syns) {
    if (!s.active) continue;
    const a = org.cellMap && org.cellMap.get(s.from);
    if (a && Math.abs(a.out) > 1e-6) s.photon = Math.random() * 0.15;
  }
}

function tick() {
  if (paused || mode !== 'owner') {
    // mirror / pause：不抢主线程权威帧，避免覆盖后端遥测
    return;
  }
  const dt = TICK_MS / 1000;
  stepPhysics(org, dt);
  marketAcc += dt;
  if (marketAcc >= 0.6) {
    marketAcc = 0;
    runMarketStep();
  }
  emitFrame();
}

function startLoop() {
  if (timerId != null) return;
  timerId = setInterval(tick, TICK_MS);
}

function stopLoop() {
  if (timerId != null) {
    clearInterval(timerId);
    timerId = null;
  }
}

self.onmessage = (ev) => {
  const msg = ev.data || {};
  switch (msg.type) {
    case 'start':
      paused = false;
      mode = msg.mode === 'mirror' ? 'mirror' : 'owner';
      startLoop();
      emitFrame();
      break;
    case 'stop':
      stopLoop();
      break;
    case 'setPaused':
      paused = !!msg.paused;
      break;
    case 'setMode':
      mode = msg.mode === 'mirror' ? 'mirror' : 'owner';
      break;
    case 'setMarket':
      if (msg.realPrice !== undefined) realPrice = msg.realPrice;
      if (msg.lastPrice !== undefined) lastPrice = msg.lastPrice;
      break;
    case 'replace':
      applySerialized(msg.org);
      emitTopology('replace');
      break;
    case 'command': {
      const cmd = msg.command;
      let ok = false;
      if (cmd === 'mitosis') ok = !!mitosis(org);
      else if (cmd === 'rewire') ok = !!rewire(org);
      else if (cmd === 'apoptosis') ok = apoptosis(org);
      else if (cmd === 'reset') {
        const seeded = seedOrganism();
        org.cells = seeded.cells;
        org.syns = seeded.syns;
        org.generation = seeded.generation || 0;
        org.phySteps = 0;
        compile(org);
        ok = true;
      }
      if (ok) emitTopology(cmd);
      else emitFrame();
      break;
    }
    case 'ping':
      self.postMessage({ type: 'pong', mode, paused, cells: org.cells.length });
      break;
    default:
      break;
  }
};

compile(org);
self.postMessage({ type: 'ready', org: serializeOrg(org) });
