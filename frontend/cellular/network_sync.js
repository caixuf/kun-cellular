/* ============================================================
 * network_sync.js - WebSocket 40Hz 直连、HTTP 容灾轮询、行情与生态场同步
 * ============================================================ */
import * as THREE from 'three';
import { scene } from './scene_setup.js';
import { getGlowTexture } from './texture_cache.js';
import { org, compile, forward } from './organism_model.js';
import { currentOrganismBounds, updateOrganismBounds } from './spatial_bounds.js';
import { rebuildViews, views, lodPointsMesh } from './lod_system.js';
import { camState } from './camera_controller.js';

export let serverOnline = false;
export let wsConnected = false;
export let clientWarpMultiplier = 1.0;
export let lastPrice = 3620;
export let realPrice = null;
export let totalActs = 0;
export let currentSelectedOrgId = null;

export function setCurrentSelectedOrgId(id) {
  currentSelectedOrgId = id;
}

let ws = null;
let wsReconnectTimer = null;
let isFetchingState = false;
let lastExtinctionTs = 0;

export function log(msg, hi = false) {
  const el = document.getElementById('eventlog');
  if (!el) return;
  const d = document.createElement('div');
  d.className = 'ev' + (hi ? ' hi' : '');
  const t = new Date().toTimeString().slice(0, 8);
  d.textContent = `[${t}] ${msg}`;
  el.prepend(d);
  while (el.children.length > 24) el.lastChild.remove();
}

/* 行情: 优先拉真实 rb 主力价 (守护进程 /api/universe), 失败回退随机游走 */
export async function fetchRealPrice() {
  try {
    const r = await fetch('/api/universe', { signal: AbortSignal.timeout(4000) });
    const arr = await r.json();
    const rb = arr.find(x => x.symbol === 'rb') || arr[0];
    if (rb && rb.last) {
      realPrice = rb.last;
      const srcEl = document.getElementById('src');
      if (srcEl) {
        srcEl.textContent = '数据源: SDSCC 真实物理后端 (40Hz 连续推演 · rb 主力行情)';
        srcEl.style.color = 'var(--emerald)';
        srcEl.style.background = 'rgba(52,211,153,0.1)';
      }
    }
  } catch (e) {
    const srcEl = document.getElementById('src');
    if (srcEl && !serverOnline) {
      srcEl.textContent = '数据源: 本地离线仿真模式';
      srcEl.style.color = 'var(--cyan)';
    }
  }
}

export function marketTick() {
  if (realPrice) lastPrice += (realPrice - lastPrice) * 0.15 + (Math.random() - 0.5) * 1.2;
  else lastPrice += (Math.random() - 0.5) * 2.4;
  const vol = 4000 + Math.random() * 3000;
  const spread = 0.8 + Math.random() * 2.5;
  const imb = Math.random() * 2 - 1;
  const actions = forward(org, [lastPrice, vol, spread, imb]);
  totalActs += org.cells.filter(c => Math.abs(c.out) > 1e-6).length;

  for (const s of org.syns) {
    if (!s.active) continue;
    const byId = new Map(org.cells.map(c => [c.id, c]));
    const a = byId.get(s.from);
    if (a && Math.abs(a.out) > 1e-6) s.photon = Math.random() * 0.15;
  }

  const pxEl = document.getElementById('px');
  if (pxEl) pxEl.textContent = lastPrice.toFixed(1);
  const pxSrcEl = document.getElementById('px-src');
  if (pxSrcEl) pxSrcEl.textContent = realPrice ? '(真实行情)' : '(合成游走)';

  if (!serverOnline) {
    const genEl = document.getElementById('st-gen');
    if (genEl) genEl.textContent = org.generation;
    const cellsEl = document.getElementById('st-cells');
    if (cellsEl) cellsEl.textContent = org.cells.length.toLocaleString();
    const synEl = document.getElementById('st-syn');
    if (synEl) synEl.textContent = org.syns.length.toLocaleString();
    const physEl = document.getElementById('st-phys');
    if (physEl) physEl.textContent = org.phySteps;
  }
  const actEl = document.getElementById('st-act');
  if (actEl) actEl.textContent = totalActs;
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v.toFixed(2);
  };
  set('v-buy', actions.buy);
  set('v-sell', actions.sell);
  const immEl = document.getElementById('v-immune');
  if (immEl) immEl.textContent = actions.immune ? '熔断!' : '—';
  const ab = document.getElementById('act-buy');
  const as = document.getElementById('act-sell');
  const ai = document.getElementById('act-immune');
  if (ab) ab.className = 'act' + (actions.buy > 0.2 ? ' flash-buy' : '');
  if (as) as.className = 'act' + (actions.sell < -0.2 ? ' flash-sell' : '');
  if (ai) ai.className = 'act' + (actions.immune ? ' flash-immune' : '');
}

export async function setWarp(speed) {
  const speedKey = (speed === 'unlimited' || speed === 'max') ? 'max' : speed;
  ['1x', '100x', '1000x', 'max'].forEach(k => {
    ['hw-', 'qw-'].forEach(prefix => {
      const btn = document.getElementById(prefix + k);
      if (btn) btn.classList.remove('active');
    });
  });

  ['hw-', 'qw-'].forEach(prefix => {
    const activeBtn = document.getElementById(prefix + speedKey);
    if (activeBtn) activeBtn.classList.add('active');
  });

  if (speed === '1x') clientWarpMultiplier = 1.0;
  else if (speed === '100x') clientWarpMultiplier = 3.5;
  else if (speed === '1000x') clientWarpMultiplier = 8.0;
  else clientWarpMultiplier = 18.0;

  const stWarpEl = document.getElementById('st-warp');
  if (stWarpEl) stWarpEl.textContent = speed;
  log(` 演化时空曲率已切至: ${speed} (物理脉冲加速 x${clientWarpMultiplier.toFixed(1)})`, true);

  try {
    fetch('/api/control/warp?speed=' + speed, { method: 'POST' }).catch(() => {});
    fetch('/api/warp?speed=' + speed).catch(() => {});
  } catch (e) {}
}

export async function setStress(level) {
  fetch("/api/stress?level=" + level).catch(() => {});
  try {
    await fetch('/api/control/stress?level=' + level, { method: 'POST' });
    log(level === 'extreme' ? ' [兵部] 触发红皇后极端流动性闪崩与对抗加压!' : '[OK] 恢复平稳演化生境', true);
  } catch (e) {
    console.error(e);
  }
}

export async function pollIslands() {
  try {
    const r = await fetch('/api/islands/status', { signal: AbortSignal.timeout(2000) });
    const j = await r.json();
    if (j.warp_mode) {
      const el = document.getElementById('st-warp');
      if (el) el.textContent = j.warp_mode;
    }
    if (j.total_generations != null) {
      const elTg = document.getElementById('st-total-gens');
      if (elTg) elTg.textContent = j.total_generations.toLocaleString();
    }
    if (j.total_migrations != null) {
      const elTm = document.getElementById('st-total-mig');
      if (elTm) elTm.textContent = j.total_migrations + ' 次';
    }

    if (j.islands && j.islands.length) {
      const html = j.islands.map(isl => {
        return `<div style="background:rgba(15,23,42,.6);border:1px solid rgba(30,41,59,.8);padding:4px 6px;border-radius:4px;">
          <div style="display:flex;justify-content:space-between;color:var(--cyan);font-weight:bold;">
            <span>Island-${isl.island_id}</span>
            <span style="color:#64748b">C${isl.core_id}</span>
          </div>
          <div style="color:#e2e8f0;font-size:9px;">Gens: <b>${isl.generations}</b></div>
          <div style="color:var(--emerald);font-size:9px;">Fit: <b>${isl.best_fitness.toFixed(1)}</b></div>
          <div style="color:#94a3b8;font-size:8px;">Mig: ⇄ ${isl.migration_in}/${isl.migration_out}</div>
        </div>`;
      }).join('');
      const grid = document.getElementById('islands-grid');
      if (grid) grid.innerHTML = html;
    }
  } catch (e) {}
}

/* ============================================================
 * 宏观生态圈层 (EcoBiosphere Dome) & 量子辐射场
 * ============================================================ */
export const NICHE_COLOR = { 'Producer': 0x4ade80, 'Herbivore': 0x38bdf8, 'Predator': 0xfb923c, 'Decomposer': 0xc084fc };
export let bioLayerVisible = true;
export let bioStep = 0;
export const bioViews = new Map();

export function toggleBioLayer() {
  bioLayerVisible = !bioLayerVisible;
  for (const v of bioViews.values()) v.sprite.visible = bioLayerVisible;
  return bioLayerVisible;
}

function nicheKey(str) { return (str || '').split('(')[0]; }

export class BioAgentView {
  constructor(agent) {
    this.id = agent.id;
    this.color = NICHE_COLOR[nicheKey(agent.niche)] || 0x94a3b8;
    const mat = new THREE.SpriteMaterial({
      map: getGlowTexture(),
      color: this.color,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.sprite = new THREE.Sprite(mat);
    scene.add(this.sprite);
    this.target = new THREE.Vector3();
    this.energy = agent.energy;
  }
  update(agent, t) {
    const v = new THREE.Vector3(agent.x, agent.y, agent.z);
    if (v.length() < 1e-3) v.set(0, 1, 0);
    v.normalize();
    const r = 300 + Math.min(60, Math.max(0, agent.energy) * 0.35) + Math.sin(t * 1.3 + agent.id) * 8;
    this.target.copy(v).multiplyScalar(r);
    this.sprite.position.lerp(this.target, 0.06);
    const s = 26 + Math.min(1, Math.max(0, agent.energy) / 120) * 34;
    this.sprite.scale.set(s, s, 1);
    this.sprite.material.opacity = 0.35 + Math.min(1, Math.max(0, agent.energy) / 120) * 0.6;
  }
  dispose() { scene.remove(this.sprite); }
}

export async function pollBiosphere() {
  try {
    const r = await fetch('/api/biosphere/status', { signal: AbortSignal.timeout(5000) });
    const j = await r.json();
    bioStep = j.step;
    const _setT = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    _setT('bio-step', `世代 ${j.step}`);
    _setT('bio-shannon', (j.shannon_diversity ?? 0).toFixed(2));
    _setT('bio-prod', j.niche_counts?.producers ?? 0);
    _setT('bio-herb', j.niche_counts?.herbivores ?? 0);
    _setT('bio-pred', j.niche_counts?.predators ?? 0);
    _setT('bio-deco', j.niche_counts?.decomposers ?? 0);

    const biomesEl = document.getElementById('bio-biomes');
    if (biomesEl) {
      biomesEl.innerHTML = (j.biomes || []).map(b => {
        const tag = b.climate.startsWith('Spring') ? '[春季]' : b.climate.startsWith('Summer') ? '[夏季]' : b.climate.startsWith('Autumn') ? '[秋季]' : '[冬季]';
        const pct = Math.min(100, b.nutrient / 12);
        return `<div><span style="color:var(--cyan);font-weight:bold">${tag}</span> ${b.name.replace('Biome-', '')} · ${b.climate.split('(')[1] || ''}
          <div style="height:3px;background:#1e293b;border-radius:2px;margin-top:2px">
            <div style="width:${pct}%;height:100%;background:#22d3ee;border-radius:2px"></div></div></div>`;
      }).join('');
    }

    const aliveIds = new Set();
    for (const a of (j.agents || [])) {
      aliveIds.add(a.id);
      if (!bioViews.has(a.id)) bioViews.set(a.id, new BioAgentView(a));
      bioViews.get(a.id).lastAgent = a;
    }
    for (const [id, view] of bioViews) {
      if (!aliveIds.has(id)) { view.dispose(); bioViews.delete(id); }
    }
    pollRadiation(j.radiation);
    const raysEl = document.getElementById('bio-rays');
    if (raysEl) raysEl.textContent = radRays.size;
    const srcEl = document.getElementById('src');
    if (srcEl) srcEl.textContent = '数据源: 守护进程 · 细胞+生态圈+辐射场三引擎在线';
  } catch (e) {
    const stepEl = document.getElementById('bio-step');
    if (stepEl) stepEl.textContent = '(离线)';
  }
}

export let radVisible = true;
let radPrevEvents = 0;
const RAD_DEFAULT_WAVES = [
  { kx: 0.08, ky: 0.04, kz: 0.02, omega: 1.2, amplitude: 1.0, phase: 0 },
  { kx: -0.05, ky: 0.07, kz: 0.03, omega: 1.5, amplitude: 0.8, phase: 1.04 },
  { kx: 0.03, ky: -0.06, kz: 0.08, omega: 0.9, amplitude: 1.2, phase: 2.09 }
];

export const radUniforms = {
  uTime: { value: 0 },
  uW: { value: [new THREE.Vector4(), new THREE.Vector4(), new THREE.Vector4()] },
  uAmp: { value: new THREE.Vector3(1, 0.8, 1.2) },
  uPhase: { value: new THREE.Vector3(0, 1.04, 2.09) }
};

export const radPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(950, 950),
  new THREE.ShaderMaterial({
    uniforms: radUniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: `varying vec3 vWorld;
      void main(){ vec4 wp = modelMatrix * vec4(position,1.0); vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp; }`,
    fragmentShader: `
      uniform float uTime; uniform vec4 uW[3]; uniform vec3 uAmp; uniform vec3 uPhase;
      varying vec3 vWorld;
      void main(){
        float psi = uAmp.x*cos(dot(uW[0].xyz, vWorld) - uW[0].w*uTime + uPhase.x)
                  + uAmp.y*cos(dot(uW[1].xyz, vWorld) - uW[1].w*uTime + uPhase.y)
                  + uAmp.z*cos(dot(uW[2].xyz, vWorld) - uW[2].w*uTime + uPhase.z);
        float I = psi*psi;
        vec3 col = mix(vec3(0.02,0.05,0.10), vec3(0.10,0.45,0.85), clamp(I*0.32,0.0,1.0));
        col += vec3(0.35,0.85,1.0)*clamp(I-2.2,0.0,1.6)*0.4;
        float alpha = 0.08 + clamp(I*0.13, 0.0, 0.42);
        gl_FragColor = vec4(col, alpha);
      }`
  })
);
radPlane.rotation.x = -Math.PI / 2;
radPlane.position.y = -150;
scene.add(radPlane);

export function toggleRadLayer() {
  radVisible = !radVisible;
  radPlane.visible = radVisible;
  for (const v of radRays.values()) {
    v.sprite.visible = radVisible;
    v.trail.visible = radVisible;
  }
  return radVisible;
}

export function setWaveUniforms(waves) {
  waves.forEach((w, i) => {
    if (i >= 3) return;
    radUniforms.uW.value[i].set(w.kx, w.ky, w.kz, w.omega);
    radUniforms.uAmp.value.setComponent(i, w.amplitude);
    radUniforms.uPhase.value.setComponent(i, w.phase);
  });
}
setWaveUniforms(RAD_DEFAULT_WAVES);

export class RadRayView {
  constructor(r) {
    this.id = r.id;
    this.origin = new THREE.Vector3(r.ox, r.oy, r.oz);
    this.dir = new THREE.Vector3(r.dx, r.dy, r.dz).normalize();
    this.dist = r.dist;
    this.speed = 120;
    this.max = 160;
    this.energy = r.energy;
    const mat = new THREE.SpriteMaterial({
      map: getGlowTexture(),
      color: 0x67e8f9,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    this.sprite = new THREE.Sprite(mat);
    this.sprite.scale.set(14, 14, 1);
    this.trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.35 })
    );
    scene.add(this.sprite, this.trail);
  }
  step(dt) {
    this.dist += this.speed * dt;
    const p = this.origin.clone().addScaledVector(this.dir, this.dist);
    this.sprite.position.copy(p);
    const s = 10 + this.energy * 0.08;
    this.sprite.scale.set(s, s, 1);
    const tail = this.origin.clone().addScaledVector(this.dir, Math.max(0, this.dist - 30));
    const pos = this.trail.geometry.attributes.position;
    pos.setXYZ(0, tail.x, tail.y, tail.z);
    pos.setXYZ(1, p.x, p.y, p.z);
    pos.needsUpdate = true;
    return this.dist < this.max;
  }
  dispose() { scene.remove(this.sprite, this.trail); }
}
export const radRays = new Map();

export function pollRadiation(rad) {
  if (!rad) return;
  if (rad.wave_sources?.length) setWaveUniforms(rad.wave_sources);
  const evEl = document.getElementById('bio-rad-ev');
  if (evEl) evEl.textContent = rad.recent_events_count ?? 0;
  if ((rad.recent_events_count ?? 0) > radPrevEvents) {
    log(`[RAD_EVENT] 辐射诱变 ×${rad.recent_events_count - radPrevEvents} (软电离/硬突变/量子隧穿)`);
  }
  radPrevEvents = rad.recent_events_count ?? 0;
  const alive = new Set();
  for (const r of (rad.active_cosmic_rays || [])) {
    alive.add(r.id);
    if (!radRays.has(r.id)) radRays.set(r.id, new RadRayView(r));
  }
  for (const [id, v] of radRays) {
    if (!alive.has(id) && v.dist > v.max * 0.9) {
      v.dispose();
      radRays.delete(id);
    }
  }
}

/* ============================================================
 * WebSocket 40Hz 直连与 REST 容灾同步
 * ============================================================ */
export function connectWebSocket() {
  if (ws) {
    try { ws.close(); } catch (e) {}
  }
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${window.location.host}/ws`;

  try {
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      wsConnected = true;
      serverOnline = true;
      const wsDot = document.getElementById("ws-dot");
      if (wsDot) wsDot.style.background = "var(--emerald)";
      const wsText = document.getElementById("ws-text");
      if (wsText) wsText.textContent = "WS 40Hz (直连)";
      const srcEl = document.getElementById("src");
      if (srcEl) {
        srcEl.textContent = "数据源: WebSocket 40Hz 零延迟双工直连 (实时力场 + 24原语前向)";
        srcEl.style.color = "var(--emerald)";
        srcEl.style.background = "rgba(52,211,153,0.1)";
      }
      log("[PULSE] [WebSocket] 已连接至 SDSCC 40Hz 实时双工全息遥测流 (零延迟直连)", true);
    };
    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.status === "ok" && data.type === "action_ack") {
          log(`[WS ACK] ${data.action}: ${JSON.stringify(data.result?.message || data.result)}`, true);
          return;
        }
        updateFromBackendState(data);
      } catch (err) {
        console.error("WS Parse error", err);
      }
    };
    ws.onclose = () => {
      wsConnected = false;
      const wsDot = document.getElementById("ws-dot");
      if (wsDot) wsDot.style.background = "var(--amber)";
      const wsText = document.getElementById("ws-text");
      if (wsText) wsText.textContent = "WS 离线重连中";
      scheduleWsReconnect();
    };
    ws.onerror = () => {
      wsConnected = false;
      const wsDot = document.getElementById("ws-dot");
      if (wsDot) wsDot.style.background = "var(--crimson)";
    };
  } catch (e) {
    wsConnected = false;
    scheduleWsReconnect();
  }
}

export function scheduleWsReconnect() {
  if (wsReconnectTimer) return;
  wsReconnectTimer = setTimeout(() => {
    wsReconnectTimer = null;
    connectWebSocket();
  }, 2500);
}

export async function sendBackendCommand(action, params = {}) {
  const payload = { action, ...params };
  if (ws && wsConnected && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
    return { status: "sent_ws" };
  } else {
    if (action === "extinction") {
      const w = params.wipeout_ratio ?? 0.8;
      const s = params.shock_scale ?? 2.5;
      const r = await fetch(`/api/extinction/trigger?wipeout_ratio=${w}&shock_scale=${s}`, { method: "POST" });
      return await r.json();
    } else if (action === "splice") {
      const name = params.name || params.organ_name || "schmitt_damping_column";
      const from_id = params.from_id !== undefined ? `&from_id=${params.from_id}` : "";
      const to_id = params.to_id !== undefined ? `&to_id=${params.to_id}` : "";
      const r = await fetch(`/api/organ/splice?name=${encodeURIComponent(name)}${from_id}${to_id}`, { method: "POST" });
      return await r.json();
    } else if (action === "lyapunov_enforce") {
      const max_g = params.max_gain ?? 0.95;
      const r = await fetch(`/api/lyapunov/enforce?max_gain=${max_g}`, { method: "POST" });
      return await r.json();
    }
  }
}

export async function syncBackendState() {
  if (wsConnected) return;
  if (isFetchingState) return;
  isFetchingState = true;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 800);
    const r = await fetch("/api/state", { signal: controller.signal });
    clearTimeout(timeoutId);
    if (r.ok) {
      const data = await r.json();
      if (!serverOnline) {
        serverOnline = true;
        const srcEl = document.getElementById("src");
        if (srcEl) {
          srcEl.textContent = "数据源: SDSCC 真实实时物理后端 (HTTP 轮询 40Hz 连续推演)";
          srcEl.style.color = "var(--emerald)";
          srcEl.style.background = "rgba(52,211,153,0.1)";
        }
        log("已连接至 SDSCC 真实物理后端引擎 (HTTP 轮询 40Hz 连续力场 + 24原语前向)", true);
      }
      updateFromBackendState(data);
    }
  } catch (e) {
    if (serverOnline && !wsConnected) {
      serverOnline = false;
      const srcEl = document.getElementById("src");
      if (srcEl) {
        srcEl.textContent = "数据源: 本地离线平滑降级 (60FPS)";
        srcEl.style.color = "var(--cyan)";
      }
    }
  } finally {
    isFetchingState = false;
  }
}

export function updateFromBackendState(data) {
  if (!data || !data.cells) return;
  if (data.organism_id) {
    currentSelectedOrgId = data.organism_id;
  }

  const topoFingerprint = `${data.organism_id || ''}_${data.cells.length}_${data.macro_synapses || (data.synapses || []).length}`;
  if (org.lastFingerprint !== topoFingerprint) {
    org.lastFingerprint = topoFingerprint;
    org.generation = data.generation;
    org.cells = data.cells.map(c => ({
      id: c.id,
      type: c.type,
      param1: c.p1 || 0.1,
      param2: c.p2 || 0.0,
      state: c.s || 0.0,
      out: c.out || 0.0,
      acts: c.acts || 0,
      _rawX: c.x !== undefined ? c.x : 0,
      _rawY: c.y !== undefined ? c.y : 0,
      _rawZ: c.z !== undefined ? c.z : 0,
      x: c.x,
      y: c.y,
      z: c.z || 0,
      vx: 0, vy: 0, vz: 0,
      glow: 0
    }));
    if (data.synapses) {
      org.syns = data.synapses.map(s => ({
        from: s.from,
        to: s.to,
        port: s.port || 0,
        w: s.w !== undefined ? s.w : (s.weight !== undefined ? s.weight : 1.0),
        active: s.active !== undefined ? s.active : true,
        photon: -1
      }));
    }
    const isNewOrg = (org.lastOrganismId !== data.organism_id);
    org.lastOrganismId = data.organism_id;
    compile(org);
    updateOrganismBounds(data);
    rebuildViews();
    if (isNewOrg) {
      camState.targetLookAt.copy(currentOrganismBounds.center);
      camState.targetCamR = currentOrganismBounds.macroDist;
      camState.isCamTransitioning = true;
    }
    log(`[GROWTH] 通用拓扑流形与宇宙相对空间重构: ${data.organism_id || 'SDSCC'} (${(currentOrganismBounds.cellScale || org.cells.length).toLocaleString()} 细胞 · 半径 ${currentOrganismBounds.radius.toFixed(1)}m · 宇宙体积占比 ${((currentOrganismBounds.volumeRatio || 0) * 100).toFixed(4)}%)`, true);
  }

  // 1. 节流更新 DOM 文本指标
  const genEl = document.getElementById("st-gen");
  if (genEl && genEl.textContent != data.generation) genEl.textContent = data.generation;

  const macroCells = data.macro_cells || data.n_macro_cells || (data.stats && data.stats.active_cells) || org.cells.length;
  const cellsEl = document.getElementById("st-cells");
  if (cellsEl) cellsEl.textContent = macroCells.toLocaleString();

  const realCells = (data.stats && data.stats.projection_cores) || data.cells_count || (data.cells && data.cells.length) || org.cells.length;
  const realCellsEl = document.getElementById("st-real-cells");
  if (realCellsEl) {
    const solidCount = (views && views.cells) ? views.cells.length : 0;
    const ptCount = (lodPointsMesh && lodPointsMesh.geometry && lodPointsMesh.geometry.attributes.position) ? lodPointsMesh.geometry.attributes.position.count : realCells;
    const cellScale = (currentOrganismBounds && currentOrganismBounds.cellScale) || macroCells || realCells;
    const isLargeScale = (cellScale >= 1000) || (realCells > 256);

    if (!isLargeScale) {
      realCellsEl.textContent = `${solidCount}/${realCells} 实体全量 (100% 显微实化)`;
    } else if (solidCount > 0) {
      realCellsEl.textContent = `${solidCount} 实体视锥局部实化 / ${ptCount.toLocaleString()} 点云流形`;
    } else {
      realCellsEl.textContent = `${ptCount.toLocaleString()} 点云流形 (宏观视距)`;
    }
    const isSurrogate = macroCells > realCells * 5;
    realCellsEl.title = isSurrogate
      ? `底座宏观规模 ${macroCells.toLocaleString()}；前端视距动态 LOD：${solidCount} 实体细胞近距实化，全视界 ${ptCount.toLocaleString()} 动力学流形点云`
      : `真实细胞数与活跃细胞数一致 (${realCells.toLocaleString()})`;
  }

  const macroSyns = data.macro_synapses || data.n_macro_synapses || (data.stats && data.stats.total_synapses) || org.syns.length;
  const synEl = document.getElementById("st-syn");
  if (synEl) synEl.textContent = macroSyns.toLocaleString();

  const physEl = document.getElementById("st-phys");
  if (physEl) physEl.textContent = data.phy_steps;

  const feEl = document.getElementById("st-fe");
  if (feEl && data.free_energy !== undefined) feEl.textContent = data.free_energy.toFixed(4);
  const vitalStabSubEl = document.getElementById("vital-stability-sub");
  if (vitalStabSubEl && data.free_energy !== undefined) vitalStabSubEl.textContent = `预测自由能 F: ${data.free_energy.toFixed(4)}`;

  const fluxEl = document.getElementById("st-flux");
  if (fluxEl && data.plasticity_flux !== undefined) fluxEl.textContent = data.plasticity_flux.toFixed(4);

  const swEl = document.getElementById("st-sw");
  if (swEl && data.clustering_coef !== undefined) swEl.textContent = `${data.clustering_coef.toFixed(2)} / ${data.avg_path_len.toFixed(2)}`;

  if (data.actions) {
    const buyEl = document.getElementById("v-buy");
    if (buyEl) buyEl.textContent = (data.actions.pos || 0).toFixed(2);
    const sellEl = document.getElementById("v-sell");
    if (sellEl) sellEl.textContent = (data.actions.neg || 0).toFixed(2);
    const lockEl = document.getElementById("v-immune");
    const isLocked = (data.actions.lock || 0) > 0.5;
    if (lockEl) lockEl.textContent = isLocked ? "LOCKED" : "NORMAL";
    const actImmune = document.getElementById("act-immune");
    if (actImmune) actImmune.classList.toggle("flash-immune", isLocked);
  }

  if (data.lyapunov) {
    const lyap = data.lyapunov;
    const isStable = !!lyap.is_stable;
    const maxGain = (lyap.max_loop_gain !== undefined ? lyap.max_loop_gain : 0).toFixed(3);
    const cycles = lyap.cycles_count || 0;

    const navGain = document.getElementById("nav-shield-gain");
    if (navGain) navGain.textContent = maxGain;
    const navIcon = document.getElementById("nav-shield-icon");
    if (navIcon) navIcon.textContent = isStable ? "[稳态]" : "[失稳]";
    const navShield = document.getElementById("nav-lyapunov-shield");
    if (navShield) {
      navShield.style.color = isStable ? "var(--emerald)" : "var(--crimson)";
      navShield.style.borderColor = isStable ? "rgba(52,211,153,0.35)" : "rgba(244,63,94,0.6)";
      navShield.style.background = isStable ? "rgba(52,211,153,0.12)" : "rgba(244,63,94,0.2)";
      navShield.classList.toggle("lyapunov-unstable-pulse", !isStable);
    }

    const stGain = document.getElementById("st-lyapunov-gain");
    if (stGain) {
      stGain.textContent = maxGain;
      stGain.style.color = isStable ? "var(--emerald)" : "var(--crimson)";
    }
    const stCycles = document.getElementById("st-lyapunov-cycles");
    if (stCycles) stCycles.textContent = cycles;
    const stBadge = document.getElementById("st-lyapunov-badge");
    if (stBadge) {
      stBadge.textContent = isStable ? "超稳态收敛" : "正反馈失稳";
      stBadge.style.color = isStable ? "var(--emerald)" : "var(--crimson)";
      stBadge.style.background = isStable ? "rgba(52,211,153,0.15)" : "rgba(244,63,94,0.2)";
    }
    const cardLyap = document.getElementById("card-evolution-axioms");
    if (cardLyap) {
      cardLyap.style.borderLeftColor = isStable ? "var(--emerald)" : "var(--crimson)";
    }
    const stShieldIcon = document.getElementById("st-shield-icon");
    if (stShieldIcon) stShieldIcon.textContent = isStable ? "[稳态]" : "[失稳]";
  }

  if (data.symbiotic_macro_cells) {
    org.symbiotic_macro_cells = data.symbiotic_macro_cells;
    const macroCount = data.symbiotic_macro_cells_count || data.symbiotic_macro_cells.length;
    const stMacroCount = document.getElementById("st-macro-count");
    if (stMacroCount) stMacroCount.textContent = `${macroCount} 个微柱`;

    const container = document.getElementById("macro-cells-badges");
    if (container && org.lastMacroCount !== macroCount) {
      org.lastMacroCount = macroCount;
      container.innerHTML = "";
      data.symbiotic_macro_cells.forEach(mc => {
        const badge = document.createElement("span");
        badge.className = "macro-badge";
        badge.style.borderColor = mc.color || "var(--cyan)";
        badge.style.color = mc.color || "var(--cyan)";
        badge.style.cursor = "pointer";
        badge.innerHTML = `<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:${mc.color || 'var(--cyan)'}"></span>${mc.label} (${mc.cells_count}胞)`;
        badge.title = `微柱ID: ${mc.id}\n内部元胞: ${mc.cells_count} 个\n感知输入端口: [${mc.sensory_ports?.join(',') || '无'}]\n效应输出端口: [${mc.effector_ports?.join(',') || '无'}]\n[点击] 在 3D 空间对焦该器官包络膜与微柱组织`;
        badge.onclick = () => {
          if (typeof window.focusOrgan === 'function') {
            window.focusOrgan(mc.id || mc.label);
          } else {
            log(`[MACRO_FOCUS] 对焦超细胞共生微柱【${mc.label}】(${mc.cells_count} 内部元胞)`, true);
          }
        };
        container.appendChild(badge);
      });
    }
  }

  if (data.last_extinction && data.last_extinction.triggered) {
    const ext = data.last_extinction;
    if (lastExtinctionTs !== ext.timestamp) {
      lastExtinctionTs = ext.timestamp;
      const repEl = document.getElementById("extinction-impact-report");
      if (repEl) {
        repEl.style.display = "block";
        repEl.innerHTML = `<b>[灭绝冲击] 冲击波已生效:</b> 抹杀 <b>${ext.wiped_synapses_count}</b> 垄断突触，保留 <b>${ext.survivors_count}</b> 幸存体，增益: ${ext.pre_extinction_gain} &rarr; <b>${ext.post_extinction_gain}</b>`;
      }
      log(`[白垩纪大灭绝] ${ext.message}`, true);
    }
  }

  const byId = new Map(data.cells.map(c => [c.id, c]));
  if (views && views.cells) {
    for (const v of views.cells) {
      const remote = byId.get(v.cell.id);
      if (remote) {
        v.cell.x += (remote.x - v.cell.x) * 0.35;
        v.cell.y += (remote.y - v.cell.y) * 0.35;
        v.cell.z += (remote.z - v.cell.z) * 0.35;
        v.cell.out = remote.out;
        v.cell.state = remote.s;
      }
    }
  }
}
