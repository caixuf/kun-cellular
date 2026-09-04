/* ============================================================
 * organism_model.js - 细胞与突触生物动力学拓扑模型
 * ============================================================ */
import { T, FAMILY, MUT_CANDIDATES } from './config.js';

let nextCellId = 1;

export function makeCell(type, p1 = 0.1, p2 = 0.0, x = 0, y = 0, z = 0) {
  return {
    id: nextCellId++,
    type,
    param1: p1,
    param2: p2,
    state: 0,
    prev: 0,
    latch: false,
    out: 0,
    acts: 0,
    x,
    y,
    z,
    _rawX: x,
    _rawY: y,
    _rawZ: z,
    vx: 0,
    vy: 0,
    vz: 0,
    fx: 0,
    fy: 0,
    fz: 0,
    glow: 0
  };
}

export function makeSyn(a, b, port, w = 1.0) {
  return {
    from: a,
    to: b,
    port,
    w,
    active: true,
    rest: 60,
    photon: -1
  };
}

export function makeMatureOrganism() {
  nextCellId = 0;
  const cells = [];
  const syns = [];

  // 1. 感知受体层 (Receptors)
  cells.push(makeCell(T.SENSE0, 1.0, 0, -220, -90));
  cells.push(makeCell(T.SENSE1, 1.0, 0, -220, -30));
  cells.push(makeCell(T.SENSE2, 1.0, 0, -220,  30));
  cells.push(makeCell(T.SENSE3, 1.0, 0, -220,  90));

  // 2. 代谢滤波与特征提取层 (Metabolic Filter & Feature Extraction)
  cells.push(makeCell(T.EMA, 0.05, 0, -140, -110));
  cells.push(makeCell(T.EMA, 0.20, 0, -140,  -60));
  cells.push(makeCell(T.DIFF, 0, 0,   -140,    0));
  cells.push(makeCell(T.INTEGRAL, 0.02, 0, -140, 50));
  cells.push(makeCell(T.OSCILLATOR, 1.2, 0.05, -140, 110));
  cells.push(makeCell(T.DELAY_N, 0.5, 0, -80, -90));
  cells.push(makeCell(T.ABS, 0, 0, -80, -40));
  cells.push(makeCell(T.RATIO, 0, 0, -80, 20));
  cells.push(makeCell(T.MUL, 0, 0, -80, 80));

  // 3. 中间因果融合与微柱中间元胞
  cells.push(makeCell(T.SUB, 0, 0, -20, -70));
  cells.push(makeCell(T.SUM, 0, 0, -20, -10));
  cells.push(makeCell(T.QUADRATIC, 0, 0, -20, 50));
  cells.push(makeCell(T.EMA, 0.12, 0, 40, -80));
  cells.push(makeCell(T.DIFF, 0, 0, 40, -30));
  cells.push(makeCell(T.SUM, 0, 0, 40, 30));

  // 4. 门控与非线性调制层 (Gating Layer)
  cells.push(makeCell(T.THRESH, 0.25, 0, 100, -90));
  cells.push(makeCell(T.HYST, 0.35, -0.20, 100, -40));
  cells.push(makeCell(T.AND, 0, 0, 100, 10));
  cells.push(makeCell(T.INHIB, 0, 0, 100, 60));
  cells.push(makeCell(T.DEADZONE, 0.15, 0, 100, 110));

  // 5. 效应与动作输出层 (Effectors)
  cells.push(makeCell('ACT_POS', 1.0, 0, 180, -70));
  cells.push(makeCell('ACT_NEG', 1.0, 0, 180, -20));
  cells.push(makeCell('ACT_RESET', 1.0, 0, 180, 30));
  cells.push(makeCell('ACT_LOCK', 1.0, 0, 180, 80));

  // 突触布线 (标准小世界网络拓扑)
  syns.push(makeSyn(0, 4, 0, 1.2));
  syns.push(makeSyn(0, 5, 0, 0.8));
  syns.push(makeSyn(1, 6, 0, 1.5));
  syns.push(makeSyn(2, 7, 0, 1.0));
  syns.push(makeSyn(3, 8, 0, 1.1));
  syns.push(makeSyn(0, 9, 0, 0.6));
  syns.push(makeSyn(1, 10, 0, 0.9));
  syns.push(makeSyn(2, 11, 0, 0.7));
  syns.push(makeSyn(3, 11, 1, 0.7));
  syns.push(makeSyn(0, 12, 0, 0.5));
  syns.push(makeSyn(3, 12, 1, 0.5));

  syns.push(makeSyn(4, 13, 0, 1.0));
  syns.push(makeSyn(5, 13, 1, -0.8));
  syns.push(makeSyn(6, 14, 0, 1.2));
  syns.push(makeSyn(7, 14, 1, 0.9));
  syns.push(makeSyn(8, 15, 0, 1.1));
  syns.push(makeSyn(9, 15, 1, 0.8));

  syns.push(makeSyn(13, 16, 0, 0.9));
  syns.push(makeSyn(14, 17, 0, 1.3));
  syns.push(makeSyn(15, 18, 0, 0.7));

  syns.push(makeSyn(16, 19, 0, 1.4));
  syns.push(makeSyn(17, 20, 0, 1.1));
  syns.push(makeSyn(18, 21, 0, 0.9));
  syns.push(makeSyn(13, 21, 1, 0.6));
  syns.push(makeSyn(14, 22, 0, -1.0));
  syns.push(makeSyn(15, 23, 0, 0.8));

  syns.push(makeSyn(19, 24, 0, 1.6));
  syns.push(makeSyn(20, 25, 0, 1.5));
  syns.push(makeSyn(21, 24, 1, 0.7));
  syns.push(makeSyn(22, 26, 0, 1.2));
  syns.push(makeSyn(23, 27, 0, 1.0));

  syns.push(makeSyn(16, 4, 1, -0.4));
  syns.push(makeSyn(18, 8, 1, -0.3));

  return {
    cells,
    syns,
    order: [],
    generation: 142,
    phySteps: 0,
    lastFingerprint: null,
    lastOrganismId: 'adas_cortex_champion'
  };
}

export function compile(org) {
  const byId = new Map(org.cells.map((c, i) => [c.id, i]));
  const indeg = new Map(org.cells.map(c => [c.id, 0]));
  const adj = new Map();
  for (const s of org.syns) {
    if (!s.active || !byId.has(s.from) || !byId.has(s.to)) continue;
    if (!adj.has(s.from)) adj.set(s.from, []);
    adj.get(s.from).push(s.to);
    indeg.set(s.to, (indeg.get(s.to) || 0) + 1);
  }
  const q = org.cells.filter(c => (indeg.get(c.id) || 0) === 0).map(c => c.id);
  const order = [];
  for (let h = 0; h < q.length; h++) {
    order.push(q[h]);
    for (const v of (adj.get(q[h]) || [])) {
      const remain = (indeg.get(v) || 0) - 1;
      indeg.set(v, remain);
      if (remain === 0) q.push(v);
    }
  }
  const seen = new Set(order);
  for (const c of org.cells) if (!seen.has(c.id)) order.push(c.id);
  org.order = order.map(id => byId.get(id));
  org.cellMap = new Map(org.cells.map(c => [c.id, c]));
}

export function forward(org, inputs) {
  const byId = new Map(org.cells.map((c, i) => [c.id, i]));
  const port = new Float64Array(org.cells.length * 2);
  for (const s of org.syns) {
    if (!s.active) continue;
    const fi = byId.get(s.from), ti = byId.get(s.to);
    if (fi === undefined || ti === undefined) continue;
    port[ti * 2 + s.port] += org.cells[fi].out * s.w;
  }
  for (const idx of org.order) {
    const c = org.cells[idx]; if (!c) continue;
    const i0 = port[idx * 2], i1 = port[idx * 2 + 1];
    switch (c.type) {
      case T.SENSE0: c.out = inputs[0] * c.param1; break;
      case T.SENSE1: c.out = inputs[1] * c.param1; break;
      case T.SENSE2: c.out = inputs[2] * c.param1; break;
      case T.SENSE3: c.out = inputs[3] * c.param1; break;
      case T.EMA: {
        const a = Math.min(1, Math.max(0.001, c.param1));
        c.state = c.acts === 0 ? i0 : a * i0 + (1 - a) * c.state;
        c.out = c.state;
        break;
      }
      case T.DIFF: c.out = i0 - c.prev; c.prev = i0; break;
      case T.INTEGRAL: c.state += i0 * c.param1; c.out = c.state; break;
      case T.SUM: c.out = i0 + i1; break;
      case T.SUB: c.out = i0 - i1; break;
      case T.MUL: c.out = i0 * i1; break;
      case T.RATIO: c.out = i0 / (Math.abs(i1) > 1e-6 ? i1 : 1e-6); break;
      case T.ABS: c.out = Math.abs(i0); break;
      case T.DELAY_N: {
        if (!c.buf) { c.buf = new Float64Array(16); c.didx = 0; }
        const k = Math.min(16, Math.max(1, Math.floor(c.param1 * 16)));
        const rIdx = (c.didx + 16 - k) & 15;
        c.out = c.buf[rIdx];
        c.buf[c.didx & 15] = i0;
        c.didx = (c.didx + 1) & 15;
        break;
      }
      case T.OSCILLATOR: {
        if (c.acts === 0 && Math.abs(c.state || 0) < 1e-6 && Math.abs(c.aux || 0) < 1e-6) c.state = 0.1;
        const mu = Math.abs(c.param1) > 1e-4 ? Math.min(5, Math.max(0.01, Math.abs(c.param1))) : 1.0;
        const dt = Math.abs(c.param2) > 1e-4 ? Math.min(0.2, Math.max(0.001, Math.abs(c.param2))) : 0.05;
        let s1 = c.state || 0, s2 = c.aux || 0;
        const ds1 = s2, ds2 = mu * (1 - s1 * s1) * s2 - s1 + i0;
        s1 += ds1 * dt; s2 += ds2 * dt;
        c.state = Math.min(10, Math.max(-10, s1));
        c.aux = Math.min(10, Math.max(-10, s2));
        c.out = c.state;
        break;
      }
      case T.QUADRATIC: c.out = c.param1 * i0 * i0 + c.param2 * i0 * i1; break;
      case T.THRESH: c.out = i0 > c.param1 ? 1 : 0; break;
      case T.HYST: {
        if (i0 > c.param1) c.latch = true; else if (i0 < c.param2) c.latch = false;
        c.out = c.latch ? 1 : -1;
        break;
      }
      case T.AND: c.out = (i0 > 0 && i1 > 0) ? 1 : 0; break;
      case T.INHIB: c.out = i0 * Math.max(0, 1 - i1); break;
      case T.DEADZONE: c.out = Math.abs(i0) > Math.abs(c.param1) ? i0 : 0; break;
      case T.MIN_MAX: c.out = c.param1 > 0.5 ? Math.max(i0, i1) : Math.min(i0, i1); break;
      case T.DAMPER: c.state = (c.state || 0) * 0.70 + i0 * 0.30; c.out = c.state; break;
      case T.AMPLIFY: c.out = Math.tanh(i0 * 2.5); break;
      case T.INVERT: c.out = -Math.tanh(i0); break;
      case T.CLIP: c.out = Math.max(-1.0, Math.min(1.0, i0)); break;
      case T.MULTIPLY: c.out = Math.tanh(i0 * 1.5); break;
      case T.THRESHOLD: c.out = i0 > 0.25 ? 1.0 : (i0 < -0.25 ? -1.0 : 0.0); break;
      case T.HYSTERESIS: {
        if (i0 > 0.15) c.state = 1.0;
        else if (i0 < -0.15) c.state = -1.0;
        c.out = c.state || 0.0;
        break;
      }
      case T.INHIBIT: {
        c.state = (c.state || 0) * 0.80 + Math.abs(i0) * 0.20;
        c.out = Math.tanh(i0) * Math.max(0.0, 1.0 - c.state);
        break;
      }
      case T.CORRELATION: {
        c.state = (c.state || 0) * 0.90 + (i0 * (c.aux || 0)) * 0.10;
        c.aux = i0;
        c.out = Math.tanh(c.state);
        break;
      }
      case T.FATIGUE: {
        c.state = Math.min(2.0, (c.state || 0) + Math.abs(i0) * 0.15) * 0.96;
        c.out = Math.tanh(i0) / (1.0 + c.state);
        break;
      }
      default: c.out = i0;
    }
    if (Math.abs(c.out) > 1e-6) { c.acts++; c.glow = Math.min(1, c.glow + 0.3); }
  }
  const actions = { buy: 0, sell: 0, reset: 0, immune: false };
  for (const c of org.cells) {
    if (c.type === 'ACT_POS') actions.buy = c.out;
    else if (c.type === 'ACT_NEG') actions.sell = c.out;
    else if (c.type === 'ACT_RESET') actions.reset = c.out;
    else if (c.type === 'ACT_LOCK' && c.out > 0.5) actions.immune = true;
  }
  return actions;
}

export function stepPhysics(org, dt = 0.016) {
  for (const c of org.cells) { c.glow = (c.glow || 0) * 0.95; }
  for (const s of org.syns) {
    if (s.photon >= 0) { s.photon += dt * 3.5; if (s.photon > 1.0) s.photon = -1.0; }
  }
  org.phySteps++;
}

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

export function mitosis(org) {
  const live = org.syns.filter(s => s.active);
  if (!live.length) return false;
  const old = pick(live);
  const type = pick(MUT_CANDIDATES);
  const nc = makeCell(
    type,
    0.01 + Math.random() * 0.99,
    -(0.01 + Math.random() * 0.99),
    (org.cells[0].x + org.cells[1].x) / 2 + (Math.random() * 60 - 30),
    (Math.random() * 60 - 30)
  );
  org.cells.push(nc);
  old.active = false;
  org.syns.push(makeSyn(old.from, nc.id, 0));
  org.syns.push(makeSyn(nc.id, old.to, old.port, old.w));
  org.generation++;
  compile(org);
  return { type, nc, old };
}

export function rewire(org) {
  if (org.cells.length < 2) return false;
  const targets = org.cells.filter(c => !c.type.startsWith('SENSE'));
  if (targets.length < 1) return false;
  const a = pick(org.cells), b = pick(targets);
  const w = Math.random() * 4 - 2;
  org.syns.push(makeSyn(a.id, b.id, Math.random() < 0.5 ? 0 : 1, w));
  org.generation++;
  compile(org);
  return { a, b, w };
}

export function apoptosis(org) {
  const useful = new Set(org.cells.filter(c => FAMILY(c.type) === 'effector').map(c => c.id));
  let grown = true;
  while (grown) {
    grown = false;
    for (const s of org.syns) {
      if (s.active && useful.has(s.to) && !useful.has(s.from)) { useful.add(s.from); grown = true; }
    }
  }
  const before = org.cells.length;
  org.cells = org.cells.filter(c => c.type.startsWith('SENSE') || useful.has(c.id));
  const alive = new Set(org.cells.map(c => c.id));
  org.syns = org.syns.filter(s => s.active && alive.has(s.from) && alive.has(s.to));
  compile(org);
  const removed = before - org.cells.length;
  return removed;
}
