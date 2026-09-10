/* ============================================================
 * frame_bus.js - Model → View 帧快照总线（类 Qt Model/View）
 *
 * 约定：
 * - Model（organism / 后端同步）只写可变源状态，再 publish 一帧只读快照
 * - View（Three.js）只读 latest()，禁止在渲染路径里改 org 拓扑
 * - 未来可把 Model 迁到 Worker：Worker postMessage(snapshot) → 主线程 publish
 * ============================================================ */

/** @typedef {{
 *   id: number,
 *   type: string,
 *   x: number, y: number, z: number,
 *   out: number, glow: number, state: number, acts: number
 * }} CellSnap */

/** @typedef {{
 *   from: number, to: number, port: number,
 *   w: number, active: boolean, photon: number
 * }} SynSnap */

/** @typedef {{
 *   revision: number,
 *   t: number,
 *   generation: number,
 *   phySteps: number,
 *   cellCount: number,
 *   synCount: number,
 *   cells: CellSnap[],
 *   syns: SynSnap[],
 *   cellById: Map<number, CellSnap>
 * }} FrameSnapshot */

function captureCells(org) {
  const cells = org && org.cells ? org.cells : [];
  const out = new Array(cells.length);
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i];
    out[i] = {
      id: c.id,
      type: c.type,
      x: c.x || 0,
      y: c.y || 0,
      z: c.z || 0,
      out: c.out || 0,
      glow: c.glow || 0,
      state: c.state || 0,
      acts: c.acts || 0
    };
  }
  return out;
}

function captureSyns(org) {
  const syns = org && org.syns ? org.syns : [];
  const out = new Array(syns.length);
  for (let i = 0; i < syns.length; i++) {
    const s = syns[i];
    const w = s.w !== undefined ? s.w : (s.weight !== undefined ? s.weight : 1.0);
    out[i] = {
      from: s.from,
      to: s.to,
      port: s.port || 0,
      w,
      active: s.active !== false,
      photon: s.photon !== undefined ? s.photon : -1
    };
  }
  return out;
}

export class FrameBus {
  constructor() {
    this.revision = 0;
    /** @type {FrameSnapshot|null} */
    this._front = null;
  }

  /**
   * 从 Model 源状态拍一帧只读快照并翻转 front。
   * @param {object} org
   * @param {{ t?: number }} [meta]
   * @returns {FrameSnapshot}
   */
  publish(org, meta = {}) {
    const cells = captureCells(org);
    const syns = captureSyns(org);
    const cellById = new Map();
    for (let i = 0; i < cells.length; i++) cellById.set(cells[i].id, cells[i]);

    this.revision += 1;
    /** @type {FrameSnapshot} */
    const snap = {
      revision: this.revision,
      t: meta.t !== undefined ? meta.t : performance.now() * 0.001,
      generation: (org && org.generation) || 0,
      phySteps: (org && org.phySteps) || 0,
      cellCount: cells.length,
      synCount: syns.length,
      cells,
      syns,
      cellById
    };
    this._front = snap;
    return snap;
  }

  /**
   * 接收 Worker / 远端已拍好的快照（类 Qt 跨线程 queued connection）。
   * @param {object} snap
   * @returns {FrameSnapshot|null}
   */
  accept(snap) {
    if (!snap || !snap.cells) return null;
    const cellById = (snap.cellById instanceof Map)
      ? snap.cellById
      : (() => {
          const m = new Map();
          for (let i = 0; i < snap.cells.length; i++) m.set(snap.cells[i].id, snap.cells[i]);
          return m;
        })();
    this.revision = snap.revision || (this.revision + 1);
    /** @type {FrameSnapshot} */
    const normalized = {
      revision: this.revision,
      t: snap.t !== undefined ? snap.t : performance.now() * 0.001,
      generation: snap.generation || 0,
      phySteps: snap.phySteps || 0,
      cellCount: snap.cells.length,
      synCount: (snap.syns && snap.syns.length) || 0,
      cells: snap.cells,
      syns: snap.syns || [],
      cellById
    };
    this._front = normalized;
    return normalized;
  }

  /** @returns {FrameSnapshot|null} */
  latest() {
    return this._front;
  }
}

export const frameBus = new FrameBus();

/** Worker 侧可复用的纯函数拍帧（避免循环依赖）。 */
export function captureFrameSnapshot(org, meta = {}) {
  const cells = captureCells(org);
  const syns = captureSyns(org);
  const cellById = new Map();
  for (let i = 0; i < cells.length; i++) cellById.set(cells[i].id, cells[i]);
  return {
    revision: meta.revision || 0,
    t: meta.t !== undefined ? meta.t : performance.now() * 0.001,
    generation: (org && org.generation) || 0,
    phySteps: (org && org.phySteps) || 0,
    cellCount: cells.length,
    synCount: syns.length,
    cells,
    syns,
    cellById
  };
}
