#!/usr/bin/env node
/**
 * P1/P2 probe: N=8000 deals/plies vs WASM rules core; illegal rate + agree checks.
 * Usage: node tools/bench_doudizhu_wasm_probe.mjs [seed=20261024] [N=8000]
 */
import { createRequire } from "module";
import path from "path";
import { fileURLToPath } from "url";
import createDouDiZhuModule from "../frontend/wasm/doudizhu_rules.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const seed = Number(process.argv[2] || 20261024);
const N = Number(process.argv[3] || 8000);

const Module = await createDouDiZhuModule();

function legalActions(player) {
  const max = 256;
  const t = Module._malloc(max * 4);
  const r = Module._malloc(max * 4);
  const c = Module._malloc(max * 4);
  const n = Module._dd_legal_actions(player, t, r, c, max);
  const out = [];
  const lim = Math.min(n, max);
  for (let i = 0; i < lim; i++) {
    out.push({
      type: Module.HEAP32[(t >> 2) + i],
      rank: Module.HEAP32[(r >> 2) + i],
      count: Module.HEAP32[(c >> 2) + i],
    });
  }
  Module._free(t); Module._free(r); Module._free(c);
  return { n, out };
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let illegal = 0;
let plies = 0;
let deals = 0;
let sumMismatch = 0;
const disagree = [];

// Independent gold: re-derive can_beat + identify from same harness subset (mirror of core).
function identifyGold(ranks) {
  const counts = new Array(15).fill(0);
  for (const r of ranks) counts[r]++;
  const n = ranks.length;
  if (n === 2 && counts[13] === 1 && counts[14] === 1) return { type: 4, rank: 14, count: 2 };
  let max_c = 0, max_r = -1;
  for (let r = 0; r < 15; r++) if (counts[r] > max_c) { max_c = counts[r]; max_r = r; }
  if (n === 4 && max_c === 4 && max_r < 13) return { type: 3, rank: max_r, count: 4 };
  if (n === 1 && max_c === 1) return { type: 1, rank: max_r, count: 1 };
  if (n === 2 && max_c === 2 && max_r < 13) return { type: 2, rank: max_r, count: 2 };
  return { type: -1, rank: -1, count: 0 };
}

function canBeatGold(neu, prev) {
  if (neu.type < 0 || neu.type === 0) return false;
  if (!prev || prev.type === 0 || prev.type < 0) return true;
  if (neu.type === 4) return !(prev && prev.type === 4);
  if (prev.type === 4) return false;
  if (neu.type === 3) {
    if (prev.type === 3) return neu.rank > prev.rank;
    return true;
  }
  if (prev.type === 3) return false;
  if (neu.type !== prev.type) return false;
  if (neu.count !== prev.count) return false;
  return neu.rank > prev.rank;
}

const rng = mulberry32(seed);
const targetPlies = N;

while (plies < targetPlies) {
  const dseed = (seed + deals * 2654435761) >>> 0;
  const landlord = deals % 3;
  const sum = Module._dd_deal(dseed, landlord);
  deals++;
  if (sum !== 54 || Module._dd_card_sum() !== 54) sumMismatch++;

  let guard = 0;
  while (plies < targetPlies && guard++ < 200) {
    const turn = Module._dd_turn();
    const { n, out } = legalActions(turn);
    if (n <= 0) { illegal++; break; }
    const pick = out[Math.floor(rng() * out.length)];
    // Agree: identify ranks reconstructed from cand should match type
    if (pick.type > 0) {
      const ranks = [];
      if (pick.type === 4) { ranks.push(13, 14); }
      else for (let i = 0; i < pick.count; i++) ranks.push(pick.rank);
      const g = identifyGold(ranks);
      const ptr = Module._malloc(ranks.length * 4);
      for (let i = 0; i < ranks.length; i++) Module.HEAP32[(ptr >> 2) + i] = ranks[i];
      const tPtr = Module._malloc(12);
      const wt = Module._dd_identify_trick(ptr, ranks.length, tPtr, tPtr + 4, tPtr + 8);
      const wr = Module.HEAP32[(tPtr + 4) >> 2];
      Module._free(ptr); Module._free(tPtr);
      if (wt !== g.type || wr !== g.rank) {
        disagree.push({ kind: "identify", dseed, pick, g, wt, wr });
      }
      const tableT = Module._dd_table_type();
      const tableR = Module._dd_table_rank();
      const prev = tableT === 0 ? null : { type: tableT, rank: tableR, count: tableT === 4 ? 2 : tableT === 3 ? 4 : tableT === 2 ? 2 : 1 };
      const goldBeat = canBeatGold({ type: pick.type, rank: pick.rank, count: pick.count }, prev);
      const wasmBeat = Module._dd_can_beat(pick.type, pick.rank, pick.count, prev ? prev.type : 0, prev ? prev.rank : -1, prev ? prev.count : 0) === 1;
      if (goldBeat !== wasmBeat) disagree.push({ kind: "can_beat", dseed, pick, prev, goldBeat, wasmBeat });
      if (!goldBeat && tableT !== 0) {
        // legal_actions should only emit beaters; count as illegal probe miss
        illegal++;
      }
    }
    const rc = Module._dd_apply_play(pick.type, pick.rank, pick.count);
    plies++;
    if (rc === 0) { illegal++; break; }
    if (rc === 2) break; // win
    if (Module._dd_card_sum() < 0 || Module._dd_card_sum() > 54) sumMismatch++;
  }
}

const rate = illegal / Math.max(plies, 1);
console.log(JSON.stringify({
  seed,
  N_target: N,
  deals,
  plies,
  illegal,
  illegal_rate: rate,
  sum_mismatch: sumMismatch,
  disagree_n: disagree.length,
  disagree_samples: disagree.slice(0, 5),
  pass_p1: rate <= 0.001,
  pass_p2: disagree.length === 0 && sumMismatch === 0,
}, null, 2));
process.exit(rate <= 0.001 && disagree.length === 0 && sumMismatch === 0 ? 0 : 1);
