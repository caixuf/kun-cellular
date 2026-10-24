/**
 * DouDiZhu WASM glue — calls emscripten exports ONLY. No local legality oracle.
 * Loads frontend/wasm/doudizhu_rules.js (createDouDiZhuModule).
 * Design polish 2026-10-24: HUD/copy owned in doudizhu.html; this file stays thin WASM calls only.
 */
(function (global) {
  "use strict";

  const TYPE_NAME = {
    0: "PASS",
    1: "SOLO",
    2: "PAIR",
    3: "BOMB",
    4: "ROCKET",
    [-1]: "INVALID",
  };

  let Module = null;
  let ready = null;

  function ensure() {
    if (ready) return ready;
    ready = (async () => {
      if (typeof createDouDiZhuModule !== "function") {
        throw new Error("createDouDiZhuModule missing — load wasm/doudizhu_rules.js first");
      }
      Module = await createDouDiZhuModule();
      return Module;
    })();
    return ready;
  }

  function heapI32(ptr, n) {
    return new Int32Array(Module.HEAP32.buffer, ptr, n);
  }

  const api = {
    async init() {
      await ensure();
      return Module._dd_version();
    },

    deal(seed, landlord) {
      return Module._dd_deal(seed >>> 0, landlord | 0);
    },

    cardSum() {
      return Module._dd_card_sum();
    },

    handCount(player, rank) {
      return Module._dd_hand_count(player | 0, rank | 0);
    },

    exportHand(player) {
      const ptr = Module._malloc(15 * 4);
      Module._dd_export_hand(player | 0, ptr);
      const arr = Array.from(heapI32(ptr, 15));
      Module._free(ptr);
      return arr;
    },

    identifyTrick(rankIdxList) {
      const n = rankIdxList.length;
      const ptr = Module._malloc(Math.max(n, 1) * 4);
      const view = heapI32(ptr, Math.max(n, 1));
      for (let i = 0; i < n; i++) view[i] = rankIdxList[i] | 0;
      const tPtr = Module._malloc(12);
      const type = Module._dd_identify_trick(ptr, n, tPtr, tPtr + 4, tPtr + 8);
      const rank = Module.HEAP32[(tPtr + 4) >> 2];
      const count = Module.HEAP32[(tPtr + 8) >> 2];
      Module._free(ptr);
      Module._free(tPtr);
      if (type < 0 || type === 0 && n > 0) {
        // type 0 with n>0 is pass-like empty; INVALID is -1
      }
      if (type < 0) return null;
      if (type === 0 && n === 0) return null;
      return {
        type: TYPE_NAME[type] || "INVALID",
        typeId: type,
        rank,
        count,
        cards: null,
        name: TYPE_NAME[type] || "INVALID",
      };
    },

    canBeat(neu, prev) {
      if (!neu) return false;
      const ntype = neu.typeId != null ? neu.typeId : typeIdFromName(neu.type);
      const nrank = neu.rank | 0;
      const ncount = neu.count != null ? neu.count : (neu.cards ? neu.cards.length : 0);
      if (!prev) {
        return Module._dd_can_beat(ntype, nrank, ncount, 0, -1, 0) === 1;
      }
      const ptype = prev.typeId != null ? prev.typeId : typeIdFromName(prev.type);
      const prank = prev.rank | 0;
      const pcount = prev.count != null ? prev.count : (prev.cards ? prev.cards.length : 0);
      return Module._dd_can_beat(ntype, nrank, ncount, ptype, prank, pcount) === 1;
    },

    legalActions(player) {
      const max = 256;
      const t = Module._malloc(max * 4);
      const r = Module._malloc(max * 4);
      const c = Module._malloc(max * 4);
      const n = Module._dd_legal_actions(player | 0, t, r, c, max);
      const out = [];
      const lim = Math.min(n, max);
      for (let i = 0; i < lim; i++) {
        out.push({
          type: Module.HEAP32[(t >> 2) + i],
          rank: Module.HEAP32[(r >> 2) + i],
          count: Module.HEAP32[(c >> 2) + i],
        });
      }
      Module._free(t);
      Module._free(r);
      Module._free(c);
      return out;
    },

    applyPlay(type, rank, count) {
      return Module._dd_apply_play(type | 0, rank | 0, count | 0);
    },

    tableType() { return Module._dd_table_type(); },
    tableRank() { return Module._dd_table_rank(); },
    turn() { return Module._dd_turn(); },
    cardsLeft(p) { return Module._dd_cards_left(p | 0); },

    telemetryTick() {
      const ptr = Module._dd_telemetry_tick();
      const arr = new Float32Array(Module.HEAPF32.buffer, ptr, 16);
      return Array.from(arr);
    },
  };

  function typeIdFromName(name) {
    if (name == null) return 0;
    const u = String(name).toUpperCase();
    if (u === "PASS" || u === "NONE") return 0;
    if (u === "SOLO") return 1;
    if (u === "PAIR") return 2;
    if (u === "BOMB") return 3;
    if (u === "ROCKET") return 4;
    return -1;
  }

  global.DouDiZhuWasm = api;
  global.DouDiZhuWasmTypeId = typeIdFromName;
})(typeof window !== "undefined" ? window : globalThis);
