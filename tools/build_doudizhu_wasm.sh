#!/usr/bin/env bash
# Build DouDiZhu rules WASM (harness-aligned single-source).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${HOME}/emsdk/emsdk_env.sh"
OUT_DIR="${ROOT}/frontend/wasm"
mkdir -p "${OUT_DIR}"
em++ "${ROOT}/tools/doudizhu_wasm_exports.cpp" \
  -I"${ROOT}" \
  -O2 \
  -s WASM=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="createDouDiZhuModule" \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s TOTAL_MEMORY=16777216 \
  -s EXPORTED_FUNCTIONS='["_dd_deal","_dd_card_sum","_dd_hand_count","_dd_export_hand","_dd_identify_trick","_dd_can_beat","_dd_legal_actions","_dd_table_type","_dd_table_rank","_dd_turn","_dd_cards_left","_dd_apply_play","_dd_telemetry_tick","_dd_version","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAP32","HEAPF32","getValue","setValue"]' \
  -o "${OUT_DIR}/doudizhu_rules.js"
echo "OK ${OUT_DIR}/doudizhu_rules.js + .wasm"
em++ -v 2>&1 | head -3
