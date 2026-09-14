#!/usr/bin/env bash
# 观测台一键：编译演示运行时 + 启动 8833。稳定性战役 #6。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8833}"

if [[ ! -d build ]]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release
fi
cmake --build build -j"$(nproc)" --target \
  kun_maze_runtime kun_doudizhu_runtime kun_eco_runtime \
  kun_immune_runtime kun_locomotion_runtime kun_slingshot_runtime \
  2>/dev/null || cmake --build build -j"$(nproc)"

echo "starting observatory on http://127.0.0.1:${PORT}/  (vehicle.html / maze.html / doudizhu.html)"
exec python3 tools/cellular_live_backend.py
