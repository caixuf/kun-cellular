#!/usr/bin/env bash
# ==============================================================================
# KunCellular SDSCC Release Packaging Script
# Generates self-contained standalone deployment tarball & SHA256 checksum
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

VERSION="${1:-v1.0.0}"
PKG_NAME="kun-cellular-${VERSION}-linux-x86_64"
DIST_DIR="${ROOT_DIR}/dist"
STAGE_DIR="${DIST_DIR}/${PKG_NAME}"
ARCHIVE_FILE="${DIST_DIR}/${PKG_NAME}.tar.gz"

echo "======================================================================"
echo "  Packaging KunCellular SDSCC Release: ${VERSION}"
echo "  Package Name: ${PKG_NAME}"
echo "  Target Archive: ${ARCHIVE_FILE}"
echo "======================================================================"

# 1. 验证编译产物存在
if [[ ! -f "${ROOT_DIR}/build/libkun_cellular_runtime.so" ]]; then
    echo "[!] libkun_cellular_runtime.so not found. Building now..."
    cmake -B "${ROOT_DIR}/build" -S "${ROOT_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${ROOT_DIR}/build" --target kun_cellular_runtime train_doudizhu_champion train_maze_navigator -j$(nproc)
fi

# 2. 清理并创建 staging 目录
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/bin"
mkdir -p "${STAGE_DIR}/lib"
mkdir -p "${STAGE_DIR}/build"
mkdir -p "${STAGE_DIR}/checkpoints"
mkdir -p "${STAGE_DIR}/models/business_lifeforms"
mkdir -p "${STAGE_DIR}/frontend/cellular"
mkdir -p "${STAGE_DIR}/tools"
mkdir -p "${STAGE_DIR}/include/kun/cellular"
mkdir -p "${STAGE_DIR}/src"

# 3. 拷贝核心动态库与原生二进制可执行文件
echo "[*] Copying runtime binaries and shared libraries..."
cp "${ROOT_DIR}/build/libkun_cellular_runtime.so" "${STAGE_DIR}/lib/"
cp "${ROOT_DIR}/build/libkun_cellular_runtime.so" "${STAGE_DIR}/build/"
cp "${ROOT_DIR}/build/libkun_cellular_runtime.so" "${STAGE_DIR}/bin/"

for bin_name in train_doudizhu_champion train_maze_navigator train_multi_asset_quant_master train_flagship_voxel test_flow_doudizhu_card_game test_universal_runtime; do
    if [[ -f "${ROOT_DIR}/build/${bin_name}" ]]; then
        cp "${ROOT_DIR}/build/${bin_name}" "${STAGE_DIR}/bin/"
        cp "${ROOT_DIR}/build/${bin_name}" "${STAGE_DIR}/build/"
    fi
done

# 4. 拷贝神圣底座头文件 (C/C++ Single Source of Truth) 与核心源码
echo "[*] Copying C11 substrate headers and source..."
cp -r "${ROOT_DIR}/include/kun/cellular/"* "${STAGE_DIR}/include/kun/cellular/"
cp -r "${ROOT_DIR}/src/"* "${STAGE_DIR}/src/"
cp "${ROOT_DIR}/CMakeLists.txt" "${STAGE_DIR}/"

# 5. 拷贝检查点与生命体清单 (SDSC-BIN v2 纯二进制)
echo "[*] Copying checkpoints & lifeforms manifest..."
cp "${ROOT_DIR}/checkpoints/"*.bin "${STAGE_DIR}/checkpoints/"
cp "${ROOT_DIR}/models/business_lifeforms/manifest.json" "${STAGE_DIR}/models/business_lifeforms/"

# 6. 拷贝前端沙盒与 3D 流形模块
echo "[*] Copying frontend sandboxes and WebGL modules..."
cp "${ROOT_DIR}/frontend/"*.html "${STAGE_DIR}/frontend/"
cp -r "${ROOT_DIR}/frontend/cellular/"* "${STAGE_DIR}/frontend/cellular/"

# 7. 拷贝后端服务与运行时驱动
echo "[*] Copying live backend server and runtime bindings..."
cp "${ROOT_DIR}/tools/cellular_live_backend.py" "${STAGE_DIR}/tools/"
cp "${ROOT_DIR}/tools/cellular_c_runtime.py" "${STAGE_DIR}/tools/"
cp "${ROOT_DIR}/tools/train_doudizhu_master_cortex.py" "${STAGE_DIR}/tools/"

# 8. 拷贝文档与说明
cp "${ROOT_DIR}/README.md" "${STAGE_DIR}/"

# 9. 生成开箱即用的一键启动脚本 start.sh
cat << 'LAUNCH_EOF' > "${STAGE_DIR}/start.sh"
#!/usr/bin/env bash
# ==============================================================================
# KunCellular SDSCC - One-Click Launcher
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${1:-8833}"

echo ""
echo "======================================================================"
echo "   ____  __.            _________        .__  .__         .__                 "
echo "  |    |/ _|__ __  ____ \\_   ___ \\  ____ |  | |  |  __ __ |  | _____ _______ "
echo "  |      < |  |  \\/    \\/    \\  \\/_/ __ \\|  | |  | |  |  \\|  | \\__  \\\\_  __ \\"
echo "  |    |  \\|  |  /   |  \\     \\___\\  ___/|  |_|  |_|  |  /|  |__/ __ \\|  | \\/"
echo "  |____|__ \\____/|___|  /\\______  /\\___  >____/____/____/ |____(____  /__|   "
echo "          \\/          \\/        \\/     \\/                           \\/        "
echo "======================================================================"
echo " Software-Defined Silicon Cellular Computer (SDSCC) v1.0.0"
echo " Zero Heap Allocation | 19ns Hard Real-Time | Non-Von-Neumann Substrate"
echo "======================================================================"
echo ""

# 检查 Python 3 环境
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is required to run the live gateway server."
    exit 1
fi

# 检查 numpy 依赖
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "[!] numpy not found. Installing numpy via pip..."
    pip3 install numpy || {
        echo "[WARN] Could not install numpy automatically. Running with fallback simulation."
    }
fi

export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${SCRIPT_DIR}/build:${LD_LIBRARY_PATH:-}"

echo "[+] Starting KunCellular Live Backend on http://localhost:${PORT}..."
echo ""
echo "----------------------------------------------------------------------"
echo "  [1] Central Hub:        http://localhost:${PORT}/"
echo "  [2] DouDiZhu Arena:     http://localhost:${PORT}/doudizhu.html"
echo "  [3] 3D Cosmic Manifold: http://localhost:${PORT}/cellular.html"
echo "  [4] ADAS Highway Sim:   http://localhost:${PORT}/vehicle.html"
echo "  [5] Maze Navigator:     http://localhost:${PORT}/maze.html"
echo "  [6] Biosphere Ecology:  http://localhost:${PORT}/ecosystem.html"
echo "----------------------------------------------------------------------"
echo "Press Ctrl+C to terminate the service gracefully."
echo ""

exec python3 tools/cellular_live_backend.py --port "${PORT}"
LAUNCH_EOF

chmod +x "${STAGE_DIR}/start.sh"
chmod +x "${STAGE_DIR}/bin/"* 2>/dev/null || true

# 10. 打包为 tar.gz
echo "[*] Creating archive: ${ARCHIVE_FILE}..."
cd "${DIST_DIR}"
tar -czf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
sha256sum "${PKG_NAME}.tar.gz" > "${PKG_NAME}.tar.gz.sha256"

PKG_SIZE=$(du -h "${ARCHIVE_FILE}" | cut -f1)
echo "======================================================================"
echo "  SUCCESS! Deployment package ready:"
echo "  Archive:  ${ARCHIVE_FILE} (${PKG_SIZE})"
echo "  Checksum: $(cat "${ARCHIVE_FILE}.sha256")"
echo "======================================================================"
