#!/usr/bin/env bash
#
# KunCellular Cloud Agent 环境安装脚本 (idempotent)
# ------------------------------------------------------------
# 复刻 .github/workflows/ci.yml 的构建/测试前置条件：
#   - 系统级 C/C++ 底座依赖 (libstdc++ / SQLite3 / OpenSSL)
#   - Python 数据管线依赖 (numpy / pytest / pyyaml / torch-cpu)
#   - 生成 SDSC-BIN v2 百万细胞检查点
#   - CMake Release 全量构建 (纯 C 底座 + 测试 + 运行时 .so)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/4] 安装系统级底座依赖 (build-essential / libstdc++ / sqlite3 / openssl)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++-13-dev \
    libsqlite3-dev \
    libssl-dev

echo "==> [2/4] 安装 Python 外围数据管线依赖 (numpy / pytest / pyyaml / torch-cpu)"
python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages numpy pytest pyyaml
python3 -m pip install --break-system-packages torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "==> [3/4] 生成 SDSC-BIN v2 百万细胞检查点 (export_sdsc_binary.py)"
mkdir -p /tmp/opencode
python3 tools/export_sdsc_binary.py

echo "==> [4/4] CMake Release 全量构建 (纯 C 底座 + 测试 + 运行时)"
# 注意: 本镜像 /usr/bin/c++ 与 /usr/bin/cc 默认指向 clang, 其无法链接 -lstdc++;
# 底座为 GCC C++20 (-fcoroutines) 工程, 因此显式钉定 gcc/g++ 与 CI (ubuntu gcc) 对齐。
CC=gcc CXX=g++ cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

echo "==> 安装完成: 底座已构建, 检查点已生成, 环境就绪。"
