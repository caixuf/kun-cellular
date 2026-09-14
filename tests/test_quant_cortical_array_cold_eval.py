#!/usr/bin/env python3
"""门禁：正式 43 柱 champion.bin 必须能被任务层冷评 loader 还原（稳定性战役量化缺口）。

只断言结构可加载，不把 OOS 数字改写成 PASS。全量 val/OOS 由
  ./build/eval_cortical_array_bin --report ...
另行落盘。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "checkpoints", "quant_cortical_array_champion.bin")
EXE_CANDIDATES = [
    os.path.join(ROOT, "build", "eval_cortical_array_bin"),
    os.path.join(ROOT, "build", "Release", "eval_cortical_array_bin"),
]


def find_exe():
    for p in EXE_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def test_official_bin_dry_load():
    exe = find_exe()
    assert exe, "eval_cortical_array_bin not built; cmake --build build --target eval_cortical_array_bin"
    assert os.path.isfile(BIN), f"missing {BIN}"
    r = subprocess.run(
        [exe, "--bin", BIN, "--dry-load"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "LOAD_OK" in r.stdout, out
    assert "cells=1032" in r.stdout, out
    assert "syns=1634" in r.stdout, out
    assert "cols=43" in r.stdout, out


if __name__ == "__main__":
    test_official_bin_dry_load()
    print("PASS  official champion.bin dry-load")
