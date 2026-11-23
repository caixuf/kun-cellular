#!/usr/bin/env python3
"""Checkpoint 命名诚实性守卫。

规则：文件名里的规模后缀必须与 SDSC-BIN header 的真实细胞数一致。
  *_320c.bin      -> 320 cells
  *_1m.bin        -> 1,000,000 cells
  *_1million.bin  -> 1,000,000 cells
  *_1b.bin        -> 1,000,000,000 cells

2026-11-23 起，`*_1m/10m/100m.bin` 这类“命名膨胀”会被 CI 直接拦下。
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

MAGIC = 0x53445343  # "SDSC"
SUFFIX_RE = re.compile(r"_(\d+)(c|m|million|b|billion)$", re.IGNORECASE)
UNIT = {
    "c": 1,
    "m": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "billion": 1_000_000_000,
}


def read_cells(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            head = f.read(24)
    except OSError as exc:
        print(f"[ERROR] cannot read {path}: {exc}")
        return None
    if len(head) < 24:
        print(f"[ERROR] {path}: header too short")
        return None
    magic, _version, n_cells, _n_syns, _in_dim, _out_dim = struct.unpack("<6I", head)
    if magic != MAGIC:
        print(f"[ERROR] {path}: bad magic 0x{magic:08x}")
        return None
    return n_cells


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.is_dir():
        print(f"[ERROR] missing {ckpt_dir}")
        return 1

    failures = 0
    checked = 0
    for path in sorted(ckpt_dir.glob("*.bin")):
        n_cells = read_cells(path)
        if n_cells is None:
            failures += 1
            continue
        match = SUFFIX_RE.search(path.stem)
        if not match:
            continue
        checked += 1
        claimed = int(match.group(1)) * UNIT[match.group(2).lower()]
        if claimed != n_cells:
            failures += 1
            print(
                f"[FAIL] {path.name}: name claims {claimed:,} cells, "
                f"header says {n_cells:,}"
            )
        else:
            print(f"[ OK ] {path.name}: {n_cells:,} cells")

    print(f"\nchecked={checked} failures={failures}")
    if failures:
        print("Rename the file to its real header cell count, or fix the header.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
