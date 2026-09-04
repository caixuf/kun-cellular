#!/usr/bin/env python3
"""
Substrate Purity Guard (底座纯洁度静态扫描器)
==============================================
扫描 include/kun/cellular/ 与 include/kun/core/ 底座代码库，
确保不存在任何反向浸润的具身业务专有词汇（ADAS、车规、量化金融等），
恪守《KunCellular 最高架构宪章》第 2 条与第 7 条绝对正交解耦准则。
"""

import os
import re
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCAN_DIRS = [
    os.path.join(ROOT_DIR, "include", "kun", "cellular"),
    os.path.join(ROOT_DIR, "include", "kun", "core"),
]

# 违禁词正则表达式
FORBIDDEN_PATTERN = re.compile(
    r"\b(ADAS|adas|steer|vehicle|PnL|pnl|quant_|lane|AEB)\b|方向盘|车道|盘口|平仓|K线"
)


def main():
    violations = []
    scanned_files = 0

    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue
        for root, _, files in os.walk(scan_dir):
            for file in files:
                if not file.endswith((".h", ".hpp", ".c", ".cpp", ".inl")):
                    continue
                scanned_files += 1
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, ROOT_DIR)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            matches = [m.group(0) for m in FORBIDDEN_PATTERN.finditer(line)]
                            if matches:
                                violations.append((rel_path, line_no, line.strip(), set(matches)))
                except Exception as e:
                    print(f"Error reading {rel_path}: {e}")
                    return 2

    print(f"============================================================")
    print(f"  KunCellular Substrate Purity Guard")
    print(f"  Scanned: {scanned_files} header/source files in include/kun/")
    print(f"============================================================")

    if violations:
        print(f"FAILED: Found {len(violations)} substrate purity violation(s)!\n")
        for rel_path, line_no, line, matches in violations:
            print(f"  {rel_path}:{line_no} [违禁词: {', '.join(sorted(matches))}]")
            print(f"    ↳ {line[:120]}")
        print("\n《KunCellular 最高架构宪章》铁律：严禁业务层专有词汇与契约浸润通用计算底座。")
        print("请将业务专有契约、感知编码器与模型适配移至 tasks/ 适配层。")
        sys.exit(1)

    print("SUCCESS: 0 forbidden domain terms found. Substrate purity verified! (Exit 0)")
    sys.exit(0)


if __name__ == "__main__":
    main()
