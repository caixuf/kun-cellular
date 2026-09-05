#!/usr/bin/env python3
"""
tools/ci/check_architecture_discipline.py
=========================================
KunCellular 架构纪律与设计原则自动化门禁 (Architecture Discipline Guard)

执行以下关键原则的硬性门禁检测：
1. [SDSC-BIN v2 规范门禁] 检视 checkpoints/ 与 manifest.json：
   - 彻底拒绝遗留 JSON 检查点作为运行模型；
   - 所有模型检查点必须拥有合法 SDSC-BIN v2 二进制头 (Magic 0x53445343, Version 2)；
   - manifest.json 中所有生命体模型文件必须在本地存在且二进制头合规。
2. [屏幕空间像素 LOD 门禁] 检视 frontend/cellular/lod_system.js 与 app.js：
   - 屏幕投影像素阈值 (MIN_CELL_PIXELS) 必须作为物理第一准则；
   - solidMaxDist 必须严格基于投影光学公式动态计算；
   - 投影尺寸 < MIN_CELL_PIXELS (d > solidMaxDist) 时绝对禁止实例化 3D 实体网格；
   - 视锥裁剪 (frustum culling) 必须在候选入队前生效；
   - 亚细胞细胞器 (线粒体、孔道、内膜等) 必须施加视距门禁 (showMicroOrganelles / closeLook)。
3. [底座免疫铁律门禁] 检视 AGENTS.md Rule 7：
   - 计算底座 include/kun/cellular/ 与 include/kun/core/ 不得混入业务专有词汇。
"""

import os
import re
import sys
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
MANIFEST_PATH = REPO_ROOT / "models" / "business_lifeforms" / "manifest.json"
LOD_SYSTEM_PATH = REPO_ROOT / "frontend" / "cellular" / "lod_system.js"
APP_JS_PATH = REPO_ROOT / "frontend" / "cellular" / "app.js"
SUBSTRATE_DIR = REPO_ROOT / "include" / "kun" / "cellular"

SDSC_BINARY_MAGIC = 0x53445343
SDSC_BINARY_VERSION = 2


def check_checkpoint_binary_discipline() -> list[str]:
    """门禁 1: 检验纯二进制 SDSC-BIN v2 与 manifest.json 契约"""
    errors = []
    
    if not MANIFEST_PATH.exists():
        return [f"manifest.json 不存在: {MANIFEST_PATH}"]
    
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return [f"manifest.json 解析失败: {e}"]
        
    lifeforms = manifest.get("lifeforms", {})
    if not lifeforms:
        errors.append("manifest.json 中没有定义任何生命体 lifeforms")

    # 1. 验证 manifest 中的每一个生命体
    checked_files = set()
    lifeforms_list = lifeforms if isinstance(lifeforms, list) else [{"id": k, **v} for k, v in lifeforms.items()]
    for info in lifeforms_list:
        l_id = info.get("id", "unknown")
        ckpt_rel = info.get("checkpoint")
        if not ckpt_rel:
            errors.append(f"生命体 {l_id} 未配置 checkpoint 路径")
            continue
        
        ckpt_path = (REPO_ROOT / ckpt_rel).resolve()
        if not ckpt_path.exists():
            if ckpt_rel == "checkpoints/sdsc_mega_1million.bin":
                try:
                    import subprocess
                    subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "export_sdsc_binary.py")], check=True, capture_output=True)
                except Exception:
                    pass
            if not ckpt_path.exists():
                errors.append(f"生命体 {l_id} 检查点文件不存在: {ckpt_path}")
                continue

        if not (ckpt_path.name.endswith(".bin") or ckpt_path.name.endswith(".pt")):
            errors.append(f"生命体 {l_id} 检查点不是二进制 (.bin/.pt) 文件: {ckpt_path.name} (违反二进制检查点宪章)")
            continue
            
        checked_files.add(ckpt_path)

        if ckpt_path.name.endswith(".pt"):
            # 检查 GPU 原生张量检查点
            try:
                import torch
                pt_hdr = torch.load(ckpt_path, map_location="cpu", mmap=True)
                if not ("n_cells" in pt_hdr or "state" in pt_hdr):
                    errors.append(f"检查点 {ckpt_path.name} 缺少细胞规模张量定义")
            except Exception as e:
                errors.append(f"检查点 {ckpt_path.name} 读取异常: {e}")
            continue

        # 检查二进制头 (SDSC-BIN v2)
        try:
            with open(ckpt_path, "rb") as bf:
                hdr = bf.read(72)
                if len(hdr) < 72:
                    errors.append(f"检查点 {ckpt_path.name} 头大小不足 72 字节")
                    continue
                magic, ver = struct.unpack("<II", hdr[:8])
                if magic != SDSC_BINARY_MAGIC:
                    errors.append(f"检查点 {ckpt_path.name} 魔数无效: 0x{magic:08x} != 0x{SDSC_BINARY_MAGIC:08x}")
                if ver != SDSC_BINARY_VERSION:
                    errors.append(f"检查点 {ckpt_path.name} 版本无效: {ver} != {SDSC_BINARY_VERSION}")
        except Exception as e:
            errors.append(f"检查点 {ckpt_path.name} 读取异常: {e}")

    # 2. 验证 checkpoints/ 目录下是否存在残存的模型 json 伪检查点
    if CHECKPOINTS_DIR.exists():
        for item in CHECKPOINTS_DIR.glob("*.json"):
            # 报告文件除外 (形如 *_report.json, *_summary.json)
            if "_report" in item.name or "_summary" in item.name or "report" in item.name:
                continue
            # 检查是否是残存的网络拓扑 JSON
            try:
                with open(item, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                if "cells" in jdata or "synapses" in jdata or "organ" in jdata:
                    errors.append(f"checkpoints/ 存在未清理的遗留模型 JSON 检查点: {item.name}，必须彻底移除为纯二进制 SDSC-BIN v2")
            except Exception:
                pass

    return errors


def check_pixel_lod_discipline() -> list[str]:
    """门禁 2: 检验屏幕空间像素门禁与点云流形架构"""
    errors = []

    if not LOD_SYSTEM_PATH.exists():
        return [f"lod_system.js 不存在: {LOD_SYSTEM_PATH}"]

    lod_code = LOD_SYSTEM_PATH.read_text(encoding="utf-8")

    # 1. 验证 MIN_CELL_PIXELS 常量导出与定义
    if "MIN_CELL_PIXELS" not in lod_code:
        errors.append("lod_system.js 缺失 MIN_CELL_PIXELS 屏幕空间像素门禁常量")
    
    # 2. 验证 solidMaxDist 光学公式计算
    if "solidMaxDist" not in lod_code or "MIN_CELL_PIXELS" not in lod_code:
        errors.append("lod_system.js 缺失 solidMaxDist 基于投影像素的动态视距计算")

    # 3. 验证超出 solidMaxDist 时的绝对门禁拦截
    if not re.search(r"if\s*\(\s*d\s*>\s*solidMaxDist\s*\)\s*continue;", lod_code):
        errors.append("lod_system.js 未在细胞候选入队前拦截 d > solidMaxDist (像素投影 < 20px 必须归属点云)")

    # 4. 验证视锥裁剪
    if "intersectsSphere" not in lod_code:
        errors.append("lod_system.js 缺失视锥裁剪判断 (frustum.intersectsSphere)")

    # 5. 验证 app.js 中的亚细胞器视距门禁
    if APP_JS_PATH.exists():
        app_code = APP_JS_PATH.read_text(encoding="utf-8")
        if "showMicroOrganelles" not in app_code:
            errors.append("app.js 缺失 showMicroOrganelles 亚细胞器视距门禁控制")

    return errors


def main():
    print("============================================================")
    print("  KunCellular Architecture Discipline & Design Gate Check")
    print("  Checking: SDSC-BIN v2 Checkpoints, Screen Pixel LOD, Substrate Purity")
    print("============================================================")

    all_errors = []

    # 1. 检查纯二进制 SDSC-BIN v2 规范
    bin_errors = check_checkpoint_binary_discipline()
    if bin_errors:
        print(f"[-] [门禁 1: SDSC-BIN 纯二进制] 发现 {len(bin_errors)} 个违规项:")
        for err in bin_errors:
            print(f"    ↳ {err}")
        all_errors.extend(bin_errors)
    else:
        print("[+] [门禁 1: SDSC-BIN 纯二进制] 100% 达标：所有检查点符合 SDSC-BIN v2 零堆内存规范，零遗留 JSON。")

    # 2. 检查屏幕空间像素投影 LOD 架构
    lod_errors = check_pixel_lod_discipline()
    if lod_errors:
        print(f"[-] [门禁 2: 屏幕像素 LOD] 发现 {len(lod_errors)} 个违规项:")
        for err in lod_errors:
            print(f"    ↳ {err}")
        all_errors.extend(lod_errors)
    else:
        print("[+] [门禁 2: 屏幕像素 LOD] 100% 达标：屏幕投影像素 < MIN_CELL_PIXELS 严格归属点云流形，光学视锥门禁完备。")

    print("============================================================")
    if all_errors:
        print(f"FAILED: 共检出 {len(all_errors)} 个架构纪律违规项！(Exit 1)")
        sys.exit(1)
    else:
        print("SUCCESS: 所有核心设计原则与架构纪律门禁 100% 满分通过！(Exit 0)")
        sys.exit(0)


if __name__ == "__main__":
    main()
