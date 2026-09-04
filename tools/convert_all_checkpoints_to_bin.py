#!/usr/bin/env python3
"""
tools/convert_all_checkpoints_to_bin.py
--------------------------------------
SDSCC 检查点全量二进制化转换与校验工具
将所有现存或训练产出的 JSON 检查点统一编译为符合 C11 硬件规范的 SDSC-BIN (v2) 紧凑二进制格式。
格式完全兼容 include/kun/cellular/sdsc_binary_runtime.h。
"""

import os
import sys
import json
import struct
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))

SDSC_BINARY_MAGIC = 0x53445343 # "SDSC"
SDSC_BINARY_VERSION = 2

# 映射到 26 原语 (sdsc_primitives.h: 0~25)
OP_ENUM_MAP = {
    "SENSE_0": 0, "SENSE_1": 1, "SENSE_2": 2, "SENSE_3": 3,
    "SENSE0": 0, "SENSE1": 1, "SENSE2": 2, "SENSE3": 3,
    "SENSE_INPUT0": 0, "SENSE_INPUT1": 1, "SENSE_INPUT2": 2, "SENSE_INPUT3": 3,
    "SENSE_VOXEL": 0, "SENSE_OCC": 1, "SENSE_SENSOR": 2,
    "SUM": 4, "OP_SUM": 4,
    "INTEGRATE": 5, "OP_INTEGRAL": 5,
    "AMPLIFY": 6,
    "INVERT": 7,
    "DAMPER": 8, "EMA": 8, "OP_EMA": 8, "OP_DAMPING": 8,
    "CLIP": 9,
    "ABS": 10,
    "MULTIPLY": 11, "OP_TENSORFIELD": 11,
    "DIFF": 12, "OP_DIFF": 12,
    "SUB": 13, "OP_SUB": 13,
    "RATIO": 14,
    "THRESHOLD": 15,
    "HYSTERESIS": 16, "HYST": 16, "GATE_HYSTERESIS": 16,
    "DEADZONE": 17, "GATE_DEADZONE": 17,
    "INHIBIT": 18,
    "AND": 19, "GATE_AND": 19,
    "MIN_MAX": 20, "GATE_MIN_MAX": 20,
    "ACT_POS": 21, "ACT_POSACTION": 21, "ACT_EFFECTOR": 21,
    "ACT_NEG": 22, "ACT_NEGACTION": 22,
    "ACT_RESET": 23, "ACT_IMMUNELOCK": 23, "ACT_LOCK": 23, "ACT_COUNTERFACTUAL": 23, "ACT_FLOW": 23,
    "CORRELATION": 24, "OP_CORRELATION": 24,
    "FATIGUE": 25, "OP_FATIGUE": 25, "OSCILLATOR": 25, "OP_WAVE": 25
}

def resolve_op_type(t_name):
    if isinstance(t_name, int):
        return t_name % 26
    norm = str(t_name).upper().strip()
    if norm in OP_ENUM_MAP:
        return OP_ENUM_MAP[norm]
    for k, v in OP_ENUM_MAP.items():
        if k in norm:
            return v
    return 4 # Default to SUM

def pack_organism_to_bin(cells, synapses, out_bin_path, generation=40, organism_id=""):
    """
    将细胞和突触打包成标准 SDSC-BIN 格式。
    cells: list of dict with {id, type, x, y, z, param1, ...}
    synapses: list of dict with {from, to, weight, active}
    """
    num_cells = len(cells)
    
    # 建立细胞 ID 连续索引映射
    cell_id_to_idx = {c["id"]: i for i, c in enumerate(cells)}
    
    # 收集突触并按 from_cell 排序构建 CSR
    adj = [[] for _ in range(num_cells)]
    for s in synapses:
        u = s.get("from")
        v = s.get("to")
        if u not in cell_id_to_idx or v not in cell_id_to_idx:
            continue
        u_idx = cell_id_to_idx[u]
        v_idx = cell_id_to_idx[v]
        w = float(s.get("weight", 1.0))
        act = s.get("active", True)
        if not act:
            w = 0.0
        adj[u_idx].append((v_idx, w))
    
    row_ptr = [0] * (num_cells + 1)
    col_idx = []
    weights = []
    
    curr_offset = 0
    for i in range(num_cells):
        row_ptr[i] = curr_offset
        for v_idx, w in adj[i]:
            col_idx.append(v_idx)
            weights.append(w)
            curr_offset += 1
    row_ptr[num_cells] = curr_offset
    num_synapses = len(col_idx)
    
    # 受体与效应器维度统计
    input_dim = 0
    output_dim = 0
    for i, c in enumerate(cells):
        op = resolve_op_type(c.get("type", "SUM"))
        if op <= 3 or "SENSE" in str(c.get("type", "")).upper() or "REC_" in str(c.get("type", "")).upper():
            input_dim += 1
        elif op in (21, 22, 23) or "ACT_" in str(c.get("type", "")).upper() or "MOTOR_" in str(c.get("type", "")).upper():
            output_dim += 1
            
    header_size = 72
    cells_offset = header_size
    cells_size = num_cells * 4 # 4 bytes each: op_type, param1, param2, flags
    
    row_ptr_offset = cells_offset + cells_size
    row_ptr_size = (num_cells + 1) * 4 # uint32
    
    col_idx_offset = row_ptr_offset + row_ptr_size
    col_idx_size = num_synapses * 4 # uint32
    
    weights_offset = col_idx_offset + col_idx_size
    weights_size = num_synapses * 4 # float32
    
    coords_offset = weights_offset + weights_size
    coords_size = num_cells * 3 * 4 # float32 x, y, z
    
    # 构造尾部元数据 (JSON)
    meta_dict = {
        "organism_id": organism_id,
        "generation": generation,
        "cells_meta": [
            {
                "id": c["id"],
                "type": str(c.get("type", "Op_EMA")),
                "layer": c.get("layer", "L2_ASSOCIATION"),
                "gain": float(c.get("param1", 1.0) or 1.0)
            }
            for c in cells
        ]
    }
    meta_bytes = json.dumps(meta_dict, ensure_ascii=False).encode("utf-8")
    meta_offset = coords_offset + coords_size
    meta_size = len(meta_bytes)
    
    # 72 bytes header: <IIIIIIQQQQQQ
    header_bytes = struct.pack(
        "<IIIIIIQQQQQQ",
        SDSC_BINARY_MAGIC,
        SDSC_BINARY_VERSION,
        num_cells,
        num_synapses,
        input_dim,
        output_dim,
        cells_offset,
        row_ptr_offset,
        col_idx_offset,
        weights_offset,
        coords_offset,
        (generation & 0xFFFFFFFF) | ((meta_size & 0xFFFFFFFF) << 32)
    )
    assert len(header_bytes) == 72
    
    # 写入文件
    with open(out_bin_path, "wb") as f:
        f.write(header_bytes)
        
        # 1. cells
        cell_bytes = bytearray(cells_size)
        for i, c in enumerate(cells):
            op_id = resolve_op_type(c.get("type", "SUM"))
            p1 = float(c.get("param1", 1.0) or 1.0)
            p1_u8 = min(255, max(0, int(p1 * 64.0)))
            p2_u8 = 0
            flags = 0
            if op_id <= 3: flags |= 0x01
            if op_id in (21, 22, 23): flags |= 0x02
            idx = i * 4
            cell_bytes[idx] = op_id
            cell_bytes[idx + 1] = p1_u8
            cell_bytes[idx + 2] = p2_u8
            cell_bytes[idx + 3] = flags
        f.write(cell_bytes)
        
        # 2. row_ptr
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())
        
        # 3. col_idx
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())
        
        # 4. weights
        f.write(np.array(weights, dtype=np.float32).tobytes())
        
        # 5. coords
        coords_arr = np.zeros((num_cells, 3), dtype=np.float32)
        for i, c in enumerate(cells):
            coords_arr[i, 0] = float(c.get("x", 0.0))
            coords_arr[i, 1] = float(c.get("y", 0.0))
            coords_arr[i, 2] = float(c.get("z", 0.0))
        f.write(coords_arr.tobytes())
        
        # 6. meta JSON
        f.write(meta_bytes)
        
    total_bytes = os.path.getsize(out_bin_path)
    return total_bytes

def verify_bin_checkpoint(bin_path):
    """验证二进制检查点文件完整性"""
    with open(bin_path, "rb") as f:
        hdr_bytes = f.read(72)
        assert len(hdr_bytes) == 72
        magic, version, nc, ns, in_dim, out_dim, c_off, rp_off, ci_off, w_off, coords_off, extra = struct.unpack("<IIIIIIQQQQQQ", hdr_bytes)
        assert magic == SDSC_BINARY_MAGIC, f"Bad magic: {hex(magic)}"
        assert version == SDSC_BINARY_VERSION, f"Bad version: {version}"
        
        f.seek(coords_off)
        coords_data = f.read(nc * 12)
        coords = np.frombuffer(coords_data, dtype=np.float32).reshape((nc, 3))
        assert not np.isnan(coords).any(), "NaN found in coords"
        
        f.seek(w_off)
        w_data = f.read(ns * 4)
        weights = np.frombuffer(w_data, dtype=np.float32)
        assert not np.isnan(weights).any(), "NaN found in weights"
        
    return nc, ns

def convert_all():
    print("================================================================")
    print("  SDSCC 检查点全量统一二进制化 (Unify Checkpoints to SDSC-BIN v2)")
    print("================================================================")
    from cellular_live_backend import SiliconCellularOrganism, load_business_lifeform_manifest
    
    manifest = load_business_lifeform_manifest()
    org_engine = SiliconCellularOrganism()
    
    ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
    converted_count = 0
    
    for lf in manifest:
        oid = lf.get("id")
        ckpt_rel = lf.get("checkpoint", "")
        if not ckpt_rel:
            continue
        
        # 统一输出路径为 .bin
        base_name = os.path.splitext(os.path.basename(ckpt_rel))[0]
        out_bin_path = os.path.join(ckpt_dir, base_name + ".bin")
        
        # 加载真实细胞拓扑
        res = org_engine.load_organism_by_id(oid)
        if isinstance(res, dict) and res.get("status") == "error":
            print(f"  [WARN] 跳过未就绪生命体: {oid}")
            continue
            
        cells = [
            {
                "id": c.id,
                "type": c.type,
                "x": c.x,
                "y": c.y,
                "z": c.z,
                "param1": c.gain,
                "layer": getattr(c, "layer", "L2_ASSOCIATION")
            }
            for c in org_engine.cells
        ]
        synapses = list(org_engine.synapses)
        
        total_sz = pack_organism_to_bin(cells, synapses, out_bin_path, generation=org_engine.generation, organism_id=oid)
        nc, ns = verify_bin_checkpoint(out_bin_path)
        print(f"  [BIN PASS] {oid:<28} -> {base_name}.bin ({total_sz:,} B, {nc:,} 细胞, {ns:,} 突触)")
        converted_count += 1
        
    print(f"\n[SUCCESS] 全部 {converted_count} 个生命体检查点已成功编译为标准 SDSC-BIN v2 二进制！")

if __name__ == "__main__":
    convert_all()
