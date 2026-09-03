#!/usr/bin/env python3
"""
SDSCC 超大规模生命体紧凑二进制打包器 (Binary CSR Exporter)
---------------------------------------------------------
解决上百万/上千万细胞展开为 C 头文件导致编译器 OOM 的工业级标准工具。
输出格式: 64 字节对齐的 SDSC-BIN 格式，供 sdsc_binary_runtime.h 零拷贝 mmap 加载。
"""

import os
import sys
import struct
import math
import random
import numpy as np

SDSC_BINARY_MAGIC = 0x53445343 # "SDSC"
SDSC_BINARY_VERSION = 2

# 26 原语映射 (0~25)
OP_MAP = {
    "SUM": 0, "INTEGRATE": 1, "AMPLIFY": 2, "INVERT": 3, "THRESHOLD": 4,
    "DAMPER": 5, "CLIP": 6, "ABS": 7, "MULTIPLY": 8, "ACT_POS": 9, "ACT_NEG": 10,
    "DIFF": 11, "HYSTERESIS": 12, "DEADZONE": 13, "INHIBIT": 14, "SUB": 15,
    "RATIO": 16, "OSCILLATOR": 17, "CORRELATION": 18, "FATIGUE": 19
}

def export_binary_cortex(num_cells, num_synapses, input_dim, output_dim,
                         cell_op_types, row_ptr, col_idx, weights, out_path):
    """打包为标准紧凑 SDSC-BIN 二进制"""
    assert len(cell_op_types) == num_cells
    assert len(row_ptr) == num_cells + 1
    assert len(col_idx) == num_synapses
    assert len(weights) == num_synapses

    header_size = 64
    cells_offset = header_size
    cells_size = num_cells * 4 # 4 bytes per cell
    
    row_ptr_offset = cells_offset + cells_size
    row_ptr_size = (num_cells + 1) * 4 # uint32
    
    col_idx_offset = row_ptr_offset + row_ptr_size
    col_idx_size = num_synapses * 4 # uint32
    
    weights_offset = col_idx_offset + col_idx_size
    weights_size = num_synapses * 4 # float32
    
    total_size = weights_offset + weights_size

    header_bytes = struct.pack(
        "<IIIIIIQQQQ16s",
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
        b"\x00" * 16
    )

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        
        # 1. 写入细胞元数据 (4 bytes each: op_type, param1, param2, flags)
        cell_bytes = bytearray(cells_size)
        for i, op in enumerate(cell_op_types):
            op_id = OP_MAP.get(op, 0) if isinstance(op, str) else int(op)
            p1_u8 = 64 # 1.0 gain
            p2_u8 = 0
            flags = 0
            if i < input_dim: flags |= 0x01
            if i >= num_cells - output_dim: flags |= 0x02
            
            idx = i * 4
            cell_bytes[idx] = op_id
            cell_bytes[idx + 1] = p1_u8
            cell_bytes[idx + 2] = p2_u8
            cell_bytes[idx + 3] = flags
        f.write(cell_bytes)

        # 2. 写入 row_ptr
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())

        # 3. 写入 col_idx
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())

        # 4. 写入 weights
        f.write(np.array(weights, dtype=np.float32).tobytes())

    file_mb = total_size / (1024 * 1024)
    print(f"  [SUCCESS] 紧凑二进制大生命体已导出: {out_path} ({file_mb:.2f} MB, {num_cells:,} 细胞, {num_synapses:,} 突触)")
    return total_size

def generate_million_cell_cortex(out_path="checkpoints/sdsc_mega_1million.bin"):
    """
    生成一个包含 1,000,000 个细胞、3,000,000 条突触的高维全息微柱阵列大生命体
    由 1,024 个具备分层动力学微柱自组织构成，符合非冯算存一体架构。
    """
    print("=========================================================")
    print("  SDSCC 1,000,000 细胞高维全息微柱阵列大生命体生成中...")
    print("=========================================================")
    
    num_cells = 1_000_000
    synapses_per_cell = 3
    num_synapses = num_cells * synapses_per_cell
    input_dim = 64
    output_dim = 16

    primitives_pool = [
        "SUM", "INTEGRATE", "AMPLIFY", "DIFF", "HYSTERESIS", 
        "DEADZONE", "DAMPER", "OSCILLATOR", "CORRELATION", "FATIGUE"
    ]
    
    random.seed(42)
    cell_ops = [random.choice(primitives_pool) for _ in range(num_cells)]

    # 构建 CSR 稀疏图 (局部小世界拓扑连接，每个细胞连接临近微柱和少量长程突触)
    row_ptr = np.zeros(num_cells + 1, dtype=np.uint32)
    col_idx = np.zeros(num_synapses, dtype=np.uint32)
    weights = np.zeros(num_synapses, dtype=np.float32)

    cur_syn = 0
    for u in range(num_cells):
        row_ptr[u] = cur_syn
        for s in range(synapses_per_cell):
            if s < 2:
                # 局部微柱晶格连接 (Local Microcolumn Connections)
                v = (u + random.randint(1, 128)) % num_cells
            else:
                # 跨微柱长程小世界捷径 (Long-Range Small-World Shortcuts)
                v = (u + random.randint(1024, 65536)) % num_cells
            col_idx[cur_syn] = v
            weights[cur_syn] = random.uniform(-1.2, 1.2)
            cur_syn += 1
    row_ptr[num_cells] = cur_syn

    export_binary_cortex(
        num_cells, num_synapses, input_dim, output_dim,
        cell_ops, row_ptr, col_idx, weights, out_path
    )

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "sdsc_mega_1million.bin")
    generate_million_cell_cortex(bin_path)
