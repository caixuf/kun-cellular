#!/usr/bin/env python3
"""
==============================================================================
KunCellular: 硅基天籁音乐与复调对位歌王演化训练器 (Music Composer Master Cortex)
==============================================================================
演化目标: 培育具备 1,024 物理细胞、196,608 因果突触的高阶音乐认知超脑。
微柱解剖结构:
  1. TonotopicHarmonicCortex   (256 细胞): 音高谐波共振与五度相空间几何映射
  2. TensionResolutionCortex    (256 细胞): 调性张力积聚与多巴胺和声解决 (ii-V-I, IV-I 终止式)
  3. RhythmGrooveCPGCortex      (256 细胞): 呼吸节律与摇摆切分中枢发生器 (CPG)
  4. MelodicMotifMemoryCortex   (256 细胞): 旋律动机长程记忆、倒影逆行与歌唱性连贯

输出规范:
  纯二进制 SDSC-BIN v2 (Magic: 0x53445343, Version: 2), 零堆内存、零 GC。
==============================================================================
"""

import os
import sys
import math
import struct
import random
import numpy as np

SDSC_BINARY_MAGIC = 0x53445343
SDSC_BINARY_VERSION = 2

# 26 原语映射
OP_SUM = 0
OP_INTEGRATE = 1
OP_AMPLIFY = 2
OP_INVERT = 3
OP_THRESHOLD = 4
OP_DAMPER = 5
OP_CLIP = 6
OP_ABS = 7
OP_MULTIPLY = 8
OP_ACT_POS = 9
OP_ACT_NEG = 10
OP_DIFF = 11
OP_GATE_HYSTERESIS = 12
OP_GATE_DEADZONE = 13
OP_GATE_INHIBIT = 14
OP_SUB = 15
OP_RATIO = 16
OP_OSCILLATOR = 17
OP_CORRELATION = 18
OP_FATIGUE = 19
OP_EMA = 20
OP_RESONANCE = 21

def build_music_composer_topology():
    """构建 1,024 细胞协同微柱认知超脑"""
    num_cells = 1024
    synapses_per_cell = 192
    num_synapses = num_cells * synapses_per_cell # 196,608
    input_dim = 32
    output_dim = 16

    op_types = np.zeros(num_cells, dtype=np.uint8)
    
    # 1. 感受野微柱 (0~255): 频域与谐波共振 (OP_OSCILLATOR, OP_CORRELATION, OP_DIFF, OP_EMA)
    for i in range(0, 256):
        r = i % 4
        if r == 0:
            op_types[i] = OP_OSCILLATOR
        elif r == 1:
            op_types[i] = OP_CORRELATION
        elif r == 2:
            op_types[i] = OP_EMA
        else:
            op_types[i] = OP_DIFF

    # 2. 调性张力与和声解决柱 (256~511): 舒曼和声能量与迟滞阻尼 (OP_INTEGRATE, OP_GATE_HYSTERESIS, OP_DAMPER, OP_SUM)
    for i in range(256, 512):
        r = i % 4
        if r == 0:
            op_types[i] = OP_INTEGRATE
        elif r == 1:
            op_types[i] = OP_GATE_HYSTERESIS
        elif r == 2:
            op_types[i] = OP_DAMPER
        else:
            op_types[i] = OP_SUM

    # 3. 节律中枢发生器 CPG (512~767): 律动脉冲与互抑制 (OP_OSCILLATOR, OP_GATE_INHIBIT, OP_FATIGUE, OP_THRESHOLD)
    for i in range(512, 768):
        r = i % 4
        if r == 0:
            op_types[i] = OP_OSCILLATOR
        elif r == 1:
            op_types[i] = OP_GATE_INHIBIT
        elif r == 2:
            op_types[i] = OP_FATIGUE
        else:
            op_types[i] = OP_THRESHOLD

    # 4. 动机叙事与记忆柱 (768~1023): 长程记忆与决策效应 (OP_EMA, OP_ACT_POS, OP_CLIP, OP_RESONANCE)
    for i in range(768, 1024):
        r = i % 4
        if r == 0:
            op_types[i] = OP_EMA
        elif r == 1:
            op_types[i] = OP_ACT_POS
        elif r == 2:
            op_types[i] = OP_CLIP
        else:
            op_types[i] = OP_RESONANCE

    # CSR 稀疏图突触连接
    row_ptr = np.zeros(num_cells + 1, dtype=np.uint32)
    col_idx = np.zeros(num_synapses, dtype=np.uint32)
    weights = np.zeros(num_synapses, dtype=np.float32)

    rng = np.random.RandomState(42)
    offset = 0
    for c in range(num_cells):
        row_ptr[c] = offset
        # 柱内密集连接 (60%) + 跨柱联络与前额抑制 (40%)
        col_list = set()
        c_col_base = (c // 256) * 256
        
        while len(col_list) < synapses_per_cell:
            if rng.rand() < 0.65:
                target = c_col_base + rng.randint(0, 256)
            else:
                target = rng.randint(0, num_cells)
            if target != c:
                col_list.add(target)

        targets = sorted(list(col_list))
        for t in targets:
            col_idx[offset] = t
            # 权重初始化遵循李雅普诺夫稳态约束 (谱半径 < 1.0)
            weights[offset] = rng.normal(0.0, 0.05)
            offset += 1

    row_ptr[num_cells] = offset
    return op_types, row_ptr, col_idx, weights

# 欧拉愉悦度与和声评分动力学 (Euler Gradus Suavitatis & Consonance Matrix)
PENTATONIC_SCALE = [0, 2, 4, 7, 9] # 大调五声音阶 (宫商角徵羽，天然极度愉快治愈)
MAJOR_DIATONIC = [0, 2, 4, 5, 7, 9, 11]

def evaluate_music_step(state, inputs, weights, op_types, row_ptr, col_idx):
    """纯张量前向步模拟"""
    num_cells = len(state)
    next_state = np.zeros_like(state)
    
    # 突触传导
    for c in range(num_cells):
        s_start = row_ptr[c]
        s_end = row_ptr[c + 1]
        syn_sum = np.dot(weights[s_start:s_end], state[col_idx[s_start:s_end]])
        
        # 加上受体输入投影
        if c < 32:
            syn_sum += inputs[c] * 0.4
            
        op = op_types[c]
        if op == OP_OSCILLATOR:
            next_state[c] = math.sin(syn_sum + state[c] * 0.8)
        elif op == OP_EMA:
            next_state[c] = 0.85 * state[c] + 0.15 * math.tanh(syn_sum)
        elif op == OP_INTEGRATE:
            next_state[c] = np.clip(state[c] + 0.1 * syn_sum, -2.0, 2.0)
        elif op == OP_GATE_HYSTERESIS:
            val = math.tanh(syn_sum)
            next_state[c] = val if abs(val) > 0.1 else state[c] * 0.95
        elif op == OP_DAMPER:
            next_state[c] = state[c] * 0.9 + syn_sum * 0.1
        elif op == OP_FATIGUE:
            next_state[c] = math.tanh(syn_sum) * (1.0 - 0.2 * abs(state[c]))
        elif op == OP_GATE_INHIBIT:
            next_state[c] = math.tanh(syn_sum) if syn_sum > 0 else 0.0
        else:
            next_state[c] = math.tanh(syn_sum)
            
    return next_state

def evolve_music_composer_master(generations=50, seed=2026):
    print("==================================================================")
    print("  KunCellular: 硅基天籁音乐与复调对位歌王超脑演化训练器")
    print(f"  目标规格: 1,024 物理细胞 | 196,608 突触 | 4大认知功能柱 | 演化代数: {generations}")
    print("==================================================================")

    op_types, row_ptr, col_idx, weights = build_music_composer_topology()
    rng = np.random.RandomState(seed)

    best_fitness = -1e9
    
    for gen in range(generations):
        # 模拟 16 个典型调性与和弦进程场景 (C大调阳光进行、ii-V-I爵士解决、卡农巴赫复调、五度相生)
        total_pleasure_score = 0.0
        total_cadence_score = 0.0
        total_singability_score = 0.0
        lyapunov_energy = 0.0
        
        state = np.zeros(1024, dtype=np.float32)
        
        for trial in range(16):
            # 构造音乐输入激励
            inputs = np.zeros(32, dtype=np.float32)
            root_pitch = (trial * 7) % 12 # 五度圈回转
            inputs[0] = math.sin(root_pitch * math.pi / 6.0)
            inputs[1] = math.cos(root_pitch * math.pi / 6.0)
            inputs[2] = 0.8 # 节奏脉冲
            inputs[3] = (trial % 4) * 0.25 # 和声进行步
            
            for step in range(8):
                state = evaluate_music_step(state, inputs, weights, op_types, row_ptr, col_idx)
                
                # 效应器解码 (取第 1008~1023 细胞作为动作输出)
                effectors = state[1008:1024]
                melody_pitch = (int((effectors[0] + 1.0) * 6.0) + root_pitch) % 12
                chord_tension = abs(effectors[1])
                groove_pulse = effectors[2]
                
                # 1. 和声与音高愉悦度 (五声与自然大调高分，三全音不协和扣分)
                if melody_pitch in PENTATONIC_SCALE:
                    total_pleasure_score += 1.5
                elif melody_pitch in MAJOR_DIATONIC:
                    total_pleasure_score += 1.0
                else:
                    total_pleasure_score -= 0.5
                    
                # 2. 终止式解决 (最后一步张力平滑解决到主和弦)
                if step == 7:
                    if chord_tension < 0.2 and melody_pitch in [0, 4, 7]: # 解决到大三和弦主音
                        total_cadence_score += 3.0
                    else:
                        total_cadence_score += 0.5
                        
                # 3. 歌唱性连贯 (步进平滑，大跳抑制)
                step_diff = abs(effectors[0] - effectors[3])
                if step_diff < 0.35:
                    total_singability_score += 0.5
                    
                lyapunov_energy += np.sum(state ** 2) / 1024.0

        fitness = (total_pleasure_score * 2.0 + total_cadence_score * 3.0 + total_singability_score * 1.5) - (lyapunov_energy * 0.01)
        
        # 变异突变突触权重 (强化高愉悦度通路)
        if fitness > best_fitness:
            best_fitness = fitness
            print(f"  ↳ [Gen {gen+1:02d}/{generations:02d}] 新冠军诞生! 适应度: {fitness:.2f} | 愉悦和声: {total_pleasure_score:.1f} | 终结解决: {total_cadence_score:.1f} | 能量稳态: {lyapunov_energy/128:.3f}")
            # 保存最佳权重
            best_weights = weights.copy()
        
        # 演化选择与微扰
        mutation_rate = 0.08 * (1.0 - gen / generations)
        weights = best_weights + rng.normal(0.0, mutation_rate, size=len(weights)).astype(np.float32)

    # 导出为标准 SDSC-BIN v2 纯二进制检查点
    out_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "music_composer_cortex.bin")

    num_cells = 1024
    num_synapses = 196608
    input_dim = 32
    output_dim = 16

    header_size = 72
    cells_offset = header_size
    cells_size = num_cells * 4
    row_ptr_offset = cells_offset + cells_size
    row_ptr_size = (num_cells + 1) * 4
    col_idx_offset = row_ptr_offset + row_ptr_size
    col_idx_size = num_synapses * 4
    weights_offset = col_idx_offset + col_idx_size
    weights_size = num_synapses * 4
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
        for i in range(num_cells):
            op_id = int(op_types[i])
            flags = 0
            if i < input_dim:
                flags |= 0x01
            if i >= num_cells - output_dim:
                flags |= 0x02
            idx = i * 4
            cell_bytes[idx] = op_id
            cell_bytes[idx + 1] = 64  # 1.0 gain
            cell_bytes[idx + 2] = 0
            cell_bytes[idx + 3] = flags
        f.write(cell_bytes)

        # 2. 写入 row_ptr (uint32)
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())

        # 3. 写入 col_idx (uint32)
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())

        # 4. 写入 weights (float32)
        f.write(np.array(best_weights, dtype=np.float32).tobytes())

    file_size_kb = os.path.getsize(out_path) / 1024.0
    print("==================================================================")
    print(f"  [SUCCESS] 硅基天籁音乐歌王超脑检查点已成功导出!")
    print(f"  路径: {out_path} ({file_size_kb:.1f} KB, 预期: {total_size / 1024:.1f} KB)")
    print(f"  规范: SDSC-BIN v2 纯二进制 (Magic 0x53445343, 0 malloc / 0 GC)")
    print("==================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KunCellular 硅基音乐歌王超脑演化训练器")
    parser.add_argument("--generations", type=int, default=50, help="演化代数 (默认: 50)")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子 (默认: 2026)")
    args = parser.parse_args()
    evolve_music_composer_master(generations=args.generations, seed=args.seed)
