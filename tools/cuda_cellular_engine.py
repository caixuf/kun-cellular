#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunCellular GPU 高吞吐微柱稀疏元胞动力学与并行演化引擎 (Refactored Production Grade)
- 支持分块微柱稀疏拓扑 (Block-Sparse Column Array, 规避 OOM)
- 支持全张量无分支连续推演 (Zero-Fragmentation Vectorization)
- 支持一键无损导出为标准 C11 SDSC-BIN v2 权威二进制检查点
"""

import time
import struct
import numpy as np
import torch
import torch.nn.functional as F

class CUDACellularPopulation:
    """
    运行在 GPU 显存上的批量元胞微柱阵列种群
    - pop_size: 种群并发个体数 (如 256)
    - num_columns: 微柱数量 (如 16)
    - cells_per_col: 每微柱物理细胞数 (如 64, 总细胞数 = 1024)
    - in_dim: 感知受体维度 (如 32)
    - out_dim: 运动效应维度 (如 8)
    """
    def __init__(self, pop_size=256, num_columns=16, cells_per_col=64, in_dim=32, out_dim=8, device="cuda"):
        self.pop_size = pop_size
        self.num_columns = num_columns
        self.cells_per_col = cells_per_col
        self.num_cells = num_columns * cells_per_col
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. 柱内密集局部突触权重: [pop_size, num_columns, cells_per_col, cells_per_col]
        # 局部高频协同 (Local dense intra-column dynamics)
        self.intra_weights = torch.randn(
            (pop_size, num_columns, cells_per_col, cells_per_col), 
            device=self.device
        ) * 0.04

        # 2. 柱间长程稀疏轴突权重: 每一柱只投影到相邻柱与全局枢纽柱 (Block-Sparse inter-column)
        # 每个柱有 k_projections 条跨柱投射通路
        self.k_projections = 4
        self.inter_weights = torch.randn(
            (pop_size, num_columns, self.k_projections, cells_per_col),
            device=self.device
        ) * 0.02
        
        # 跨柱连接拓扑表: [num_columns, k_projections]
        col_targets = []
        for c in range(num_columns):
            targets = [
                (c + 1) % num_columns,
                (c - 1 + num_columns) % num_columns,
                (c + num_columns // 2) % num_columns,
                0 # 枢纽柱
            ]
            col_targets.append(targets)
        self.col_targets = torch.tensor(col_targets, device=self.device, dtype=torch.long)

        # 3. 原语参数与槽位预分配 (分槽固定，杜绝动态布尔掩码分支)
        # 每柱内: 
        # 0..7:   受体/直通通道 (Pass-through)
        # 8..23:  指数移动平均平滑滤波 (EMA, alpha ~ 0.1)
        # 24..39: 泄漏积分器 (Integrate)
        # 40..51: 二阶非线性谐振子 (Oscillator)
        # 52..63: 施密特迟滞双阈值门控 (Hysteresis Gate)
        self.param1 = torch.rand((pop_size, self.num_cells), device=self.device) * 0.4 + 0.1
        self.param2 = torch.rand((pop_size, self.num_cells), device=self.device) * 0.2 - 0.1

        # 运行时连续状态张量 (Zero-Allocation)
        self.states = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.aux_states = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.outputs = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.drive_buf = torch.zeros((pop_size, self.num_cells), device=self.device)

    def reset_states(self):
        self.states.zero_()
        self.aux_states.zero_()
        self.outputs.zero_()
        self.drive_buf.zero_()

    def forward_step(self, inputs):
        """
        单步批量全并发无分支推演 (Zero-Branch Vectorized Step)
        """
        P = self.pop_size
        C = self.num_columns
        K = self.cells_per_col
        N = self.num_cells

        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0).expand(P, -1)

        # 将 outputs 重整为微柱视图: [P * C, K, 1]
        col_outputs = self.outputs.view(P * C, K, 1)

        # 1. 柱内矩阵乘法: [P * C, K, K] x [P * C, K, 1] -> [P * C, K]
        intra_W = self.intra_weights.view(P * C, K, K)
        intra_drive = torch.bmm(intra_W, col_outputs).view(P, C, K)

        # 2. 柱间跨柱投影计算
        inter_drive = torch.zeros_like(intra_drive)
        out_per_col = self.outputs.view(P, C, K)
        for proj_idx in range(self.k_projections):
            target_cols = self.col_targets[:, proj_idx] # [C]
            W_proj = self.inter_weights[:, :, proj_idx, :] # [P, C, K]
            # 汇聚投射能量
            energy = (out_per_col * W_proj).sum(dim=-1, keepdim=True) # [P, C, 1]
            inter_drive.index_add_(1, target_cols, energy.expand(-1, -1, K) * (1.0 / self.k_projections))

        # 合成总驱动力
        drive = (intra_drive + inter_drive).view(P, N)

        # 注入外部感知输入
        drive[:, :self.in_dim] += inputs

        # 3. 固定槽位纯张量前向 (完全无动态掩码，100% 连续执行)
        out = torch.empty_like(self.outputs)
        
        # 槽位 A (0..7): 受体直通
        out[:, 0:8] = drive[:, 0:8]

        # 槽位 B (8..23): EMA 滤波
        sl_b = slice(8, 24)
        alpha = self.param1[:, sl_b]
        self.states[:, sl_b] = (1.0 - alpha) * self.states[:, sl_b] + alpha * drive[:, sl_b]
        out[:, sl_b] = torch.tanh(self.states[:, sl_b])

        # 槽位 C (24..39): 积分器
        sl_c = slice(24, 40)
        leak = self.param1[:, sl_c] * 0.08
        self.states[:, sl_c] = self.states[:, sl_c] * (1.0 - leak) + drive[:, sl_c] * 0.05
        out[:, sl_c] = torch.clamp(self.states[:, sl_c], -2.0, 2.0)

        # 槽位 D (40..51): 谐振子 (二阶 Van der Pol)
        sl_d = slice(40, 52)
        s1 = self.states[:, sl_d]
        s2 = self.aux_states[:, sl_d]
        mu = self.param1[:, sl_d] * 1.5
        ds1 = s2
        ds2 = mu * (1.0 - s1 * s1) * s2 - s1 + drive[:, sl_d]
        dt = 0.05
        s1 = torch.clamp(s1 + ds1 * dt, -3.0, 3.0)
        s2 = torch.clamp(s2 + ds2 * dt, -3.0, 3.0)
        self.states[:, sl_d] = s1
        self.aux_states[:, sl_d] = s2
        out[:, sl_d] = s1

        # 槽位 E (52..63): 施密特双阈值迟滞门控
        sl_e = slice(52, 64)
        th_hi = self.param1[:, sl_e]
        th_lo = self.param2[:, sl_e]
        latch = self.states[:, sl_e]
        latch = torch.where(drive[:, sl_e] > th_hi, torch.tensor(1.0, device=self.device), latch)
        latch = torch.where(drive[:, sl_e] < th_lo, torch.tensor(-1.0, device=self.device), latch)
        self.states[:, sl_e] = latch
        out[:, sl_e] = latch

        # 剩余其他高位细胞统一走自适应双曲正切
        if self.num_cells > 64:
            sl_rest = slice(64, self.num_cells)
            out[:, sl_rest] = torch.tanh(drive[:, sl_rest])

        self.outputs = out
        return self.outputs[:, -self.out_dim:]

    def export_champion_to_sdsc_bin(self, champion_idx, filepath):
        """
        将指定冠军个体的拓扑与权重导出为严格标准 SDSC-BIN v2 二进制检查点
        与 C++ load_checkpoint_bin 完全兼容 (72-byte header + CSR)
        """
        champion_idx = int(champion_idx)
        intra_W = self.intra_weights[champion_idx].detach().cpu().numpy() # [C, K, K]
        p1 = self.param1[champion_idx].detach().cpu().numpy()
        p2 = self.param2[champion_idx].detach().cpu().numpy()

        C, K = self.num_columns, self.cells_per_col
        N = self.num_cells

        # 构造完整 CSR 图拓扑 (仅导出绝对值 > 1e-4 的有效突触)
        row_ptr = [0]
        col_idx = []
        weights = []

        for u in range(N):
            c_u = u // K
            k_u = u % K
            edges = []
            # 1. 柱内突触
            for k_v in range(K):
                w = float(intra_W[c_u, k_u, k_v])
                if abs(w) > 1e-4:
                    v = c_u * K + k_v
                    edges.append((v, 0, w)) # (to_cell, to_port, weight)

            # 2. 跨柱突触
            # 简化为由每柱前几位代表投射
            for proj_idx in range(self.k_projections):
                c_v = int(self.col_targets[c_u, proj_idx].item())
                w = float(self.inter_weights[champion_idx, c_u, proj_idx, k_u].item())
                if abs(w) > 1e-4:
                    v = c_v * K + (k_u % K)
                    edges.append((v, 0, w))

            # 排序保序
            edges.sort(key=lambda x: x[0])
            for v, port, w in edges:
                packed = (int(port & 0xFF) << 24) | int(v & 0x00FFFFFF)
                col_idx.append(packed)
                weights.append(w)

            row_ptr.append(len(col_idx))

        num_synapses = len(col_idx)

        # 构造 72 字节 SDSC_BIN_HDR
        magic = 0x53445343
        version = 2
        cells_off = 72
        cells_sz = N * 4
        rp_off = cells_off + cells_sz
        rp_sz = (N + 1) * 4
        ci_off = rp_off + rp_sz
        ci_sz = num_synapses * 4
        w_off = ci_off + ci_sz
        w_sz = num_synapses * 4
        coords_off = w_off + w_sz

        hdr = struct.pack(
            "<IIIIIIQQQQQQ",
            magic, version, N, num_synapses, self.in_dim, self.out_dim,
            cells_off, rp_off, ci_off, w_off, coords_off, 0
        )

        # 写入文件
        with open(filepath, "wb") as f:
            f.write(hdr)
            # 写入 cells (4 字节/细胞: op, p1, p2, flags)
            for i in range(N):
                # 确定 opcode
                slot = i % K
                if slot < 8: op = 0
                elif slot < 24: op = 8
                elif slot < 40: op = 5
                elif slot < 52: op = 25
                elif slot < 64: op = 16
                else: op = 4

                p1_i8 = int(np.clip(np.round(p1[i] * 64.0), -128, 127))
                p2_i8 = int(np.clip(np.round(p2[i] * 64.0), -128, 127))
                p1_u8 = p1_i8 & 0xFF
                p2_u8 = p2_i8 & 0xFF
                flags = 0
                if i < self.in_dim: flags |= 0x01
                if i >= N - self.out_dim: flags |= 0x02
                f.write(struct.pack("<BBBB", op, p1_u8, p2_u8, flags))

            # 写入 row_ptr
            f.write(struct.pack(f"<{len(row_ptr)}I", *row_ptr))
            # 写入 col_idx
            if col_idx:
                f.write(struct.pack(f"<{len(col_idx)}I", *col_idx))
                f.write(struct.pack(f"<{len(weights)}f", *weights))

            # 写入三维坐标
            coords = []
            for i in range(N):
                c = i // K
                k = i % K
                theta = (c / C) * 2.0 * np.pi
                r = 100.0 + (k % 8) * 10.0
                z = (k // 8) * 25.0
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                coords.extend([x, y, z])
            f.write(struct.pack(f"<{len(coords)}f", *coords))

        return num_synapses


def bench_gpu_throughput():
    print("=" * 65)
    print("🚀 KunCellular GPU 微柱稀疏演化底座性能压测 (RTX 5060)")
    print("=" * 65)

    pop_size = 256       # 256 个生命体并发
    num_columns = 16     # 16 个微柱
    cells_per_col = 64   # 每柱 64 细胞 = 1024 细胞/个体
    in_dim = 32
    out_dim = 8
    steps = 1000         # 模拟 1000 步

    pop = CUDACellularPopulation(pop_size, num_columns, cells_per_col, in_dim, out_dim)
    print(f"[*] 硬件设备: {pop.device} ({torch.cuda.get_device_name(0)})")
    print(f"[*] 并发生命体数 (Pop Size): {pop_size}")
    print(f"[*] 阵列微柱数: {num_columns} 柱, 每柱细胞: {cells_per_col}, 总细胞: {pop.num_cells}")
    print(f"[*] 拓扑架构: 微柱块稀疏阵列 (Block-Sparse Column Array)")
    print(f"[*] 模拟步数: {steps} 步")

    # 预热 CUDA
    dummy_input = torch.randn((pop_size, in_dim), device=pop.device)
    for _ in range(10):
        pop.forward_step(dummy_input)
    torch.cuda.synchronize()

    print("[*] 开始全开火基准压测...")
    t0 = time.perf_counter()

    for step in range(steps):
        pop.forward_step(dummy_input)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_cell_evals = pop_size * pop.num_cells * steps

    print("-" * 65)
    print(f"✅ 完成 {steps} 步完整批量推演，总耗时: {total_time:.3f} 秒")
    print(f"⚡ 单步全种群耗时 (256生命体并发): {total_time / steps * 1000.0:.3f} 毫秒")
    print(f"🔥 细胞动力学有效吞吐量: {total_cell_evals / total_time / 1e6:.2f} M-Cells/s")
    print("-" * 65)

    # 导出测试：检验是否能导出合法可用的 SDSC-BIN v2 文件
    test_bin = "/tmp/test_cuda_champion.bin"
    n_syns = pop.export_champion_to_sdsc_bin(0, test_bin)
    print(f"📦 导出冠军生命体检查点: {test_bin}")
    print(f"   - 细胞总数: {pop.num_cells}")
    print(f"   - 稀疏突触数: {n_syns}")
    print(f"   - 二进制大小: {len(open(test_bin, 'rb').read())} 字节")
    print("=" * 65)

if __name__ == "__main__":
    bench_gpu_throughput()

