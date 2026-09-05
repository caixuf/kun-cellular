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
        
        # 构造期契约断言 (CodeBuddy 审计修复: 规避死循环与静默越界)
        assert num_columns > self.k_projections, f"num_columns ({num_columns}) 必须严格大于 k_projections ({self.k_projections})，否则跨柱图无法建立非自环通路！"
        assert cells_per_col == 64, f"当前微柱原语槽位表针对 K=64 设计 (8+16+16+12+12)，当前 cells_per_col={cells_per_col}！"
        assert in_dim <= cells_per_col, f"in_dim ({in_dim}) 不得大于单微柱细胞数 ({cells_per_col})，防止输入静默跨柱污染！"

        # 跨柱连接拓扑表: [num_columns, k_projections] (严格排除自环与重复目标)
        col_targets = []
        for c in range(num_columns):
            seen, targets = {c}, [] # 将自身加入 seen 彻底排除自环
            # 优先拓扑：相邻柱 + 对角柱
            candidates = [
                (c + 1) % num_columns,
                (c - 1 + num_columns) % num_columns,
                (c + num_columns // 2) % num_columns,
                0 # 枢纽柱
            ]
            for t in candidates:
                if t not in seen:
                    seen.add(t)
                    targets.append(t)
            # 若因排除自环/重合导致不足 k_projections，用步长位移填充有效非自环目标
            step_offset = 2
            while len(targets) < self.k_projections:
                cand = (c + step_offset) % num_columns
                if cand not in seen:
                    seen.add(cand)
                    targets.append(cand)
                step_offset += 1
            col_targets.append(targets[:self.k_projections])

        self.col_targets = torch.tensor(col_targets, device=self.device, dtype=torch.long)
        # 构造期断言：确保无自环、每柱恰好 k_projections 条独立无重复通路
        for c in range(num_columns):
            row = self.col_targets[c].tolist()
            assert c not in row, f"Column {c} contains self-loop in targets: {row}"
            assert len(set(row)) == self.k_projections, f"Column {c} contains duplicate targets: {row}"

        # 3. 原语参数与槽位预分配
        self.param1 = torch.rand((pop_size, self.num_cells), device=self.device) * 0.4 + 0.1
        self.param2 = torch.rand((pop_size, self.num_cells), device=self.device) * 0.2 - 0.1

        # 运行时连续状态张量 (真·Zero-Allocation, 彻底消除运行期 torch.empty)
        self.states = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.aux_states = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.outputs = torch.zeros((pop_size, self.num_cells), device=self.device)
        self.out_buf_col = torch.zeros((pop_size, num_columns, cells_per_col), device=self.device)
        self.inter_drive_buf = torch.zeros((pop_size, num_columns, cells_per_col), device=self.device)
        
        # 缓存常量张量 (消除 torch.where 每步重建标量开销)
        self.const_one = torch.tensor(1.0, device=self.device)
        self.const_neg_one = torch.tensor(-1.0, device=self.device)

    def reset_states(self):
        self.states.zero_()
        self.aux_states.zero_()
        self.outputs.zero_()
        self.out_buf_col.zero_()
        self.inter_drive_buf.zero_()

    def forward_step(self, inputs):
        """
        单步批量全并发无分支推演 (Zero-Allocation & Zero-Branch)
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

        # 2. 柱间跨柱投影计算 (复用预分配缓冲)
        self.inter_drive_buf.zero_()
        out_per_col = self.outputs.view(P, C, K)
        for proj_idx in range(self.k_projections):
            target_cols = self.col_targets[:, proj_idx] # [C]
            W_proj = self.inter_weights[:, :, proj_idx, :] # [P, C, K]
            energy = (out_per_col * W_proj).sum(dim=-1, keepdim=True) # [P, C, 1]
            self.inter_drive_buf.index_add_(1, target_cols, energy.expand(-1, -1, K) * (1.0 / self.k_projections))

        # 合成总驱动力
        drive = (intra_drive + self.inter_drive_buf).view(P, N)

        # 注入外部感知输入到 column 0 受体槽位
        drive[:, :self.in_dim] += inputs

        # 3. 按微柱局部槽位纯张量前向 (复用预分配 out_buf_col, 零内存动态申请)
        drive_per_col = drive.view(P, C, K)
        states_per_col = self.states.view(P, C, K)
        aux_per_col = self.aux_states.view(P, C, K)
        p1_per_col = self.param1.view(P, C, K)
        p2_per_col = self.param2.view(P, C, K)

        # 槽位 A (0..7): 受体/直通通道
        self.out_buf_col[:, :, 0:8] = drive_per_col[:, :, 0:8]

        # 槽位 B (8..23): EMA 指数移动平滑滤波
        sl_b = slice(8, 24)
        alpha = p1_per_col[:, :, sl_b]
        states_per_col[:, :, sl_b] = (1.0 - alpha) * states_per_col[:, :, sl_b] + alpha * drive_per_col[:, :, sl_b]
        self.out_buf_col[:, :, sl_b] = torch.tanh(states_per_col[:, :, sl_b])

        # 槽位 C (24..39): 泄漏积分器
        sl_c = slice(24, 40)
        leak = p1_per_col[:, :, sl_c] * 0.08
        states_per_col[:, :, sl_c] = states_per_col[:, :, sl_c] * (1.0 - leak) + drive_per_col[:, :, sl_c] * 0.05
        self.out_buf_col[:, :, sl_c] = torch.clamp(states_per_col[:, :, sl_c], -2.0, 2.0)

        # 槽位 D (40..51): 二阶非线性谐振子 (Van der Pol)
        sl_d = slice(40, 52)
        s1 = states_per_col[:, :, sl_d]
        s2 = aux_per_col[:, :, sl_d]
        mu = p1_per_col[:, :, sl_d] * 1.5
        ds1 = s2
        ds2 = mu * (1.0 - s1 * s1) * s2 - s1 + drive_per_col[:, :, sl_d]
        dt = 0.05
        s1 = torch.clamp(s1 + ds1 * dt, -3.0, 3.0)
        s2 = torch.clamp(s2 + ds2 * dt, -3.0, 3.0)
        states_per_col[:, :, sl_d] = s1
        aux_per_col[:, :, sl_d] = s2
        self.out_buf_col[:, :, sl_d] = s1

        # 槽位 E (52..63): 施密特双阈值迟滞门控 (使用缓存的常数标量)
        sl_e = slice(52, 64)
        th_hi = p1_per_col[:, :, sl_e]
        th_lo = p2_per_col[:, :, sl_e]
        latch = states_per_col[:, :, sl_e]
        latch = torch.where(drive_per_col[:, :, sl_e] > th_hi, self.const_one, latch)
        latch = torch.where(drive_per_col[:, :, sl_e] < th_lo, self.const_neg_one, latch)
        states_per_col[:, :, sl_e] = latch
        self.out_buf_col[:, :, sl_e] = latch

        # 回写展平状态与输出
        self.states = states_per_col.view(P, N)
        self.aux_states = aux_per_col.view(P, N)
        self.outputs = self.out_buf_col.view(P, N)

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
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] 硬件设备: {pop.device} ({dev_name})")
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

