#!/usr/bin/env python3
# ============================================================================
# test_v5_receptor_reconciliation.py — SDSC-BIN v5 受体映射位级对账测试
# 不变量: GPU 逐列注入前向 ≡ C11 sdsc_binary_runtime v5 前向 (同权重同输入)
# 流程: 构建小种群 → 固定输入跑 T 步记录效应器输出 → 导出 v5 bin →
#       C11 harness 编译运行同输入 → 逐位/容差对账
# ============================================================================
import os
import sys
import struct
import subprocess
import tempfile

import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)
from tools.cuda_cellular_engine import CUDACellularPopulation
from tools.train_quant_population_cuda import QuantArrayPopulation

C, K, IN_DIM, OUT_DIM, T, P = 43, 24, 4, 2, 12, 2


def build_population():
    pop = QuantArrayPopulation(pop_size=P, device="cuda" if torch.cuda.is_available() else "cpu")
    return pop


def gpu_reference(pop, inputs_seq):
    """GPU 逐列注入 T 步, 返回成员 0 的效应器输出序列 [T, OUT_DIM]"""
    pop.reset_states()
    outs = []
    for t in range(T):
        x = inputs_seq[t].to(pop.device).unsqueeze(0).expand(pop.pop_size, -1).contiguous()
        pop.forward_step(x)
        eff = pop.outputs.view(pop.pop_size, C, K)[:, :, -2:]
        outs.append(eff[0, -1].detach().cpu().numpy().copy())
    return outs


C_TEMPLATE = r"""
#include "kun/cellular/sdsc_binary_runtime.h"
#include <stdio.h>
#include <math.h>
int main(int argc, char** argv) {
    SDSCBinaryGraph* g = sdsc_binary_load(argv[1]);
    if (!g) { printf("LOAD_FAIL\n"); return 1; }
    printf("v%%u cells=%%u syns=%%u in=%%u out=%%u map=%%d\n", g->header.version,
           g->header.num_cells, g->header.num_synapses,
           g->header.input_dim, g->header.output_dim,
           g->receptor_map != NULL);
    const int IND = %d;
    for (int t = 0; t < %d; ++t) {
        float inputs[172];
        for (int k = 0; k < IND; ++k) {
            /* 与 GPU 侧同式: sin(t*0.7 + c*1.3 + j*0.9) * 0.8, k = c*IND+j */
            int c = k / IND, j = k %% IND;
            inputs[k] = (float)(sin(t * 0.7 + c * 1.3 + j * 0.9) * 0.8);
        }
        float out[1032];  /* 全细胞输出 (sdsc_binary_forward 写满 num_cells) */
        sdsc_binary_forward(g, inputs, out);
        printf("STEP %%d", t);
        for (int o = 0; o < %d; ++o) printf(" %%.9f", out[%d + o]);
        printf("\n");
    }
    return 0;
}
"""


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    pop = build_population()

    # 固定确定性输入序列 (逐列异列: 每列不同特征)
    inputs_seq = []
    for t in range(T):
        x = torch.zeros(C * IN_DIM)
        for c in range(C):
            for j in range(IN_DIM):
                x[c * IN_DIM + j] = float(np.sin(t * 0.7 + c * 1.3 + j * 0.9)) * 0.8
        inputs_seq.append(x)

    gpu_out = gpu_reference(pop, inputs_seq)

    # 导出 v5
    bin_path = os.path.join(tempfile.gettempdir(), "test_v5_receptor.bin")
    pop.export_champion_to_sdsc_bin(0, bin_path, per_column=True)

    # 头部结构校验
    with open(bin_path, "rb") as f:
        hdr = struct.unpack("<IIIIIIQQQQQQ", f.read(72))
    magic, version, n_cells, n_syns, in_dim, out_dim = hdr[:6]
    assert magic == 0x53445343 and version == 5, f"v5 magic/version 错误: {magic:#x} v{version}"
    assert in_dim == C * IN_DIM and out_dim == OUT_DIM, "in/out 维度错误"
    print(f"  ✓ v5 头部: version=5, cells={n_cells}, syns={n_syns}, in={in_dim}, out={out_dim}")

    # C11 harness 编译运行
    c_src = C_TEMPLATE % (C * IN_DIM, T, OUT_DIM, C * K - OUT_DIM)
    c_path = os.path.join(tempfile.gettempdir(), "test_v5_harness.c")
    bin_exec = os.path.join(tempfile.gettempdir(), "test_v5_harness")
    with open(c_path, "w") as f:
        f.write(c_src)
    cc = subprocess.run(f"gcc -O2 -I {WORKSPACE}/include {c_path} -o {bin_exec} -lm",
                        shell=True, capture_output=True, text=True)
    assert cc.returncode == 0, f"C11 harness 编译失败: {cc.stderr}"
    run = subprocess.run(bin_exec + " " + bin_path, shell=True, capture_output=True, text=True)
    assert "LOAD_FAIL" not in run.stdout, "C11 加载 v5 bin 失败"
    lines = [l for l in run.stdout.strip().splitlines() if l.startswith("STEP")]
    assert len(lines) == T, f"C11 步数不足: {len(lines)}"

    # 逐位/容差对账
    max_diff = 0.0
    for t, line in enumerate(lines):
        parts = line.split()
        c11_vals = [float(v) for v in parts[2:2 + OUT_DIM]]
        for o in range(OUT_DIM):
            d = abs(c11_vals[o] - float(gpu_out[t][o]))
            max_diff = max(max_diff, d)
    print(f"  ✓ C11 harness: {lines[0].split()[0]}..{lines[-1].split()[0]} | 最大输出差分: {max_diff:.2e}")
    assert max_diff < 1e-5, f"位级对账失败: max_diff={max_diff:.2e}"

    # 清理
    os.remove(bin_path); os.remove(c_path); os.remove(bin_exec)
    print("[PASS] test_v5_receptor_reconciliation — GPU ≡ C11 (v5 受体映射) 对账通过!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
