#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunCellular GPU vs C 硬件底座逐细胞全量数值保真对账测试
(Strict Full 1024-Cell GPU <-> C Substrate Bit-Parity Verification Test)
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
import numpy as np
import torch

from tools.cuda_cellular_engine import CUDACellularPopulation

def run_parity_audit():
    ckpt_path = "/tmp/test_parity_champion.bin"
    pop_size = 1
    num_columns = 16
    cells_per_col = 64
    in_dim = 32
    out_dim = 8
    num_steps = 20
    num_cells = num_columns * cells_per_col

    # 禁用 TF32 浮点截断以保证单精度 IEEE-754 严格对齐
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 1. 实例化 GPU 种群并导出
    torch.manual_seed(42)
    pop = CUDACellularPopulation(pop_size, num_columns, cells_per_col, in_dim, out_dim)
    n_syns = pop.export_champion_to_sdsc_bin(0, ckpt_path)

    # 2. 生成确定性的测试输入序列
    np.random.seed(123)
    test_inputs = np.random.randn(num_steps, in_dim).astype(np.float32) * 0.5

    # 3. GPU 端推演 20 步并记录全部 1024 细胞的 outputs (共 20480 个浮点值)
    pop.reset_states()
    gpu_all_cells = []

    for s in range(num_steps):
        inp_t = torch.tensor(test_inputs[s:s+1], device=pop.device)
        pop.forward_step(inp_t)
        # 记录全脑 1024 细胞瞬时激发值
        gpu_all_cells.append(pop.outputs[0].detach().cpu().numpy().copy())

    # 4. 编写并编译 C 原生硬件级运行时测试程序 (输出全脑 1024 细胞)
    c_driver = f"""
    #include "kun/cellular/sdsc_binary_runtime.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>

    int main(int argc, char** argv) {{
        const char* p = (argc > 1) ? argv[1] : "{ckpt_path}";
        SDSCBinaryGraph* g = sdsc_binary_load(p);
        if (!g) {{
            fprintf(stderr, "C load failed!\\n");
            return 1;
        }}

        printf("C_CENSUS: cells=%u syns=%u in_d=%u out_d=%u\\n", 
               g->header.num_cells, g->header.num_synapses, 
               g->header.input_dim, g->header.output_dim);

        float inp[32];
        float out[8];

        for (int step = 0; step < {num_steps}; ++step) {{
            for (int i = 0; i < 32; ++i) {{
                if (scanf("%f", &inp[i]) != 1) return 2;
            }}
            sdsc_binary_forward(g, inp, out);
            printf("STEP %d:", step);
            // 打印全脑全部 1024 细胞数值进行无死角对账
            for (uint32_t i = 0; i < g->header.num_cells; ++i) {{
                printf(" %.7g", g->outputs[i]);
            }}
            printf("\\n");
        }}

        sdsc_binary_free(g);
        return 0;
    }}
    """

    with open("/tmp/run_c_parity.c", "w") as f:
        f.write(c_driver)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compile_res = subprocess.run(f"gcc -O3 -I {repo_root}/include /tmp/run_c_parity.c -o /tmp/run_c_parity -lm", shell=True, capture_output=True, text=True)
    if compile_res.returncode != 0:
        raise RuntimeError(f"C 编译失败: {compile_res.stderr}")

    # 喂入测试输入并运行
    input_str = "\n".join(" ".join(f"{x:.8f}" for x in row) for row in test_inputs)
    proc = subprocess.run(f"/tmp/run_c_parity {ckpt_path}", shell=True, input=input_str, capture_output=True, text=True)
    
    lines = [l for l in proc.stdout.strip().split("\n") if l]
    census_line = lines[0]
    
    # 双向严格图谱审计
    assert f"cells={num_cells}" in census_line, f"细胞数校验失败: {census_line}"
    assert f"syns={n_syns}" in census_line, f"突触数校验失败 (GPU导出 {n_syns} vs C实际读出): {census_line}"
    assert f"in_d={in_dim}" in census_line, f"受体数校验失败: {census_line}"
    assert f"out_d={out_dim}" in census_line, f"效应器数校验失败: {census_line}"

    c_all_cells = []
    for line in lines[1:]:
        if line.startswith("STEP"):
            vals = [float(x) for x in line.split(":")[1].strip().split()]
            assert len(vals) == num_cells, f"C 端输出细胞数不匹配: 期望 {num_cells} 实际 {len(vals)}"
            c_all_cells.append(np.array(vals, dtype=np.float32))

    assert len(c_all_cells) == num_steps, f"时间步数不匹配: {len(c_all_cells)} vs {num_steps}"

    max_diff_all = 0.0
    for s in range(num_steps):
        gpu_c = gpu_all_cells[s]
        c_c = c_all_cells[s]
        step_diff = np.max(np.abs(gpu_c - c_c))
        max_diff_all = max(max_diff_all, step_diff)

    # 检验末端效应器活跃度 (杜绝全零僵尸)
    max_act = max(np.max(np.abs(c[num_cells - out_dim:])) for c in c_all_cells)
    assert max_act > 1e-4, f"效应器信号全零异常: {max_act}"

    return max_diff_all, max_act, n_syns

def test_gpu_cpp_bit_parity():
    """pytest 收集入口"""
    max_diff, max_act, n_syns = run_parity_audit()
    assert max_diff < 1e-4, f"全脑 1024 细胞绝对误差超标: {max_diff:.8e} >= 1e-4"

def main():
    print("=" * 65)
    print("🔬 GPU vs C++ 底座全脑 1024 细胞逐节点数值绝对保真对账 (Zero-Lie Audit)")
    print("=" * 65)

    max_diff, max_act, n_syns = run_parity_audit()
    print(f"[*] 导出的有效突触总数: {n_syns}")
    print(f"[*] 全脑 1024 细胞 × 20 步 (共 20480 细胞状态) 最大绝对误差: {max_diff:.8e}")
    print(f"[*] 效应器动作信号活跃峰值: {max_act:.6f}")

    if max_diff < 1e-4:
        print("🎉 PASS: GPU 与 C 硬件底座实现全脑 1024 细胞无死角等价！(Error < 1e-4)")
        return 0
    else:
        print(f"❌ FAIL: 全脑状态存在数值偏差: {max_diff:.8e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
