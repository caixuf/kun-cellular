#!/usr/bin/env python3
"""
C11 导出体 ↔ Python 演化体 数值对账
==================================
把 `include/kun/cellular/sdsc_cortex.h` 编译成可执行体，喂入与 Python
`AdasCortexOrgan.forward` 完全相同的随机输入序列（含时序状态：INTEGRATE /
DAMPER 细胞带记忆，必须按序推进而非独立采样），逐帧比对两侧输出。

任何一帧 |Δ| > 1e-5 即 FAIL —— 保证"训练出来的细胞"和"车上跑的细胞"是同一个。
"""

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from train_adas_cortex import AdasCortexOrgan, SDSC_PRIMITIVES  # noqa: E402
from export_sdsc_cortex import build_header, load_cortex_from_bin  # noqa: E402

HEADER = os.path.join(ROOT, "include", "kun", "cellular", "sdsc_cortex.h")
CHECKPOINT = os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin")
N_FRAMES = 400
TOL = 1e-5

C_MAIN = r"""
#include <stdio.h>
#include "sdsc_cortex.h"
#include "tasks/adas/sdsc_adas_adapter.h"

int main(void) {
    SdscCortex ctx;
    sdsc_cortex_init_default_adas(&ctx);
    float in[6], out[2];
    while (scanf("%f %f %f %f %f %f",
                 &in[0], &in[1], &in[2], &in[3], &in[4], &in[5]) == 6) {
        sdsc_cortex_forward(&ctx, in, out);
        printf("%.9g %.9g\n", out[0], out[1]);
    }
    return 0;
}
"""


def verify_parity(organ, header_content, test_name):
    organ.reset_state()
    random.seed(4242)
    frames = []
    for _ in range(N_FRAMES):
        frames.append((
            random.uniform(-1.0, 1.0),   # cte_n
            random.uniform(-1.0, 1.0),   # dpsi_n
            random.uniform(-1.0, 1.0),   # kappa_n
            random.uniform(0.0, 1.0),    # v_n
            random.uniform(-1.0, 1.0),   # verr_n
            random.uniform(0.0, 1.0),    # danger_n
        ))

    py_out = [organ.forward(*fr) for fr in frames]

    with tempfile.TemporaryDirectory() as td:
        hdr_path = os.path.join(td, "sdsc_cortex.h")
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(header_content)
        src = os.path.join(td, "parity.c")
        exe = os.path.join(td, "parity")
        with open(src, "w", encoding="utf-8") as f:
            f.write(C_MAIN)
        cc = subprocess.run(
            ["cc", "-std=c11", "-O2", "-I", td, "-I", ROOT,
             src, "-o", exe, "-lm"],
            capture_output=True, text=True)
        if cc.returncode != 0:
            print(f"FAIL [{test_name}]: C 编译失败\n" + cc.stderr)
            return False

        stdin = "\n".join(" ".join(f"{v:.9g}" for v in fr) for fr in frames)
        run = subprocess.run([exe], input=stdin, capture_output=True, text=True)
        if run.returncode != 0:
            print(f"FAIL [{test_name}]: C 运行失败\n" + run.stderr)
            return False

    c_out = [tuple(float(x) for x in line.split())
             for line in run.stdout.strip().splitlines()]

    if len(c_out) != len(py_out):
        print(f"FAIL [{test_name}]: 帧数不一致 C={len(c_out)} Python={len(py_out)}")
        return False

    max_ds = max_da = 0.0
    worst = -1
    for i, ((ps, pa), (cs, ca)) in enumerate(zip(py_out, c_out)):
        ds, da = abs(ps - cs), abs(pa - ca)
        if max(ds, da) > max(max_ds, max_da):
            worst = i
        max_ds, max_da = max(max_ds, ds), max(max_da, da)

    print(f"[{test_name}] 帧数 {len(frames)}  最大 steer 偏差 {max_ds:.3e}  最大 accel 偏差 {max_da:.3e}")
    if max(max_ds, max_da) > TOL:
        i = worst
        print(f"FAIL: 第 {i} 帧 Python={py_out[i]} C={c_out[i]} (容差 {TOL})")
        return False
    print(f"PASS [{test_name}]: C11 导出体与 Python 演化体逐帧数值一致")
    return True


def test_checkpoint_parity():
    if CHECKPOINT.endswith(".bin"):
        ck = load_cortex_from_bin(CHECKPOINT)
    else:
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            ck = json.load(f)
    organ = AdasCortexOrgan.deserialize(ck["organ"])
    with open(HEADER, "r", encoding="utf-8") as f:
        hdr = f.read()
    assert verify_parity(organ, hdr, "Champion Checkpoint")


def test_synthetic_all_primitives_parity():
    """覆盖 SDSC_PRIMITIVES 中所有 18 大完备控制与高阶认知原语"""
    organ = AdasCortexOrgan(n_hidden=len(SDSC_PRIMITIVES), _empty=True)
    organ.hidden_types = list(SDSC_PRIMITIVES)
    organ.synapses = []

    n_rec = organ.n_receptors
    n_hid = len(organ.hidden_types)
    mot = n_rec + n_hid
    m_steer = mot + 4
    m_accel = mot + 5

    # 让每个输入串联到隐藏原语，再汇聚到输出
    for i in range(n_hid):
        hid_idx = n_rec + i
        rec_idx = i % n_rec
        organ.synapses.append((rec_idx, hid_idx, 0.75))
        organ.synapses.append((hid_idx, m_steer, 0.4 if i % 2 == 0 else -0.4))
        organ.synapses.append((hid_idx, m_accel, 0.3 if i % 3 == 0 else -0.3))

    organ.build()
    # 导出临时头文件
    ck = {
        "trainer": "test_synthetic",
        "organ": organ.serialize(),
        "metrics": {}
    }
    hdr_content = build_header(ck)
    assert verify_parity(organ, hdr_content, "All 18 Primitives Synthetic")


def main():
    test_checkpoint_parity()
    test_synthetic_all_primitives_parity()
    print("\n>>> 全部对账测试 100% PASS <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())

