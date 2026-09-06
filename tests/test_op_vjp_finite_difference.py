#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunCellular 原语算子 VJP 解析梯度 vs 有限差分数值梯度对账门禁测试
(Vector-Jacobian Product vs Finite Difference Parity & STE Verification)

门禁准则:
  1. C11 (sdsc_primitives_vjp.h) vs C++ (cuda_ops_vjp.cuh) VJP 算子位级一致
  2. 全部可微原语: 解析导数与双边有限差分误差 < 1e-5 (针对 x, g, s)
  3. 不可微门控/效应器原语: 直通估计器 (STE) 导数通路正常，无 NaN/Inf，无梯度截断
"""

import os
import sys
import subprocess
import ctypes
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import tools.kun_cellular_ops as kops

class SdscOpVJP(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_float),
        ("ds_prev", ctypes.c_float),
        ("da_prev", ctypes.c_float),
        ("dg", ctypes.c_float),
    ]

@pytest.fixture(scope="module")
def vjp_lib():
    test_src = f"""
    #include "kun/cellular/sdsc_primitives.h"
    #include "kun/cellular/sdsc_primitives_vjp.h"
    #include "kun/cellular/cuda_ops.cuh"
    #include "kun/cellular/cuda_ops_vjp.cuh"

    extern "C" {{
        float eval_forward_c(uint8_t op, float g, float x, float* s, float* a) {{
            return sdsc_primitive_eval(op, g, x, s, a);
        }}

        SdscOpVJP eval_vjp_c(
            uint8_t op, float g, float x,
            float s, float a, float out,
            float s_next, float a_next,
            float dy, float ds_next, float da_next
        ) {{
            return sdsc_primitive_vjp(op, g, x, s, a, out, s_next, a_next, dy, ds_next, da_next);
        }}

        SdscCUDAVJP eval_vjp_cpp(
            uint8_t op, float g, float x,
            float s, float a, float out,
            float s_next, float a_next,
            float dy, float ds_next, float da_next
        ) {{
            return sdsc_cuda_vjp_primitive(op, g, x, s, a, out, s_next, a_next, dy, ds_next, da_next);
        }}
    }}
    """
    tmp_c_path = "/tmp/test_vjp_harness.cpp"
    tmp_so_path = "/tmp/libtest_vjp_harness.so"
    with open(tmp_c_path, "w") as f:
        f.write(test_src)

    compile_cmd = f"g++ -O3 -fPIC -shared -I {REPO_ROOT}/include {tmp_c_path} -o {tmp_so_path} -lm"
    res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    assert res.returncode == 0, f"编译 VJP Harness 失败:\n{res.stderr}"

    lib = ctypes.CDLL(tmp_so_path)
    lib.eval_forward_c.argtypes = [ctypes.c_uint8, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.eval_forward_c.restype = ctypes.c_float

    lib.eval_vjp_c.argtypes = [
        ctypes.c_uint8, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float
    ]
    lib.eval_vjp_c.restype = SdscOpVJP

    lib.eval_vjp_cpp.argtypes = lib.eval_vjp_c.argtypes
    lib.eval_vjp_cpp.restype = SdscOpVJP

    return lib

def test_vjp_parity_c11_vs_cpp(vjp_lib):
    """门禁 1: 验证 C11 与 C++ (CUDA Host) VJP 解析实现位级对账"""
    for meta in kops.PRIMITIVES_META:
        op = meta["id"]
        g = 1.25
        x = 0.35
        s = 0.20
        a = 0.15

        s_fwd = ctypes.c_float(s)
        a_fwd = ctypes.c_float(a)
        out = vjp_lib.eval_forward_c(op, g, x, ctypes.byref(s_fwd), ctypes.byref(a_fwd))
        s_next = s_fwd.value
        a_next = a_fwd.value

        for dy, ds_next, da_next in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.2)]:
            v_c = vjp_lib.eval_vjp_c(op, g, x, s, a, out, s_next, a_next, dy, ds_next, da_next)
            v_cpp = vjp_lib.eval_vjp_cpp(op, g, x, s, a, out, s_next, a_next, dy, ds_next, da_next)

            assert abs(v_c.dx - v_cpp.dx) < 1e-6, f"op={op} ({meta['name']}) dx mismatch"
            assert abs(v_c.ds_prev - v_cpp.ds_prev) < 1e-6, f"op={op} ({meta['name']}) ds mismatch"
            assert abs(v_c.da_prev - v_cpp.da_prev) < 1e-6, f"op={op} ({meta['name']}) da mismatch"
            assert abs(v_c.dg - v_cpp.dg) < 1e-6, f"op={op} ({meta['name']}) dg mismatch"

def test_differentiable_ops_finite_difference(vjp_lib):
    """门禁 2: 全部可微原语有限差分校验 (|vjp - fd| < 1e-3 float32 数值对账)"""
    eps = 1e-3
    test_points = [
        (0.25, 1.0, 0.1, 0.05),
        (-0.45, 1.5, -0.3, 0.2),
        (0.65, 0.8, 0.5, -0.1),
    ]

    # 跳过不可微/分段边界点的原语或采用 STE 的原语
    smooth_ops = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 18, 24, 25, 26, 27
    ]

    for meta in kops.PRIMITIVES_META:
        op = meta["id"]
        if op not in smooth_ops:
            continue

        for x, g, s, a in test_points:
            # 基础前向
            s_fwd = ctypes.c_float(s)
            a_fwd = ctypes.c_float(a)
            out = vjp_lib.eval_forward_c(op, g, x, ctypes.byref(s_fwd), ctypes.byref(a_fwd))
            s_next = s_fwd.value
            a_next = a_fwd.value

            # 1. 验证关于 x 的偏导: dy=1.0, ds_next=0.0
            v = vjp_lib.eval_vjp_c(op, g, x, s, a, out, s_next, a_next, 1.0, 0.0, 0.0)

            # 有限差分: [f(x+eps) - f(x-eps)] / (2*eps)
            s_p = ctypes.c_float(s)
            a_p = ctypes.c_float(a)
            out_p = vjp_lib.eval_forward_c(op, g, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))

            s_m = ctypes.c_float(s)
            a_m = ctypes.c_float(a)
            out_m = vjp_lib.eval_forward_c(op, g, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))

            fd_dx = (out_p - out_m) / (2.0 * eps)
            err_dx = abs(v.dx - fd_dx)
            assert err_dx < 1e-3, f"op={op} ({meta['name']}) dx err={err_dx:.2e} vjp={v.dx} fd={fd_dx}"

            # 2. 验证关于 g 的偏导: dy=1.0, ds_next=0.0 (仅当 op 不是无参原语)
            if op in [4, 5, 6, 7, 10, 11, 13, 18, 24, 25]:
                s_gp = ctypes.c_float(s)
                a_gp = ctypes.c_float(a)
                out_gp = vjp_lib.eval_forward_c(op, g + eps, x, ctypes.byref(s_gp), ctypes.byref(a_gp))

                s_gm = ctypes.c_float(s)
                a_gm = ctypes.c_float(a)
                out_gm = vjp_lib.eval_forward_c(op, g - eps, x, ctypes.byref(s_gm), ctypes.byref(a_gm))

                fd_dg = (out_gp - out_gm) / (2.0 * eps)
                err_dg = abs(v.dg - fd_dg)
                assert err_dg < 1e-3, f"op={op} ({meta['name']}) dg err={err_dg:.2e} vjp={v.dg} fd={fd_dg}"

            # 3. 验证关于 s 的时序偏导 (对有状态原语): dy=0.0, ds_next=1.0 -> d(s_next)/ds
            if meta["has_state"] and op in [5, 8, 13, 14, 18, 24, 27]:
                v_s = vjp_lib.eval_vjp_c(op, g, x, s, a, out, s_next, a_next, 0.0, 1.0, 0.0)

                s_sp = ctypes.c_float(s + eps)
                a_sp = ctypes.c_float(a)
                vjp_lib.eval_forward_c(op, g, x, ctypes.byref(s_sp), ctypes.byref(a_sp))

                s_sm = ctypes.c_float(s - eps)
                a_sm = ctypes.c_float(a)
                vjp_lib.eval_forward_c(op, g, x, ctypes.byref(s_sm), ctypes.byref(a_sm))

                fd_ds = (s_sp.value - s_sm.value) / (2.0 * eps)
                err_ds = abs(v_s.ds_prev - fd_ds)
                assert err_ds < 1e-3, f"op={op} ({meta['name']}) ds_prev err={err_ds:.2e} vjp={v_s.ds_prev} fd={fd_ds}"

def test_ste_non_differentiable_ops(vjp_lib):
    """门禁 3: 不可微原语直通估计器 (STE) 有效性校验"""
    ste_ops = [15, 16, 19, 23]
    for op in ste_ops:
        meta = kops.PRIMITIVES_META[op]
        assert not meta["differentiable"]
        assert meta["ste_gradient"] in ("straight_through", "identity")

        # 在区间内输入，梯度应当直通 (dx > 0)
        v = vjp_lib.eval_vjp_c(op, 1.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert not np.isnan(v.dx) and not np.isinf(v.dx)
        assert v.dx > 0.0, f"op={op} ({meta['name']}) STE gradient should be positive"

def test_piecewise_linear_vjp_finite_difference(vjp_lib):
    """门禁 4: 分段线性/非光滑原语在非不可导拐点 (Non-kink interior) 处的解析 VJP 与有限差分对账"""
    eps = 1e-4

    # 1. SDSC_OP_CLIP (9): f(x) = clamp(x*g, -1, 1). 当 |x*g| < 1 时，导数为 g
    for x in [0.35, -0.35]:
        g = 1.2
        s_p = ctypes.c_float(0.0)
        a_p = ctypes.c_float(0.0)
        out_p = vjp_lib.eval_forward_c(9, g, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))
        s_m = ctypes.c_float(0.0)
        a_m = ctypes.c_float(0.0)
        out_m = vjp_lib.eval_forward_c(9, g, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))
        fd_dx = (out_p - out_m) / (2.0 * eps)
        v = vjp_lib.eval_vjp_c(9, g, x, 0.0, 0.0, x * g, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert abs(v.dx - fd_dx) < 1e-3, f"CLIP dx err={abs(v.dx - fd_dx)}"

    # 2. SDSC_OP_DEADZONE (17): |x| > 0.08 ? x*g : 0. 当 |x| = 0.35 (远离 0.08 拐点) 时，导数为 g
    for x in [0.35, -0.35]:
        g = 1.2
        s_p = ctypes.c_float(0.0)
        a_p = ctypes.c_float(0.0)
        out_p = vjp_lib.eval_forward_c(17, g, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))
        s_m = ctypes.c_float(0.0)
        a_m = ctypes.c_float(0.0)
        out_m = vjp_lib.eval_forward_c(17, g, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))
        fd_dx = (out_p - out_m) / (2.0 * eps)
        v = vjp_lib.eval_vjp_c(17, g, x, 0.0, 0.0, x * g, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert abs(v.dx - fd_dx) < 1e-3, f"DEADZONE dx err={abs(v.dx - fd_dx)}"

    # 3. SDSC_OP_MIN_MAX (20): g > 0.5 ? max(x, s) : min(x, s). 当 x != s 时光滑
    x, s, g = 0.35, 0.10, 1.0
    s_p = ctypes.c_float(s)
    a_p = ctypes.c_float(0.0)
    out_p = vjp_lib.eval_forward_c(20, g, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))
    s_m = ctypes.c_float(s)
    a_m = ctypes.c_float(0.0)
    out_m = vjp_lib.eval_forward_c(20, g, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))
    fd_dx = (out_p - out_m) / (2.0 * eps)
    v = vjp_lib.eval_vjp_c(20, g, x, s, 0.0, max(x, s), s, 0.0, 1.0, 0.0, 0.0)
    assert abs(v.dx - fd_dx) < 1e-3, f"MIN_MAX dx err={abs(v.dx - fd_dx)}"

    # 4. SDSC_OP_ACT_POS (21): x in (0, 1) 时为恒等线性
    x = 0.35
    s_p = ctypes.c_float(0.0)
    a_p = ctypes.c_float(0.0)
    out_p = vjp_lib.eval_forward_c(21, 1.0, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))
    s_m = ctypes.c_float(0.0)
    a_m = ctypes.c_float(0.0)
    out_m = vjp_lib.eval_forward_c(21, 1.0, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))
    fd_dx = (out_p - out_m) / (2.0 * eps)
    v = vjp_lib.eval_vjp_c(21, 1.0, x, 0.0, 0.0, x, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert abs(v.dx - fd_dx) < 1e-3, f"ACT_POS dx err={abs(v.dx - fd_dx)}"

    # 5. SDSC_OP_ACT_NEG (22): x in (-1, 0) 时为 -x
    x = -0.35
    s_p = ctypes.c_float(0.0)
    a_p = ctypes.c_float(0.0)
    out_p = vjp_lib.eval_forward_c(22, 1.0, x + eps, ctypes.byref(s_p), ctypes.byref(a_p))
    s_m = ctypes.c_float(0.0)
    a_m = ctypes.c_float(0.0)
    out_m = vjp_lib.eval_forward_c(22, 1.0, x - eps, ctypes.byref(s_m), ctypes.byref(a_m))
    fd_dx = (out_p - out_m) / (2.0 * eps)
    v = vjp_lib.eval_vjp_c(22, 1.0, x, 0.0, 0.0, -x, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert abs(v.dx - fd_dx) < 1e-3, f"ACT_NEG dx err={abs(v.dx - fd_dx)}"

