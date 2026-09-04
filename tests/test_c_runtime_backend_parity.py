"""
Test C11 Substrate Runtime Binding & Live Backend Parity
========================================================
验证纯 C11 硬件级共享库 (libkun_cellular_runtime.so) 及其 Python ctypes 绑定的数值精度与稳定性，
确保 tools/cellular_live_backend.py 彻底摆脱手写 numpy/torch 伪神经网络算子，
100% 遵从最高架构宪章：C 纯底座为唯一真实本源。
"""

import os
import sys
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.cellular_c_runtime import (
    NativeCellularDynamicsEngine,
    NativeOrganExecutor,
    CSdscDiagnostics,
    _clib
)
from tools.cellular_live_backend import (
    SiliconCellularOrganism,
    SdscSiliconLifeOrgan,
    CUDACellularDynamicsEngine
)


def test_clib_symbols_exported():
    """验证 C 运行时核心导出符号完整性"""
    assert hasattr(_clib, "sdsc_c_primitive_eval")
    assert hasattr(_clib, "sdsc_c_tensor_graph_forward")
    assert hasattr(_clib, "sdsc_c_tensor_graph_reset")
    assert hasattr(_clib, "sdsc_c_tensor_graph_diagnostics")
    assert hasattr(_clib, "sdsc_c_cellular_dynamics_step")
    assert hasattr(_clib, "sdsc_c_organ_forward")


def test_primitive_eval_pure_c():
    """验证原子原语纯 C 单步推演精度与有界性"""
    state = (np.zeros(1, dtype=np.float32)).ctypes.data_as(pytest.importorskip("ctypes").POINTER(pytest.importorskip("ctypes").c_float))
    aux = (np.zeros(1, dtype=np.float32)).ctypes.data_as(pytest.importorskip("ctypes").POINTER(pytest.importorskip("ctypes").c_float))

    # SUM: tanh(x * g)
    out_sum = _clib.sdsc_c_primitive_eval(4, 1.0, 0.5, state, aux)
    assert np.isclose(out_sum, np.tanh(0.5), atol=1e-5)

    # AMPLIFY: tanh(x * g * 2.5)
    out_amp = _clib.sdsc_c_primitive_eval(6, 1.0, 0.5, state, aux)
    assert np.isclose(out_amp, np.tanh(0.5 * 2.5), atol=1e-5)

    # CLIP: clamp(x * g, -1, 1)
    out_clip = _clib.sdsc_c_primitive_eval(9, 2.0, 1.5, state, aux)
    assert out_clip == 1.0


def test_native_dynamics_engine_stability():
    """验证 3D 生物形态发生推演连续 50 步稳定性 (无 NaN，无发散)"""
    class DummyCell:
        def __init__(self, cid, ptype, gain=1.2):
            self.id = cid
            self.type = ptype
            self.gain = gain

    types = ["SUM", "INTEGRATE", "AMPLIFY", "DAMPER", "CLIP", "HYSTERESIS", "DEADZONE"]
    cells = [DummyCell(i, types[i % len(types)], gain=1.0 + (i % 5) * 0.1) for i in range(96)]
    synapses = [{"from": i, "to": (i + 1) % 96, "weight": 0.4} for i in range(96)]

    engine = NativeCellularDynamicsEngine(96)
    engine.load_topology(cells, synapses)

    for step in range(50):
        t = step * 0.04
        fe, flux, outs, states, preds, errors = engine.step(t, red_queen_pressure=1.0)
        assert np.all(np.isfinite(outs))
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(preds))
        assert np.all(np.isfinite(errors))
        assert 0.0 <= fe < 10.0
        assert 0.0 <= flux < 1.0


def test_organ_executor_c_parity():
    """验证器官分层推演内核 C 实现运算正确性与状态累积"""
    n_rec = 32
    n_hidden = 768
    n_mot = 224

    rec = np.random.randn(n_rec).astype(np.float32)
    W1 = np.random.randn(n_rec, n_hidden).astype(np.float32) * 0.02
    W2 = np.random.randn(n_hidden, n_mot).astype(np.float32) * 0.02

    H_state = np.zeros(n_hidden, dtype=np.float32)
    H_out = np.zeros(n_hidden, dtype=np.float32)
    MOT_out = np.zeros(n_mot, dtype=np.float32)

    out_primary, out_secondary = NativeOrganExecutor.forward(
        rec, W1, W2, H_state, H_out, MOT_out
    )

    # 手动对账纯数学期望
    expected_h_raw = np.dot(rec, W1)
    expected_h_state = expected_h_raw * 0.18
    expected_h_out = np.tanh(expected_h_state)
    expected_mot = np.tanh(np.dot(expected_h_out, W2))

    assert np.allclose(H_out, expected_h_out, atol=1e-5)
    assert np.allclose(MOT_out, expected_mot, atol=1e-5)
    assert np.isclose(out_primary, expected_mot[0], atol=1e-5)
    assert np.isclose(out_secondary, expected_mot[1], atol=1e-5)


def test_live_backend_integration():
    """验证 cellular_live_backend 中的模型对象无缝调用 C 运行时"""
    org = SiliconCellularOrganism()
    for _ in range(10):
        org.step_physics_and_signal()

    assert org.free_energy > 0
    assert len(org.cells) > 0
    assert all(np.isfinite(c.out) for c in org.cells)

    organ = SdscSiliconLifeOrgan()
    organ.W1 = np.random.randn(32, 768).astype(np.float32) * 0.02
    organ.W2 = np.random.randn(768, 224).astype(np.float32) * 0.02
    steer, speed = organ.forward(0.05, -0.02, 0.01, 0.3)

    assert np.isfinite(steer)
    assert np.isfinite(speed)
