"""
KunCellular C-ABI Substrate Runtime Binding (Python ctypes Wrapper)
==================================================================
严格遵循《KunCellular 最高架构宪章》：
- C/C++ 是唯一的绝对计算底座 (Single Source of Truth)
- 零手写伪神经网络算子，Python 仅作为外围驱动与数据管线
- 通过 C-ABI / ctypes 直接调度纳秒级纯 C11 运行时推演
"""

import os
import sys
import ctypes
from ctypes import (
    c_uint8, c_uint32, c_float, c_bool,
    POINTER, Structure, byref
)
from typing import Tuple, Optional
import numpy as np

# 1. 动态探测并加载纯 C11 运行时共享库 libkun_cellular_runtime.so
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSSIBLE_LIB_PATHS = [
    os.path.join(ROOT_DIR, "build", "libkun_cellular_runtime.so"),
    os.path.join(ROOT_DIR, "build", "lib", "libkun_cellular_runtime.so"),
    os.path.join(ROOT_DIR, "lib", "libkun_cellular_runtime.so"),
    "libkun_cellular_runtime.so"
]

_clib = None
_lib_path_loaded = None

for path in POSSIBLE_LIB_PATHS:
    if os.path.exists(path):
        try:
            _clib = ctypes.CDLL(path)
            _lib_path_loaded = path
            break
        except Exception as e:
            pass

if _clib is None:
    try:
        _clib = ctypes.CDLL("libkun_cellular_runtime.so")
        _lib_path_loaded = "system:libkun_cellular_runtime.so"
    except Exception:
        pass

if _clib is None:
    raise RuntimeError(
        f"CRITICAL: Failed to load libkun_cellular_runtime.so! Searched paths: {POSSIBLE_LIB_PATHS}. "
        "Please build the project first: `cmake -B build -S . && cmake --build build --target kun_cellular_runtime`."
    )


# 2. 结构体与 C 函数签名定义
class CSdscDiagnostics(Structure):
    _fields_ = [
        ("total_lyapunov_energy", c_float),
        ("max_output_amplitude", c_float),
        ("active_cell_count", c_uint32),
        ("is_bibo_stable", c_bool),
    ]


# sdsc_c_primitive_eval
_clib.sdsc_c_primitive_eval.argtypes = [
    c_uint8, c_float, c_float, POINTER(c_float), POINTER(c_float)
]
_clib.sdsc_c_primitive_eval.restype = c_float

# sdsc_c_tensor_graph_forward
_clib.sdsc_c_tensor_graph_forward.argtypes = [
    c_uint32, c_uint32, c_uint32, c_uint32,
    POINTER(c_uint8), POINTER(c_float),
    POINTER(c_uint32), POINTER(c_uint32), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_uint32)
]
_clib.sdsc_c_tensor_graph_forward.restype = None

# sdsc_c_tensor_graph_reset
_clib.sdsc_c_tensor_graph_reset.argtypes = [
    c_uint32, POINTER(c_float), POINTER(c_float), POINTER(c_float)
]
_clib.sdsc_c_tensor_graph_reset.restype = None

# sdsc_c_tensor_graph_diagnostics
_clib.sdsc_c_tensor_graph_diagnostics.argtypes = [
    c_uint32, POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(CSdscDiagnostics)
]
_clib.sdsc_c_tensor_graph_diagnostics.restype = None

# sdsc_c_cellular_dynamics_step
_clib.sdsc_c_cellular_dynamics_step.argtypes = [
    c_uint32, c_float, c_float, c_float, c_float,
    POINTER(c_uint8), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float)
]
_clib.sdsc_c_cellular_dynamics_step.restype = None

# sdsc_c_organ_forward
_clib.sdsc_c_organ_forward.argtypes = [
    c_uint32, c_uint32, c_uint32,
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float), POINTER(c_float),
    POINTER(c_float), POINTER(c_float)
]
_clib.sdsc_c_organ_forward.restype = None


# 3. 原生 C11 运行时代谢与动力学执行器封装
SDSC_PRIMITIVE_NAME_TO_OP = {
    "SENSE_0": 0, "SENSE_1": 1, "SENSE_2": 2, "SENSE_3": 3,
    "SUM": 4, "INTEGRATE": 5, "AMPLIFY": 6, "INVERT": 7,
    "DAMPER": 8, "CLIP": 9, "ABS": 10, "MULTIPLY": 11,
    "DIFF": 12, "SUB": 13, "RATIO": 14,
    "THRESHOLD": 15, "HYSTERESIS": 16, "DEADZONE": 17,
    "INHIBIT": 18, "AND": 19, "MIN_MAX": 20,
    "ACT_POS": 21, "ACT_NEG": 22, "ACT_RESET": 23,
    "CORRELATION": 24, "FATIGUE": 25, "PASSTHRU": 26
}


class NativeCellularDynamicsEngine:
    """
    纯 C11 硬件级细胞动力学与 STDP 塑性执行器
    - 纳秒级推演 26 大原子动力学原语
    - 求解膜电位微分方程、自由能与在线 STDP+Oja 突触塑性重塑
    - 彻底剥离手写 PyTorch/GPU 胶水算子，回归绝对单一本源
    """
    def __init__(self, n_cells: int = 96):
        self.n_cells = n_cells
        self._alloc_buffers(n_cells)

    def _alloc_buffers(self, n_cells: int):
        self.n_cells = n_cells
        self.op_types = np.zeros(n_cells, dtype=np.uint8)
        self.gains = np.ones(n_cells, dtype=np.float32)
        self.states = np.zeros(n_cells, dtype=np.float32)
        self.aux_states = np.zeros(n_cells, dtype=np.float32)
        self.outputs = np.zeros(n_cells, dtype=np.float32)
        self.preds = np.zeros(n_cells, dtype=np.float32)
        self.errors = np.zeros(n_cells, dtype=np.float32)
        self.W = np.zeros((n_cells, n_cells), dtype=np.float32)
        self.mask = np.zeros((n_cells, n_cells), dtype=np.float32)

    def load_topology(self, cells, synapses):
        n = len(cells)
        if n != self.n_cells:
            self._alloc_buffers(n)
        else:
            self.states.fill(0.0)
            self.aux_states.fill(0.0)
            self.outputs.fill(0.0)
            self.preds.fill(0.0)
            self.errors.fill(0.0)
            self.W.fill(0.0)
            self.mask.fill(0.0)

        id_to_idx = {c.id: idx for idx, c in enumerate(cells)}
        for idx, c in enumerate(cells):
            op = SDSC_PRIMITIVE_NAME_TO_OP.get(c.type, 4)  # 默认 SUM
            self.op_types[idx] = op
            self.gains[idx] = getattr(c, "gain", 1.0)

        for s in synapses:
            u = id_to_idx.get(s.get("from"))
            v = id_to_idx.get(s.get("to"))
            w = float(s.get("weight", 1.0))
            if u is not None and v is not None and u < self.n_cells and v < self.n_cells:
                self.W[u, v] = w
                self.mask[u, v] = 1.0

    def step(
        self,
        t: float,
        red_queen_pressure: float = 1.0,
        eta: float = 0.006,
        alpha: float = 0.012
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """调用纯 C11 底座单步执行动力学演化"""
        if self.n_cells == 0:
            return 0.0, 0.0, self.outputs, self.states, self.preds, self.errors

        free_energy = c_float(0.0)
        plasticity_flux = c_float(0.0)

        _clib.sdsc_c_cellular_dynamics_step(
            c_uint32(self.n_cells),
            c_float(t),
            c_float(red_queen_pressure),
            c_float(eta),
            c_float(alpha),
            self.op_types.ctypes.data_as(POINTER(c_uint8)),
            self.gains.ctypes.data_as(POINTER(c_float)),
            self.states.ctypes.data_as(POINTER(c_float)),
            self.aux_states.ctypes.data_as(POINTER(c_float)),
            self.outputs.ctypes.data_as(POINTER(c_float)),
            self.preds.ctypes.data_as(POINTER(c_float)),
            self.errors.ctypes.data_as(POINTER(c_float)),
            self.W.ctypes.data_as(POINTER(c_float)),
            self.mask.ctypes.data_as(POINTER(c_float)),
            byref(free_energy),
            byref(plasticity_flux)
        )

        return (
            float(free_energy.value),
            float(plasticity_flux.value),
            self.outputs,
            self.states,
            self.preds,
            self.errors
        )


class NativeOrganExecutor:
    """
    纯 C11 硬件级器官分层推演内核 (受体层 32 -> 联络层 768 -> 运动效应层 224)
    """
    @staticmethod
    def forward(
        rec: np.ndarray,
        W1: Optional[np.ndarray],
        W2: Optional[np.ndarray],
        H_state: np.ndarray,
        H_out: np.ndarray,
        MOT_out: np.ndarray
    ) -> Tuple[float, float]:
        n_rec = len(rec)
        n_hidden = len(H_state)
        n_mot = len(MOT_out)

        out_primary = c_float(0.0)
        out_secondary = c_float(0.0)

        rec_c = rec.astype(np.float32)
        H_state_c = H_state.astype(np.float32)
        H_out_c = H_out.astype(np.float32)
        MOT_out_c = MOT_out.astype(np.float32)

        p_w1 = W1.astype(np.float32).ctypes.data_as(POINTER(c_float)) if W1 is not None else None
        p_w2 = W2.astype(np.float32).ctypes.data_as(POINTER(c_float)) if W2 is not None else None

        _clib.sdsc_c_organ_forward(
            c_uint32(n_rec),
            c_uint32(n_hidden),
            c_uint32(n_mot),
            rec_c.ctypes.data_as(POINTER(c_float)),
            p_w1,
            p_w2,
            H_state_c.ctypes.data_as(POINTER(c_float)),
            H_out_c.ctypes.data_as(POINTER(c_float)),
            MOT_out_c.ctypes.data_as(POINTER(c_float)),
            byref(out_primary),
            byref(out_secondary)
        )

        # 拷贝状态回传
        np.copyto(H_state, H_state_c)
        np.copyto(H_out, H_out_c)
        np.copyto(MOT_out, MOT_out_c)

        return float(out_primary.value), float(out_secondary.value)
