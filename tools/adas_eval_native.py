#!/usr/bin/env python3
"""adas_eval_native — tools/adas_eval_native.c 的 ctypes 包装（任务层）。

    from adas_eval_native import NativeEvaluator
    ev = NativeEvaluator(T)                 # T = 已加载的 train_adas_cortex 模块
    ev.bind(organ)                          # 冻结拓扑
    cost = ev.evaluate(gains, weights, seed=7)
    costs = ev.evaluate_batch(G, W, seeds=[1, 2])   # OpenMP 并行

首次使用自动用 gcc 编译到 build/libadas_eval_native.so（源码更新则重编）。
`python3 tools/adas_eval_native.py` 运行与 Python 参考实现的位级对账。
"""
import ctypes as C
import os
import subprocess
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "adas_eval_native.c")
SO = os.path.join(ROOT, "build", "libadas_eval_native.so")

PRIM_CODE = {
    "SUM": 0, "INTEGRATE": 1, "AMPLIFY": 2, "INVERT": 3, "THRESHOLD": 4, "DAMPER": 5,
    "CLIP": 6, "ABS": 7, "MULTIPLY": 8, "DIFF": 9, "HYSTERESIS": 10, "DEADZONE": 11,
    "INHIBIT": 12, "SUB": 13, "RATIO": 14, "OSCILLATOR": 15, "CORRELATION": 16, "FATIGUE": 17,
}
PASSTHROUGH = 18
PATH_KIND = {"straight": 0, "sine": 1, "arc": 2}
SPD_KIND = {"cruise": 0, "cruise_fast": 1, "stop_go": 2, "follow": 3, "ramp": 4}


class EvalConfig(C.Structure):
    _fields_ = [(k, C.c_double) for k in (
        "wheelbase", "dt", "max_lateral_accel", "stg_a_lat_max", "stg_curve_safety",
        "cte_fail", "max_speed", "accel_max", "brake_max",
        "steer_rate_max", "steer_lag_tau", "accel_lag_tau",
        "meas_noise_cte", "meas_noise_psi", "gust_period_s", "gust_accel",
        "lat_env_cruise", "lat_env_maneuver")]


class Topology(C.Structure):
    _fields_ = [("n_cells", C.c_int), ("n_rec", C.c_int), ("n_syn", C.c_int),
                ("steer_id", C.c_int), ("accel_id", C.c_int),
                ("types", C.POINTER(C.c_int32)), ("syn_from", C.POINTER(C.c_int32)),
                ("syn_to", C.POINTER(C.c_int32))]


class Scenario(C.Structure):
    _fields_ = [("path_kind", C.c_int), ("amp", C.c_double), ("wavelen", C.c_double),
                ("kappa", C.c_double), ("spd_kind", C.c_int), ("v0", C.c_double),
                ("duration", C.c_double), ("lead_on", C.c_int)]


class Metrics(C.Structure):
    _fields_ = [("cost", C.c_double), ("avg_cte", C.c_double), ("max_cte", C.c_double),
                ("avg_verr", C.c_double), ("avg_dsteer", C.c_double),
                ("ok", C.c_int), ("steps", C.c_int), ("total", C.c_int)]

    def as_dict(self):
        return {"cost": self.cost, "avg_cte": self.avg_cte, "max_cte": self.max_cte,
                "avg_verr": self.avg_verr, "avg_dsteer": self.avg_dsteer,
                "ok": bool(self.ok), "steps": self.steps, "total": self.total}


def _build():
    os.makedirs(os.path.dirname(SO), exist_ok=True)
    if os.path.exists(SO) and os.path.getmtime(SO) >= os.path.getmtime(SRC):
        return
    cmd = ["gcc", "-O2", "-fopenmp", "-shared", "-fPIC", "-o", SO, SRC, "-lm"]
    subprocess.run(cmd, check=True)


def _load():
    _build()
    lib = C.CDLL(SO)
    lib.adas_eval_scenario.restype = C.c_double
    lib.adas_eval_scenario.argtypes = [C.POINTER(EvalConfig), C.POINTER(Topology),
                                       C.POINTER(C.c_double), C.POINTER(C.c_double),
                                       C.POINTER(Scenario), C.c_uint64, C.POINTER(Metrics)]
    lib.adas_evaluate.restype = C.c_double
    lib.adas_evaluate.argtypes = [C.POINTER(EvalConfig), C.POINTER(Topology),
                                  C.POINTER(C.c_double), C.POINTER(C.c_double),
                                  C.POINTER(Scenario), C.c_int, C.c_uint64, C.POINTER(Metrics)]
    lib.adas_evaluate_batch.restype = None
    lib.adas_evaluate_batch.argtypes = [C.POINTER(EvalConfig), C.POINTER(Topology), C.c_int,
                                        C.POINTER(C.c_double), C.POINTER(C.c_double),
                                        C.POINTER(Scenario), C.c_int,
                                        C.POINTER(C.c_uint64), C.c_int, C.POINTER(C.c_double)]
    lib.adas_eval_num_threads.restype = C.c_int
    lib.adas_eval_num_threads.argtypes = []
    return lib


def load_trainer():
    path = os.path.join(ROOT, "tools", "train_adas_cortex.py")
    T = types.ModuleType("adas_trainer")
    T.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), T.__dict__)
    return T


def config_from_trainer(T):
    return EvalConfig(
        T.WHEELBASE, T.DT, T.MAX_LATERAL_ACCEL, T.STG_A_LAT_MAX, T.STG_CURVE_SAFETY,
        T.CTE_FAIL, T.MAX_SPEED, T.ACCEL_MAX, T.BRAKE_MAX,
        T.STEER_RATE_MAX, T.STEER_LAG_TAU, T.ACCEL_LAG_TAU,
        T.MEAS_NOISE_CTE, T.MEAS_NOISE_PSI, T.GUST_PERIOD_S, T.GUST_ACCEL,
        T.LAT_ENV_CRUISE, T.LAT_ENV_MANEUVER)


def scenarios_to_c(scn):
    arr = (Scenario * len(scn))()
    for i, (name, path, spd, v0, dur, lead) in enumerate(scn):
        arr[i] = Scenario(PATH_KIND[path.kind], float(path.amp), float(path.wavelen),
                          float(path.kappa), SPD_KIND[spd], float(v0), float(dur), int(lead))
    return arr


class NativeEvaluator:
    def __init__(self, T):
        self.T = T
        self.lib = _load()
        self.cfg = config_from_trainer(T)
        self.threads = self.lib.adas_eval_num_threads()

    def bind(self, organ):
        """冻结拓扑：细胞类型 + 突触 (from,to) 列表序。"""
        T = self.T
        ptypes = [PRIM_CODE.get(c.ptype, PASSTHROUGH) for c in organ.cells]
        self._types = np.array(ptypes, dtype=np.int32)
        self._from = np.array([f for f, _, _ in organ.synapses], dtype=np.int32)
        self._to = np.array([t for _, t, _ in organ.synapses], dtype=np.int32)
        self.n_cells, self.n_syn = len(organ.cells), len(organ.synapses)
        self.topo = Topology(self.n_cells, len(T.RECEPTOR_TYPES), self.n_syn,
                             organ.steer_id, organ.accel_id,
                             self._types.ctypes.data_as(C.POINTER(C.c_int32)),
                             self._from.ctypes.data_as(C.POINTER(C.c_int32)),
                             self._to.ctypes.data_as(C.POINTER(C.c_int32)))
        self.gains0 = np.array([c.gain for c in organ.cells], dtype=np.float64)
        self.weights0 = np.array([w for _, _, w in organ.synapses], dtype=np.float64)
        return self

    @staticmethod
    def _dp(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a, a.ctypes.data_as(C.POINTER(C.c_double))

    def run_scenario(self, gains, weights, scenario, seed):
        g, gp = self._dp(gains)
        w, wp = self._dp(weights)
        sc = scenarios_to_c([scenario])
        m = Metrics()
        cost = self.lib.adas_eval_scenario(C.byref(self.cfg), C.byref(self.topo), gp, wp,
                                           sc, seed, C.byref(m))
        return cost, m.as_dict()

    def evaluate(self, gains, weights, scenarios=None, seed=0, detail=False):
        scn = scenarios or self.T.SCENARIOS
        g, gp = self._dp(gains)
        w, wp = self._dp(weights)
        sc = scenarios_to_c(scn)
        ms = (Metrics * len(scn))()
        total = self.lib.adas_evaluate(C.byref(self.cfg), C.byref(self.topo), gp, wp,
                                       sc, len(scn), seed, ms)
        if detail:
            return total, {scn[i][0]: ms[i].as_dict() for i in range(len(scn))}
        return total

    def evaluate_batch(self, gains_all, weights_all, seeds, scenarios=None):
        scn = scenarios or self.T.SCENARIOS
        G, gp = self._dp(np.asarray(gains_all).reshape(-1, self.n_cells))
        W, wp = self._dp(np.asarray(weights_all).reshape(-1, self.n_syn))
        n = G.shape[0]
        sc = scenarios_to_c(scn)
        seeds_arr = np.array(seeds, dtype=np.uint64)
        out = np.zeros(n, dtype=np.float64)
        self.lib.adas_evaluate_batch(C.byref(self.cfg), C.byref(self.topo), n, gp, wp,
                                     sc, len(scn),
                                     seeds_arr.ctypes.data_as(C.POINTER(C.c_uint64)), len(seeds),
                                     out.ctypes.data_as(C.POINTER(C.c_double)))
        return out


def parity_check(bin_path=None, seeds=(0, 7, 20260904)):
    """与 Python 参考实现逐场景位级对账。"""
    import time
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from export_sdsc_cortex import load_cortex_from_bin
    T = load_trainer()
    bin_path = bin_path or os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin")
    organ = T.AdasCortexOrgan.deserialize(load_cortex_from_bin(bin_path)["organ"])
    ev = NativeEvaluator(T).bind(organ)
    print(f"native threads={ev.threads}  cells={ev.n_cells} syn={ev.n_syn}")
    worst = 0.0
    n_exact = n_total = 0
    for seed in seeds:
        for name, path, spd, v0, dur, lead in list(T.SCENARIOS) + list(T.VAL_SCENARIOS):
            cp, okp, mp_ = T.run_scenario(organ, path, spd, v0, dur, lead_on=lead, seed=seed)
            cc, mc = ev.run_scenario(ev.gains0, ev.weights0, (name, path, spd, v0, dur, lead), seed)
            d = max(abs(cp - cc), abs(mp_["avg_cte"] - mc["avg_cte"]), abs(mp_["max_cte"] - mc["max_cte"]))
            exact = (cp == cc and mp_["steps"] == mc["steps"])
            n_exact += exact
            n_total += 1
            worst = max(worst, d)
            flag = "==" if exact else f"Δ={d:.3e}"
            print(f"  seed={seed:<9} {name:16s} py {cp:9.4f} ({mp_['steps']:3d})  c {cc:9.4f} ({mc['steps']:3d})  {flag}")
    t0 = time.time(); T.evaluate(organ, noise_seed=7); tp = time.time() - t0
    t0 = time.time(); ev.evaluate(ev.gains0, ev.weights0, seed=7); tc = time.time() - t0
    G = np.tile(ev.gains0, (48, 1)); W = np.tile(ev.weights0, (48, 1))
    t0 = time.time(); ev.evaluate_batch(G, W, [1, 2]); tb = time.time() - t0
    print(f"bit-exact {n_exact}/{n_total}  worst |Δ|={worst:.3e}")
    print(f"python evaluate {tp*1e3:.0f} ms | native {tc*1e3:.2f} ms (x{tp/tc:.0f}) | batch 48x2 {tb*1e3:.0f} ms = {tb/96*1e3:.2f} ms/eval")
    return n_exact == n_total


if __name__ == "__main__":
    ok = parity_check(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if ok else 1)
