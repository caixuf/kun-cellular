#!/usr/bin/env python3
"""稳定性战役 #1：在 L3 拓扑上加入体育场公路代价，sep-CMA 只调增益/权重。

不覆盖 adas_cortex_champion.bin；输出 checkpoints/adas_cortex_champion_stadium.bin
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def load_trainer():
    path = os.path.join(ROOT, "tools", "train_adas_cortex.py")
    T = types.ModuleType("adas_trainer")
    T.__file__ = path
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), T.__dict__)
    return T


T = load_trainer()
from export_sdsc_cortex import load_cortex_from_bin  # noqa: E402


class Stadium:
    L, R, WB = 70.0, 32.0, 2.7

    def __init__(self):
        self.lap = 2 * self.L + 2 * math.pi * self.R

    def project(self, xm, ym):
        half_L, R = self.L * 0.5, self.R
        if xm >= half_L:
            phi = math.atan2(ym, xm - half_L)
            cx = half_L + R * math.cos(phi)
            cy = R * math.sin(phi)
            th = (phi + math.pi * 0.5) % math.tau
            kap = 1.0 / R
        elif xm <= -half_L:
            phi = math.atan2(ym, xm + half_L)
            cx = -half_L + R * math.cos(phi)
            cy = R * math.sin(phi)
            th = (phi + math.pi * 0.5) % math.tau
            kap = 1.0 / R
        elif ym <= 0:
            cx, cy, th, kap = xm, -R, 0.0, 0.0
        else:
            cx, cy, th, kap = xm, R, math.pi, 0.0
        cte = math.cos(th) * (cy - ym) - math.sin(th) * (cx - xm)
        return cx, cy, th, kap, cte

    def start(self):
        return -self.L * 0.5, -self.R, 0.0


def stadium_cost(organ, duration=25.0, dt=0.04, seed=0):
    """闭环体育场一圈量级；失败离轨重罚。"""
    organ.reset_state()
    st = Stadium()
    xm, ym, th = st.start()
    v, delta, accel_act = 12.0, 0.0, 0.0
    cum_cte = 0.0
    steps = int(duration / dt)
    ok_steps = 0
    rng = random.Random(seed)
    for i in range(steps):
        _, _, th_b, kap, cte = st.project(xm, ym)
        if abs(cte) > 5.0:
            return 80.0 + abs(cte), False
        heading_err = (th_b - th + math.pi) % math.tau - math.pi
        # 前瞻曲率限速（与 live 后端同构）
        target_v = 13.5
        if abs(kap) > 1e-4:
            target_v = min(target_v, 0.78 * math.sqrt(4.0 / abs(kap)))
        target_v = max(4.0, min(T.MAX_SPEED, target_v))
        cte_n = max(-1.0, min(1.0, (cte + rng.gauss(0, 0.01)) / 2.0))
        dpsi_n = max(-1.0, min(1.0, (heading_err + rng.gauss(0, 0.005)) / 0.5))
        kap_n = max(-1.0, min(1.0, kap * 20.0))
        v_n = max(0.0, min(1.0, v / T.MAX_SPEED))
        verr_n = max(-1.0, min(1.0, (target_v - v) / 5.0))
        danger = min(1.0, abs(cte) / 2.0)
        steer_n, accel_n = organ.forward(cte_n, dpsi_n, kap_n, v_n, verr_n, danger)
        lim = T.adaptive_steer_limit(v, cte)
        steer_req = max(-lim, min(lim, float(steer_n) * lim))
        d_max = T.STEER_RATE_MAX * dt
        steer_req = delta + max(-d_max, min(d_max, steer_req - delta))
        delta += (steer_req - delta) * min(1.0, dt / max(1e-3, T.STEER_LAG_TAU))
        delta = max(-lim, min(lim, delta))
        accel_req = float(accel_n) * T.ACCEL_MAX if accel_n > 0 else float(accel_n) * 6.0
        accel_act += (accel_req - accel_act) * min(1.0, dt / max(1e-3, T.ACCEL_LAG_TAU))
        v = max(1.0, min(T.MAX_SPEED, v + accel_act * dt))
        yaw_rate = (v / st.WB) * math.tan(delta)
        half = st.WB * 0.5
        xm += (v * math.cos(th) - half * math.sin(th) * yaw_rate) * dt
        ym += (v * math.sin(th) + half * math.cos(th) * yaw_rate) * dt
        th = (th + yaw_rate * dt + math.pi) % math.tau - math.pi
        cum_cte += abs(cte)
        ok_steps += 1
    avg = cum_cte / max(1, ok_steps)
    cost = avg * 25.0 + (0.0 if ok_steps == steps else 40.0)
    return cost, ok_steps == steps


def apply_params(organ, x, n_cells):
    for i, c in enumerate(organ.cells):
        c.gain = math.exp(x[i])
    organ.synapses = [(f, t, float(w)) for (f, t, _), w in zip(organ.synapses, x[n_cells:])]
    organ.compile_incoming()


def params_from_organ(organ):
    g = np.log(np.array([max(c.gain, 1e-3) for c in organ.cells]))
    w = np.array([ww for _, _, ww in organ.synapses], dtype=float)
    return np.concatenate([g, w])


def write_bin(out_path, organ, generation, metrics, note):
    ser = organ.serialize()
    n_rec, n_mot = len(T.RECEPTOR_TYPES), len(T.MOTOR_TYPES)
    n_cells = len(organ.cells)
    meta = json.dumps({
        "organism_id": "adas_cortex_champion_stadium",
        "generation": generation,
        "organ": ser,
        "metrics": metrics,
        "tuning": note,
    }, ensure_ascii=False).encode("utf-8")
    adj = [[] for _ in range(n_cells)]
    for f, t, w in organ.synapses:
        adj[f].append((t, float(w)))
    row_ptr, col_idx, weights = [0] * (n_cells + 1), [], []
    for i in range(n_cells):
        row_ptr[i] = len(col_idx)
        for v, w in adj[i]:
            col_idx.append(v)
            weights.append(w)
    row_ptr[n_cells] = len(col_idx)
    n_syn = len(col_idx)
    hdr = 72
    cells_off = hdr
    rp_off = cells_off + n_cells * 4
    ci_off = rp_off + (n_cells + 1) * 4
    w_off = ci_off + n_syn * 4
    co_off = w_off + n_syn * 4
    header = struct.pack(
        "<IIIIIIQQQQQQ", 0x53445343, 2, n_cells, n_syn, 6, 2,
        cells_off, rp_off, ci_off, w_off, co_off,
        (generation & 0xFFFFFFFF) | ((len(meta) & 0xFFFFFFFF) << 32),
    )
    cell_bytes = bytearray(n_cells * 4)
    for i, c in enumerate(organ.cells):
        cell_bytes[i * 4] = 4
        cell_bytes[i * 4 + 1] = min(255, max(0, int(c.gain * 64.0)))
        flags = (0x01 if i < n_rec else 0) | (0x02 if i >= n_cells - n_mot else 0)
        cell_bytes[i * 4 + 3] = flags
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(cell_bytes)
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())
        f.write(np.array(weights, dtype=np.float32).tobytes())
        f.write(np.zeros((n_cells, 3), dtype=np.float32).tobytes())
        f.write(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "adas_cortex_champion_stadium.bin"))
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--stadium-weight", type=float, default=1.5)
    args = ap.parse_args()

    organ0 = T.AdasCortexOrgan.deserialize(load_cortex_from_bin(args.inp)["organ"])
    x0 = params_from_organ(organ0)
    n_cells = len(organ0.cells)
    dim = len(x0)
    lam = min(16, max(8, dim // 40))
    rng = np.random.default_rng(20260914)
    # 轻量对角 CMA
    m = x0.copy()
    sigma = 0.08
    C = np.ones(dim)

    def fitness(x):
        o = T.AdasCortexOrgan.deserialize(organ0.serialize())
        apply_params(o, x, n_cells)
        train_c, _, _ = T.evaluate(o, T.SCENARIOS[:6], noise_seed=7)  # 子集保交互速度
        st_c, st_ok = stadium_cost(o, duration=20.0, seed=11)
        return train_c + args.stadium_weight * st_c + (0.0 if st_ok else 20.0), st_c, st_ok

    best_x, best_f, best_st = m.copy(), float("inf"), None
    t0 = time.time()
    hist = []
    for it in range(1, args.iters + 1):
        zs = rng.standard_normal((lam, dim))
        xs = m + sigma * zs * np.sqrt(C)
        scored = []
        for x in xs:
            f, st_c, st_ok = fitness(x)
            scored.append((f, x, st_c, st_ok))
        scored.sort(key=lambda r: r[0])
        if scored[0][0] < best_f:
            best_f, best_x = scored[0][0], scored[0][1].copy()
            best_st = {"stadium_cost": scored[0][2], "stadium_ok": scored[0][3]}
        # 更新均值（μ=lam//2）
        mu = max(2, lam // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        m = sum(w[i] * scored[i][1] for i in range(mu))
        hist.append({"iter": it, "best_f": round(best_f, 3), **(best_st or {})})
        if it % 10 == 0 or it == 1:
            print(f"iter {it}/{args.iters} best_f={best_f:.3f} st={best_st}")

    best = T.AdasCortexOrgan.deserialize(organ0.serialize())
    apply_params(best, best_x, n_cells)
    # 全量验证
    full_c, full_ok, detail = T.evaluate(best, T.SCENARIOS, noise_seed=21)
    st_c, st_ok = stadium_cost(best, duration=25.0, seed=21)
    metrics = {
        "full_train_cost": full_c,
        "full_train_ok": full_ok,
        "stadium_cost": st_c,
        "stadium_ok": st_ok,
        "history": hist[-20:],
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_bin(args.out, best, args.iters, metrics, {
        "note": "stadium-domain sep-CMA; L3 lock untouched",
        "in": os.path.relpath(args.inp, ROOT),
    })
    report = os.path.join(ROOT, "runs", "adas_stadium_tune_20260914.json")
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"out": args.out, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    print("wrote", args.out)
    print("report", report, metrics)


if __name__ == "__main__":
    main()
