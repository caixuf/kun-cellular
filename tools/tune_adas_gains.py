#!/usr/bin/env python3
"""L3 — 结构/参数分离：固定已演化拓扑，用 sep-CMA-ES 只调连续参数。

论文 §7.3 L3 的第一次落地。演化（train_adas_cortex.py）负责搜拓扑：
隐藏细胞类型 + 突触连接图；本工具接手其冠军，**不改任何一条边、任何一个
细胞类型**，仅优化：

  * 每个细胞的增益 gain（对数空间，保证正）
  * 每条突触的权重 w（线性空间）

适应度与训练器完全相同（`evaluate` 全 12 场景代价和，代际轮换噪声种子），
所以任何改进都来自参数调优本身，不来自更换目标。

这是任务层工具：不触碰 include/kun/cellular/。

用法：
  python3 tools/tune_adas_gains.py --in checkpoints/adas_cortex_champion.bin \
      --out checkpoints/adas_cortex_champion_tuned.bin --iters 200
"""
import argparse
import json
import math
import os
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
from adas_eval_native import NativeEvaluator  # noqa: E402

def apply_params(organ, x, n_cells):
    for i, c in enumerate(organ.cells):
        c.gain = math.exp(x[i])
    ws = x[n_cells:]
    organ.synapses = [(f, t, float(w)) for (f, t, _), w in zip(organ.synapses, ws)]
    organ.compile_incoming()


def params_from_organ(organ):
    g = np.log(np.array([max(c.gain, 1e-3) for c in organ.cells]))
    w = np.array([w for _, _, w in organ.synapses], dtype=float)
    return np.concatenate([g, w])


def table(organ, seeds):
    """训练 + 验证全场景 avg_cte 的 mean±std。"""
    out = {}
    for split, scn in (("train", T.SCENARIOS), ("val", T.VAL_SCENARIOS)):
        for name, path, spd, v0, dur, lead in scn:
            v = [T.run_scenario(organ, path, spd, v0, dur, lead_on=lead, seed=s)[2]
                 for s in seeds]  # 报表用 Python 参考实现，与原生位级一致
            out[name] = {
                "split": split,
                "avg_cte_mean": float(np.mean([m["avg_cte"] for m in v])),
                "avg_cte_std": float(np.std([m["avg_cte"] for m in v], ddof=1)),
                "max_cte_mean": float(np.mean([m["max_cte"] for m in v])),
                "ok_all": all(m["steps"] == m["total"] for m in v),
            }
    return out


def write_bin(out_path, organ, generation, metrics, note):
    ser = organ.serialize()
    n_rec, n_mot = len(T.RECEPTOR_TYPES), len(T.MOTOR_TYPES)
    n_cells, n_syn = len(organ.cells), len(organ.synapses)
    meta = json.dumps({
        "organism_id": "adas_cortex_champion_tuned",
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

    hdr = 72
    cells_off = hdr
    rp_off = cells_off + n_cells * 4
    ci_off = rp_off + (n_cells + 1) * 4
    w_off = ci_off + n_syn * 4
    co_off = w_off + n_syn * 4
    header = struct.pack("<IIIIIIQQQQQQ", 0x53445343, 2, n_cells, n_syn, 6, 2,
                         cells_off, rp_off, ci_off, w_off, co_off,
                         (generation & 0xFFFFFFFF) | ((len(meta) & 0xFFFFFFFF) << 32))
    cell_bytes = bytearray(n_cells * 4)
    for i, c in enumerate(organ.cells):
        cell_bytes[i * 4] = 4
        cell_bytes[i * 4 + 1] = min(255, max(0, int(c.gain * 64.0)))
        flags = (0x01 if i < n_rec else 0) | (0x02 if i >= n_cells - n_mot else 0)
        cell_bytes[i * 4 + 3] = flags
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(cell_bytes)
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())
        f.write(np.array(weights, dtype=np.float32).tobytes())
        f.write(np.zeros((n_cells, 3), dtype=np.float32).tobytes())
        f.write(meta)


class SepCMAES:
    """对角协方差 CMA-ES（Ros & Hansen 2008），适合几百维、评估昂贵的场景。"""

    def __init__(self, x0, sigma0, lam, rng):
        self.n = len(x0)
        self.m = np.array(x0, dtype=float)
        self.sigma = sigma0
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        n = self.n
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        # sep-CMA: 学习率按 (n+2)/3 放大
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff) * (n + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff)
                       / ((n + 2) ** 2 + self.mueff) * (n + 2) / 3)
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.ps = np.zeros(n)
        self.pc = np.zeros(n)
        self.C = np.ones(n)
        self.rng = rng
        self.gen = 0

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.n))
        self.y = self.z * np.sqrt(self.C)
        return self.m + self.sigma * self.y

    def tell(self, fit):
        idx = np.argsort(fit)[: self.mu]
        yw = np.sum(self.w[:, None] * self.y[idx], axis=0)
        zw = np.sum(self.w[:, None] * self.z[idx], axis=0)
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        self.gen += 1
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                / self.chin) < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = np.sum(self.w[:, None] * self.y[idx] ** 2, axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chin - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin"))
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "adas_cortex_champion_tuned.bin"))
    ap.add_argument("--report", default=os.path.join(ROOT, "runs", "adas_gain_tuning_report.json"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--eval-seeds", type=int, default=2, help="每个候选平均几个噪声种子")
    args = ap.parse_args()

    ck = load_cortex_from_bin(args.inp)
    organ_ser = ck["organ"]
    organ = T.AdasCortexOrgan.deserialize(organ_ser)
    n_cells = len(organ.cells)
    x0 = params_from_organ(organ)
    print("=" * 72)
    print("  L3 结构/参数分离调参 (sep-CMA-ES)  —— 拓扑冻结, 仅调 gain + weight")
    print(f"  输入: {args.inp}")
    print(f"  细胞={n_cells} 突触={len(organ.synapses)} 参数维度={len(x0)}")
    print(f"  pop={args.pop} iters={args.iters} sigma0={args.sigma0}")
    print("=" * 72)

    print("[before] 10 种子 avg_cte (m):")
    before = table(organ, range(1, 11))
    for k, v in before.items():
        print(f"  {k:16s} {v['avg_cte_mean']:.3f} ± {v['avg_cte_std']:.3f}  max {v['max_cte_mean']:.3f}")

    rng = np.random.default_rng(args.seed)
    es = SepCMAES(x0, args.sigma0, args.pop, rng)
    ev = NativeEvaluator(T).bind(organ)
    print(f"  原生评估器: {ev.threads} 线程, 与 Python 位级一致 (tools/adas_eval_native.py)")
    best_x, best_f = x0.copy(), float("inf")
    t0 = time.time()
    hist = []

    def batch(cands, seeds):
        X = np.asarray(cands)
        G = np.exp(X[:, :n_cells])
        W = X[:, n_cells:]
        return ev.evaluate_batch(G, W, seeds)

    for it in range(1, args.iters + 1):
        seeds = [args.seed + it * 7919 + k * 104729 for k in range(args.eval_seeds)]
        X = es.ask()
        # 精英与当前均值一并用当代种子复评，避免旧种子上的偶然好成绩霸位
        cand = list(X) + [best_x, es.m.copy()]
        fit = batch(cand, seeds)
        es.tell(np.array(fit[: args.pop]))
        j = int(np.argmin(fit))
        if fit[j] < fit[args.pop]:      # 比复评后的精英更好才替换
            best_x, best_f = cand[j].copy(), float(fit[j])
        else:
            best_f = float(fit[args.pop])
        hist.append({"iter": it, "best": best_f, "gen_min": float(min(fit[: args.pop])),
                     "sigma": es.sigma})
        if it % 10 == 0 or it == 1:
            print(f"  iter {it:4d}  best {best_f:8.2f}  gen_min {min(fit[:args.pop]):8.2f}"
                  f"  sigma {es.sigma:.4f}  {time.time() - t0:6.1f}s", flush=True)

    apply_params(organ, best_x, n_cells)
    print("[after] 10 种子 avg_cte (m):")
    after = table(organ, range(1, 11))
    for k, v in after.items():
        b = before[k]["avg_cte_mean"]
        a = v["avg_cte_mean"]
        print(f"  {k:16s} {a:.3f} ± {v['avg_cte_std']:.3f}  max {v['max_cte_mean']:.3f}"
              f"   ({b:.3f} → {a:.3f}, x{b / max(a, 1e-9):.2f})")

    note = {"method": "sep-CMA-ES", "iters": args.iters, "pop": args.pop,
            "sigma0": args.sigma0, "seed": args.seed, "eval_seeds": args.eval_seeds,
            "params": int(len(x0)), "topology_frozen": True, "source": os.path.relpath(args.inp, ROOT)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_bin(args.out, organ, ck.get("generation", 0) if isinstance(ck, dict) else 0,
              {k: {kk: vv for kk, vv in v.items()} for k, v in after.items()}, note)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"note": note, "before": before, "after": after, "history": hist}, f, indent=1)
    print(f"[存盘] {args.out}")
    print(f"[报告] {args.report}")


if __name__ == "__main__":
    main()
