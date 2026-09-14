#!/usr/bin/env python3
"""ADAS 拓扑因果消融：同度数重连 + E/I 符号翻转。

论文 §6.13 预注册门禁（稳定性战役 #4）：
  锁档冠军若能力来自拓扑先验，则
    (A) 同度数随机重连应显著变差；
    (B) 全突触符号翻转应显著变差。
  显著 = 验证+训练全场景 cost 比值 ≥ 1.3 或 all_ok 从 True 掉到 False。
  不显著则如实记 NEGATIVE，不得宣称「拓扑即能力已证」。

不改 include/kun/cellular/；不覆盖锁档 bin。

用法:
  python3 tools/ablate_topology_prior.py
  python3 tools/ablate_topology_prior.py --bin checkpoints/adas_cortex_champion.bin --rewires 8
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import types

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


def clone_organ(organ):
    return T.AdasCortexOrgan.deserialize(organ.serialize())


def apply_synapses(organ, synapses):
    organ.synapses = [(int(f), int(t), float(w)) for f, t, w in synapses]
    organ.compile_incoming()


def directed_degrees(synapses, n_cells):
    out_d = [0] * n_cells
    in_d = [0] * n_cells
    for f, t, _ in synapses:
        if 0 <= f < n_cells and 0 <= t < n_cells:
            out_d[f] += 1
            in_d[t] += 1
    return out_d, in_d


def degree_preserving_rewire(synapses, n_cells, n_receptors, rng, n_swaps=None):
    """Maslov–Sneppen 有向边交换：保持每个节点入/出度，禁止指向感受器。"""
    edges = [(int(f), int(t), float(w)) for f, t, w in synapses
             if 0 <= f < n_cells and n_receptors <= t < n_cells]
    if len(edges) < 2:
        return edges
    existing = {(f, t) for f, t, _ in edges}
    n_swaps = n_swaps or max(200, 12 * len(edges))
    m = len(edges)
    accepted = 0
    attempts = 0
    while accepted < n_swaps and attempts < n_swaps * 20:
        attempts += 1
        i, j = rng.randrange(m), rng.randrange(m)
        if i == j:
            continue
        a, b, wa = edges[i]
        c, d, wc = edges[j]
        # 交换目标：a→d, c→b
        if a == d or c == b or b < n_receptors or d < n_receptors:
            continue
        if (a, d) in existing or (c, b) in existing:
            continue
        existing.discard((a, b))
        existing.discard((c, d))
        edges[i] = (a, d, wa)
        edges[j] = (c, b, wc)
        existing.add((a, d))
        existing.add((c, b))
        accepted += 1
    return edges, {"swaps": accepted, "attempts": attempts}


def flip_signs(synapses):
    return [(f, t, -w) for f, t, w in synapses]


def summarize_detail(detail):
    rows = {}
    for name, m in detail.items():
        if not isinstance(m, dict):
            continue
        rows[name] = {
            "ok": bool(m.get("steps") == m.get("total")),
            "avg_cte": round(float(m.get("avg_cte", 0.0)), 4),
            "max_cte": round(float(m.get("max_cte", 0.0)), 4),
        }
    return rows


def eval_pack(organ, noise_seed):
    scn = list(T.SCENARIOS) + list(T.VAL_SCENARIOS)
    cost, ok, detail = T.evaluate(organ, scn, noise_seed=noise_seed)
    # 去掉突触数惩罚以便消融公平（边数不变）
    cost_raw = cost - len(organ.synapses) * 0.005
    return {
        "cost": round(cost_raw, 4),
        "all_ok": bool(ok),
        "scenes": summarize_detail(detail),
    }


def ratio_pass(lock_cost, other_cost, lock_ok, other_ok, thresh=1.3):
    if lock_cost <= 0:
        return False, 0.0
    r = other_cost / lock_cost
    degraded = (r >= thresh) or (lock_ok and not other_ok)
    return degraded, round(r, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", action="append", default=[],
                    help="可重复；默认 L3 锁档 + stadium 适配档")
    ap.add_argument("--rewires", type=int, default=8)
    ap.add_argument("--noise-seed", type=int, default=21)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs",
                    "adas_topology_ablation_20260914.json"))
    args = ap.parse_args()
    bins = args.bin or [
        os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin"),
        os.path.join(ROOT, "checkpoints", "adas_cortex_champion_stadium.bin"),
    ]

    t0 = time.time()
    reports = []
    for bin_path in bins:
        if not os.path.exists(bin_path):
            reports.append({"bin": bin_path, "status": "missing"})
            continue
        organ = T.AdasCortexOrgan.deserialize(load_cortex_from_bin(bin_path)["organ"])
        n_cells = len(organ.cells)
        n_rec = organ.n_receptors
        out0, in0 = directed_degrees(organ.synapses, n_cells)
        lock = eval_pack(organ, args.noise_seed)

        rewire_rows = []
        for k in range(args.rewires):
            syn, meta = degree_preserving_rewire(
                organ.synapses, n_cells, n_rec, random.Random(20260914 + k * 97))
            out1, in1 = directed_degrees(syn, n_cells)
            assert out0 == out1 and in0 == in1, "degree not preserved"
            child = clone_organ(organ)
            apply_synapses(child, syn)
            row = eval_pack(child, args.noise_seed)
            row.update({"k": k, **meta})
            rewire_rows.append(row)

        ei_child = clone_organ(organ)
        apply_synapses(ei_child, flip_signs(organ.synapses))
        ei = eval_pack(ei_child, args.noise_seed)

        rw_costs = [r["cost"] for r in rewire_rows]
        rw_mean = sum(rw_costs) / len(rw_costs)
        rw_ok_rate = sum(1 for r in rewire_rows if r["all_ok"]) / len(rewire_rows)
        a_ok, a_ratio = ratio_pass(lock["cost"], rw_mean, lock["all_ok"], rw_ok_rate >= 0.99)
        b_ok, b_ratio = ratio_pass(lock["cost"], ei["cost"], lock["all_ok"], ei["all_ok"])

        reports.append({
            "bin": os.path.relpath(bin_path, ROOT),
            "n_cells": n_cells,
            "n_synapses": len(organ.synapses),
            "lock": lock,
            "rewire": {
                "n": len(rewire_rows),
                "cost_mean": round(rw_mean, 4),
                "cost_min": round(min(rw_costs), 4),
                "cost_max": round(max(rw_costs), 4),
                "ok_rate": round(rw_ok_rate, 3),
                "ratio_vs_lock": a_ratio,
                "degraded": a_ok,
                "rows": rewire_rows,
            },
            "ei_flip": {**ei, "ratio_vs_lock": b_ratio, "degraded": b_ok},
            "gate": {
                "A_rewire": a_ok,
                "B_ei_flip": b_ok,
                "pass": bool(a_ok and b_ok),
            },
        })

    overall = all(r.get("gate", {}).get("pass") for r in reports if "gate" in r)
    out = {
        "protocol": "degree_preserving_rewire + full_sign_flip",
        "criterion": "cost_ratio>=1.3 or all_ok True→False",
        "overall_pass": overall,
        "elapsed_s": round(time.time() - t0, 2),
        "reports": reports,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=" * 64)
    print("  ADAS 拓扑先验消融  (rewire / E-I flip)")
    print("=" * 64)
    for r in reports:
        if "gate" not in r:
            print(f"  SKIP {r['bin']}")
            continue
        print(f"  {r['bin']}")
        print(f"    lock     cost={r['lock']['cost']:.3f}  ok={r['lock']['all_ok']}")
        print(f"    rewire   mean={r['rewire']['cost_mean']:.3f}  "
              f"ok_rate={r['rewire']['ok_rate']:.2f}  "
              f"ratio={r['rewire']['ratio_vs_lock']}  "
              f"{'DEGRADE' if r['rewire']['degraded'] else 'NO-EFFECT'}")
        print(f"    ei_flip  cost={r['ei_flip']['cost']:.3f}  "
              f"ok={r['ei_flip']['all_ok']}  "
              f"ratio={r['ei_flip']['ratio_vs_lock']}  "
              f"{'DEGRADE' if r['ei_flip']['degraded'] else 'NO-EFFECT'}")
        print(f"    gate     {'PASS' if r['gate']['pass'] else 'FAIL'}")
    print("-" * 64)
    print(f"  overall={'PASS' if overall else 'NEGATIVE'}  wrote {args.out}")
    print("=" * 64)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
