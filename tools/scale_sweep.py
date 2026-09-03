#!/usr/bin/env python3
"""
细胞规模扫描：更多细胞 = 开得更好吗？
=====================================
控制变量（同代数 / 同种群 / 同随机种子 / 同 5 场景闭环），只变隐藏细胞数，
测量：闭环驾驶代价、各场景 CTE、演化耗时、C 侧单帧推理延迟。

用途：回答"要不要上百万细胞"。结论必须来自实测而非直觉。
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from train_adas_cortex import AdasCortexOrgan, evaluate, SCENARIOS  # noqa: E402

LATENCY_C = r"""
#include <stdio.h>
#include <time.h>
#include "sdsc_cortex.h"
int main(void) {
    SdscCortex ctx; sdsc_cortex_init_default_adas(&ctx);
    float in[6] = {0.3f,-0.2f,0.1f,0.6f,0.2f,0.0f}, out[2];
    for (int i=0;i<2000;++i) sdsc_cortex_forward(&ctx,in,out);   /* warmup */
    const int N = 200000;
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    for (int i=0;i<N;++i) { in[0] = (float)(i%100)/100.0f; sdsc_cortex_forward(&ctx,in,out); }
    clock_gettime(CLOCK_MONOTONIC,&b);
    double ns = ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec)) / (double)N;
    printf("%.1f\n", ns);
    return 0;
}
"""


def evolve(hidden, gens, pop, seed):
    random.seed(seed)
    population = [AdasCortexOrgan(n_hidden=hidden) for _ in range(pop)]
    best, best_cost, best_detail = None, float("inf"), None
    t0 = time.time()
    for _ in range(gens):
        scored = sorted(((evaluate(o), o) for o in population), key=lambda r: r[0][0])
        (c, ok, d), o = scored[0]
        if c < best_cost:
            best_cost, best, best_detail = c, o, d
        survivors = [r[1] for r in scored[: max(2, pop // 4)]]
        population = [best] + survivors
        while len(population) < pop:
            population.append(random.choice(survivors).mutate())
    return best, best_cost, best_detail, time.time() - t0


def measure_latency(organ, cost):
    """导出为 C11 并实测单帧推理延迟（ns）。"""
    ck = {"trainer": "scale_sweep", "champion_cost": cost,
          "all_scenarios_passed": True, "metrics": {}, "organ": organ.serialize()}
    with tempfile.TemporaryDirectory() as td:
        ckp = os.path.join(td, "ck.json")
        with open(ckp, "w", encoding="utf-8") as f:
            json.dump(ck, f)
        inc = os.path.join(td, "inc")
        os.makedirs(inc)
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "export_sdsc_cortex.py"),
             "--checkpoint", ckp, "--targets", os.path.join(inc, "sdsc_cortex.h")],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None, f"export failed: {r.stderr.strip()[:200]}"
        src, exe = os.path.join(td, "l.c"), os.path.join(td, "l")
        with open(src, "w", encoding="utf-8") as f:
            f.write(LATENCY_C)
        cc = subprocess.run(["cc", "-std=c11", "-O2", "-I", inc, src, "-o", exe, "-lm"],
                            capture_output=True, text=True)
        if cc.returncode != 0:
            return None, f"cc failed: {cc.stderr.strip()[:200]}"
        run = subprocess.run([exe], capture_output=True, text=True)
        hdr_kb = os.path.getsize(os.path.join(inc, "sdsc_cortex.h")) / 1024.0
        return (float(run.stdout.strip()), hdr_kb), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[24, 48, 192, 768])
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    print("=" * 96)
    print("  细胞规模扫描（控制变量：同代数/同种群/同种子，只变隐藏细胞数）")
    print(f"  代数={args.generations} 种群={args.pop} 种子={args.seed} 场景={len(SCENARIOS)}")
    print("=" * 96)
    print(f"{'隐藏':>6} {'细胞':>6} {'突触':>6} {'代价':>9} {'直道CTE':>9} {'S弯CTE':>9} "
          f"{'弯道CTE':>9} {'速度误差':>9} {'演化耗时':>9} {'推理延迟':>10} {'头文件':>9}")
    print("-" * 96)

    rows = []
    for h in args.hidden:
        organ, cost, detail, secs = evolve(h, args.generations, args.pop, args.seed)
        lat, err = measure_latency(organ, cost)
        lat_s = f"{lat[0]:.0f} ns" if lat else (err or "n/a")[:10]
        hdr_s = f"{lat[1]:.0f} KB" if lat else "-"
        rows.append({
            "hidden": h, "cells": len(organ.cells), "synapses": len(organ.synapses),
            "cost": cost, "evolve_seconds": secs,
            "latency_ns": lat[0] if lat else None,
            "header_kb": lat[1] if lat else None,
            "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                        for k, v in detail.items()},
        })
        print(f"{h:>6} {len(organ.cells):>6} {len(organ.synapses):>6} {cost:>9.3f} "
              f"{detail['straight_cruise']['avg_cte']*100:>8.2f}cm "
              f"{detail['s_curve']['avg_cte']*100:>8.2f}cm "
              f"{detail['tight_curve']['avg_cte']*100:>8.2f}cm "
              f"{detail['stop_go']['avg_verr']:>8.2f}  "
              f"{secs:>8.1f}s {lat_s:>10} {hdr_s:>9}")

    print("-" * 96)
    best = min(rows, key=lambda r: r["cost"])
    print(f"  最优规模: hidden={best['hidden']} ({best['cells']} 细胞) cost={best['cost']:.3f}")
    worst = max(rows, key=lambda r: r["cost"])
    print(f"  最差规模: hidden={worst['hidden']} ({worst['cells']} 细胞) cost={worst['cost']:.3f}")
    print("=" * 96)

    out = os.path.join(ROOT, "runs", "scale_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": rows}, f, indent=2)
    print(f"[结果已存盘] -> {out}")


if __name__ == "__main__":
    main()
