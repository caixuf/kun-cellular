#!/usr/bin/env python3
"""bench_adas_vs_stanley_seeds — 同一 ADAS 冠军 × 多噪声种子 vs Stanley（评估方差锁档）。

用法:
  python3 tools/bench_adas_vs_stanley_seeds.py
  python3 tools/bench_adas_vs_stanley_seeds.py --seeds 1-10 --out runs/adas_champion_vs_stanley_seeds1-10.json

口径: 评估噪声方差，不是独立训练方差（见 docs/superpowers/plans/2026-09-11-adas-multiseed-lock.md）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def load_trainer():
    path = os.path.join(ROOT, "tools", "train_adas_cortex.py")
    T = types.ModuleType("adas_trainer")
    T.__file__ = path
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), T.__dict__)
    return T


class Stanley:
    """与 test_adas_cortex_contract.check_env_trackable 同构的几何基线。"""

    def __init__(self, T):
        self.T = T

    def reset_state(self):
        pass

    def forward(self, cte_n, dpsi_n, kap_n, v_n, verr_n, danger_n):
        T = self.T
        cte = cte_n * 2.0
        dpsi = dpsi_n * 0.5
        kap = kap_n / 20.0
        v = max(v_n * T.MAX_SPEED, 1.0)
        delta = (dpsi + math.atan(2.0 * cte / v) + math.atan(kap * T.WHEELBASE))
        lim = T.adaptive_steer_limit(v, cte)
        return (max(-1.0, min(1.0, delta / lim)),
                max(-1.0, min(1.0, verr_n * 1.5)))


def parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin"))
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs",
                    "adas_champion_vs_stanley_seeds1-10_repro.json"),
                    help="复现输出；勿默认覆盖论文引用的历史 seeds1-10.json")
    ap.add_argument("--baseline", default=os.path.join(ROOT, "runs",
                    "adas_champion_vs_stanley_seeds1-10.json"),
                    help="漂移对照（论文/README 历史表）")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    from export_sdsc_cortex import load_cortex_from_bin

    T = load_trainer()
    organ = T.AdasCortexOrgan.deserialize(load_cortex_from_bin(args.bin)["organ"])
    stanley = Stanley(T)

    # 跑前读旧基线（若 out 将覆盖，先读）
    old = None
    if os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            old = json.load(f)

    report = {}
    wins = losses = ties = 0
    for split, scn in (("train", T.SCENARIOS), ("val", T.VAL_SCENARIOS)):
        for name, path, spd, v0, dur, lead in scn:
            c_vals, s_vals, ok_c, ok_s = [], [], True, True
            for s in seeds:
                _, ok1, m1 = T.run_scenario(organ, path, spd, v0, dur, lead_on=lead, seed=s)
                _, ok2, m2 = T.run_scenario(stanley, path, spd, v0, dur, lead_on=lead, seed=s)
                c_vals.append(m1["avg_cte"])
                s_vals.append(m2["avg_cte"])
                ok_c = ok_c and ok1 and (m1["steps"] == m1["total"])
                ok_s = ok_s and ok2 and (m2["steps"] == m2["total"])
            cm, sm = float(np.mean(c_vals)), float(np.mean(s_vals))
            report[name] = {
                "split": split,
                "champ_mean": cm,
                "champ_std": float(np.std(c_vals, ddof=1)),
                "stanley_mean": sm,
                "stanley_std": float(np.std(s_vals, ddof=1)),
                "ok_all_champ": ok_c,
                "ok_all_stanley": ok_s,
                "seeds": seeds,
            }
            if cm < sm - 1e-12:
                wins += 1
                tag = "W"
            elif sm < cm - 1e-12:
                losses += 1
                tag = "L"
            else:
                ties += 1
                tag = "="
            print(f"  {tag} {name:16s} champ {cm:.4f}±{report[name]['champ_std']:.4f}  "
                  f"stanley {sm:.4f}±{report[name]['stanley_std']:.4f}  "
                  f"ok={ok_c and ok_s}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
        f.write("\n")

    # 门禁
    a1 = all(v["ok_all_champ"] and v["ok_all_stanley"] for v in report.values())
    a2 = wins in (6, 7, 8)
    sc = report["straight_cruise"]["champ_mean"]
    a3 = 0.028 <= sc <= 0.045
    a4 = True
    max_drift = 0.0
    if old:
        for name, v in report.items():
            if name not in old:
                continue
            d = abs(v["champ_mean"] - float(old[name]["champ_mean"]))
            max_drift = max(max_drift, d)
            if d > 0.02:
                a4 = False
    else:
        a4 = True  # 无旧文件时跳过漂移门，仍报

    print("---------------------------------------------------------------------")
    print(f"  战绩: {wins}W / {losses}L / {ties}T  (seeds={seeds[0]}..{seeds[-1]})")
    print(f"  straight_cruise champ_mean={sc:.6f}")
    print(f"  max |Δchamp| vs baseline={max_drift:.6f}")
    print(f"  A1完赛={a1} A2胜场带={a2}({wins}) A3直道={a3} A4漂移={a4}")
    print(f"  JSON {{\"wins\":{wins},\"losses\":{losses},\"ties\":{ties},"
          f"\"straight_cruise\":{sc:.6f},\"max_drift\":{max_drift:.6f},"
          f"\"a1\":{str(a1).lower()},\"a2\":{str(a2).lower()},"
          f"\"a3\":{str(a3).lower()},\"a4\":{str(a4).lower()}}}")
    print(f"  已写: {args.out}")
    ok = a1 and a2 and a3 and a4
    if not ok:
        print("  [NEGATIVE] T3 门禁未全过")
        return 2
    print("  [PASS] T3 多种子评估锁档")
    return 0


if __name__ == "__main__":
    sys.exit(main())
