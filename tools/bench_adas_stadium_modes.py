#!/usr/bin/env python3
"""体育场公路：导师 / 协同 / 学生 三模式 CTE 对照（走 8833 真后端）。

预注册成功线（稳定性战役 #1）：student 模式 p95 CTE < 0.5 m，且无频繁离轨重置。

用法:
  # 需已启动: python3 tools/cellular_live_backend.py
  python3 tools/bench_adas_stadium_modes.py
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8833"


def api(path: str):
    with urllib.request.urlopen(BASE + path, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def sample_mode(mode: str, seconds: float = 8.0, period: float = 0.05):
    api(f"/api/vehicle/set_mode?mode={mode}")
    api("/api/vehicle/reset")
    time.sleep(0.3)
    ctes = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        d = api("/api/vehicle/status")
        c = d.get("car") or {}
        ctes.append(float(c.get("cte_m") or 0.0))
        time.sleep(period)
    ctes.sort()
    n = len(ctes)
    p95 = ctes[min(n - 1, int(0.95 * n))] if n else None
    return {
        "mode": mode,
        "n": n,
        "mean": round(statistics.fmean(ctes), 4) if ctes else None,
        "max": round(max(ctes), 4) if ctes else None,
        "p95": round(p95, 4) if p95 is not None else None,
        "loop": (api("/api/vehicle/status").get("control_loop")),
        "pass_p95_0_5": bool(p95 is not None and p95 < 0.5),
    }


def main():
    try:
        api("/api/vehicle/status")
    except Exception as e:
        raise SystemExit(f"后端未就绪 ({e})；先启动 python3 tools/cellular_live_backend.py")

    rows = []
    for mode in ("teacher", "co_driver", "student"):
        rows.append(sample_mode(mode))
        print(json.dumps(rows[-1], ensure_ascii=False))

    out = {
        "protocol": "stadium_live_cte_8s",
        "criterion": "student p95 CTE < 0.5 m",
        "rows": rows,
        "student_pass": next((r["pass_p95_0_5"] for r in rows if r["mode"] == "student"), False),
    }
    path = "runs/adas_stadium_modes_cte_20260914.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote", path, "student_pass=", out["student_pass"])


if __name__ == "__main__":
    main()
