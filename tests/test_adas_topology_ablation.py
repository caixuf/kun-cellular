#!/usr/bin/env python3
"""门禁：ADAS 锁档必须显著优于同度数重连与 E/I 翻转（稳定性战役 #4）。"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_topology_prior_degrades_on_rewire_and_sign_flip():
    script = os.path.join(ROOT, "tools", "ablate_topology_prior.py")
    lock = os.path.join(ROOT, "checkpoints", "adas_cortex_champion.bin")
    out = os.path.join(ROOT, "runs", "adas_topology_ablation_ci.json")
    cmd = [sys.executable, script, "--bin", lock, "--rewires", "2", "--out", out]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert "overall=PASS" in r.stdout


if __name__ == "__main__":
    test_topology_prior_degrades_on_rewire_and_sign_flip()
    print("PASS")
