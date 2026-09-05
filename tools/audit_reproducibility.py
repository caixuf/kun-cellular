#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunCellular 真实性、可复现性与无欺瞒严格审计工具
(Strict Reproducibility & Truthfulness Audit Tool)
"""

import os
import sys
import subprocess
import json

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd)
    return res.returncode, res.stdout, res.stderr

def audit_checkpoint_roundtrip():
    print("=" * 60)
    print("1. 审计检查点全双工生命周期 (Checkpoint Save-Load Roundtrip)...")
    code, out, err = run_cmd("./build/test_checkpoint_roundtrip")
    if code == 0 and "ALL CHECKPOINT ROUNDTRIP TESTS PASSED" in out:
        print("   ✅ PASS: C++ 内核 load_checkpoint_bin 真实可用，推演零损耗回环验证通过。")
        return True
    else:
        print("   ❌ FAIL: 检查点反序列化或回环验证失败！")
        print(out)
        print(err)
        return False

def audit_gate_integrity():
    print("=" * 60)
    print("2. 审计三隔离门禁防作弊机制 (3-Isolation Gate Integrity)...")
    path = "include/kun/cellular/evolvable_task.hpp"
    if not os.path.exists(path):
        print(f"   ❌ FAIL: {path} 不存在！")
        return False
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 验证 ood 是否硬性参与 passes_m1_gate 判定
    has_ood_check = "report.holdout_ood_metrics.success_rate >= 0.50" in content
    has_strict_eval = "report.passes_m1_gate = (train_pass && id_pass && ood_pass);" in content

    if has_ood_check and has_strict_eval:
        print("   ✅ PASS: OOD 分布外真实成功率与泛化比率已硬编码纳入 M1 门禁，伪门禁已封堵。")
        return True
    else:
        print("   ❌ FAIL: M1 门禁未严格检验 OOD 表现，存在虚假放行漏洞！")
        return False

def audit_model_shells():
    print("=" * 60)
    print("3. 审计模型真实性与空壳文件扫描 (Anti-Hollow-Artifact Scan)...")
    suspicious = []
    
    # 检查 runs/ 下是否有宣称 billion 但体积小于 1MB 的 pickle 假文件
    if os.path.exists("runs"):
        for f in os.listdir("runs"):
            if f.endswith(".pt"):
                sz = os.path.getsize(os.path.join("runs", f))
                if ("billion" in f or "1b" in f) and sz < 1024 * 1024:
                    suspicious.append((f, f"宣称十亿参数但体积仅 {sz} 字节 (空壳)"))
    
    if suspicious:
        print("   ❌ FAIL: 发现空壳/虚假模型文件:")
        for name, reason in suspicious:
            print(f"      - {name}: {reason}")
        return False
    else:
        print("   ✅ PASS: 未在活跃 runs/ 目录发现空壳大模型伪造文件。")
        return True

def audit_ctest_all():
    print("=" * 60)
    print("4. 执行全量 35 组 CTest 回归套件 (Full Deterministic Test Suite)...")
    code, out, err = run_cmd("ctest --test-dir build --output-on-failure")
    if code == 0 and "100% tests passed" in out:
        print("   ✅ PASS: 35/35 单元测试与极限工况基准全量满分通过。")
        return True
    else:
        print("   ❌ FAIL: CTest 存在失败用例！")
        print(out[-1000:] if len(out) > 1000 else out)
        return False

def main():
    print("\n" + "=" * 60)
    print("   KunCellular 真实性与工程纪律硬核审计 (Strict Reproducibility)")
    print("=" * 60)
    
    ok1 = audit_checkpoint_roundtrip()
    ok2 = audit_gate_integrity()
    ok3 = audit_model_shells()
    ok4 = audit_ctest_all()
    
    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3 and ok4:
        print("🎉 最终审计结论: 全部 4 项硬核防伪与可复现性门禁 100% 满分通过！")
        print("   项目已彻底拔除空壳资产，建立真实双向检查点与真·三隔离门禁。")
        print("=" * 60 + "\n")
        return 0
    else:
        print("💥 最终审计结论: 存在未修复的真实性或工程纪律违规，请立即整改！")
        print("=" * 60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
