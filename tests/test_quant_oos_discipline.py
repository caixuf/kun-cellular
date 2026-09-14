#!/usr/bin/env python3
"""量化 OOS 纪律门禁：冻结负结果，禁止用选择集夏普冒充样本外。

稳定性战役 #5。证据文件：
  checkpoints/quant_cortical_array_oos_repro_20260911.json

判据（预注册，负结果锁档）：
  Q2 FAIL：OOS 年化夏普 < 0
  过拟合签名：val.sharpe > 1.0 且 oos.sharpe < 0
若未来真正翻正，必须换新报告文件并改写本测试——不得改这个 JSON 里的数字装 PASS。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "checkpoints", "quant_cortical_array_oos_repro_20260911.json")


def test_quant_oos_frozen_negative():
    assert os.path.isfile(REPORT), f"missing frozen OOS report: {REPORT}"
    with open(REPORT, encoding="utf-8") as f:
        d = json.load(f)
    oos = d["oos"]
    val = d["val"]
    sharpe = float(oos["sharpe"])
    val_s = float(val["sharpe"])
    assert sharpe < 0.0, (
        f"OOS sharpe={sharpe} is not negative; if this is a real turnaround, "
        "register a NEW report file and rewrite this gate — do not edit the frozen JSON"
    )
    assert val_s > 1.0 and sharpe < 0.0, "overfit signature missing (val boom / OOS bust)"
    paper = d.get("paper_anchor") or {}
    # 论文锚点 0.22 不得被本文件改写成“已复现”
    assert abs(float(paper.get("sharpe", 0.22)) - 0.22) < 1e-6
    assert abs(sharpe - (-0.031879)) < 1e-4


def test_quant_oos_not_claimed_gate_true():
    """复训报告不得带 gate:true（该文件没有 gate 字段；有则必须 false）。"""
    with open(REPORT, encoding="utf-8") as f:
        d = json.load(f)
    assert d.get("gate") in (None, False, 0)


if __name__ == "__main__":
    test_quant_oos_frozen_negative()
    test_quant_oos_not_claimed_gate_true()
    print("PASS  OOS frozen FAIL (sharpe=-0.03) locked")
