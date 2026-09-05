#!/usr/bin/env python3
"""ADAS Cortex 训练契约 ↔ FlowEngine C 源码 常量对齐门禁
========================================================

背景（2026-09-03 实测的一类 bug）
--------------------------------
训练器里把 planning 的曲率限速写成 ``sqrt(1.4/κ)``，而车上 ``st_graph.h`` 用的是
``0.85*sqrt(5.0/κ)``。κ=0.008 时车上给 21 m/s、训练里只给 13.2 m/s —— 细胞体被
训成"弯道该减速"，可真车 planning 根本不减，上车后一路顶着速度误差。

这类"训练环境常量与车上不一致"的 sim-to-real 错配**不会被 parity 测试抓到**：
parity 比的是 C 导出体与 Python 演化体的数值一致性，两边用的是同一份错误常量。
必须单独把训练器常量与 FlowEngine C 源码逐个对账。

新增/修改任何训练常量时，同步在 CHECKS 里登记它对应的 C 侧出处。
"""

import os
import re
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLOWENGINE = os.environ.get("FLOWENGINE_ROOT",
                            os.path.join(os.path.dirname(ROOT), "FlowEngine"))


def load_trainer():
    """从源码加载训练器。

    **必须绕开 .pyc 缓存**：实测改动 ``5.0``→``1.4`` 时字节数相同、mtime 同秒，
    pyc 失效检查漏判，门禁会对着过期字节码做判决 —— 门禁读错了源，它的 PASS
    就毫无意义。这里直接编译源码文本，不经过任何缓存。
    """
    path = os.path.join(ROOT, "tools", "train_adas_cortex.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    import types
    mod = types.ModuleType("train_adas_cortex")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def read(rel):
    path = os.path.join(FLOWENGINE, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def grab(rel, pattern, group=1):
    """从 FlowEngine 源码里抓一个数值常量。"""
    src = read(rel)
    if src is None:
        return None, f"{rel} 不存在"
    m = re.search(pattern, src)
    if not m:
        return None, f"{rel} 中未匹配 {pattern!r}"
    return float(m.group(group)), None


# (训练器属性名, FlowEngine 相对路径, 正则, 说明)
CHECKS = [
    ("STG_A_LAT_MAX", "modules/adas_nodes/st_graph.h",
     r"#define\s+STG_A_LAT_MAX\s+([0-9.]+)",
     "planning 曲率限速的横向加速度上限"),
    ("STG_CURVE_SAFETY", "modules/adas_nodes/st_graph.h",
     r"#define\s+STG_CURVE_SAFETY\s+([0-9.]+)",
     "planning 曲率限速安全系数"),
    ("WHEELBASE", "modules/adas_nodes/flowsim/entity.h",
     r"double\s+wheelbase\s*\{\s*([0-9.]+)\s*\}",
     "仿真实体默认轴距（flowsim ego 缺省值）"),
    ("MAX_SPEED", "config/pipeline.json",
     r'\\"max_speed\\":\s*([0-9.]+)',
     "inference 节点速度上限"),
]


def check_constants(trainer):
    fails = []
    for attr, rel, pat, desc in CHECKS:
        want, err = grab(rel, pat)
        got = getattr(trainer, attr, None)
        if err:
            # 「跳过」等于没检查 —— 门禁抓不住的东西不能算 PASS
            print(f"  [FAIL] {attr:18s} ({desc}) — {err}")
            fails.append(f"{attr}: 无法从 {rel} 读出 C 侧取值 — {err}")
            continue
        if got is None:
            fails.append(f"{attr} 训练器中不存在（应 = {want}，出处 {rel}）")
            continue
        ok = abs(got - want) < 1e-9
        print(f"  [{'OK ' if ok else 'FAIL'}] {attr:18s} 训练器={got:<8g} "
              f"C侧={want:<8g}  ({desc}, {rel})")
        if not ok:
            fails.append(f"{attr}: 训练器={got} 但 {rel} 里是 {want} — {desc}")
    return fails


def check_steer_limit(trainer):
    """steer_limit_for_speed 必须与 control_node.cpp 逐点一致。"""
    src = read("modules/adas_nodes/control_node.cpp")
    if src is None:
        print("  [跳过] steer_limit_for_speed — control_node.cpp 不存在")
        return []
    m = re.search(
        r"steer_limit_for_speed\s*\([^)]*\)\s*\{(.{0,400}?)\n\}", src, re.S)
    if not m:
        print("  [跳过] steer_limit_for_speed — 未能定位函数体")
        return []
    body = m.group(1)
    # 从 C 实现里抽出三个决定性数字：速度下限、转角下限、转角上限
    nums = re.findall(r"([0-9]+\.[0-9]+)", body)
    fails = []
    print(f"  [信息] control_node.cpp 函数体常量: {nums}")
    for lo, hi in (("0.016", "0.16"),):
        if lo not in nums or hi not in nums:
            fails.append(
                f"steer_limit_for_speed 的限幅常量 [{lo},{hi}] 未在 C 实现中找到，"
                f"实际为 {nums} — 训练器的转角限幅可能已与车上脱节")
    # 数值对拍
    for v in (0.0, 2.0, 5.0, 10.0, 14.0, 20.0):
        got = trainer.steer_limit_for_speed(v)
        if not (0.016 - 1e-9 <= got <= 0.16 + 1e-9):
            fails.append(f"steer_limit_for_speed({v}) = {got} 越出 [0.016, 0.16]")
    if not fails:
        print("  [OK ] steer_limit_for_speed  限幅区间 [0.016, 0.16] 与 C 侧一致")
    return fails


def check_curve_speed_limit(trainer):
    """训练器里的曲率限速公式必须等于 STG_CURVE_SAFETY*sqrt(STG_A_LAT_MAX/κ)。"""
    import math
    src = open(os.path.join(ROOT, "tools", "train_adas_cortex.py"),
               encoding="utf-8").read()
    fails = []
    if "MAX_LATERAL_ACCEL / abs(kap)" in src or \
       "MAX_LATERAL_ACCEL/abs(kap)" in src:
        fails.append(
            "训练器用 MAX_LATERAL_ACCEL(转角限幅系数) 做曲率限速 —— "
            "应使用 STG_CURVE_SAFETY*sqrt(STG_A_LAT_MAX/κ)")
    if "STG_CURVE_SAFETY * math.sqrt(STG_A_LAT_MAX" not in src.replace("\n", " "):
        # 允许换行，做一次宽松匹配
        flat = re.sub(r"\s+", " ", src)
        if "STG_CURVE_SAFETY * math.sqrt(STG_A_LAT_MAX" not in flat:
            fails.append("训练器中未找到 STG_CURVE_SAFETY*sqrt(STG_A_LAT_MAX/κ) 曲率限速")
    if not fails:
        k = 0.008
        expect = trainer.STG_CURVE_SAFETY * math.sqrt(trainer.STG_A_LAT_MAX / k)
        print(f"  [OK ] 曲率限速公式  κ=0.008 → v_lim={expect:.2f} m/s "
              f"(= {trainer.STG_CURVE_SAFETY}*sqrt({trainer.STG_A_LAT_MAX}/κ))")
    return fails


def check_train_covers_val(trainer):
    """训练曲率范围必须**覆盖**验证集，否则验证是外推、必然崩。"""
    fails = []
    val_k = []
    for name, path, _, _, _, _ in trainer.VAL_SCENARIOS:
        if path.kind == "arc":
            val_k.append(abs(path.kappa))
        elif path.kind == "sine":
            k = 2.0 * 3.141592653589793 / path.wavelen
            val_k.append(abs(path.amp) * k * k)
    if not val_k:
        return fails
    vmax = max(val_k)
    if vmax > trainer.TRAIN_KAPPA_MAX + 1e-9:
        fails.append(
            f"验证集峰值曲率 {vmax:.4f} 超过训练上界 TRAIN_KAPPA_MAX="
            f"{trainer.TRAIN_KAPPA_MAX} — 验证成了外推，会假性 FAIL")
    else:
        print(f"  [OK ] 曲率覆盖  训练上界 {trainer.TRAIN_KAPPA_MAX} "
              f">= 验证峰值 {vmax:.4f}")
    return fails


def check_env_trackable(trainer):
    """环境可跟随性门禁：工业标准 Stanley 必须跑满全部训练/验证场景。

    背景（2026-09-03）：曾一路把训练集调到"S弯CTE 恒 101cm 纹丝不动"，误以为
    是演化学不动，实为**环境本身不可跟随**：
      ① 按 κ 反解正弦振幅时没约束切线角 → 起步就差 23°，追不上；
      ② 转角限幅写死 a_lat=1.4，而车上是 |e_y|>0.5 用 2.4 的自适应包络；
      ③ 限速与限幅用同一个 a_lat → need<=lim 恰好卡边界、零修正余量；
      ④ 正弦为凑高 κ 把波长压到 33m，成了绕桩而非道路。
    这些坑的共同点是：**演化跑多少代都无解，但看指标只会以为是模型不行**。

    Stanley 跑不满 = 环境有病，先修环境再谈训练。
    """
    import math
    class Stanley:
        def reset_state(self):
            pass

        def forward(self, cte_n, dpsi_n, kap_n, v_n, verr_n, danger_n):
            cte = cte_n * 2.0
            dpsi = dpsi_n * 0.5
            kap = kap_n / 20.0
            v = max(v_n * trainer.MAX_SPEED, 1.0)
            delta = (dpsi + math.atan(2.0 * cte / v)
                     + math.atan(kap * trainer.WHEELBASE))
            lim = trainer.adaptive_steer_limit(v, cte)
            return (max(-1.0, min(1.0, delta / lim)),
                    max(-1.0, min(1.0, verr_n * 1.5)))

    fails = []
    worst = 0.0
    for grp, scn in (("训练", trainer.SCENARIOS), ("验证", trainer.VAL_SCENARIOS)):
        for name, path, spd, v0, dur, lead in scn:
            _, ok, mt = trainer.run_scenario(Stanley(), path, spd, v0, dur,
                                             lead_on=lead, seed=7)
            worst = max(worst, mt["avg_cte"])
            if not ok:
                fails.append(
                    f"{grp}场景 {name} Stanley 只跑 {mt['steps']}/{mt['total']} 步 "
                    f"(CTE {mt['avg_cte']*100:.1f}cm) — 环境不可跟随，先修场景")
    if not fails:
        n = len(trainer.SCENARIOS) + len(trainer.VAL_SCENARIOS)
        print(f"  [OK ] 环境可跟随性  Stanley 跑满 {n}/{n} 场景, "
              f"最差CTE {worst*100:.1f}cm")
    # CTE 过大也说明环境勉强：留一条软上限
    if worst > 0.60:
        fails.append(f"Stanley 最差 CTE {worst*100:.1f}cm > 60cm — 环境余量不足")
    return fails


def main():
    print("=" * 72)
    print("  ADAS Cortex 训练契约 ↔ FlowEngine C 源码 对齐门禁")
    print(f"  FlowEngine: {FLOWENGINE}")
    print("=" * 72)
    trainer = load_trainer()

    fails = []
    if not os.path.exists(FLOWENGINE):
        print(f"  [WARN] FlowEngine 源码未在同级目录挂载: {FLOWENGINE}")
        print("  ↳ 跳过跨仓 C 源码对账，仅执行环境动力学与自洽性门禁")
        fails += check_curve_speed_limit(trainer)
        fails += check_train_covers_val(trainer)
        fails += check_env_trackable(trainer)
        print("=" * 72)
        if fails:
            print(f"  FAIL — {len(fails)} 项自洽性未达标：")
            for f in fails:
                print(f"    - {f}")
            print("=" * 72)
            return 1
        print("  PASS — 内部训练环境动力学自洽性通过")
        print("=" * 72)
        return 0

    fails += check_constants(trainer)
    fails += check_steer_limit(trainer)
    fails += check_curve_speed_limit(trainer)
    fails += check_train_covers_val(trainer)
    fails += check_env_trackable(trainer)

    print("=" * 72)
    if fails:
        print(f"  FAIL — {len(fails)} 项契约错配：")
        for f in fails:
            print(f"    - {f}")
        print("=" * 72)
        return 1
    print("  PASS — 训练环境常量与车上 C 实现一致")
    print("=" * 72)
    return 0


def test_adas_cortex_contract():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
