#!/usr/bin/env python3
"""
SDSC ADAS Cortex 演化训练器 (FlowEngine 轨迹跟随契约)
=====================================================
与 tools/train_vehicle_cortex.py 的区别：
  - train_vehicle_cortex.py：玩具赛道、内部单位、2.5 m/s、只训横向循迹
  - 本训练器：对齐 FlowEngine `control_node` 的真实契约 —— 消费 planning
    轨迹的跟踪误差，同时输出横向 steer 与纵向 accel，车体模型与
    `modules/adas_nodes/flowsim/physics.cpp` 的运动学自行车模型一致
    （车辆中心参考点、half_wb 切向项、throttle 3.33 / brake 8.0、dt=0.05）。

契约（与 include/sdsc_cortex.h 导出体一致）：
  inputs[6]:
    0 cte_n    横向跟踪误差 / 2.0      (正 = 目标路径在车左侧)
    1 dpsi_n   航向误差 / 0.5 rad
    2 kappa_n  前视曲率 * 20.0
    3 v_n      车速 / 20.0
    4 verr_n   (v_target - v) / 5.0
    5 danger_n 危险度 = 1 - min(ttc,10)/10
  outputs[2]:
    0 steer_n  [-1,1] → steer = steer_n * steer_limit(v)
    1 accel_n  [-1,1] → accel = accel_n>0 ? accel_n*3.5 : accel_n*6.0

训练产物: checkpoints/adas_cortex_champion.json
导出:     tools/export_sdsc_cortex.py (读取该 checkpoint 生成 C11 头)
"""

import math
import random
import json
import os
import time
import argparse

# ── 与 FlowEngine 对齐的常量 ─────────────────────────────────────
WHEELBASE = 2.7
DT = 0.05
MAX_LATERAL_ACCEL = 1.4     # control_node.cpp steer_limit_for_speed 的转角限幅系数
# planning 的曲率限速（st_graph.h）：v_lim = STG_CURVE_SAFETY * sqrt(STG_A_LAT_MAX/κ)
# 注意这与上面的 1.4 是**两回事**：1.4 限方向盘转角，5.0 限规划速度。
# 训练里若误用 1.4 去限速，细胞体会被训成"弯道该减速"，而真车 planning 根本不减
# → 上车后一路顶着速度误差。必须逐字对齐 C 侧常量。
STG_A_LAT_MAX = 5.0
STG_CURVE_SAFETY = 0.85
CTE_FAIL = 2.0          # 超过判定出路沿，等价 demo_evaluator 的 road departure
MAX_SPEED = 20.0        # 与 pipeline.json max_speed 一致
ACCEL_MAX = 3.5         # accel_n>0 时的标定上限
BRAKE_MAX = 8.0         # 车体可达的制动上限（physics.cpp）

# ── 执行器真实性（车上有、旧训练环境没有 → 学出来的策略过于激进）──
STEER_RATE_MAX = 0.35   # rad/s，方向盘转角速率上限
STEER_LAG_TAU = 0.06    # s，转向一阶滞后
ACCEL_LAG_TAU = 0.12    # s，纵向执行器一阶滞后

# ── 扰动与噪声（防过拟合到"只收敛一次初始偏差"）──────────────
MEAS_NOISE_CTE = 0.02   # m
MEAS_NOISE_PSI = 0.004  # rad
GUST_PERIOD_S = 3.5     # 侧向扰动脉冲周期
GUST_ACCEL = 0.35       # m/s^2 等效侧向扰动

SDSC_PRIMITIVES = [
    "SUM", "INTEGRATE", "AMPLIFY", "INVERT",
    "THRESHOLD", "DAMPER", "CLIP", "ABS", "MULTIPLY",
    "DIFF", "HYSTERESIS", "DEADZONE", "INHIBIT",
    "SUB", "RATIO", "OSCILLATOR", "CORRELATION", "FATIGUE",
]

# 感受器 / 运动器名称（导出到 C 时作为注释，便于对账）
RECEPTOR_TYPES = [
    "REC_CTE_L", "REC_CTE_R", "REC_CTE_COARSE_L", "REC_CTE_COARSE_R",
    "REC_PSI", "REC_PSI_STRONG", "REC_KAPPA", "REC_CENTRIPETAL",
    "REC_SPEED", "REC_VERR", "REC_VERR_NEG", "REC_DANGER",
]
MOTOR_TYPES = [
    "MOT_STEER_P", "MOT_STEER_D", "MOT_ACC", "MOT_BRK",
    "EFFECTOR_STEER", "EFFECTOR_ACCEL",
]


LAT_ENV_CRUISE = 1.4    # 巡航舒适包络
LAT_ENV_MANEUVER = 2.4  # 大横向误差（变道/避障/跟弯）包络


def steer_limit_for_speed(v, a_lat=MAX_LATERAL_ACCEL):
    """与 control_node.cpp steer_limit_for_speed 完全一致。"""
    s = max(v, 2.0)
    return min(max(math.atan(a_lat * WHEELBASE / (s * s)), 0.016), 0.16)


def adaptive_steer_limit(v, lat_err):
    """车上的自适应转向包络（control_node.cpp:924）：|e_y|>0.5 用 2.4，否则 1.4。

    坑（2026-09-03 实测）：训练里写死 1.4，R=95m 弯在 v=13.9 需要
    δ=atan(2.7/95)=0.0284 rad，而 steer_limit_for_speed(13.9, 1.4)=0.0196 —— 
    **限幅比过弯所需还小**，连工业标准 Stanley 都只能跑满 8/16 场景，CTE 单调
    发散到 2m。车上靠这个自适应包络化解，训练里丢掉了就等于人为制造死局。
    """
    env = LAT_ENV_MANEUVER if abs(lat_err) > 0.5 else LAT_ENV_CRUISE
    return steer_limit_for_speed(v, env)


class SdscCell:
    __slots__ = ("ptype", "state", "aux_state", "output", "gain")

    def __init__(self, ptype, gain=None):
        self.ptype = ptype
        self.state = 0.0
        self.aux_state = 0.0
        self.output = 0.0
        self.gain = gain if gain is not None else random.uniform(0.6, 2.2)

    def forward_fast(self, x):
        pt, g = self.ptype, self.gain
        if pt == "SUM":
            self.output = math.tanh(x * g)
        elif pt == "INTEGRATE":
            self.state = self.state * 0.85 + x * 0.15
            self.output = math.tanh(self.state * g)
        elif pt == "AMPLIFY":
            self.output = math.tanh(x * g * 2.5)
        elif pt == "INVERT":
            self.output = -math.tanh(x * g)
        elif pt == "THRESHOLD":
            self.output = 1.0 if x > 0.25 else (-1.0 if x < -0.25 else 0.0)
        elif pt == "DAMPER":
            self.state = self.state * 0.70 + x * 0.30
            self.output = self.state
        elif pt == "CLIP":
            self.output = max(-1.0, min(1.0, x * g))
        elif pt == "ABS":
            self.output = abs(math.tanh(x * g))
        elif pt == "MULTIPLY":
            self.output = math.tanh(x * g * 1.5)
        elif pt == "DIFF":
            self.output = x - self.state
            self.state = x
        elif pt == "HYSTERESIS":
            if x > 0.15:
                self.state = 1.0
            elif x < -0.15:
                self.state = -1.0
            self.output = self.state
        elif pt == "DEADZONE":
            self.output = x * g if abs(x) > 0.08 else 0.0
        elif pt == "INHIBIT":
            self.state = self.state * 0.80 + abs(x) * 0.20
            self.output = math.tanh(x * g) * max(0.0, 1.0 - self.state)
        elif pt == "SUB":
            self.state = self.state * 0.60 + x * 0.40
            self.output = math.tanh((x - self.state) * g)
        elif pt == "RATIO":
            self.state = self.state * 0.85 + abs(x) * 0.15
            self.output = max(-2.0, min(2.0, x / (self.state + 0.1)))
        elif pt == "OSCILLATOR":
            s1 = self.state
            s2 = self.aux_state
            ds1 = s2
            ds2 = 1.0 * (1.0 - s1 * s1) * s2 - s1 + x
            dt = 0.05
            s1 = max(-3.0, min(3.0, s1 + ds1 * dt))
            s2 = max(-3.0, min(3.0, s2 + ds2 * dt))
            self.state = s1
            self.aux_state = s2
            self.output = math.tanh(s1)
        elif pt == "CORRELATION":
            self.state = self.state * 0.90 + (x * self.aux_state) * 0.10
            self.aux_state = x
            self.output = math.tanh(self.state * g)
        elif pt == "FATIGUE":
            self.state = min(2.0, self.state + abs(x) * 0.15) * 0.96
            self.output = math.tanh(x * g) / (1.0 + self.state)
        else:
            self.output = x
        return self.output


class AdasCortexOrgan:
    """6 输入 / 2 输出的 ADAS 轨迹跟随皮层器官。

    细胞按索引分三段：[感受器 | 隐藏层 | 运动器]，前向按索引序单遍推进
    （与 C 导出体逐字一致；反向边天然读到上一拍值，等价循环突触）。
    """

    def __init__(self, n_hidden=48, _empty=False):
        self.n_receptors = len(RECEPTOR_TYPES)
        self.n_motors = len(MOTOR_TYPES)
        self.n_hidden = n_hidden
        if _empty:
            return
        self.hidden_types = [random.choice(SDSC_PRIMITIVES) for _ in range(n_hidden)]
        self.synapses = []
        self._seed_synapses()
        self.build()

    # ── 结构 ────────────────────────────────────────────────
    def build(self):
        self.cells = [SdscCell(t) for t in RECEPTOR_TYPES]
        self.cells += [SdscCell(t) for t in self.hidden_types]
        self.cells += [SdscCell(t) for t in MOTOR_TYPES]
        self.mot_offset = self.n_receptors + self.n_hidden
        self.steer_id = self.mot_offset + MOTOR_TYPES.index("EFFECTOR_STEER")
        self.accel_id = self.mot_offset + MOTOR_TYPES.index("EFFECTOR_ACCEL")
        self.compile_incoming()

    def compile_incoming(self):
        self.incoming = [[] for _ in range(len(self.cells))]
        for (f, t, w) in self.synapses:
            if 0 <= f < len(self.cells) and self.n_receptors <= t < len(self.cells):
                self.incoming[t].append((f, w))

    def _seed_synapses(self):
        """播种先验通路：给演化一个可用起点，而非纯随机游走。"""
        n_rec, n_hid = self.n_receptors, self.n_hidden
        mot = n_rec + n_hid
        m_steer_p, m_steer_d = mot + 0, mot + 1
        m_acc, m_brk = mot + 2, mot + 3
        eff_steer, eff_accel = mot + 4, mot + 5

        l1 = list(range(n_rec, n_rec + n_hid // 2))
        l2 = list(range(n_rec + n_hid // 2, n_rec + n_hid))

        # 横向先锋通路：航向误差 + 横向误差 → 转向比例柱
        self.synapses += [
            (4, m_steer_p, random.uniform(1.0, 1.6)),
            (5, m_steer_p, random.uniform(0.4, 0.9)),
            (0, m_steer_p, random.uniform(0.6, 1.2)),
            (1, m_steer_p, -random.uniform(0.6, 1.2)),
            (2, m_steer_p, random.uniform(0.5, 1.1)),
            (3, m_steer_p, -random.uniform(0.5, 1.1)),
            (6, m_steer_p, random.uniform(0.5, 1.2)),   # 曲率前馈
            (7, m_steer_d, random.uniform(0.3, 0.8)),   # 向心阻尼
        ]
        # 纵向先锋通路：速度误差 → 加速柱；危险度 → 制动柱
        self.synapses += [
            (9, m_acc, random.uniform(0.8, 1.6)),
            (8, m_acc, -random.uniform(0.2, 0.6)),
            (10, m_brk, random.uniform(0.8, 1.6)),
            (11, m_brk, random.uniform(1.0, 2.0)),
            (7, m_brk, random.uniform(0.2, 0.8)),       # 弯道降速
        ]
        # 运动柱 → 效应器
        self.synapses += [
            (m_steer_p, eff_steer, 1.0),
            (m_steer_d, eff_steer, -0.3),
            (m_acc, eff_accel, 1.0),
            (m_brk, eff_accel, -1.2),
        ]
        # 皮层代谢连接
        for r in range(n_rec):
            for t in random.sample(l1, min(4, len(l1))):
                self.synapses.append((r, t, random.choice([-1.0, 1.0])))
        for s in l1:
            for t in random.sample(l2, min(3, len(l2))):
                self.synapses.append((s, t, random.choice([-1.0, 1.0])))
        for s in l2:
            self.synapses.append((s, m_steer_p, random.choice([-0.3, 0.3])))
            self.synapses.append((s, m_acc, random.choice([-0.3, 0.3])))

    # ── 前向 ────────────────────────────────────────────────
    def reset_state(self):
        for c in self.cells:
            c.state = 0.0
            c.aux_state = 0.0
            c.output = 0.0

    def forward(self, cte_n, dpsi_n, kappa_n, v_n, verr_n, danger_n):
        cells = self.cells
        cells[0].output = max(0.0, -cte_n)
        cells[1].output = max(0.0, cte_n)
        cells[2].output = max(0.0, -cte_n * 2.0 - 0.5)
        cells[3].output = max(0.0, cte_n * 2.0 - 0.5)
        cells[4].output = max(-1.0, min(1.0, dpsi_n))
        cells[5].output = max(-1.0, min(1.0, dpsi_n * 1.5))
        cells[6].output = max(-1.0, min(1.0, kappa_n))
        cells[7].output = max(-1.0, min(1.0, kappa_n * v_n))
        cells[8].output = max(0.0, min(1.0, v_n))
        cells[9].output = max(-1.0, min(1.0, verr_n))
        cells[10].output = max(0.0, min(1.0, -verr_n))
        cells[11].output = max(0.0, min(1.0, danger_n))

        inc = self.incoming
        for i in range(self.n_receptors, len(cells)):
            e = inc[i]
            if e:
                cells[i].forward_fast(sum(cells[f].output * w for f, w in e))
            else:
                cells[i].output = cells[i].state * 0.90

        s = max(-1.0, min(1.0, cells[self.steer_id].output))
        a = max(-1.0, min(1.0, cells[self.accel_id].output))
        return s, a

    # ── 演化 ────────────────────────────────────────────────
    def mutate(self):
        child = AdasCortexOrgan(n_hidden=self.n_hidden, _empty=True)
        child.hidden_types = list(self.hidden_types)
        child.synapses = list(self.synapses)
        gains = [c.gain for c in self.cells]

        for _ in range(random.randint(1, 3)):
            child.hidden_types[random.randrange(len(child.hidden_types))] = \
                random.choice(SDSC_PRIMITIVES)

        for _ in range(random.randint(3, 10)):
            i = random.randrange(len(child.synapses))
            f, t, w = child.synapses[i]
            child.synapses[i] = (f, t, w * random.uniform(0.75, 1.30)
                                 if random.random() < 0.75 else -w)

        total = child.n_receptors + child.n_hidden + child.n_motors
        for _ in range(random.randint(1, 5)):
            f = random.randrange(total)
            t = random.randrange(child.n_receptors, total)
            if f != t:
                child.synapses.append((f, t, random.choice([-0.6, 0.6])))

        cap = max(400, child.n_hidden * 6)
        while len(child.synapses) > cap:
            child.synapses.pop(random.randrange(len(child.synapses)))

        child.build()
        for i, c in enumerate(child.cells):
            if i < len(gains):
                c.gain = gains[i]
            if random.random() < 0.25:
                c.gain *= random.uniform(0.88, 1.14)
        return child

    def serialize(self):
        return {
            "contract": {"inputs": 6, "outputs": 2},
            "n_hidden": self.n_hidden,
            "hidden_types": self.hidden_types,
            "synapses": [[int(f), int(t), float(w)] for f, t, w in self.synapses],
            "cell_gains": [c.gain for c in self.cells],
        }

    @staticmethod
    def deserialize(data):
        organ = AdasCortexOrgan(n_hidden=data.get("n_hidden", 48), _empty=True)
        organ.hidden_types = list(data["hidden_types"])
        organ.synapses = [tuple(s) for s in data["synapses"]]
        organ.build()
        for i, g in enumerate(data.get("cell_gains", [])):
            if i < len(organ.cells):
                organ.cells[i].gain = g
        return organ


# ══════════════════════════════════════════════════════════════
#  训练场景：参考路径 + 目标速度剖面（模拟 planning 轨迹）
# ══════════════════════════════════════════════════════════════

class RefPath:
    """参考路径：x(s), y(s), heading(s), kappa(s)。直道 / 正弦 S 弯 / 定曲率弯。"""

    def __init__(self, kind, amp=0.0, wavelen=200.0, kappa=0.0):
        self.kind, self.amp, self.wavelen, self.kappa = kind, amp, wavelen, kappa

    def at(self, s):
        if self.kind == "straight":
            return s, 0.0, 0.0, 0.0
        if self.kind == "sine":
            k = 2.0 * math.pi / self.wavelen
            y = self.amp * math.sin(k * s)
            dy = self.amp * k * math.cos(k * s)
            ddy = -self.amp * k * k * math.sin(k * s)
            h = math.atan(dy)
            kap = ddy / (1.0 + dy * dy) ** 1.5
            return s, y, h, kap
        # 定曲率圆弧
        r = 1.0 / self.kappa
        th = s * self.kappa
        return r * math.sin(th), r * (1.0 - math.cos(th)), th, self.kappa


def speed_profile(kind, t, duration):
    """目标速度剖面（模拟 planning command_speed：巡航 / 红灯刹停 / 跟车)。"""
    if kind == "cruise":
        return 14.0
    if kind == "cruise_fast":
        return 19.0          # 覆盖 v_n 0.7~1.0 区间（旧训练完全没见过）
    if kind == "stop_go":
        if t < duration * 0.35:
            return 14.0
        if t < duration * 0.55:
            return 0.0          # 红灯刹停
        return 14.0             # 绿灯起步
    if kind == "follow":
        return 14.0 - 6.0 * max(0.0, math.sin(2.0 * math.pi * t / duration))
    if kind == "ramp":
        # 匝道汇入：低速起步一路加到高速巡航
        return 6.0 + 13.0 * min(1.0, t / (duration * 0.6))
    return 14.0


def achievable_ref(v_ref, v_target, dt):
    """把阶跃的 v_target 过一遍车体加减速上限，得到**物理可达**的速度参考。

    旧代价直接罚 |v_target - v|：红灯 14→0 阶跃在 -8 m/s^2 下至少要 1.75 s，
    起步 0→14 至少 4 s，这段误差积分是**执行器决定的下界，不可学**。
    罚它只会把演化推向到处 bang-bang（实测所有规模 verr 都卡在 2.2~2.6，
    加大权重后 768 细胞直接崩溃）。改罚 |v_ref - v| 才是可学信号。
    """
    dv = v_target - v_ref
    lim = (ACCEL_MAX if dv > 0 else BRAKE_MAX) * dt
    return v_ref + max(-lim, min(lim, dv))


class LeadVehicle:
    """前车模型：给出真实连续 TTC，让 danger 输入的语义与车上一致。

    旧训练里 danger = (v_target<0.5 ? 1 : 0) 是二值的，而 inference_node.cpp
    喂的是 1 - clamp(ttc,0,10)/10 的连续量 —— 中间值演化体从没见过。
    """

    def __init__(self, enabled, gap0=45.0, v=11.0):
        self.enabled, self.s, self.v = enabled, gap0, v

    def step(self, ego_v, dt, t):
        if not self.enabled:
            return 99.0
        self.v = 11.0 + 3.0 * math.sin(0.35 * t)
        self.s += (self.v - ego_v) * dt
        if self.s < 3.0:
            self.s = 3.0
        rel_v = self.v - ego_v
        return (self.s / -rel_v) if rel_v < -0.1 else 99.0


MAX_PATH_HEADING = 0.20   # rad，参考路径相对 x 轴的最大切线角（约 11.5°）
MIN_SINE_WAVELEN = 90.0   # m，正弦波长下限：再短就是绕桩不是道路


def sine_path_with_kappa(kappa_max, wavelen, max_heading=MAX_PATH_HEADING):
    """按目标峰值曲率反解正弦振幅，并约束路径切线角。

    坑（2026-09-03 实测）：只按 kappa = amp*(2pi/L)^2 反解，κ=0.0088/L=300
    会解出 amp=20.06 m —— 该正弦在 s=0 处切线角 0.398 rad(23°)，而车 heading
    从 0 起步，**一出生就差 23°**，任何控制器都追不上，CTE 直接发散到 2m。
    实测连纯比例控制器都在 13 步内崩，演化自然也学不动（S弯 CTE 恒在 101cm）。

    真实道路的曲率与航向是耦合的：车道级 S 弯振幅是米级不是几十米。所以按
    峰值切线角 amp*(2pi/L) <= max_heading 再压一次振幅；曲率不足则缩短波长
    补回来，保证既有目标曲率、又不出现起步就追不上的大航向角。
    """
    for _ in range(24):
        k = 2.0 * math.pi / wavelen
        amp = kappa_max / (k * k)
        if amp * k <= max_heading:
            return RefPath("sine", amp=amp, wavelen=wavelen)
        if wavelen * 0.85 < MIN_SINE_WAVELEN:
            break                # 触底：宁可降曲率也不生成绕桩路径
        wavelen *= 0.85          # 缩波长：κ 不变时振幅按 L^2 下降，切线角按 L 下降
    wavelen = max(wavelen, MIN_SINE_WAVELEN)
    k = 2.0 * math.pi / wavelen
    return RefPath("sine", amp=max_heading / k, wavelen=wavelen)


# 训练曲率上界必须**严格宽于**验证集（验证集峰值 0.012），否则就是外推。
TRAIN_KAPPA_MAX = 0.035


def make_train_scenarios(rng=None):
    """分层固定训练集：跨曲率全域均匀铺开，**代际间不变**。

    为什么不是"每代重采样"（2026-09-03 实测教训）
    --------------------------------------------
    先试过每代 rng 重抽场景参数，结果 cost 在 200~350 之间震荡不收敛：
    好抽签给 47、坏抽签给 350，代际间难度不可比，选择压力被抽签噪声完全淹没。

    域覆盖应当来自**场景集的广度**，而不是来自"每次换一批"。这里按 κ 分层
    铺满 [0, TRAIN_KAPPA_MAX]，配合每代轮换的测量噪声/扰动种子拿多样性 ——
    难度恒定、代际可比，同时不给演化留下"背下固定几条轨迹"的空间。

    ``rng`` 参数保留只为兼容旧调用，不再使用。
    """
    def sine_k(kappa, wavelen):
        return sine_path_with_kappa(kappa, wavelen)

    K = TRAIN_KAPPA_MAX
    return [
        # 直道 / 缓弯：基本循迹与速度保持
        ("straight_cruise", RefPath("straight"),          "cruise",      12.0, 20.0, False),
        ("gentle_s",        sine_k(0.004, 340.0),         "cruise",      13.0, 22.0, False),
        # S 弯分层：0.25K / 0.55K / 1.0K
        ("s_curve",         sine_k(0.25 * K, 300.0),      "cruise",      12.0, 25.0, False),
        ("s_curve_mid",     sine_k(0.55 * K, 230.0),      "cruise",      12.0, 22.0, False),
        # 注：正弦的极端曲率靠缩波长实现，缩过头会变成"绕桩"而非道路
        # （实测 κ=0.035 被压到 L=33m，2.1s 翻一次曲率符号，真实路网不存在）。
        # 所以 S 弯只覆盖到 0.55K，更高曲率交给下面的 arc 场景。
        ("s_curve_hard",    sine_k(0.75 * K, 200.0),      "cruise",      11.0, 22.0, False),
        # 定曲率弯分层：0.3K / 0.7K / 1.0K（含曲率限速必须生效的工况）
        ("curve_easy",      RefPath("arc", kappa=0.30 * K), "cruise",    12.0, 20.0, False),
        ("tight_curve",     RefPath("arc", kappa=0.70 * K), "cruise",    10.0, 20.0, False),
        ("tight_curve_max", RefPath("arc", kappa=1.00 * K), "cruise",     9.0, 20.0, False),
        # 纵向工况
        ("stop_go",         sine_k(0.003, 500.0),         "stop_go",     12.0, 22.0, False),
        ("follow",          sine_k(0.008, 300.0),         "follow",      12.0, 22.0, True),
        ("highway",         sine_k(0.005, 420.0),         "cruise_fast", 16.0, 22.0, False),
        ("ramp_merge",      RefPath("arc", kappa=0.45 * K), "ramp",       6.0, 22.0, True),
    ]


# 训练集：分层固定，跨代不变
SCENARIOS = make_train_scenarios()

# 留出验证集：不同振幅/波长/曲率/初速，训练中**不参与选择**，只报告泛化成绩
# 验证路径同样必须满足 MAX_PATH_HEADING —— 否则起步就有追不上的航向角，
# 验证会给出假性 FAIL（上一轮 val_s_curve 480 步只跑 14 步，实为路径 h0=0.364rad
# 不可跟随，被误读成"过拟合 8.63x"）。参数刻意取训练集**没有**的中间值。
VAL_SCENARIOS = [
    ("val_s_curve",   sine_path_with_kappa(0.0140, 210.0),      "cruise",      14.0, 24.0, False),
    ("val_curve",     RefPath("arc", kappa=0.0120),             "cruise",      11.0, 20.0, False),
    ("val_highway",   sine_path_with_kappa(0.0065, 470.0),      "cruise_fast", 18.0, 20.0, True),
    ("val_stop_go",   sine_path_with_kappa(0.0021, 610.0),      "stop_go",     13.0, 22.0, True),
]


def run_scenario(organ, path, spd_kind, v0, duration, lead_on=False,
                 seed=0, collect=False):
    """闭环试跑单场景，返回 (cost, ok, 指标)。"""
    organ.reset_state()
    rng = random.Random(seed)
    # 起始带初始偏差，逼迫演化学会收敛而非恰好停在中心
    x, y, heading, v = 0.0, 0.6, 0.0, v0
    steer, prev_steer = 0.0, 0.0
    accel_act = 0.0
    v_ref = v0
    lead = LeadVehicle(lead_on)
    s_ref = 0.0
    n = int(duration / DT)

    cum_cte = cum_verr = cum_dsteer = 0.0
    max_cte = 0.0
    steps = 0
    trace = []

    for i in range(n):
        t = i * DT
        # 参考点：沿弧长投影（近似取最近点，步进搜索）
        best_s, best_d2 = s_ref, 1e18
        probe = s_ref
        while probe < s_ref + 30.0:
            px, py, _, _ = path.at(probe)
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2, best_s = d2, probe
            probe += 0.5
        s_ref = best_s
        px, py, ph, _ = path.at(s_ref)
        # 前视点曲率（0.8 s 前视，等价 control 的 lookahead）
        _, _, _, kap = path.at(s_ref + max(v * 0.8, 2.0))

        # 跟踪误差（正 = 参考路径在车左侧）
        cte = math.cos(ph) * (py - y) - math.sin(ph) * (px - x)
        dpsi = (ph - heading + math.pi) % (2 * math.pi) - math.pi
        v_target = speed_profile(spd_kind, t, duration)
        # planning 的曲率限速（真车上由 st_graph 给出）：物理过不去的弯必须减速。
        # 旧训练 v_target 恒定，急弯里"减速"会被 verr 罚 → 代价函数自相矛盾，
        # 演化只能硬闯 → tight_curve CTE 21cm。
        if v_target > 0.5 and abs(kap) > 1e-4:
            # ① planning 的曲率限速（st_graph.h）
            v_target = min(v_target,
                           STG_CURVE_SAFETY * math.sqrt(STG_A_LAT_MAX / abs(kap)))
            # ② 转向包络自洽限速：规划速度必须是转角限幅真能转过去的速度。
            #    实测（2026-09-03）κ=0.035 时 ①给 10.2 m/s，但 2.4 包络只允许
            #    δ=0.0627 rad，过弯需要 0.0942 —— 规划出的速度自己转不过去，
            #    连 Stanley 都只能跑满 8/16，CTE 必然单调发散。
            #    a_lat 限速上限必须 <= 转向包络的 a_lat，否则前后矛盾。
            #    余量系数 0.75：need<=lim 等价于 v<=sqrt(a/κ)，两处用同一个 a
            #    会**恰好卡在边界、零余量** —— 稳态勉强够，但修正误差、抗扰动
            #    一点转角都不剩（实测 curve_easy 跑满 400 步却 CTE 94cm）。
            #    限速的 a 必须小于限幅的 a，差额就是控制器的可用权限。
            v_target = min(v_target,
                           0.75 * math.sqrt(LAT_ENV_MANEUVER / abs(kap)))
        v_ref = achievable_ref(v_ref, v_target, DT)
        ttc = lead.step(v, DT, t)
        danger = 1.0 - min(max(ttc, 0.0), 10.0) / 10.0
        if v_target < 0.5:
            danger = max(danger, 1.0)   # 刹停指令等价高危信号

        # 感知带噪声：车上 cte/dpsi 由定位与轨迹几何算出，不可能无噪
        cte_m = cte + rng.gauss(0.0, MEAS_NOISE_CTE)
        dpsi_m = dpsi + rng.gauss(0.0, MEAS_NOISE_PSI)

        steer_n, accel_n = organ.forward(
            max(-1.0, min(1.0, cte_m / 2.0)),
            max(-1.0, min(1.0, dpsi_m / 0.5)),
            max(-1.0, min(1.0, kap * 20.0)),
            max(0.0, min(1.0, v / MAX_SPEED)),
            max(-1.0, min(1.0, (v_target - v) / 5.0)),
            danger,
        )

        lim = adaptive_steer_limit(v, cte)
        steer_req = max(-lim, min(lim, steer_n * lim))
        # 执行器：速率限幅 + 一阶滞后（车上真实存在）
        d_max = STEER_RATE_MAX * DT
        steer_req = steer + max(-d_max, min(d_max, steer_req - steer))
        steer += (steer_req - steer) * min(1.0, DT / STEER_LAG_TAU)
        steer_cmd = max(-lim, min(lim, steer))

        accel_req = accel_n * ACCEL_MAX if accel_n > 0 else accel_n * 6.0
        accel_act += (accel_req - accel_act) * min(1.0, DT / ACCEL_LAG_TAU)
        accel = accel_act

        # 车体：与 physics.cpp step_bicycle 一致（中心参考点 + half_wb 切向项）
        if accel >= 0:
            v += min(accel, ACCEL_MAX) * DT
            v = min(v, MAX_SPEED)
        else:
            v += max(accel, -BRAKE_MAX) * DT
            v = max(v, 0.0)
        yaw_rate = v / WHEELBASE * math.tan(steer_cmd)
        half_wb = WHEELBASE * 0.5
        x += (v * math.cos(heading) - half_wb * math.sin(heading) * yaw_rate) * DT
        y += (v * math.sin(heading) + half_wb * math.cos(heading) * yaw_rate) * DT
        heading += yaw_rate * DT
        # 侧向扰动（阵风/路拱）：逼迫持续抑制误差，而非只收敛一次初始偏差
        gust = GUST_ACCEL * math.sin(2.0 * math.pi * t / GUST_PERIOD_S)
        heading += (gust / max(v, 3.0)) * DT

        acte = abs(cte)
        cum_cte += acte
        max_cte = max(max_cte, acte)
        cum_verr += abs(v_ref - v)          # 罚可达参考，不罚执行器物理下界
        cum_dsteer += abs(steer_cmd - prev_steer)
        prev_steer = steer_cmd
        steps += 1
        if collect:
            trace.append((x, y, v, steer_cmd, cte))

        if acte > CTE_FAIL:
            break

    ok = (steps == n)
    avg_cte = cum_cte / max(1, steps)
    avg_verr = cum_verr / max(1, steps)
    avg_dsteer = cum_dsteer / max(1, steps)
    # 代价：跟踪精度为主，速度跟随次之，抖动惩罚防 bang-bang；未跑满重罚
    cost = avg_cte * 10.0 + avg_verr * 3.0 + avg_dsteer * 40.0
    if not ok:
        cost += 50.0 * (1.0 - steps / n) + 20.0
    return cost, ok, {
        "avg_cte": avg_cte, "max_cte": max_cte,
        "avg_verr": avg_verr, "avg_dsteer": avg_dsteer,
        "steps": steps, "total": n, "trace": trace,
    }


def evaluate(organ, scenarios=None, noise_seed=0):
    total, all_ok, detail = 0.0, True, {}
    for name, path, spd, v0, dur, lead in (scenarios or SCENARIOS):
        c, ok, m = run_scenario(organ, path, spd, v0, dur, lead_on=lead,
                                seed=noise_seed)
        total += c
        all_ok = all_ok and ok
        detail[name] = m
    # 奥卡姆剃刀：轻微惩罚突触过度膨胀，防过拟合与死通路堆积
    total += len(organ.synapses) * 0.005
    return total, all_ok, detail


def main():
    ap = argparse.ArgumentParser(description="SDSC ADAS Cortex 演化训练器")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(root, "checkpoints", "adas_cortex_champion.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 72)
    print("  SDSC ADAS Cortex 演化训练器 (FlowEngine 轨迹跟随契约, 6-in / 2-out)")
    print(f"  场景: {', '.join(s[0] for s in SCENARIOS)}")
    print("=" * 72)

    t0 = time.time()
    pop = [AdasCortexOrgan(n_hidden=args.hidden) for _ in range(args.pop)]
    best, best_cost, best_detail = None, float("inf"), None

    for gen in range(1, args.generations + 1):
        # 每代换噪声实现（域随机化）：防止把某一条噪声轨迹背下来
        nseed = args.seed + gen * 7919
        gen_scn = SCENARIOS      # 场景固定 → 代际可比；多样性来自轮换的噪声种子
        scored = []
        for o in pop:
            c, ok, d = evaluate(o, gen_scn, noise_seed=nseed)
            scored.append((c, ok, d, o))
        scored.sort(key=lambda r: r[0])
        c, ok, d, o = scored[0]
        # 冠军也用当代噪声复评，避免"旧噪声下的好成绩"永久霸占精英位
        if best is None:
            best_cost, best, best_detail = c, o, d
        else:
            bc, _, bd = evaluate(best, gen_scn, noise_seed=nseed)
            if c < bc:
                best_cost, best, best_detail = c, o, d
            else:
                best_cost, best_detail = bc, bd
        print(f"[代际 {gen:3d}] cost={c:8.3f} 全场景通过={ok} "
              f"| 直道CTE={d['straight_cruise']['avg_cte']*100:5.1f}cm "
              f"S弯CTE={d['s_curve_hard']['avg_cte']*100:5.1f}cm "
              f"急弯CTE={d['tight_curve_max']['avg_cte']*100:5.1f}cm "
              f"高速CTE={d['highway']['avg_cte']*100:5.1f}cm "
              f"速度误差={d['stop_go']['avg_verr']:4.2f}m/s")

        survivors = [r[3] for r in scored[: max(2, args.pop // 4)]]
        newpop = [best] + survivors
        while len(newpop) < args.pop:
            newpop.append(random.choice(survivors).mutate())
        pop = newpop

    elapsed = time.time() - t0
    hold_seed = args.seed + 991
    final_cost, final_ok, final_detail = evaluate(best, noise_seed=hold_seed)
    val_cost, val_ok, val_detail = evaluate(best, VAL_SCENARIOS, noise_seed=hold_seed)
    print("\n" + "=" * 72)
    print(f"  冠军复核: cost={final_cost:.3f} 全场景通过={final_ok} 用时 {elapsed:.1f}s")
    for name, _, _, _, _, _ in SCENARIOS:
        m = final_detail[name]
        print(f"    {name:16s} 步数 {m['steps']:4d}/{m['total']:4d} "
              f"平均CTE {m['avg_cte']*100:6.2f}cm 最大CTE {m['max_cte']*100:6.2f}cm "
              f"速度误差 {m['avg_verr']:5.2f} m/s 抖动 {m['avg_dsteer']*1000:5.2f}mrad")
    print(f"  留出验证集(训练中未参与选择): cost={val_cost:.3f} 全场景通过={val_ok}")
    for name, _, _, _, _, _ in VAL_SCENARIOS:
        m = val_detail[name]
        print(f"    {name:16s} 步数 {m['steps']:4d}/{m['total']:4d} "
              f"平均CTE {m['avg_cte']*100:6.2f}cm 最大CTE {m['max_cte']*100:6.2f}cm "
              f"速度误差 {m['avg_verr']:5.2f} m/s 抖动 {m['avg_dsteer']*1000:5.2f}mrad")
    gap = val_cost / max(final_cost, 1e-6)
    print(f"  泛化差距 val/train = {gap:.2f}x" + ("  [过拟合警告]" if gap > 2.5 else ""))
    print("=" * 72)

    payload = {
        "trainer": "train_adas_cortex.py",
        "trained_time_seconds": round(elapsed, 2),
        "generations": args.generations,
        "population": args.pop,
        "seed": args.seed,
        "champion_cost": round(final_cost, 4),
        "all_scenarios_passed": final_ok,
        "hidden": args.hidden,
        "val_cost": round(val_cost, 4),
        "val_all_passed": val_ok,
        "val_metrics": {
            k: {kk: round(vv, 6) for kk, vv in v.items() if kk != "trace"}
            for k, v in val_detail.items()
        },
        "metrics": {
            k: {kk: round(vv, 6) for kk, vv in v.items() if kk != "trace"}
            for k, v in final_detail.items()
        },
        "checkpoint": out_path,
    }

    # 存盘为统一标准 SDSC-BIN (v2) 格式
    import struct
    import numpy as np

    organ_ser = best.serialize()
    hidden = organ_ser["hidden_types"]
    gains = organ_ser["cell_gains"]
    synapses = organ_ser["synapses"]

    n_rec = len(RECEPTOR_TYPES)
    n_mot = len(MOTOR_TYPES)
    n_hid = len(hidden)
    n_cells = n_rec + n_hid + n_mot
    n_syn = len(synapses)

    meta_dict = {
        "organism_id": "adas_cortex_champion",
        "generation": args.generations,
        "organ": organ_ser,
        "metrics": {
            k: {kk: round(vv, 6) for kk, vv in v.items() if kk != "trace"}
            for k, v in final_detail.items()
        }
    }
    meta_bytes = json.dumps(meta_dict, ensure_ascii=False).encode("utf-8")
    meta_size = len(meta_bytes)

    adj = [[] for _ in range(n_cells)]
    for (f, t, w) in synapses:
        adj[f].append((t, float(w)))

    row_ptr = [0] * (n_cells + 1)
    col_idx = []
    weights = []
    curr = 0
    for i in range(n_cells):
        row_ptr[i] = curr
        for v, w in adj[i]:
            col_idx.append(v)
            weights.append(w)
            curr += 1
    row_ptr[n_cells] = curr

    header_size = 72
    cells_size = n_cells * 4
    row_ptr_size = (n_cells + 1) * 4
    col_idx_size = n_syn * 4
    weights_size = n_syn * 4
    coords_size = n_cells * 3 * 4

    cells_offset = header_size
    row_ptr_offset = cells_offset + cells_size
    col_idx_offset = row_ptr_offset + row_ptr_size
    weights_offset = col_idx_offset + col_idx_size
    coords_offset = weights_offset + weights_size

    header_bytes = struct.pack(
        "<IIIIIIQQQQQQ",
        0x53445343,  # SDSC
        2,           # Version 2
        n_cells,
        n_syn,
        6,
        2,
        cells_offset,
        row_ptr_offset,
        col_idx_offset,
        weights_offset,
        coords_offset,
        (args.generations & 0xFFFFFFFF) | ((meta_size & 0xFFFFFFFF) << 32)
    )

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        cell_bytes = bytearray(cells_size)
        for i in range(n_cells):
            idx = i * 4
            cell_bytes[idx] = 4
            p1 = gains[i] if i < len(gains) else 1.0
            cell_bytes[idx + 1] = min(255, max(0, int(p1 * 64.0)))
            cell_bytes[idx + 2] = 0
            flags = 0
            if i < n_rec: flags |= 0x01
            if i >= n_cells - n_mot: flags |= 0x02
            cell_bytes[idx + 3] = flags
        f.write(cell_bytes)
        f.write(np.array(row_ptr, dtype=np.uint32).tobytes())
        f.write(np.array(col_idx, dtype=np.uint32).tobytes())
        f.write(np.array(weights, dtype=np.float32).tobytes())
        f.write(np.zeros((n_cells, 3), dtype=np.float32).tobytes())
        f.write(meta_bytes)

    report_path = os.path.join(root, "checkpoints", "adas_cortex_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[二进制检查点已存盘] -> {out_path}")
    print(f"[门禁评估报告已存盘] -> {report_path}")
    print(f"  细胞数={len(best.cells)} 突触数={len(best.synapses)}")


if __name__ == "__main__":
    main()
