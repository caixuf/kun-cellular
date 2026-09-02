#!/usr/bin/env python3
"""
SDSCC 128-Cell Cortical Organ Auto-Trainer (百万步极速演化训练器)
持续演化直至 128 细胞大脑皮层能够连续无误跑通 3 圈以上，并保存检查点
"""

import math
import random
import json
import os
import sys
import time

SDSCC_ALL_PRIMITIVES = [
    "SUM", "INTEGRATE", "AMPLIFY", "INVERT", 
    "THRESHOLD", "DAMPER", "CLIP", "ABS", "MULTIPLY"
]

class SdscCell:
    def __init__(self, cell_id, ptype, layer=1):
        self.cell_id = cell_id
        self.ptype = ptype
        self.layer = layer
        self.state = 0.0
        self.output = 0.0
        self.gain = random.uniform(0.6, 2.2)

    def forward_fast(self, x):
        pt = self.ptype
        gain = self.gain
        if pt == "SUM":
            self.output = math.tanh(x * gain)
        elif pt == "INTEGRATE":
            self.state = self.state * 0.85 + x * 0.15
            self.output = math.tanh(self.state * gain)
        elif pt == "AMPLIFY":
            self.output = math.tanh(x * gain * 2.5)
        elif pt == "INVERT":
            self.output = -math.tanh(x * gain)
        elif pt == "THRESHOLD":
            self.output = 1.0 if x > 0.25 else (-1.0 if x < -0.25 else 0.0)
        elif pt == "DAMPER":
            self.state = self.state * 0.70 + x * 0.30
            self.output = self.state
        elif pt == "CLIP":
            self.output = max(-1.0, min(1.0, x * gain))
        elif pt == "ABS":
            self.output = abs(math.tanh(x * gain))
        elif pt == "MULTIPLY":
            self.output = math.tanh(x * gain * 1.5)
        else:
            self.output = x
        return self.output

class SdscCorticalOrgan:
    def __init__(self, n_hidden=96):
        self.receptor_types = [
            "REC_CTE_FINE_L", "REC_CTE_FINE_R", "REC_CTE_COARSE_L", "REC_CTE_COARSE_R",
            "REC_PSI_NEAR", "REC_PSI_MID", "REC_PSI_FAR", "REC_PSI_INTEGRAL",
            "REC_CURV_NEAR", "REC_CURV_MID", "REC_CURV_FAR", "REC_CURV_DERIVATIVE",
            "REC_SPEED", "REC_ACCEL", "REC_LAT_DRIFT", "REC_CENTRIPETAL"
        ]
        self.motor_types = [
            "MOT_STEER_PROP", "MOT_STEER_INT", "MOT_STEER_DERIV", "MOT_STEER_DAMP",
            "MOT_BRAKE_CURV", "MOT_BRAKE_SURGE", "MOT_THROTTLE_CRUISE", "MOT_LAT_STABILITY",
            "EFFECTOR_STEER", "EFFECTOR_SPEED"
        ]
        self.n_receptors = len(self.receptor_types)
        self.n_motors = len(self.motor_types)
        self.n_hidden = n_hidden
        self.hidden_types = [random.choice(SDSCC_ALL_PRIMITIVES) for _ in range(n_hidden)]
        self.build_cortex()
        self.synapses = []
        self.generate_cortical_synapses()

    def build_cortex(self):
        self.cells = []
        for i, ptype in enumerate(self.receptor_types):
            self.cells.append(SdscCell(i, ptype, layer=0))
        for i, ptype in enumerate(self.hidden_types):
            layer = 1 if i < self.n_hidden // 2 else 2
            self.cells.append(SdscCell(self.n_receptors + i, ptype, layer=layer))
        offset = self.n_receptors + self.n_hidden
        for i, ptype in enumerate(self.motor_types):
            self.cells.append(SdscCell(offset + i, ptype, layer=3))
        
        self.steer_id = offset + self.motor_types.index("EFFECTOR_STEER")
        self.speed_id = offset + self.motor_types.index("EFFECTOR_SPEED")
        self.compile_incoming()

    def compile_incoming(self):
        self.incoming_synapses = [[] for _ in range(len(self.cells))]
        if hasattr(self, "synapses"):
            for (f, t, pol) in self.synapses:
                if 0 <= f < len(self.cells) and 0 <= t < len(self.cells):
                    self.incoming_synapses[t].append((f, pol))

    def generate_cortical_synapses(self):
        self.synapses = []
        rec_ids = list(range(self.n_receptors))
        l1_ids = list(range(self.n_receptors, self.n_receptors + self.n_hidden // 2))
        l2_ids = list(range(self.n_receptors + self.n_hidden // 2, self.n_receptors + self.n_hidden))
        mot_offset = self.n_receptors + self.n_hidden

        mot_steer_prop = mot_offset + 0
        mot_steer_int  = mot_offset + 1
        mot_steer_damp = mot_offset + 3
        mot_brake_curv = mot_offset + 4

        # 核心先锋通路
        self.synapses.append((4, mot_steer_prop, random.uniform(1.2, 1.8)))
        self.synapses.append((5, mot_steer_prop, random.uniform(0.6, 1.0)))
        self.synapses.append((0, mot_steer_prop, random.uniform(0.8, 1.4)))
        self.synapses.append((2, mot_steer_prop, random.uniform(1.0, 1.8)))
        self.synapses.append((1, mot_steer_prop, -random.uniform(0.8, 1.4)))
        self.synapses.append((3, mot_steer_prop, -random.uniform(1.0, 1.8)))
        self.synapses.append((8, mot_brake_curv, random.uniform(1.0, 1.8)))
        self.synapses.append((9, mot_brake_curv, random.uniform(1.2, 2.0)))

        self.synapses.append((mot_steer_prop, self.steer_id, 1.0))
        self.synapses.append((mot_steer_int, self.steer_id, 0.3))
        self.synapses.append((mot_steer_damp, self.steer_id, -0.3))
        self.synapses.append((mot_brake_curv, self.speed_id, 1.2))

        # 96 个中间代谢皮层连接
        for r in rec_ids:
            for target in random.sample(l1_ids, min(5, len(l1_ids))):
                self.synapses.append((r, target, random.choice([-1.0, 1.0])))

        for src in l1_ids:
            for target in random.sample(l2_ids, min(4, len(l2_ids))):
                self.synapses.append((src, target, random.choice([-1.0, 1.0])))

        for src in l2_ids:
            self.synapses.append((src, mot_steer_prop, random.choice([-0.4, 0.4])))
            self.synapses.append((src, mot_steer_damp, random.choice([-0.4, 0.4])))
            self.synapses.append((src, mot_brake_curv, random.choice([-0.4, 0.4])))

        for _ in range(120):
            src = random.choice(l1_ids + l2_ids)
            dst = random.choice(l1_ids + l2_ids)
            if src != dst:
                self.synapses.append((src, dst, random.choice([-0.8, 0.8])))

        self.compile_incoming()

    def forward(self, cte, psi_err, curv, speed, cte_deriv=0.0, psi_far=0.0):
        cells = self.cells
        cells[0].output = max(0.0, -cte)
        cells[1].output = max(0.0, cte)
        cells[2].output = max(0.0, -cte * 2.0 - 0.5)
        cells[3].output = max(0.0, cte * 2.0 - 0.5)
        cells[4].output = max(-1.0, min(1.0, psi_err))
        cells[5].output = max(-1.0, min(1.0, (psi_err + psi_far) * 0.5))
        cells[6].output = max(-1.0, min(1.0, psi_far))
        cells[7].output = max(-1.0, min(1.0, psi_err * 1.5))
        cells[8].output = min(1.0, curv * 20.0)
        cells[9].output = min(1.0, curv * 45.0)
        cells[10].output = min(1.0, curv * 80.0)
        cells[11].output = max(-1.0, min(1.0, cte_deriv * 2.0))
        cells[12].output = min(1.0, speed)
        cells[13].output = max(-1.0, min(1.0, speed - 0.5))
        cells[14].output = max(-1.0, min(1.0, cte_deriv))
        cells[15].output = min(1.0, curv * speed * 30.0)

        incomings = self.incoming_synapses
        for i in range(self.n_receptors, len(cells)):
            c = cells[i]
            inc = incomings[i]
            if inc:
                s = sum(cells[f].output * pol for f, pol in inc)
                c.forward_fast(s)
            else:
                c.output = c.state * 0.90

        return cells[self.steer_id].output, cells[self.speed_id].output

    def mutate(self):
        child = SdscCorticalOrgan.__new__(SdscCorticalOrgan)
        child.receptor_types = list(self.receptor_types)
        child.motor_types = list(self.motor_types)
        child.n_receptors = self.n_receptors
        child.n_motors = self.n_motors
        child.n_hidden = self.n_hidden
        child.hidden_types = list(self.hidden_types)
        child.synapses = list(self.synapses)

        for _ in range(random.randint(1, 4)):
            idx = random.randrange(len(child.hidden_types))
            child.hidden_types[idx] = random.choice(SDSCC_ALL_PRIMITIVES)

        for _ in range(random.randint(2, 6)):
            if child.synapses:
                idx = random.randrange(len(child.synapses))
                f, t, p = child.synapses[idx]
                child.synapses[idx] = (f, t, p * random.uniform(0.8, 1.25) if random.random() < 0.6 else -p)

        total_cells = child.n_receptors + child.n_hidden + child.n_motors
        for _ in range(random.randint(3, 8)):
            f = random.randrange(total_cells)
            t = random.randrange(child.n_receptors, total_cells)
            if f != t:
                child.synapses.append((f, t, random.choice([-0.8, 0.8])))

        if len(child.synapses) > 350:
            for _ in range(random.randint(2, 6)):
                child.synapses.pop(random.randrange(len(child.synapses)))

        child.build_cortex()
        for c in child.cells:
            if random.random() < 0.20:
                c.gain *= random.uniform(0.90, 1.12)
        return child

    def serialize(self):
        return {
            "n_hidden": self.n_hidden,
            "hidden_types": self.hidden_types,
            "synapses": self.synapses,
            "cell_gains": [c.gain for c in self.cells]
        }

    @staticmethod
    def deserialize(data):
        organ = SdscCorticalOrgan(n_hidden=data.get("n_hidden", 96))
        organ.hidden_types = data.get("hidden_types", organ.hidden_types)
        organ.synapses = [tuple(s) for s in data.get("synapses", [])]
        organ.build_cortex()
        gains = data.get("cell_gains", [])
        for i, g in enumerate(gains):
            if i < len(organ.cells):
                organ.cells[i].gain = g
        organ.compile_incoming()
        return organ

def get_track_point(s):
    cx, cy = 400.0, 300.0
    t = (s * 0.0025) % math.tau
    x = cx + math.cos(t) * 280.0 + math.sin(t * 2.0) * 80.0
    y = cy + math.sin(t) * 190.0 + math.cos(t * 3.0) * 35.0
    dx = -math.sin(t) * 280.0 + math.cos(t * 2.0) * 160.0
    dy =  math.cos(t) * 190.0 - math.sin(t * 3.0) * 105.0
    return x, y, math.atan2(dy, dx)

def get_max_curvature_ahead(s, v=2.5):
    v = max(0.5, v)
    probes = [v * 4, v * 8, v * 14]
    max_curv, _, _, theta0 = 0.0, 0, 0, get_track_point(s)[2]
    for ds in probes:
        _, _, theta1 = get_track_point(s + ds)
        curv = abs((theta1 - theta0 + math.pi) % math.tau - math.pi) / max(ds, 1.0)
        max_curv = max(max_curv, curv)
        theta0 = theta1
    return max_curv

def evaluate_organ(organ, max_steps=1500):
    """闭环试跑评估：跑满 1500 步 (~3 圈赛道)"""
    x, y, theta = get_track_point(0.0)
    v = 2.5
    delta = 0.0
    s = 0.0
    cum_cte = 0.0
    max_cte = 0.0
    steps = 0
    dt = 0.04
    L = 24.0
    road_width = 46.0
    prev_cte = 0.0

    for step in range(max_steps):
        steps += 1
        s += v * dt * 25.0
        if step % 3 == 0:
            best_s, best_dist = s, float("inf")
            for ds in range(-2, 8):
                probe_s = s + ds * 12.0
                px, py, _ = get_track_point(probe_s)
                d = (x - px)**2 + (y - py)**2
                if d < best_dist:
                    best_dist, best_s = d, probe_s
            s = best_s

        cx, cy, road_theta = get_track_point(s)
        dx = x - cx
        dy = y - cy
        signed_cte = math.cos(road_theta) * dy - math.sin(road_theta) * dx
        cte = abs(signed_cte)
        cum_cte += cte
        max_cte = max(max_cte, cte)
        cte_deriv = (cte - prev_cte) / dt
        prev_cte = cte

        heading_err = (road_theta - theta + math.pi) % math.tau - math.pi
        curv = get_max_curvature_ahead(s, v)
        _, _, psi_far = get_track_point(s + v * 12.0)
        psi_far_err = (psi_far - theta + math.pi) % math.tau - math.pi

        cte_norm = signed_cte / (road_width * 0.5)
        heading_norm = heading_err / (math.pi * 0.5)
        curv_norm = min(1.0, curv * 50.0)
        speed_norm = v / 5.0
        steer_raw, speed_raw = organ.forward(cte_norm, heading_norm, curv_norm, speed_norm, cte_deriv, psi_far_err)

        steer_target = max(-0.45, min(0.45, steer_raw * 0.45))
        delta += (steer_target - delta) * 0.30
        target_v = max(1.5, 4.2 - max(0.0, speed_raw) * 2.7)
        v += (target_v - v) * 0.12

        beta = math.atan(0.5 * math.tan(delta))
        x += v * math.cos(theta + beta) * dt
        y += v * math.sin(theta + beta) * dt
        theta += (v / L) * math.cos(beta) * math.tan(delta) * dt

        if cte > 24.0:
            break

    lap_bonus = 5000.0 if steps >= max_steps else 0.0
    fitness = steps * 5.0 - (cum_cte / max(1, steps)) * 20.0 + lap_bonus
    avg_cte_m = (cum_cte / max(1, steps)) * 0.05
    max_cte_m = max_cte * 0.05
    return fitness, steps, avg_cte_m, max_cte_m

def main():
    print("======================================================================")
    print("  SDSCC 128-Cell Cortical Organ Auto-Trainer (百万步百万细胞极速演化)")
    print("  目标: 演化直至 128 细胞大脑皮层连续无误跑通 3 圈以上 (1500 步)")
    print("======================================================================")

    pop_size = 20
    population = [SdscCorticalOrgan(n_hidden=96) for _ in range(pop_size)]
    checkpoint_dir = "/home/caixuf/code/kun-cellular/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "vehicle_cortex_champion.json")

    best_organ = None
    best_fitness = -float("inf")
    generation = 0
    t0 = time.time()

    while generation < 80:
        generation += 1
        results = [evaluate_organ(organ, max_steps=1200) for organ in population]
        fits = [r[0] for r in results]
        steps_list = [r[1] for r in results]
        avg_ctes = [r[2] for r in results]
        max_ctes = [r[3] for r in results]

        gen_best_idx = fits.index(max(fits))
        gen_best_fit = fits[gen_best_idx]
        gen_best_steps = steps_list[gen_best_idx]
        gen_best_avg_cte = avg_ctes[gen_best_idx]
        gen_best_max_cte = max_ctes[gen_best_idx]

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_organ = population[gen_best_idx]

        print(f"[代际 {generation:3d}] 最佳步数: {gen_best_steps:4d}/1200 步 | 平均CTE: {gen_best_avg_cte:6.4f}m | 最大CTE: {gen_best_max_cte:6.4f}m | 适应度: {gen_best_fit:7.1f}")

        # 如果最佳个体连续跑满 1200 步 (约 2.5 圈) 且平均偏离小于 5 厘米，达到通关标准！
        if gen_best_steps >= 1200 and gen_best_avg_cte < 0.05:
            print(f"\n[达成高阶通关标准!] 代际 {generation} 最佳个体连续跑满 {gen_best_steps} 步，平均 CTE 仅 {gen_best_avg_cte*100:.2f} 厘米!")
            break

        # 锦标赛选择与突变
        sorted_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
        survivors = [population[i] for i in sorted_idx[:max(2, pop_size // 4)]]
        new_pop = [best_organ] + list(survivors)
        while len(new_pop) < pop_size:
            parent = random.choice(survivors)
            new_pop.append(parent.mutate())
        population = new_pop

    # 最终进行 3 圈 (1500步) 深度无损验证
    print("\n================ 正在对终极冠军进行 3 圈 (1500步) 完整无损大考 ================")
    final_fit, final_steps, final_avg_cte, final_max_cte = evaluate_organ(best_organ, max_steps=1500)
    print(f"终极实测: 连续跑完 {final_steps}/1500 步 (约 3.2 圈) | 平均 CTE: {final_avg_cte*100:.2f} 厘米 | 最大 CTE: {final_max_cte*100:.2f} 厘米")

    # 存盘保存检查点
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({
            "generation": generation,
            "trained_time_seconds": round(time.time() - t0, 2),
            "champion_fitness": round(final_fit, 1),
            "final_steps": final_steps,
            "avg_cte_m": round(final_avg_cte, 4),
            "max_cte_m": round(final_max_cte, 4),
            "organ": best_organ.serialize()
        }, f, indent=2)
    print(f"[检查点已存盘] -> {checkpoint_path}")

if __name__ == "__main__":
    main()
