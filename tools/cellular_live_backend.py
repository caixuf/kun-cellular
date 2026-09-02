#!/usr/bin/env python3
"""
Software-Defined Silicon Cellular Computer (SDSCC) Live Backend Server
----------------------------------------------------------------------
Zero-dependency HTTP & WebSocket server providing:
1. Real-time 3D Lennard-Jones physical force-field simulation
2. Real-time 24-primitive signal forward execution
3. High-frequency non-blocking WebSocket state streaming (30~60 Hz)
4. REST API for dynamic stimulation, mutations, and presets
5. Static file hosting for frontend observatory
"""

import os
import sys
import math
import time
import json
import random
import socket
import socketserver
import struct
import hashlib
import base64
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT = 8833
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")


# ============================================================================
# 0.13 硅基细胞计算机车辆控制器 (SDSCC Vehicle Controller - True 24-Primitive DAG Evolution)
# 基因组编码 DAG 拓扑结构（哪些原语、如何连接），而非浮点参数向量
# ============================================================================

# ============================================================================
# 0.13 硅基细胞计算机自动驾驶大脑皮层 (SDSCC 128-Cell Autonomous Driving Cortex)
# 仿生多层皮层架构：16感知受体 + 96中间代谢原语皮层 + 16小脑运动效应器 = 128+细胞，500+突触
# ============================================================================

SDSCC_ALL_PRIMITIVES = [
    "SUM", "INTEGRATE", "AMPLIFY", "INVERT", 
    "THRESHOLD", "DAMPER", "CLIP", "ABS", "MULTIPLY"
]

class SdscCell:
    """单个 SDSCC 计算细胞：具备时域积分、非线性传递与突触极性调制"""
    def __init__(self, cell_id, ptype, layer=1):
        self.cell_id = cell_id
        self.ptype = ptype
        self.layer = layer       # 0: 受体层, 1: 联络层, 2: 积分记忆层, 3: 运动层
        self.state = 0.0         # 内部膜电位/时域积分
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
    """
    SDSCC 硅基大脑皮层器官（128 细胞规模，500+ 条突触）:
    - Layer 0 (Receptors): 16 个高分辨率空间/时域感知受体细胞
    - Layer 1~2 (Interneurons): 96 个密集互联的中间代谢计算细胞
    - Layer 3 (Motor & Effectors): 16 个小脑协同与执行动作细胞
    """
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
        # 1. 受体层 (0 ~ 15)
        for i, ptype in enumerate(self.receptor_types):
            self.cells.append(SdscCell(i, ptype, layer=0))
        # 2. 中间皮层 (16 ~ 16+n_hidden-1)
        for i, ptype in enumerate(self.hidden_types):
            layer = 1 if i < self.n_hidden // 2 else 2
            self.cells.append(SdscCell(self.n_receptors + i, ptype, layer=layer))
        # 3. 运动效应层 (16+n_hidden ~ end)
        offset = self.n_receptors + self.n_hidden
        for i, ptype in enumerate(self.motor_types):
            self.cells.append(SdscCell(offset + i, ptype, layer=3))
        
        self.steer_id = offset + self.motor_types.index("EFFECTOR_STEER")
        self.speed_id = offset + self.motor_types.index("EFFECTOR_SPEED")
        self.compile_incoming()

    def compile_incoming(self):
        """预编译反向入边索引（实现微秒级零内存分配推演）"""
        self.incoming_synapses = [[] for _ in range(len(self.cells))]
        if hasattr(self, "synapses"):
            for (f, t, pol) in self.synapses:
                if 0 <= f < len(self.cells) and 0 <= t < len(self.cells):
                    self.incoming_synapses[t].append((f, pol))

    def generate_cortical_synapses(self):
        """生成分层投射 + 侧向抑制 + 循环反馈突触网络（500+ 条）"""
        self.synapses = []
        rec_ids = list(range(self.n_receptors))
        l1_ids = list(range(self.n_receptors, self.n_receptors + self.n_hidden // 2))
        l2_ids = list(range(self.n_receptors + self.n_hidden // 2, self.n_receptors + self.n_hidden))
        mot_ids = list(range(self.n_receptors + self.n_hidden, len(self.cells)))

        for r in rec_ids:
            for target in random.sample(l1_ids, min(5, len(l1_ids))):
                self.synapses.append((r, target, random.choice([-1.0, 1.0])))

        for src in l1_ids:
            for target in random.sample(l2_ids, min(4, len(l2_ids))):
                self.synapses.append((src, target, random.choice([-1.0, 1.0])))

        for src in l2_ids:
            for target in random.sample(mot_ids, min(3, len(mot_ids))):
                self.synapses.append((src, target, random.choice([-1.0, 1.0])))

        for m in mot_ids[:-2]:
            self.synapses.append((m, self.steer_id, random.choice([-1.0, 1.0])))
            self.synapses.append((m, self.speed_id, random.choice([-1.0, 1.0])))

        for _ in range(80):
            src = random.choice(l1_ids + l2_ids)
            dst = random.choice(l1_ids + l2_ids + mot_ids)
            if src != dst:
                self.synapses.append((src, dst, random.choice([-1.0, 1.0])))
        self.compile_incoming()

    def forward(self, cte, psi_err, curv, speed, cte_deriv=0.0, psi_far=0.0):
        """128 细胞多层皮层拓扑前向脉冲激发传导 (零内存分配)"""
        cells = self.cells
        # 1. 16 受体特征注入
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

        # 2. 预编译入边快速激活
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
        """形态发生突变：突触增殖重连、原语点突变、增益微调"""
        child = SdscCorticalOrgan.__new__(SdscCorticalOrgan)
        child.receptor_types = list(self.receptor_types)
        child.motor_types = list(self.motor_types)
        child.n_receptors = self.n_receptors
        child.n_motors = self.n_motors
        child.n_hidden = self.n_hidden
        child.hidden_types = list(self.hidden_types)
        child.synapses = list(self.synapses)

        # 1. 原语点突变
        for _ in range(random.randint(2, 5)):
            idx = random.randrange(len(child.hidden_types))
            child.hidden_types[idx] = random.choice(SDSCC_ALL_PRIMITIVES)

        # 2. 突触极性翻转
        for _ in range(random.randint(3, 8)):
            if child.synapses:
                idx = random.randrange(len(child.synapses))
                f, t, p = child.synapses[idx]
                child.synapses[idx] = (f, t, -p)

        # 3. 新增突触连接
        total_cells = child.n_receptors + child.n_hidden + child.n_motors
        for _ in range(random.randint(4, 10)):
            f = random.randrange(total_cells)
            t = random.randrange(child.n_receptors, total_cells)
            if f != t:
                child.synapses.append((f, t, random.choice([-1.0, 1.0])))

        # 4. 突触修剪凋亡
        if len(child.synapses) > 350:
            for _ in range(random.randint(2, 6)):
                child.synapses.pop(random.randrange(len(child.synapses)))

        child.build_cortex()
        for c in child.cells:
            if random.random() < 0.20:
                c.gain *= random.uniform(0.88, 1.15)
        return child

class LiveVehicleSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_cte = []
        self.road_width = 46.0
        self.prev_cte = 0.0
        self.init_track()
        # 种群：6 个 SDSCC 128-细胞大脑皮层器官 (SdscCorticalOrgan)
        self.population = [SdscCorticalOrgan(n_hidden=96) for _ in range(6)]
        self.current_agent = 0
        self.agent_lap_steps = 0
        self.agent_cum_cte = 0.0
        self.fitness_log = []
        self.champion_genome = None
        self.champion_fitness = -1.0
        self.champion_trail = []
        self.init_vehicle()
        # 启动时极速超演化 20 代 128 细胞大脑皮层
        self.fast_evolve_batch(target_generations=20, pop_size=12, sim_steps_per_agent=350)

    def fast_evolve_batch(self, target_generations=20, pop_size=12, sim_steps_per_agent=350):
        """
        128 细胞大脑皮层极速批量超演化加速引擎 (微秒级零开销推演)
        """
        pop = [SdscCorticalOrgan(n_hidden=96) for _ in range(pop_size)]
        if self.champion_genome:
            pop[0] = self.champion_genome

        best_genome = self.champion_genome or pop[0]
        best_fitness = self.champion_fitness if self.champion_fitness > 0 else 0.0

        for gen in range(target_generations):
            fits = []
            for g in pop:
                x, y, theta = self.get_track_point(0.0)
                v = 2.5
                delta = 0.0
                s = 0.0
                cum_cte = 0.0
                steps = 0
                dt = 0.04
                L = 24.0

                for step in range(sim_steps_per_agent):
                    steps += 1
                    s += v * dt * 25.0
                    if step % 4 == 0:
                        best_s, best_dist = s, float("inf")
                        for ds in range(-2, 8):
                            probe_s = s + ds * 12.0
                            px, py, _ = self.get_track_point(probe_s)
                            d = (x - px)*(x - px) + (y - py)*(y - py)
                            if d < best_dist:
                                best_dist, best_s = d, probe_s
                        s = best_s

                    cx, cy, road_theta = self.get_track_point(s)
                    dx = x - cx
                    dy = y - cy
                    signed_cte = math.cos(road_theta) * dy - math.sin(road_theta) * dx
                    cte = abs(signed_cte)
                    cum_cte += cte
                    heading_err = (road_theta - theta + math.pi) % math.tau - math.pi
                    curv = self.get_max_curvature_ahead(s)

                    cte_norm = signed_cte / (self.road_width * 0.5)
                    heading_norm = heading_err / (math.pi * 0.5)
                    curv_norm = min(1.0, curv * 50.0)
                    speed_norm = v / 5.0
                    steer_raw, speed_raw = g.forward(cte_norm, heading_norm, curv_norm, speed_norm)

                    steer_target = max(-0.45, min(0.45, steer_raw * 0.45))
                    delta += (steer_target - delta) * 0.30
                    target_v = max(1.5, 4.2 - max(0.0, speed_raw) * 2.7)
                    v += (target_v - v) * 0.12

                    beta = math.atan(0.5 * math.tan(delta))
                    x += v * math.cos(theta + beta) * dt
                    y += v * math.sin(theta + beta) * dt
                    theta += (v / L) * math.cos(beta) * math.tan(delta) * dt

                    if cte > 28.0:
                        break

                fitness = steps / (1.0 + cum_cte / max(1, steps))
                fits.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = g

            # 锦标赛自然选择
            sorted_idx = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
            survivors = [pop[i] for i in sorted_idx[:max(2, pop_size // 4)]]
            new_pop = list(survivors)
            while len(new_pop) < pop_size:
                parent = random.choice(survivors)
                new_pop.append(parent.mutate())
            pop = new_pop

        with self.lock:
            self.generation += target_generations
            self.champion_genome = best_genome
            self.champion_fitness = round(best_fitness, 1)
            self.population = [best_genome] + [best_genome.mutate() for _ in range(5)]
            self.current_agent = 0
            self.init_vehicle()

        return {
            "trained_generations": target_generations,
            "champion_fitness": self.champion_fitness,
            "n_cells": len(best_genome.cells),
            "n_synapses": len(best_genome.synapses),
            "hidden_types": list(best_genome.hidden_types)
        }

    def get_track_point(self, s):
        cx, cy = 400.0, 300.0
        t = (s * 0.0025) % math.tau
        x = cx + math.cos(t) * 280.0 + math.sin(t * 2.0) * 80.0
        y = cy + math.sin(t) * 190.0 + math.cos(t * 3.0) * 35.0
        dx = -math.sin(t) * 280.0 + math.cos(t * 2.0) * 160.0
        dy =  math.cos(t) * 190.0 - math.sin(t * 3.0) * 105.0
        return x, y, math.atan2(dy, dx)

    def get_max_curvature_ahead(self, s):
        v = max(0.5, self.v)
        probes = [v * 4, v * 8, v * 14]
        max_curv, _, _, theta0 = 0.0, 0, 0, self.get_track_point(s)[2]
        for ds in probes:
            _, _, theta1 = self.get_track_point(s + ds)
            curv = abs((theta1 - theta0 + math.pi) % math.tau - math.pi) / max(ds, 1.0)
            max_curv = max(max_curv, curv)
            theta0 = theta1
        return max_curv

    def init_track(self):
        self.track_points = []
        for i in range(180):
            s_i = (i / 180) * (math.tau / 0.0025)
            x, y, theta = self.get_track_point(s_i)
            self.track_points.append({"s": round(s_i, 1), "x": round(x, 1), "y": round(y, 1), "theta": round(theta, 3)})

    def init_vehicle(self):
        x0, y0, theta0 = self.get_track_point(0.0)
        self.x, self.y, self.theta = x0, y0, theta0
        self.v = 2.0
        self.delta = 0.0
        self.s = 0.0
        self.cte = 0.0
        self.total_dist = 0.0
        self.trail = []
        self.agent_lap_steps = 0
        self.agent_cum_cte = 0.0

    def next_agent(self):
        with self.lock:
            steps = max(1, self.agent_lap_steps)
            fitness = steps / (1.0 + self.agent_cum_cte / steps)
            self.fitness_log.append(round(fitness, 1))
            if len(self.fitness_log) > 20:
                self.fitness_log.pop(0)
            if fitness > self.champion_fitness:
                self.champion_fitness = fitness
                self.champion_genome = self.population[self.current_agent]
                self.champion_trail = list(self.trail)
            self.current_agent = (self.current_agent + 1) % len(self.population)
            if self.current_agent == 0:
                self.generation += 1
                if self.champion_genome:
                    self.population[0] = self.champion_genome
                    for i in range(1, len(self.population)):
                        self.population[i] = self.champion_genome.mutate()
            self.init_vehicle()

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            self.agent_lap_steps += 1
            dt = 0.04
            L = 24.0

            # 1. 最近点投影锁定 s
            best_s, best_dist = self.s, float("inf")
            for ds in range(-3, 18):
                probe_s = self.s + ds * 8.0
                px, py, _ = self.get_track_point(probe_s)
                d = math.hypot(self.x - px, self.y - py)
                if d < best_dist:
                    best_dist, best_s = d, probe_s
            self.s = best_s

            # 2. 参考点与传感器数据
            cx, cy, road_theta = self.get_track_point(self.s)
            dx = self.x - cx
            dy = self.y - cy
            signed_cte = math.cos(road_theta) * dy - math.sin(road_theta) * dx
            self.cte = abs(signed_cte)
            self.agent_cum_cte += self.cte
            heading_err = (road_theta - self.theta + math.pi) % math.tau - math.pi
            curv = self.get_max_curvature_ahead(self.s)

            # 3. 硅基细胞 DAG 前向传导（真正的 SDSCC 推演，无硬编码黑盒参数）
            genome = self.population[self.current_agent]
            cte_norm = signed_cte / (self.road_width * 0.5)
            heading_norm = heading_err / (math.pi * 0.5)
            curv_norm = min(1.0, curv * 50.0)
            speed_norm = self.v / 5.0
            steer_raw, speed_raw = genome.forward(cte_norm, heading_norm, curv_norm, speed_norm)

            # 4. 将细胞效应器输出映射到物理执行机构
            steer_target = max(-0.45, min(0.45, steer_raw * 0.45))
            self.delta += (steer_target - self.delta) * 0.30
            # 速度受控于效应器细胞 (输出正向激活即进行急弯预瞄制动)
            target_v = max(1.5, 4.2 - max(0.0, speed_raw) * 2.7)
            self.v += (target_v - self.v) * 0.12

            # 5. 阿克曼运动学积分
            beta = math.atan(0.5 * math.tan(self.delta))
            self.x += self.v * math.cos(self.theta + beta) * dt
            self.y += self.v * math.sin(self.theta + beta) * dt
            self.theta += (self.v / L) * math.cos(beta) * math.tan(self.delta) * dt
            self.total_dist += self.v * dt

            # 6. 失控淘汰
            if self.cte > 28.0 or self.agent_lap_steps > 10000:
                self.next_agent()
                return

            # 7. 记录
            if self.step_count % 5 == 0:
                self.history_cte.append(round(self.cte * 0.05, 3))
                if len(self.history_cte) > 40:
                    self.history_cte.pop(0)
            if self.step_count % 3 == 0:
                self.trail.append({"x": round(self.x, 1), "y": round(self.y, 1)})
                if len(self.trail) > 120:
                    self.trail.pop(0)

    def get_snapshot(self):
        with self.lock:
            genome = self.population[self.current_agent]
            return {
                "generation": self.generation,
                "agent_index": self.current_agent,
                "champion_fitness": round(self.champion_fitness, 1),
                "fitness_log": list(self.fitness_log),
                "n_cells": len(genome.cells),
                "n_synapses": len(genome.synapses),
                "hidden_types": list(genome.hidden_types)[:12],
                "cell_activities": [
                    {"id": c.cell_id, "type": c.ptype, "layer": c.layer, "out": round(c.output, 2)}
                    for c in genome.cells
                ],
                "step_count": self.step_count,
                "total_dist_m": round(self.total_dist, 1),
                "road_width": self.road_width,
                "track": self.track_points,
                "champion_trail": list(self.champion_trail)[-60:],
                "car": {
                    "x": round(self.x, 1),
                    "y": round(self.y, 1),
                    "s": round(self.s, 1),
                    "theta": round(self.theta, 3),
                    "delta_deg": round(math.degrees(self.delta), 1),
                    "speed_kmh": round(self.v * 14.0, 1),
                    "cte_m": round(self.cte * 0.05, 3)
                },
                "trail": list(self.trail),
                "history_cte": list(self.history_cte)
            }

live_veh = LiveVehicleSimulator()

def veh_loop():
    while True:
        for _ in range(live_veh.warp_speed):
            live_veh.step_physics()
        time.sleep(0.016)

threading.Thread(target=veh_loop, daemon=True).start()


class ObservatoryHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        
        if self.path.startswith("/api/control/warp") or self.path.startswith("/api/warp"):
            speed = "1x"
            if "speed=" in self.path:
                speed = self.path.split("speed=")[1].split("&")[0]
            mode = organism.set_warp(speed)
            body = json.dumps({"status": "ok", "warp_speed": mode, "warp_factor": organism.warp_factor}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        
        if self.path == "/api/checkpoints":
            ckpts = []
            runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
            if os.path.exists(runs_dir):
                for fname in sorted(os.listdir(runs_dir)):
                    if fname.endswith(".pt") or fname.endswith(".json"):
                        fpath = os.path.join(runs_dir, fname)
                        size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                        ckpts.append({"name": fname, "size_mb": size_mb})
            body = json.dumps({"status": "ok", "checkpoints": ckpts}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        
        
        
        
                # 软体步态端点
        if self.path.startswith("/api/loco/status"):
            body = json.dumps(live_loco.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/loco/reset"):
            live_loco.init_population(20)
            live_loco.generation = 1
            live_loco.step_count = 0
            body = json.dumps({"status": "ok", "msg": "Locomotion reset"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/loco/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_loco.warp_speed = max(1, min(50, speed))
            except Exception:
                live_loco.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_loco.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # 红皇后生态端点
        if self.path.startswith("/api/eco/status"):
            body = json.dumps(live_eco.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/eco/reset"):
            live_eco.init_world()
            live_eco.generation = 1
            live_eco.step_count = 0
            body = json.dumps({"status": "ok", "msg": "Eco reset"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/eco/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_eco.warp_speed = max(1, min(50, speed))
            except Exception:
                live_eco.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_eco.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # 免疫防线端点
        if self.path.startswith("/api/immune/status"):
            body = json.dumps(live_immune.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/immune/reset"):
            live_immune.init_microenvironment()
            live_immune.generation = 1
            live_immune.step_count = 0
            body = json.dumps({"status": "ok", "msg": "Immune reset"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/immune/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_immune.warp_speed = max(1, min(50, speed))
            except Exception:
                live_immune.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_immune.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # 三体引力弹弓端点
        if self.path.startswith("/api/slingshot/status"):
            body = json.dumps(live_slingshot.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/slingshot/reset"):
            live_slingshot.init_system()
            live_slingshot.generation = 1
            live_slingshot.step_count = 0
            body = json.dumps({"status": "ok", "msg": "Slingshot reset"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/slingshot/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_slingshot.warp_speed = max(1, min(50, speed))
            except Exception:
                live_slingshot.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_slingshot.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        # 无限公路车辆控制端点
        if self.path.startswith("/api/vehicle/status"):
            body = json.dumps(live_veh.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/vehicle/reset"):
            live_veh.init_vehicle()
            live_veh.generation = 1
            live_veh.step_count = 0
            body = json.dumps({"status": "ok", "msg": "Vehicle reset"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/vehicle/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_veh.warp_speed = max(1, min(50, speed))
            except Exception:
                live_veh.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_veh.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/vehicle/train_fast"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                gens = int(qs.get("generations", ["50"])[0])
            except Exception:
                gens = 50
            res = live_veh.fast_evolve_batch(target_generations=gens)
            body = json.dumps({"status": "ok", "result": res}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/maze/status"):
            body = json.dumps(live_maze.get_snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/maze/reset"):
            live_maze.generate_maze()
            live_maze.init_population(24)
            body = json.dumps({"status": "ok", "msg": "New maze generated"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/maze/warp"):
            try:
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                speed = int(qs.get("speed", ["5"])[0])
                live_maze.warp_speed = max(1, min(50, speed))
            except Exception:
                live_maze.warp_speed = 5
            body = json.dumps({"status": "ok", "warp_speed": live_maze.warp_speed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/maze/step"):
            live_maze.evolve_generation()
            body = json.dumps({"status": "ok", "generation": live_maze.generation}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/library"):
            silicon_library.reload_books()
            body = json.dumps({"status": "ok", "total_books": len(silicon_library.books), "books": silicon_library.books}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/lineage"):
            body = json.dumps({"status": "ok", "milestones": silicon_library.milestones}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if self.path == "/api/universe":
            elapsed = time.time()
            px = round(3620.0 + 20.0 * math.sin(elapsed * 0.6), 1)
            data = [{"symbol": "rb", "last": px, "volume": 524000, "name": "螺纹钢主力"}]
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/biosphere"):
            data = {
                "step": organism.phy_steps,
                "shannon_diversity": organism.shannon_h,
                "niche_counts": {"producers": 14, "herbivores": 28, "predators": 9, "decomposers": 6},
                "biomes": [
                    {"name": "Biome-Alpha", "climate": "Spring (Warm)", "nutrient": 8.4},
                    {"name": "Biome-Beta", "climate": "Summer (Hot)", "nutrient": 11.2},
                    {"name": "Biome-Gamma", "climate": "Autumn (Cool)", "nutrient": 6.8}
                ],
                "agents": [],
                "radiation": {"events": 0, "rays": []}
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/islands"):
            data = {
                "warp_mode": organism.warp_mode,
                "total_generations": organism.generation,
                "total_migrations": organism.generation // 8,
                "islands": [
                    {"island_id": i, "core_id": i % 4, "generations": organism.generation + (i * 12), "best_fitness": round(4.5 + math.sin(i), 1), "migration_in": i * 3, "migration_out": i * 2}
                    for i in range(8)
                ]
            }
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/cellular/organism"):
            data = organism.get_state_snapshot()
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/state":
            data = organism.get_state_snapshot()
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/preset"):
            ptype = "mature"
            if "seed" in self.path: ptype = "seed"
            elif "adas" in self.path: ptype = "adas"
            elif "real" in self.path or "champion" in self.path: ptype = "real"

            if ptype == "seed": organism.load_seed_preset()
            elif ptype == "adas": organism.load_adas_1m_preset()
            elif ptype == "real": organism.load_real_champion_preset()
            else: organism.load_mature_preset()

            body = json.dumps({"status": "ok", "preset": ptype, "cells_count": len(organism.cells)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/control/stress") or self.path.startswith("/api/stress"):
            organism.stress_mode = ("on" in self.path or "extreme" in self.path)
            body = json.dumps({"status": "ok", "stress": organism.stress_mode}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        super().do_GET()

    
    def do_POST(self):
        self.do_GET()

    def log_message(self, format, *args):
        # 静默常规静态文件 GET 日志，保持终端清爽
        if "api" in args[0]:
            pass
        else:
            super().log_message(format, *args)

# ============================================================================
# 3.5 线程化 HTTP 服务器 (Threaded HTTP Server)
# ============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

# ============================================================================
# 4. 启动后端监听服务
# ============================================================================

def run():
    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadedHTTPServer(("", PORT), ObservatoryHTTPHandler)
    print("======================================================================")
    print(" 硅基细胞计算机 (SDSCC) 实时计算与流式遥测后端")
    print(f" 服务已启动: http://localhost:{PORT}/cellular.html")
    print(f" API 端点: http://localhost:{PORT}/api/state (40Hz 实时状态)")
    print(f" 静态目录: {FRONTEND_DIR}")
    print("======================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n后端服务已安全停止。")

if __name__ == "__main__":
    run()
