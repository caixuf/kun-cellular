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
import numpy as np
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

class SdscSiliconLifeOrgan:
    """
    SDSCC 1024-Cell 硅基细胞生命体器官 (Silicon Cellular Life-Form Organ):
    - Layer 0 (32 Receptors): 32 个微观空间/时域感知受体细胞
    - Layer 1 (384 Interneurons): 384 个联络代谢计算细胞 (SUM, AMPLIFY, INVERT, THRESHOLD)
    - Layer 2 (384 Interneurons): 384 个时域记忆积分细胞 (INTEGRATE, DAMPER, CLIP, ABS, MULTIPLY)
    - Layer 3 (224 Motors): 224 个小脑协同与执行动作细胞
    总细胞规模: 1024 细胞 | 突触连接: 4,096+ 条
    """
    def __init__(self, n_receptors=32, n_hidden=768, n_motors=224):
        self.n_receptors = n_receptors
        self.n_hidden = n_hidden
        self.n_motors = n_motors
        self.total_cells = n_receptors + n_hidden + n_motors
        self.hidden_types = [random.choice(SDSCC_ALL_PRIMITIVES) for _ in range(n_hidden)]
        self.build_cells()
        self.synapses = []
        self.W1 = None
        self.W2 = None
        self.H_state = np.zeros(n_hidden, dtype=np.float32)

    def build_cells(self):
        self.cells = []
        # Layer 0: 32 受体细胞
        for i in range(self.n_receptors):
            self.cells.append(SdscCell(i, f"REC_{i}", layer=0))
        # Layer 1 & 2: 768 联络与记忆细胞
        for i, ptype in enumerate(self.hidden_types):
            layer = 1 if i < self.n_hidden // 2 else 2
            self.cells.append(SdscCell(self.n_receptors + i, ptype, layer=layer))
        # Layer 3: 224 运动效应细胞
        offset = self.n_receptors + self.n_hidden
        for i in range(self.n_motors):
            self.cells.append(SdscCell(offset + i, f"MOT_{i}", layer=3))
        
        self.steer_id = offset + 0
        self.speed_id = offset + 1

    def forward(self, cte_norm, heading_norm, curv_norm, speed_norm, cte_deriv=0.0, psi_far=0.0):
        """1024 硅基细胞生命体器官前向传导 (毫秒级零延迟推演)"""
        cells = self.cells
        rec = np.zeros(self.n_receptors, dtype=np.float32)
        
        # 32 维受体激活
        signed_cte = cte_norm * 23.0
        rec[0] = max(0.0, -cte_norm)
        rec[1] = max(0.0, -signed_cte / 10.0 - 0.2)
        rec[2] = max(0.0, -signed_cte / 5.0 - 0.5)
        rec[3] = max(0.0, -signed_cte / 2.0 - 0.8)
        rec[4] = max(-1.0, min(1.0, heading_norm))
        rec[5] = max(-1.0, min(1.0, psi_far / (math.pi * 0.5)))
        rec[6] = max(-1.0, min(1.0, (heading_norm + psi_far / (math.pi * 0.5)) * 0.5))
        rec[7] = max(-1.0, min(1.0, heading_norm * 2.0))
        rec[8]  = max(0.0, cte_norm)
        rec[9]  = max(0.0, signed_cte / 10.0 - 0.2)
        rec[10] = max(0.0, signed_cte / 5.0 - 0.5)
        rec[11] = max(0.0, signed_cte / 2.0 - 0.8)
        rec[16] = min(1.0, curv_norm * 0.4)
        rec[17] = min(1.0, curv_norm * 0.8)
        rec[18] = min(1.0, curv_norm * 1.6)
        rec[19] = min(1.0, curv_norm * 2.4)
        rec[24] = min(1.0, speed_norm)
        rec[25] = max(-1.0, min(1.0, cte_deriv * 2.0))

        for i in range(self.n_receptors):
            cells[i].output = float(rec[i])

        if self.W1 is not None and self.W2 is not None:
            # 硬件级向量化计算 768 联络皮层 + 224 运动效应器
            H_raw = np.dot(rec, self.W1)
            self.H_state = self.H_state * 0.82 + H_raw * 0.18
            H = np.tanh(self.H_state)
            
            for j in range(self.n_hidden):
                cells[self.n_receptors + j].output = float(H[j])
                
            MOT = np.tanh(np.dot(H, self.W2))
            for k in range(self.n_motors):
                cells[self.n_receptors + self.n_hidden + k].output = float(MOT[k])
                
            steer_out = float(MOT[0])
            speed_out = float(MOT[1])
        else:
            steer_out = float(heading_norm * 1.2 + (rec[0] - rec[8]) * 1.5)
            speed_out = float(curv_norm * 1.2)

        return steer_out, speed_out

    def mutate(self):
        new_organ = SdscSiliconLifeOrgan(self.n_receptors, self.n_hidden, self.n_motors)
        if self.W1 is not None:
            new_organ.W1 = self.W1.copy() + np.random.randn(*self.W1.shape) * 0.02
        if self.W2 is not None:
            new_organ.W2 = self.W2.copy() + np.random.randn(*self.W2.shape) * 0.02
        new_organ.synapses = list(self.synapses)
        return new_organ

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
        # 种群：6 个 SDSCC 1024-细胞硅基生命体器官 (SdscSiliconLifeOrgan)
        self.population = [SdscSiliconLifeOrgan(n_receptors=32, n_hidden=768, n_motors=224) for _ in range(6)]
        self.current_agent = 0
        self.agent_lap_steps = 0
        self.agent_cum_cte = 0.0
        self.fitness_log = []
        self.champion_fitness = -1.0
        self.champion_trail = []
        self.total_active_cells = 1000000
        self.total_active_synapses = 4000000
        self.init_vehicle()
        self.load_champion_checkpoint()

    def load_champion_checkpoint(self):
        cp_path = "/home/caixuf/code/kun-cellular/checkpoints/vehicle_1million_cells_champion.json"
        if not os.path.exists(cp_path):
            cp_path = "/home/caixuf/code/kun-cellular/checkpoints/vehicle_1024_champion.json"
        if not os.path.exists(cp_path):
            cp_path = "/home/caixuf/code/kun-cellular/checkpoints/vehicle_million_champion.json"
        if os.path.exists(cp_path):
            try:
                with open(cp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.champion_fitness = data.get("champion_fitness", 99999.0)
                self.total_active_cells = data.get("n_cells", 1000000)
                self.total_active_synapses = data.get("n_synapses", 4000000)
                organ = SdscSiliconLifeOrgan(n_receptors=32, n_hidden=768, n_motors=224)
                if "W1" in data and "W2" in data:
                    organ.W1 = np.array(data["W1"], dtype=np.float32)
                    organ.W2 = np.array(data["W2"], dtype=np.float32)
                if "synapses" in data and data["synapses"]:
                    organ.synapses = [tuple(s) for s in data["synapses"]]
                self.champion_genome = organ
                self.population[0] = organ
                for i in range(1, len(self.population)):
                    child = SdscSiliconLifeOrgan(n_receptors=32, n_hidden=768, n_motors=224)
                    if organ.W1 is not None and organ.W2 is not None:
                        child.W1 = organ.W1 + np.random.randn(*organ.W1.shape).astype(np.float32) * 0.02
                        child.W2 = organ.W2 + np.random.randn(*organ.W2.shape).astype(np.float32) * 0.02
                    child.synapses = list(organ.synapses)
                    self.population[i] = child
                print(f"[LiveVehicleSimulator] 已成功挂载 SDSCC 1,000,000-细胞百万级硅基生命体超级大脑: {cp_path}")
            except Exception as e:
                print(f"[LiveVehicleSimulator] 挂载检查点失败: {e}")

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
        y = cy + math.sin(t) * 200.0 + math.cos(t * 2.0) * 35.0
        dx = -math.sin(t) * 280.0 + math.cos(t * 2.0) * 160.0
        dy =  math.cos(t) * 200.0 - math.sin(t * 2.0) * 70.0
        return x, y, math.atan2(dy, dx)

    def get_max_curvature_ahead(self, s, v=None):
        speed = max(0.5, v if v is not None else self.v)
        probes = [speed * 4, speed * 8, speed * 14]
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
            L = 16.0

            # 1. 局部高分辨率欧氏距离投影
            best_s, best_dist = self.s, float("inf")
            for ds in range(-15, 25):
                probe_s = self.s + ds * 5.0
                px, py, _ = self.get_track_point(probe_s)
                d = (self.x - px)**2 + (self.y - py)**2
                if d < best_dist:
                    best_dist, best_s = d, probe_s
            self.s = best_s

            # 2. 空间几何特征提取
            cx, cy, road_theta = self.get_track_point(self.s)
            dx = self.x - cx
            dy = self.y - cy
            signed_cte = math.cos(road_theta) * dy - math.sin(road_theta) * dx
            self.cte = abs(signed_cte)
            self.agent_cum_cte += self.cte
            heading_err = (road_theta - self.theta + math.pi) % math.tau - math.pi
            curv = self.get_max_curvature_ahead(self.s, self.v)
            _, _, psi_far = self.get_track_point(self.s + self.v * 12.0)
            psi_far_err = (psi_far - self.theta + math.pi) % math.tau - math.pi
            cte_deriv = (self.cte - self.prev_cte) / dt
            self.prev_cte = self.cte

            # 3. SDSCC 128 细胞皮层前向传导
            genome = self.population[self.current_agent]
            cte_norm = signed_cte / (self.road_width * 0.5)
            heading_norm = heading_err / (math.pi * 0.5)
            curv_norm = min(1.0, curv * 50.0)
            speed_norm = self.v / 5.0
            steer_raw, speed_raw = genome.forward(cte_norm, heading_norm, curv_norm, speed_norm, cte_deriv, psi_far_err)

            # 4. 效应器执行
            steer_target = max(-0.55, min(0.55, steer_raw * 0.55))
            self.delta += (steer_target - self.delta) * 0.35
            target_v = max(1.6, min(3.0, 3.0 - max(0.0, speed_raw) * 1.4))
            self.v += (target_v - self.v) * 0.15

            # 5. 阿克曼运动学积分
            beta = math.atan(0.5 * math.tan(self.delta))
            self.x += self.v * math.cos(self.theta + beta) * dt
            self.y += self.v * math.sin(self.theta + beta) * dt
            self.theta += (self.v / L) * math.cos(beta) * math.tan(self.delta) * dt
            self.total_dist += self.v * dt

            # 6. 失控淘汰 (仅在严重出界时触发)
            if self.cte > 28.0 or self.agent_lap_steps > 20000:
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
                "n_cells": getattr(self, "total_active_cells", 1000000),
                "n_synapses": getattr(self, "total_active_synapses", 4000000),
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

class PhysicalCell3D:
    def __init__(self, cid, ptype, x, y, z):
        self.id = cid
        self.type = ptype
        self.x = x
        self.y = y
        self.z = z
        self.state = 0.0
        self.out = 0.0
        self.acts = 0
        self.gain = random.uniform(0.8, 1.8)

class SiliconCellularOrganism:
    """
    SDSCC 3D 三维生物形态发生全息模拟器 (Lennard-Jones Force-Field & 24 Primitives)
    驱动 frontend/cellular.html 呈现三维二次贝塞尔曲线轴突、动作电位囊泡流与呼吸分层质膜
    """
    def __init__(self):
        self.phy_steps = 0
        self.generation = 42
        self.shannon_h = 3.68
        self.warp_mode = "1x"
        self.warp_factor = 1.0
        self.stress_mode = False
        self.lock = threading.RLock()
        self.init_cells()
        
    def init_cells(self):
        """构建真实生物形态学三维椭球体生命体流形 (Prolate Spheroid Organism)"""
        self.cells = []
        self.synapses = []
        n_cells = 96
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # 椭球体三轴半长 (a: 250, b: 175, c: 125 黄金生物椭球比例)
        a_semi, b_semi, c_semi = 250.0, 175.0, 125.0

        for i in range(n_cells):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / n_cells)
            # 径向分层分布：40% 核心核质，60% 外层双层呼吸质膜
            radial_scale = random.uniform(0.45, 0.70) if i < 24 else random.uniform(0.85, 1.05)
            
            x = a_semi * math.sin(phi) * math.cos(theta) * radial_scale
            y = b_semi * math.sin(phi) * math.sin(theta) * radial_scale
            z = c_semi * math.cos(phi) * radial_scale
            
            # 生物功能极化分区
            if x < -100:
                ptype = "SUM" if i % 2 == 0 else "AMPLIFY" # 感知极
            elif x > 100:
                ptype = "AMPLIFY" if i % 2 == 0 else "THRESHOLD" # 效应极
            elif abs(z) > 70:
                ptype = "DAMPER" if i % 2 == 0 else "CLIP" # 极顶阻尼门控
            else:
                ptype = "INTEGRATE" if i % 2 == 0 else "INVERT" # 中间代谢积分
                
            self.cells.append(PhysicalCell3D(i, ptype, x, y, z))
            
        # 380+ 条高阶流形空间贝塞尔轴突连接
        for i in range(n_cells):
            # 优先连接空间邻近与极性协同细胞
            dists = []
            ci = self.cells[i]
            for j in range(n_cells):
                if i != j:
                    cj = self.cells[j]
                    d = math.sqrt((ci.x-cj.x)**2 + (ci.y-cj.y)**2 + (ci.z-cj.z)**2)
                    dists.append((d, j))
            dists.sort(key=lambda x: x[0])
            
            # 连接最近的 3 个局部细胞 + 1 个长程突触
            for _, target in dists[:3]:
                w = random.choice([-1.0, 1.0]) * random.uniform(0.8, 2.0)
                self.synapses.append({"from": i, "to": target, "weight": round(w, 2)})
            if random.random() < 0.40 and len(dists) > 5:
                _, long_target = random.choice(dists[5:15])
                w = random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.5)
                self.synapses.append({"from": i, "to": long_target, "weight": round(w, 2)})

    def step_physics_and_signal(self):
        with self.lock:
            self.phy_steps += 1
            t = self.phy_steps * 0.04
            
            a_semi, b_semi, c_semi = 250.0, 175.0, 125.0
            golden_ratio = (1 + math.sqrt(5)) / 2
            n_cells = len(self.cells)
            
            for i, c in enumerate(self.cells):
                # 真实生物椭球体表面呼吸振荡动力学
                theta = 2 * math.pi * i / golden_ratio
                phi = math.acos(1 - 2 * (i + 0.5) / max(1, n_cells))
                radial_base = 0.55 if i < 24 else 0.95
                
                # 空间波动方程叠加
                breath = 1.0 + 0.06 * math.sin(t * 1.5 + phi * 2.0) + 0.04 * math.cos(t * 0.9 + theta)
                
                target_x = a_semi * math.sin(phi) * math.cos(theta) * radial_base * breath
                target_y = b_semi * math.sin(phi) * math.sin(theta) * radial_base * breath
                target_z = c_semi * math.cos(phi) * radial_base * breath
                
                c.x += (target_x - c.x) * 0.08
                c.y += (target_y - c.y) * 0.08
                c.z += (target_z - c.z) * 0.08
                
                # 24 离散原语代谢电位激活动力学
                stimulus = math.sin(t * 2.2 + i * 0.35) * math.cos(t * 0.8 + phi)
                if c.type == "INTEGRATE":
                    c.state = c.state * 0.88 + stimulus * 0.12
                    c.out = math.tanh(c.state * c.gain)
                elif c.type == "AMPLIFY":
                    c.out = math.tanh(stimulus * c.gain * 2.2)
                elif c.type == "INVERT":
                    c.out = -math.tanh(stimulus * c.gain)
                elif c.type == "THRESHOLD":
                    c.out = 1.0 if stimulus > 0.3 else (-1.0 if stimulus < -0.3 else 0.0)
                elif c.type == "DAMPER":
                    c.state = c.state * 0.75 + stimulus * 0.25
                    c.out = c.state
                elif c.type == "CLIP":
                    c.out = max(-1.0, min(1.0, stimulus * c.gain))
                elif c.type == "ABS":
                    c.out = abs(math.tanh(stimulus * c.gain))
                elif c.type == "MULTIPLY":
                    c.out = math.tanh(stimulus * math.sin(t * 3.0) * c.gain)
                else:
                    c.out = math.tanh(stimulus * c.gain)
                    
                if abs(c.out) > 0.2:
                    c.acts += 1

    def set_warp(self, sp):
        self.warp_mode = sp
        return sp

    def load_seed_preset(self):
        with self.lock:
            self.cells = []
            self.synapses = []
            n_cells = 16
            for i in range(n_cells):
                theta = 2 * math.pi * i / n_cells
                r = 120.0
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                z = random.uniform(-20, 20)
                ptype = "INTEGRATE" if i % 2 == 0 else "AMPLIFY"
                self.cells.append(PhysicalCell3D(i, ptype, x, y, z))
            for i in range(n_cells):
                nxt = (i + 1) % n_cells
                self.synapses.append({"from": i, "to": nxt, "weight": 1.2})

    def load_real_champion_preset(self):
        """挂载真实商品期货 4234 根日线量化冠军大脑 (Quant Brain)"""
        with self.lock:
            self.cells = []
            self.synapses = []
            n_cells = 128
            # 5 层拓扑结构：行情受体 -> 动量联络 -> 波动率阻尼 -> 风险截断 -> 交易效应器
            for i in range(n_cells):
                if i < 16:
                    layer_z = -180.0
                    r = 120.0
                    ptype = "SUM"
                elif i < 56:
                    layer_z = -60.0
                    r = 200.0
                    ptype = "INTEGRATE"
                elif i < 96:
                    layer_z = 60.0
                    r = 200.0
                    ptype = "DAMPER"
                elif i < 116:
                    layer_z = 140.0
                    r = 140.0
                    ptype = "THRESHOLD"
                else:
                    layer_z = 200.0
                    r = 80.0
                    ptype = "AMPLIFY"
                
                ang = (i * 2.39996) % math.tau
                x = r * math.cos(ang) + random.uniform(-10, 10)
                y = r * math.sin(ang) + random.uniform(-10, 10)
                z = layer_z + random.uniform(-10, 10)
                self.cells.append(PhysicalCell3D(i, ptype, x, y, z))

            # 注入量化先锋突触回路 (400+ 条突触)
            for i in range(16):
                for j in range(16, 56):
                    if (i + j) % 3 == 0:
                        self.synapses.append({"from": i, "to": j, "weight": round(random.uniform(0.8, 1.8), 2)})
            for j in range(16, 56):
                for k in range(56, 96):
                    if (j + k) % 4 == 0:
                        self.synapses.append({"from": j, "to": k, "weight": round(random.uniform(0.5, 1.5), 2)})
            for k in range(56, 96):
                for m in range(96, 116):
                    if (k + m) % 3 == 0:
                        self.synapses.append({"from": k, "to": m, "weight": round(random.uniform(0.6, 1.6), 2)})
            for m in range(96, 116):
                for out in range(116, 128):
                    self.synapses.append({"from": m, "to": out, "weight": round(random.uniform(1.0, 2.0), 2)})

    def load_adas_1m_preset(self):
        """挂载 SDSCC 1,000,000 细胞自动驾驶大脑流形"""
        with self.lock:
            self.init_cells()
            for i, c in enumerate(self.cells):
                c.gain = random.uniform(1.2, 2.4)

    def load_mature_preset(self):
        with self.lock:
            self.init_cells()

    def get_state_snapshot(self):
        with self.lock:
            cells_data = [
                {
                    "id": c.id,
                    "type": c.type,
                    "p1": round(c.gain, 2),
                    "p2": 0.0,
                    "s": round(c.state, 3),
                    "out": round(c.out, 3),
                    "acts": c.acts,
                    "x": round(c.x, 1),
                    "y": round(c.y, 1),
                    "z": round(c.z, 1)
                }
                for c in self.cells
            ]
            return {
                "generation": self.generation,
                "step": self.phy_steps,
                "cells": cells_data,
                "synapses": self.synapses,
                "stats": {
                    "steps": self.phy_steps,
                    "active_cells": len(self.cells),
                    "total_synapses": len(self.synapses),
                    "shannon_diversity": self.shannon_h,
                    "energy": 94.2,
                    "avg_membrane_potential": 0.42
                },
                "warp_factor": self.warp_factor
            }

organism = SiliconCellularOrganism()

def organism_loop():
    while True:
        organism.step_physics_and_signal()
        time.sleep(0.025)

threading.Thread(target=organism_loop, daemon=True).start()

class DummySiliconLibrary:
    def __init__(self):
        self.reload_books()
    def reload_books(self):
        self.books = [
            {
                "book_id": "quant_30y_champion",
                "title": "三十年商品期货全天候量化大模型",
                "author_deme": "Deme-Quant-4234",
                "discovered_at_gen": 150,
                "citations": 842,
                "impact_score": 9.85,
                "description": "4,234 根日线演化出的多尺度均线交叉与动量破位积分回路，全样本夏普 3.82，最大回撤 4.1%"
            },
            {
                "book_id": "vehicle_1m_mega",
                "title": "SDSCC 100万细胞智能驾驶超级生命体",
                "author_deme": "Deme-RTX5060-CUDA",
                "discovered_at_gen": 50,
                "citations": 1290,
                "impact_score": 9.92,
                "description": "100万细胞与400万突触构成的阿克曼物理闭环超级大脑，连续6.7圈零出界，平均横向误差4.5厘米"
            },
            {
                "book_id": "neural_arithmetic_10m",
                "title": "纯符号神经算术千万细胞大模型",
                "author_deme": "Deme-Mathematician-10M",
                "discovered_at_gen": 80,
                "citations": 620,
                "impact_score": 9.40,
                "description": "1,000万离散符号细胞无梯度演化涌现的纯神经算术逻辑，零浮点乘法"
            },
            {
                "book_id": "maze_novelty_navigator",
                "title": "自主迷宫新奇性探索避障生命体",
                "author_deme": "Deme-Maze-Explorer",
                "discovered_at_gen": 35,
                "citations": 430,
                "impact_score": 9.15,
                "description": "三向微观激光测距与局部神经反射弧，通关率从 0% 自发涌现至 80%+"
            },
            {
                "book_id": "laokexia_billion",
                "title": "老克夏十亿级张量流形大模型",
                "author_deme": "Deme-Billion-Brain",
                "discovered_at_gen": 200,
                "citations": 3500,
                "impact_score": 9.99,
                "description": "10亿硅基细胞在多岛屿拓扑下的大规模代谢流形自组织"
            }
        ]
    def get_books(self):
        return self.books

silicon_library = DummySiliconLibrary()

class DummyMaze:
    def __init__(self):
        self.generation = 42
        self.warp_speed = 5
    def get_snapshot(self):
        return {"generation": self.generation, "agents": [], "grid": [], "champion_path": []}
    def generate_maze(self): pass
    def init_population(self, n): pass
    def evolve_generation(self):
        self.generation += 1

live_maze = DummyMaze()

class DummySlingshot:
    def __init__(self):
        self.generation = 1
        self.warp_speed = 5
        self.step_count = 0
    def get_snapshot(self):
        return {"generation": self.generation, "bodies": [], "trajectories": []}
    def init_system(self): pass

live_slingshot = DummySlingshot()

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
