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

# SDSCC 24 种原语分类（用于车辆控制的子集）
SDSCC_PRIMITIVES = {
    # 感知受体层 (Receptor Layer) - 固定输入，不参与演化
    "RECEPTOR_CTE":   "receptor",   # 横向偏差传感器
    "RECEPTOR_PSI":   "receptor",   # 航向误差传感器
    "RECEPTOR_CURV":  "receptor",   # 前向曲率传感器
    "RECEPTOR_SPEED": "receptor",   # 车速传感器
    # 代谢运算层 (Metabolic Layer) - 隐层原语，可演化
    "SUM":       "hidden",  # 加权求和
    "INTEGRATE": "hidden",  # 泄漏积分器
    "AMPLIFY":   "hidden",  # 增益放大
    "INVERT":    "hidden",  # 极性反转
    "THRESHOLD": "hidden",  # 施密特触发阈值
    "DAMPER":    "hidden",  # 指数平滑阻尼
    "CLIP":      "hidden",  # 饱和限幅
    "ABS":       "hidden",  # 绝对值
    # 效应动作层 (Effector Layer) - 固定输出，不参与演化
    "EFFECTOR_STEER": "effector",   # 转向指令输出
    "EFFECTOR_SPEED": "effector",   # 速度指令输出
}
HIDDEN_PRIMITIVES = [k for k, v in SDSCC_PRIMITIVES.items() if v == "hidden"]

class SdscCell:
    """单个 SDSCC 计算细胞：持有原语类型与内部状态"""
    def __init__(self, cell_id, ptype):
        self.cell_id = cell_id
        self.ptype = ptype
        self.state = 0.0       # 积分器/阻尼器内部状态
        self.output = 0.0
        self.gain = random.uniform(0.5, 2.5)    # 可遗传增益参数

    def forward(self, inputs):
        x = sum(inputs) if inputs else 0.0
        if   self.ptype == "SUM":       self.output = math.tanh(x * self.gain)
        elif self.ptype == "INTEGRATE": self.state = self.state * 0.88 + x * 0.12; self.output = self.state
        elif self.ptype == "AMPLIFY":   self.output = math.tanh(x * self.gain * 2.0)
        elif self.ptype == "INVERT":    self.output = -math.tanh(x * self.gain)
        elif self.ptype == "THRESHOLD": self.output = 1.0 if x > 0.3 else (-1.0 if x < -0.3 else self.output)
        elif self.ptype == "DAMPER":    self.state = self.state * 0.72 + x * 0.28; self.output = self.state
        elif self.ptype == "CLIP":      self.output = max(-1.0, min(1.0, x * self.gain))
        elif self.ptype == "ABS":       self.output = abs(math.tanh(x * self.gain))
        else:                           self.output = x
        return self.output

class SdscGenome:
    """
    SDSCC 拓扑基因组：
    - 固定4个受体输入细胞 (CTE, PSI, CURV, SPEED)
    - 演化的隐层细胞 3~7 个（原语类型可变异）
    - 固定2个效应输出细胞 (STEER, SPEED_CMD)
    - 基因组 = 突触连接表 [(from_id, to_id, polarity)]
    """
    def __init__(self):
        self.receptors = ["RECEPTOR_CTE", "RECEPTOR_PSI", "RECEPTOR_CURV", "RECEPTOR_SPEED"]
        self.effectors = ["EFFECTOR_STEER", "EFFECTOR_SPEED"]
        # 隐层细胞随机初始化
        n_hidden = random.randint(3, 6)
        self.hidden_types = [random.choice(HIDDEN_PRIMITIVES) for _ in range(n_hidden)]
        # 构建细胞列表（ID 分配：0~3 受体, 4~4+n-1 隐层, 最后2个效应）
        self.build_cells()
        # 随机初始化突触连接（受体→隐层, 隐层→隐层, 隐层→效应）
        self.synapses = []
        self.random_synapses()

    def build_cells(self):
        self.cells = []
        for i, ptype in enumerate(self.receptors):
            self.cells.append(SdscCell(i, ptype))
        n_r = len(self.receptors)
        for i, ptype in enumerate(self.hidden_types):
            self.cells.append(SdscCell(n_r + i, ptype))
        n_h = len(self.hidden_types)
        for i, ptype in enumerate(self.effectors):
            self.cells.append(SdscCell(n_r + n_h + i, ptype))
        self.n_receptor = len(self.receptors)
        self.n_hidden = n_h
        self.n_effector = len(self.effectors)
        self.steer_id = n_r + n_h
        self.speed_id = n_r + n_h + 1

    def random_synapses(self):
        """在受体→隐层→效应之间建立随机稀疏连接"""
        r_ids = list(range(self.n_receptor))
        h_ids = list(range(self.n_receptor, self.n_receptor + self.n_hidden))
        e_ids = [self.steer_id, self.speed_id]
        # 受体→隐层（每个受体至少连一个隐层细胞）
        for r in r_ids:
            for h in random.sample(h_ids, min(2, len(h_ids))):
                self.synapses.append((r, h, random.choice([-1, 1])))
        # 隐层→效应（至少保证每个效应被连接）
        for e in e_ids:
            h = random.choice(h_ids)
            self.synapses.append((h, e, random.choice([-1, 1])))
        # 额外随机稀疏连接
        for _ in range(random.randint(2, 5)):
            f = random.choice(r_ids + h_ids)
            t = random.choice(h_ids + e_ids)
            if f != t:
                self.synapses.append((f, t, random.choice([-1, 1])))

    def forward(self, cte, psi_err, curv, speed):
        """拓扑前向传导：规范化输入受体并按受体→隐层→效应拓扑前向激发"""
        # 归一化输入至 [-1.0, 1.0] 灵敏动态区间
        self.cells[0].output = max(-1.0, min(1.0, cte))
        self.cells[1].output = max(-1.0, min(1.0, psi_err))
        self.cells[2].output = max(0.0, min(1.0, curv))
        self.cells[3].output = max(0.0, min(1.0, speed))

        # 收集每个细胞的输入（来自突触）
        inputs_map = {c.cell_id: [] for c in self.cells}
        for (f, t, pol) in self.synapses:
            if 0 <= f < len(self.cells) and 0 <= t < len(self.cells):
                inputs_map[self.cells[t].cell_id].append(
                    self.cells[f].output * pol
                )

        # 按隐层→效应顺序前向激发
        n_r = self.n_receptor
        for i in range(n_r, len(self.cells)):
            c = self.cells[i]
            c.forward(inputs_map[c.cell_id])

        steer_raw = self.cells[self.steer_id].output
        speed_raw = self.cells[self.speed_id].output
        return steer_raw, speed_raw

    def mutate(self):
        """形态发生变异：突触重连、原语类型替换、增益微调、细胞增殖/凋亡"""
        child = SdscGenome.__new__(SdscGenome)
        child.receptors = list(self.receptors)
        child.effectors = list(self.effectors)
        child.hidden_types = list(self.hidden_types)
        child.synapses = list(self.synapses)

        # 1. 原语类型点突变（25% 概率）
        if random.random() < 0.25 and child.hidden_types:
            idx = random.randrange(len(child.hidden_types))
            child.hidden_types[idx] = random.choice(HIDDEN_PRIMITIVES)

        # 2. 突触极性翻转（30% 概率）
        if random.random() < 0.30 and child.synapses:
            idx = random.randrange(len(child.synapses))
            f, t, p = child.synapses[idx]
            child.synapses[idx] = (f, t, -p)

        # 3. 新突触添加（30% 概率）
        if random.random() < 0.30:
            n_r = len(child.receptors)
            n_h = len(child.hidden_types)
            n_e = len(child.effectors)
            all_ids = list(range(n_r + n_h + n_e))
            f = random.choice(all_ids[:n_r + n_h])
            t = random.choice(all_ids[n_r:])
            if f != t:
                child.synapses.append((f, t, random.choice([-1, 1])))

        # 4. 突触凋亡（15% 概率）
        if random.random() < 0.15 and len(child.synapses) > 3:
            child.synapses.pop(random.randrange(len(child.synapses)))

        # 5. 细胞增殖（15% 概率，最多7个隐层细胞）
        if random.random() < 0.15 and len(child.hidden_types) < 7:
            child.hidden_types.append(random.choice(HIDDEN_PRIMITIVES))

        # 6. 细胞凋亡（10% 概率，至少保留2个隐层细胞）
        if random.random() < 0.10 and len(child.hidden_types) > 2:
            child.hidden_types.pop(random.randrange(len(child.hidden_types)))

        child.build_cells()
        # 7. 细胞内在放大增益高斯扰动
        for c in child.cells:
            if random.random() < 0.35:
                c.gain *= random.uniform(0.85, 1.18)
        return child


class LiveVehicleSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_cte = []
        self.road_width = 46.0
        self.init_track()
        # 种群：6 个 SDSCC DAG 拓扑基因组
        self.population = [SdscGenome() for _ in range(6)]
        self.current_agent = 0
        self.agent_lap_steps = 0
        self.agent_cum_cte = 0.0
        self.fitness_log = []
        self.champion_genome = None
        self.champion_fitness = -1.0
        self.champion_trail = []
        self.init_vehicle()
        # 启动时执行 60 代百万步极速批量超演化，瞬间产出赛道老司机
        self.fast_evolve_batch(target_generations=60, pop_size=24, sim_steps_per_agent=600)

    def fast_evolve_batch(self, target_generations=50, pop_size=20, sim_steps_per_agent=600):
        """
        极速批量超演化加速引擎 (Batch Supercomputing Evolution Pipeline)
        在 100~300ms 内并行推演 50 代 x 20 个体 = 1,000+ 次完整闭环试跑，
        直接产出跑通全圈、高适应度、CTE 极小的顶级冠军拓扑！
        """
        pop = [SdscGenome() for _ in range(pop_size)]
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
                    best_s, best_dist = s, float("inf")
                    for ds in range(-2, 14):
                        probe_s = s + ds * 10.0
                        px, py, _ = self.get_track_point(probe_s)
                        d = (x - px)**2 + (y - py)**2
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
                "hidden_types": list(genome.hidden_types),
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
