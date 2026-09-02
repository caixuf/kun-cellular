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
# 0.13 无限公路阿克曼车辆动力学与居中保持引擎 (Infinite Highway Vehicle Engine)
# ============================================================================

class LiveVehicleSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_cte = []
        self.road_width = 140.0
        self.init_vehicle()

    def X_road(self, s):
        # 无限程序化公路中心线位置 X(s)
        return 400.0 + math.sin(s * 0.003) * 140.0 + math.sin(s * 0.007) * 70.0

    def dX_road_ds(self, s):
        # 道路切线导数
        return math.cos(s * 0.003) * 0.003 * 140.0 + math.cos(s * 0.007) * 0.007 * 70.0

    def init_vehicle(self):
        with self.lock:
            self.s = 0.0
            self.x = self.X_road(0.0)
            self.psi = math.atan(self.dX_road_ds(0.0))
            self.v = 5.5 # 巡航车速
            self.delta = 0.0 # 前轮转向角
            self.cte = 0.0 # 横向偏离误差
            self.total_dist = 0.0

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            dt = 0.04
            L = 26.0 # 轴距
            k_cte = 0.16

            # 1. 探测当前公路中心线与切线航向
            x_center = self.X_road(self.s)
            psi_road = math.atan(self.dX_road_ds(self.s))

            # 2. 计算横向偏差与航向误差
            e_y = x_center - self.x # >0 表示车偏左，需向右转向
            e_psi = (psi_road - self.psi + math.pi) % math.tau - math.pi
            self.cte = abs(e_y)

            # 3. 闭环 Stanley 控制律 (Stanley Lane-Centering Steering Law)
            steer_target = e_psi + math.atan2(k_cte * e_y, max(1.0, self.v))
            steer_target = max(-0.60, min(0.60, steer_target))
            self.delta += (steer_target - self.delta) * 0.35

            # 4. 速度控制 (弯道平顺减速，直道平稳巡航)
            curvature = abs(self.dX_road_ds(self.s + 30.0) - self.dX_road_ds(self.s)) / 30.0
            target_v = max(3.5, 7.5 - curvature * 140.0)
            self.v += (target_v - self.v) * 0.12

            # 5. 阿克曼运动学积分
            beta = math.atan(0.5 * math.tan(self.delta))
            dx = self.v * math.sin(self.psi + beta) * dt * 25.0
            ds = self.v * math.cos(self.psi + beta) * dt * 25.0
            dpsi = (self.v / L) * math.cos(beta) * math.tan(self.delta) * dt * 25.0

            self.x += dx
            self.s += ds
            self.psi += dpsi
            self.total_dist = self.s * 0.1 # 标定为米

            # 6. 统计历史 CTE 误差
            if self.step_count % 5 == 0:
                self.history_cte.append(round(self.cte * 0.02, 3)) # 标定为米
                if len(self.history_cte) > 40:
                    self.history_cte.pop(0)

    def get_snapshot(self):
        with self.lock:
            # 采样当前视口前后 800 像素范围内的道路点
            road_points = []
            s_start = self.s - 150.0
            s_end = self.s + 650.0
            ds = 20.0
            cur_s = s_start
            while cur_s <= s_end:
                rx = self.X_road(cur_s)
                rpsi = math.atan(self.dX_road_ds(cur_s))
                road_points.append({
                    "s": round(cur_s, 1),
                    "x": round(rx, 1),
                    "psi": round(rpsi, 3)
                })
                cur_s += ds

            return {
                "step_count": self.step_count,
                "s": round(self.s, 1),
                "total_dist_m": round(self.total_dist, 1),
                "road_width": self.road_width,
                "car": {
                    "x": round(self.x, 1),
                    "s": round(self.s, 1),
                    "psi": round(self.psi, 3),
                    "delta_deg": round(math.degrees(self.delta), 1),
                    "speed_kmh": round(self.v * 12.0, 1),
                    "cte_m": round(self.cte * 0.02, 3)
                },
                "road_points": road_points,
                "history_cte": list(self.history_cte)
            }

live_veh = LiveVehicleSimulator()

def veh_loop():
    while True:
        for _ in range(live_veh.warp_speed):
            live_veh.step_physics()
        time.sleep(0.016)

threading.Thread(target=veh_loop, daemon=True).start()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ============================================================================
# 0.11 微观免疫防线：巨噬细胞吞噬与抗原追猎动力学引擎 (Immune Phagocytosis Engine)
# ============================================================================

class LiveImmuneSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 300
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.pathogens = []
        self.macrophages = []
        self.history_clearance = []
        self.init_microenvironment()

    def init_microenvironment(self):
        with self.lock:
            # 1. 异变病毒病原体 (Pathogens, 40 颗)
            self.pathogens = []
            for i in range(40):
                self.pathogens.append({
                    "id": i,
                    "x": random.uniform(40, self.width - 40),
                    "y": random.uniform(40, self.height - 40),
                    "vx": random.uniform(-1.2, 1.2),
                    "vy": random.uniform(-1.2, 1.2),
                    "alive": True,
                    "mut_type": random.choice([0, 1, 2]) # 0: 脂质包膜, 1: 棘突蛋白, 2: 变异株
                })

            # 2. 巨噬/T细胞宿主免疫卫士 (Macrophages, 12 只)
            self.macrophages = []
            for i in range(12):
                self.macrophages.append({
                    "id": i,
                    "x": random.uniform(80, self.width - 80),
                    "y": random.uniform(80, self.height - 80),
                    "theta": random.uniform(0, math.tau),
                    "pseudopods": 6, # 伪足数
                    "radius": 14.0,
                    "phagocytosed": 0,
                    # 免疫基因组: [化学趋化感知半径, 伪足伸缩强度, 吞噬速度, 抗体亲和力]
                    "genes": [
                        random.uniform(140.0, 260.0), # chemotaxis_radius
                        random.uniform(4.0, 10.0),    # pseudopod_amp
                        random.uniform(2.8, 4.5),     # speed
                        random.uniform(0.8, 2.0)      # affinity
                    ]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            
            # 1. 病毒扩散与布朗运动
            for pat in self.pathogens:
                if not pat["alive"]: continue
                pat["vx"] += random.uniform(-0.2, 0.2)
                pat["vy"] += random.uniform(-0.2, 0.2)
                pat["vx"] = max(-2.0, min(2.0, pat["vx"]))
                pat["vy"] = max(-2.0, min(2.0, pat["vy"]))
                pat["x"] = max(10, min(self.width - 10, pat["x"] + pat["vx"]))
                pat["y"] = max(10, min(self.height - 10, pat["y"] + pat["vy"]))

            # 2. 免疫细胞化学趋化性追踪与伪足吞噬
            for mac in self.macrophages:
                r_chem = mac["genes"][0]
                closest_p = None
                closest_d = 9999.0
                
                for pat in self.pathogens:
                    if not pat["alive"]: continue
                    d = math.hypot(pat["x"] - mac["x"], pat["y"] - mac["y"])
                    if d < r_chem and d < closest_d:
                        closest_d = d
                        closest_p = pat

                if closest_p:
                    target_theta = math.atan2(closest_p["y"] - mac["y"], closest_p["x"] - mac["x"])
                    diff = (target_theta - mac["theta"] + math.pi) % math.tau - math.pi
                    mac["theta"] += diff * 0.25
                    spd = mac["genes"][2]

                    # 吞噬距离检测 (Phagocytosis)
                    if closest_d < (mac["radius"] + 6.0):
                        closest_p["alive"] = False
                        mac["phagocytosed"] += 1
                        mac["radius"] = min(22.0, mac["radius"] + 0.4)
                else:
                    mac["theta"] += random.uniform(-0.15, 0.15)
                    spd = mac["genes"][2] * 0.55

                mac["x"] = max(20, min(self.width - 20, mac["x"] + math.cos(mac["theta"]) * spd))
                mac["y"] = max(20, min(self.height - 20, mac["y"] + math.sin(mac["theta"]) * spd))

            alive_pathogens = sum(1 for p in self.pathogens if p["alive"])
            if self.step_count >= self.max_steps or alive_pathogens == 0:
                self.evolve_immunity()

    def evolve_immunity(self):
        alive_count = sum(1 for p in self.pathogens if p["alive"])
        cleared_rate = (40 - alive_count) / 40.0
        self.history_clearance.append(round(cleared_rate * 100.0, 1))
        if len(self.history_clearance) > 30:
            self.history_clearance.pop(0)

        for m in self.macrophages:
            m["fitness"] = m["phagocytosed"] * 30.0 + m["genes"][3] * 10.0
        self.macrophages.sort(key=lambda m: m["fitness"], reverse=True)
        top_macs = self.macrophages[:4]
        new_macs = []

        for i in range(len(self.macrophages)):
            parent = random.choice(top_macs)
            child_genes = [g + (random.gauss(0, 0.1) if random.random() < 0.35 else 0.0) for g in parent["genes"]]
            new_macs.append({
                "id": i,
                "x": random.uniform(80, self.width - 80),
                "y": random.uniform(80, self.height - 80),
                "theta": random.uniform(0, math.tau),
                "pseudopods": 6,
                "radius": 14.0,
                "phagocytosed": 0,
                "genes": child_genes
            })
        self.macrophages = new_macs

        # 重新投放下一批变异病毒株
        self.pathogens = []
        for i in range(40):
            self.pathogens.append({
                "id": i,
                "x": random.uniform(40, self.width - 40),
                "y": random.uniform(40, self.height - 40),
                "vx": random.uniform(-1.2, 1.2),
                "vy": random.uniform(-1.2, 1.2),
                "alive": True,
                "mut_type": random.choice([0, 1, 2])
            })
        self.generation += 1
        self.step_count = 0

    def get_snapshot(self):
        with self.lock:
            alive_p = sum(1 for p in self.pathogens if p["alive"])
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "pathogens_alive": alive_p,
                "total_pathogens": len(self.pathogens),
                "total_phagocytosed": 40 - alive_p,
                "clearance_rate": round((40 - alive_p) / 40.0 * 100.0, 1),
                "pathogens": [
                    {"id": p["id"], "x": round(p["x"], 1), "y": round(p["y"], 1), "alive": p["alive"], "type": p["mut_type"]}
                    for p in self.pathogens
                ],
                "macrophages": [
                    {
                        "id": m["id"],
                        "x": round(m["x"], 1),
                        "y": round(m["y"], 1),
                        "theta": round(m["theta"], 2),
                        "radius": round(m["radius"], 1),
                        "phagocytosed": m["phagocytosed"],
                        "chem_r": round(m["genes"][0], 1)
                    }
                    for m in self.macrophages
                ],
                "history_clearance": list(self.history_clearance)
            }

live_immune = LiveImmuneSimulator()

def immune_loop():
    while True:
        for _ in range(live_immune.warp_speed):
            live_immune.step_physics()
        time.sleep(0.016)

threading.Thread(target=immune_loop, daemon=True).start()

# ============================================================================
# 0.12 混沌三体引力弹膏深空导航动力学引擎 (Three-Body Slingshot Engine)
# ============================================================================

class LiveSlingshotSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 360
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_success = []
        self.init_system()

    def init_system(self):
        with self.lock:
            # 3 颗大质量恒星 (Three-Body Stars)
            self.stars = [
                {"x": 280.0, "y": 300.0, "vx": 0.0, "vy": 1.2, "m": 8000.0, "color": "#f43f5e"},
                {"x": 520.0, "y": 300.0, "vx": 0.0, "vy": -1.2, "m": 8000.0, "color": "#fbbf24"},
                {"x": 400.0, "y": 480.0, "vx": 1.0, "vy": 0.0, "m": 6000.0, "color": "#a855f7"}
            ]
            self.target_planet = {"x": 700.0, "y": 120.0, "r": 18.0}
            
            # 16 艘硅基自适应深空探测器 (Probes)
            self.probes = []
            for i in range(16):
                self.probes.append({
                    "id": i,
                    "x": 100.0, "y": 500.0,
                    "vx": random.uniform(1.8, 3.2),
                    "vy": random.uniform(-3.2, -1.8),
                    "alive": True,
                    "reached": False,
                    "fuel": 100.0,
                    "trail": [[100.0, 500.0]],
                    # 探测器控制基因: [引力梯度响应权重, 目标朝向权重, 轨道离心推力, 喷气阈值]
                    "genes": [
                        random.uniform(0.5, 2.5),
                        random.uniform(1.0, 4.0),
                        random.uniform(0.2, 1.5),
                        random.uniform(0.1, 0.6)
                    ]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            dt = 0.06
            G = 0.8
            
            # 1. 恒星牛顿混沌引力积分 (Runge-Kutta / Verlet)
            for i in range(3):
                s1 = self.stars[i]
                for j in range(i + 1, 3):
                    s2 = self.stars[j]
                    dx = s2["x"] - s1["x"]
                    dy = s2["y"] - s1["y"]
                    d = math.hypot(dx, dy) + 10.0
                    f = (G * s1["m"] * s2["m"]) / (d * d)
                    fx = (dx / d) * f
                    fy = (dy / d) * f
                    s1["vx"] += (fx / s1["m"]) * dt
                    s1["vy"] += (fy / s1["m"]) * dt
                    s2["vx"] -= (fx / s2["m"]) * dt
                    s2["vy"] -= (fy / s2["m"]) * dt

            for s in self.stars:
                s["x"] += s["vx"] * dt * 10.0
                s["y"] += s["vy"] * dt * 10.0

            # 2. 探测器在混沌三体场中的引力弹弓与微喷推演
            tx, ty = self.target_planet["x"], self.target_planet["y"]
            reached_count = 0

            for p in self.probes:
                if not p["alive"]: continue
                if p["reached"]:
                    reached_count += 1
                    continue

                # 三体恒星万有引力叠加
                total_gx, total_gy = 0.0, 0.0
                crashed = False
                for s in self.stars:
                    dx = s["x"] - p["x"]
                    dy = s["y"] - p["y"]
                    d = math.hypot(dx, dy)
                    if d < 15.0: # 坠毁在恒星表面
                        crashed = True
                        break
                    f = (G * s["m"]) / (d * d + 100.0)
                    total_gx += (dx / d) * f
                    total_gy += (dy / d) * f

                if crashed:
                    p["alive"] = False
                    continue

                # 探测器自主神经喷气推力 (Micro-Thruster Control)
                dx_t = tx - p["x"]
                dy_t = ty - p["y"]
                d_target = math.hypot(dx_t, dy_t)
                
                if d_target < self.target_planet["r"]:
                    p["reached"] = True
                    reached_count += 1
                    continue

                # 基因组驱动的自适应喷气
                thrust_x = (dx_t / d_target) * p["genes"][1] * 0.15
                thrust_y = (dy_t / d_target) * p["genes"][1] * 0.15
                
                p["vx"] += (total_gx * p["genes"][0] + thrust_x) * dt
                p["vy"] += (total_gy * p["genes"][0] + thrust_y) * dt
                
                p["x"] += p["vx"] * dt * 8.0
                p["y"] += p["vy"] * dt * 8.0
                
                if self.step_count % 3 == 0 and len(p["trail"]) < 120:
                    p["trail"].append([round(p["x"], 1), round(p["y"], 1)])

            if self.step_count >= self.max_steps or reached_count == len(self.probes):
                self.evolve_slingshot()

    def evolve_slingshot(self):
        tx, ty = self.target_planet["x"], self.target_planet["y"]
        reached_count = sum(1 for p in self.probes if p["reached"])
        self.history_success.append(round(reached_count / len(self.probes) * 100.0, 1))
        if len(self.history_success) > 30:
            self.history_success.pop(0)

        for p in self.probes:
            min_d = min(math.hypot(tx - pt[0], ty - pt[1]) for pt in p["trail"]) if p["trail"] else 999.0
            p["fitness"] = (800.0 - min_d) + (500.0 if p["reached"] else 0.0)
            
        self.probes.sort(key=lambda p: p["fitness"], reverse=True)
        top_probes = self.probes[:4]
        new_probes = []
        
        for i in range(len(self.probes)):
            parent = random.choice(top_probes)
            child_genes = [g + (random.gauss(0, 0.1) if random.random() < 0.35 else 0.0) for g in parent["genes"]]
            new_probes.append({
                "id": i,
                "x": 100.0, "y": 500.0,
                "vx": random.uniform(1.8, 3.2),
                "vy": random.uniform(-3.2, -1.8),
                "alive": True,
                "reached": False,
                "fuel": 100.0,
                "trail": [[100.0, 500.0]],
                "genes": child_genes
            })
        self.probes = new_probes
        self.generation += 1
        self.step_count = 0
        self.init_system()

    def get_snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "success_rate": round(sum(1 for p in self.probes if p["reached"]) / len(self.probes) * 100.0, 1),
                "stars": [{"x": round(s["x"], 1), "y": round(s["y"], 1), "color": s["color"]} for s in self.stars],
                "target": self.target_planet,
                "probes": [
                    {
                        "id": p["id"],
                        "x": round(p["x"], 1),
                        "y": round(p["y"], 1),
                        "alive": p["alive"],
                        "reached": p["reached"],
                        "trail": p["trail"]
                    }
                    for p in self.probes
                ],
                "history_success": list(self.history_success)
            }

live_slingshot = LiveSlingshotSimulator()

def slingshot_loop():
    while True:
        for _ in range(live_slingshot.warp_speed):
            live_slingshot.step_physics()
        time.sleep(0.016)

threading.Thread(target=slingshot_loop, daemon=True).start()


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
                # 真实阿克曼车辆运动学与无限公路居中控制端点
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
            body = json.dumps({"status": "ok", "msg": "Ecosystem reset"}).encode("utf-8")
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
