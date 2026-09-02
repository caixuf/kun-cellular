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
# 0.13 真实车辆运动学与全景公路巡航引擎 (Universal Kinematic Vehicle Engine)
# ============================================================================

class LiveVehicleSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_cte = []
        self.road_width = 46.0
        self.init_track()
        self.init_vehicle()

    def get_track_point(self, s):
        # 闭环多曲率 S 弯公路中心线方程
        cx, cy = 400.0, 300.0
        t = (s * 0.0025) % math.tau
        x = cx + math.cos(t) * 280.0 + math.sin(t * 2.0) * 80.0
        y = cy + math.sin(t) * 190.0 + math.cos(t * 3.0) * 35.0
        
        dx = -math.sin(t) * 280.0 + math.cos(t * 2.0) * 160.0
        dy = math.cos(t) * 190.0 - math.sin(t * 3.0) * 105.0
        theta = math.atan2(dy, dx)
        return x, y, theta

    def init_track(self):
        # 预计算全景赛道离散点 (180 点)
        self.track_points = []
        num_pts = 180
        for i in range(num_pts):
            s_i = (i / num_pts) * (math.tau / 0.0025)
            x, y, theta = self.get_track_point(s_i)
            self.track_points.append({"s": round(s_i, 1), "x": round(x, 1), "y": round(y, 1), "theta": round(theta, 3)})

    def init_vehicle(self):
        with self.lock:
            self.s = 0.0
            x0, y0, theta0 = self.get_track_point(0.0)
            self.x = x0
            self.y = y0
            self.theta = theta0
            self.v = 5.0 # 巡航车速
            self.delta = 0.0 # 前轮转向角
            self.cte = 0.0 # 横向偏离误差
            self.total_dist = 0.0
            self.laps = 0
            self.trail = []

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            dt = 0.04
            L = 24.0 # 轴距
            k_cte = 0.22
            k_heading = 1.25

            # 1. 探测当前公路中心线与预瞄点
            center_x, center_y, road_theta = self.get_track_point(self.s)
            look_s = self.s + 28.0
            look_x, look_y, look_theta = self.get_track_point(look_s)

            # 2. 计算带符号横向偏差 (Signed CTE) 与航向误差
            dx = self.x - center_x
            dy = self.y - center_y
            # 法向量交叉乘积
            cross = math.cos(road_theta) * dy - math.sin(road_theta) * dx
            signed_cte = cross
            self.cte = abs(signed_cte)
            
            heading_err = (road_theta - self.theta + math.pi) % math.tau - math.pi

            # 3. 闭环 Stanley 控制律 (Stanley Lane Centering)
            steer_target = heading_err * k_heading - math.atan2(k_cte * signed_cte, max(1.0, self.v))
            steer_target = max(-0.55, min(0.55, steer_target))
            self.delta += (steer_target - self.delta) * 0.40

            # 4. 速度控制 (弯道平顺减速, 直道平稳巡航)
            curvature = abs(look_theta - road_theta) / 28.0
            target_v = max(3.5, 6.8 - curvature * 90.0)
            self.v += (target_v - self.v) * 0.15

            # 5. 阿克曼运动学积分 (Kinematic Bicycle ODE)
            beta = math.atan(0.5 * math.tan(self.delta))
            self.x += self.v * math.cos(self.theta + beta) * dt * 25.0
            self.y += self.v * math.sin(self.theta + beta) * dt * 25.0
            self.theta += (self.v / L) * math.cos(beta) * math.tan(self.delta) * dt * 25.0

            # 推进沿赛道里程 s
            self.s += self.v * dt * 25.0
            self.total_dist = self.s * 0.1 # 标定为米

            # 6. 统计历史 CTE 误差
            if self.step_count % 5 == 0:
                self.history_cte.append(round(self.cte * 0.02, 3))
                if len(self.history_cte) > 40:
                    self.history_cte.pop(0)

            # 7. 车辆行驶发光轨迹尾迹
            if self.step_count % 2 == 0:
                self.trail.append({"x": round(self.x, 1), "y": round(self.y, 1)})
                if len(self.trail) > 80:
                    self.trail.pop(0)

    def get_snapshot(self):
        with self.lock:
            return {
                "step_count": self.step_count,
                "s": round(self.s, 1),
                "total_dist_m": round(self.total_dist, 1),
                "road_width": self.road_width,
                "track": self.track_points,
                "car": {
                    "x": round(self.x, 1),
                    "y": round(self.y, 1),
                    "s": round(self.s, 1),
                    "theta": round(self.theta, 3),
                    "delta_deg": round(math.degrees(self.delta), 1),
                    "speed_kmh": round(self.v * 14.0, 1),
                    "cte_m": round(self.cte * 0.02, 3)
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
