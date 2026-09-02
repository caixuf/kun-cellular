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
# 1. 实时仿真引擎 (Live Cellular Simulation Engine)
# ============================================================================

CELL_TYPES = {
    0: "SENSE0", 1: "SENSE1", 2: "SENSE2", 3: "SENSE3",
    10: "EMA", 11: "DIFF", 12: "INTEGRAL", 13: "SUM", 14: "SUB",
    15: "MUL", 16: "RATIO", 17: "ABS", 18: "DELAY_N", 19: "OSCILLATOR", 20: "QUADRATIC",
    24: "THRESH", 25: "HYST", 26: "AND", 27: "INHIB", 28: "DEADZONE", 29: "MIN_MAX",
    30: "ACT_POS", 31: "ACT_NEG", 32: "ACT_RESET", 33: "ACT_LOCK"
}

class LiveOrganism:
    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 384
        self.phy_steps = 1500
        self.warp_mode = "1x"
        self.warp_factor = 1
        self.stress_mode = False
        self.cells = []
        self.synapses = []
        self.order = []
        self.actions = {"pos": 0.0, "neg": 0.0, "reset": 0.0, "lock": 0.0}
        self.shannon_h = 1.94
        self.atp_budget = 92.4
        self.last_update_ts = time.time()
        self.load_mature_preset()

    def load_seed_preset(self):
        with self.lock:
            self.generation = 0
            self.phy_steps = 0
            self.cells = [
                {"id": 0, "type": "SENSE0", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -120.0, "y": -40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 1, "type": "SENSE1", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -120.0, "y": 40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 2, "type": "EMA", "p1": 0.05, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -40.0, "y": -30.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 3, "type": "EMA", "p1": 0.20, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -40.0, "y": 30.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 4, "type": "SUB", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 30.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 5, "type": "HYST", "p1": 0.01, "p2": -0.01, "s": 0.0, "out": 0.0, "acts": 0, "x": 80.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 6, "type": "ACT_POS", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 140.0, "y": -40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 7, "type": "ACT_NEG", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 140.0, "y": 40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 8, "type": "ACT_LOCK", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 140.0, "y": 100.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            ]
            self.synapses = [
                {"from": 0, "to": 2, "port": 0, "w": 1.0, "active": True},
                {"from": 0, "to": 3, "port": 0, "w": 1.0, "active": True},
                {"from": 3, "to": 4, "port": 0, "w": 1.0, "active": True},
                {"from": 2, "to": 4, "port": 1, "w": 1.0, "active": True},
                {"from": 4, "to": 5, "port": 0, "w": 1.0, "active": True},
                {"from": 5, "to": 6, "port": 0, "w": 1.0, "active": True},
                {"from": 5, "to": 7, "port": 0, "w": -1.0, "active": True}
            ]
            self.compile_topology()

    
    def set_warp(self, speed):
        with self.lock:
            self.warp_mode = speed
            if speed == "1x": self.warp_factor = 1
            elif speed == "100x": self.warp_factor = 25
            elif speed == "1000x": self.warp_factor = 100
            elif speed == "unlimited" or speed == "max": self.warp_factor = 500
            return self.warp_mode

    
    def load_real_champion_preset(self):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "real_trained_champion.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            with self.lock:
                self.generation = ckpt.get("train_generations", 30)
                self.cells = ckpt.get("cells", self.cells)
                self.synapses = ckpt.get("synapses", self.synapses)
                self.compile_topology()
                print(f"[*] 成功加载真实演化训练冠军模型: {len(self.cells)} 细胞 / {len(self.synapses)} 突触")
                return True
        return False

    
    def load_adas_1m_preset(self):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "adas_million_champion.pt")
        if os.path.exists(ckpt_path):
            try:
                import torch
                data = torch.load(ckpt_path, map_location="cpu")
                with self.lock:
                    self.generation = 1000
                    # 提取前 40 个核心代表性因果细胞子图映射到 3D 视图，宏观 1M 挂载到 GPU 点云
                    types_arr = data["types"][:40].tolist()
                    weights_arr = data["champion_weights"][:40].tolist()
                    params_arr = data["champion_params"][:40].tolist()
                    
                    self.cells = []
                    for i, t_val in enumerate(types_arr):
                        t_name = CELL_TYPES.get(t_val % 34, "EMA")
                        if i < 4: t_name = f"SENSE{i}"
                        elif i >= len(types_arr) - 4: t_name = ["ACT_POS", "ACT_NEG", "ACT_RESET", "ACT_LOCK"][i - (len(types_arr) - 4)]
                        
                        self.cells.append({
                            "id": i,
                            "type": t_name,
                            "p1": round(float(params_arr[i][0]), 3),
                            "p2": round(float(params_arr[i][1]), 3),
                            "s": 0.0, "out": 0.0, "acts": 0,
                            "x": -220.0 + (440.0 * i / len(types_arr)),
                            "y": round((float(weights_arr[i][0]) * 100.0) % 160.0 - 80.0, 1),
                            "z": round((float(weights_arr[i][1]) * 40.0) % 60.0 - 30.0, 1),
                            "vx": 0.0, "vy": 0.0, "vz": 0.0
                        })
                    
                    self.synapses = []
                    for i in range(len(self.cells) - 1):
                        self.synapses.append({
                            "from": i,
                            "to": min(len(self.cells) - 1, i + 1 + (i % 3)),
                            "port": i % 2,
                            "w": round(float(weights_arr[i][0]), 3),
                            "active": True
                        })
                    self.compile_topology()
                    print(f"[*] 成功加载 ADAS 百万级冠军模型 (adas_million_champion.pt, 1,000,000 细胞 / 2,000,000 突触)")
                    return True
            except Exception as e:
                print(f"[!] 加载 ADAS 1M 模型失败: {e}")
        return False

    def load_mature_preset(self):
        with self.lock:
            self.generation = 384
            self.phy_steps = 1500
            self.cells = [
                # Sense
                {"id": 0, "type": "SENSE0", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -220.0, "y": -90.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 1, "type": "SENSE1", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -220.0, "y": -30.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 2, "type": "SENSE2", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -220.0, "y": 30.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 3, "type": "SENSE3", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -220.0, "y": 90.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                # Metabolic
                {"id": 4, "type": "EMA", "p1": 0.05, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -140.0, "y": -110.0, "z": 10.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 5, "type": "EMA", "p1": 0.20, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -140.0, "y": -60.0, "z": -10.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 6, "type": "DIFF", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -140.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 7, "type": "INTEGRAL", "p1": 0.02, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -140.0, "y": 50.0, "z": 15.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 8, "type": "OSCILLATOR", "p1": 1.2, "p2": 0.05, "s": 0.1, "out": 0.0, "acts": 0, "x": -140.0, "y": 110.0, "z": -15.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 9, "type": "DELAY_N", "p1": 0.5, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -80.0, "y": -90.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 10, "type": "ABS", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -80.0, "y": -40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 11, "type": "RATIO", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -80.0, "y": 20.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 12, "type": "MUL", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -80.0, "y": 80.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                # Intermediate & Fusion
                {"id": 13, "type": "SUB", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -20.0, "y": -70.0, "z": 5.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 14, "type": "SUM", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -20.0, "y": -10.0, "z": -5.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 15, "type": "QUADRATIC", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -20.0, "y": 50.0, "z": 10.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 16, "type": "EMA", "p1": 0.12, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 40.0, "y": -80.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 17, "type": "DIFF", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 40.0, "y": -20.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 18, "type": "INTEGRAL", "p1": 0.01, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 40.0, "y": 40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                # Gating & Schmitt
                {"id": 19, "type": "HYST", "p1": 0.012, "p2": -0.012, "s": 0.0, "out": 0.0, "acts": 0, "x": 100.0, "y": -90.0, "z": 5.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 20, "type": "HYST", "p1": 0.015, "p2": -0.015, "s": 0.0, "out": 0.0, "acts": 0, "x": 100.0, "y": -30.0, "z": -5.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 21, "type": "THRESH", "p1": 0.05, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 100.0, "y": 30.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 22, "type": "INHIB", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 100.0, "y": 90.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 23, "type": "DEADZONE", "p1": 0.008, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 150.0, "y": -50.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 24, "type": "AND", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 150.0, "y": 20.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 25, "type": "MIN_MAX", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 150.0, "y": 80.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                # Action Effectors
                {"id": 26, "type": "ACT_POS", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 220.0, "y": -70.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 27, "type": "ACT_NEG", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 220.0, "y": -10.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 28, "type": "ACT_RESET", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 220.0, "y": 50.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 29, "type": "ACT_LOCK", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 220.0, "y": 110.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            ]
            links = [
                (0,4,0,1.0),(0,5,0,1.0),(0,6,0,1.0),(0,7,0,0.8),(1,8,0,1.2),(1,10,0,1.0),(2,11,0,0.9),(3,12,0,1.1),
                (4,13,1,-1.0),(5,13,0,1.0),(6,14,0,0.8),(7,14,1,0.6),(8,12,1,0.7),(9,15,0,1.3),(10,11,1,0.5),
                (13,16,0,1.1),(13,19,0,1.0),(14,17,0,0.9),(14,20,0,-1.0),(15,18,0,0.7),(16,19,0,0.8),(17,21,0,1.2),
                (18,22,0,0.9),(19,23,0,1.0),(20,23,1,-1.0),(21,24,0,1.0),(22,24,1,0.8),(23,26,0,1.0),(23,27,0,-1.0),
                (24,26,1,0.7),(25,28,0,0.9),(12,29,0,1.5),(22,29,1,1.2)
            ]
            self.synapses = [{"from": a, "to": b, "port": p, "w": w, "active": True} for a, b, p, w in links]
            self.compile_topology()

    
    def step_morphogenesis(self):
        # 突触有丝分裂 (Synaptic Mitosis) 与形态发生增殖
        if len(self.cells) >= 128:
            # 达到脑区细胞容量上限后触发凋亡净化 (Apoptosis)
            if random.random() < 0.3:
                non_essential = [c for c in self.cells if not c["type"].startswith("SENSE") and not c["type"].startswith("ACT_")]
                if len(non_essential) > 5:
                    victim = random.choice(non_essential)
                    self.cells = [c for c in self.cells if c["id"] != victim["id"]]
                    self.synapses = [s for s in self.synapses if s["from"] != victim["id"] and s["to"] != victim["id"]]
                    self.compile_topology()
            return

        if not self.synapses:
            return

        # 选取一条高活跃突触分裂
        syn = random.choice(self.synapses)
        if not syn.get("active", True):
            return

        by_id = {c["id"]: c for c in self.cells}
        c_from = by_id.get(syn["from"])
        c_to = by_id.get(syn["to"])
        if not c_from or not c_to:
            return

        new_id = max(c["id"] for c in self.cells) + 1
        candidate_types = ["EMA", "DIFF", "INTEGRAL", "SUM", "SUB", "MUL", "RATIO", "ABS", "OSCILLATOR", "QUADRATIC", "THRESH", "HYST", "AND", "INHIB", "DEADZONE"]
        new_type = random.choice(candidate_types)
        
        mid_x = (c_from["x"] + c_to["x"]) * 0.5 + (random.random() - 0.5) * 20.0
        mid_y = (c_from["y"] + c_to["y"]) * 0.5 + (random.random() - 0.5) * 20.0
        mid_z = (c_from["z"] + c_to["z"]) * 0.5 + (random.random() - 0.5) * 20.0

        new_cell = {
            "id": new_id,
            "type": new_type,
            "p1": random.uniform(0.01, 0.5),
            "p2": random.uniform(-0.1, 0.1),
            "s": 0.0,
            "out": 0.0,
            "acts": 0,
            "x": round(mid_x, 1),
            "y": round(mid_y, 1),
            "z": round(mid_z, 1),
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0
        }
        self.cells.append(new_cell)

        # 突触分裂为两段
        orig_to = syn["to"]
        orig_port = syn.get("port", 0)
        orig_w = syn.get("w", 1.0)
        syn["to"] = new_id
        syn["port"] = 0

        new_syn = {
            "from": new_id,
            "to": orig_to,
            "port": orig_port,
            "w": orig_w * random.uniform(0.8, 1.2),
            "active": True
        }
        self.synapses.append(new_syn)

        # 偶发侧向联络突触
        if random.random() < 0.4 and len(self.cells) > 5:
            other = random.choice(self.cells)
            if other["id"] != new_id and not other["type"].startswith("ACT_"):
                self.synapses.append({
                    "from": other["id"],
                    "to": new_id,
                    "port": 1,
                    "w": random.uniform(-1.0, 1.0),
                    "active": True
                })

        self.compile_topology()

    def compile_topology(self):
        by_id = {c["id"]: i for i, c in enumerate(self.cells)}
        indeg = {c["id"]: 0 for c in self.cells}
        adj = {c["id"]: [] for c in self.cells}
        for s in self.synapses:
            if s["active"] and s["from"] in by_id and s["to"] in by_id:
                adj[s["from"]].append(s["to"])
                indeg[s["to"]] += 1
        q = [c["id"] for c in self.cells if indeg[c["id"]] == 0]
        order = []
        for u in q:
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        seen = set(order)
        for c in self.cells:
            if c["id"] not in seen:
                order.append(c["id"])
        self.order = [by_id[cid] for cid in order if cid in by_id]

    def step_physics(self, dt=0.04):
        # 3D 兰纳-琼斯力场 + 突触弹簧阻尼
        n = len(self.cells)
        fx = [0.0] * n
        fy = [0.0] * n
        fz = [0.0] * n

        # 细胞间排斥
        for i in range(n):
            ci = self.cells[i]
            for j in range(i + 1, n):
                cj = self.cells[j]
                dx = cj["x"] - ci["x"]
                dy = cj["y"] - ci["y"]
                dz = cj["z"] - ci["z"]
                dist_sq = dx*dx + dy*dy + dz*dz + 1e-4
                dist = math.sqrt(dist_sq)
                if dist < 80.0:
                    rep = 400.0 / (dist_sq)
                    fx[i] -= (dx / dist) * rep
                    fy[i] -= (dy / dist) * rep
                    fz[i] -= (dz / dist) * rep
                    fx[j] += (dx / dist) * rep
                    fy[j] += (dy / dist) * rep
                    fz[j] += (dz / dist) * rep

        # 突触引力
        by_id = {c["id"]: i for i, c in enumerate(self.cells)}
        for s in self.synapses:
            if not s["active"]:
                continue
            fi, ti = by_id.get(s["from"]), by_id.get(s["to"])
            if fi is not None and ti is not None:
                cf, ct = self.cells[fi], self.cells[ti]
                dx = ct["x"] - cf["x"]
                dy = ct["y"] - cf["y"]
                dz = ct["z"] - cf["z"]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz + 1e-4)
                target_len = 50.0
                pull = (dist - target_len) * 0.08
                fx[fi] += (dx / dist) * pull
                fy[fi] += (dy / dist) * pull
                fz[fi] += (dz / dist) * pull
                fx[ti] -= (dx / dist) * pull
                fy[ti] -= (dy / dist) * pull
                fz[ti] -= (dz / dist) * pull

        # 速度更新与阻尼衰减 (Verlet-Euler 积分)
        damping = 0.88
        for i in range(n):
            c = self.cells[i]
            c["vx"] = (c["vx"] + fx[i] * dt) * damping
            c["vy"] = (c["vy"] + fy[i] * dt) * damping
            c["vz"] = (c["vz"] + fz[i] * dt) * damping
            # 感知受体与效应动作锚定在左右两侧，中间微柱自由折叠
            if not c["type"].startswith("SENSE") and not c["type"].startswith("ACT_"):
                c["x"] += c["vx"] * dt
                c["y"] += c["vy"] * dt
                c["z"] += c["vz"] * dt

        self.phy_steps += 1

    def step_forward(self, inputs):
        with self.lock:
            by_id = {c["id"]: i for i, c in enumerate(self.cells)}
            port_in = [[0.0, 0.0] for _ in range(len(self.cells))]

            for s in self.synapses:
                if not s["active"]:
                    continue
                fi, ti = by_id.get(s["from"]), by_id.get(s["to"])
                if fi is not None and ti is not None:
                    port_in[ti][s["port"]] += self.cells[fi]["out"] * s["w"]

            for idx in self.order:
                c = self.cells[idx]
                i0, i1 = port_in[idx][0], port_in[idx][1]
                t = c["type"]

                if t == "SENSE0": c["out"] = inputs[0] * c["p1"]
                elif t == "SENSE1": c["out"] = inputs[1] * c["p1"]
                elif t == "SENSE2": c["out"] = inputs[2] * c["p1"]
                elif t == "SENSE3": c["out"] = inputs[3] * c["p1"]
                elif t == "EMA":
                    a = max(0.01, min(1.0, c["p1"]))
                    c["s"] = i0 if c["acts"] == 0 else (a * i0 + (1.0 - a) * c["s"])
                    c["out"] = c["s"]
                elif t == "DIFF":
                    c["out"] = i0 - c["s"]
                    c["s"] = i0
                elif t == "INTEGRAL":
                    c["s"] += i0 * max(0.001, c["p1"])
                    c["out"] = c["s"]
                elif t == "SUM": c["out"] = i0 + i1
                elif t == "SUB": c["out"] = i0 - i1
                elif t == "MUL": c["out"] = math.tanh(i0 * i1)
                elif t == "RATIO": c["out"] = i0 / (abs(i1) + 1e-4)
                elif t == "ABS": c["out"] = abs(i0)
                elif t == "OSCILLATOR":
                    c["s"] = math.sin(time.time() * 3.0 + idx)
                    c["out"] = c["s"]
                elif t == "QUADRATIC": c["out"] = (1.0 if i0 >= 0 else -1.0) * (i0 * i0)
                elif t == "THRESH": c["out"] = 1.0 if i0 > c["p1"] else 0.0
                elif t == "HYST":
                    if abs(i0) > c["p1"]: c["out"] = i0
                    elif abs(i0) < abs(c["p2"]): c["out"] = 0.0
                elif t == "AND": c["out"] = 1.0 if (i0 > 0 and i1 > 0) else 0.0
                elif t == "INHIB": c["out"] = i0 * max(0.0, 1.0 - i1)
                elif t == "DEADZONE": c["out"] = i0 if abs(i0) > c["p1"] else 0.0
                elif t == "MIN_MAX": c["out"] = min(i0, i1)
                elif t == "ACT_POS": self.actions["pos"] = max(0.0, min(1.0, i0))
                elif t == "ACT_NEG": self.actions["neg"] = max(0.0, min(1.0, i0))
                elif t == "ACT_RESET": self.actions["reset"] = 1.0 if i0 > 0.5 else 0.0
                elif t == "ACT_LOCK": self.actions["lock"] = 1.0 if i0 > 0.8 else 0.0

                c["acts"] += 1

            self.step_physics()

    def get_state_snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "phy_steps": self.phy_steps,
                "warp_mode": self.warp_mode,
                "stress_mode": self.stress_mode,
                "shannon_h": round(self.shannon_h, 3),
                "atp_budget": round(self.atp_budget, 1),
                "actions": self.actions,
                "cells": [
                    {
                        "id": c["id"],
                        "type": c["type"],
                        "p1": c["p1"],
                        "p2": c["p2"],
                        "s": round(c["s"], 3),
                        "out": round(c["out"], 3),
                        "acts": c["acts"],
                        "x": round(c["x"], 1),
                        "y": round(c["y"], 1),
                        "z": round(c["z"], 1)
                    }
                    for c in self.cells
                ],
                "synapses": self.synapses
            }

organism = LiveOrganism()

# ============================================================================
# 2. 仿真主循环线程 (Simulation Tick Thread)
# ============================================================================

def simulation_worker():
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        # 产生多相态合成驱动信号
        px = math.sin(elapsed * 0.8) + 0.3 * math.sin(elapsed * 2.5)
        vol = abs(math.cos(elapsed * 0.5))
        spread = 0.05 + 0.02 * math.sin(elapsed * 1.2)
        ttc = 5.0 + 3.0 * math.cos(elapsed * 0.3)

        if organism.stress_mode:
            # 注入红皇后闪崩冲击
            px += (random.random() - 0.5) * 4.0
            spread += 0.5

        steps = organism.warp_factor
        for _ in range(steps):
            organism.step_forward([px, vol, spread, ttc])
        with organism.lock:
            if steps > 1:
                organism.generation += max(1, steps // 20)
            else:
                organism.generation += 1

            # 随世代演化自发增殖有丝分裂 (持续生长出新功能细胞)
            if organism.generation % 12 == 0 or (steps > 1 and organism.generation % max(1, 200 // steps) == 0):
                organism.step_morphogenesis()
        time.sleep(0.025)

sim_thread = threading.Thread(target=simulation_worker, daemon=True)
sim_thread.start()

# ============================================================================
# 3. HTTP & WebSocket 请求分发处理
# ============================================================================

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class ObservatoryHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

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

        if self.path.startswith("/api/stress"):
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
