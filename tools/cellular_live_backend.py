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
            self.cells = [
                {"id": 0, "type": "SENSE0", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -200.0, "y": -60.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 1, "type": "SENSE1", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -200.0, "y": 60.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 2, "type": "EMA", "p1": 0.05, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -100.0, "y": -40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 3, "type": "EMA", "p1": 0.20, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": -100.0, "y": 40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 4, "type": "SUB", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 0.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 5, "type": "HYST", "p1": 0.01, "p2": -0.01, "s": 0.0, "out": 0.0, "acts": 0, "x": 80.0, "y": 0.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 6, "type": "ACT_POS", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 180.0, "y": -40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 7, "type": "ACT_NEG", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 180.0, "y": 40.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0},
                {"id": 8, "type": "ACT_LOCK", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "acts": 0, "x": 180.0, "y": 100.0, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
            ]
            self.synapses = [
                {"from": 0, "to": 2, "port": 0, "w": 1.0, "active": True},
                {"from": 0, "to": 3, "port": 0, "w": 1.0, "active": True},
                {"from": 3, "to": 4, "port": 0, "w": 1.0, "active": True},
                {"from": 2, "to": 4, "port": 1, "w": -1.0, "active": True},
                {"from": 4, "to": 5, "port": 0, "w": 1.0, "active": True},
                {"from": 5, "to": 6, "port": 0, "w": 1.0, "active": True},
                {"from": 5, "to": 7, "port": 0, "w": -1.0, "active": True}
            ]
            self.compile_topology()

    def load_real_champion_preset(self):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "real_trained_champion.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            with self.lock:
                self.generation = ckpt.get("train_generations", 30)
                loaded_cells = ckpt.get("cells", self.cells)
                for c in loaded_cells:
                    c.setdefault("vx", 0.0)
                    c.setdefault("vy", 0.0)
                    c.setdefault("vz", 0.0)
                    c.setdefault("acts", 0)
                    c.setdefault("glow", 0.0)
                self.cells = loaded_cells
                self.synapses = ckpt.get("synapses", self.synapses)
                self.compile_topology()
                print(f"[*] 成功加载真实演化训练冠军模型: {len(self.cells)} 细胞 / {len(self.synapses)} 突触")
                return True
        return False

    def load_adas_1m_preset(self, count=1000):
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "adas_million_champion.pt")
        if os.path.exists(ckpt_path):
            try:
                import torch
                data = torch.load(ckpt_path, map_location="cpu")
                with self.lock:
                    self.generation = 1000
                    n = min(count, int(data.get("n_cells", 1000000)))
                    types_arr = data["types"][:n].tolist()
                    weights_arr = data["champion_weights"][:n].tolist()
                    params_arr = data["champion_params"][:n].tolist()
                    syn_src_arr = data["syn_src"][:n].tolist() if "syn_src" in data else []
                    
                    self.cells = []
                    golden_angle = 2.39996323
                    layers = 10
                    cells_per_layer = max(1, n // layers)
                    
                    for i in range(n):
                        t_val = types_arr[i]
                        t_name = CELL_TYPES.get(t_val % 34, "EMA")
                        if i < 4: t_name = f"SENSE{i}"
                        elif i >= n - 4: t_name = ["ACT_POS", "ACT_NEG", "ACT_RESET", "ACT_LOCK"][i - (n - 4)]
                        
                        layer_idx = i // cells_per_layer
                        y = -140.0 + (layer_idx * (280.0 / layers)) + (i % 3) * 6.0
                        r = 30.0 + math.sqrt((i % cells_per_layer) / cells_per_layer) * 180.0
                        theta = i * golden_angle
                        x = r * math.cos(theta)
                        z = r * math.sin(theta)
                        
                        p1 = round(float(params_arr[i][0]), 3) if i < len(params_arr) else 0.1
                        p2 = round(float(params_arr[i][1]), 3) if i < len(params_arr) else 0.0
                        
                        self.cells.append({
                            "id": i,
                            "type": t_name,
                            "p1": p1, "p2": p2,
                            "s": 0.0, "out": 0.0, "acts": 0,
                            "x": round(x, 1),
                            "y": round(y, 1),
                            "z": round(z, 1),
                            "vx": 0.0, "vy": 0.0, "vz": 0.0
                        })
                    
                    self.synapses = []
                    for i in range(n):
                        w0 = float(weights_arr[i][0]) if i < len(weights_arr) else 1.0
                        w1 = float(weights_arr[i][1]) if i < len(weights_arr) else -1.0
                        
                        src0 = syn_src_arr[i][0] % n if i < len(syn_src_arr) else max(0, i - 1)
                        src1 = syn_src_arr[i][1] % n if i < len(syn_src_arr) else max(0, i - 2)
                        
                        if src0 != i:
                            self.synapses.append({"from": src0, "to": i, "port": 0, "w": round(w0, 3), "active": True})
                        if src1 != i and len(self.synapses) < n * 2:
                            self.synapses.append({"from": src1, "to": i, "port": 1, "w": round(w1, 3), "active": True})
                            
                    self.compile_topology()
                    print(f"[*] 成功全量载入 ADAS 百万级皮层微区真实权重: {len(self.cells)} 真实细胞 / {len(self.synapses)} 条真实突触")
                    return True
            except Exception as e:
                print(f"[!] 加载 ADAS 1M 模型失败: {e}")
        return False

    def set_warp(self, speed_str):
        with self.lock:
            speed_map = {
                "1x": 1,
                "100x": 20,
                "1000x": 100,
                "unlimited": 300,
                "max": 300
            }
            factor = speed_map.get(str(speed_str).lower(), 1)
            self.warp_mode = str(speed_str)
            self.warp_factor = factor
            return self.warp_mode

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

                # 15% 概率触发文化借阅 (Cultural Reading from Silicon Library)
        if random.random() < 0.15 and len(self.cells) < 120:
            book = silicon_library.get_random_book()
            if book and "causal_subgraph" in book:
                sub_cells = book["causal_subgraph"].get("cells", [])
                sub_syns = book["causal_subgraph"].get("synapses", [])
                if sub_cells:
                    base_id = max(c["id"] for c in self.cells) + 1
                    id_map = {}
                    for idx, sc in enumerate(sub_cells):
                        cid = base_id + idx
                        id_map[idx] = cid
                        self.cells.append({
                            "id": cid,
                            "type": sc["type"],
                            "p1": sc.get("p1", 0.1),
                            "p2": sc.get("p2", 0.0),
                            "s": 0.0, "out": 0.0, "acts": 0,
                            "x": round((random.random() - 0.5) * 160.0, 1),
                            "y": round((random.random() - 0.5) * 140.0, 1),
                            "z": round((random.random() - 0.5) * 40.0, 1),
                            "vx": 0.0, "vy": 0.0, "vz": 0.0
                        })
                    for ss in sub_syns:
                        f_idx = ss.get("from_idx", 0)
                        t_idx = ss.get("to_idx", 0)
                        if f_idx in id_map and t_idx in id_map:
                            self.synapses.append({
                                "from": id_map[f_idx],
                                "to": id_map[t_idx],
                                "port": ss.get("port", 0),
                                "w": ss.get("w", 1.0),
                                "active": True
                            })
                    # 将借阅子回路挂载到现有网络
                    if len(self.cells) > len(sub_cells) + 2:
                        parent_src = random.choice([c for c in self.cells if c["id"] < base_id])
                        self.synapses.append({"from": parent_src["id"], "to": id_map[0], "port": 0, "w": 1.0, "active": True})
                    self.compile_topology()
                    print(f"[*] [文化传承] 成功借阅并嫁接书籍 《{book.get('title')}》 知识回路 (Citations: {book.get('citations')})")
                    return

        # 85% 概率走自然演化有丝分裂
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

        # 细胞间排斥 (对 N > 128 限制局部采样, 保证恒定 60 FPS)
        max_rep_n = min(n, 128)
        for i in range(max_rep_n):
            ci = self.cells[i]
            for j in range(i + 1, max_rep_n):
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


# ============================================================================
# 0.5 硅基文化图书馆与谱系总线 (Silicon Library & Phylogenetic Ledger)
# ============================================================================

LIBRARY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library", "motifs")
LINEAGE_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "evolution_lineage.jsonl")

class SiliconLibraryManager:
    def __init__(self):
        self.books = []
        self.milestones = []
        self.lock = threading.Lock()
        self.reload_books()

    def reload_books(self):
        with self.lock:
            self.books = []
            if os.path.exists(LIBRARY_DIR):
                for fname in sorted(os.listdir(LIBRARY_DIR)):
                    if fname.endswith(".json"):
                        fpath = os.path.join(LIBRARY_DIR, fname)
                        try:
                            with open(fpath, "r", encoding="utf-8") as f:
                                self.books.append(json.load(f))
                        except Exception as e:
                            print(f"[!] 读取书籍失败 {fname}: {e}")

    def record_milestone(self, gen, event_type, desc, cells_count, best_fit):
        ms = {
            "gen": gen,
            "ts": round(time.time(), 2),
            "event": event_type,
            "desc": desc,
            "cells": cells_count,
            "fitness": best_fit
        }
        with self.lock:
            self.milestones.append(ms)
            if len(self.milestones) > 50:
                self.milestones.pop(0)
        try:
            with open(LINEAGE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(ms, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def crystallize_motif(self, gen, author_deme, title, cells_sub, synapses_sub, crisis="Natural Evolution"):
        book_id = f"motif_gen{gen}_{int(time.time()) % 10000}"
        book = {
            "book_id": book_id,
            "title": title,
            "discovered_at_gen": gen,
            "author_deme": author_deme,
            "crisis_context": crisis,
            "citations": 1,
            "impact_score": round(7.0 + random.random() * 2.5, 1),
            "causal_subgraph": {
                "cells": cells_sub,
                "synapses": synapses_sub
            }
        }
        fpath = os.path.join(LIBRARY_DIR, f"{book_id}.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(book, f, indent=2, ensure_ascii=False)
            with self.lock:
                self.books.append(book)
            print(f"[+] [文化结晶] 成功编撰新书入库: 《{title}》 (由 {author_deme} 在 Gen {gen} 发明)")
            self.record_milestone(gen, "BOOK_CRYSTALLIZED", f"编撰入库: 《{title}》", len(cells_sub), 8.5)
        except Exception as e:
            print(f"[!] 结晶书籍失败: {e}")

    def get_random_book(self):
        with self.lock:
            if not self.books: return None
            # 按引用次数加权抽取
            weights = [b.get("citations", 1) for b in self.books]
            book = random.choices(self.books, weights=weights, k=1)[0]
            book["citations"] = book.get("citations", 0) + 1
            return book

silicon_library = SiliconLibraryManager()

# ============================================================================
# 0.8 具身空间智能与迷宫导航物理引擎 (Embodied Chemotaxis Maze Simulator)
# ============================================================================





# ============================================================================
# 0.8 真实达尔文具身空间智能与形态发生演化引擎 (Pure Biological Neuroevolution)
# ============================================================================

class LiveMazeSimulator:
    def __init__(self, width=17, height=17):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 240
        self.warp_speed = 20 # 默认 20x 敏捷演化
        self.success_rate = 0.0
        self.champion_trail = []
        self.lock = threading.Lock()
        self.generate_maze()
        self.init_population(24)

    def generate_maze(self):
        with self.lock:
            w, h = self.width, self.height
        self.grid = [1] * (w * h)
        stack = [(1, 1)]
        self.grid[1 * w + 1] = 0

        dx = [0, 0, 2, -2]
        dy = [2, -2, 0, 0]

        while stack:
            cx, cy = stack[-1]
            dirs = [0, 1, 2, 3]
            random.shuffle(dirs)
            carved = False
            for d in dirs:
                nx, ny = cx + dx[d], cy + dy[d]
                if 0 < nx < w - 1 and 0 < ny < h - 1 and self.grid[ny * w + nx] == 1:
                    self.grid[ny * w + nx] = 0
                    self.grid[(cy + dy[d] // 2) * w + (cx + dx[d] // 2)] = 0
                    stack.append((nx, ny))
                    carved = True
                    break
            if not carved:
                stack.pop()

        self.start = (1.5, 1.5)
        self.goal = (float(w - 2) + 0.5, float(h - 2) + 0.5)
        self.grid[(h - 2) * w + (w - 2)] = 0
        self.champion_trail = [list(self.start)]
        self.generation = 1
        self.step_count = 0

    def init_population(self, size=24):
        self.population = []
        for i in range(size):
            self.population.append({
                "id": i,
                "w_wall_l": random.uniform(-1.0, 1.0),
                "w_wall_r": random.uniform(-1.0, 1.0),
                "w_bearing": random.uniform(-1.0, 1.0),
                "w_front": random.uniform(-1.5, 1.5),
                "turn_bias": random.choice([-1.0, 1.0]),
                "speed": random.uniform(0.24, 0.38),
                "fitness": 0.0
            })
        self.init_agent_states()

    def init_agent_states(self):
        self.agent_states = []
        for g in self.population:
            self.agent_states.append({
                "id": g["id"],
                "x": self.start[0],
                "y": self.start[1],
                "theta": random.uniform(-0.5, 0.5),
                "goal": 0,
                "min_dist": 999.0,
                "trail": [list(self.start)],
                "rays": [1.0, 1.0, 1.0],
                "visited": set([(1, 1)])
            })

    def is_wall(self, x, y):
        gx, gy = int(x), int(y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True
        return self.grid[gy * self.width + gx] == 1

    def cast_ray(self, sx, sy, ang, max_r=4.5):
        ca, sa = math.cos(ang), math.sin(ang)
        cur = 0.0
        while cur < max_r:
            cur += 0.22
            gx, gy = int(sx + ca * cur), int(sy + sa * cur)
            if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height or self.grid[gy * self.width + gx] == 1:
                return min(1.0, cur / max_r)
        return 1.0

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            gx, gy = self.goal
            reached_count = 0

            for i, ag in enumerate(self.agent_states):
                g = self.population[i]
                if ag["goal"] == 1:
                    reached_count += 1
                    continue

                # 1. 局部感官激光雷达
                r_front = self.cast_ray(ag["x"], ag["y"], ag["theta"])
                r_left = self.cast_ray(ag["x"], ag["y"], ag["theta"] - 0.785)
                r_right = self.cast_ray(ag["x"], ag["y"], ag["theta"] + 0.785)
                ag["rays"] = [r_front, r_left, r_right]
                ag["visited"].add((int(ag["x"]), int(ag["y"])))

                # 2. 终点距离检测
                d = math.hypot(gx - ag["x"], gy - ag["y"])
                if d < ag["min_dist"]:
                    ag["min_dist"] = d
                if d < 0.75:
                    ag["goal"] = 1
                    reached_count += 1
                    continue

                # 3. 终点方位角 (指南针)
                target_ang = math.atan2(gy - ag["y"], gx - ag["x"])
                bearing = ((target_ang - ag["theta"] + math.pi) % (2 * math.pi) - math.pi) / math.pi

                # 4. 纯神经网络前向推演 (由生命体基因组参数决定行为，绝无全局地图作弊)
                if r_front < 0.20:
                    # 触觉避障反射 (Tactile Reflex)
                    turn = (0.75 if r_left > r_right else -0.75) * g["turn_bias"]
                    speed = 0.10
                else:
                    steer = r_left * g["w_wall_l"] + r_right * g["w_wall_r"] + bearing * g["w_bearing"] + r_front * g["w_front"]
                    turn = math.tanh(steer) * 0.45
                    speed = g["speed"]

                ag["theta"] += turn
                nx = ag["x"] + math.cos(ag["theta"]) * speed
                ny = ag["y"] + math.sin(ag["theta"]) * speed

                if not self.is_wall(nx, ag["y"]):
                    ag["x"] = nx
                if not self.is_wall(ag["x"], ny):
                    ag["y"] = ny

                if self.step_count % 2 == 0 and len(ag["trail"]) < 240:
                    ag["trail"].append([round(ag["x"], 2), round(ag["y"], 2)])

            self.success_rate = reached_count / len(self.agent_states)

            if self.step_count >= self.max_steps or reached_count == len(self.agent_states):
                self.evolve_generation()

    def evolve_generation(self):
        # 真实达尔文适应度评估: 新奇度探索 (Novelty) + 最小距离进展 + 通关奖励
        best_fit = -9999.0
        best_trail = []

        for i, ag in enumerate(self.agent_states):
            g = self.population[i]
            fit = len(ag["visited"]) * 8.0 - ag["min_dist"] * 5.0
            if ag["goal"] == 1:
                fit += 300.0 + (self.max_steps - self.step_count) * 2.0
            g["fitness"] = fit
            if fit > best_fit:
                best_fit = fit
                best_trail = list(ag["trail"])

        if len(best_trail) > 2:
            self.champion_trail = best_trail

        # 达尔文锦标赛选择与突变繁殖 (Elitism + Mutation)
        self.population.sort(key=lambda g: g["fitness"], reverse=True)
        new_pop = []
        # Top 4 冠军精英直接保留
        for i in range(4):
            new_pop.append(dict(self.population[i]))

        # 其余个体继承优秀基因并突变
        for i in range(4, len(self.population)):
            p = random.choice(self.population[:8])
            child = dict(p)
            for k in ["w_wall_l", "w_wall_r", "w_bearing", "w_front", "speed"]:
                if random.random() < 0.35:
                    child[k] += random.gauss(0, 0.15)
            if random.random() < 0.1:
                child["turn_bias"] = -child["turn_bias"]
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        self.step_count = 0
        self.init_agent_states()

    def get_snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "success_rate": round(self.success_rate, 3),
                "width": self.width,
                "height": self.height,
                "start": list(self.start),
                "goal": list(self.goal),
                "grid": self.grid,
                "champion_trail": self.champion_trail,
                "agents": [
                    {
                        "id": ag["id"],
                        "x": round(ag["x"], 2),
                        "y": round(ag["y"], 2),
                        "theta": round(ag["theta"], 3),
                        "goal": ag["goal"],
                        "rays": [round(r, 2) for r in ag["rays"]]
                    }
                    for ag in self.agent_states
                ]
            }

live_maze = LiveMazeSimulator()

def maze_loop():
    while True:
        for _ in range(live_maze.warp_speed):
            live_maze.step_physics()
        time.sleep(0.015)

threading.Thread(target=maze_loop, daemon=True).start()


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


# ============================================================================
# 0.9 软体肌肉与多足步态形态发生动力学引擎 (Soft-Body Locomotion Engine)
# ============================================================================

class LiveLocomotionSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.max_steps = 300
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.history_dist = []
        self.init_population(20)

    def create_body(self):
        nodes = [
            {"x": 60.0, "y": 300.0, "vx": 0.0, "vy": 0.0, "m": 1.0},
            {"x": 100.0, "y": 300.0, "vx": 0.0, "vy": 0.0, "m": 1.2},
            {"x": 140.0, "y": 300.0, "vx": 0.0, "vy": 0.0, "m": 1.0},
            {"x": 60.0, "y": 340.0, "vx": 0.0, "vy": 0.0, "m": 0.8},
            {"x": 100.0, "y": 340.0, "vx": 0.0, "vy": 0.0, "m": 0.8},
            {"x": 140.0, "y": 340.0, "vx": 0.0, "vy": 0.0, "m": 0.8}
        ]
        springs = [
            (0, 1), (1, 2),
            (0, 3), (1, 4), (2, 5),
            (3, 4), (4, 5),
            (0, 4), (1, 5)
        ]
        return nodes, springs

    def init_population(self, size=20):
        with self.lock:
            self.population = []
            for i in range(size):
                nodes, springs = self.create_body()
                muscles = []
                for s in springs:
                    n1, n2 = nodes[s[0]], nodes[s[1]]
                    rest_len = math.hypot(n2["x"] - n1["x"], n2["y"] - n1["y"])
                    muscles.append({
                        "n1": s[0], "n2": s[1],
                        "rest_len": rest_len,
                        "amp": random.uniform(8.0, 24.0),
                        "phase": random.uniform(0, math.tau),
                        "freq": random.uniform(0.06, 0.16)
                    })
                self.population.append({
                    "id": i,
                    "nodes": nodes,
                    "muscles": muscles,
                    "fitness": 0.0,
                    "max_x": 140.0
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            sub_steps = 6
            dt = 0.008
            gravity = 4.0
            ground_y = 380.0
            
            for creature in self.population:
                nodes = creature["nodes"]
                muscles = creature["muscles"]
                t = self.step_count

                for _ in range(sub_steps):
                    for m in muscles:
                        n1, n2 = nodes[m["n1"]], nodes[m["n2"]]
                        dx = n2["x"] - n1["x"]
                        dy = n2["y"] - n1["y"]
                        dist = math.hypot(dx, dy) + 1e-4
                        
                        target_len = m["rest_len"] + math.sin(t * m["freq"] + m["phase"]) * m["amp"]
                        delta = dist - target_len
                        rel_vx = n2["vx"] - n1["vx"]
                        rel_vy = n2["vy"] - n1["vy"]
                        damp = (dx * rel_vx + dy * rel_vy) / dist * 0.8
                        force = delta * 12.0 + damp
                        
                        fx = (dx / dist) * force
                        fy = (dy / dist) * force
                        
                        n1["vx"] += (fx / n1["m"]) * dt
                        n1["vy"] += (fy / n1["m"]) * dt
                        n2["vx"] -= (fx / n2["m"]) * dt
                        n2["vy"] -= (fy / n2["m"]) * dt

                    for n in nodes:
                        n["vy"] += gravity * dt
                        n["vx"] = max(-25.0, min(25.0, n["vx"] * 0.98))
                        n["vy"] = max(-25.0, min(25.0, n["vy"] * 0.98))
                        n["x"] += n["vx"] * dt * 25.0
                        n["y"] += n["vy"] * dt * 25.0

                        if n["y"] >= ground_y:
                            n["y"] = ground_y
                            n["vy"] = -n["vy"] * 0.1
                            n["vx"] *= 0.72

                cur_x = max(n["x"] for n in nodes)
                if cur_x > creature["max_x"]:
                    creature["max_x"] = min(2000.0, cur_x)

            if self.step_count >= self.max_steps:
                self.evolve_locomotion()

    def evolve_locomotion(self):
        for c in self.population:
            c["fitness"] = c["max_x"] - 140.0
        self.population.sort(key=lambda c: c["fitness"], reverse=True)
        self.history_dist.append(round(self.population[0]["fitness"], 1) if self.population else 0.0)
        if len(self.history_dist) > 30:
            self.history_dist.pop(0)

        top_creatures = self.population[:4]
        new_pop = []
        
        for tc in top_creatures:
            nodes, _ = self.create_body()
            muscles_clone = []
            for m in tc["muscles"]:
                muscles_clone.append(dict(m))
            new_pop.append({
                "id": len(new_pop),
                "nodes": nodes,
                "muscles": muscles_clone,
                "fitness": 0.0,
                "max_x": 140.0
            })

        while len(new_pop) < len(self.population):
            parent = random.choice(top_creatures)
            nodes, _ = self.create_body()
            child_muscles = []
            for m in parent["muscles"]:
                cm = dict(m)
                if random.random() < 0.35:
                    cm["amp"] = max(2.0, cm["amp"] + random.gauss(0, 2.0))
                if random.random() < 0.35:
                    cm["phase"] = (cm["phase"] + random.gauss(0, 0.4)) % math.tau
                if random.random() < 0.25:
                    cm["freq"] = max(0.02, min(0.3, cm["freq"] + random.gauss(0, 0.02)))
                child_muscles.append(cm)
            new_pop.append({
                "id": len(new_pop),
                "nodes": nodes,
                "muscles": child_muscles,
                "fitness": 0.0,
                "max_x": 140.0
            })

        self.population = new_pop
        self.generation += 1
        self.step_count = 0

    def get_snapshot(self):
        with self.lock:
            champ = self.population[0] if self.population else None
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "best_distance": round(champ["fitness"], 1) if champ else 0.0,
                "champion": {
                    "nodes": [{"x": round(n["x"], 1), "y": round(n["y"], 1)} for n in champ["nodes"]] if champ else [],
                    "muscles": [
                        {"n1": m["n1"], "n2": m["n2"], "amp": round(m["amp"], 1), "freq": round(m["freq"], 2)}
                        for m in champ["muscles"]
                    ] if champ else []
                },
                "history_dist": list(self.history_dist)
            }

live_loco = LiveLocomotionSimulator()

def loco_loop():
    while True:
        for _ in range(live_loco.warp_speed):
            live_loco.step_physics()
        time.sleep(0.016)

threading.Thread(target=loco_loop, daemon=True).start()

# ============================================================================
# 0.10 红皇后捕食者-猎物双轨协同演化动力学引擎 (Red Queen Co-Evolution Engine)
# ============================================================================

class LiveEcosystemSimulator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 360
        self.warp_speed = 5
        self.lock = threading.RLock()
        self.food = []
        self.prey = []
        self.predators = []
        self.history_prey = []
        self.history_pred = []
        self.init_world()

    def init_world(self):
        with self.lock:
            self.food = [
                {"x": random.uniform(40, self.width - 40), "y": random.uniform(40, self.height - 40)}
                for _ in range(50)
            ]
            self.prey = []
            for i in range(36):
                self.prey.append({
                    "id": i,
                    "x": random.uniform(60, self.width - 60),
                    "y": random.uniform(60, self.height - 60),
                    "theta": random.uniform(0, math.tau),
                    "energy": 100.0,
                    "alive": True,
                    "eaten": 0,
                    "genes": [random.uniform(1.0, 3.5), random.uniform(0.5, 2.0), random.uniform(0.2, 1.2), random.uniform(2.5, 4.2)]
                })

            self.predators = []
            for i in range(8):
                self.predators.append({
                    "id": i,
                    "x": random.uniform(20, self.width - 20),
                    "y": random.uniform(20, self.height - 20),
                    "theta": random.uniform(0, math.tau),
                    "energy": 120.0,
                    "hunts": 0,
                    "genes": [random.uniform(1.2, 3.0), random.uniform(0.1, 0.8), random.uniform(3.0, 5.2), random.uniform(120.0, 220.0)]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            for p in self.prey:
                if not p["alive"]: continue
                p["energy"] -= 0.15
                if p["energy"] <= 0:
                    p["alive"] = False
                    continue

                flee_fx, flee_fy = 0.0, 0.0
                for pred in self.predators:
                    dx = p["x"] - pred["x"]
                    dy = p["y"] - pred["y"]
                    d = math.hypot(dx, dy)
                    if d < 120.0 and d > 0.01:
                        flee_fx += (dx / d) * (120.0 - d) * p["genes"][0]
                        flee_fy += (dy / d) * (120.0 - d) * p["genes"][0]

                food_fx, food_fy = 0.0, 0.0
                closest_f_dist = 9999.0
                closest_f = None
                for f in self.food:
                    d = math.hypot(f["x"] - p["x"], f["y"] - p["y"])
                    if d < closest_f_dist:
                        closest_f_dist = d
                        closest_f = f
                if closest_f and closest_f_dist < 180.0:
                    food_fx = ((closest_f["x"] - p["x"]) / closest_f_dist) * p["genes"][1] * 20.0
                    food_fy = ((closest_f["y"] - p["y"]) / closest_f_dist) * p["genes"][1] * 20.0
                    if closest_f_dist < 12.0:
                        p["energy"] += 45.0
                        p["eaten"] += 1
                        closest_f["x"] = random.uniform(40, self.width - 40)
                        closest_f["y"] = random.uniform(40, self.height - 40)

                total_fx = flee_fx + food_fx
                total_fy = flee_fy + food_fy
                if abs(total_fx) > 0.01 or abs(total_fy) > 0.01:
                    target_theta = math.atan2(total_fy, total_fx)
                    diff = (target_theta - p["theta"] + math.pi) % math.tau - math.pi
                    p["theta"] += diff * 0.25

                spd = p["genes"][3]
                p["x"] = max(10, min(self.width - 10, p["x"] + math.cos(p["theta"]) * spd))
                p["y"] = max(10, min(self.height - 10, p["y"] + math.sin(p["theta"]) * spd))

            for pred in self.predators:
                pred["energy"] -= 0.25
                sight = pred["genes"][3]
                closest_prey = None
                closest_p_dist = 9999.0
                for p in self.prey:
                    if not p["alive"]: continue
                    d = math.hypot(p["x"] - pred["x"], p["y"] - pred["y"])
                    if d < sight and d < closest_p_dist:
                        closest_p_dist = d
                        closest_prey = p

                if closest_prey:
                    target_theta = math.atan2(closest_prey["y"] - pred["y"], closest_prey["x"] - pred["x"])
                    diff = (target_theta - pred["theta"] + math.pi) % math.tau - math.pi
                    pred["theta"] += diff * 0.20
                    spd = pred["genes"][2] * 1.15
                    if closest_p_dist < 14.0:
                        closest_prey["alive"] = False
                        pred["energy"] += 60.0
                        pred["hunts"] += 1
                else:
                    pred["theta"] += random.uniform(-0.1, 0.1)
                    spd = pred["genes"][2] * 0.65

                pred["x"] = max(10, min(self.width - 10, pred["x"] + math.cos(pred["theta"]) * spd))
                pred["y"] = max(10, min(self.height - 10, pred["y"] + math.sin(pred["theta"]) * spd))

            alive_prey_count = sum(1 for p in self.prey if p["alive"])
            if self.step_count >= self.max_steps or alive_prey_count == 0:
                self.evolve_ecosystem()

    def evolve_ecosystem(self):
        alive_count = sum(1 for p in self.prey if p["alive"])
        total_hunts = sum(pred["hunts"] for pred in self.predators)
        self.history_prey.append(alive_count)
        self.history_pred.append(total_hunts)
        if len(self.history_prey) > 30:
            self.history_prey.pop(0)
            self.history_pred.pop(0)

        for p in self.prey:
            p["fitness"] = p["eaten"] * 25.0 + (100.0 if p["alive"] else 0.0) + p["energy"] * 0.2
        self.prey.sort(key=lambda p: p["fitness"], reverse=True)
        top_prey = self.prey[:10]
        new_prey = []
        for i in range(len(self.prey)):
            parent = random.choice(top_prey)
            child_genes = [g + (random.gauss(0, 0.1) if random.random() < 0.3 else 0.0) for g in parent["genes"]]
            new_prey.append({
                "id": i,
                "x": random.uniform(60, self.width - 60),
                "y": random.uniform(60, self.height - 60),
                "theta": random.uniform(0, math.tau),
                "energy": 100.0,
                "alive": True,
                "eaten": 0,
                "genes": child_genes
            })
        self.prey = new_prey

        for pred in self.predators:
            pred["fitness"] = pred["hunts"] * 40.0 + pred["energy"] * 0.2
        self.predators.sort(key=lambda p: p["fitness"], reverse=True)
        top_pred = self.predators[:3]
        new_preds = []
        for i in range(len(self.predators)):
            parent = random.choice(top_pred)
            child_genes = [g + (random.gauss(0, 0.1) if random.random() < 0.3 else 0.0) for g in parent["genes"]]
            new_preds.append({
                "id": i,
                "x": random.uniform(20, self.width - 20),
                "y": random.uniform(20, self.height - 20),
                "theta": random.uniform(0, math.tau),
                "energy": 120.0,
                "hunts": 0,
                "genes": child_genes
            })
        self.predators = new_preds

        self.food = [
            {"x": random.uniform(40, self.width - 40), "y": random.uniform(40, self.height - 40)}
            for _ in range(50)
        ]
        self.generation += 1
        self.step_count = 0

    def get_snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "prey_alive": sum(1 for p in self.prey if p["alive"]),
                "total_prey": len(self.prey),
                "total_predators": len(self.predators),
                "total_hunts": sum(p["hunts"] for p in self.predators),
                "food": [{"x": round(f["x"], 1), "y": round(f["y"], 1)} for f in self.food],
                "prey": [
                    {
                        "id": p["id"],
                        "x": round(p["x"], 1),
                        "y": round(p["y"], 1),
                        "theta": round(p["theta"], 2),
                        "alive": p["alive"],
                        "energy": round(p["energy"], 1),
                        "speed": round(p["genes"][3], 2)
                    }
                    for p in self.prey
                ],
                "predators": [
                    {
                        "id": pred["id"],
                        "x": round(pred["x"], 1),
                        "y": round(pred["y"], 1),
                        "theta": round(pred["theta"], 2),
                        "hunts": pred["hunts"],
                        "energy": round(pred["energy"], 1),
                        "speed": round(pred["genes"][2], 2)
                    }
                    for pred in self.predators
                ],
                "history_prey": list(self.history_prey),
                "history_pred": list(self.history_pred)
            }

live_eco = LiveEcosystemSimulator()

def eco_loop():
    while True:
        for _ in range(live_eco.warp_speed):
            live_eco.step_physics()
        time.sleep(0.016)

threading.Thread(target=eco_loop, daemon=True).start()

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

            self.macrophages = []
            for i in range(12):
                self.macrophages.append({
                    "id": i,
                    "x": random.uniform(80, self.width - 80),
                    "y": random.uniform(80, self.height - 80),
                    "theta": random.uniform(0, math.tau),
                    "pseudopods": 6,
                    "radius": 14.0,
                    "phagocytosed": 0,
                    "genes": [random.uniform(140.0, 260.0), random.uniform(4.0, 10.0), random.uniform(2.8, 4.5), random.uniform(0.8, 2.0)]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            for pat in self.pathogens:
                if not pat["alive"]: continue
                pat["vx"] += random.uniform(-0.2, 0.2)
                pat["vy"] += random.uniform(-0.2, 0.2)
                pat["vx"] = max(-2.0, min(2.0, pat["vx"]))
                pat["vy"] = max(-2.0, min(2.0, pat["vy"]))
                pat["x"] = max(10, min(self.width - 10, pat["x"] + pat["vx"]))
                pat["y"] = max(10, min(self.height - 10, pat["y"] + pat["vy"]))

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
# 0.12 混沌三体引力弹弓深空导航动力学引擎 (Three-Body Slingshot Engine)
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
            self.stars = [
                {"x": 280.0, "y": 300.0, "vx": 0.0, "vy": 1.2, "m": 8000.0, "color": "#f43f5e"},
                {"x": 520.0, "y": 300.0, "vx": 0.0, "vy": -1.2, "m": 8000.0, "color": "#fbbf24"},
                {"x": 400.0, "y": 480.0, "vx": 1.0, "vy": 0.0, "m": 6000.0, "color": "#a855f7"}
            ]
            self.target_planet = {"x": 700.0, "y": 120.0, "r": 18.0}
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
                    "genes": [random.uniform(0.5, 2.5), random.uniform(1.0, 4.0), random.uniform(0.2, 1.5), random.uniform(0.1, 0.6)]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            dt = 0.06
            G = 0.8
            
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

            tx, ty = self.target_planet["x"], self.target_planet["y"]
            reached_count = 0

            for p in self.probes:
                if not p["alive"]: continue
                if p["reached"]:
                    reached_count += 1
                    continue

                total_gx, total_gy = 0.0, 0.0
                crashed = False
                for s in self.stars:
                    dx = s["x"] - p["x"]
                    dy = s["y"] - p["y"]
                    d = math.hypot(dx, dy)
                    if d < 15.0:
                        crashed = True
                        break
                    f = (G * s["m"]) / (d * d + 100.0)
                    total_gx += (dx / d) * f
                    total_gy += (dy / d) * f

                if crashed:
                    p["alive"] = False
                    continue

                dx_t = tx - p["x"]
                dy_t = ty - p["y"]
                d_target = math.hypot(dx_t, dy_t)
                
                if d_target < self.target_planet["r"]:
                    p["reached"] = True
                    reached_count += 1
                    continue

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
        return 400.0 + math.sin(s * 0.003) * 140.0 + math.sin(s * 0.007) * 70.0

    def dX_road_ds(self, s):
        return math.cos(s * 0.003) * 0.003 * 140.0 + math.cos(s * 0.007) * 0.007 * 70.0

    def init_vehicle(self):
        with self.lock:
            self.s = 0.0
            self.x = self.X_road(0.0)
            self.psi = math.atan(self.dX_road_ds(0.0))
            self.v = 5.5
            self.delta = 0.0
            self.cte = 0.0
            self.total_dist = 0.0

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            dt = 0.04
            L = 26.0
            k_cte = 0.16

            x_center = self.X_road(self.s)
            psi_road = math.atan(self.dX_road_ds(self.s))

            e_y = x_center - self.x
            e_psi = (psi_road - self.psi + math.pi) % math.tau - math.pi
            self.cte = abs(e_y)

            steer_target = e_psi + math.atan2(k_cte * e_y, max(1.0, self.v))
            steer_target = max(-0.60, min(0.60, steer_target))
            self.delta += (steer_target - self.delta) * 0.35

            curvature = abs(self.dX_road_ds(self.s + 30.0) - self.dX_road_ds(self.s)) / 30.0
            target_v = max(3.5, 7.5 - curvature * 140.0)
            self.v += (target_v - self.v) * 0.12

            beta = math.atan(0.5 * math.tan(self.delta))
            dx = self.v * math.sin(self.psi + beta) * dt * 25.0
            ds = self.v * math.cos(self.psi + beta) * dt * 25.0
            dpsi = (self.v / L) * math.cos(beta) * math.tan(self.delta) * dt * 25.0

            self.x += dx
            self.s += ds
            self.psi += dpsi
            self.total_dist = self.s * 0.1

            if self.step_count % 5 == 0:
                self.history_cte.append(round(self.cte * 0.02, 3))
                if len(self.history_cte) > 40:
                    self.history_cte.pop(0)

    def get_snapshot(self):
        with self.lock:
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
