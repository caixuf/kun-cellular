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


class LiveMazeSimulator:
    def __init__(self, width=21, height=21):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 320
        self.warp_speed = 20 # 默认 20x 极速
        self.success_rate = 0.0
        self.champion_trail = []
        self.dist_field = []
        self.lock = threading.Lock()
        self.generate_maze()
        self.init_agents(24)

    def generate_maze(self):
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
        self.compute_distance_field()

    def compute_distance_field(self):
        # 预计算拓扑势能场 (BFS 到终点的真实网格最短步数)
        w, h = self.width, self.height
        gx, gy = int(self.goal[0]), int(self.goal[1])
        self.dist_field = [999] * (w * h)
        self.dist_field[gy * w + gx] = 0
        q = [(gx, gy)]
        dx = [0, 0, 1, -1]
        dy = [1, -1, 0, 0]
        while q:
            cx, cy = q.pop(0)
            cur_d = self.dist_field[cy * w + cx]
            for i in range(4):
                nx, ny = cx + dx[i], cy + dy[i]
                if 0 <= nx < w and 0 <= ny < h and self.grid[ny * w + nx] == 0:
                    if self.dist_field[ny * w + nx] > cur_d + 1:
                        self.dist_field[ny * w + nx] = cur_d + 1
                        q.append((nx, ny))

    def init_agents(self, num=24):
        self.agents = []
        for i in range(num):
            self.agents.append({
                "id": i,
                "x": self.start[0],
                "y": self.start[1],
                "theta": random.uniform(-0.5, 0.5),
                "goal": 0,
                "min_dist": 999.0,
                "trail": [list(self.start)],
                "rays": [1.0, 1.0, 1.0],
                "bias": 1.0 if i % 2 == 0 else -1.0
            })

    def is_wall(self, x, y):
        gx, gy = int(x), int(y)
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return True
        return self.grid[gy * self.width + gx] == 1

    def cast_ray(self, sx, sy, ang, max_r=5.0):
        step = 0.1
        cur_r = 0.0
        while cur_r < max_r:
            cur_r += step
            rx = sx + math.cos(ang) * cur_r
            ry = sy + math.sin(ang) * cur_r
            if self.is_wall(rx, ry):
                return min(1.0, cur_r / max_r)
        return 1.0

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            gx, gy = self.goal
            reached_count = 0
            w, h = self.width, self.height

            for ag in self.agents:
                if ag["goal"] == 1:
                    reached_count += 1
                    continue

                # 1. 3路激光测距
                r_front = self.cast_ray(ag["x"], ag["y"], ag["theta"])
                r_left = self.cast_ray(ag["x"], ag["y"], ag["theta"] - 0.785398)
                r_right = self.cast_ray(ag["x"], ag["y"], ag["theta"] + 0.785398)
                ag["rays"] = [r_front, r_left, r_right]

                # 2. 终点距离检测
                dx = gx - ag["x"]
                dy = gy - ag["y"]
                dist = math.hypot(dx, dy)
                if dist < ag["min_dist"]:
                    ag["min_dist"] = dist
                if dist < 0.8:
                    ag["goal"] = 1
                    reached_count += 1
                    continue

                # 3. 经典死胡同 U 型掉头反射 (Cul-de-sac 180° U-Turn Reflection)
                # 当正前方撞墙且两侧均受阻时, 触发迟滞掉头锁
                if r_front < 0.18 and r_left < 0.28 and r_right < 0.28:
                    ag["theta"] += math.pi + random.uniform(-0.2, 0.2)
                    ag["u_turn_lock"] = 6 # 保持 6 步防回头保护
                
                if ag.get("u_turn_lock", 0) > 0:
                    ag["u_turn_lock"] -= 1
                    speed = 0.30
                    turn = 0.0
                else:
                    # 4. 拓扑势能场全局全局极小值搜索 (严格沿 BFS 梯度下降, 绝不贪心直扑欧式直线)
                    cgx, cgy = max(0, min(w-1, int(ag["x"]))), max(0, min(h-1, int(ag["y"])))
                    best_ang = ag["theta"]
                    min_td = 9999

                    # 探测周边可行走方向中的绝对最小拓扑步数
                    for test_ang in [0, 0.523, 1.047, 1.57, 2.094, 2.618, 3.141, -2.618, -2.094, -1.57, -1.047, -0.523]:
                        tx = ag["x"] + math.cos(test_ang) * 0.85
                        ty = ag["y"] + math.sin(test_ang) * 0.85
                        if not self.is_wall(tx, ty):
                            tgx, tgy = int(tx), int(ty)
                            td = self.dist_field[tgy * w + tgx] if (0 <= tgx < w and 0 <= tgy < h) else 9999
                            if td < min_td:
                                min_td = td
                                best_ang = test_ang

                    diff_ang = (best_ang - ag["theta"] + math.pi) % (2 * math.pi) - math.pi

                    if r_front < 0.20:
                        turn = 0.85 if r_left > r_right else -0.85
                        speed = 0.12
                    else:
                        turn = diff_ang * 0.55 + (0.22 if r_left < 0.22 else 0.0) - (0.22 if r_right < 0.22 else 0.0)
                        speed = 0.38

                ag["theta"] += turn
                nx = ag["x"] + math.cos(ag["theta"]) * speed
                ny = ag["y"] + math.sin(ag["theta"]) * speed

                if not self.is_wall(nx, ag["y"]):
                    ag["x"] = nx
                if not self.is_wall(ag["x"], ny):
                    ag["y"] = ny

                if self.step_count % 2 == 0 and len(ag["trail"]) < 240:
                    ag["trail"].append([round(ag["x"], 2), round(ag["y"], 2)])

            self.success_rate = reached_count / len(self.agents)

            if self.step_count >= self.max_steps or reached_count == len(self.agents):
                self.evolve_generation()

    def evolve_generation(self):
        for ag in self.agents:
            base_fit = 30.0 - ag["min_dist"]
            if ag["goal"] == 1:
                base_fit += 150.0
            ag["fit"] = base_fit

        self.agents.sort(key=lambda a: a["fit"], reverse=True)
        best = self.agents[0]
        if len(best["trail"]) > 2:
            self.champion_trail = list(best["trail"])

        self.init_agents(24)
        self.generation += 1
        self.step_count = 0

    def get_snapshot(self):
        with self.lock:
            return {
                "generation": self.generation,
                "step_count": self.step_count,
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
                    for ag in self.agents
                ]
            }

live_maze = LiveMazeSimulator()

def maze_loop():
    while True:
        for _ in range(live_maze.warp_speed):
            live_maze.step_physics()
        time.sleep(0.008) # 120Hz 高频极速动力学

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
            live_maze.init_agents(24)
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
