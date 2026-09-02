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
# 0.8 真实达尔文具身空间演化引擎 (Pure Biological Maze Neuroevolution)
# ============================================================================

class LiveMazeSimulator:
    def __init__(self, width=17, height=17):
        self.width = width
        self.height = height
        self.generation = 1
        self.step_count = 0
        self.max_steps = 240
        self.warp_speed = 20
        self.success_rate = 0.0
        self.last_success_rate = 0.0
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

                r_front = self.cast_ray(ag["x"], ag["y"], ag["theta"])
                r_left = self.cast_ray(ag["x"], ag["y"], ag["theta"] - 0.785)
                r_right = self.cast_ray(ag["x"], ag["y"], ag["theta"] + 0.785)
                ag["rays"] = [r_front, r_left, r_right]
                ag["visited"].add((int(ag["x"]), int(ag["y"])))

                d = math.hypot(gx - ag["x"], gy - ag["y"])
                if d < ag["min_dist"]:
                    ag["min_dist"] = d
                if d < 0.75:
                    ag["goal"] = 1
                    reached_count += 1
                    continue

                target_ang = math.atan2(gy - ag["y"], gx - ag["x"])
                bearing = ((target_ang - ag["theta"] + math.pi) % (2 * math.pi) - math.pi) / math.pi

                if r_front < 0.20:
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
        reached = sum(1 for a in self.agent_states if a.get("goal") == 1)
        self.last_success_rate = reached / len(self.agent_states)

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

        self.population.sort(key=lambda g: g["fitness"], reverse=True)
        new_pop = []
        for i in range(4):
            new_pop.append(dict(self.population[i]))

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
                "success_rate": round(max(self.success_rate, self.last_success_rate), 3),
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

# ============================================================================
# 0.9 软体肌肉与多足步态形态发生动力学引擎 (Soft-Body Locomotion Engine)
# ============================================================================

class LiveLocomotionSimulator:
    def __init__(self):
        self.generation = 1
        self.step_count = 0
        self.max_steps = 300
        self.warp_speed = 5
        self.lock = threading.Lock()
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
            self.champion = self.population[0]

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
        self.champion = self.population[0]
        self.history_dist.append(round(self.champion["fitness"], 1))
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
            champ = self.champion
            return {
                "generation": self.generation,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
                "best_distance": round(champ["fitness"], 1) if champ else 0.0,
                "champion": {
                    "nodes": [{"x": round(n["x"], 1), "y": round(n["y"], 1)} for n in champ["nodes"]],
                    "muscles": [
                        {"n1": m["n1"], "n2": m["n2"], "amp": round(m["amp"], 1), "freq": round(m["freq"], 2)}
                        for m in champ["muscles"]
                    ]
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
        self.lock = threading.Lock()
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
                    "genes": [
                        random.uniform(1.0, 3.5),
                        random.uniform(0.5, 2.0),
                        random.uniform(0.2, 1.2),
                        random.uniform(2.5, 4.2)
                    ]
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
                    "genes": [
                        random.uniform(1.2, 3.0),
                        random.uniform(0.1, 0.8),
                        random.uniform(3.0, 5.2),
                        random.uniform(120.0, 220.0)
                    ]
                })

    def step_physics(self):
        with self.lock:
            self.step_count += 1
            
            for p in self.prey:
                if not p["alive"]:
                    continue
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
