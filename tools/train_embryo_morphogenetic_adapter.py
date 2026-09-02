#!/usr/bin/env python3
"""
Embryo Morphogenesis Adapter Evolution Trainer & Benchmark
===========================================================
基于受精卵胚胎发育、卵裂扩增、原肠力场极性迁移与形态素浓度诱导分化的
自适应神经控制适配器（Embryo Adapter）演化训练与对照实验引擎。
"""

import math
import random
import json
import os
import sys
import time

PRIMITIVES = [
    "EMA", "DIFF", "INTEGRAL", "SUM", "SUB", "MULTIPLY", "RATIO", "ABS",
    "DELAY_N", "OSCILLATOR", "QUADRATIC", "THRESHOLD", "HYSTERESIS",
    "AND", "INHIBIT", "DEADZONE", "MIN_MAX"
]

class EmbryoGenome:
    def __init__(self, genome_id=0):
        self.genome_id = genome_id
        self.maternal_loci = []
        self.cleavage_factor = random.uniform(1.2, 2.5) # 卵裂增殖指数
        self.morphogen_gradient_sensitivity = random.uniform(0.5, 1.8) # 形态素敏感度

    def mutate(self):
        for loc in self.maternal_loci:
            if random.random() < 0.20:
                loc["weight"] += random.gauss(0.0, 0.25)
            if random.random() < 0.08:
                loc["op_type"] = random.choice(PRIMITIVES)
        if random.random() < 0.10:
            self.cleavage_factor = max(1.0, min(3.0, self.cleavage_factor + random.gauss(0.0, 0.15)))
        if random.random() < 0.10:
            self.morphogen_gradient_sensitivity = max(0.2, min(2.5, self.morphogen_gradient_sensitivity + random.gauss(0.0, 0.15)))

class EmbryoAdapter:
    def __init__(self, genome):
        self.genome = genome
        self.cells = []
        self.synapses = []
        self.states = []
        self.develop()

    def develop(self):
        # 1. 受精卵启动 (Zygote)
        # 2. 卵裂 (Cleavage)
        n_cells = max(6, int(round(len(self.genome.maternal_loci) * self.genome.cleavage_factor)))
        positions = []
        for i in range(n_cells):
            px = random.uniform(-60.0, 60.0)
            py = random.uniform(-40.0, 40.0)
            positions.append([px, py])

        # 3. 原肠期 (Gastrulation) 模拟 3D 力场排斥与极性拉伸
        for _ in range(8):
            for i in range(n_cells):
                for j in range(i + 1, n_cells):
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-4
                    if dist < 40.0:
                        repulsion = (40.0 - dist) * 0.15
                        positions[i][0] -= (dx / dist) * repulsion
                        positions[j][0] += (dx / dist) * repulsion
                        positions[i][1] -= (dy / dist) * repulsion
                        positions[j][1] += (dy / dist) * repulsion

        # 4. 形态素梯度诱导命运分化 (Differentiation)
        # 沿 X 轴排序：最前端为感觉受体，最后端为效应动作，中间为功能细胞
        sorted_indices = sorted(range(n_cells), key=lambda idx: positions[idx][0])

        self.cells = []
        # 前端 2 个受体
        self.cells.append({"id": sorted_indices[0], "type": "SENSE_0", "weight": 1.0, "x": -100.0, "y": -30.0})
        self.cells.append({"id": sorted_indices[1], "type": "SENSE_1", "weight": 1.0, "x": -100.0, "y": 30.0})

        # 后端 2 个效应器
        self.cells.append({"id": sorted_indices[-2], "type": "ACT_POS", "weight": 1.0, "x": 120.0, "y": -30.0})
        self.cells.append({"id": sorted_indices[-1], "type": "ACT_NEG", "weight": 1.0, "x": 120.0, "y": 30.0})

        # 中间代谢与门控细胞
        for k in range(2, n_cells - 2):
            cid = sorted_indices[k]
            loc_idx = (k - 2) % len(self.genome.maternal_loci)
            loc = self.genome.maternal_loci[loc_idx]
            self.cells.append({
                "id": cid,
                "type": loc["op_type"],
                "weight": loc["weight"],
                "x": positions[cid][0],
                "y": positions[cid][1]
            })

        # 5. 轴突生长与突触成型 (Synaptogenesis)
        self.synapses = []
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                ci = self.cells[i]
                cj = self.cells[j]
                if cj["x"] > ci["x"] + 15.0: # 严格前向轴突投射
                    dx = cj["x"] - ci["x"]
                    dy = cj["y"] - ci["y"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 140.0:
                        w = 1.0 if cj["type"] != "ACT_NEG" else -1.0
                        self.synapses.append((i, j, w * ci["weight"]))

        self.states = [0.0] * len(self.cells)

    def reset(self):
        self.states = [0.0] * len(self.cells)

    def forward(self, inputs):
        self.states[0] = inputs[0] # CTE / 距离误差
        self.states[1] = inputs[1] # 速度误差 / 航向误差

        for syn in self.synapses:
            src_idx, dst_idx, w = syn
            src_val = self.states[src_idx]
            dst_cell = self.cells[dst_idx]
            op = dst_cell["type"]

            if op == "EMA":
                self.states[dst_idx] = self.states[dst_idx] * 0.8 + (src_val * w) * 0.2
            elif op == "DIFF":
                self.states[dst_idx] = (src_val * w) - self.states[dst_idx]
            elif op == "INTEGRAL":
                self.states[dst_idx] = max(-4.0, min(4.0, self.states[dst_idx] + (src_val * w) * 0.05))
            elif op == "THRESHOLD":
                self.states[dst_idx] = 1.0 if src_val * w > 0.1 else (-1.0 if src_val * w < -0.1 else 0.0)
            elif op == "HYSTERESIS":
                if src_val * w > 0.25:
                    self.states[dst_idx] = 1.0
                elif src_val * w < -0.25:
                    self.states[dst_idx] = -1.0
            else:
                self.states[dst_idx] = math.tanh(self.states[dst_idx] + src_val * w)

        pos_act = math.tanh(self.states[2])
        neg_act = math.tanh(self.states[3])
        return pos_act, neg_act

def evaluate_embryo_fitness(adapter):
    adapter.reset()
    total_error = 0.0
    x, y, heading, v = 0.0, 0.0, 0.0, 8.0

    for t in range(120):
        s = t * 0.15
        ref_y = math.sin(s * 0.25) * 4.5 + math.cos(s * 0.12) * 2.0
        ref_h = math.atan2(0.25 * math.cos(s * 0.25) * 4.5 - 0.12 * math.sin(s * 0.12) * 2.0, 1.0)
        
        cte = y - ref_y
        h_err = heading - ref_h
        steer, brake = adapter.forward([cte, h_err])

        dt = 0.05
        heading += (v / 2.7) * math.tan(steer * 0.5) * dt
        x += v * math.cos(heading) * dt
        y += v * math.sin(heading) * dt
        v += (1.0 - abs(brake) * 0.8) * dt
        v = max(1.0, min(20.0, v))

        total_error += abs(cte) + abs(h_err) * 1.5

    return 1000.0 / (1.0 + total_error)

def run_embryo_adapter_experiment(n_generations=20, pop_size=40):
    print("======================================================================")
    print("  SDSCC 胚胎发育形态发生自适应适配器 (Embryo Adapter) 演化训练")
    print(f"  演化规模: {n_generations} 代 | 种群 {pop_size} 胚胎个体")
    print("======================================================================")

    # 创建初始母源基因种群
    population = []
    for i in range(pop_size):
        eg = EmbryoGenome(i + 1)
        for j in range(6):
            eg.maternal_loci.append({
                "gene_id": j,
                "op_type": random.choice(PRIMITIVES),
                "weight": random.uniform(-1.2, 1.2)
            })
        population.append(eg)

    best_fitness = 0.0
    best_genome = None

    for gen in range(1, n_generations + 1):
        fitnesses = []
        for eg in population:
            adapter = EmbryoAdapter(eg)
            fit = evaluate_embryo_fitness(adapter)
            fitnesses.append((eg, fit))
            if fit > best_fitness:
                best_fitness = fit
                best_genome = eg

        fitnesses.sort(key=lambda x: x[1], reverse=True)
        avg_fit = sum(f for _, f in fitnesses) / len(fitnesses)

        print(f"  [Gen {gen:02d}/{n_generations}] 最高适应度: {best_fitness:.2f} | 均值: {avg_fit:.2f} | 精英细胞数: {len(EmbryoAdapter(best_genome).cells)}")

        # 演化繁殖
        next_gen = [fitnesses[0][0], fitnesses[1][0]] # 精英保留
        while len(next_gen) < pop_size:
            parent = random.choice(fitnesses[:pop_size // 2])[0]
            child = EmbryoGenome(random.randint(1, 1000000))
            child.maternal_loci = [dict(loc) for loc in parent.maternal_loci]
            child.cleavage_factor = parent.cleavage_factor
            child.morphogen_gradient_sensitivity = parent.morphogen_gradient_sensitivity
            child.mutate()
            next_gen.append(child)
        population = next_gen

    # 保存胚胎适配器检查点
    out_dir = "/home/caixuf/code/kun-cellular/models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embryo_morphogenetic_adapter_checkpoint.json")

    mature_adapter = EmbryoAdapter(best_genome)
    export_data = {
        "model_type": "EmbryoMorphogeneticAdapter",
        "best_fitness": best_fitness,
        "mature_cells_count": len(mature_adapter.cells),
        "mature_synapses_count": len(mature_adapter.synapses),
        "cleavage_factor": best_genome.cleavage_factor,
        "morphogen_gradient_sensitivity": best_genome.morphogen_gradient_sensitivity,
        "maternal_loci": best_genome.maternal_loci,
        "cells": mature_adapter.cells
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print("======================================================================")
    print(f"  ✓ 胚胎发育适配器训练圆满成功！模型已保存至: {out_path}")
    print(f"  ✓ 成熟细胞规模: {len(mature_adapter.cells)} | 突触数: {len(mature_adapter.synapses)} | 最佳适应度: {best_fitness:.2f}")
    print("======================================================================")

if __name__ == "__main__":
    run_embryo_adapter_experiment()
