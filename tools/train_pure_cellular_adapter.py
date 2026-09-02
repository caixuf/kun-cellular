#!/usr/bin/env python3
"""
Pure Physical Speciation Cellular Adapter Evolution Trainer
============================================================
基于纯物理 24 原语动力学元胞、基因拓扑兼容距离 (Speciation Niche)
与两性基因重组 (Sexual Crossover) 的自适应控制适配器演化训练器。
"""

import math
import random
import json
import os
import sys
import time
import numpy as np

# 24 种离散动力学原语
PRIMITIVES = [
    "EMA", "DIFF", "INTEGRAL", "SUM", "SUB", "MULTIPLY", "RATIO", "ABS",
    "DELAY_N", "OSCILLATOR", "QUADRATIC", "THRESHOLD", "HYSTERESIS",
    "AND", "INHIBIT", "DEADZONE", "MIN_MAX"
]

class CellularLocus:
    def __init__(self, gene_id, op_type, weight=1.0, metabolic_cost=0.02):
        self.gene_id = gene_id
        self.op_type = op_type
        self.weight = weight
        self.metabolic_cost = metabolic_cost

class PureCellularGenome:
    def __init__(self, genome_id=0):
        self.genome_id = genome_id
        self.loci = []
        self.lineage_hash = "GEN00"

    def mutate(self, rng_seed=None):
        if rng_seed:
            random.seed(rng_seed)
        # 点突变：权重微调或算子突变
        for loc in self.loci:
            if random.random() < 0.15:
                loc.weight += random.gauss(0.0, 0.2)
            if random.random() < 0.05:
                loc.op_type = random.choice(PRIMITIVES)
        # 拓扑扩展/剪枝
        if random.random() < 0.08 and len(self.loci) < 24:
            new_id = len(self.loci)
            self.loci.append(CellularLocus(new_id, random.choice(PRIMITIVES), random.uniform(-1.0, 1.0)))
        elif random.random() < 0.04 and len(self.loci) > 4:
            idx = random.randint(0, len(self.loci) - 1)
            self.loci.pop(idx)

    @staticmethod
    def compatibility_distance(g1, g2, c1=1.0, c2=0.4):
        n1, n2 = len(g1.loci), len(g2.loci)
        max_n = max(n1, n2)
        if max_n == 0:
            return 0.0
        min_n = min(n1, n2)
        match_count = 0
        w_diff_sum = 0.0
        for i in range(min_n):
            if g1.loci[i].op_type == g2.loci[i].op_type:
                match_count += 1
                w_diff_sum += abs(g1.loci[i].weight - g2.loci[i].weight)
        disjoint = max_n - match_count
        avg_w_diff = (w_diff_sum / match_count) if match_count > 0 else 1.0
        return (c1 * disjoint / max_n) + (c2 * avg_w_diff)

    @staticmethod
    def crossover(mom, dad):
        child = PureCellularGenome(random.randint(1, 100000000))
        min_len = min(len(mom.loci), len(dad.loci))
        cp = random.randint(1, max(1, min_len))
        for i in range(cp):
            loc = mom.loci[i]
            child.loci.append(CellularLocus(loc.gene_id, loc.op_type, loc.weight, loc.metabolic_cost))
        for i in range(cp, len(dad.loci)):
            loc = dad.loci[i]
            child.loci.append(CellularLocus(loc.gene_id, loc.op_type, loc.weight, loc.metabolic_cost))
        child.lineage_hash = mom.lineage_hash[:3] + dad.lineage_hash[:3]
        return child

class CellularAdapterPhenotype:
    def __init__(self, genome):
        self.genome = genome
        self.states = [0.0] * len(genome.loci)

    def reset(self):
        self.states = [0.0] * len(self.genome.loci)

    def forward(self, inputs):
        # inputs: [cte, speed_error, target_curv, heading_err]
        cte, spd_err, curv, h_err = inputs
        s0 = cte * 1.5 + h_err * 2.0
        s1 = spd_err * 0.8 - curv * 1.2

        current_val = s0
        for i, loc in enumerate(self.genome.loci):
            op = loc.op_type
            w = loc.weight
            if op == "EMA":
                self.states[i] = self.states[i] * 0.8 + current_val * 0.2
                current_val = self.states[i] * w
            elif op == "DIFF":
                diff = current_val - self.states[i]
                self.states[i] = current_val
                current_val = diff * w
            elif op == "INTEGRAL":
                self.states[i] = max(-5.0, min(5.0, self.states[i] + current_val * 0.05))
                current_val = self.states[i] * w
            elif op == "SUM":
                current_val = (current_val + s1) * w
            elif op == "SUB":
                current_val = (current_val - s1) * w
            elif op == "MULTIPLY":
                current_val = (current_val * (1.0 + math.tanh(s1))) * w
            elif op == "ABS":
                current_val = abs(current_val) * w
            elif op == "THRESHOLD":
                current_val = w if current_val > 0.1 else (-w if current_val < -0.1 else 0.0)
            elif op == "HYSTERESIS":
                if current_val > 0.3:
                    self.states[i] = 1.0
                elif current_val < -0.3:
                    self.states[i] = -1.0
                current_val = self.states[i] * w
            elif op == "DEADZONE":
                current_val = 0.0 if abs(current_val) < 0.05 else current_val * w
            else:
                current_val = math.tanh(current_val) * w

        steer_cmd = math.tanh(current_val)
        speed_cmd = max(0.0, 1.0 - abs(steer_cmd) * 0.6)
        return steer_cmd, speed_cmd

def evaluate_adapter_fitness(adapter):
    adapter.reset()
    # 模拟复杂赛道循迹：S弯、发卡弯与急加速急减速
    total_error = 0.0
    steps = 150
    x, y, heading = 0.0, 0.0, 0.0
    v = 10.0 # 初始车速 10 m/s

    for t in range(steps):
        s = t * 0.1
        # 目标道路中心线与目标航向
        ref_y = math.sin(s * 0.2) * 5.0 + math.cos(s * 0.08) * 3.0
        ref_heading = math.atan2(0.2 * math.cos(s * 0.2) * 5.0 - 0.08 * math.sin(s * 0.08) * 3.0, 1.0)
        ref_curv = abs(math.sin(s * 0.2) * 0.04)

        cte = y - ref_y
        h_err = heading - ref_heading
        target_v = 15.0 / (1.0 + ref_curv * 20.0)
        spd_err = target_v - v

        steer, throttle = adapter.forward([cte, spd_err, ref_curv, h_err])

        # 车辆运动学模型积分
        dt = 0.05
        heading += (v / 2.7) * math.tan(steer * 0.45) * dt
        x += v * math.cos(heading) * dt
        y += v * math.sin(heading) * dt
        v += (throttle * 2.0 - 0.5) * dt
        v = max(1.0, min(25.0, v))

        total_error += abs(cte) + abs(h_err) * 2.0

    # 适应度函数：误差越小适应度越高
    fitness = 1000.0 / (1.0 + total_error)
    return fitness

class Species:
    def __init__(self, species_id, representative):
        self.species_id = species_id
        self.representative = representative
        self.members = []
        self.adjusted_fitness = 0.0

def train_pure_cellular_adapter(n_generations=25, pop_size=48, speciation_threshold=0.35):
    print("======================================================================")
    print("  SDSCC 纯物理物种自适应控制适配器 (Speciation Cellular Adapter) 演化训练")
    print(f"  配置: {n_generations} 代 | 种群 {pop_size} 个体 | 物种利基阈值 {speciation_threshold}")
    print("======================================================================")

    # 初始化始祖种群
    population = []
    for i in range(pop_size):
        g = PureCellularGenome(i + 1)
        g.lineage_hash = f"Lin{i:02d}"
        for j in range(6):
            g.loci.append(CellularLocus(j, random.choice(PRIMITIVES), random.uniform(-1.0, 1.0)))
        population.append(g)

    species_list = []
    best_overall_fitness = 0.0
    best_overall_genome = None

    for gen in range(1, n_generations + 1):
        # 1. 评估适应度
        fitnesses = []
        for g in population:
            pheno = CellularAdapterPhenotype(g)
            fit = evaluate_adapter_fitness(pheno)
            fitnesses.append((g, fit))
            if fit > best_overall_fitness:
                best_overall_fitness = fit
                best_overall_genome = g

        # 2. 物种聚类 (Speciation Clustering)
        for sp in species_list:
            sp.members.clear()

        for g, fit in fitnesses:
            placed = False
            for sp in species_list:
                dist = PureCellularGenome.compatibility_distance(g, sp.representative)
                if dist < speciation_threshold:
                    sp.members.append((g, fit))
                    placed = True
                    break
            if not placed:
                new_sp = Species(len(species_list) + 1, g)
                new_sp.members.append((g, fit))
                species_list.append(new_sp)

        # 清除灭绝物种
        species_list = [sp for sp in species_list if len(sp.members) > 0]

        # 3. 显式适应度共享 (Explicit Fitness Sharing)
        for sp in species_list:
            sp_size = len(sp.members)
            sp.members.sort(key=lambda x: x[1], reverse=True)
            sp.adjusted_fitness = sum(f / sp_size for _, f in sp.members)

        # 4. 物种内精英保留与两性基因重组繁衍
        new_population = []
        total_adj = sum(sp.adjusted_fitness for sp in species_list) + 1e-6

        for sp in species_list:
            # 精英保留 (Elitism)
            new_population.append(sp.members[0][0])
            # 根据物种贡献度分配繁衍配额
            quota = max(1, int(round((sp.adjusted_fitness / total_adj) * pop_size)))
            for _ in range(quota - 1):
                parent_a = random.choice(sp.members[:max(1, len(sp.members)//2)])[0]
                parent_b = random.choice(sp.members)[0]
                child = PureCellularGenome.crossover(parent_a, parent_b)
                child.mutate()
                new_population.append(child)

        # 补齐或截断到固定种群规模
        while len(new_population) < pop_size:
            sp = random.choice(species_list)
            p = random.choice(sp.members)[0]
            c = PureCellularGenome(random.randint(1, 1000000))
            c.loci = [CellularLocus(l.gene_id, l.op_type, l.weight) for l in p.loci]
            c.mutate()
            new_population.append(c)
        population = new_population[:pop_size]

        avg_fit = sum(f for _, f in fitnesses) / len(fitnesses)
        print(f"  [Gen {gen:02d}/{n_generations}] 最高适应度: {best_overall_fitness:.2f} | 种群均值: {avg_fit:.2f} | 繁衍物种数: {len(species_list)} 种")

    # 保存最优自适应适配器检查点
    out_dir = "/home/caixuf/code/kun-cellular/models"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pure_cellular_adapter_checkpoint.json")

    export_data = {
        "model_type": "PureCellularAdapter",
        "best_fitness": best_overall_fitness,
        "species_count": len(species_list),
        "loci_count": len(best_overall_genome.loci),
        "loci": [
            {"gene_id": loc.gene_id, "op_type": loc.op_type, "weight": loc.weight}
            for loc in best_overall_genome.loci
        ]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print("======================================================================")
    print(f"  ✓ 演化适配器训练圆满完成！最优基因组已导出至: {out_path}")
    print(f"  ✓ 最佳适应度: {best_overall_fitness:.2f} | 基因位点数: {len(best_overall_genome.loci)}")
    print("======================================================================")
    return best_overall_genome

if __name__ == "__main__":
    train_pure_cellular_adapter()
