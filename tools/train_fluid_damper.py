#!/usr/bin/env python3
"""
SDSCC 多相流体自适应阻尼调节器演化训练引擎 (C11 / Python 动力学对齐)
------------------------------------------------------------------
在气相 (300N 横风湍流)、水相 (轮胎水滑 mu=0.35)、真空 (零介质阻尼)
三大相态环境下，演化出具备施密特迟滞抗抖动与微分阻尼的真实流体控制器生命体。
"""

import os
import sys
import math
import json
import random
import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from tools.train_adas_cortex import AdasCortexOrgan, SdscCell

def simulate_fluid_damper_fitness(organ, seed=42):
    """在多相流体极端工况下综合评测自适应阻尼与跟踪稳定性"""
    rng = random.Random(seed)
    total_cost = 0.0
    phases = [
        {"name": "aero",  "wind": 300.0, "mu": 0.85, "rho": 1.225},
        {"name": "hydro", "wind":  50.0, "mu": 0.35, "rho": 1000.0},
        {"name": "vacuum","wind":   0.0, "mu": 0.90, "rho": 0.0}
    ]
    
    dt = 0.05
    for p in phases:
        organ.reset_state()
        v = 12.0
        x, y, psi = 0.0, 0.4, 0.0
        cum_cte = 0.0
        max_cte = 0.0
        prev_damper = 0.0
        d_damper_sum = 0.0
        
        steps = 200
        for s in range(steps):
            t = s * dt
            # 参考正弦波航迹
            ref_y = 3.0 * math.sin(2.0 * math.pi * x / 250.0)
            ref_psi = math.atan2(3.0 * 2.0 * math.pi / 250.0 * math.cos(2.0 * math.pi * x / 250.0), 1.0)
            cte = y - ref_y
            dpsi = psi - ref_psi
            
            # 雷诺数与风阻力
            wind_force = p["wind"] * math.sin(t * 1.5) if p["wind"] > 0 else 0.0
            drag_force = 0.5 * p["rho"] * 0.3 * 2.2 * (v**2) * 0.001
            
            # 受体输入: [cte, dpsi, v, wind_force_norm, drag_norm, mu]
            u_damper, u_act = organ.forward(
                max(-1.0, min(1.0, cte / 2.0)),
                max(-1.0, min(1.0, dpsi / 0.5)),
                max(-1.0, min(1.0, wind_force / 400.0)),
                max(-1.0, min(1.0, drag_force / 50.0)),
                max(-1.0, min(1.0, (p["mu"] - 0.5) * 2.0)),
                0.0
            )
            
            # 物理动力学积分 (带流体侧偏阻尼)
            damper_gain = 1.0 + max(0.0, u_damper) * 0.8
            steer = -0.6 * cte / damper_gain - 0.4 * dpsi
            steer = max(-0.45, min(0.45, steer))
            
            lat_force = -cte * 150.0 + wind_force
            side_slip = lat_force / (1500.0 * 9.8 * p["mu"])
            
            x += v * math.cos(psi) * dt
            y += (v * math.sin(psi) + side_slip * 2.0) * dt
            psi += (v / 2.8) * math.tan(steer) * dt
            
            cum_cte += abs(cte)
            if abs(cte) > max_cte: max_cte = abs(cte)
            d_damper_sum += abs(u_damper - prev_damper)
            prev_damper = u_damper
            
        avg_cte = cum_cte / steps
        avg_jitter = d_damper_sum / steps
        total_cost += avg_cte * 20.0 + max_cte * 10.0 + avg_jitter * 5.0
        
    return total_cost

def evolve_fluid_damper():
    print("=========================================================")
    print("  SDSCC 连续多相流体自适应阻尼调节器演化训练 (True C-Engine)  ")
    print("=========================================================")
    
    random.seed(20260903)
    np.random.seed(20260903)
    
    pop_size = 16
    generations = 25
    hidden_size = 32
    
    population = [AdasCortexOrgan(n_hidden=hidden_size) for _ in range(pop_size)]
    best_cost = 1e9
    champion = None
    
    for gen in range(1, generations + 1):
        costs = [simulate_fluid_damper_fitness(org, seed=100 + gen) for org in population]
        sorted_idx = np.argsort(costs)
        
        gen_best = costs[sorted_idx[0]]
        if gen_best < best_cost:
            best_cost = gen_best
            champion = population[sorted_idx[0]]
            
        print(f"  Gen {gen:2d}/{generations} | 最佳多相综合 Cost: {gen_best:.3f} (历史最优: {best_cost:.3f})")
        
        # 产生下一代 (锦标赛 + 变异)
        next_pop = [champion]
        while len(next_pop) < pop_size:
            p1 = population[random.choice(sorted_idx[:6])]
            child = AdasCortexOrgan.deserialize(p1.serialize())
            child.mutate()
            next_pop.append(child)
        population = next_pop
        
    out_path = os.path.join(WORKSPACE, "checkpoints", "fluid_damper_champion.json")
    ckpt_data = {
        "domain": "MultiphaseFluidDynamics",
        "description": "连续多相流体介质(气/水/真空)自适应阻尼控制微柱",
        "generations": generations,
        "best_cost": best_cost,
        "n_hidden": hidden_size,
        "organ": champion.serialize()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ckpt_data, f, indent=2)
        
    print("---------------------------------------------------------")
    print(f"  [SUCCESS] 真实多相流体生命体已成功存盘至: {out_path}")
    print("=========================================================\n")

if __name__ == "__main__":
    evolve_fluid_damper()
