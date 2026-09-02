#!/usr/bin/env python3
"""
Real Evolutionary Training Pipeline for Software-Defined Silicon Cellular Computer
---------------------------------------------------------------------------------
1. Loads real historical commodity market data from kunquant/data/history/rb.csv
2. Trains a population of cellular organisms across multiple generations
3. Evaluates real fitness (Sharpe Ratio, Cumulative Return, Max Drawdown)
4. Saves the genuinely trained champion organism to checkpoints/real_trained_champion.json
"""

import os
import csv
import math
import json
import random
import copy

CSV_PATH = "/home/caixuf/code/kunquant/data/history/rb.csv"
CHECKPOINT_PATH = "/home/caixuf/code/kun-cellular/checkpoints/real_trained_champion.json"

# ============================================================================
# 1. 真实历史数据加载 (Load Real Historical Market Data)
# ============================================================================

def load_real_data(filepath=CSV_PATH):
    bars = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        for row in reader:
            if len(row) < 7:
                continue
            date_str = row[1]
            open_p = float(row[2])
            high_p = float(row[3])
            low_p = float(row[4])
            close_p = float(row[5])
            volume = float(row[6])
            bars.append({
                "date": date_str,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume
            })
    print(f"[*] 成功加载真实历史行情数据: {len(bars)} 根日线数据 (跨度: {bars[0]['date']} 至 {bars[-1]['date']})")
    return bars

# ============================================================================
# 2. 细胞有机体与演化评估 (Cellular Organism & Real Fitness Evaluator)
# ============================================================================

CANDIDATE_TYPES = [
    "EMA", "DIFF", "INTEGRAL", "SUM", "SUB", "MUL", "RATIO", "ABS", 
    "OSCILLATOR", "QUADRATIC", "THRESH", "HYST", "AND", "INHIB", "DEADZONE"
]

class RealCellularIndividual:
    def __init__(self, id_counter=10):
        self.cells = [
            {"id": 0, "type": "SENSE0", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": -200.0, "y": -60.0, "z": 0.0},
            {"id": 1, "type": "SENSE1", "p1": 1.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": -200.0, "y": 60.0, "z": 0.0},
            {"id": 2, "type": "EMA", "p1": 0.05, "p2": 0.0, "s": 0.0, "out": 0.0, "x": -100.0, "y": -40.0, "z": 0.0},
            {"id": 3, "type": "EMA", "p1": 0.20, "p2": 0.0, "s": 0.0, "out": 0.0, "x": -100.0, "y": 40.0, "z": 0.0},
            {"id": 4, "type": "SUB", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 5, "type": "HYST", "p1": 0.01, "p2": -0.01, "s": 0.0, "out": 0.0, "x": 80.0, "y": 0.0, "z": 0.0},
            {"id": 6, "type": "ACT_POS", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": 180.0, "y": -40.0, "z": 0.0},
            {"id": 7, "type": "ACT_NEG", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": 180.0, "y": 40.0, "z": 0.0},
            {"id": 8, "type": "ACT_LOCK", "p1": 0.0, "p2": 0.0, "s": 0.0, "out": 0.0, "x": 180.0, "y": 100.0, "z": 0.0}
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
        self.fitness = -999.0
        self.pnl = 0.0
        self.sharpe = 0.0
        self.max_dd = 0.0

    def reset_state(self):
        for c in self.cells:
            c["s"] = 0.0
            c["out"] = 0.0

    def forward_step(self, inputs):
        by_id = {c["id"]: i for i, c in enumerate(self.cells)}
        port_in = [[0.0, 0.0] for _ in range(len(self.cells))]
        
        for s in self.synapses:
            if not s["active"]: continue
            fi, ti = by_id.get(s["from"]), by_id.get(s["to"])
            if fi is not None and ti is not None:
                port_in[ti][s["port"]] += self.cells[fi]["out"] * s["w"]

        act_buy = 0.0
        act_sell = 0.0
        act_lock = 0.0

        for i, c in enumerate(self.cells):
            i0, i1 = port_in[i][0], port_in[i][1]
            t = c["type"]
            if t == "SENSE0": c["out"] = inputs[0] * c["p1"]
            elif t == "SENSE1": c["out"] = inputs[1] * c["p1"]
            elif t == "EMA":
                a = max(0.005, min(0.99, c["p1"]))
                c["s"] = a * i0 + (1.0 - a) * c["s"]
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
            elif t == "QUADRATIC": c["out"] = (1.0 if i0 >= 0 else -1.0) * (i0 * i0)
            elif t == "THRESH": c["out"] = 1.0 if i0 > c["p1"] else 0.0
            elif t == "HYST":
                if abs(i0) > c["p1"]: c["out"] = i0
                elif abs(i0) < abs(c["p2"]): c["out"] = 0.0
            elif t == "AND": c["out"] = 1.0 if (i0 > 0 and i1 > 0) else 0.0
            elif t == "INHIB": c["out"] = i0 * max(0.0, 1.0 - i1)
            elif t == "DEADZONE": c["out"] = i0 if abs(i0) > c["p1"] else 0.0
            elif t == "ACT_POS": act_buy = max(0.0, min(1.0, i0))
            elif t == "ACT_NEG": act_sell = max(0.0, min(1.0, i0))
            elif t == "ACT_LOCK": act_lock = 1.0 if i0 > 0.8 else 0.0

        return act_buy, act_sell, act_lock

    def mutate(self):
        # 突触权重变异
        for s in self.synapses:
            if random.random() < 0.2:
                s["w"] += random.gauss(0, 0.2)
        # 细胞参数微调
        for c in self.cells:
            if random.random() < 0.15:
                c["p1"] = max(0.001, min(2.0, c["p1"] + random.gauss(0, 0.05)))
        # 结构有丝分裂 (Synaptic Mitosis)
        if random.random() < 0.25 and len(self.cells) < 40:
            syn = random.choice(self.synapses)
            new_id = max(c["id"] for c in self.cells) + 1
            new_type = random.choice(CANDIDATE_TYPES)
            new_cell = {
                "id": new_id,
                "type": new_type,
                "p1": random.uniform(0.01, 0.5),
                "p2": random.uniform(-0.1, 0.1),
                "s": 0.0, "out": 0.0,
                "x": round(random.uniform(-100, 100), 1),
                "y": round(random.uniform(-80, 80), 1),
                "z": round(random.uniform(-20, 20), 1)
            }
            self.cells.append(new_cell)
            orig_to = syn["to"]
            syn["to"] = new_id
            self.synapses.append({
                "from": new_id,
                "to": orig_to,
                "port": 0,
                "w": random.uniform(0.5, 1.5),
                "active": True
            })

def evaluate_on_real_data(ind, bars):
    ind.reset_state()
    capital = 1000000.0
    position = 0.0 # -1.0 to 1.0
    equity_curve = [capital]
    returns = []
    
    # 标准化基准
    p0 = bars[0]["close"]
    vol0 = bars[0]["volume"]

    for i in range(1, len(bars)):
        prev_bar = bars[i-1]
        cur_bar = bars[i]
        
        # 输入特征: 归一化价格变化率与成交量倍比
        d_price = (prev_bar["close"] - p0) / p0
        d_vol = math.log(max(1.0, prev_bar["volume"] / max(1.0, vol0)))
        
        buy_sig, sell_sig, lock_sig = ind.forward_step([d_price, d_vol])
        
        # 目标仓位决策
        if lock_sig > 0.5:
            target_pos = 0.0 # 免疫熔断平仓
        else:
            if buy_sig > 0.3 and buy_sig > sell_sig:
                target_pos = 1.0
            elif sell_sig > 0.3 and sell_sig > buy_sig:
                target_pos = -1.0
            else:
                target_pos = position

        # 计算收益 (T+1 开盘成交)
        price_ret = (cur_bar["close"] - cur_bar["open"]) / cur_bar["open"]
        step_pnl = position * price_ret * capital
        
        # 手续费摩擦 (调仓产生 1.5 bp 佣金)
        if abs(target_pos - position) > 0.1:
            capital -= capital * 0.00015
            
        capital += step_pnl
        equity_curve.append(capital)
        returns.append(step_pnl / equity_curve[-2])
        position = target_pos

    # 计算夏普比率与最大回撤
    if not returns or capital <= 0:
        ind.fitness = -999.0
        return ind.fitness

    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret)**2 for r in returns) / len(returns)
    std_ret = math.sqrt(variance) if variance > 1e-12 else 1e-6
    annual_sharpe = (mean_ret / std_ret) * math.sqrt(252)

    # 计算最大回撤
    peak = equity_curve[0]
    max_drawdown = 0.0
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > max_drawdown: max_drawdown = dd

    cum_return = (capital - 1000000.0) / 1000000.0
    
    # 综合适应度评分: 夏普比率 * (1 - 最大回撤) + 累计收益率
    ind.fitness = annual_sharpe * (1.0 - max_drawdown) + cum_return * 0.5
    ind.pnl = cum_return
    ind.sharpe = annual_sharpe
    ind.max_dd = max_drawdown
    return ind.fitness

# ============================================================================
# 3. 演化训练主循环 (Evolutionary Training Loop)
# ============================================================================

def train():
    bars = load_real_data()
    # 划分样本内训练集 (前 70%) 与样本外测试集 (后 30%)
    train_split = int(len(bars) * 0.7)
    train_bars = bars[:train_split]
    test_bars = bars[train_split:]
    print(f"[*] 样本内训练集: {len(train_bars)} 根日线 ({train_bars[0]['date']} ~ {train_bars[-1]['date']})")
    print(f"[*] 样本外盲测集: {len(test_bars)} 根日线 ({test_bars[0]['date']} ~ {test_bars[-1]['date']})")
    
    pop_size = 32
    generations = 30
    population = [RealCellularIndividual() for _ in range(pop_size)]
    
    print("\n======================================================================")
    print(" 开始在真实大宗商品期货历史大数据上进行形态发生细胞演化训练...")
    print("======================================================================")

    best_global = None

    for gen in range(1, generations + 1):
        for ind in population:
            evaluate_on_real_data(ind, train_bars)

        population.sort(key=lambda x: x.fitness, reverse=True)
        best_cur = population[0]

        if best_global is None or best_cur.fitness > best_global.fitness:
            best_global = copy.deepcopy(best_cur)

        if gen % 5 == 0 or gen == 1 or gen == generations:
            print(f"Gen {gen:02d}/{generations:02d} | 最佳适应度: {best_cur.fitness:6.3f} | 收益率: {best_cur.pnl*100:+6.2f}% | 夏普比: {best_cur.sharpe:5.2f} | 最大回撤: {best_cur.max_dd*100:5.2f}% | 细胞数: {len(best_cur.cells)}")

        # 锦标赛选择与突变繁衍
        new_pop = [copy.deepcopy(best_global)] # 保留全局最优精英
        while len(new_pop) < pop_size:
            p1 = random.choice(population[:8])
            child = copy.deepcopy(p1)
            child.mutate()
            new_pop.append(child)
        population = new_pop

    # 样本外盲测检验
    print("\n======================================================================")
    print(" 训练完成，对冠军个体进行严格的样本外盲测审计 (Holdout Out-of-Sample)...")
    print("======================================================================")
    evaluate_on_real_data(best_global, test_bars)
    print(f"[+] 样本外盲测结果:")
    print(f"    - 样本外累计收益率: {best_global.pnl * 100:+.2f}%")
    print(f"    - 样本外年化夏普比: {best_global.sharpe:.2f}")
    print(f"    - 样本外最大历史回撤: {best_global.max_dd * 100:.2f}%")
    print(f"    - 演化拓扑结构: {len(best_global.cells)} 个功能细胞, {len(best_global.synapses)} 条突触连接")

    # 保存检查点文件
    checkpoint_data = {
        "model_type": "Real_Trained_SDSCC_Champion",
        "dataset": "kunquant/data/history/rb.csv",
        "train_generations": generations,
        "sample_out_pnl": best_global.pnl,
        "sample_out_sharpe": best_global.sharpe,
        "sample_out_max_dd": best_global.max_dd,
        "cells": best_global.cells,
        "synapses": best_global.synapses
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"[+] 真实训练冠军模型已成功存盘: {CHECKPOINT_PATH}")

if __name__ == "__main__":
    train()
