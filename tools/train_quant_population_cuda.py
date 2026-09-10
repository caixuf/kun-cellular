#!/usr/bin/env python3
# ============================================================================
# train_quant_population_cuda.py — L2 任务层: 量化种群生态 GPU 战役
#
# 大规模演化并行加速主战役 (用户裁决 + 底座动工授权 2026-09-10):
#   - 256 成员全 GPU 批量演化 (CUDACellularPopulation per_column_input 逐列注入模式)
#   - 43 资产 × 24 细胞, in_dim=4/列 (逐列独立特征注入), 组合模拟全向量化驻留 GPU
#   - 生态池: val 选型 + 相关性门 (0.90) + 体制标签 ≥2 + OOS 一次性盲报 (预注册纪律同 v2)
#
# 如实声明 (新世系, Stage 1):
#   1. GPU 拓扑 = 稠密列内突触 + 环柱轴突, 算子池无 CORRELATION/INHIBIT — 与 CSR 稀疏微柱不同构
#   2. SDSC-BIN 导出可行但 **C11 位级对账未达成** (现行 bin 运行时为全局前 in_dim 注入语义,
#      逐列异输入无法表达) — Stage 2 立项: SDSC-BIN v5 受体映射格式 (底座评审, 用户已授权)
# ============================================================================
import os
import sys
import csv
import math
import time
import json

import numpy as np
import torch

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE)

from tools.cuda_cellular_engine import CUDACellularPopulation

DATA_DIR = "/home/caixuf/code/kunquant/data/history"

# ── 预注册冻结常数 (GPU 战役 v2, 启动后不可改) ──
# v2 修订 (首战实证: 引擎 GA 锦标赛克隆 → 256 成员 Gen20 前全部坍缩, corr 门拒 255/256):
#   新增 GPU 列级杂交 (整列掩码混合, 与 CPU column_crossover 同语义);
#   多样性哨兵: 每 10 代监控成员收益相关矩阵, corr>0.9 占比>50% → 50% 非精英重置为随机移民;
#   其余不变: pop 256 / gen 200 / mut 0.06 / elite 0.10 / 池 K5 / corr 门 0.90。
POP_SIZE = 256
GENERATIONS = 200
NUM_COLS = 43
CELLS_PER_COL = 24
IN_DIM = 4
OUT_DIM = 2
POOL_K = 5
CORR_GATE = 0.90
MUT_RATE = 0.06
ELITE_RATIO = 0.10
CROSSOVER_PROB = 0.5
WATCHDOG_EVERY = 10
WATCHDOG_CORR_FRAC = 0.5


def load_series():
    syms = ["IF", "IC", "au", "ag", "cu", "al", "zn", "ni", "sn", "pb", "ss", "rb", "hc",
            "i", "j", "jm", "sc", "fu", "bu", "ta", "MA", "ru", "l", "pp", "v", "eg", "eb",
            "pg", "sp", "ur", "sa", "fg", "m", "y", "p", "oi", "c", "cs", "a", "rm",
            "cf", "sr", "ap", "jd"]
    series = []
    for sym in syms:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            continue  # 与 C++ load_all_assets 同款防御: 缺文件品种跳过
        rows = {}
        with open(path) as f:
            rdr = csv.reader(f)
            next(rdr)
            for tk in rdr:
                if len(tk) >= 7:
                    rows[tk[1]] = (float(tk[2]), float(tk[3]), float(tk[4]), float(tk[5]), float(tk[6]))
        series.append(rows)
    all_dates = sorted(set().union(*[set(s.keys()) for s in series]))
    train_d = [d for d in all_dates if d < "2013-01-01"]
    val_d = [d for d in all_dates if "2013-01-01" <= d < "2016-01-01"]
    test_d = [d for d in all_dates if d >= "2016-01-01"]
    return series, train_d, val_d, test_d


def precompute(series, dates):
    """numpy 移植 CorticalQuantTask::precompute_all (逐式同式)
    返回: feat[T,43,4], vol_inv[T,43], ret_next[T,43], has_next[T,43], has_cur[T,43]"""
    T, A = len(dates), len(series)
    feat = np.zeros((T, A, 4), dtype=np.float32)
    vol_inv = np.ones((T, A), dtype=np.float32)
    ret_next = np.zeros((T, A), dtype=np.float32)
    has_next = np.zeros((T, A), dtype=bool)
    has_cur = np.zeros((T, A), dtype=bool)

    for a, rows in enumerate(series):
        for d_idx, d in enumerate(dates):
            if d not in rows:
                continue
            o, h, l, c, v = rows[d]
            has_cur[d_idx, a] = True
            vol_pct = max(0.01, (h - l) / (c + 1e-4))
            vol_inv[d_idx, a] = 1.0 / vol_pct
            if d_idx >= 20:
                pd = dates[d_idx - 1]
                if pd in rows:
                    pc, pv = rows[pd][3], rows[pd][4]
                    ret = (c - pc) / (pc + 1e-4)
                    sum5 = sum20 = 0.0
                    valid = 0
                    for i in range(20):
                        dd = rows.get(dates[d_idx - i])
                        if dd is not None:
                            cc = dd[3]
                            if i < 5:
                                sum5 += cc
                            sum20 += cc
                            valid += 1
                    ma_diff = ((sum5 / 5.0) - (sum20 / valid)) / c if (valid >= 15 and c > 1e-4) else 0.0
                    vol_ratio = (v / pv - 1.0) if pv > 0 else 0.0
                    feat[d_idx, a, 0] = np.clip(ret * 20.0, -1.0, 1.0)
                    feat[d_idx, a, 1] = np.clip(ma_diff * 30.0, -1.0, 1.0)
                    feat[d_idx, a, 2] = np.clip((vol_pct - 0.02) * 40.0, -1.0, 1.0)
                    feat[d_idx, a, 3] = np.clip(vol_ratio * 0.5, -1.0, 1.0)
            if d_idx + 1 < T and dates[d_idx + 1] in rows:
                no, _, _, nc, _ = series[a][dates[d_idx + 1]]
                ret_next[d_idx, a] = (nc - no) / (no + 1e-4)
                has_next[d_idx, a] = True
    return feat, vol_inv, ret_next, has_next, has_cur


class QuantArrayPopulation(CUDACellularPopulation):
    """43 资产微柱 × 24 细胞 (逐列注入 in4 / 全局效应器 out2) — GPU 变体拓扑"""

    def __init__(self, pop_size=POP_SIZE, device="cuda"):
        super().__init__(pop_size=pop_size, num_columns=NUM_COLS, cells_per_col=CELLS_PER_COL,
                         in_dim=IN_DIM, out_dim=OUT_DIM, device=device, per_column_input=True)
        K = CELLS_PER_COL
        POOL = [4, 12, 5, 8, 16, 17, 11, 10, 14, 6]  # 引擎可用算子池 (无 CORRELATION/INHIBIT)
        for c in range(NUM_COLS):
            b = c * K
            self.op_types[b + 0:b + 4] = 0          # SENSE
            self.flags[b + 0:b + 4] = 0x01          # RECEPTOR
            self.op_types[b + 4] = 5                # INTEGRATE slow
            self.param1[:, b + 4] = 0.20
            self.op_types[b + 5] = 5                # INTEGRATE fast
            self.param1[:, b + 5] = 0.60
            self.op_types[b + 6] = 13               # SUB
            self.op_types[b + 7] = 16               # HYSTERESIS
            self.op_types[b + 8] = 12               # DIFF
            self.op_types[b + 9] = 17               # DEADZONE
            self.op_types[b + 10] = 4               # SUM
            self.op_types[b + 11] = 8               # DAMPER
            self.param1[:, b + 11] = 0.80
            for i in range(12, K - 2):
                self.op_types[b + i] = POOL[(i + c) % len(POOL)]
            self.op_types[b + K - 2] = 21           # ACT_POS
            self.op_types[b + K - 1] = 22           # ACT_NEG
            self.flags[b + K - 2:b + K] = 0x02      # EFFECTOR
        self.param1 = torch.clamp(torch.round(self.param1 * (255.0 / 4.0)), 0, 255) * (4.0 / 255.0)
        self._rebuild_masks()

    def _rebuild_masks(self):
        t = lambda op: torch.tensor(np.where(self.op_types == op)[0], device=self.device, dtype=torch.long)
        self.idx_sense = t(0); self.idx_sum = t(4); self.idx_integral = t(5)
        self.idx_amplify = t(6); self.idx_invert = t(7); self.idx_damper = t(8)
        self.idx_clip = t(9); self.idx_abs = t(10); self.idx_multiply = t(11)
        self.idx_diff = t(12); self.idx_sub = t(13); self.idx_ratio = t(14)
        self.idx_hyst = t(16); self.idx_deadzone = t(17); self.idx_fatigue = t(25)
        self.idx_act_pos = t(21); self.idx_act_neg = t(22); self.idx_act_reset = t(23)
        self.idx_passthru = t(26)
        m = {0: 'sense', 4: 'sum', 5: 'integral', 6: 'amplify', 7: 'invert', 8: 'damper',
             9: 'clip', 10: 'abs', 11: 'multiply', 12: 'diff', 13: 'sub', 14: 'ratio',
             16: 'hyst', 17: 'deadzone', 21: 'act_pos', 22: 'act_neg', 23: 'act_reset',
             25: 'fatigue', 26: 'passthru'}
        for op, name in m.items():
            setattr(self, f"has_{name}", getattr(self, f"idx_{name}").numel() > 0)

    def effector_signals(self):
        """全成员逐资产原始信号 [P, C] (读全细胞输出, 不止全局末尾切片)"""
        eff = self.outputs.view(self.pop_size, self.num_columns, self.cells_per_col)[:, :, -2:]
        return eff[:, :, 0] - eff[:, :, 1]


class BatchedPortfolioSim:
    """P 成员组合模拟全向量化 (C++ CorticalQuantTask::step_day 逐式语义对齐)"""

    def __init__(self, pop_size, feat, vol_inv, ret_next, has_next, has_cur, device):
        self.P = pop_size
        self.device = device
        self.feat = torch.tensor(feat, device=device)                    # [T,43,4]
        self.vol_inv = torch.tensor(vol_inv, device=device)
        self.ret_next = torch.tensor(ret_next, device=device)
        self.has_next_f = torch.tensor(has_next, dtype=torch.float32, device=device)
        self.has_cur_f = torch.tensor(has_cur, dtype=torch.float32, device=device)
        self.T = self.feat.shape[0]
        self.A = self.feat.shape[1]
        self.reset()

    def reset(self):
        dev = self.device
        self.capital = torch.full((self.P,), 1_000_000.0, device=dev)
        self.peak = self.capital.clone()
        self.mdd = torch.zeros(self.P, device=dev)
        self.positions = torch.zeros(self.P, self.A, device=dev)
        self.ema = torch.zeros(self.P, self.A, device=dev)
        self.trades = torch.zeros(self.P, dtype=torch.long, device=dev)
        self.returns = torch.zeros(self.P, self.T - 21, device=dev)
        self.done = torch.zeros(self.P, dtype=torch.bool, device=dev)
        self.done_day = torch.full((self.P,), self.T - 21, dtype=torch.long, device=dev)
        self.t = 20

    def step(self, raw_signals):
        """raw_signals [P,43] → 推进一日; 返回 False 当全员终局"""
        t = self.t
        if t >= self.T - 1:
            return False
        active = ~self.done
        af = active.float()
        # EMA (0.92/0.08, 仅 has_cur 资产; 与 C++ 一致)
        self.ema = torch.where(active.unsqueeze(1) & (self.has_cur_f[t] > 0).unsqueeze(0),
                               0.92 * self.ema + 0.08 * raw_signals, self.ema)
        ranked = torch.argsort(self.ema, dim=1)
        top_idx = ranked[:, -5:]
        bot_idx = ranked[:, :5]
        sig_top = torch.gather(self.ema, 1, top_idx)
        sig_bot = torch.gather(self.ema, 1, bot_idx)
        w_row = self.vol_inv[t].unsqueeze(0).expand(self.P, -1)
        target = torch.zeros_like(self.positions)
        target.scatter_(1, top_idx, torch.where(sig_top > 0.02, torch.gather(w_row, 1, top_idx),
                                                torch.zeros(self.P, 5, device=raw_signals.device)))
        target.scatter_(1, bot_idx, torch.where(sig_bot < -0.02, -torch.gather(w_row, 1, bot_idx),
                                                torch.zeros(self.P, 5, device=raw_signals.device)))
        total_abs = target.abs().sum(dim=1, keepdim=True)
        scale = torch.where(total_abs > 1e-4, 0.80 / total_abs.clamp(min=1e-9), torch.zeros_like(total_abs))
        target = target * scale

        cur = self.positions
        delta = (target - cur).abs()
        traded = (delta > 0.10).float()
        eff_target = torch.where(traded > 0, target, cur)               # C++: 未交易保持旧仓
        cost_rate = (delta * 0.00015 * traded).sum(dim=1)               # C++: capital -= capital*delta*0.00015
        cap_after_cost = self.capital * (1.0 - cost_rate)
        # pnl 用扣费后资金 × 旧仓位 × ret_next (has_next 掩码)
        pnl_assets = cur * self.ret_next[t].unsqueeze(0) * self.has_next_f[t].unsqueeze(0)
        pnl = pnl_assets.sum(dim=1) * cap_after_cost * af
        day_ret = torch.where((cap_after_cost > 0) & active,
                              pnl / cap_after_cost.clamp(min=1e-9), torch.zeros_like(pnl))
        self.returns[:, t - 20] = day_ret
        self.capital = torch.where(active, cap_after_cost + pnl, self.capital)
        self.positions = torch.where(active.unsqueeze(1), eff_target, cur)
        self.trades += (traded.sum(dim=1).long()) * active.long()
        self.peak = torch.maximum(self.peak, self.capital)
        self.mdd = torch.maximum(self.mdd, (1.0 - self.capital / self.peak.clamp(min=1e-4)).clamp(min=0))
        newly_done = active & (self.capital <= 100_000.0)
        self.done_day = torch.where(newly_done, torch.full_like(self.done_day, t - 20 + 1), self.done_day)
        self.done |= newly_done
        self.t += 1
        return bool(active.any())

    def metrics(self):
        """向量化 sharpe/cum/mdd/trades → fitness (fitness_from_task 逐式对齐)"""
        r = self.returns
        L = torch.where(self.done, self.done_day,
                        torch.full_like(self.done_day, r.shape[1])).clamp(min=1, max=r.shape[1])
        idx = (L - 1).unsqueeze(1)
        cs = r.cumsum(dim=1)
        cs2 = (r * r).cumsum(dim=1)
        mean = cs.gather(1, idx).squeeze(1) / L.float()
        var = (cs2.gather(1, idx).squeeze(1) / L.float()) - mean * mean
        std = var.clamp(min=0).sqrt()
        sharpe = torch.where(std > 1e-7, mean / std * math.sqrt(252.0), torch.zeros_like(mean))
        sharpe = torch.where(L >= 20, sharpe, torch.full_like(sharpe, -1.0))
        cum = (self.capital - 1_000_000.0) / 1_000_000.0
        fitness = torch.where(
            sharpe > 0.0,
            sharpe * 2.5 + cum * 0.5 - self.mdd * 3.0 - self.trades.float() / 50000.0,
            sharpe * 2.0 - self.mdd * 4.0 + torch.where(cum < 0, cum * 0.5, torch.zeros_like(cum)))
        fitness = torch.where(self.trades >= 40, fitness, torch.full_like(fitness, -10.0))
        return fitness, sharpe, cum, self.mdd, self.trades


def run_episode(pop, sim):
    sim.reset()
    pop.reset_states()
    x = torch.zeros(pop.pop_size, NUM_COLS * IN_DIM, device=pop.device)
    while True:
        d = sim.t
        if d >= sim.T - 1:
            break
        # 同日全成员输入一致 (特征只依赖日期); 逐列展开 [P, 43*4]
        x.copy_(sim.feat[d].reshape(1, -1).expand(pop.pop_size, -1))
        pop.forward_step(x)
        if not sim.step(pop.effector_signals()):
            break
    return sim.metrics()


def main():
    generations = int(sys.argv[sys.argv.index("--gen") + 1]) if "--gen" in sys.argv else GENERATIONS
    pop_size = int(sys.argv[sys.argv.index("--pop") + 1]) if "--pop" in sys.argv else POP_SIZE

    print("=" * 75)
    print("🧬 SDSCC 量化种群生态 GPU 战役 (Stage 1: 逐列注入新世系, C11 对账未达成·如实)")
    print(f"   43 资产微柱 × 24 细胞 (逐列注入 in4) | 种群 {pop_size} | 代数 {generations}")
    print("=" * 75)

    series, train_d, val_d, test_d = load_series()
    print(f"  ↳ 演化集 {len(train_d)} / 选择集 {len(val_d)} / 盲测集 {len(test_d)} 交易日")
    tr = precompute(series, train_d)
    va = precompute(series, val_d)
    te = precompute(series, test_d)

    pop = QuantArrayPopulation(pop_size=pop_size, device="cuda")
    sim_tr = BatchedPortfolioSim(pop_size, *tr, device=pop.device)

    t0 = time.perf_counter()
    best_fit, best_idx = -float("inf"), 0
    repop_events = 0
    for gen in range(1, generations + 1):
        fitness, sharpe, cum, mdd, trades = run_episode(pop, sim_tr)
        cur_best = fitness.max().item()
        if cur_best > best_fit:
            best_fit = cur_best
            best_idx = int(torch.argmax(fitness).item())
        # GPU 列级有性重组 (v2) + 变异
        num_elites = pop.crossover_population(fitness, elite_ratio=ELITE_RATIO,
                                              tournament_k=4, crossover_prob=CROSSOVER_PROB)
        mut_scale = max(0.015, 0.08 * (1.0 - gen / generations))
        pop.mutate(mutation_rate=MUT_RATE, mutation_power=mut_scale, num_elites=num_elites)
        # 多样性哨兵 (v2): 每 10 代监控相关矩阵, 坍缩即重繁
        if gen % WATCHDOG_EVERY == 0:
            Rg = sim_tr.returns
            Z = (Rg - Rg.mean(dim=1, keepdim=True)) / Rg.std(dim=1, keepdim=True).clamp(min=1e-9)
            Cmat = (Z @ Z.T) / max(1, Rg.shape[1])
            off_mask = ~torch.eye(pop_size, dtype=torch.bool, device=pop.device)
            frac_high = ((Cmat.abs() > CORR_GATE) & off_mask).float().sum() / off_mask.sum()
            if frac_high.item() > WATCHDOG_CORR_FRAC:
                n = pop.repopulate_random(fitness, elite_ratio=ELITE_RATIO, frac=0.5)
                repop_events += 1
        if gen % 20 == 0 or gen == 1:
            print(f"  [Gen {gen:3d}/{generations}] 最佳fitness: {cur_best:7.3f} | "
                  f"夏普均值: {sharpe.mean().item():5.2f} | 盈利成员: {(cum > 0).float().mean().item()*100:.1f}%"
                  + (f" | 重繁: {repop_events}" if repop_events else ""))
    print(f"  [✓] GPU 演化完毕: {time.perf_counter()-t0:.1f}s (种群 {pop_size} × {generations} 代 | 重繁 {repop_events} 次)")

    # 种群快照落盘 (选型阶段崩溃可续跑, 免整场重演)
    torch.save({"intra": pop.intra_weights, "inter": pop.inter_weights,
                "param1": pop.param1, "param2": pop.param2},
               "/tmp/opencode/quant_gpu_pop_state.pt")

    # ── val 选型 (预注册纪律同 v2) ──
    sim_va = BatchedPortfolioSim(pop_size, *va, device=pop.device)
    fit_va, sh_va, cum_va, mdd_va, _ = run_episode(pop, sim_va)
    eligible = torch.where(sh_va > 0)[0].tolist()
    print(f"\n  ↳ 诚实条款: val 夏普>0 合格成员 {len(eligible)}/{pop_size}")
    R = sim_va.returns.cpu().numpy()
    order = sorted(eligible, key=lambda i: -fit_va[i].item())
    selected, rejected = [], 0
    for i in order:
        if len(selected) >= POOL_K:
            break
        if any(abs(float(np.corrcoef(R[i], R[j])[0, 1])) > CORR_GATE for j in selected):
            rejected += 1
            continue
        selected.append(i)
    print(f"  ↳ 相关性门: 拒绝 {rejected} 个克隆候选")
    if len(selected) < 2:
        print("  [FAIL] 相关性门后 <2 成员, 战役失败")
        return 1

    # 池融合 val 回放 (融合信号 = 入池成员 raw 等权平均)
    class _FusedPop:
        def __init__(self):
            self.device = pop.device
            self.pop_size = 1
            self.outputs = None
        def reset_states(self):
            pop.reset_states()
        def forward_step(self, x):
            if x.shape[0] != pop.pop_size:
                x = x.expand(pop.pop_size, -1)   # 单行输入广播至全种群 (融合语义只用入池成员行)
            pop.forward_step(x)
            eff = pop.outputs.view(pop.pop_size, NUM_COLS, CELLS_PER_COL)[:, :, -2:]
            fused = (eff[:, :, 0] - eff[:, :, 1])[selected].mean(dim=0)
            self.outputs = torch.zeros(1, pop.num_cells, device=pop.device)
            self.outputs.view(1, NUM_COLS, CELLS_PER_COL)[:, :, -2] = fused.clamp(min=0)
            self.outputs.view(1, NUM_COLS, CELLS_PER_COL)[:, :, -1] = (-fused).clamp(min=0)
            return self.outputs[:, pop.num_cells - pop.out_dim:]

    sim_fv = BatchedPortfolioSim(1, *va, device=pop.device)
    _, sh_fv, cum_fv, mdd_fv, _ = run_episode(_FusedPop(), sim_fv)
    print(f"  ↳ [池·融合] val 夏普: {sh_fv[0].item():.2f} | 收益: {cum_fv[0].item()*100:.1f}% | 回撤: {mdd_fv[0].item()*100:.1f}%")
    if mdd_fv[0].item() > 0.15:
        print("  [FAIL] val 池回撤 >15% 诚实阈值, 战役失败")
        return 1

    # ── OOS 一次性盲报 + 全量披露 + 导出 ──
    sim_te = BatchedPortfolioSim(pop_size, *te, device=pop.device)
    _, sh_te, cum_te, mdd_te, _ = run_episode(pop, sim_te)
    sim_ft = BatchedPortfolioSim(1, *te, device=pop.device)
    _, sh_ft, cum_ft, mdd_ft, tr_ft = run_episode(_FusedPop(), sim_ft)
    print("\n" + "=" * 70)
    print("  样本外一次性盲报 (OOS 2016-2026) — 最终池形态, 无任何后验选择")
    print("=" * 70)
    print(f"  ↳ [OOS 池·融合] 夏普: {sh_ft[0].item():.2f} | 收益: {cum_ft[0].item()*100:.1f}% | "
          f"回撤: {mdd_ft[0].item()*100:.1f}% | 换手: {int(tr_ft[0].item())} 次")
    print(f"  ↳ 期末资金: {1e6 * (1.0 + cum_ft[0].item()):,.2f} 元")
    print(f"  ↳ [全量披露] 入池 {len(selected)} 成员个体 OOS:")
    for i in selected:
        print(f"    成员{i} | 夏普: {sh_te[i].item():5.2f} | 收益: {cum_te[i].item()*100:6.1f}% | 回撤: {mdd_te[i].item()*100:4.1f}%")

    os.makedirs("checkpoints", exist_ok=True)
    manifest = {"organism_id": "quant_ecology_gpu",
                "pool_size": len(selected),
                "protocol_note": "GPU 变体新世系 (稠密列内+环柱轴突+逐列注入; 算子池无 CORRELATION/INHIBIT); "
                                 "C11 位级对账未达成 (SDSC-BIN v5 受体映射立项中); "
                                 "val-selected frozen-consts, OOS single-shot",
                "pool_val_sharpe": sh_fv[0].item(),
                "pool_oos_sharpe": sh_ft[0].item(),
                "pool_oos_return": cum_ft[0].item(),
                "pool_oos_mdd": mdd_ft[0].item(),
                "members": []}
    for i in selected:
        path = f"checkpoints/quant_ecology_gpu_m{i}.bin"
        pop.export_champion_to_sdsc_bin(i, path)
        manifest["members"].append({"checkpoint_path": path, "member_idx": i,
                                    "val_sharpe": sh_va[i].item(), "oos_sharpe": sh_te[i].item()})
    with open("checkpoints/quant_ecology_gpu_pool.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n  [SUCCESS] GPU 生态池入库: checkpoints/quant_ecology_gpu_pool.json (+{len(selected)} SDSC-BIN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
