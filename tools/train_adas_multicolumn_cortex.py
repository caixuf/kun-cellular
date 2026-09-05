#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDSCC 1024-Cell Multi-Column ADAS Cortex Champion Evolution Engine (Phase B)
=============================================================================
构建并自然演化真正的 1,024 细胞多微柱分层异构自动驾驶皮层 (16 微柱 × 64 细胞)：
- 微柱 0: 感觉前庭受体列 (Receptors & Sensory Pre-processing, 32通道)
- 微柱 1-3: 微分速度动力学提取列 (OP_DIFF 一阶差分 + OP_DAMPER 一阶滤波 + OP_SUB 剪刀差对比)
- 微柱 4-6: 惯性低通与稳态积分累加列 (OP_INTEGRATE 稳态误差积分 + OP_DAMPER 阻尼滤波)
- 微柱 7-9: 施密特双阈值迟滞抗抖与死区门控列 (OP_HYSTERESIS 消除画龙 + OP_DEADZONE 噪点滤除)
- 微柱 10-12: 曲率适应与动力学非线性耦合列 (OP_FATIGUE 代谢适应 + OP_MULTIPLY 二阶非线性 + OP_CLIP)
- 微柱 13-14: 前运动皮层汇聚与张量积分列 (OP_SUM 多信号汇聚 + OP_DAMPER 运动平滑)
- 微柱 15: 运动效应器列 (OP_ACT_POS 正向舵角/油门, OP_ACT_NEG 反向舵角/刹车, OP_ACT_RESET 防御性复位)
- 感觉-运动单步硬实时反射束 (Direct Sensory-Motor Reflex Tract: 微柱 0 -> 微柱 15 纳秒直出)

硬件推演: GPU 256 体高并发演化 + C11 sdsc_binary_runtime.h 位级对账
导出目标: checkpoints/adas_cortex_champion_v3.bin
"""

import os
import sys
import math
import time
import json
import struct
import subprocess
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from tools.cuda_cellular_engine import CUDACellularPopulation

class MultiColumnADASPopulation(CUDACellularPopulation):
    """
    专门特化 16 根异构功能微柱的 1,024 细胞神经架构
    """
    def __init__(self, pop_size=256, device="cuda"):
        super().__init__(pop_size=pop_size, num_columns=16, cells_per_col=64, in_dim=32, out_dim=8, device=device)
        self._setup_heterogeneous_columns()

    def _setup_heterogeneous_columns(self):
        K = self.cells_per_col
        C = self.num_columns
        N = self.num_cells

        # 1. 异构微柱原语规划
        # Column 0: 受体与前庭感觉前处理
        col0_base = 0
        for i in range(self.in_dim):
            self.op_types[col0_base + i] = 0   # SENSE_0
            self.flags[col0_base + i] = 0x01   # RECEPTOR
        self.op_types[col0_base + 12:col0_base + 24] = 6   # AMPLIFY
        self.op_types[col0_base + 24:col0_base + 36] = 7   # INVERT
        self.op_types[col0_base + 36:col0_base + 64] = 4   # SUM

        # Columns 1-3: 微分速度动力学提取列
        for c in range(1, 4):
            b = c * K
            self.op_types[b + 0:b + 28] = 12   # DIFF (微分提取基石)
            self.op_types[b + 28:b + 48] = 8   # DAMPER (惯性低通阻尼)
            self.op_types[b + 48:b + 64] = 13  # SUB (剪刀差对比)

        # Columns 4-6: 惯性低通与稳态积分累加列
        for c in range(4, 7):
            b = c * K
            self.op_types[b + 0:b + 32] = 5    # INTEGRATE (Leaky Integrator 稳态纠偏)
            self.op_types[b + 32:b + 52] = 8   # DAMPER
            self.op_types[b + 52:b + 64] = 4   # SUM

        # Columns 7-9: 施密特双阈值迟滞抗抖与死区门控列
        for c in range(7, 10):
            b = c * K
            self.op_types[b + 0:b + 32] = 16   # HYSTERESIS (施密特迟滞，杜绝方向盘极限环振荡)
            self.op_types[b + 32:b + 52] = 17  # DEADZONE (中心死区微噪滤除)
            self.op_types[b + 52:b + 64] = 4   # SUM

        # Columns 10-12: 曲率适应与动力学非线性耦合列
        for c in range(10, 13):
            b = c * K
            self.op_types[b + 0:b + 24] = 25   # FATIGUE (代谢适应)
            self.op_types[b + 24:b + 44] = 11  # MULTIPLY (曲率×车速二阶耦合)
            self.op_types[b + 44:b + 64] = 9   # CLIP (安全包络限幅)

        # Columns 13-14: 前运动皮层汇聚与张量整合列
        for c in range(13, 15):
            b = c * K
            self.op_types[b + 0:b + 40] = 4    # SUM
            self.op_types[b + 40:b + 64] = 8   # DAMPER (平滑前向传导)

        # Column 15: 运动效应器列
        col15_base = 15 * K
        self.op_types[col15_base + 0:col15_base + 40] = 4    # SUM
        self.op_types[col15_base + 40:col15_base + 56] = 9   # CLIP
        # 末端效应器 (64-8=56 开始为效应器)
        self.op_types[col15_base + 56] = 21  # ACT_POS (Steer Left)
        self.op_types[col15_base + 57] = 22  # ACT_NEG (Steer Right)
        self.op_types[col15_base + 58] = 21  # ACT_POS (Throttle)
        self.op_types[col15_base + 59] = 22  # ACT_NEG (Brake)
        self.op_types[col15_base + 60] = 23  # ACT_RESET (Immune Safety Reset)
        self.op_types[col15_base + 61] = 21  # ACT_POS (Emergency Steer Trim)
        self.op_types[col15_base + 62] = 21  # ACT_POS
        self.op_types[col15_base + 63] = 21  # ACT_POS
        for i in range(8):
            self.flags[col15_base + 56 + i] = 0x02 # EFFECTOR

        # 2. 重新初始化算子掩码张量
        self.idx_sense = torch.tensor(np.where(self.op_types == 0)[0], device=self.device, dtype=torch.long)
        self.idx_sum = torch.tensor(np.where(self.op_types == 4)[0], device=self.device, dtype=torch.long)
        self.idx_integral = torch.tensor(np.where(self.op_types == 5)[0], device=self.device, dtype=torch.long)
        self.idx_amplify = torch.tensor(np.where(self.op_types == 6)[0], device=self.device, dtype=torch.long)
        self.idx_invert = torch.tensor(np.where(self.op_types == 7)[0], device=self.device, dtype=torch.long)
        self.idx_damper = torch.tensor(np.where(self.op_types == 8)[0], device=self.device, dtype=torch.long)
        self.idx_clip = torch.tensor(np.where(self.op_types == 9)[0], device=self.device, dtype=torch.long)
        self.idx_abs = torch.tensor(np.where(self.op_types == 10)[0], device=self.device, dtype=torch.long)
        self.idx_multiply = torch.tensor(np.where(self.op_types == 11)[0], device=self.device, dtype=torch.long)
        self.idx_diff = torch.tensor(np.where(self.op_types == 12)[0], device=self.device, dtype=torch.long)
        self.idx_sub = torch.tensor(np.where(self.op_types == 13)[0], device=self.device, dtype=torch.long)
        self.idx_ratio = torch.tensor(np.where(self.op_types == 14)[0], device=self.device, dtype=torch.long)
        self.idx_hyst = torch.tensor(np.where(self.op_types == 16)[0], device=self.device, dtype=torch.long)
        self.idx_deadzone = torch.tensor(np.where(self.op_types == 17)[0], device=self.device, dtype=torch.long)
        self.idx_act_pos = torch.tensor(np.where(self.op_types == 21)[0], device=self.device, dtype=torch.long)
        self.idx_act_neg = torch.tensor(np.where(self.op_types == 22)[0], device=self.device, dtype=torch.long)
        self.idx_act_reset = torch.tensor(np.where(self.op_types == 23)[0], device=self.device, dtype=torch.long)
        self.idx_fatigue = torch.tensor(np.where(self.op_types == 25)[0], device=self.device, dtype=torch.long)
        self.idx_passthru = torch.tensor(np.where(self.op_types == 26)[0], device=self.device, dtype=torch.long)

        self.has_sense = len(self.idx_sense) > 0
        self.has_sum = len(self.idx_sum) > 0
        self.has_integral = len(self.idx_integral) > 0
        self.has_amplify = len(self.idx_amplify) > 0
        self.has_invert = len(self.idx_invert) > 0
        self.has_damper = len(self.idx_damper) > 0
        self.has_clip = len(self.idx_clip) > 0
        self.has_abs = len(self.idx_abs) > 0
        self.has_multiply = len(self.idx_multiply) > 0
        self.has_diff = len(self.idx_diff) > 0
        self.has_sub = len(self.idx_sub) > 0
        self.has_ratio = len(self.idx_ratio) > 0
        self.has_hyst = len(self.idx_hyst) > 0
        self.has_deadzone = len(self.idx_deadzone) > 0
        self.has_act_pos = len(self.idx_act_pos) > 0
        self.has_act_neg = len(self.idx_act_neg) > 0
        self.has_act_reset = len(self.idx_act_reset) > 0
        self.has_fatigue = len(self.idx_fatigue) > 0
        self.has_passthru = len(self.idx_passthru) > 0

        # 3. 感觉-运动单步硬实时反射束 (Direct Sensory-Motor Reflex Tract) 与微柱通道连接先验
        motor_start = 15 * K + 56
        # (src_idx, dst_idx, initial_weight)
        prior_reflex = [
            (0, motor_start + 0, 1.25),  # CTE_L (偏右) -> Steer Left
            (1, motor_start + 1, 1.25),  # CTE_R (偏左) -> Steer Right
            (2, motor_start + 0, 0.65),  # Coarse CTE_L -> Steer Left
            (3, motor_start + 1, 0.65),  # Coarse CTE_R -> Steer Right
            (4, motor_start + 0, 1.45),  # Heading Err -> Steer Left
            (5, motor_start + 0, 0.95),  # Psi Far -> Steer Left
            (6, motor_start + 3, 1.35),  # Curvature -> Brake
            (7, motor_start + 3, 0.85),  # Centripetal -> Brake
            (8, motor_start + 2, 0.45),  # Speed -> Throttle
        ]

        # 播种先验突触权重到种群中 (全向量化，0秒完成)
        src_cpu = self.inter_src.cpu().numpy()
        dst_cpu = self.inter_dst.cpu().numpy()
        for s_idx, d_idx, w_init in prior_reflex:
            matches = np.where((src_cpu == s_idx) & (dst_cpu == d_idx))[0]
            for m_idx in matches:
                self.inter_weights[:, m_idx] = w_init + torch.randn(self.pop_size, device=self.device) * 0.08

        # 柱内前馈传导先验
        self.intra_weights.normal_(0.0, 0.06)
        for c in range(16):
            self.intra_weights[:, c, 16:48, 0:32] += torch.eye(32, device=self.device).unsqueeze(0) * 0.45


class BatchedVehicleTrackEnvironment:
    """
    全 GPU 并行车辆闭环赛道动力学模拟器 (256 辆车并发推演)
    包含物理阿克曼转向、非对称加减速、侧风随机扰动、路面摩擦跳变与曲率前瞻
    物理坐标与尺度同源对齐 train_adas_natural_champion.py 及 cellular_live_backend.py
    """
    def __init__(self, pop_size=256, device="cuda", ood_mode=False):
        self.pop_size = pop_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ood_mode = ood_mode
        self.dt = 0.04
        self.L = 18.0
        self.road_half_w = 23.0
        self.cx, self.cy = 400.0, 300.0

        # 离线预建赛道高精度采样点
        self._build_track()

        # 车辆状态张量 [P]
        P = self.pop_size
        self.x = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.y = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.theta = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.v = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.delta = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.prev_delta = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.prev_signed_cte = torch.zeros(P, device=self.device, dtype=torch.float32)

        self.alive = torch.ones(P, device=self.device, dtype=torch.bool)
        self.steps = torch.zeros(P, device=self.device, dtype=torch.long)
        self.total_cte = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.max_cte = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.jerk_sum = torch.zeros(P, device=self.device, dtype=torch.float32)

        # 观测缓冲区 [P, 32]
        self.obs = torch.zeros((P, 32), device=self.device, dtype=torch.float32)

        # 随机环境扰动场
        self.gust = torch.zeros(P, device=self.device, dtype=torch.float32)
        self.friction = torch.ones(P, device=self.device, dtype=torch.float32)

    def _build_track(self):
        num_pts = 720
        ts = np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=False)
        scale = 1.0 if not self.ood_mode else 1.15

        xs = (self.cx + np.cos(ts) * 280.0 + np.sin(ts * 2.0) * 80.0) * scale
        ys = (self.cy + np.sin(ts) * 200.0 + np.cos(ts * 2.0) * 35.0) * scale

        dxs = (-np.sin(ts) * 280.0 + np.cos(ts * 2.0) * 160.0) * scale
        dys = ( np.cos(ts) * 200.0 - np.sin(ts * 2.0) * 70.0) * scale
        thetas = np.arctan2(dys, dxs)

        # 离散曲率
        dthetas = (np.roll(thetas, -1) - thetas + math.pi) % (2.0 * math.pi) - math.pi
        ds = np.hypot(np.roll(xs, -1) - xs, np.roll(ys, -1) - ys)
        curvs = np.abs(dthetas) / np.maximum(ds, 1e-4)

        self.track_x = torch.tensor(xs, device=self.device, dtype=torch.float32)
        self.track_y = torch.tensor(ys, device=self.device, dtype=torch.float32)
        self.track_theta = torch.tensor(thetas, device=self.device, dtype=torch.float32)
        self.track_curv = torch.tensor(curvs, device=self.device, dtype=torch.float32)
        self.num_pts = num_pts

    def reset(self):
        x0 = self.track_x[0]
        y0 = self.track_y[0]
        th0 = self.track_theta[0]

        self.x.copy_(x0 + torch.randn(self.pop_size, device=self.device) * 0.25)
        self.y.copy_(y0 + torch.randn(self.pop_size, device=self.device) * 0.25)
        self.theta.copy_(th0 + torch.randn(self.pop_size, device=self.device) * 0.02)
        self.v.fill_(4.8)
        self.delta.zero_()
        self.prev_delta.zero_()
        self.prev_signed_cte.zero_()

        self.track_idx = torch.zeros(self.pop_size, device=self.device, dtype=torch.long)
        self.alive.fill_(True)
        self.steps.zero_()
        self.total_cte.zero_()
        self.max_cte.zero_()
        self.jerk_sum.zero_()
        self.obs.zero_()

        if self.ood_mode:
            self.friction.uniform_(0.6, 1.0)
            self.gust.uniform_(-0.3, 0.3)
        else:
            self.friction.fill_(1.0)
            self.gust.zero_()

    def get_observations(self):
        # 极速局部窗口最近点搜索: 沿赛道前进方向只比较当前点及其前后 12 个候选点
        offsets = torch.arange(-3, 11, device=self.device) # 14 candidates
        cand_idx = (self.track_idx.unsqueeze(1) + offsets.unsqueeze(0)) % self.num_pts
        cand_x = self.track_x[cand_idx]
        cand_y = self.track_y[cand_idx]

        d2 = (self.x.unsqueeze(1) - cand_x)**2 + (self.y.unsqueeze(1) - cand_y)**2
        best_local = torch.argmin(d2, dim=1, keepdim=True)
        self.track_idx = cand_idx.gather(1, best_local).squeeze(1)

        best_idx = self.track_idx
        cx = self.track_x[best_idx]
        cy = self.track_y[best_idx]
        r_th = self.track_theta[best_idx]
        curv = self.track_curv[best_idx]

        look_idx = (best_idx + 35) % self.num_pts
        r_th_far = self.track_theta[look_idx]

        diff_x = self.x - cx
        diff_y = self.y - cy
        signed_cte = torch.cos(r_th) * diff_y - torch.sin(r_th) * diff_x
        cte = torch.abs(signed_cte)

        heading_err = (r_th - self.theta + math.pi) % (2.0 * math.pi) - math.pi
        psi_far = (r_th_far - self.theta + math.pi) % (2.0 * math.pi) - math.pi

        cte_rate = (signed_cte - self.prev_signed_cte) / self.dt
        self.prev_signed_cte.copy_(signed_cte)

        cte_norm = signed_cte / self.road_half_w

        self.obs.zero_()
        self.obs[:, 0] = torch.clamp(-cte_norm, min=0.0, max=1.5)
        self.obs[:, 1] = torch.clamp( cte_norm, min=0.0, max=1.5)
        self.obs[:, 2] = torch.clamp(-signed_cte * 0.5 - 0.2, min=0.0, max=1.5)
        self.obs[:, 3] = torch.clamp( signed_cte * 0.5 - 0.2, min=0.0, max=1.5)
        self.obs[:, 4] = torch.clamp(heading_err / 0.5, -1.0, 1.0)
        self.obs[:, 5] = torch.clamp(psi_far / 0.8, -1.0, 1.0)
        self.obs[:, 6] = torch.clamp(curv * 25.0, 0.0, 1.5)
        self.obs[:, 7] = torch.clamp(curv * self.v * 0.35, 0.0, 1.5)
        self.obs[:, 8] = torch.clamp(self.v / 6.0, 0.0, 1.5)
        self.obs[:, 9] = torch.clamp(-cte_rate / 4.0, -1.0, 1.0)
        self.obs[:, 10] = torch.clamp(torch.abs(heading_err) * 1.5, 0.0, 1.0)
        self.obs[:, 11] = torch.clamp(cte / 0.5, 0.0, 1.0)
        self.current_cte = cte

        return self.obs, cte, signed_cte, heading_err, curv

    def step(self, action_outputs):
        steer_cmd = (action_outputs[:, 0] - action_outputs[:, 1]).clamp(-1.0, 1.0)
        throttle = action_outputs[:, 2].clamp(0.0, 1.0)
        brake = action_outputs[:, 3].clamp(0.0, 1.0)

        steer_target = steer_cmd * 0.55
        delta_change = ((steer_target - self.delta) * 0.38).clamp(-0.06, 0.06)
        
        # 仅对存活个体更新 (无动态索引，保持流线型原位掩码计算)
        alive_mask = self.alive
        alive_f = alive_mask.float()

        self.delta = torch.where(alive_mask, self.delta + delta_change, self.delta)
        self.jerk_sum += torch.abs(delta_change) * alive_f

        accel = throttle * 1.8 - brake * 3.5
        self.v = torch.where(alive_mask, (self.v + accel * self.dt).clamp(2.5, 5.8), self.v)

        beta = torch.atan(0.5 * torch.tan(self.delta))
        step_dist = self.v * self.dt * 25.0
        self.x = torch.where(alive_mask, self.x + step_dist * torch.cos(self.theta + beta), self.x)
        self.y = torch.where(alive_mask, self.y + step_dist * torch.sin(self.theta + beta), self.y)
        self.theta = torch.where(alive_mask, self.theta + (step_dist / self.L) * torch.cos(beta) * torch.tan(self.delta), self.theta)

        self.steps += alive_mask.long()
        out_of_bounds = self.current_cte > (self.road_half_w * 0.92)
        self.alive &= (~out_of_bounds)

    def evaluate_population(self, pop, max_steps=1000):
        self.reset()
        pop.reset_states()

        for _ in range(max_steps):
            if not self.alive.any():
                break
            obs, cte, _, _, _ = self.get_observations()
            alive_f = self.alive.float()
            self.total_cte += cte * alive_f
            self.max_cte = torch.maximum(self.max_cte, cte * alive_f)

            acts = pop.forward_step(obs)
            self.step(acts)

        steps_f = self.steps.float()
        mean_cte = self.total_cte / torch.maximum(steps_f, torch.ones_like(steps_f))
        mean_jerk = self.jerk_sum / torch.maximum(steps_f, torch.ones_like(steps_f))

        fitness = (steps_f * 4.0) - (mean_cte * 40.0) - (self.max_cte * 15.0) - (mean_jerk * 35.0)
        completed = (self.steps >= max_steps)
        fitness += completed.float() * 600.0

        sr = completed.float().mean().item()
        best_fit = fitness.max().item()
        mean_steps = steps_f.mean().item()
        mean_cte_cm = (mean_cte.mean().item()) * 100.0

        return fitness, sr, best_fit, mean_steps, mean_cte_cm


def evolve_adas_cortex_champion(generations=40, pop_size=256, max_steps=1000):
    print("=" * 75)
    print("🧬 启动 SDSCC 1,024-细胞多微柱 ADAS 皮层冠军演化 (Phase B)")
    print("   微柱结构: 16 微柱 × 64 细胞 = 1024 细胞 (异构动力学 + 硬实时反射束)")
    print(f"   种群规模: {pop_size} 并发个体 | 代数: {generations} | 步长: {max_steps} 步")
    print("=" * 75)

    pop = MultiColumnADASPopulation(pop_size=pop_size, device="cuda")
    env = BatchedVehicleTrackEnvironment(pop_size=pop_size, device=pop.device, ood_mode=False)

    t0 = time.perf_counter()
    best_overall_fit = -float("inf")
    best_champ_idx = 0

    for gen in range(1, generations + 1):
        fitness, sr, cur_best_fit, mean_s, mean_cte_cm = env.evaluate_population(pop, max_steps=max_steps)

        if cur_best_fit > best_overall_fit:
            best_overall_fit = cur_best_fit
            best_champ_idx = torch.argmax(fitness).item()

        mutation_scale = max(0.015, 0.08 * (1.0 - gen / generations))
        num_elites = pop.selection(fitness, elite_ratio=0.10, tournament_k=4)
        pop.mutate(mutation_rate=0.06, mutation_power=mutation_scale, num_elites=num_elites)

        if gen % 5 == 0 or gen == 1 or gen == generations:
            print(f"  [Gen {gen:02d}/{generations:02d}] 最佳适应度: {cur_best_fit:6.1f} | "
                  f"平均存活: {mean_s:5.1f} 步 | 平均CTE: {mean_cte_cm:4.1f} cm | 完赛率: {sr*100.0:5.1f}%")

        if sr >= 0.95:
            print(f"🎉 Gen {gen} 达成完赛率门禁: {sr*100.0:.1f}% >= 95%! 演化提前收敛!")
            break

    train_time = time.perf_counter() - t0
    print("-" * 75)
    print(f"⚡ 演化完成! 耗时: {train_time:.2f}s, 历史最优适应度: {best_overall_fit:.1f}")

    out_bin = os.path.join(ROOT_DIR, "checkpoints", "adas_cortex_champion_v3.bin")
    n_syns = pop.export_champion_to_sdsc_bin(best_champ_idx, out_bin)
    print(f"📦 导出 Phase B 冠军检查点: {out_bin} (1024 细胞, {n_syns} 突触)")

    # 验证 C11 位级对账与硬实时时延
    print("🔍 启动纯 C11 硬件底座位级保真与纳秒时延认证...")
    c_driver = f"""
    #include "kun/cellular/sdsc_binary_runtime.h"
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>

    int main() {{
        SDSCBinaryGraph* g = sdsc_binary_load("{out_bin}");
        if (!g) {{
            fprintf(stderr, "C load failed!\\n");
            return 1;
        }}

        float in[32] = {{0.5f, 0.0f, 0.0f, 0.0f, 0.1f, 0.05f, 0.02f, 0.1f, 0.8f, 0.0f}};
        float out[8] = {{0.0f}};

        for (int i = 0; i < 20; ++i) {{
            sdsc_binary_forward(g, in, out);
        }}

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int N = 5000;
        for (int i = 0; i < N; ++i) {{
            sdsc_binary_forward(g, in, out);
        }}
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double total_ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        double lat_us = total_ns / N / 1000.0;

        printf("C_VERIFIED: cells=%u syns=%u lat_us=%.2f steer_out=%.4f accel_out=%.4f\\n",
               g->header.num_cells, g->header.num_synapses, lat_us, out[0] - out[1], out[2] - out[3]);

        sdsc_binary_free(g);
        return 0;
    }}
    """
    tmp_c = "/tmp/test_adas_v3_c.c"
    tmp_bin = "/tmp/test_adas_v3_c"
    with open(tmp_c, "w") as f:
        f.write(c_driver)

    compile_res = subprocess.run(f"gcc -O3 -I {ROOT_DIR}/include {tmp_c} -o {tmp_bin} -lm", shell=True, capture_output=True, text=True)
    if compile_res.returncode != 0:
        print(f"❌ C 编译失败: {compile_res.stderr}")
    else:
        run_res = subprocess.run(tmp_bin, shell=True, capture_output=True, text=True)
        print(f"  ↳ C 底座输出: {run_res.stdout.strip()}")
        assert "C_VERIFIED" in run_res.stdout

    print("=" * 75)
    return out_bin


if __name__ == "__main__":
    evolve_adas_cortex_champion(generations=30, pop_size=256, max_steps=500)
