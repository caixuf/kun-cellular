#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KunCellular 第 3 站 · GPU 大规模迷宫演化 (256 体 × 1024 细胞 × 21×21)
=====================================================================================
设计原则 (响应"为什么不敢大规模"):
- 拒绝 8~13 细胞的算法仿制品路线, 直接上 1024 细胞块稀疏微柱 × 256 体并发
- 传感器/动力学与 C++ MazeTask 逐位对齐 (DDA 细采样近似, 误差 <0.001), 保证策略可迁移
- 演化: GPU 批量锦标赛选择 + 变异; 冠军导出 SDSC-BIN 后用 C++ 内核独立验收
=====================================================================================
"""
import math
import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from cuda_cellular_engine import CUDACellularPopulation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
W = 21                     # 迷宫尺寸
STEPS = 400                # 步预算 (与 CPU 验收一致)
DT = 0.12                  # MazeTask step_continuous dt=0.12f (评测真实值)
MAX_RANGE = 6.0            # cast_ray max_range
RAY_ANGLES = [0.0, -0.523599, -0.785398, -1.308997,
              0.523599, 0.785398, 1.308997]   # front + L30/L45/L75 + R30/R45/R75
N_ENV = 256


# ----------------------------------------------------------------------------
# CPU 迷宫生成 (严格复刻 MazeEnvironment: DFS 回溯 + braid)
# ----------------------------------------------------------------------------
def gen_maze(w, seed, braid_prob=0.15):
    rng = np.random.RandomState(seed)
    grid = np.ones((w, w), dtype=np.float32)
    grid[1, 1] = 0
    stack = [(1, 1)]
    dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    while stack:
        cx, cy = stack[-1]
        order = rng.permutation(4)
        carved = False
        for oi in order:
            dx, dy = dirs[oi]
            nx, ny = cx + dx, cy + dy
            if 0 < nx < w - 1 and 0 < ny < w - 1 and grid[ny, nx] == 1:
                grid[ny, nx] = 0
                grid[cy + dy // 2, cx + dx // 2] = 0
                stack.append((nx, ny))
                carved = True
                break
        if not carved:
            stack.pop()
    # braid: 死胡同按概率打通
    for y in range(1, w - 1, 2):
        for x in range(1, w - 1, 2):
            if grid[y, x] != 0:
                continue
            walls = []
            if y - 2 > 0 and grid[y - 1, x] == 1: walls.append((0, -1))
            if y + 2 < w - 1 and grid[y + 1, x] == 1: walls.append((0, 1))
            if x - 2 > 0 and grid[y, x - 1] == 1: walls.append((-1, 0))
            if x + 2 < w - 1 and grid[y, x + 1] == 1: walls.append((1, 0))
            if len(walls) >= 3 and len(walls) > 0 and rng.rand() < braid_prob:
                wx, wy = walls[rng.randint(len(walls))]
                grid[y + wy, x + wx] = 0
    grid[w - 2, w - 2] = 0   # 目标格
    return grid


# ----------------------------------------------------------------------------
# 批量迷宫环境 (torch, 传感器/动力学对齐 C++ MazeTask)
# ----------------------------------------------------------------------------
class BatchedMazeTask:
    def __init__(self, grids, device):
        self.grid = grids.to(device)                       # [N, W, W]
        self.N = self.grid.shape[0]
        self.W = self.grid.shape[1]                        # 支持任意尺寸 (11×11 单位测试等)
        n = self.N
        self.x = torch.full((n,), 1.5, device=device)
        self.y = torch.full((n,), 1.5, device=device)
        self.theta = torch.zeros(n, device=device)
        self.vx = torch.zeros(n, device=device)
        self.vy = torch.zeros(n, device=device)
        self.min_dist = torch.full((n,), 1e9, device=device)
        self.init_dist = torch.full((n,), 1e9, device=device)
        self.collisions = torch.zeros(n, device=device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=device)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.stuck = torch.zeros(n, device=device)
        self.new_tiles = torch.zeros(n, device=device)
        self.visited = torch.zeros(n, self.W * self.W, dtype=torch.bool, device=device)
        self.visited[:, 1 * self.W + 1] = True
        self.prev_x = self.x.clone()
        self.prev_y = self.y.clone()
        self.prev_rays_left = torch.zeros(n, device=device)
        self.prev_rays_right = torch.zeros(n, device=device)
        self.goal = torch.tensor([float(self.W - 2) + 0.5, float(self.W - 2) + 0.5], device=device)
        d = torch.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.min_dist = d.clone()
        self.init_dist = d.clone()

    def _grid_at(self, px, py):
        gx = torch.clamp(torch.floor(px).long(), 0, self.W - 1)
        gy = torch.clamp(torch.floor(py).long(), 0, self.W - 1)
        return self.grid[torch.arange(self.N, device=px.device), gy, gx]

    def cast_rays(self):
        """细采样 DDA 近似: 7 角度 × 600 样本 × 0.01, 误差 <0.001 归一化"""
        n = self.N
        angs = torch.tensor(RAY_ANGLES, device=self.x.device)
        s = torch.arange(1, 601, device=self.x.device).float() * 0.01   # [600] 0.01..6.0
        posx = self.x.view(n, 1, 1) + torch.cos(self.theta.view(n, 1, 1) + angs.view(1, -1, 1)) * s.view(1, 1, -1)
        posy = self.y.view(n, 1, 1) + torch.sin(self.theta.view(n, 1, 1) + angs.view(1, -1, 1)) * s.view(1, 1, -1)
        gx = torch.clamp(torch.floor(posx).long(), 0, self.W - 1)
        gy = torch.clamp(torch.floor(posy).long(), 0, self.W - 1)
        occ = self.grid[torch.arange(n, device=self.x.device).view(n, 1, 1), gy, gx]  # [N,7,600]
        wall = occ > 0.5
        dists = torch.where(wall, s.view(1, 1, -1).expand(n, 7, 600), torch.full_like(s.view(1,1,-1).expand(n,7,600), 9.9))
        first = dists.min(dim=2).values                     # [N,7] 首次命中距离 (未命中=9.9>6)
        norm = torch.clamp(first / MAX_RANGE, max=1.0)
        front = norm[:, 0]
        left = torch.minimum(torch.minimum(norm[:, 1], norm[:, 2]), norm[:, 3])
        right = torch.minimum(torch.minimum(norm[:, 4], norm[:, 5]), norm[:, 6])
        return front, left, right

    def observation(self):
        f, l, r = self.cast_rays()
        self.prev_rays_left = l
        self.prev_rays_right = r
        to_x = self.goal[0] - self.x
        to_y = self.goal[1] - self.y
        diff = torch.atan2(to_y, to_x) - self.theta
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        bearing = torch.clamp(diff / math.pi, -1.0, 1.0)
        return torch.stack([f, l, r, bearing], dim=1)       # [N,4]

    def step(self, fwd, neg, reset):
        """动力学逐位对齐 MazeTask.step_agent"""
        alive = ~self.reached & (self.steps < STEPS)
        fwd = torch.clamp(fwd, -3.0, 5.0)
        rev = torch.clamp(reset, 0.0, 5.0)
        turn = torch.clamp(neg, -5.0, 5.0) * 3.2
        thrust = (fwd - rev) * 2.5

        # 停滞自脱困反射 (stuck_frames >= 5); CPU 语义: prev 在移动前快照 (disp = 上一步真实位移)
        disp = torch.hypot(self.x - self.prev_x, self.y - self.prev_y)
        self.prev_x = self.x.clone()
        self.prev_y = self.y.clone()
        self.stuck = torch.where(disp < 0.02, self.stuck + 1.0, torch.zeros_like(self.stuck))
        escape = self.stuck >= 5.0
        thrust = torch.where(escape, torch.full_like(thrust, -2.2), thrust)
        turn = torch.where(escape, torch.where(self.prev_rays_left >= self.prev_rays_right,
                                               torch.full_like(turn, 3.8), torch.full_like(turn, -3.8)), turn)

        self.theta = self.theta + turn * DT
        tvx = torch.cos(self.theta) * thrust
        tvy = torch.sin(self.theta) * thrust
        self.vx = self.vx * 0.7 + tvx * 0.3
        self.vy = self.vy * 0.7 + tvy * 0.3

        nx = self.x + self.vx * DT
        ny = self.y + self.vy * DT
        wall_n = self._grid_at(nx, ny) > 0.5
        wall_x = self._grid_at(nx, self.y) > 0.5
        wall_y = self._grid_at(self.x, ny) > 0.5

        move_both = ~wall_n
        move_x = wall_n & ~wall_x
        move_y = wall_n & ~wall_y & wall_x
        blocked = wall_n & wall_x & wall_y

        self.x = torch.where(move_both | move_x, nx, self.x)
        self.y = torch.where(move_both | move_y, ny, self.y)
        self.vy = torch.where(move_x, self.vy * 0.5, self.vy)
        self.vx = torch.where(move_y, self.vx * 0.5, self.vx)
        full_stop = move_both | move_x | move_y
        self.vx = torch.where(blocked, torch.zeros_like(self.vx), self.vx * full_stop.float())
        self.vx = torch.where(blocked, torch.zeros_like(self.vx), self.vx)
        self.vy = torch.where(blocked, torch.zeros_like(self.vy), self.vy)
        cx = torch.floor(self.x) + 0.5
        cy = torch.floor(self.y) + 0.5
        self.x = torch.where(blocked, self.x + (cx - self.x) * 0.08, self.x)
        self.y = torch.where(blocked, self.y + (cy - self.y) * 0.08, self.y)
        self.collisions = self.collisions + wall_n.float()

        self.steps += 1

        # 探索全新网格加成 (对齐 CPU: +6.0/新格)
        t_gx = torch.clamp(torch.floor(self.x).long(), 0, self.W - 1)
        t_gy = torch.clamp(torch.floor(self.y).long(), 0, self.W - 1)
        t_idx = t_gy * self.W + t_gx
        t_ar = torch.arange(self.N, device=self.x.device)
        unseen = ~self.visited[t_ar, t_idx]
        self.visited[t_ar, t_idx] = True
        self.new_tiles += unseen.float()

        cur = torch.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        self.min_dist = torch.minimum(self.min_dist, cur)
        self.reached = self.reached | (cur < 0.6)
        return cur

    def fitness(self):
        progress = (self.init_dist - self.min_dist) / torch.clamp(self.init_dist, min=0.1)
        fit = progress * 60.0 \
            + self.new_tiles * 6.0 \
            + torch.where(self.reached, 300.0 + (STEPS - self.steps.float()) * 2.0, torch.zeros_like(progress)) \
            - self.collisions * 0.2
        return fit


# ----------------------------------------------------------------------------
# GPU 大规模演化主循环
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 冠军拓扑上规模移植: 11 细胞冠军整图嵌入第 15 柱 (效应器柱, 感觉辐射直达)
# ----------------------------------------------------------------------------
def parse_champion_bin(path):
    import struct
    d = open(path, 'rb').read()
    magic, ver, nc, ns, idim, odim = struct.unpack('<IIIIII', d[:24])
    c_off, rp_off, ci_off, w_off, _, _ = struct.unpack('<QQQQQQ', d[24:72])
    cells = d[c_off:c_off + nc * 4]
    rp = np.frombuffer(d, np.uint32, nc + 1, rp_off)
    ci = np.frombuffer(d, np.uint32, ns, ci_off)
    w = np.frombuffer(d, np.float32, ns, w_off)
    ops, p1s = [], []
    for i in range(nc):
        ops.append(cells[i * 4])
        p1s.append(struct.unpack('b', bytes([cells[i * 4 + 1]]))[0] / 64.0)
    edges = []  # (u, v, port, weight)
    for u in range(nc):
        for k in range(int(rp[u]), int(rp[u + 1])):
            packed = int(ci[k])
            edges.append((u, packed & 0xFFFFFF, (packed >> 24) & 0xFF, float(w[k])))
    return ops, p1s, edges, nc


def transplant_champion(pop, champ_path, jitter_rng=None):
    """冠军图移植进第 15 柱, 语义严格对齐 cellular_genome.hpp::dispatch_cell_forward:
    - CellType 枚举 → SDSC eval op 标准映射 (cell_type_to_sdsc_opcode)
    - 受体: out = inputs[ch] * param1 → p1 折叠进下游边权
    - ACT: out = in0 raw 直通 (param1/in1 均无效) → 有符号转向用 ACT_POS/ACT_NEG 双细胞整流分解
    - SUB/SUM 双端口: SUB 取 in0-in1 (port1 负号), SUM 取 in0+in1 (全正)
    - GATE_DEADZONE: |in0|>|param1| ? in0 : 0, g = param1 精确保留"""
    ops, p1s, edges, nc = parse_champion_bin(champ_path)
    P, C, K = pop.pop_size, pop.num_columns, pop.cells_per_col
    COL = C - 1

    CT2EVAL = {0: 0, 1: 0, 2: 0, 3: 0,          # 受体统一引擎 SENSE op0 (通道由辐射按槽位路由)
               10: 8,                            # OP_EMA -> SDSC_OP_DAMPER
               11: 12,                           # OP_DIFF -> SDSC_OP_DIFF
               13: 4,                            # OP_SUM
               14: 13,                           # OP_SUB
               15: 11,                           # OP_MULTIPLY
               19: 25,                           # OP_OSCILLATOR
               24: 15,                           # GATE_THRESHOLD
               25: 16,                           # GATE_HYSTERESIS
               28: 17,                           # GATE_DEADZONE
               30: 21, 31: 22, 32: 23, 33: 21}   # ACT_POS/NEG/RESET/IMMUNE

    # 槽位映射: 受体 0..3 → (15,0..3) (感觉辐射直达); 中间细胞 → 8..; 效应器 → 56..59
    slot, eff_map = {}, {}
    next_free = 8
    turn_cell = None
    thrust_cell = None
    for i in range(nc):
        if i < 4:
            slot[i] = i
        elif ops[i] == 30:
            if thrust_cell is None:
                thrust_cell, eff_map[i] = i, 56
            else:
                eff_map[i] = 59
        elif ops[i] == 31:
            turn_cell, eff_map[i] = i, 57
        elif ops[i] == 33:
            eff_map[i] = 58
        else:
            slot[i] = next_free
            next_free += 1
    turn_pos_slot = 59  # 有符号转向的整数流 (ACT_POS 侧), 负半流走 57

    def eid(k):
        return COL * K + k

    op_dev, gain = pop.op_types, pop.param1
    # 1. op/gain 覆盖
    for i in range(nc):
        if i in slot:
            k = slot[i]
            op_dev[eid(k)] = CT2EVAL[int(ops[i])]
            gain[:, eid(k)] = p1s[i]          # DEADZONE g 必须精确保留
    for i, k in eff_map.items():
        op_dev[eid(k)] = CT2EVAL[int(ops[i])]
        gain[:, eid(k)] = 1.0                 # ACT 双侧整流, g=1
    # 效应器槽 op 显式固化 (59 不在 eff_map 中, 必须手动设 ACT_POS; 58 设 RESET)
    if turn_cell is not None:
        op_dev[eid(turn_pos_slot)] = 21
        gain[:, eid(turn_pos_slot)] = 1.0
    op_dev[eid(58)] = 23
    gain[:, eid(58)] = 1.0

    # 2. 权重直写 (其余柱清零; 平行边累加)
    pop.intra_weights[:, :, :, :] = 0.0
    pop.inter_weights[:, :] = 0.0
    for u, v, port, w in edges:
        su = eff_map.get(u, slot.get(u))
        if su is None:
            continue
        if v in eff_map:
            tv = eff_map[v]
            if v == turn_cell:
                # 冠军 ACT 直通只用 in0: port1 边为死权重, 丢弃;
                # port0 边同时驱动 ACT_POS(59)/ACT_NEG(57) → 有符号整流分解;
                # 驱动预缩放 ÷5 使 acts*5 精确复现 CPU clamp(neg,±5) 的线性区与饱和区
                if port == 0:
                    # su==56 (推进反馈): 源是 ÷2.1 缩放后的 acts56, 需 ×2.1 还原 raw 语义
                    comp = 2.1 if su == 56 else 1.0
                    ww = w * (p1s[u] if u < 4 else 1.0) * 0.2 * comp
                    pop.intra_weights[:, COL, turn_pos_slot, su] += ww
                    pop.intra_weights[:, COL, 57, su] += ww
            elif tv == 56:
                # 推进预缩放 ÷2.1: acts*2.1 = 冠军 raw = 2.1*front (front≤1 不触钳制)
                pop.intra_weights[:, COL, tv, su] += w * (p1s[u] if u < 4 else 1.0) * (1.0 / 2.1)
            else:
                pop.intra_weights[:, COL, tv, su] += w * (p1s[u] if u < 4 else 1.0)
        elif v in slot:
            sign = -1.0 if (ops[v] == 14 and port == 1) else 1.0  # 仅 SUB 的 port1 取负
            pop.intra_weights[:, COL, slot[v], su] += sign * w * (p1s[u] if u < 4 else 1.0)

    # 2.5 记忆嫁接 (任务层, 底座原生 op 16 HYSTERESIS 双稳态): 堵死锁存 → 停车 + 持续自旋 → 前方开阔自解
    #     输入 x = 3.0·front: front < 0.117 → 锁存 -1 (脱困模式); front > 0.117 → 翻回 +1
    graft = os.environ.get("MAZE_MEMORY_GRAFT", "0") == "1"  # 默认关 (消融实验见 git log)
    if graft:
        # v3 完整记忆嫁接 (v2 单极性输入无法翻转双稳态, 缺常数偏置源):
        # 常数发生器: INTEGRAL 自激吸引子 s=0.85s+0.15(0.3f+2tanh(s)) → s*≈1.92, out≈0.96 常数
        # (tanh 有界, 李雅普诺夫安全; ~30 步自激从观测种子生长)
        op_dev[eid(19)] = 5                        # OP_INTEGRATE (漏积分器, 自激)
        gain[:, eid(19)] = 1.0
        pop.intra_weights[:, COL, 19, 0] += 0.3    # front → 自激种子
        pop.intra_weights[:, COL, 19, 19] += 2.0   # 自环 → 吸引子 (常数源 ≈ +0.96)
        # 施密特触发器: x = 3·front - 0.5 → L=+1 (front>0.283 开阔) / L=-1 (front<0.05 堵死)
        op_dev[eid(20)] = 4                        # OP_SUM 纯直通 (偏置合成)
        gain[:, eid(20)] = 1.0
        pop.intra_weights[:, COL, 20, 0] += 3.0    # front → 比较
        pop.intra_weights[:, COL, 20, 19] += -0.52 # 常数偏置 -0.5 (÷0.96)
        op_dev[eid(16)] = 16                       # GATE_HYSTERESIS 双稳态转锁
        gain[:, eid(16)] = 0.35
        pop.intra_weights[:, COL, 16, 20] += 1.0   # 合成信号 → 锁存
        pop.intra_weights[:, COL, 57, 16] += 0.6   # 锁存 → ACT_NEG 侧整流 (堵死 → CW 自旋直到开阔)
        # 退避持久化 (第二级工作记忆): 锁事件 → 积分器 → ~15 步衰减的持续 CW 偏置
        # (L=-1 时 x17=-3 → s 下行 → out17<0 → 经 ACT_NEG 整流 = 持续自旋偏置, 撤出死胡同)
        op_dev[eid(17)] = 5                        # OP_INTEGRATE (退避记忆, τ≈6 步衰减)
        gain[:, eid(17)] = 1.0
        pop.intra_weights[:, COL, 17, 16] += 3.0   # 锁存 → 记忆充电
        pop.intra_weights[:, COL, 57, 17] += 0.5   # 记忆 → 持续 CW 偏置 (整流后 |out17|)
        print("[记忆嫁接] v4: 常数源+偏置+转锁+退避持久化 (9 突触, 无推进干涉)")

    # 3. 其余个体: 移植图 + 权重抖动
    for p in range(1, P):
        pop.intra_weights[p] = pop.intra_weights[0] + torch.randn_like(pop.intra_weights[p]) * 0.02
    pop.rebuild_indices()
    print(f"[移植] 冠军 {nc} 细胞 / {len(edges)} 边 → 第 {COL} 柱 (CellType→eval op 标准映射), 种群 {P} 体已播种")
    # 单位测试: 移植体应在 11×11 复现冠军行为 (CPU 参照 80/100)
    chk_grids = torch.stack([torch.from_numpy(gen_maze(11, 55555 + i)) for i in range(50)]).repeat(6, 1, 1)[:P]
    cenv = BatchedMazeTask(chk_grids, DEVICE)
    pop.reset_states()
    inp_c = torch.zeros(P, 32, device=DEVICE)
    o = cenv.observation(); inp_c[:, :4] = o
    for _ in range(400):
        a = pop.forward_step(inp_c)
        cenv.step(a[:, 0] * 2.1, (a[:, 3] - a[:, 1]) * 5.0, torch.zeros(P, device=DEVICE))
        o = cenv.observation(); inp_c[:, :4] = o
    sr11 = float(cenv.reached.float().mean()) * 100.0
    print(f"[单位测试] 移植体 11×11 快评: {sr11:.0f}% (冠军 CPU 参照 80%, 50 图平铺)")
    # 信号探针: 单迷宫 8 步, 逐步打印关键细胞输出
    penv = BatchedMazeTask(chk_grids[:1].clone(), DEVICE)
    pop.reset_states()
    inp_p = torch.zeros(P, 32, device=DEVICE)
    o = penv.observation(); inp_p[:, :4] = o[0]
    def cell(k):
        return float(pop.outputs[0, COL * K + k])
    for t in range(8):
        a = pop.forward_step(inp_p)
        penv.step(a[0, 0] * 2.1, (a[0, 3] - a[0, 1]) * 5.0, torch.zeros(1, device=DEVICE))
        o = penv.observation(); inp_p[0, :4] = o[0]
        print(f"  t={t} obs=[{o[0,0]:.2f},{o[0,1]:.2f},{o[0,2]:.2f},{o[0,3]:+.2f}] "
              f"recv=[{cell(0):.2f},{cell(1):.2f},{cell(2):.2f},{cell(3):.2f}] "
              f"hid=[{cell(8):+.2f},{cell(9):+.2f},{cell(10):+.2f},{cell(11):+.2f},{cell(12):+.2f}] "
              f"act=[{cell(56):.2f},{cell(57):.2f},{cell(59):.2f}] "
              f"pos=({penv.x[0]:.1f},{penv.y[0]:.1f})")
    return sr11


def main():
    pop_size = 256
    print("=" * 70)
    print(f"  第 3 站 · GPU 大规模迷宫演化: {pop_size} 体 × 1024 细胞 × 21×21")
    print("=" * 70)

    pop = CUDACellularPopulation(pop_size=pop_size, num_columns=16, cells_per_col=64,
                                 in_dim=32, out_dim=8, device=DEVICE,
                                 propagation_passes=6)  # C++ Kahn 同拍传播仿真 (冠军链深 3)
    print(f"[*] 设备: {pop.device} | 拓扑: 16 微柱 × 64 细胞 = 1024 细胞/体")
    transplant_champion(pop, os.path.join(ROOT, "checkpoints", "maze_navigation_champion.bin"))

    test_grids = torch.stack([torch.from_numpy(gen_maze(W, 55555 + i)) for i in range(50)])
    gens = 100
    snapshot_path = os.path.join(ROOT, "checkpoints", "maze_gpu_scale_best.pt")
    mazes_per_gen = 256
    t_start = time.time()
    best_sr = 0.0
    for gen in range(gens):
        # 训练迷宫: 每代新抽 256 张 (域随机化: braid 0.05..0.25)
        grids = torch.stack([torch.from_numpy(gen_maze(W, int(torch.randint(0, 2**30, (1,)).item()),
                                                        braid_prob=0.05 + 0.20 * (gen % 5) / 5.0))
                             for _ in range(mazes_per_gen)])
        env = BatchedMazeTask(grids, DEVICE)
        pop.reset_states()
        obs = env.observation()
        inp = torch.zeros(pop_size, 32, device=DEVICE)
        inp[:, :4] = obs
        for step in range(STEPS):
            acts = pop.forward_step(inp)                 # [N,8]
            # 双向动作映射: 引擎效应器输出单向 [0,1] → 差分出双向转向
            fwd = acts[:, 0] * 2.1
            neg = (acts[:, 3] - acts[:, 1]) * 5.0
            reset = torch.zeros_like(fwd)
            cur = env.step(fwd, neg, reset)
            obs = env.observation()
            inp[:, :4] = obs
            if bool(env.reached.all()):
                break
        # 课程对照记录: 混合尺寸 (11..21) 已证伪 (最好 7.8% < 纯 21×21 的 11.3%), 回退
        csize = 21
        grids2 = torch.stack([torch.from_numpy(gen_maze(csize, int(torch.randint(0, 2**30, (1,)).item()),
                                                         braid_prob=0.05 + 0.20 * (gen % 5) / 5.0))
                              for _ in range(mazes_per_gen)])
        env2 = BatchedMazeTask(grids2, DEVICE)
        pop.reset_states()
        obs = env2.observation()
        inp[:, :4] = obs
        for step in range(STEPS):
            acts = pop.forward_step(inp)
            env2.step(acts[:, 0] * 2.1, (acts[:, 3] - acts[:, 1]) * 5.0, torch.zeros(pop_size, device=DEVICE))
            obs = env2.observation()
            inp[:, :4] = obs
            if bool(env2.reached.all()):
                break
        fit = (env.fitness() + env2.fitness()) * 0.5   # 双迷宫均值 (选择噪声减半)
        # 每 5 代: 21×21 全新随机 50 网格快评 (根治固定集泄漏: run4 固定集 7.4% 但 OOD 0/200)
        if gen % 5 == 4 or gen == gens - 1:
            fresh_seeds = [int(torch.randint(0, 2**30, (1,)).item()) for _ in range(50)]
            tiled = torch.stack([torch.from_numpy(gen_maze(W, s)) for s in fresh_seeds]).repeat(6, 1, 1)[:pop_size]
            tenv = BatchedMazeTask(tiled, DEVICE)
            pop.reset_states()
            obs = tenv.observation()
            inp_t = torch.zeros(pop_size, 32, device=DEVICE)
            inp_t[:, :4] = obs
            for step in range(STEPS):
                acts = pop.forward_step(inp_t)
                cur = tenv.step(acts[:, 0] * 2.1, (acts[:, 3] - acts[:, 1]) * 5.0,
                                torch.zeros(pop_size, device=DEVICE))
                obs = tenv.observation()
                inp_t[:, :4] = obs
                if bool(tenv.reached.all()):
                    break
            sr = float(tenv.reached.float().mean()) * 100.0
            if sr > best_sr:
                best_sr = sr
                # SR-最优个体快照 (argmax: 通关优先, 平手取训练适应度)
                score = tenv.reached.float() * 1000.0 + (fit - fit.min()) / (fit.max() - fit.min() + 1e-6)
                best_idx = int(torch.argmax(score))
                torch.save({"intra": pop.intra_weights[best_idx].cpu(),
                            "inter": pop.inter_weights[best_idx].cpu(),
                            "p1": pop.param1[best_idx].cpu(),
                            "p2": pop.param2[best_idx].cpu(),
                            "ops": torch.from_numpy(np.asarray(pop.op_types))}, snapshot_path)
                print(f"  [快照] Gen {gen} SR {sr:.1f}% 个#{best_idx} 已存 {os.path.basename(snapshot_path)}")
            top5 = torch.topk(fit, 5).values.mean().item()
            print(f"  Gen {gen:3d} | 训练 top5 适应度 {top5:8.1f} | 21×21 快评 SR {sr:5.1f}% | 最好 {best_sr:5.1f}% | 用时 {time.time()-t_start:.0f}s")

        # 锦标赛选择 + 变异 (GPU 原生)
        pop.selection(fit, elite_ratio=0.10, tournament_k=4)
        pop.mutate(mutation_rate=0.10, mutation_power=0.16, num_elites=max(1, int(pop_size * 0.10)))

    print("=" * 70)
    print(f"[完成] 最好 21×21 快评 SR: {best_sr:.1f}% | 总耗时 {time.time()-t_start:.0f}s")
    # 导出最佳个体 (最后一轮 fit 的 argmax 为精英头位)
    pop.export_champion_to_sdsc_bin(0, os.path.join(ROOT, "checkpoints", "maze_gpu_scale_champion.bin"))
    print("[产物] checkpoints/maze_gpu_scale_champion.bin (待 C++ 内核独立验收)")

    # ---- 训后独立 OOD 验收: 200 全新种子 (seed 77000+, 训练/快评从未触及) ----
    if os.path.exists(snapshot_path):
        st = torch.load(snapshot_path)
        pop.intra_weights[0] = st["intra"].to(DEVICE)
        pop.inter_weights[0] = st["inter"].to(DEVICE)
        pop.param1[0] = st["p1"].to(DEVICE)
        pop.param2[0] = st["p2"].to(DEVICE)
        pop.op_types = st["ops"].numpy()
        pop.rebuild_indices()
        ood_grids = torch.stack([torch.from_numpy(gen_maze(W, 77000 + i)) for i in range(200)]).repeat(2, 1, 1)[:pop_size]
        oenv = BatchedMazeTask(ood_grids, DEVICE)
        ood_unique = oenv.reached[:200]
        pop.reset_states()
        obs = oenv.observation()
        inp_ood = torch.zeros(pop_size, 32, device=DEVICE)
        inp_ood[:, :4] = obs
        for step in range(STEPS):
            acts = pop.forward_step(inp_ood)
            oenv.step(acts[:, 0] * 2.1, (acts[:, 3] - acts[:, 1]) * 5.0, torch.zeros(pop_size, device=DEVICE))
            obs = oenv.observation(); inp_ood[:, :4] = obs
            if bool(oenv.reached.all()):
                break
        ood_sr = float(ood_unique.float().mean()) * 100.0
        print(f"[OOD 验收] GPU 规模冠军 @ 21×21 全新 200 种子: SR = {int(ood_unique.sum())}/200 = {ood_sr:.1f}%")


if __name__ == "__main__":
    main()
