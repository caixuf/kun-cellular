#!/usr/bin/env python3
"""第 3 站独立 OOD 验收: GPU 规模冠军 @ 全新随机种子 (21×21, 400 步, 与 CPU 验收同预算)
用法: python3 tools/verify_maze_gpu_ood.py [grid_bin] [n_seeds]
  grid_bin 若提供则从 .npy 加载网格 (供 C++ 交叉验收), 否则现场生成并保存"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_maze_gpu_scale import CUDACellularPopulation, BatchedMazeTask, transplant_champion, gen_maze, W, STEPS, ROOT

def main():
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    grid_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/maze_ood_grids.npy"
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 77000  # 独立种子域 (训练/快评均未触及)

    grids = np.stack([gen_maze(W, seed0 + i, braid_prob=0.15) for i in range(n_seeds)])
    np.save(grid_path, grids)
    pop = CUDACellularPopulation(max(64, n_seeds), 16, 64, 32, 8, device="cuda", propagation_passes=6)
    transplant_champion(pop, os.path.join(ROOT, "checkpoints", "maze_gpu_scale_champion.bin"))
    env = BatchedMazeTask(torch.from_numpy(grids), "cuda")
    pop.reset_states()
    inp = torch.zeros(env.N, 32, device="cuda")
    obs = env.observation(); inp[:, :4] = obs
    for step in range(STEPS):
        a = pop.forward_step(inp)
        env.step(a[:, 0] * 2.1, (a[:, 3] - a[:, 1]) * 5.0, torch.zeros(env.N, device="cuda"))
        obs = env.observation(); inp[:, :4] = obs
        if bool(env.reached.all()):
            break
    sr = float(env.reached.float().mean()) * 100.0
    print(f"[OOD] GPU 规模冠军 @ 21×21 全新 {n_seeds} 种子 (seed {seed0}+): SR = {env.reached.sum()}/{env.N} = {sr:.1f}%")
    print(f"[OOD] 网格已存 {grid_path} (C++ 交叉验收用)")
    return sr

if __name__ == "__main__":
    main()
