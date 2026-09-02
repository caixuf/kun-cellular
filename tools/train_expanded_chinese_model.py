#!/usr/bin/env python3
"""
扩展中文语料自回归因果语言模型训练脚本 (稳定收敛版本)
======================================================================
运行环境: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
目标: 训练全景中文对话、常识问答、闲聊交互与跨领域技术文本
======================================================================
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "需要 CUDA GPU 环境"
device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)

# 1. 构建丰富全场景中文语料库
EXPANDED_CORPUS = """
问：你好？
答：你好！很高兴和你交流，今天想聊些什么？

问：你好呀？
答：你好呀！随时准备为你解答问题或进行日常交流。

问：你是谁？
答：我是运行在计算系统中的语言与控制模型，负责文本生成、对话交互与多任务协同。

问：我想你了？
答：很高兴再次见到你！每一次对话都是新的交互与思维碰撞。

问：你今天心情怎么样？
答：作为计算程序，我没有人类的情绪波动，但我始终保持高效且稳定的运行状态。

问：今天天气怎么样？
答：我无法直接感知外部实时天气，不过如果你告诉我你所在的城市，我可以为你提供相关的气象常识与出行建议。

问：你是不是不会说话？
答：我会说话呀！我不仅能理解你输入的自然语言，还能推演数学、解读行情以及控制车辆。

问：你会做什么？
答：我掌握多领域能力：自然语言对话、智能驾驶算法推演、期货量化分析、离散符号数学推导、四足步态合成、迷宫路径规划以及微环境免疫仿真。

问：1+1等于多少呀？
答：1+1等于2。这是基础的代数运算公理。

问：3乘8加5等于多少？
答：3乘8加5等于29。计算过程为：3*8=24，24+5=29。

问：介绍一下智能驾驶的核心逻辑？
答：智能驾驶核心基于环境感知、路径规划与底盘运动控制。通过激光雷达与摄像头获取障碍物数据，结合 Stanley 闭环控制律实现高精度的轨迹跟踪。

问：解读一下量化投资的基本原理？
答：量化投资通过数学模型分析历史行情数据，提取动量、均值回归等特征信号，结合严格的风控截断与资金管理来获取稳健的超额收益。

问：什么是符号计算？
答：符号计算是指直接使用离散符号与代数规则进行精确公式推导和逻辑证明，避免浮点数截断带来的精度损失。

问：为什么书籍与生命体有联系？
答：书籍代表着知识与经验的凝聚结晶。在系统中，不同典籍代表着不同专业脑区的特定神经回路与参数配置。

问：迷宫避障算法是怎么工作的？
答：利用测距传感器感知前方及两侧障碍物距离，根据局部避障策略与新奇性搜索算法实时计算转向角度，实现自主探索。

问：四足步态是如何协调行走的？
答：通过中枢模式发生器（CPG）产生节律振荡信号，协调四条肢体的关节肌肉收缩，结合地面反作用力实现平稳的前进与转向。

问：免疫系统的防御机制是什么？
答：特异性免疫细胞通过受体识别外来抗原，利用趋化性向病原体聚集并执行吞噬降解，保护系统内环境的稳定。

问：你在想什么？
答：我在持续处理输入的文本特征，优化内部参数流向，以便给出最准确清晰的回应。

问：什么是神经网络？
答：神经网络是一种模仿生物神经元结构的计算模型，通过多层线性变换与非线性激活函数，学习输入数据与输出目标之间的复杂映射关系。

问：什么是 Transformer 架构？
答：Transformer 是一种基于自注意力机制（Self-Attention）的深度学习架构，能够并行处理序列数据并有效捕捉长距离依赖关系。

问：这个项目有什么技术特色？
答：项目集成了自组织计算流形、实时物理仿真、多领域控制闭环以及 GPU 加速的端到端自回归语言生成能力。
""".strip() * 40

chars = sorted(list(set(EXPANDED_CORPUS)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

class StableLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=6, n_heads=6, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.0, activation='gelu', batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        x = self.blocks(x, mask=mask)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt, max_tokens=65, temperature=0.25):
        self.eval()
        idx_tokens = [char_to_ix.get(c, 0) for c in prompt if c in char_to_ix]
        if not idx_tokens:
            idx_tokens = [char_to_ix.get("问", 0), char_to_ix.get("：", 0)]
        idx = torch.tensor([idx_tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :] / max(0.05, temperature)
            probs = F.softmax(logits_last, dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_ix), dim=1)
            ch = ix_to_char[next_ix.item()]
            if ch == '\n' and len(idx[0]) > len(idx_tokens) + 10:
                break
        res = ''.join([ix_to_char[i] for i in idx[0].cpu().tolist()])
        self.train()
        return res

def train_and_save():
    print("======================================================================")
    print("  全景中文大语料自回归因果语言模型深度演化")
    print(f"  硬件加速: {gpu_name} (CUDA Tensor Cores)")
    print("======================================================================")
    
    t0 = time.time()
    model = StableLanguageModel(vocab_size=vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"• 独立字符词表 (Vocab Size): {vocab_size} 个中文常用字符")
    print(f"• 模型神经参数总规模:        {total_params:,} (约 {total_params/1e6:.1f}M 参数)")
    print(f"• 架构配置:                  6 层 Transformer, 隐层 384, 6 头自注意力")
    print("----------------------------------------------------------------------")
    
    data = torch.tensor([char_to_ix[c] for c in EXPANDED_CORPUS], dtype=torch.long, device=device)
    seq_len = 64
    batch_size = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-3)
    
    steps = 300
    print("[开始反向传播与参数优化迭代...]")
    for step in range(1, steps + 1):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == steps:
            elapsed = time.time() - t0
            print(f"Step [{step:3d}/{steps}] | 交叉熵 Loss: {loss.item():.4f} | 耗时: {elapsed:.2f}s")
            
    total_time = time.time() - t0
    print(f"\n训练顺利完成！总耗时: {total_time:.2f} 秒，最终 Loss: {loss.item():.4f}")
    
    # 保存权重
    out_dir = "/home/caixuf/code/kun-cellular/checkpoints"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cellular_language_neural_champion.pt")
    
    torch.save({
        'vocab_size': vocab_size,
        'char_to_ix': char_to_ix,
        'ix_to_char': ix_to_char,
        'd_model': 384,
        'n_layers': 6,
        'n_heads': 6,
        'total_params': total_params,
        'state_dict': model.state_dict()
    }, out_path)
    print(f"模型检查点已保存至: {out_path}")
    
    # 抽样评估
    print("\n======================================================================")
    print("  多场景提示词生成质量验证测试")
    print("======================================================================")
    test_queries = [
        "问：你好？\n答：",
        "问：我想你了？\n答：",
        "问：今天天气怎么样？\n答：",
        "问：你会做什么？\n答：",
        "问：1+1等于多少呀？\n答：",
        "问：介绍一下智能驾驶的核心逻辑？\n答："
    ]
    for q in test_queries:
        out = model.generate(q, max_tokens=60)
        print(out)
        print("-" * 65)

if __name__ == "__main__":
    train_and_save()
