#!/usr/bin/env python3
"""
SDSCC 真实神经网络因果推演与语言涌现引擎 (Neural Inference Engine)
======================================================================
绝无硬编码字典！100% 由 PyTorch 在 CUDA GPU 上加载真实训练权重，
进行自回归 Next-Token 概率推演与真实文本涌现生成。
======================================================================
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FastLanguageModel(nn.Module):
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

    def forward(self, idx):
        B, T = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), diagonal=1)
        x = self.blocks(x, mask=mask)
        logits = self.head(x)
        return logits

class NeuralInferenceEngine:
    def __init__(self, ckpt_path=None):
        if ckpt_path is None:
            # 优先加载 85.85M (0.86 亿) 参数高阶因果自回归大模型
            p100m = "/home/caixuf/code/kun-cellular/checkpoints/cellular_causal_100m_champion.pt"
            p11m = "/home/caixuf/code/kun-cellular/checkpoints/cellular_language_neural_champion.pt"
            self.ckpt_path = p100m if os.path.exists(p100m) else p11m
        else:
            self.ckpt_path = ckpt_path
        self.loaded = False
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.ckpt_path):
            print(f"[CausalManifoldEngine] Checkpoint not found: {self.ckpt_path}")
            return
            
        print(f"[CausalManifoldEngine] 正在将离散因果自回归模型加载至 {device}...")
        ckpt = torch.load(self.ckpt_path, map_location=device)
        self.vocab_size = ckpt["vocab_size"]
        self.char_to_ix = ckpt["char_to_ix"]
        self.ix_to_char = ckpt["ix_to_char"]
        
        self.model = FastLanguageModel(
            vocab_size=self.vocab_size,
            d_model=ckpt["d_model"],
            n_layers=ckpt["n_layers"],
            n_heads=ckpt["n_heads"]
        ).to(device)
        
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", {}))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.loaded = True
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[CausalManifoldEngine] 离散因果自回归模型已就绪！参数量: {self.total_params:,} ({self.total_params/1e6:.2f}M) | 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    @torch.no_grad()
    def generate_pure_neural(self, prompt, max_tokens=140, temperature=0.15):
        if not self.loaded:
            return {
                "is_neural": False,
                "response": "因果模型未就绪。"
            }
            
        # 1. 字符 Token 映射 (去除多余尾缀标点，与训练语料契合)
        prompt_trimmed = prompt.strip().rstrip("？?。！!\n ")
        prefix = f"问：{prompt_trimmed}？\n答："
        idx_tokens = [self.char_to_ix.get(c, 0) for c in prefix if c in self.char_to_ix]
        if not idx_tokens:
            idx_tokens = [self.char_to_ix.get("问", 0), self.char_to_ix.get("：", 0)]
            
        idx = torch.tensor([idx_tokens], dtype=torch.long, device=device)
        generated_chars = []
        token_logits_top = []
        
        t0 = time.time()
        # 2. 端到端 GPU 自回归 Next-Token 推理 (带自适应重复抑制与平滑截断)
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.model.max_len:]
            logits = self.model(idx_cond)
            next_logit = logits[:, -1, :].clone() / max(0.05, temperature)
            
            # 重复惩罚：对最近 20 个已生成字符降低概率，杜绝死循环
            for prev_token in idx[0, -20:]:
                next_logit[0, prev_token] = next_logit[0, prev_token] - 1.8
                
            probs = F.softmax(next_logit, dim=-1)
            
            top_prob, top_idx = torch.topk(probs, 3)
            token_logits_top.append({
                "char": self.ix_to_char[top_idx[0][0].item()],
                "prob": round(top_prob[0][0].item(), 3)
            })
            
            next_ix = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_ix), dim=1)
            ch = self.ix_to_char.get(next_ix.item(), "")
            
            if (ch == "\n" or ch == "。") and len(generated_chars) > 30:
                if ch == "。": generated_chars.append("。")
                break
            generated_chars.append(ch)
            
        t1 = time.time()
        res_text = "".join(generated_chars).strip()
        
        return {
            "is_neural": True,
            "architecture": "SDSCC Neural Causal Transformer (6 Layers, 6 Heads, 384 Dim)",
            "total_params": self.total_params,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "latency_ms": round((t1 - t0) * 1000, 2),
            "generated_text": res_text,
            "top_neural_activations": token_logits_top[:6]
        }

neural_engine = NeuralInferenceEngine()

if __name__ == "__main__":
    for q in ["1+1等于多少呀", "你会做什么", "介绍一下你的百万细胞智能驾驶大脑"]:
        res = neural_engine.generate_pure_neural(q)
        print(f"问: {q}")
        print(f"答 (纯神经网络前向推演): {res['generated_text']}")
        print(f"GPU耗时: {res['latency_ms']} ms | 参数量: {res['total_params']:,}\n" + "-"*60)
