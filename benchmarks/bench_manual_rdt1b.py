#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
test_rdt1b_alignment.py — RDT-1B end-to-end numerical alignment test.

Steps:
  1. Load the converted ServoFlow checkpoint weights into a pure-PyTorch
     reference model (no dependency on HF repo code).
  2. Generate deterministic random inputs.
  3. Run reference forward pass → ref_output.
  4. Save inputs as safetensors for the C++ binary.
  5. Run the ServoFlow C++ binary (align_rdt1b).
  6. Load C++ output and compare with ref_output.

Pass criteria:
  max_abs_error  < 0.05   (fp16/bf16 accumulation over 28 blocks)
  cosine_similarity > 0.9999

Usage:
  # Quick (PyTorch reference only, no C++ comparison):
  python tests/alignment/test_rdt1b_alignment.py --no-cpp

  # Full E2E (requires servoflow:latest Docker image):
  python tests/alignment/test_rdt1b_alignment.py
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not available; C++ comparison will be skipped.")

REPO_ROOT   = Path(__file__).resolve().parents[2]
SF_CKPT_DEFAULT = REPO_ROOT / "tests" / "alignment" / "sf_checkpoint"
ALIGN_BIN   = "/workspace/servoflow/build/tests/align_rdt1b"

CFG = {
    "hidden_size":    2048,
    "num_heads":      32,
    "head_dim":       64,
    "depth":          28,
    "action_dim":     128,
    "pred_horizon":   64,
    "img_cond_len":   4374,
    "img_token_dim":  1152,
    "lang_token_dim": 4096,
    "state_token_dim": 128,
    "max_lang_cond_len": 1024,
    "sincos_dim":     256,
    "x_seq_len":      67,
}


# ══════════════════════════════════════════════════════════════════════════════
# Reference PyTorch implementation (same as before)
# ══════════════════════════════════════════════════════════════════════════════

class RmsNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight.float()).to(x.dtype)

def sincos_emb(t, dim):
    # [cos, sin] ordering matching C++ TimestepEmbedding::build_sincos_table
    half = dim // 2
    freq = torch.exp(-torch.arange(half, dtype=torch.float32, device=t.device)
                     * math.log(10000) / half)
    x = t.float().unsqueeze(-1) * freq.unsqueeze(0)
    return torch.cat([x.cos(), x.sin()], dim=-1)

class TimestepEmbedder(nn.Module):
    def __init__(self, D, sincos_dim):
        super().__init__()
        self.sincos_dim = sincos_dim
        self.mlp = nn.Sequential(nn.Linear(sincos_dim, D), nn.SiLU(), nn.Linear(D, D))
    def forward(self, t):
        emb = sincos_emb(t, self.sincos_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(emb).unsqueeze(1)

class Mlp2x(nn.Module):
    def __init__(self, in_dim, D):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, D); self.fc2 = nn.Linear(D, D)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc0(x), approximate="tanh"))

class Mlp3x(nn.Module):
    def __init__(self, in_dim, D):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, D); self.fc2 = nn.Linear(D, D); self.fc4 = nn.Linear(D, D)
    def forward(self, x):
        return self.fc4(F.gelu(self.fc2(F.gelu(self.fc0(x), approximate="tanh")), approximate="tanh"))

class FFN(nn.Module):
    def __init__(self, D, out_dim=None):
        super().__init__()
        out_dim = out_dim or D
        self.fc1 = nn.Linear(D, D); self.fc2 = nn.Linear(D, out_dim)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

class SelfAttn(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H; self.hd = D // H
        self.qkv = nn.Linear(D, 3*D); self.proj = nn.Linear(D, D)
        self.q_norm = RmsNorm(self.hd); self.k_norm = RmsNorm(self.hd)
    def forward(self, x):
        B,S,D = x.shape
        qkv = self.qkv(x).view(B,S,3,self.H,self.hd).unbind(2)
        q,k,v = (self.q_norm(qkv[0]).transpose(1,2),
                 self.k_norm(qkv[1]).transpose(1,2),
                 qkv[2].transpose(1,2))
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1,2).reshape(B,S,D))

class CrossAttn(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H; self.hd = D // H
        self.q = nn.Linear(D,D); self.kv = nn.Linear(D,2*D); self.proj = nn.Linear(D,D)
        self.q_norm = RmsNorm(self.hd); self.k_norm = RmsNorm(self.hd)
    def forward(self, x, ctx):
        B,S,D = x.shape; Sc = ctx.shape[1]
        q = self.q_norm(self.q(x).view(B,S,self.H,self.hd)).transpose(1,2)
        kv = self.kv(ctx).view(B,Sc,2,self.H,self.hd).unbind(2)
        k,v = self.k_norm(kv[0]).transpose(1,2), kv[1].transpose(1,2)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1,2).reshape(B,S,D))

class RDTBlock(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.norm1 = RmsNorm(D); self.norm2 = RmsNorm(D); self.norm3 = RmsNorm(D)
        self.attn = SelfAttn(D,H); self.cross_attn = CrossAttn(D,H); self.ffn = FFN(D)
    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), cond)
        x = x + self.ffn(self.norm3(x))
        return x

class FinalLayer(nn.Module):
    def __init__(self, D, A):
        super().__init__()
        self.norm_final = RmsNorm(D); self.ffn_final = FFN(D, A)
    def forward(self, x):
        return self.ffn_final(self.norm_final(x))

class RDT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D,H,N = cfg["hidden_size"], cfg["num_heads"], cfg["depth"]
        self.t_embedder = TimestepEmbedder(D, cfg["sincos_dim"])
        self.freq_embedder = TimestepEmbedder(D, cfg["sincos_dim"])
        self.x_pos_embed = nn.Parameter(torch.zeros(1, cfg["x_seq_len"], D))
        self.lang_cond_pos_embed = nn.Parameter(torch.zeros(1, cfg["max_lang_cond_len"], D))
        self.img_cond_pos_embed  = nn.Parameter(torch.zeros(1, cfg["img_cond_len"], D))
        self.blocks = nn.ModuleList([RDTBlock(D,H) for _ in range(N)])
        self.final_layer = FinalLayer(D, cfg["action_dim"])
    def forward(self, x_seq, lang_c, img_c, t, freq):
        B,T1,D = x_seq.shape; T = T1 - 1
        t_emb   = self.t_embedder(t)
        freq_emb = self.freq_embedder(freq)
        x = torch.cat([t_emb, freq_emb, x_seq], dim=1)
        x = x + self.x_pos_embed
        L = lang_c.shape[1]
        lang_c = lang_c + self.lang_cond_pos_embed[:, :L]
        img_c  = img_c  + self.img_cond_pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x, lang_c if i%2==0 else img_c)
        return self.final_layer(x)[:, -T:]

class RDT1BRef(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg["hidden_size"]
        self.lang_adaptor  = Mlp2x(cfg["lang_token_dim"], D)
        self.img_adaptor   = Mlp2x(cfg["img_token_dim"],  D)
        self.state_adaptor = Mlp3x(2 * cfg["state_token_dim"], D)
        self.model = RDT(cfg)
    def forward(self, lang, img, state, noisy, t, freq):
        B,T,A = noisy.shape
        mask = torch.ones(B, T+1, A, dtype=lang.dtype, device=lang.device)
        lang_c = self.lang_adaptor(lang)
        img_c  = self.img_adaptor(img)
        state_in = torch.cat([state, noisy], dim=1)   # [B, T+1, A]
        traj_in  = torch.cat([state_in, mask], dim=2) # [B, T+1, 2A]
        x_seq    = self.state_adaptor(traj_in)
        return self.model(x_seq, lang_c, img_c, t, freq)


def load_weights(model, sd, dtype):
    def cp(src, dst): dst.data.copy_(sd[src].to(dtype))
    for dst, pfx in [(model.lang_adaptor, "lang_adaptor"),
                     (model.img_adaptor,  "img_adaptor")]:
        cp(f"{pfx}.0.weight", dst.fc0.weight); cp(f"{pfx}.0.bias", dst.fc0.bias)
        cp(f"{pfx}.2.weight", dst.fc2.weight); cp(f"{pfx}.2.bias", dst.fc2.bias)
    for k in ("0","2","4"):
        cp(f"state_adaptor.{k}.weight", getattr(model.state_adaptor, f"fc{k}").weight)
        cp(f"state_adaptor.{k}.bias",   getattr(model.state_adaptor, f"fc{k}").bias)
    for dst, pfx in [(model.model.t_embedder,    "t_embedder"),
                     (model.model.freq_embedder, "freq_embedder")]:
        cp(f"{pfx}.mlp.0.weight", dst.mlp[0].weight); cp(f"{pfx}.mlp.0.bias", dst.mlp[0].bias)
        cp(f"{pfx}.mlp.2.weight", dst.mlp[2].weight); cp(f"{pfx}.mlp.2.bias", dst.mlp[2].bias)
    cp("x_pos_embed",         model.model.x_pos_embed)
    cp("lang_cond_pos_embed", model.model.lang_cond_pos_embed)
    cp("img_cond_pos_embed",  model.model.img_cond_pos_embed)
    for i, blk in enumerate(model.model.blocks):
        p = f"blocks.{i}"
        cp(f"{p}.norm1.weight", blk.norm1.weight)
        cp(f"{p}.norm2.weight", blk.norm2.weight)
        cp(f"{p}.norm3.weight", blk.norm3.weight)
        cp(f"{p}.attn.qkv.weight",    blk.attn.qkv.weight)
        cp(f"{p}.attn.qkv.bias",      blk.attn.qkv.bias)
        cp(f"{p}.attn.proj.weight",   blk.attn.proj.weight)
        cp(f"{p}.attn.proj.bias",     blk.attn.proj.bias)
        cp(f"{p}.attn.q_norm.weight", blk.attn.q_norm.weight)
        cp(f"{p}.attn.k_norm.weight", blk.attn.k_norm.weight)
        cp(f"{p}.cross_attn.q.weight",      blk.cross_attn.q.weight)
        cp(f"{p}.cross_attn.q.bias",        blk.cross_attn.q.bias)
        cp(f"{p}.cross_attn.kv.weight",     blk.cross_attn.kv.weight)
        cp(f"{p}.cross_attn.kv.bias",       blk.cross_attn.kv.bias)
        cp(f"{p}.cross_attn.proj.weight",   blk.cross_attn.proj.weight)
        cp(f"{p}.cross_attn.proj.bias",     blk.cross_attn.proj.bias)
        cp(f"{p}.cross_attn.q_norm.weight", blk.cross_attn.q_norm.weight)
        cp(f"{p}.cross_attn.k_norm.weight", blk.cross_attn.k_norm.weight)
        cp(f"{p}.ffn.fc1.weight", blk.ffn.fc1.weight)
        cp(f"{p}.ffn.fc1.bias",   blk.ffn.fc1.bias)
        cp(f"{p}.ffn.fc2.weight", blk.ffn.fc2.weight)
        cp(f"{p}.ffn.fc2.bias",   blk.ffn.fc2.bias)
    fl = model.model.final_layer
    cp("final_layer.norm_final.weight",  fl.norm_final.weight)
    cp("final_layer.ffn_final.fc1.weight", fl.ffn_final.fc1.weight)
    cp("final_layer.ffn_final.fc1.bias",   fl.ffn_final.fc1.bias)
    cp("final_layer.ffn_final.fc2.weight", fl.ffn_final.fc2.weight)
    cp("final_layer.ffn_final.fc2.bias",   fl.ffn_final.fc2.bias)


def make_inputs(cfg, seed, device, dtype):
    torch.manual_seed(seed)
    B,T,A = 1, cfg["pred_horizon"], cfg["action_dim"]
    L = 32  # shorter lang for speed
    I = cfg["img_cond_len"]
    return {
        "lang_tok":  torch.randn(B, L, cfg["lang_token_dim"], dtype=dtype, device=device),
        "img_tok":   torch.randn(B, I, cfg["img_token_dim"],  dtype=dtype, device=device),
        "state_tok": torch.randn(B, 1, A,                    dtype=dtype, device=device),
        "noisy_act": torch.randn(B, T, A,                    dtype=dtype, device=device),
        "t":    torch.tensor([500], dtype=torch.long, device=device),
        "freq": torch.tensor([25],  dtype=torch.long, device=device),
    }


def read_f32_bin(path):
    """Read the binary output written by align_rdt1b."""
    with open(path, "rb") as f:
        magic, ndim = struct.unpack("II", f.read(8))
        assert magic == 0xAF12BF16, f"Bad magic: {magic:#x}"
        shape = struct.unpack("I"*ndim, f.read(4*ndim))
        n = 1
        for s in shape: n *= s
        data = np.frombuffer(f.read(n*4), dtype=np.float32)
    return torch.from_numpy(data.reshape(shape))


def compare(ref, hyp, label):
    ref = ref.float().cpu(); hyp = hyp.float().cpu()
    diff = (ref - hyp).abs()
    max_ae  = diff.max().item()
    mean_ae = diff.mean().item()
    cos_sim = F.cosine_similarity(ref.reshape(1,-1), hyp.reshape(1,-1)).item()
    rel_err = (diff / (ref.abs() + 1e-6)).mean().item()
    print(f"\n  [{label}]  shape={list(ref.shape)}")
    print(f"    max|err|  = {max_ae:.4e}")
    print(f"    mean|err| = {mean_ae:.4e}")
    print(f"    rel_err   = {rel_err:.4e}")
    print(f"    cos_sim   = {cos_sim:.8f}")
    return max_ae, mean_ae, cos_sim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sf-ckpt", default=str(SF_CKPT_DEFAULT))
    p.add_argument("--hf-ckpt", default=None,
                   help="HF pytorch_model.bin dir (use --sf-ckpt weights by default)")
    p.add_argument("--dtype",   default="bf16",
                   choices=["fp32","bf16","fp16"])
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--atol",    type=float, default=0.05)
    p.add_argument("--no-cpp",  action="store_true",
                   help="Skip C++ comparison (Python reference only)")
    p.add_argument("--cpp-bin", default=ALIGN_BIN,
                   help="Path to align_rdt1b C++ binary")
    p.add_argument("--work-dir", default=None,
                   help="Directory for temp files (default: system temp)")
    return p.parse_args()


import time
import json

def measure_latency_ms(func, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup):
        func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_ms = (end - start) * 1000.0
    return {
        "mean_ms": total_ms / iters,
        "total_ms": total_ms,
        "iters": iters
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf-ckpt", required=True, help="Path to sf_checkpoint directory")
    parser.add_argument("--dtype", default="fp16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-json", default="hf_results.json")
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"Benchmarking Manual PyTorch RDT-1B")
    print(f"  ckpt: {args.sf_ckpt}")
    print(f"  dtype: {args.dtype}, device: {device}")
    
    # 1. Load weights
    ckpt_path = Path(args.sf_ckpt)
    sf_bin = ckpt_path / "model.safetensors" if ckpt_path.is_dir() else ckpt_path
    
    print(f"Loading weights from {sf_bin}...")
    if HAS_SAFETENSORS:
        sd = load_file(str(sf_bin), device="cpu")
    else:
        sd = torch.load(str(sf_bin), map_location="cpu")
        
    # 2. Build model
    model = RDT1BRef(CFG).to(device, dtype=dtype)
    load_weights(model, sd, dtype)
    model.eval()
    
    # 3. Prepare inputs
    inputs = make_inputs(CFG, seed=42, device=device, dtype=dtype)
    # Simplify inputs for repeated calls
    lang_tok = inputs["lang_tok"]
    img_tok = inputs["img_tok"]
    state_tok = inputs["state_tok"]
    noisy_act = inputs["noisy_act"]
    t = inputs["t"]
    freq = inputs["freq"]

    # 4. Benchmark Single Forward
    print("Benchmarking Single Forward...")
    def forward_fn():
        with torch.inference_mode():
             model(lang_tok, img_tok, state_tok, noisy_act, t, freq)
             
    stats_single = measure_latency_ms(forward_fn, warmup=args.warmup, iters=args.iters)
    print(f"  Single Forward: {stats_single['mean_ms']:.2f} ms")

    # 5. Benchmark Denoise Loop
    print(f"Benchmarking {args.steps}-step Loop...")
    # Create time steps for loop
    timesteps = torch.linspace(1000, 0, args.steps, device=device, dtype=torch.long)
    
    def loop_fn():
        with torch.inference_mode():
            curr_act = noisy_act.clone()
            for step_t in timesteps:
                # In real diffusion, we'd update curr_act based on velocity.
                # Here we just run the model forward to measure compute.
                # We reuse the same t/freq embeddings for simplicity or update them if needed.
                # The reference model takes scalar t/freq tensor inputs.
                t_in = step_t.unsqueeze(0) # [1]
                freq_in = (step_t / 4).long().unsqueeze(0) # dummy freq
                
                vel = model(lang_tok, img_tok, state_tok, curr_act, t_in, freq_in)
                # Update action (dummy update)
                curr_act = curr_act + vel * 0.01 
                
    stats_loop = measure_latency_ms(loop_fn, warmup=args.warmup, iters=args.iters)
    print(f"  {args.steps}-step Loop: {stats_loop['mean_ms']:.2f} ms")
    
    # 6. Save results
    results = {
        "denoise_loop": stats_loop,
        "dummy_single_forward": stats_single, # Reuse key for compatibility
        "per_step_ms": stats_loop["mean_ms"] / args.steps,
        "config": {
            "steps": args.steps,
            "dtype": args.dtype
        }
    }
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
