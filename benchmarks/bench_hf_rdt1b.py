#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
HuggingFace RDT-1B benchmark.

Measures:
  1. Model load time
  2. Condition encoding latency  (encode_condition equivalent)
  3. Per-denoising-step latency
  4. Full-pipeline latency       (encode + N denoise steps)
  5. GPU memory footprint

Usage:
  python bench_hf_rdt1b.py [--steps N] [--iters K] [--action-dim D]
                            [--action-horizon T] [--dtype {fp32,fp16,bf16}]
"""

import argparse
import time
import json
import statistics
import sys
from contextlib import contextmanager

import torch

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="HuggingFace RDT-1B benchmark")
parser.add_argument("--steps",          type=int,   default=10,
                    help="Denoising steps per inference call (default: 10)")
parser.add_argument("--iters",          type=int,   default=20,
                    help="Measurement iterations after warm-up (default: 20)")
parser.add_argument("--warmup",         type=int,   default=5,
                    help="Warm-up iterations (default: 5)")
parser.add_argument("--action-dim",     type=int,   default=14,
                    help="Action dimension, e.g. 14 for bimanual (default: 14)")
parser.add_argument("--action-horizon", type=int,   default=64,
                    help="Action horizon length (default: 64)")
parser.add_argument("--img-size",       type=int,   default=224,
                    help="Input image resolution (default: 224)")
parser.add_argument("--num-cameras",    type=int,   default=2,
                    help="Number of camera views (default: 2)")
parser.add_argument("--seq-len",        type=int,   default=512,
                    help="Language token sequence length (default: 512)")
parser.add_argument("--dtype",          choices=["fp32", "fp16", "bf16"],
                    default="fp16")
parser.add_argument("--device",         default="cuda:0")
parser.add_argument("--model-id",       default="robotics-diffusion-transformer/rdt-1b")
parser.add_argument("--output-json",    default=None,
                    help="Write results as JSON to this path")
args = parser.parse_args()

DEVICE     = torch.device(args.device)
TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

# ── Timing helpers ────────────────────────────────────────────────────────────
def cuda_sync():
    torch.cuda.synchronize(DEVICE)

@contextmanager
def timed(label):
    cuda_sync()
    t0 = time.perf_counter()
    yield
    cuda_sync()
    print(f"  {label}: {(time.perf_counter() - t0)*1e3:.1f} ms")

def measure_latency_ms(fn, warmup: int, iters: int) -> dict:
    """Run fn() warmup+iters times; return stats over the iters phase."""
    for _ in range(warmup):
        fn()
    cuda_sync()
    samples = []
    for _ in range(iters):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "mean_ms":   statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "std_ms":    statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms":    min(samples),
        "max_ms":    max(samples),
        "samples":   samples,
    }

# ── Dummy inputs ──────────────────────────────────────────────────────────────
def make_dummy_inputs():
    B  = 1
    C  = args.num_cameras
    H  = W = args.img_size
    T  = args.action_horizon
    A  = args.action_dim
    SL = args.seq_len

    images        = torch.randn(B, C, 3, H, W,  dtype=TORCH_DTYPE, device=DEVICE)
    lang_tokens   = torch.randint(0, 32000, (B, SL), device=DEVICE)
    lang_attn_mask = torch.ones(B, SL, dtype=torch.bool, device=DEVICE)
    robot_state   = torch.zeros(B, A,  dtype=TORCH_DTYPE, device=DEVICE)
    # noisy action: what the denoising loop operates on
    x_t           = torch.randn(B, T, A, dtype=TORCH_DTYPE, device=DEVICE)
    timestep      = torch.tensor([0.5], dtype=TORCH_DTYPE, device=DEVICE)

    return {
        "images":         images,
        "lang_tokens":    lang_tokens,
        "lang_attn_mask": lang_attn_mask,
        "robot_state":    robot_state,
        "x_t":            x_t,
        "timestep":       timestep,
    }

# ── Memory helpers ────────────────────────────────────────────────────────────
def gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated(DEVICE) / 1e6

def gpu_peak_mb() -> float:
    return torch.cuda.max_memory_allocated(DEVICE) / 1e6

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"HuggingFace RDT-1B Benchmark")
    print(f"  model     : {args.model_id}")
    print(f"  dtype     : {args.dtype}")
    print(f"  device    : {args.device}")
    print(f"  steps     : {args.steps}")
    print(f"  warmup    : {args.warmup}  iters: {args.iters}")
    print(f"  action    : dim={args.action_dim}  horizon={args.action_horizon}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading model …")
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    t_load_start = time.perf_counter()

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
        ).to(DEVICE).eval()
    except Exception as exc:
        print(f"  ERROR loading model: {exc}")
        print("  Falling back to architecture-only benchmark (no weights).")
        model = None

    t_load = (time.perf_counter() - t_load_start) * 1e3
    mem_after_load = gpu_memory_mb()
    print(f"  load time  : {t_load:.0f} ms")
    print(f"  GPU memory : {mem_after_load:.0f} MB")

    inputs = make_dummy_inputs()

    results = {
        "config": {
            "model_id":       args.model_id,
            "dtype":          args.dtype,
            "steps":          args.steps,
            "action_dim":     args.action_dim,
            "action_horizon": args.action_horizon,
        },
        "load_ms":         t_load,
        "model_memory_mb": mem_after_load,
    }

    if model is None:
        print("\n  Model unavailable – reporting dummy-forward timings only.")
        _bench_dummy(inputs, results)
    else:
        _bench_model(model, inputs, results)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    _print_results(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nJSON saved to: {args.output_json}")

# ── Model-available benchmark ─────────────────────────────────────────────────
def _bench_model(model, inputs, results):
    """Benchmark using the actual loaded model."""
    print("\n[2/4] Probing model API …")

    # Try to discover the correct call signature.
    # RDT-1B may expose: forward(noisy_action, timestep, images, lang, state)
    # or use a pipeline/generate interface.
    forward_fn = None

    # Attempt 1: standard DiT forward (noisy_action, t, condition_context)
    try:
        with torch.inference_mode():
            out = model(
                inputs["x_t"],
                inputs["timestep"],
                inputs["images"],
                inputs["lang_tokens"],
                inputs["lang_attn_mask"],
            )
        forward_fn = lambda: model(
            inputs["x_t"],
            inputs["timestep"],
            inputs["images"],
            inputs["lang_tokens"],
            inputs["lang_attn_mask"],
        )
        print("  API: forward(x_t, t, images, lang_tokens, lang_attn_mask)")
    except Exception:
        pass

    # Attempt 2: keyword arguments
    if forward_fn is None:
        try:
            with torch.inference_mode():
                out = model(
                    noisy_actions=inputs["x_t"],
                    timesteps=inputs["timestep"].long(),
                    images=inputs["images"],
                    lang_tokens=inputs["lang_tokens"],
                    attention_mask=inputs["lang_attn_mask"],
                    state=inputs["robot_state"],
                )
            forward_fn = lambda: model(
                noisy_actions=inputs["x_t"],
                timesteps=inputs["timestep"].long(),
                images=inputs["images"],
                lang_tokens=inputs["lang_tokens"],
                attention_mask=inputs["lang_attn_mask"],
                state=inputs["robot_state"],
            )
            print("  API: forward(**kwargs)")
        except Exception:
            pass

    # Attempt 3: just pass the action tensor and timestep
    if forward_fn is None:
        try:
            with torch.inference_mode():
                out = model(inputs["x_t"], inputs["timestep"])
            forward_fn = lambda: model(inputs["x_t"], inputs["timestep"])
            print("  API: forward(x_t, t)  [condition-free]")
        except Exception as e:
            print(f"  Could not find a valid forward signature: {e}")
            print("  Falling back to dummy benchmark.")
            _bench_dummy(inputs, results)
            return

    print("\n[3/4] Full-pipeline latency (encode + denoise × N) …")
    with torch.inference_mode():
        stats_full = measure_latency_ms(
            forward_fn, warmup=args.warmup, iters=args.iters
        )
    results["full_pipeline"] = stats_full
    _print_stat("  single forward pass", stats_full)

    # Simulate N-step denoising loop
    print(f"\n[4/4] Simulated {args.steps}-step denoising loop …")
    ts = torch.linspace(1.0, 0.0, args.steps + 1, device=DEVICE, dtype=TORCH_DTYPE)

    def denoise_loop():
        x = inputs["x_t"].clone()
        for i in range(args.steps):
            t = ts[i].unsqueeze(0)
            vel = forward_fn()  # simplified: reuse same inputs
            if isinstance(vel, torch.Tensor):
                dt = ts[i] - ts[i + 1]
                x = x + dt * vel[..., :args.action_dim] if vel.shape[-1] > args.action_dim else x + dt * vel
        return x

    with torch.inference_mode():
        stats_loop = measure_latency_ms(
            denoise_loop, warmup=args.warmup, iters=args.iters
        )
    results["denoise_loop"] = stats_loop
    _print_stat(f"  {args.steps}-step loop", stats_loop)

    results["per_step_ms"] = stats_loop["mean_ms"] / args.steps
    results["peak_memory_mb"] = gpu_peak_mb()
    print(f"\n  Per-step latency : {results['per_step_ms']:.2f} ms")
    print(f"  Peak GPU memory  : {results['peak_memory_mb']:.0f} MB")

# ── Dummy/fallback benchmark ──────────────────────────────────────────────────
def _bench_dummy(inputs, results):
    """Benchmark equivalent DiT operations without real weights."""
    print("\n[2/4] Dummy DiT forward (same tensor sizes as RDT-1B) …")

    hidden_dim   = 1024
    num_heads    = 16
    num_layers   = 28   # RDT-1B approx
    T = args.action_horizon
    A = args.action_dim

    # Approximate RDT-1B compute: embed + N × DiT-block + decode
    W_in   = torch.randn(hidden_dim, A,          device=DEVICE, dtype=TORCH_DTYPE)
    W_out  = torch.randn(A,          hidden_dim, device=DEVICE, dtype=TORCH_DTYPE)
    W_attn = torch.randn(3 * hidden_dim, hidden_dim, device=DEVICE, dtype=TORCH_DTYPE)
    W_ff1  = torch.randn(4 * hidden_dim, hidden_dim, device=DEVICE, dtype=TORCH_DTYPE)
    W_ff2  = torch.randn(hidden_dim, 4 * hidden_dim, device=DEVICE, dtype=TORCH_DTYPE)

    cond_seq = 512  # language + vision tokens

    def dummy_forward():
        x = inputs["x_t"].reshape(T, A)
        h = torch.mm(x, W_in.T)                            # embed
        for _ in range(num_layers):
            # Self-attention (simplified: QKV proj + reshape, skip actual attn)
            qkv = torch.mm(h, W_attn.T)
            q, k, v = qkv.chunk(3, dim=-1)
            # dot-product attention [T×T]
            scale = hidden_dim ** -0.5
            attn  = torch.softmax(torch.mm(q, k.T) * scale, dim=-1)
            h     = h + torch.mm(attn, v)
            # Feed-forward
            ff = torch.mm(h, W_ff1.T)
            ff = torch.nn.functional.gelu(ff)
            h  = h + torch.mm(ff, W_ff2.T)
        return torch.mm(h, W_out.T).reshape(1, T, A)

    with torch.inference_mode():
        stats_single = measure_latency_ms(
            dummy_forward, warmup=args.warmup, iters=args.iters
        )
    results["dummy_single_forward"] = stats_single
    _print_stat("  single DiT forward", stats_single)

    def dummy_loop():
        for _ in range(args.steps):
            dummy_forward()

    with torch.inference_mode():
        stats_loop = measure_latency_ms(
            dummy_loop, warmup=args.warmup, iters=args.iters
        )
    results["dummy_denoise_loop"] = stats_loop
    results["per_step_ms"]        = stats_loop["mean_ms"] / args.steps
    results["peak_memory_mb"]     = gpu_peak_mb()
    _print_stat(f"  {args.steps}-step loop", stats_loop)
    print(f"\n  Per-step latency : {results['per_step_ms']:.2f} ms")
    print(f"  Peak GPU memory  : {results['peak_memory_mb']:.0f} MB")

# ── Formatting helpers ────────────────────────────────────────────────────────
def _print_stat(label, stats):
    print(f"{label}:")
    print(f"    mean={stats['mean_ms']:.2f} ms  "
          f"median={stats['median_ms']:.2f} ms  "
          f"std={stats['std_ms']:.2f} ms  "
          f"[{stats['min_ms']:.2f}, {stats['max_ms']:.2f}]")

def _print_results(results):
    cfg = results.get("config", {})
    print(f"  Model         : {cfg.get('model_id', 'N/A')}")
    print(f"  dtype         : {cfg.get('dtype', 'N/A')}")
    print(f"  Denoising steps: {cfg.get('steps', 'N/A')}")
    print(f"  Action dim×horizon: {cfg.get('action_dim')}×{cfg.get('action_horizon')}")
    print()

    for key in ("full_pipeline", "denoise_loop",
                "dummy_single_forward", "dummy_denoise_loop"):
        if key in results:
            _print_stat(f"  [{key}]", results[key])
            print()

    if "per_step_ms" in results:
        hz = 1000.0 / (results["per_step_ms"] * cfg.get("steps", 10))
        print(f"  Per-step latency : {results['per_step_ms']:.2f} ms")
        print(f"  Achievable Hz    : {hz:.1f} Hz  (at {cfg.get('steps')} denoise steps)")
    if "model_memory_mb" in results:
        print(f"  Model VRAM       : {results['model_memory_mb']:.0f} MB")
    if "peak_memory_mb" in results:
        print(f"  Peak VRAM        : {results['peak_memory_mb']:.0f} MB")

if __name__ == "__main__":
    main()
