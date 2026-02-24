#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
hf_to_servoflow.py — Convert an HuggingFace RDT-1B checkpoint to ServoFlow format.

The HuggingFace checkpoint is RDTRunner, which contains:
  model.*           → RDT backbone (blocks, pos_embeds, t/freq_embedder)
  lang_adaptor.*    → mlp2x_gelu for language tokens
  img_adaptor.*     → mlp2x_gelu for image tokens
  state_adaptor.*   → mlp3x_gelu for state+action tokens

ServoFlow stripping convention:
  model.*            → strip "model." prefix
  lang_adaptor.*     → keep as-is
  img_adaptor.*      → keep as-is
  state_adaptor.*    → keep as-is

Usage:
  python tools/convert/hf_to_servoflow.py \\
      --input  robotics-diffusion-transformer/rdt-1b  (HF hub id or local path)
      --output /path/to/servoflow_checkpoint
      --dtype  fp16

Requirements:
  pip install safetensors torch huggingface_hub
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Weight key remapping
# ─────────────────────────────────────────────────────────────────────────────
def remap_key(k: str) -> Optional[str]:
    """Map one HuggingFace weight name to its ServoFlow equivalent.
    Returns None if the key should be skipped.
    """
    # Skip EMA shadow parameters (not needed for inference).
    if ".shadow_params" in k or k.startswith("ema_model."):
        return None
    # Skip optimizer / scheduler states.
    if "optimizer" in k or "scheduler" in k:
        return None

    # ── RDT backbone: strip "model." prefix ──────────────────────────────
    if k.startswith("model."):
        return k[len("model."):]  # e.g. "model.blocks.0.norm1.weight" → "blocks.0.norm1.weight"

    # ── Adaptors: keep as-is ──────────────────────────────────────────────
    if k.startswith(("lang_adaptor.", "img_adaptor.", "state_adaptor.")):
        return k

    # ── Action normalisation stats ────────────────────────────────────────
    if k.startswith("action_norm."):
        return k

    # Skip anything else (vision encoder if present, etc.)
    return None


def remap_all(hf_keys: list) -> dict:
    mapping = {}
    for k in hf_keys:
        sf_key = remap_key(k)
        if sf_key is not None:
            mapping[k] = sf_key
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (supports safetensors and pytorch_model.bin)
# ─────────────────────────────────────────────────────────────────────────────
DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def load_hf_checkpoint(input_dir: Path) -> Dict[str, torch.Tensor]:
    """Load tensors from safetensors shards or pytorch_model.bin."""
    from safetensors.torch import load_file

    # Try safetensors first.
    shards = sorted(input_dir.glob("*.safetensors"))
    if shards:
        tensors = {}
        for shard in shards:
            print(f"  Loading {shard.name}")
            tensors.update(load_file(str(shard), device="cpu"))
        print(f"  Loaded {len(tensors)} tensors from {len(shards)} safetensors shard(s).")
        return tensors

    # Fallback: pytorch_model.bin (single or sharded).
    bin_files = sorted(input_dir.glob("pytorch_model*.bin"))
    if bin_files:
        tensors = {}
        for bf in bin_files:
            print(f"  Loading {bf.name}")
            tensors.update(torch.load(str(bf), map_location="cpu"))
        print(f"  Loaded {len(tensors)} tensors from {len(bin_files)} .bin file(s).")
        return tensors

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model*.bin files found in {input_dir}"
    )


def download_hf_model(model_id: str, cache_dir: Optional[str] = None) -> Path:
    """Download a HuggingFace model to a local directory and return the path."""
    from huggingface_hub import snapshot_download

    print(f"  Downloading {model_id} from HuggingFace Hub...")
    local_dir = snapshot_download(
        model_id,
        local_dir=cache_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*",
                         "tf_model*", "rust_model*"],
    )
    return Path(local_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────
def convert(input_path: str, output_dir: Path, dtype: torch.dtype,
            max_shard_gb: float = 4.0) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input ─────────────────────────────────────────────────────
    input_dir = Path(input_path)
    if not input_dir.is_dir():
        # Assume it's a HuggingFace hub model id.
        print(f"\n[0/5] Downloading model from HuggingFace Hub: {input_path}")
        input_dir = download_hf_model(input_path)

    print(f"\n[1/5] Loading HuggingFace checkpoint from: {input_dir}")
    hf_tensors = load_hf_checkpoint(input_dir)

    print(f"\n[2/5] Remapping weight keys...")
    key_map = remap_all(list(hf_tensors.keys()))
    skipped = [k for k in hf_tensors if k not in key_map]
    print(f"  Mapped:  {len(key_map)} tensors")
    print(f"  Skipped: {len(skipped)} tensors")
    for k in skipped[:5]:
        print(f"    skip: {k}")
    if len(skipped) > 5:
        print(f"    ... and {len(skipped) - 5} more")

    # Print a sample of the mapped keys for verification.
    print("\n  Sample key mappings:")
    for hf_k, sf_k in list(key_map.items())[:10]:
        print(f"    {hf_k:60s} → {sf_k}")

    print(f"\n[3/5] Casting to {dtype} ...")
    sf_tensors: Dict[str, torch.Tensor] = {}
    for hf_key, sf_key in key_map.items():
        t = hf_tensors[hf_key]
        # Keep action normalisation statistics in fp32 for precision.
        target = torch.float32 if sf_key.startswith("action_norm.") else dtype
        sf_tensors[sf_key] = t.to(target).contiguous()

    print(f"\n[4/5] Writing safetensors checkpoint(s)...")
    total_bytes = sum(t.nbytes for t in sf_tensors.values())
    max_bytes   = int(max_shard_gb * 1024 ** 3)

    if total_bytes <= max_bytes:
        out_path = output_dir / "model.safetensors"
        print(f"  Writing {out_path.name}  ({total_bytes / 1024**3:.2f} GB)")
        save_file(sf_tensors, str(out_path))
    else:
        shard_idx  = 1
        current    = {}
        current_sz = 0
        for name, tensor in sf_tensors.items():
            if current_sz + tensor.nbytes > max_bytes and current:
                fname = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                print(f"  Writing shard {shard_idx}  ({current_sz / 1024**3:.2f} GB)")
                save_file(current, str(fname))
                shard_idx += 1
                current    = {}
                current_sz = 0
            current[name]  = tensor
            current_sz    += tensor.nbytes
        if current:
            fname = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current, str(fname))
        total_shards = shard_idx
        for i, old in enumerate(
            sorted(output_dir.glob("model-*-of-XXXXX.safetensors")), 1
        ):
            new = output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old.rename(new)

    print(f"\n[5/5] Writing config.json...")
    write_servoflow_config(input_dir, output_dir, dtype)

    print(f"\nDone. ServoFlow checkpoint written to: {output_dir}")
    print(f"Total weight size: {total_bytes / 1024**3:.2f} GB  ({dtype})")
    print("\nVerification keys present:")
    required = [
        "t_embedder.mlp.0.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.cross_attn.q.weight",
        "blocks.0.ffn.fc1.weight",
        "final_layer.norm_final.weight",
        "lang_adaptor.0.weight",
        "x_pos_embed",
    ]
    for k in required:
        present = k in sf_tensors
        shape   = tuple(sf_tensors[k].shape) if present else "MISSING"
        print(f"  {'✓' if present else '✗'} {k}: {shape}")


def write_servoflow_config(input_dir: Path, output_dir: Path,
                            dtype: torch.dtype) -> None:
    """Write ServoFlow config.json, adapting from HuggingFace config.json."""
    hf_cfg = {}
    hf_cfg_path = input_dir / "config.json"
    if hf_cfg_path.exists():
        with open(hf_cfg_path) as f:
            hf_cfg = json.load(f)

    # Build ServoFlow config from HF fields.
    rdt = hf_cfg.get("rdt", {})
    ns  = hf_cfg.get("noise_scheduler", {})

    sf_cfg = {
        # RDT backbone.
        "hidden_size":          rdt.get("hidden_size",   2048),
        "num_hidden_layers":    rdt.get("depth",         28),
        "num_attention_heads":  rdt.get("num_heads",     32),
        # Architecture: no mlp expansion in RDT-1B (1× hidden).
        "mlp_ratio":            1.0,
        # Action / robot.
        "action_dim":           hf_cfg.get("action_dim",    128),
        "pred_horizon":         hf_cfg.get("pred_horizon",  64),
        # Condition tokens.
        "img_cond_len":         hf_cfg.get("img_cond_len",  4374),
        "img_token_dim":        hf_cfg.get("img_token_dim", 1152),
        "lang_token_dim":       hf_cfg.get("lang_token_dim", 4096),
        "max_lang_cond_len":    hf_cfg.get("max_lang_cond_len", 1024),
        "state_token_dim":      hf_cfg.get("state_token_dim",   128),
        # DDPM.
        "num_train_timesteps":     ns.get("num_train_timesteps",     1000),
        "num_inference_timesteps": ns.get("num_inference_timesteps", 5),
        # Normalisation.
        "rms_norm_eps":         1e-6,
        "freq_dim":             256,
        # Compute.
        "compute_dtype": {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }[dtype],
        "servoflow_version": "0.2.0",
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(sf_cfg, f, indent=2)
    print(f"  Wrote config.json: depth={sf_cfg['num_hidden_layers']}, "
          f"hidden={sf_cfg['hidden_size']}, heads={sf_cfg['num_attention_heads']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace RDT-1B checkpoint to ServoFlow format."
    )
    parser.add_argument(
        "--input", required=True,
        help="Local HF checkpoint directory or HuggingFace model id "
             "(e.g. 'robotics-diffusion-transformer/rdt-1b')."
    )
    parser.add_argument("--output", required=True,
                        help="Output ServoFlow checkpoint directory.")
    parser.add_argument("--dtype", default="fp16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--max-shard-gb", type=float, default=4.0)
    args = parser.parse_args()

    convert(args.input, Path(args.output), DTYPE_MAP[args.dtype], args.max_shard_gb)


if __name__ == "__main__":
    main()
