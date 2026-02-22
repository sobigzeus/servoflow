#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
hf_to_servoflow.py — Convert a HuggingFace RDT-1B checkpoint to ServoFlow format.

What this script does:
  1. Loads the HuggingFace safetensors checkpoint (model.safetensors or shards).
  2. Remaps weight keys from HF naming conventions to ServoFlow naming conventions.
  3. Optionally quantises weights (fp32 → fp16 / bf16).
  4. Saves action normalisation statistics (mean / std).
  5. Writes a ServoFlow-compatible config.json.
  6. Outputs one or more .safetensors files in the target directory.

Usage:
  python tools/convert/hf_to_servoflow.py \\
      --input  /path/to/hf_rdt1b_checkpoint \\
      --output /path/to/servoflow_checkpoint \\
      --dtype  fp16

Requirements:
  pip install safetensors torch huggingface_hub
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file


# ─────────────────────────────────────────────────────────────────────────────
# Key remapping: HuggingFace → ServoFlow naming convention.
#
# ServoFlow uses a flat naming scheme that maps directly to C++ struct members.
# HuggingFace checkpoints typically use PyTorch module hierarchy naming.
# ─────────────────────────────────────────────────────────────────────────────
def build_key_map(hf_keys: list[str]) -> dict[str, str]:
    """Build a HF-key → ServoFlow-key mapping for known patterns."""
    mapping = {}

    for k in hf_keys:
        sf_key = remap_key(k)
        if sf_key is not None:
            mapping[k] = sf_key

    return mapping


def remap_key(k: str) -> Optional[str]:
    """
    Map one HuggingFace tensor name to its ServoFlow equivalent.
    Returns None if the key should be skipped (e.g. T5 encoder weights).
    """
    # Skip T5 language encoder (not bundled in ServoFlow Phase 1).
    if k.startswith("language_model.") or k.startswith("t5_encoder."):
        return None

    # Skip optimizer states if accidentally included.
    if "optimizer" in k or "scheduler" in k:
        return None

    # ── Timestep embedding ────────────────────────────────────────────────
    if k.startswith("time_embed."):
        return k  # same naming

    # ── Vision projection ─────────────────────────────────────────────────
    if k.startswith("vision_proj."):
        return k

    # ── Language projection ───────────────────────────────────────────────
    if k.startswith("lang_proj."):
        return k

    # ── Action input projection ───────────────────────────────────────────
    if k in ("action_proj.weight", "action_proj.bias",
             "action_in.weight",   "action_in.bias"):
        return k.replace("action_proj.", "action_in_proj.") \
                 .replace("action_in.", "action_in_proj.")

    # ── State tokeniser ───────────────────────────────────────────────────
    if k.startswith("state_tok.") or k.startswith("state_token."):
        return k.replace("state_token.", "state_tok.")

    # ── DiT blocks ────────────────────────────────────────────────────────
    # HF: model.blocks.{i}.{...}  →  SF: dit.blocks.{i}.{...}
    if k.startswith("model.blocks.") or k.startswith("dit_model.blocks."):
        return k.replace("model.blocks.", "dit.blocks.") \
                 .replace("dit_model.blocks.", "dit.blocks.")

    # Already in SF format.
    if k.startswith("dit.blocks."):
        return k

    # ── Final layer ───────────────────────────────────────────────────────
    if k.startswith("final_layer."):
        return k

    # ── Action normalisation ──────────────────────────────────────────────
    if k.startswith("action_norm."):
        return k

    # Unknown key — keep as-is and let the loader warn about unrecognised weights.
    return k


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────
DTYPE_MAP = {
    "fp32":    torch.float32,
    "fp16":    torch.float16,
    "bf16":    torch.bfloat16,
}


def load_hf_checkpoint(input_dir: Path) -> Dict[str, torch.Tensor]:
    """Load all safetensors shards from a HuggingFace checkpoint directory."""
    shards = sorted(input_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    tensors = {}
    for shard in shards:
        print(f"  Loading shard: {shard.name}")
        tensors.update(load_file(str(shard), device="cpu"))

    print(f"  Loaded {len(tensors)} tensors from {len(shards)} shard(s).")
    return tensors


def convert(input_dir: Path, output_dir: Path, dtype: torch.dtype,
            max_shard_gb: float = 4.0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading HuggingFace checkpoint from: {input_dir}")
    hf_tensors = load_hf_checkpoint(input_dir)

    print(f"\n[2/4] Remapping weight keys...")
    key_map = build_key_map(list(hf_tensors.keys()))
    skipped = [k for k in hf_tensors if k not in key_map]
    if skipped:
        print(f"  Skipped {len(skipped)} tensors (T5 encoder / unknown):")
        for k in skipped[:5]:
            print(f"    {k}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    print(f"\n[3/4] Casting to {dtype} and saving...")
    sf_tensors: Dict[str, torch.Tensor] = {}
    for hf_key, sf_key in key_map.items():
        t = hf_tensors[hf_key]
        # Keep action normalisation in fp32 for numerical precision.
        target = torch.float32 if sf_key.startswith("action_norm.") else dtype
        sf_tensors[sf_key] = t.to(target).contiguous()

    # Shard output if total size exceeds max_shard_gb.
    total_bytes = sum(t.nbytes() for t in sf_tensors.values())
    max_bytes   = int(max_shard_gb * 1024 ** 3)

    if total_bytes <= max_bytes:
        out_path = output_dir / "model.safetensors"
        print(f"  Writing {out_path.name}  ({total_bytes / 1024**3:.2f} GB)")
        save_file(sf_tensors, str(out_path))
    else:
        # Split into shards.
        shard_idx   = 1
        current     = {}
        current_sz  = 0

        for name, tensor in sf_tensors.items():
            if current_sz + tensor.nbytes() > max_bytes and current:
                fname = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                print(f"  Writing shard {shard_idx}  ({current_sz / 1024**3:.2f} GB)")
                save_file(current, str(fname))
                shard_idx += 1
                current   = {}
                current_sz = 0
            current[name]  = tensor
            current_sz    += tensor.nbytes()

        if current:
            fname = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            save_file(current, str(fname))

        # Rename shards to include total count.
        total_shards = shard_idx
        for i, old in enumerate(
            sorted(output_dir.glob("model-*-of-XXXXX.safetensors")), 1
        ):
            new = output_dir / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            old.rename(new)

    print(f"\n[4/4] Writing config.json...")
    _write_config(input_dir, output_dir, dtype)

    print(f"\nDone. ServoFlow checkpoint written to: {output_dir}")
    print(f"Total size: {total_bytes / 1024**3:.2f} GB  ({dtype})")


def _write_config(input_dir: Path, output_dir: Path, dtype: torch.dtype) -> None:
    """Write ServoFlow config.json, adapting from HF config.json if available."""
    hf_cfg_path = input_dir / "config.json"
    sf_cfg: dict = {}

    if hf_cfg_path.exists():
        with open(hf_cfg_path) as f:
            hf_cfg = json.load(f)
        # Translate common HF field names.
        field_map = {
            "hidden_size":           "hidden_size",
            "num_hidden_layers":     "num_hidden_layers",
            "num_attention_heads":   "num_attention_heads",
            "intermediate_size":     None,  # we use mlp_ratio
            "action_dim":            "action_dim",
            "action_horizon":        "action_horizon",
            "num_cameras":           "num_cameras",
            "lang_max_tokens":       "lang_max_tokens",
            "vision_embed_dim":      "vision_embed_dim",
            "num_image_tokens":      "num_image_tokens",
        }
        for hf_k, sf_k in field_map.items():
            if sf_k and hf_k in hf_cfg:
                sf_cfg[sf_k] = hf_cfg[hf_k]

    # Add ServoFlow-specific fields.
    sf_cfg.setdefault("hidden_size",         2048)
    sf_cfg.setdefault("num_hidden_layers",   24)
    sf_cfg.setdefault("num_attention_heads", 32)
    sf_cfg.setdefault("mlp_ratio",           4.0)
    sf_cfg.setdefault("action_dim",          128)
    sf_cfg.setdefault("action_horizon",      64)
    sf_cfg.setdefault("freq_dim",            256)
    sf_cfg.setdefault("vision_embed_dim",    1152)
    sf_cfg.setdefault("num_image_tokens",    729)
    sf_cfg.setdefault("num_cameras",         2)
    sf_cfg.setdefault("lang_max_tokens",     128)
    sf_cfg.setdefault("state_dim",           128)
    sf_cfg.setdefault("layer_norm_eps",      1e-6)
    sf_cfg["servoflow_version"] = "0.1.0"
    sf_cfg["compute_dtype"] = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]

    with open(output_dir / "config.json", "w") as f:
        json.dump(sf_cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace RDT-1B checkpoint to ServoFlow format."
    )
    parser.add_argument("--input",  required=True,
                        help="Path to the HuggingFace checkpoint directory.")
    parser.add_argument("--output", required=True,
                        help="Path to the output ServoFlow checkpoint directory.")
    parser.add_argument("--dtype",  default="fp16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Target weight dtype (default: fp16).")
    parser.add_argument("--max-shard-gb", type=float, default=4.0,
                        help="Maximum size of each output shard in GB (default: 4).")
    args = parser.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    convert(Path(args.input), Path(args.output), dtype, args.max_shard_gb)


if __name__ == "__main__":
    main()
