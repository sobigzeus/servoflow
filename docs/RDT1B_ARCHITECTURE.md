# RDT-1B Model Architecture (HuggingFace)

This document describes the **exact architecture** of the RDT-1B model from HuggingFace (`robotics-diffusion-transformer/rdt-1b`), derived from the official config, modeling code, and repository.

---

## 1. Config Summary (config.json)

| Parameter | Value |
|-----------|-------|
| `action_dim` | 128 |
| `pred_horizon` | 64 |
| `img_cond_len` | 4374 |
| `img_token_dim` | 1152 |
| `lang_token_dim` | 4096 |
| `max_lang_cond_len` | 1024 |
| `state_token_dim` | 128 |
| **RDT backbone** | |
| `rdt.depth` | **28** |
| `rdt.hidden_size` | **2048** |
| `rdt.num_heads` | **32** |
| `rdt.cond_pos_embed_type` | multimodal |
| **Adaptors** | |
| `lang_adaptor` | mlp2x_gelu |
| `img_adaptor` | mlp2x_gelu |
| `state_adaptor` | mlp3x_gelu |
| **Position embedding configs** | |
| `img_pos_embed_config` | `[["image", [2, 3, -729]]]` |
| `lang_pos_embed_config` | `[["lang", -1024]]` |

---

## 2. DiT Block Structure (RDTBlock)

Each RDT block uses **RMSNorm** (not LayerNorm) and alternates **language** and **image** conditioning in the cross-attention:

```
RDTBlock:
  1. norm1 (RmsNorm)
  2. self-attention (timm Attention, qkv_bias=True, qk_norm=True)
  3. residual
  4. norm2 (RmsNorm)
  5. cross-attention (Q from x, KV from condition)
     - odd blocks  → condition = language
     - even blocks → condition = image
  6. residual
  7. norm3 (RmsNorm)
  8. MLP (fc1 → GELU → fc2, hidden_features = hidden_size, i.e. no expansion)
  9. residual
```

### Key details

- **No AdaLN (adaptive layer norm)**: unlike standard DiT, RDT does not inject timestep via adaptive scaling in each block. Timestep and control frequency are prepended as tokens.
- **Self-attention**: timm `Attention` with `qkv_bias=True`, `qk_norm=True`, `norm_layer=RmsNorm`, head_dim = 2048/32 = 64.
- **Cross-attention**: custom `CrossAttention` with separate Q (from x) and KV (from condition), same head dim, qk_norm.
- **MLP**: `Mlp(in_features=hidden_size, hidden_features=hidden_size, ...)` → **no expansion**. FFN is 2048 → 2048 → 2048.

---

## 3. Token Processing Flow

### 3.1 Input sequence construction

1. **State–action trajectory**  
   Input `x` is `[B, horizon+1, state_token_dim*2]`:
   - `state_token_dim*2` because state/action is concatenated with an action mask.
   - After `state_adaptor`: `[B, 1 + horizon, hidden_size]` (1 state token + 64 action tokens).

2. **Timestep and control frequency**  
   - `t_embedder(t)` → `[B, 1, 2048]`  
   - `freq_embedder(freq)` → `[B, 1, 2048]`  
   - Prepended to `x`: `x = [t, freq, state, action_1, ..., action_64]` → `[B, 3 + horizon, 2048]` = `[B, 67, 2048]`.

3. **Position embeddings**  
   `x_pos_embed` has shape `[1, 67, 2048]` for positions:
   - `timestep` (1)
   - `ctrl_freq` (1)
   - `state` (1)
   - `action` (64)

   Initialized with sin/cos multimodal conditioning via `get_multimodal_cond_pos_embed`.

4. **Condition tokens**  
   - **Language**: `lang_c` of variable length up to 1024, projected to `hidden_size` by `lang_adaptor`.
   - **Image**: `img_c` of length 4374, projected to `hidden_size` by `img_adaptor`.
   - Position embeddings:
     - `lang_cond_pos_embed`: `[1, 1024, 2048]`
     - `img_cond_pos_embed`: `[1, 4374, 2048]` with `img_pos_embed_config` grid `[2, 3, -729]` (2 frames × 3 cameras × 729 patches; last dim uses no pos embed, hence -729).

### 3.2 Forward through blocks

```python
x = x + self.x_pos_embed
lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]
img_c = img_c + self.img_cond_pos_embed

for i, block in enumerate(self.blocks):
    c, mask = (lang_c, lang_mask) if i % 2 == 0 else (img_c, img_mask)
    x = block(x, c, mask)
```

So:
- **Even-index blocks (0, 2, 4, …)**: cross-attend to **language**.
- **Odd-index blocks (1, 3, 5, …)**: cross-attend to **image**.

### 3.3 Output

- `FinalLayer` runs on full sequence `[B, 67, 2048]`.
- Only the last `horizon` tokens are kept: `x[:, -self.horizon:]` → `[B, 64, 128]` (output_dim = 128).

---

## 4. Adaptors

| Adaptor | Type | Input dim | Output dim |
|---------|------|-----------|------------|
| `lang_adaptor` | mlp2x_gelu | 4096 | 2048 |
| `img_adaptor` | mlp2x_gelu | 1152 | 2048 |
| `state_adaptor` | mlp3x_gelu | 256 (128×2) | 2048 |

- **mlp2x_gelu**: `Linear → GELU → Linear` (depth 2).
- **mlp3x_gelu**: `Linear → GELU → Linear → GELU → Linear` (depth 3).

---

## 5. Timestep Embedding

- Sinusoidal encoding: `frequency_embedding_size=256`, `max_period=10000`.
- MLP: `256 → 2048` (Linear–SiLU–Linear).
- Same structure used for **control frequency** via `freq_embedder`.

---

## 6. Weight Naming (PyTorch state_dict)

The model is saved as `pytorch_model.bin` (not safetensors). Top-level keys:

### RDTRunner (wrapper)

- `model.*` – RDT backbone
- `lang_adaptor.*`
- `img_adaptor.*`
- `state_adaptor.*`
- `noise_scheduler.*` (config, not weights)

### RDT model

```
model.t_embedder.mlp.0.weight, model.t_embedder.mlp.0.bias
model.t_embedder.mlp.2.weight, model.t_embedder.mlp.2.bias
model.freq_embedder.mlp.0.weight, ...
model.freq_embedder.mlp.2.weight, ...

model.x_pos_embed              # [1, 67, 2048]
model.lang_cond_pos_embed      # [1, 1024, 2048]
model.img_cond_pos_embed       # [1, 4374, 2048]

model.blocks.{i}.norm1.weight  # RmsNorm
model.blocks.{i}.attn.qkv.weight, model.blocks.{i}.attn.qkv.bias
model.blocks.{i}.attn.q_norm.weight, model.blocks.{i}.attn.k_norm.weight
model.blocks.{i}.attn.proj.weight, model.blocks.{i}.attn.proj.bias

model.blocks.{i}.norm2.weight
model.blocks.{i}.cross_attn.q.weight, model.blocks.{i}.cross_attn.q.bias
model.blocks.{i}.cross_attn.kv.weight, model.blocks.{i}.cross_attn.kv.bias
model.blocks.{i}.cross_attn.q_norm.weight, model.blocks.{i}.cross_attn.k_norm.weight
model.blocks.{i}.cross_attn.proj.weight, model.blocks.{i}.cross_attn.proj.bias

model.blocks.{i}.norm3.weight
model.blocks.{i}.ffn.fc1.weight, model.blocks.{i}.ffn.fc1.bias
model.blocks.{i}.ffn.fc2.weight, model.blocks.{i}.ffn.fc2.bias

model.final_layer.norm_final.weight
model.final_layer.ffn_final.fc1.weight, model.final_layer.ffn_final.fc1.bias
model.final_layer.ffn_final.fc2.weight, model.final_layer.ffn_final.fc2.bias
```

### Adaptors

```
lang_adaptor.0.weight, lang_adaptor.0.bias   # first linear
lang_adaptor.2.weight, lang_adaptor.2.bias   # second linear (mlp2x)

img_adaptor.0.weight, img_adaptor.0.bias
img_adaptor.2.weight, img_adaptor.2.bias

state_adaptor.0.weight, state_adaptor.0.bias
state_adaptor.2.weight, state_adaptor.2.bias
state_adaptor.4.weight, state_adaptor.4.bias  # mlp3x
```

---

## 7. File Layout on HuggingFace

| File | Size | Description |
|------|------|-------------|
| `config.json` | 919 B | Architecture and training config |
| `pytorch_model.bin` | ~2.46 GB | Full model weights (no safetensors) |
| `README.md` | 5.5 KB | Usage and citation |

There is **no** `model.safetensors.index.json`; the hub provides `pytorch_model.bin` only.

---

## 8. Parameter Count (diffusion core)

Reported in training as “Diffusion params” = RDT + adaptors:

- **RDT**: ~1.0B parameters (depth 28, hidden 2048, 32 heads).
- **Adaptors**: lang_adaptor, img_adaptor, state_adaptor (order of tens of millions).
- **Encoders** (SigLIP, T5) are **frozen** and not part of “diffusion params”.

---

## 9. Diffusion / Noise Scheduler

- **Type**: DDPM
- **Train timesteps**: 1000
- **Inference timesteps**: 5 (DPMSolverMultistep)
- **Schedule**: squaredcos_cap_v2
- **Prediction type**: sample (predict `x_0`, not noise)

---

## 10. References

- Config: `https://huggingface.co/robotics-diffusion-transformer/rdt-1b/resolve/main/config.json`
- Code: `https://github.com/thu-ml/RoboticsDiffusionTransformer`
- Core modules: `models/rdt/model.py`, `models/rdt/blocks.py`, `models/rdt_runner.py`
- Paper: https://arxiv.org/abs/2410.07864
