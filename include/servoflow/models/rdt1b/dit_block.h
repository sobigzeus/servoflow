// SPDX-License-Identifier: Apache-2.0
#pragma once

// RDT-1B transformer block (RDTBlock).
//
// Architecture (from https://github.com/thu-ml/RoboticsDiffusionTransformer):
//   norm1 (RmsNorm, eps=1e-6)
//   → self-attn (timm Attention: qkv_bias=True, qk_norm=True via RmsNorm(head_dim))
//   → residual
//   norm2 (RmsNorm)
//   → cross-attn (CrossAttention: q from x, kv from condition;
//                 qkv_bias=True, qk_norm=True; alternating: even=lang, odd=img)
//   → residual
//   norm3 (RmsNorm)
//   → MLP (timm Mlp: fc1 D→D + GELU(tanh) + fc2 D→D)
//   → residual
//
// FinalLayer: norm_final (RmsNorm) + ffn_final (Mlp D→D→action_dim)
//
// TimestepEmbedding: sinusoidal(256) → Linear(256→D) → SiLU → Linear(D→D)
//   Produces a single [B, D] token that is prepended to x as a position token.

#include "servoflow/models/rdt1b/config.h"
#include "servoflow/backend/backend.h"
#include "servoflow/core/tensor.h"
#include <string>
#include <unordered_map>

namespace sf {
namespace rdt1b {

using WeightMap = std::unordered_map<std::string, Tensor>;

// ─────────────────────────────────────────────────────────────────────────────
// TimestepEmbedding
//
// Embeds a scalar integer timestep t ∈ [0, num_train_timesteps) into a
// dense vector: sinusoidal(t, 256) → Linear+SiLU → Linear → [1, hidden_dim]
// ─────────────────────────────────────────────────────────────────────────────
class TimestepEmbedding {
public:
    TimestepEmbedding() = default;

    // prefix: e.g. "t_embedder." or "freq_embedder."
    // Weight names: prefix + "mlp.0.weight/bias", "mlp.2.weight/bias"
    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // t: integer diffusion timestep (0 .. num_train_timesteps-1)
    // Returns [1, hidden_dim]
    Tensor forward(int64_t t, BackendPtr backend, StreamHandle stream) const;

private:
    Tensor mlp0_weight_;   // [hidden_dim, freq_dim]
    Tensor mlp0_bias_;     // [hidden_dim]
    Tensor mlp2_weight_;   // [hidden_dim, hidden_dim]
    Tensor mlp2_bias_;     // [hidden_dim]

    // Pre-computed sinusoidal table: [num_train_timesteps, freq_dim] on device.
    Tensor sincos_table_;

    int64_t freq_dim_      = 256;
    int64_t embed_dim_     = 2048;
    int64_t max_timesteps_ = 1000;

    void build_sincos_table(BackendPtr backend, StreamHandle stream);
};

// ─────────────────────────────────────────────────────────────────────────────
// MLP — timm's Mlp with 1× hidden (no expansion)
//
// forward(x): fc1(x) → GELU(tanh) → fc2(x)
// Weight names: prefix + "fc1.weight/bias", "fc2.weight/bias"
// ─────────────────────────────────────────────────────────────────────────────
class Mlp {
public:
    Mlp() = default;

    // in_dim: input/hidden dim; out_dim: output dim (= in_dim for blocks,
    //         = action_dim for final layer's ffn_final)
    void load(const WeightMap& weights, const std::string& prefix,
              int64_t in_dim, int64_t out_dim, DType dt,
              BackendPtr backend, StreamHandle stream);

    // x: [B, S, in_dim] → out: [B, S, out_dim]
    Tensor forward(const Tensor& x, BackendPtr backend, StreamHandle stream) const;

private:
    Tensor fc1_weight_;   // [in_dim, in_dim]
    Tensor fc1_bias_;     // [in_dim]
    Tensor fc2_weight_;   // [out_dim, in_dim]
    Tensor fc2_bias_;     // [out_dim]
    int64_t in_dim_  = 2048;
    int64_t out_dim_ = 2048;
};

// ─────────────────────────────────────────────────────────────────────────────
// SelfAttention — timm's Attention module
//
// qkv = Linear(x, 3D, bias=True)  → split into q, k, v [B, H, S, head_dim]
// q_norm, k_norm = RmsNorm(head_dim) on q and k (per-head normalisation)
// out = scaled_dot_product_attention(q, k, v)  (FlashAttention)
// return proj(reshape(out))
//
// Weight names: prefix + "qkv.weight/bias", "q_norm.weight", "k_norm.weight",
//               "proj.weight/bias"
// ─────────────────────────────────────────────────────────────────────────────
class SelfAttention {
public:
    SelfAttention() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // x: [B, S, D] → out: [B, S, D]
    Tensor forward(const Tensor& x, BackendPtr backend, StreamHandle stream) const;

private:
    Tensor qkv_weight_;      // [3*D, D]
    Tensor qkv_bias_;        // [3*D]
    Tensor q_norm_weight_;   // [head_dim]  RmsNorm weight
    Tensor k_norm_weight_;   // [head_dim]
    Tensor proj_weight_;     // [D, D]
    Tensor proj_bias_;       // [D]

    int64_t hidden_dim_ = 2048;
    int64_t num_heads_  = 32;
    int64_t head_dim_   = 64;
    float   norm_eps_   = 1e-6f;

    // Apply per-head RmsNorm to Q or K of shape [B, H, S, head_dim].
    void apply_qk_norm(Tensor& qk, const Tensor& weight,
                       BackendPtr backend, StreamHandle stream) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// CrossAttention
//
// q from x [B, Sq, D], kv from condition c [B, Sk, D]
// Same qk_norm as SelfAttention.
// Optional key-padding mask [B, Sk] (True = keep, False = mask out).
//
// Weight names: prefix + "q.weight/bias", "kv.weight/bias",
//               "q_norm.weight", "k_norm.weight", "proj.weight/bias"
// ─────────────────────────────────────────────────────────────────────────────
class CrossAttention {
public:
    CrossAttention() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // x: [B, Sq, D]  c: [B, Sk, D]  mask: nullptr or [B, Sk] bool (host)
    // Returns [B, Sq, D]
    Tensor forward(const Tensor& x, const Tensor& c,
                   BackendPtr backend, StreamHandle stream) const;

private:
    Tensor q_weight_;        // [D, D]
    Tensor q_bias_;          // [D]
    Tensor kv_weight_;       // [2D, D]
    Tensor kv_bias_;         // [2D]
    Tensor q_norm_weight_;   // [head_dim]
    Tensor k_norm_weight_;   // [head_dim]
    Tensor proj_weight_;     // [D, D]
    Tensor proj_bias_;       // [D]

    int64_t hidden_dim_ = 2048;
    int64_t num_heads_  = 32;
    int64_t head_dim_   = 64;
    float   norm_eps_   = 1e-6f;

    void apply_qk_norm(Tensor& qk, const Tensor& weight,
                       BackendPtr backend, StreamHandle stream) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// RDTBlock — one transformer block
//
// forward(x, cond, block_idx):
//   x = x + self_attn(norm1(x))
//   c = lang_cond if block_idx % 2 == 0 else img_cond
//   x = x + cross_attn(norm2(x), c)
//   x = x + mlp(norm3(x))
//   return x
// ─────────────────────────────────────────────────────────────────────────────
class RDTBlock {
public:
    RDTBlock() = default;

    // prefix: e.g. "blocks.0."
    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // x: [B, S, D]
    // lang_cond: [B, L, D]   img_cond: [B, I, D]
    // block_idx: used to alternate cross-attn condition (even=lang, odd=img)
    Tensor forward(const Tensor& x,
                   const Tensor& lang_cond, const Tensor& img_cond,
                   int block_idx,
                   BackendPtr backend, StreamHandle stream) const;

private:
    // 3 RmsNorm layers (weight only, no bias in RmsNorm)
    Tensor norm1_weight_;  // [D]
    Tensor norm2_weight_;  // [D]
    Tensor norm3_weight_;  // [D]

    SelfAttention  self_attn_;
    CrossAttention cross_attn_;
    Mlp            ffn_;

    int64_t hidden_dim_ = 2048;
    float   norm_eps_   = 1e-6f;
};

// ─────────────────────────────────────────────────────────────────────────────
// FinalLayer
//
// norm_final (RmsNorm) + ffn_final (Mlp: D→D→action_dim)
// Applied to all tokens; caller slices last action_horizon tokens.
// ─────────────────────────────────────────────────────────────────────────────
class FinalLayer {
public:
    FinalLayer() = default;

    // prefix: "final_layer."
    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // x: [B, S, D] → out: [B, S, action_dim]
    Tensor forward(const Tensor& x, BackendPtr backend, StreamHandle stream) const;

private:
    Tensor norm_final_weight_;   // [D]  RmsNorm
    Mlp    ffn_final_;           // D→D→action_dim

    int64_t hidden_dim_ = 2048;
    float   norm_eps_   = 1e-6f;
};

}  // namespace rdt1b
}  // namespace sf
