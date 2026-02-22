// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/models/rdt1b/config.h"
#include "servoflow/backend/backend.h"
#include "servoflow/core/tensor.h"
#include <string>
#include <unordered_map>

namespace sf {
namespace rdt1b {

// Alias for the weight map loaded from a checkpoint.
using WeightMap = std::unordered_map<std::string, Tensor>;

// ─────────────────────────────────────────────────────────────────────────────
// TimestepEmbedding
//
// Converts a scalar timestep t ∈ [0,1] into a dense embedding vector.
// Architecture (standard for DiT / RDT):
//   1. Sinusoidal positional encoding: t → R^{freq_dim}
//   2. Linear + SiLU → R^{time_embed_dim}
//   3. Linear         → R^{time_embed_dim}
// ─────────────────────────────────────────────────────────────────────────────
class TimestepEmbedding {
public:
    TimestepEmbedding() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // Input: t (scalar, host-side float)
    // Output: embedding [1, time_embed_dim]
    Tensor forward(float t, BackendPtr backend, StreamHandle stream) const;

private:
    // Sinusoidal encoding lookup table (pre-computed at load time).
    // Shape: [max_timesteps, freq_dim], stored on device.
    Tensor sincos_table_;   // [1000, freq_dim]

    // MLP weights.
    Tensor linear1_weight_;  // [time_embed_dim, freq_dim]
    Tensor linear1_bias_;    // [time_embed_dim]
    Tensor linear2_weight_;  // [time_embed_dim, time_embed_dim]
    Tensor linear2_bias_;    // [time_embed_dim]

    int64_t freq_dim_      = 256;
    int64_t embed_dim_     = 2048;

    void build_sincos_table(BackendPtr backend, StreamHandle stream);
};

// ─────────────────────────────────────────────────────────────────────────────
// AdaLNModulation
//
// Adaptive Layer Norm zero (adaLN-zero): given a conditioning vector c,
// produce scale and shift for a layer norm output.
//
// c → SiLU → Linear → [scale, shift]  (both R^{hidden_dim})
//
// Applied as: out = (1 + scale) * LayerNorm(x) + shift
// The "(1 + ...)" trick initialises the block as identity at t=0.
// ─────────────────────────────────────────────────────────────────────────────
class AdaLNModulation {
public:
    AdaLNModulation() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // Returns [scale, shift], each [B, hidden_dim].
    // c: condition vector [B, time_embed_dim]
    struct Params { Tensor scale; Tensor shift; };
    Params forward(const Tensor& c, BackendPtr backend,
                   StreamHandle stream) const;

private:
    Tensor linear_weight_;  // [2 * hidden_dim, time_embed_dim]
    Tensor linear_bias_;    // [2 * hidden_dim]
    int64_t hidden_dim_ = 2048;
};

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward (SwiGLU variant used in RDT-1B)
//
// x → [gate, up] = Linear(x, 2*ffn_dim)
// out = down(SiLU(gate) * up)
//
// This is the gated linear unit (GLU) variant, which outperforms plain GELU
// FFN in practice and is used in modern transformer architectures.
// ─────────────────────────────────────────────────────────────────────────────
class FeedForward {
public:
    FeedForward() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, BackendPtr backend, StreamHandle stream);

    // x: [B, S, hidden_dim] → out: [B, S, hidden_dim]
    void forward(const Tensor& x, Tensor& out,
                 BackendPtr backend, StreamHandle stream) const;

private:
    Tensor gate_up_weight_;  // [2 * ffn_dim, hidden_dim]
    Tensor gate_up_bias_;    // [2 * ffn_dim]
    Tensor down_weight_;     // [hidden_dim, ffn_dim]
    Tensor down_bias_;       // [hidden_dim]

    int64_t hidden_dim_ = 2048;
    int64_t ffn_dim_    = 8192;
};

// ─────────────────────────────────────────────────────────────────────────────
// MultiHeadAttention
//
// Supports both self-attention and cross-attention.
// Uses FlashAttention on CUDA backend when inputs are fp16/bf16.
// ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention {
public:
    MultiHeadAttention() = default;

    // prefix: e.g. "blocks.0.attn."
    // if cross_attn=true, keys/values come from a separate context sequence.
    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg, bool cross_attn,
              BackendPtr backend, StreamHandle stream);

    // Self-attention: x → x   (x: [B, S, D])
    void forward(const Tensor& x, Tensor& out,
                 bool is_causal,
                 BackendPtr backend, StreamHandle stream) const;

    // Cross-attention: query from x, key/value from context.
    // x:       [B, Sq, D]
    // context: [B, Sk, D]
    void forward_cross(const Tensor& x, const Tensor& context,
                       Tensor& out,
                       BackendPtr backend, StreamHandle stream) const;

private:
    // For self-attention: fused Q,K,V projection [3*hidden, hidden].
    Tensor qkv_weight_;   // [3 * hidden_dim, hidden_dim]
    Tensor qkv_bias_;     // [3 * hidden_dim]

    // For cross-attention: separate Q vs K,V projections.
    Tensor q_weight_;     // [hidden_dim, hidden_dim]
    Tensor q_bias_;
    Tensor kv_weight_;    // [2 * hidden_dim, hidden_dim]
    Tensor kv_bias_;

    Tensor out_weight_;   // [hidden_dim, hidden_dim]
    Tensor out_bias_;     // [hidden_dim]

    int64_t hidden_dim_ = 2048;
    int64_t num_heads_  = 32;
    int64_t head_dim_   = 64;
    bool    cross_attn_ = false;

    // Splits [B, S, 3D] into three [B, H, S, head_dim] tensors.
    void split_qkv(const Tensor& qkv,
                   Tensor& Q, Tensor& K, Tensor& V,
                   BackendPtr backend, StreamHandle stream) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// DiTBlock: one transformer block with adaLN-zero conditioning.
//
// Forward pass:
//   1. adaLN modulation from timestep embedding c.
//   2. Self-attention with pre-norm (applied on full sequence).
//   3. Residual add.
//   4. FFN with pre-norm.
//   5. Residual add.
//
// The action token subsequence uses causal masking within itself;
// condition tokens (image, language) are fully visible to all tokens.
// ─────────────────────────────────────────────────────────────────────────────
class DiTBlock {
public:
    DiTBlock() = default;

    // prefix: e.g. "dit.blocks.0."
    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg,
              BackendPtr backend, StreamHandle stream);

    // x: [B, S_total, hidden_dim]
    // c: timestep+condition embedding [B, time_embed_dim]
    // cond_len: number of condition tokens (first cond_len tokens are condition)
    // out: [B, S_total, hidden_dim]
    void forward(const Tensor& x, const Tensor& c,
                 int64_t cond_len, Tensor& out,
                 BackendPtr backend, StreamHandle stream) const;

private:
    AdaLNModulation  adaln_;
    MultiHeadAttention attn_;
    FeedForward        ffn_;

    // Pre-norm layer norms.
    Tensor norm1_weight_;  // [hidden_dim]
    Tensor norm1_bias_;
    Tensor norm2_weight_;  // [hidden_dim]
    Tensor norm2_bias_;

    int64_t hidden_dim_ = 2048;
    float   norm_eps_   = 1e-6f;
};

// ─────────────────────────────────────────────────────────────────────────────
// FinalLayer
//
// After all DiT blocks, project from hidden_dim back to action_dim.
// Also uses adaLN-zero conditioning.
// ─────────────────────────────────────────────────────────────────────────────
class FinalLayer {
public:
    FinalLayer() = default;

    void load(const WeightMap& weights, const std::string& prefix,
              const RDT1BConfig& cfg,
              BackendPtr backend, StreamHandle stream);

    // x: [B, S_total, hidden_dim] → out: [B, T_action, action_dim]
    // Only the action token subsequence (last action_horizon tokens) is output.
    void forward(const Tensor& x, const Tensor& c,
                 int64_t action_horizon, Tensor& out,
                 BackendPtr backend, StreamHandle stream) const;

private:
    AdaLNModulation adaln_;

    Tensor norm_weight_;   // [hidden_dim]
    Tensor norm_bias_;
    Tensor linear_weight_; // [action_dim, hidden_dim]
    Tensor linear_bias_;   // [action_dim]

    int64_t hidden_dim_ = 2048;
    float   norm_eps_   = 1e-6f;
};

}  // namespace rdt1b
}  // namespace sf
