// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/rdt1b/dit_block.h"
#include "backend/cuda/ops/split_ops.cuh"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// Utility: load a named weight from the WeightMap, cast to target dtype,
// and upload to device if needed.
// ─────────────────────────────────────────────────────────────────────────────
static Tensor load_weight(const WeightMap& weights, const std::string& key,
                           DType target_dtype, BackendPtr backend,
                           StreamHandle stream) {
    auto it = weights.find(key);
    if (it == weights.end())
        throw std::runtime_error("Missing weight key: " + key);

    const Tensor& src = it->second;
    Tensor dst = backend->alloc(src.shape(), target_dtype, stream);

    if (src.device().is_cpu()) {
        if (src.dtype() == target_dtype) {
            backend->copy(dst, src, stream);
        } else {
            Tensor tmp = backend->alloc(src.shape(), src.dtype(), stream);
            backend->copy(tmp, src, stream);
            backend->cast(tmp, dst, stream);
        }
    } else {
        if (src.dtype() == target_dtype)
            backend->copy(dst, src, stream);
        else
            backend->cast(src, dst, stream);
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// TimestepEmbedding
//
// Exact match of PyTorch implementation in blocks.py:
//
//   def timestep_embedding(t, dim, max_period=10000):
//       half = dim // 2
//       freqs = exp(-log(max_period) * arange(0, half) / half)
//       args  = t[:, None].float() * freqs[None]
//       embedding = cat([cos(args), sin(args)], dim=-1)
//       return embedding
//
//   MLP: sinusoidal → Linear(freq_dim, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)
// ─────────────────────────────────────────────────────────────────────────────
void TimestepEmbedding::build_sincos_table(BackendPtr backend,
                                            StreamHandle stream) {
    std::vector<float> table(max_timesteps_ * freq_dim_);
    const float max_period = 10000.f;
    const int   half       = freq_dim_ / 2;

    for (int step = 0; step < max_timesteps_; ++step) {
        float t = static_cast<float>(step);
        for (int i = 0; i < half; ++i) {
            float freq  = std::exp(-std::log(max_period) * i / static_cast<float>(half));
            float angle = t * freq;
            table[step * freq_dim_ + i]        = std::cos(angle);
            table[step * freq_dim_ + half + i] = std::sin(angle);
        }
    }

    Tensor cpu_table = backend->alloc_pinned(
        Shape({max_timesteps_, freq_dim_}), DType::Float32);
    std::memcpy(cpu_table.raw_data_ptr(), table.data(),
                table.size() * sizeof(float));

    sincos_table_ = backend->alloc(
        Shape({max_timesteps_, freq_dim_}), DType::Float32, stream);
    backend->copy(sincos_table_, cpu_table, stream);
}

void TimestepEmbedding::load(const WeightMap& weights,
                               const std::string& prefix,
                               const RDT1BConfig& cfg,
                               BackendPtr backend, StreamHandle stream) {
    freq_dim_      = cfg.freq_dim;
    embed_dim_     = cfg.time_embed_dim;
    max_timesteps_ = cfg.num_train_timesteps;
    DType dt       = cfg.compute_dtype;

    // HF weight names: prefix + "mlp.0.weight/bias" and "mlp.2.weight/bias"
    mlp0_weight_ = load_weight(weights, prefix + "mlp.0.weight", dt, backend, stream);
    mlp0_bias_   = load_weight(weights, prefix + "mlp.0.bias",   dt, backend, stream);
    mlp2_weight_ = load_weight(weights, prefix + "mlp.2.weight", dt, backend, stream);
    mlp2_bias_   = load_weight(weights, prefix + "mlp.2.bias",   dt, backend, stream);

    build_sincos_table(backend, stream);
}

Tensor TimestepEmbedding::forward(int64_t t, BackendPtr backend,
                                   StreamHandle stream) const {
    // Clamp to valid range.
    t = std::max(int64_t(0), std::min(t, max_timesteps_ - 1));

    // Slice sinusoidal row: [1, freq_dim]
    Tensor sincos_row = sincos_table_.slice(t, t + 1);  // [1, freq_dim]

    DType dt   = mlp0_weight_.dtype();
    Tensor emb = backend->alloc(Shape({1, freq_dim_}), dt, stream);
    backend->cast(sincos_row, emb, stream);

    // Linear1: [1, freq_dim] × [hidden_dim, freq_dim]^T → [1, hidden_dim]
    Tensor h = backend->alloc(Shape({1, embed_dim_}), dt, stream);
    backend->gemm(emb, mlp0_weight_, h, 1.f, 0.f, false, true, stream);
    backend->add(h, mlp0_bias_.view({1, embed_dim_}), h, stream);
    backend->silu(h, h, stream);

    // Linear2: [1, hidden_dim] × [hidden_dim, hidden_dim]^T → [1, hidden_dim]
    Tensor out = backend->alloc(Shape({1, embed_dim_}), dt, stream);
    backend->gemm(h, mlp2_weight_, out, 1.f, 0.f, false, true, stream);
    backend->add(out, mlp2_bias_.view({1, embed_dim_}), out, stream);

    return out;  // [1, hidden_dim]
}

// ─────────────────────────────────────────────────────────────────────────────
// Mlp — timm's Mlp, 1× hidden, GELU(tanh approx)
// ─────────────────────────────────────────────────────────────────────────────
void Mlp::load(const WeightMap& weights, const std::string& prefix,
                int64_t in_dim, int64_t out_dim, DType dt,
                BackendPtr backend, StreamHandle stream) {
    in_dim_     = in_dim;
    out_dim_    = out_dim;
    fc1_weight_ = load_weight(weights, prefix + "fc1.weight", dt, backend, stream);
    fc1_bias_   = load_weight(weights, prefix + "fc1.bias",   dt, backend, stream);
    fc2_weight_ = load_weight(weights, prefix + "fc2.weight", dt, backend, stream);
    fc2_bias_   = load_weight(weights, prefix + "fc2.bias",   dt, backend, stream);
}

Tensor Mlp::forward(const Tensor& x, BackendPtr backend,
                     StreamHandle stream) const {
    DType   dt  = x.dtype();
    int64_t B   = x.shape()[0];
    int64_t S   = x.shape()[1];

    // fc1: [B*S, in_dim] → [B*S, in_dim]
    Tensor x_2d = x.view({B * S, in_dim_});
    Tensor h    = backend->alloc(Shape({B * S, in_dim_}), dt, stream);
    backend->gemm(x_2d, fc1_weight_, h, 1.f, 0.f, false, true, stream);
    backend->add(h, fc1_bias_.view({1, in_dim_}), h, stream);
    backend->gelu(h, h, stream);  // GELU with tanh approximation

    // fc2: [B*S, in_dim] → [B*S, out_dim]
    Tensor out_2d = backend->alloc(Shape({B * S, out_dim_}), dt, stream);
    backend->gemm(h, fc2_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, fc2_bias_.view({1, out_dim_}), out_2d, stream);

    return out_2d.view({B, S, out_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// SelfAttention (timm Attention with qkv_bias=True, qk_norm=True)
// ─────────────────────────────────────────────────────────────────────────────
void SelfAttention::load(const WeightMap& weights, const std::string& prefix,
                          const RDT1BConfig& cfg,
                          BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    num_heads_  = cfg.num_heads;
    head_dim_   = cfg.head_dim;
    norm_eps_   = cfg.rms_norm_eps;
    DType dt    = cfg.compute_dtype;

    qkv_weight_    = load_weight(weights, prefix + "qkv.weight",    dt, backend, stream);
    qkv_bias_      = load_weight(weights, prefix + "qkv.bias",      dt, backend, stream);
    q_norm_weight_ = load_weight(weights, prefix + "q_norm.weight", dt, backend, stream);
    k_norm_weight_ = load_weight(weights, prefix + "k_norm.weight", dt, backend, stream);
    proj_weight_   = load_weight(weights, prefix + "proj.weight",   dt, backend, stream);
    proj_bias_     = load_weight(weights, prefix + "proj.bias",     dt, backend, stream);
}

void SelfAttention::apply_qk_norm(Tensor& qk, const Tensor& weight,
                                   BackendPtr backend, StreamHandle stream) const {
    // qk: [B, H, S, head_dim] → reshape to [B*H*S, head_dim] → rms_norm → reshape back
    int64_t B = qk.shape()[0];
    int64_t H = qk.shape()[1];
    int64_t S = qk.shape()[2];

    Tensor flat = qk.view({B * H * S, head_dim_});
    Tensor out  = backend->alloc(flat.shape(), flat.dtype(), stream);
    backend->rms_norm(flat, weight, out, norm_eps_, stream);
    qk = out.view({B, H, S, head_dim_});
}

Tensor SelfAttention::forward(const Tensor& x, BackendPtr backend,
                               StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t S  = x.shape()[1];

    // QKV projection: [B*S, D] → [B*S, 3D]
    Tensor x_2d   = x.view({B * S, hidden_dim_});
    Tensor qkv_2d = backend->alloc(Shape({B * S, 3 * hidden_dim_}), dt, stream);
    backend->gemm(x_2d, qkv_weight_, qkv_2d, 1.f, 0.f, false, true, stream);
    backend->add(qkv_2d, qkv_bias_.view({1, 3 * hidden_dim_}), qkv_2d, stream);

    // Split QKV: [B, S, 3D] → three [B, H, S, head_dim]
    Tensor qkv = qkv_2d.view({B, S, 3 * hidden_dim_});
    Tensor Q = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    Tensor K = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    Tensor V = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    cuda_ops::split_qkv_kernel(qkv, Q, K, V, num_heads_, head_dim_, backend, stream);

    // qk_norm: per-head RmsNorm
    apply_qk_norm(Q, q_norm_weight_, backend, stream);
    apply_qk_norm(K, k_norm_weight_, backend, stream);

    // FlashAttention: [B, H, S, D] → [B, H, S, D]
    Tensor attn_out = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    backend->attention(Q, K, V, attn_out, nullptr, 0.f, /*is_causal=*/false, stream);

    // Convert attention output [B, H, S, D] → [B, S, H*D] for projection GEMM.
    Tensor attn_seq = backend->alloc(Shape({B, S, hidden_dim_}), dt, stream);
    cuda_ops::head_to_seq(attn_out, attn_seq, num_heads_, head_dim_, backend, stream);

    // Output projection: [B*S, D] → [B*S, D]
    Tensor attn_2d = attn_seq.view({B * S, hidden_dim_});
    Tensor out_2d  = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
    backend->gemm(attn_2d, proj_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, proj_bias_.view({1, hidden_dim_}), out_2d, stream);

    return out_2d.view({B, S, hidden_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// CrossAttention
// ─────────────────────────────────────────────────────────────────────────────
void CrossAttention::load(const WeightMap& weights, const std::string& prefix,
                           const RDT1BConfig& cfg,
                           BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    num_heads_  = cfg.num_heads;
    head_dim_   = cfg.head_dim;
    norm_eps_   = cfg.rms_norm_eps;
    DType dt    = cfg.compute_dtype;

    q_weight_      = load_weight(weights, prefix + "q.weight",      dt, backend, stream);
    q_bias_        = load_weight(weights, prefix + "q.bias",        dt, backend, stream);
    kv_weight_     = load_weight(weights, prefix + "kv.weight",     dt, backend, stream);
    kv_bias_       = load_weight(weights, prefix + "kv.bias",       dt, backend, stream);
    q_norm_weight_ = load_weight(weights, prefix + "q_norm.weight", dt, backend, stream);
    k_norm_weight_ = load_weight(weights, prefix + "k_norm.weight", dt, backend, stream);
    proj_weight_   = load_weight(weights, prefix + "proj.weight",   dt, backend, stream);
    proj_bias_     = load_weight(weights, prefix + "proj.bias",     dt, backend, stream);
}

void CrossAttention::apply_qk_norm(Tensor& qk, const Tensor& weight,
                                    BackendPtr backend, StreamHandle stream) const {
    int64_t B = qk.shape()[0];
    int64_t H = qk.shape()[1];
    int64_t S = qk.shape()[2];

    Tensor flat = qk.view({B * H * S, head_dim_});
    Tensor out  = backend->alloc(flat.shape(), flat.dtype(), stream);
    backend->rms_norm(flat, weight, out, norm_eps_, stream);
    qk = out.view({B, H, S, head_dim_});
}

Tensor CrossAttention::forward(const Tensor& x, const Tensor& c,
                                BackendPtr backend, StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t Sq = x.shape()[1];
    int64_t Sk = c.shape()[1];

    // Q from x: [B*Sq, D] → [B*Sq, D]
    Tensor x_2d  = x.view({B * Sq, hidden_dim_});
    Tensor q_2d  = backend->alloc(Shape({B * Sq, hidden_dim_}), dt, stream);
    backend->gemm(x_2d, q_weight_, q_2d, 1.f, 0.f, false, true, stream);
    backend->add(q_2d, q_bias_.view({1, hidden_dim_}), q_2d, stream);

    // KV from c: [B*Sk, D] → [B*Sk, 2D]
    Tensor c_2d  = c.view({B * Sk, hidden_dim_});
    Tensor kv_2d = backend->alloc(Shape({B * Sk, 2 * hidden_dim_}), dt, stream);
    backend->gemm(c_2d, kv_weight_, kv_2d, 1.f, 0.f, false, true, stream);
    backend->add(kv_2d, kv_bias_.view({1, 2 * hidden_dim_}), kv_2d, stream);

    // Convert Q: [B*Sq, D] seq-major → [B, H, Sq, D] head-major for attention.
    Tensor Q_seq = q_2d.view({B, Sq, hidden_dim_});  // [B, Sq, H*D]
    Tensor Q  = backend->alloc(Shape({B, num_heads_, Sq, head_dim_}), dt, stream);
    cuda_ops::seq_to_head(Q_seq, Q, num_heads_, head_dim_, backend, stream);

    // Convert KV: [B, Sk, 2*H*D] → K, V in [B, H, Sk, D] head-major.
    Tensor KV = kv_2d.view({B, Sk, 2 * hidden_dim_});
    Tensor K  = backend->alloc(Shape({B, num_heads_, Sk, head_dim_}), dt, stream);
    Tensor V  = backend->alloc(Shape({B, num_heads_, Sk, head_dim_}), dt, stream);
    cuda_ops::split_kv_kernel(KV, K, V, num_heads_, head_dim_, backend, stream);

    // qk_norm (both now in head-major [B, H, S, D])
    apply_qk_norm(Q, q_norm_weight_, backend, stream);
    apply_qk_norm(K, k_norm_weight_, backend, stream);

    // Attention: inputs/output [B, H, S, D]
    Tensor attn_out = backend->alloc(Shape({B, num_heads_, Sq, head_dim_}), dt, stream);
    backend->attention(Q, K, V, attn_out, nullptr, 0.f, /*is_causal=*/false, stream);

    // Convert attention output [B, H, Sq, D] → [B, Sq, H*D] seq-major.
    Tensor attn_seq = backend->alloc(Shape({B, Sq, hidden_dim_}), dt, stream);
    cuda_ops::head_to_seq(attn_out, attn_seq, num_heads_, head_dim_, backend, stream);

    // Output projection: [B*Sq, D] → [B*Sq, D]
    Tensor attn_2d = attn_seq.view({B * Sq, hidden_dim_});
    Tensor out_2d  = backend->alloc(Shape({B * Sq, hidden_dim_}), dt, stream);
    backend->gemm(attn_2d, proj_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, proj_bias_.view({1, hidden_dim_}), out_2d, stream);

    return out_2d.view({B, Sq, hidden_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// RDTBlock
// ─────────────────────────────────────────────────────────────────────────────
void RDTBlock::load(const WeightMap& weights, const std::string& prefix,
                     const RDT1BConfig& cfg,
                     BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    norm_eps_   = cfg.rms_norm_eps;
    DType dt    = cfg.compute_dtype;

    // 3 RmsNorm layers (weight only, shape [hidden_dim])
    norm1_weight_ = load_weight(weights, prefix + "norm1.weight", dt, backend, stream);
    norm2_weight_ = load_weight(weights, prefix + "norm2.weight", dt, backend, stream);
    norm3_weight_ = load_weight(weights, prefix + "norm3.weight", dt, backend, stream);

    self_attn_.load(weights, prefix + "attn.",        cfg, backend, stream);
    cross_attn_.load(weights, prefix + "cross_attn.", cfg, backend, stream);
    ffn_.load(weights, prefix + "ffn.", cfg.hidden_dim, cfg.hidden_dim,
              cfg.compute_dtype, backend, stream);
}

Tensor RDTBlock::forward(const Tensor& x,
                          const Tensor& lang_cond, const Tensor& img_cond,
                          int block_idx,
                          BackendPtr backend, StreamHandle stream) const {
    DType dt = x.dtype();
    (void)dt;

    // ── Self-attention branch ──────────────────────────────────────────────
    Tensor normed1 = backend->alloc(x.shape(), x.dtype(), stream);
    backend->rms_norm(x, norm1_weight_, normed1, norm_eps_, stream);

    Tensor attn_out = self_attn_.forward(normed1, backend, stream);
    Tensor x1 = backend->alloc(x.shape(), x.dtype(), stream);
    backend->add(x, attn_out, x1, stream);

    // ── Cross-attention branch (alternating: even=lang, odd=img) ──────────
    const Tensor& cond = (block_idx % 2 == 0) ? lang_cond : img_cond;

    Tensor normed2 = backend->alloc(x1.shape(), x1.dtype(), stream);
    backend->rms_norm(x1, norm2_weight_, normed2, norm_eps_, stream);

    Tensor cross_out = cross_attn_.forward(normed2, cond, backend, stream);
    Tensor x2 = backend->alloc(x1.shape(), x1.dtype(), stream);
    backend->add(x1, cross_out, x2, stream);

    // ── MLP branch ────────────────────────────────────────────────────────
    Tensor normed3 = backend->alloc(x2.shape(), x2.dtype(), stream);
    backend->rms_norm(x2, norm3_weight_, normed3, norm_eps_, stream);

    Tensor ffn_out = ffn_.forward(normed3, backend, stream);
    Tensor x3 = backend->alloc(x2.shape(), x2.dtype(), stream);
    backend->add(x2, ffn_out, x3, stream);

    return x3;
}

// ─────────────────────────────────────────────────────────────────────────────
// FinalLayer
// ─────────────────────────────────────────────────────────────────────────────
void FinalLayer::load(const WeightMap& weights, const std::string& prefix,
                       const RDT1BConfig& cfg,
                       BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    norm_eps_   = cfg.rms_norm_eps;
    DType dt    = cfg.compute_dtype;

    norm_final_weight_ = load_weight(weights, prefix + "norm_final.weight",
                                     dt, backend, stream);
    // ffn_final: hidden_dim → hidden_dim → action_dim
    ffn_final_.load(weights, prefix + "ffn_final.",
                    cfg.hidden_dim, cfg.action_dim,
                    cfg.compute_dtype, backend, stream);
}

Tensor FinalLayer::forward(const Tensor& x, BackendPtr backend,
                            StreamHandle stream) const {
    Tensor normed = backend->alloc(x.shape(), x.dtype(), stream);
    backend->rms_norm(x, norm_final_weight_, normed, norm_eps_, stream);
    return ffn_final_.forward(normed, backend, stream);
}

}  // namespace rdt1b
}  // namespace sf
