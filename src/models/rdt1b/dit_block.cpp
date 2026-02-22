// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/rdt1b/dit_block.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// Utility: load a named weight, cast to target dtype, move to device.
// ─────────────────────────────────────────────────────────────────────────────
static Tensor load_weight(const WeightMap& weights, const std::string& key,
                          DType target_dtype, BackendPtr backend,
                          StreamHandle stream) {
    auto it = weights.find(key);
    if (it == weights.end())
        throw std::runtime_error("Missing weight: " + key);

    const Tensor& src = it->second;
    // Allocate on device.
    Tensor dst = backend->alloc(src.shape(), target_dtype, stream);

    if (src.device().is_cpu()) {
        // Upload: H2D copy, then cast if needed.
        if (src.dtype() == target_dtype) {
            backend->copy(dst, src, stream);
        } else {
            Tensor tmp = backend->alloc(src.shape(), src.dtype(), stream);
            backend->copy(tmp, src, stream);
            backend->cast(tmp, dst, stream);
        }
    } else {
        if (src.dtype() == target_dtype) {
            backend->copy(dst, src, stream);
        } else {
            backend->cast(src, dst, stream);
        }
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// TimestepEmbedding
// ─────────────────────────────────────────────────────────────────────────────
void TimestepEmbedding::build_sincos_table(BackendPtr backend,
                                            StreamHandle stream) {
    // Pre-compute sinusoidal embeddings for t in [0, 1] mapped to 1000 steps.
    // The table has shape [1000, freq_dim]; at inference we pick the row
    // corresponding to round(t * 999).
    constexpr int kMaxSteps = 1000;
    std::vector<float> table(kMaxSteps * freq_dim_);
    float max_period = 10000.f;

    for (int step = 0; step < kMaxSteps; ++step) {
        float t = static_cast<float>(step);
        for (int i = 0; i < freq_dim_ / 2; ++i) {
            float freq = std::exp(-std::log(max_period)
                                  * i / (freq_dim_ / 2 - 1));
            float angle = t * freq;
            table[step * freq_dim_ + i]                 = std::cos(angle);
            table[step * freq_dim_ + freq_dim_ / 2 + i] = std::sin(angle);
        }
    }

    // Upload to device.
    Tensor cpu_table = backend->alloc_pinned(
        Shape({kMaxSteps, freq_dim_}), DType::Float32);
    std::memcpy(cpu_table.raw_data_ptr(), table.data(),
                table.size() * sizeof(float));

    sincos_table_ = backend->alloc(
        Shape({kMaxSteps, freq_dim_}), DType::Float32, stream);
    backend->copy(sincos_table_, cpu_table, stream);
}

void TimestepEmbedding::load(const WeightMap& weights,
                              const std::string& prefix,
                              const RDT1BConfig& cfg,
                              BackendPtr backend, StreamHandle stream) {
    freq_dim_  = cfg.freq_dim;
    embed_dim_ = cfg.time_embed_dim;
    DType dt   = cfg.compute_dtype;

    linear1_weight_ = load_weight(weights, prefix + "linear1.weight", dt, backend, stream);
    linear1_bias_   = load_weight(weights, prefix + "linear1.bias",   dt, backend, stream);
    linear2_weight_ = load_weight(weights, prefix + "linear2.weight", dt, backend, stream);
    linear2_bias_   = load_weight(weights, prefix + "linear2.bias",   dt, backend, stream);

    build_sincos_table(backend, stream);
}

Tensor TimestepEmbedding::forward(float t, BackendPtr backend,
                                   StreamHandle stream) const {
    // Pick sinusoidal row: row = round(t * 999).
    int row = static_cast<int>(std::round(t * 999.f));
    row     = std::max(0, std::min(999, row));

    // Slice [row, :] → [1, freq_dim]
    Tensor sincos = sincos_table_.slice(row, row + 1);  // [1, freq_dim]

    // Cast to compute dtype for GEMM.
    DType dt = linear1_weight_.dtype();
    Tensor emb = backend->alloc(Shape({1, freq_dim_}), dt, stream);
    backend->cast(sincos, emb, stream);

    // Linear1: [1, freq_dim] × [freq_dim, embed_dim] → [1, embed_dim]
    Tensor h = backend->alloc(Shape({1, embed_dim_}), dt, stream);
    backend->gemm(emb, linear1_weight_, h,
                  1.f, 0.f, false, true, stream);
    backend->add(h, linear1_bias_.view({1, embed_dim_}), h, stream);
    backend->silu(h, h, stream);

    // Linear2: [1, embed_dim] × [embed_dim, embed_dim] → [1, embed_dim]
    Tensor out = backend->alloc(Shape({1, embed_dim_}), dt, stream);
    backend->gemm(h, linear2_weight_, out,
                  1.f, 0.f, false, true, stream);
    backend->add(out, linear2_bias_.view({1, embed_dim_}), out, stream);

    return out;  // [1, embed_dim]
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaLNModulation
// ─────────────────────────────────────────────────────────────────────────────
void AdaLNModulation::load(const WeightMap& weights,
                            const std::string& prefix,
                            const RDT1BConfig& cfg,
                            BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    DType dt    = cfg.compute_dtype;

    linear_weight_ = load_weight(weights, prefix + "linear.weight", dt, backend, stream);
    linear_bias_   = load_weight(weights, prefix + "linear.bias",   dt, backend, stream);
}

AdaLNModulation::Params AdaLNModulation::forward(const Tensor& c,
                                                   BackendPtr backend,
                                                   StreamHandle stream) const {
    DType dt = c.dtype();
    int64_t B = c.shape()[0];

    // SiLU(c): [B, time_embed_dim]
    Tensor act = backend->alloc(c.shape(), dt, stream);
    backend->silu(c, act, stream);

    // Linear: [B, time_embed_dim] → [B, 2*hidden_dim]
    Tensor proj = backend->alloc(Shape({B, 2 * hidden_dim_}), dt, stream);
    backend->gemm(act, linear_weight_, proj,
                  1.f, 0.f, false, true, stream);
    backend->add(proj, linear_bias_.view({1, 2 * hidden_dim_}), proj, stream);

    // Split into scale and shift, each [B, hidden_dim].
    Tensor scale = proj.slice(0, B);   // uses view into same storage
    // We need actual separate slices along dim 1; use view + offset arithmetic.
    // Since split is along the last dim, we do it via view + manual slice.
    // TODO: implement a proper dim-1 split in the backend.
    // For now, allocate and copy each half.
    Tensor scale_out = backend->alloc(Shape({B, hidden_dim_}), dt, stream);
    Tensor shift_out = backend->alloc(Shape({B, hidden_dim_}), dt, stream);

    // Copy first hidden_dim columns to scale, second to shift.
    // This is a strided copy; backend::cat in reverse, or a custom kernel.
    // We express it as two GEMM with identity slices using an eye matrix,
    // but that's expensive. Instead, we rely on the split_last_dim helper
    // which we implement as a dedicated elementwise copy kernel.
    split_last_dim(proj, scale_out, shift_out, backend, stream);

    return {scale_out, shift_out};
}

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward (SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────
void FeedForward::load(const WeightMap& weights, const std::string& prefix,
                        const RDT1BConfig& cfg,
                        BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    ffn_dim_    = cfg.ffn_dim();
    DType dt    = cfg.compute_dtype;

    gate_up_weight_ = load_weight(weights, prefix + "gate_up_proj.weight",
                                  dt, backend, stream);
    gate_up_bias_   = load_weight(weights, prefix + "gate_up_proj.bias",
                                  dt, backend, stream);
    down_weight_    = load_weight(weights, prefix + "down_proj.weight",
                                  dt, backend, stream);
    down_bias_      = load_weight(weights, prefix + "down_proj.bias",
                                  dt, backend, stream);
}

void FeedForward::forward(const Tensor& x, Tensor& out,
                           BackendPtr backend, StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t S  = x.shape()[1];

    // gate_up = x @ gate_up_weight^T  → [B*S, 2*ffn_dim]
    Tensor x_2d       = x.view({B * S, hidden_dim_});
    Tensor gate_up    = backend->alloc(Shape({B * S, 2 * ffn_dim_}), dt, stream);
    backend->gemm(x_2d, gate_up_weight_, gate_up,
                  1.f, 0.f, false, true, stream);
    backend->add(gate_up,
                 gate_up_bias_.view({1, 2 * ffn_dim_}),
                 gate_up, stream);

    // Split gate and up: each [B*S, ffn_dim].
    Tensor gate = backend->alloc(Shape({B * S, ffn_dim_}), dt, stream);
    Tensor up   = backend->alloc(Shape({B * S, ffn_dim_}), dt, stream);
    split_last_dim(gate_up, gate, up, backend, stream);

    // SwiGLU: out_ffn = SiLU(gate) * up
    backend->silu(gate, gate, stream);
    backend->mul(gate, up, gate, stream);  // reuse gate buffer

    // down = gate @ down_weight^T  → [B*S, hidden_dim]
    Tensor out_2d = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
    backend->gemm(gate, down_weight_, out_2d,
                  1.f, 0.f, false, true, stream);
    backend->add(out_2d,
                 down_bias_.view({1, hidden_dim_}),
                 out_2d, stream);

    out = out_2d.view({B, S, hidden_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiHeadAttention
// ─────────────────────────────────────────────────────────────────────────────
void MultiHeadAttention::load(const WeightMap& weights,
                               const std::string& prefix,
                               const RDT1BConfig& cfg, bool cross_attn,
                               BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    num_heads_  = cfg.num_heads;
    head_dim_   = cfg.head_dim;
    cross_attn_ = cross_attn;
    DType dt    = cfg.compute_dtype;

    if (!cross_attn_) {
        qkv_weight_ = load_weight(weights, prefix + "qkv.weight",
                                  dt, backend, stream);
        qkv_bias_   = load_weight(weights, prefix + "qkv.bias",
                                  dt, backend, stream);
    } else {
        q_weight_  = load_weight(weights, prefix + "q.weight",  dt, backend, stream);
        q_bias_    = load_weight(weights, prefix + "q.bias",    dt, backend, stream);
        kv_weight_ = load_weight(weights, prefix + "kv.weight", dt, backend, stream);
        kv_bias_   = load_weight(weights, prefix + "kv.bias",   dt, backend, stream);
    }

    out_weight_ = load_weight(weights, prefix + "proj.weight", dt, backend, stream);
    out_bias_   = load_weight(weights, prefix + "proj.bias",   dt, backend, stream);
}

void MultiHeadAttention::split_qkv(const Tensor& qkv,
                                    Tensor& Q, Tensor& K, Tensor& V,
                                    BackendPtr backend, StreamHandle stream) const {
    // qkv: [B, S, 3*H*head_dim]
    int64_t B = qkv.shape()[0];
    int64_t S = qkv.shape()[1];
    DType dt  = qkv.dtype();

    Q = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    K = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    V = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);

    split_qkv_kernel(qkv, Q, K, V, num_heads_, head_dim_, backend, stream);
}

void MultiHeadAttention::forward(const Tensor& x, Tensor& out,
                                  bool is_causal,
                                  BackendPtr backend, StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t S  = x.shape()[1];

    // Project to QKV: [B, S, hidden] → [B, S, 3*hidden]
    Tensor x_2d  = x.view({B * S, hidden_dim_});
    Tensor qkv_2d = backend->alloc(Shape({B * S, 3 * hidden_dim_}), dt, stream);
    backend->gemm(x_2d, qkv_weight_, qkv_2d, 1.f, 0.f, false, true, stream);
    backend->add(qkv_2d,
                 qkv_bias_.view({1, 3 * hidden_dim_}),
                 qkv_2d, stream);

    Tensor qkv = qkv_2d.view({B, S, 3 * hidden_dim_});
    Tensor Q, K, V;
    split_qkv(qkv, Q, K, V, backend, stream);

    // FlashAttention: [B, H, S, D] → [B, H, S, D]
    Tensor attn_out = backend->alloc(Shape({B, num_heads_, S, head_dim_}), dt, stream);
    backend->attention(Q, K, V, attn_out, nullptr, 0.f, is_causal, stream);

    // Reshape [B, H, S, D] → [B, S, H*D] and project out.
    Tensor attn_2d = attn_out.view({B * S, hidden_dim_});
    Tensor out_2d  = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
    backend->gemm(attn_2d, out_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d,
                 out_bias_.view({1, hidden_dim_}),
                 out_2d, stream);

    out = out_2d.view({B, S, hidden_dim_});
}

void MultiHeadAttention::forward_cross(const Tensor& x, const Tensor& context,
                                        Tensor& out,
                                        BackendPtr backend,
                                        StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t Sq = x.shape()[1];
    int64_t Sk = context.shape()[1];

    // Q from x: [B*Sq, hidden] → [B*Sq, hidden]
    Tensor x_2d   = x.view({B * Sq, hidden_dim_});
    Tensor q_2d   = backend->alloc(Shape({B * Sq, hidden_dim_}), dt, stream);
    backend->gemm(x_2d, q_weight_, q_2d, 1.f, 0.f, false, true, stream);
    backend->add(q_2d, q_bias_.view({1, hidden_dim_}), q_2d, stream);

    // K,V from context: [B*Sk, hidden] → [B*Sk, 2*hidden]
    Tensor ctx_2d = context.view({B * Sk, hidden_dim_});
    Tensor kv_2d  = backend->alloc(Shape({B * Sk, 2 * hidden_dim_}), dt, stream);
    backend->gemm(ctx_2d, kv_weight_, kv_2d, 1.f, 0.f, false, true, stream);
    backend->add(kv_2d, kv_bias_.view({1, 2 * hidden_dim_}), kv_2d, stream);

    // Reshape to [B, H, S, D].
    Tensor Q = q_2d.view({B, num_heads_, Sq, head_dim_});
    Tensor KV = kv_2d.view({B, Sk, 2 * hidden_dim_});
    Tensor K = backend->alloc(Shape({B, num_heads_, Sk, head_dim_}), dt, stream);
    Tensor V = backend->alloc(Shape({B, num_heads_, Sk, head_dim_}), dt, stream);
    split_kv_kernel(KV, K, V, num_heads_, head_dim_, backend, stream);

    Tensor attn_out = backend->alloc(Shape({B, num_heads_, Sq, head_dim_}), dt, stream);
    backend->attention(Q, K, V, attn_out, nullptr, 0.f, false, stream);

    Tensor attn_2d = attn_out.view({B * Sq, hidden_dim_});
    Tensor out_2d  = backend->alloc(Shape({B * Sq, hidden_dim_}), dt, stream);
    backend->gemm(attn_2d, out_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, out_bias_.view({1, hidden_dim_}), out_2d, stream);

    out = out_2d.view({B, Sq, hidden_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// DiTBlock
// ─────────────────────────────────────────────────────────────────────────────
void DiTBlock::load(const WeightMap& weights, const std::string& prefix,
                     const RDT1BConfig& cfg,
                     BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    norm_eps_   = cfg.layer_norm_eps;
    DType dt    = cfg.compute_dtype;

    adaln_.load(weights, prefix + "adaLN_modulation.", cfg, backend, stream);
    attn_ .load(weights, prefix + "attn.", cfg, /*cross_attn=*/false, backend, stream);
    ffn_  .load(weights, prefix + "ffn.",  cfg, backend, stream);

    norm1_weight_ = load_weight(weights, prefix + "norm1.weight", dt, backend, stream);
    norm1_bias_   = load_weight(weights, prefix + "norm1.bias",   dt, backend, stream);
    norm2_weight_ = load_weight(weights, prefix + "norm2.weight", dt, backend, stream);
    norm2_bias_   = load_weight(weights, prefix + "norm2.bias",   dt, backend, stream);
}

void DiTBlock::forward(const Tensor& x, const Tensor& c,
                        int64_t cond_len, Tensor& out,
                        BackendPtr backend, StreamHandle stream) const {
    DType   dt = x.dtype();
    int64_t B  = x.shape()[0];
    int64_t S  = x.shape()[1];

    // ── adaLN modulation ─────────────────────────────────────────────────
    auto [scale1, shift1] = adaln_.forward(c, backend, stream);

    // ── Self-attention branch ─────────────────────────────────────────────
    // Pre-norm.
    Tensor normed1 = backend->alloc(x.shape(), dt, stream);
    backend->layer_norm(x, norm1_weight_, norm1_bias_, normed1, norm_eps_, stream);

    // Apply adaLN: normed = (1 + scale) * normed + shift
    apply_adaln(normed1, scale1, shift1, backend, stream);

    // Self-attention: condition tokens are fully visible to all tokens;
    // action tokens attend to everything (no causal mask in the full sequence).
    Tensor attn_out = backend->alloc(x.shape(), dt, stream);
    attn_.forward(normed1, attn_out, /*is_causal=*/false, backend, stream);

    // Residual.
    Tensor x1 = backend->alloc(x.shape(), dt, stream);
    backend->add(x, attn_out, x1, stream);

    // ── FFN branch ────────────────────────────────────────────────────────
    auto [scale2, shift2] = adaln_.forward(c, backend, stream);

    Tensor normed2 = backend->alloc(x1.shape(), dt, stream);
    backend->layer_norm(x1, norm2_weight_, norm2_bias_, normed2, norm_eps_, stream);
    apply_adaln(normed2, scale2, shift2, backend, stream);

    Tensor ffn_out = backend->alloc(x1.shape(), dt, stream);
    ffn_.forward(normed2, ffn_out, backend, stream);

    out = backend->alloc(x1.shape(), dt, stream);
    backend->add(x1, ffn_out, out, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// FinalLayer
// ─────────────────────────────────────────────────────────────────────────────
void FinalLayer::load(const WeightMap& weights, const std::string& prefix,
                       const RDT1BConfig& cfg,
                       BackendPtr backend, StreamHandle stream) {
    hidden_dim_ = cfg.hidden_dim;
    norm_eps_   = cfg.layer_norm_eps;
    DType dt    = cfg.compute_dtype;

    adaln_.load(weights, prefix + "adaLN_modulation.", cfg, backend, stream);
    norm_weight_   = load_weight(weights, prefix + "norm_final.weight", dt, backend, stream);
    norm_bias_     = load_weight(weights, prefix + "norm_final.bias",   dt, backend, stream);
    linear_weight_ = load_weight(weights, prefix + "linear.weight",     dt, backend, stream);
    linear_bias_   = load_weight(weights, prefix + "linear.bias",       dt, backend, stream);
}

void FinalLayer::forward(const Tensor& x, const Tensor& c,
                          int64_t action_horizon, Tensor& out,
                          BackendPtr backend, StreamHandle stream) const {
    DType   dt  = x.dtype();
    int64_t B   = x.shape()[0];
    int64_t S   = x.shape()[1];
    int64_t A   = linear_bias_.shape()[0];  // action_dim

    // Extract only the action subsequence (last action_horizon tokens).
    int64_t cond_len = S - action_horizon;
    Tensor action_x = x.slice(cond_len * hidden_dim_,
                               S * hidden_dim_);  // approximate; TODO: proper dim-1 slice
    action_x = x.view({B, S, hidden_dim_}).slice(cond_len, S);  // [B, T_action, D]

    auto [scale, shift] = adaln_.forward(c, backend, stream);

    Tensor normed = backend->alloc(action_x.shape(), dt, stream);
    backend->layer_norm(action_x, norm_weight_, norm_bias_, normed, norm_eps_, stream);
    apply_adaln(normed, scale, shift, backend, stream);

    int64_t T = action_horizon;
    Tensor normed_2d = normed.view({B * T, hidden_dim_});
    Tensor out_2d    = backend->alloc(Shape({B * T, A}), dt, stream);
    backend->gemm(normed_2d, linear_weight_, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, linear_bias_.view({1, A}), out_2d, stream);

    out = out_2d.view({B, T, A});
}

}  // namespace rdt1b
}  // namespace sf
