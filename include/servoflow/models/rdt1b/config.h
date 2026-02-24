// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/dtype.h"
#include <cstdint>
#include <string>

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// RDT1BConfig: matches the published RDT-1B architecture exactly.
//
// Reference: "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation"
//            (Liu et al., 2024, Tsinghua University)
//            https://huggingface.co/robotics-diffusion-transformer/rdt-1b
//
// Block architecture (RDTBlock, NOT DiT-style adaLN):
//   norm1 (RmsNorm) → self-attn (qkv_bias=True, qk_norm=True) → residual
//   norm2 (RmsNorm) → cross-attn (alternating: even=lang, odd=img) → residual
//   norm3 (RmsNorm) → MLP (1× expansion, GELU tanh approx) → residual
//
// Input token sequence to transformer:
//   [t_emb (1), freq_emb (1), state (1), action_0..action_63 (64)] = 67 tokens
// ─────────────────────────────────────────────────────────────────────────────
struct RDT1BConfig {
    // ── DiT backbone (actual RDT-1B values) ─────────────────────────────
    int64_t hidden_dim     = 2048;
    int64_t num_layers     = 28;       // NOT 24 — 28 in the published model
    int64_t num_heads      = 32;
    int64_t head_dim       = 64;       // hidden_dim / num_heads

    // ── Action / robot ────────────────────────────────────────────────────
    int64_t action_dim     = 128;      // unified action space
    int64_t action_horizon = 64;       // T: action chunk length

    // ── Timestep / control-freq embedding ────────────────────────────────
    int64_t freq_dim       = 256;      // sinusoidal frequency embedding dim
    int64_t time_embed_dim = 2048;     // output dim after MLP (= hidden_dim)

    // ── Condition token dimensions ────────────────────────────────────────
    int64_t img_token_dim    = 1152;   // SigLIP ViT-So400M output dim
    int64_t img_cond_len     = 4374;   // fixed: 2 cameras × 3 frames × 729
    int64_t lang_token_dim   = 4096;   // T5-XXL output dim
    int64_t max_lang_cond_len = 1024;  // padded language sequence length
    int64_t state_token_dim  = 128;    // robot state / proprioception dim

    // ── DDPM noise schedule ───────────────────────────────────────────────
    int64_t num_train_timesteps    = 1000;
    int64_t ctrl_freq              = 25;   // control frequency Hz
    int64_t num_inference_timesteps = 5;   // DPM-Solver++ default

    // ── Normalisation ─────────────────────────────────────────────────────
    float rms_norm_eps  = 1e-6f;

    // ── Compute ───────────────────────────────────────────────────────────
    DType compute_dtype = DType::Float16;

    // ── Derived helpers ───────────────────────────────────────────────────
    // MLP hidden = hidden_dim × 1 (NO expansion in RDT-1B, unlike DiT)
    int64_t mlp_hidden_dim() const { return hidden_dim; }

    // Total transformer input sequence: t_emb + freq_emb + state + actions
    int64_t x_seq_len() const { return 1 + 1 + 1 + action_horizon; }  // = 67

    // Load from a JSON config file (e.g. config.json in HF checkpoint dir).
    static RDT1BConfig from_json(const std::string& path);

    // Validate consistency of fields.
    void validate() const;
};

}  // namespace rdt1b
}  // namespace sf
