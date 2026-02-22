// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/dtype.h"
#include <cstdint>
#include <string>

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// RDT1BConfig: matches the published RDT-1B architecture.
//
// Reference: "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation"
//            (Liu et al., 2024, Tsinghua University)
//
// Key numbers from the paper / open-source checkpoint:
//   hidden_dim   = 2048
//   num_layers   = 24
//   num_heads    = 32   → head_dim = 64
//   mlp_ratio    = 4.0  → ffn_dim = 8192
//   action_dim   = 128  (unified action space, robot-specific subsets used)
//   action_horizon = 64 (action chunk length)
// ─────────────────────────────────────────────────────────────────────────────
struct RDT1BConfig {
    // ── DiT backbone ──────────────────────────────────────────────────────
    int64_t hidden_dim     = 2048;
    int64_t num_layers     = 24;
    int64_t num_heads      = 32;
    int64_t head_dim       = 64;    // hidden_dim / num_heads
    double  mlp_ratio      = 4.0;   // FFN dim = hidden_dim * mlp_ratio
    float   attn_dropout   = 0.0f;
    float   ffn_dropout    = 0.0f;

    // ── Action / robot ────────────────────────────────────────────────────
    int64_t action_dim     = 128;   // full unified action space
    int64_t action_horizon = 64;    // T_action: steps in the action chunk

    // ── Timestep embedding ────────────────────────────────────────────────
    int64_t freq_dim       = 256;   // sinusoidal frequency dimension
    int64_t time_embed_dim = 2048;  // after MLP projection

    // ── Vision encoder (SigLIP ViT-So400M) ───────────────────────────────
    int64_t vision_embed_dim     = 1152;  // SigLIP output dim
    int64_t num_image_tokens     = 729;   // 27×27 patches for 378×378 input
    int64_t num_cameras          = 2;     // primary + wrist (configurable)
    int64_t projected_vision_dim = 2048;  // projected to match hidden_dim

    // ── Language encoder (T5-XXL) ─────────────────────────────────────────
    int64_t lang_embed_dim  = 4096;   // T5-XXL output dim
    int64_t lang_max_tokens = 128;
    int64_t projected_lang_dim = 2048;  // projected to match hidden_dim

    // ── Robot state ───────────────────────────────────────────────────────
    int64_t state_dim = 128;  // matches action_dim for unified representation

    // ── Normalisation ─────────────────────────────────────────────────────
    float layer_norm_eps = 1e-6f;

    // ── Compute ───────────────────────────────────────────────────────────
    DType compute_dtype = DType::Float16;

    // ── Derived helpers ───────────────────────────────────────────────────
    int64_t ffn_dim() const {
        return static_cast<int64_t>(hidden_dim * mlp_ratio);
    }

    // Total condition sequence length (vision + language + state tokens).
    int64_t cond_seq_len() const {
        return num_cameras * num_image_tokens  // vision
             + lang_max_tokens                 // language
             + 1;                              // robot state (1 token)
    }

    // Full sequence length passed through the DiT.
    int64_t total_seq_len() const {
        return cond_seq_len() + action_horizon;
    }

    // Load from a JSON config file (e.g. config.json in HF checkpoint dir).
    static RDT1BConfig from_json(const std::string& path);

    // Validate consistency of fields.
    void validate() const;
};

}  // namespace rdt1b
}  // namespace sf
