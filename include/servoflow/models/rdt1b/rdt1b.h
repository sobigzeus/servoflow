// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/rdt1b/config.h"
#include "servoflow/models/rdt1b/dit_block.h"
#include <memory>
#include <string>
#include <vector>

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// RDT1BModel: full model implementation of IVLAModel.
//
// Computational graph:
//
//  [Images × N_cam]  [Language tokens]  [Robot state]
//       │                   │                  │
//   SigLIP ViT           T5 Enc (frozen)   Linear proj
//       │                   │                  │
//   Linear proj         Linear proj        [1, state_dim]
//       │                   │
//  [B, N_cam*S_v, D]  [B, S_l, D]
//       └──────── cat ──────┘
//                  │                        [B, 1, D]
//          [B, S_cond, D]  ←────────────────────┘
//                  │
//        ┌─────────┴─────────────────────────────┐
//        │         [B, T_action, D]              │
//        │              │                        │
//        │         Linear proj (action_in)       │
//        │              │                        │
//        └──────── cat ──────────────────────────┘
//                  │
//         [B, S_cond + T_action, D]   ← full sequence
//                  │
//              DiT Blocks × 24
//                  │
//             FinalLayer
//                  │
//         velocity [B, T_action, action_dim]
// ─────────────────────────────────────────────────────────────────────────────
class RDT1BModel : public IVLAModel {
public:
    // Create from a config; weights loaded separately via load_weights().
    explicit RDT1BModel(RDT1BConfig config, BackendPtr backend);

    // Load weights from a ServoFlow checkpoint directory.
    // Call this after constructing the model, before first inference.
    void load_weights(const std::string& checkpoint_dir,
                      StreamHandle stream = nullptr);

    // ── IVLAModel interface ────────────────────────────────────────────────
    Tensor encode_condition(const VLAInput& input,
                            BackendPtr backend,
                            StreamHandle stream) override;

    void denoise_step(const Tensor& x_t, float t,
                      const Tensor& condition,
                      Tensor& velocity,
                      BackendPtr backend,
                      StreamHandle stream) override;

    Tensor decode_action(const Tensor& raw,
                         BackendPtr backend,
                         StreamHandle stream) override;

    int64_t action_dim()     const override { return cfg_.action_dim;     }
    int64_t action_horizon() const override { return cfg_.action_horizon; }
    DType   dtype()          const override { return cfg_.compute_dtype;  }

    const RDT1BConfig& config() const { return cfg_; }

private:
    // Vision encoder: SigLIP ViT-So400M.
    // In phase 1 we treat this as opaque and load its output directly
    // (pre-computed embeddings) or use a lightweight wrapper.
    // Full ViT implementation is a separate module.
    struct VisionEncoder {
        // Input: image [1, C, H, W] → output: [1, num_image_tokens, vision_embed_dim]
        Tensor forward(const Tensor& image, BackendPtr backend,
                       StreamHandle stream) const;

        // Projection: [vision_embed_dim] → [hidden_dim]
        Tensor proj_weight;  // [hidden_dim, vision_embed_dim]
        Tensor proj_bias;    // [hidden_dim]

        // ViT weights (large; stored but not fully implemented in Phase 1).
        // Phase 1: accept pre-computed embeddings, skip the ViT forward pass.
        bool use_precomputed_embeddings = false;
        WeightMap vit_weights;

        int64_t vision_embed_dim = 1152;
        int64_t hidden_dim       = 2048;
    };

    // Language encoder: T5-XXL encoder is frozen and run separately.
    // ServoFlow accepts pre-computed T5 embeddings as input to avoid
    // bundling a 10 GB T5 encoder in the inference device.
    struct LanguageProjection {
        Tensor proj_weight;   // [hidden_dim, lang_embed_dim]
        Tensor proj_bias;     // [hidden_dim]
        int64_t lang_dim   = 4096;
        int64_t hidden_dim = 2048;

        // [B, S_l, lang_embed_dim] → [B, S_l, hidden_dim]
        Tensor forward(const Tensor& lang_emb, BackendPtr backend,
                       StreamHandle stream) const;
    };

    // Action input projection: maps noisy action to token space.
    struct ActionInputProj {
        Tensor weight;  // [hidden_dim, action_dim]
        Tensor bias;    // [hidden_dim]

        // [B, T_action, action_dim] → [B, T_action, hidden_dim]
        Tensor forward(const Tensor& action, BackendPtr backend,
                       StreamHandle stream) const;
    };

    // Robot state tokeniser: maps state vector to a single token.
    struct StateTokeniser {
        Tensor weight;  // [hidden_dim, state_dim]
        Tensor bias;    // [hidden_dim]

        // [B, state_dim] → [B, 1, hidden_dim]
        Tensor forward(const Tensor& state, BackendPtr backend,
                       StreamHandle stream) const;
    };

    // Normalisation statistics for action un-normalisation.
    struct ActionNorm {
        Tensor mean;   // [action_dim]
        Tensor std;    // [action_dim]
    };

    RDT1BConfig        cfg_;
    BackendPtr         backend_;

    TimestepEmbedding  time_embed_;
    VisionEncoder      vision_enc_;
    LanguageProjection lang_proj_;
    ActionInputProj    action_in_proj_;
    StateTokeniser     state_tok_;
    std::vector<DiTBlock> dit_blocks_;  // cfg_.num_layers blocks
    FinalLayer         final_layer_;
    ActionNorm         action_norm_;

    // Pre-allocated working buffers for denoise_step (stable addresses for CUDA Graph).
    Tensor buf_time_emb_;     // [1, time_embed_dim]
    Tensor buf_action_proj_;  // [1, T_action, hidden_dim]
    Tensor buf_seq_;          // [1, S_total, hidden_dim]
    Tensor buf_block_out_;    // [1, S_total, hidden_dim]

    void preallocate_work_buffers(StreamHandle stream);
    void load_dit_weights(const WeightMap& weights, StreamHandle stream);
};

// ─────────────────────────────────────────────────────────────────────────────
// Factory function: create and return a fully loaded RDT-1B model.
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<RDT1BModel> load_rdt1b(const std::string& checkpoint_dir,
                                        Device device = kCUDA0);

}  // namespace rdt1b
}  // namespace sf
