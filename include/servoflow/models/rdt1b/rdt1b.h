// SPDX-License-Identifier: Apache-2.0
#pragma once

// RDT1BModel: ServoFlow implementation of the RDT-1B model.
// Architecture: https://huggingface.co/robotics-diffusion-transformer/rdt-1b

#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/rdt1b/config.h"
#include "servoflow/models/rdt1b/dit_block.h"
#include <memory>
#include <string>
#include <vector>

namespace sf {
namespace rdt1b {

class RDT1BModel : public IVLAModel {
public:
    explicit RDT1BModel(RDT1BConfig config, BackendPtr backend);

    void load_weights(const std::string& checkpoint_dir,
                      StreamHandle stream = nullptr);

    // ── IVLAModel interface ────────────────────────────────────────────────
    Tensor encode_condition(const VLAInput& input,
                            BackendPtr backend,
                            StreamHandle stream = nullptr) override;

    void denoise_step(const Tensor& x_t, float t,
                      const Tensor& condition,
                      Tensor& out,
                      BackendPtr backend,
                      StreamHandle stream = nullptr) override;

    Tensor decode_action(const Tensor& raw,
                         BackendPtr backend,
                         StreamHandle stream = nullptr) override { return raw; }

    int64_t action_dim()    const override { return cfg_.action_dim; }
    int64_t action_horizon() const override { return cfg_.action_horizon; }
    DType   dtype()          const override { return cfg_.compute_dtype; }

    // ── E2E alignment test interface ──────────────────────────────────────
    // Runs the full forward pass with raw feature tensors (B=1).
    // Exactly matches the PyTorch reference. Returns [1, T, action_dim].
    Tensor forward_raw(
        const Tensor& lang_features,   // [1, L, lang_token_dim]
        const Tensor& img_features,    // [1, I, img_token_dim]
        const Tensor& state_tok,       // [1, 1, action_dim]
        const Tensor& noisy_action,    // [1, T, action_dim]
        int64_t t,
        int64_t freq,
        StreamHandle stream = nullptr) const;

private:
    // ── MLP adaptor (mlp2x_gelu or mlp3x_gelu) ────────────────────────────
    struct MlpAdaptor {
        int64_t in_dim_     = 0;
        int64_t hidden_dim_ = 0;
        int depth_          = 2;

        Tensor fc0_weight_, fc0_bias_;
        Tensor fc2_weight_, fc2_bias_;
        Tensor fc4_weight_, fc4_bias_;   // depth==3 only

        void load(const WeightMap& weights, const std::string& prefix,
                  int64_t in_dim, int64_t hidden_dim, int64_t depth,
                  DType dt, BackendPtr backend, StreamHandle stream);

        Tensor forward(const Tensor& x,
                       BackendPtr backend, StreamHandle stream) const;
    };

    // ── Members ────────────────────────────────────────────────────────────
    RDT1BConfig          cfg_;
    BackendPtr           backend_;

    MlpAdaptor           lang_adaptor_;
    MlpAdaptor           img_adaptor_;
    MlpAdaptor           state_adaptor_;

    rdt1b::TimestepEmbedding  t_embedder_;
    rdt1b::TimestepEmbedding  freq_embedder_;

    Tensor x_pos_embed_;
    Tensor lang_cond_pos_embed_;
    Tensor img_cond_pos_embed_;

    std::vector<rdt1b::RDTBlock> blocks_;
    rdt1b::FinalLayer            final_layer_;

    struct { Tensor mean, std; } action_norm_;

    void load_dit_weights(const WeightMap& weights, StreamHandle stream);
};

std::shared_ptr<RDT1BModel> load_rdt1b(const std::string& checkpoint_dir,
                                         BackendPtr backend = nullptr,
                                         Device device = Device{DeviceType::CUDA});

}  // namespace rdt1b
}  // namespace sf
