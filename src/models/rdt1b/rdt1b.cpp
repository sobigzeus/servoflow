// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/loader/safetensors.h"

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// Sub-module forward implementations
// ─────────────────────────────────────────────────────────────────────────────

Tensor RDT1BModel::VisionEncoder::forward(const Tensor& image,
                                           BackendPtr backend,
                                           StreamHandle stream) const {
    if (use_precomputed_embeddings) {
        // image is already [1, num_tokens, vision_embed_dim]; just project.
        int64_t B = image.shape()[0];
        int64_t S = image.shape()[1];
        DType dt  = proj_weight.dtype();

        Tensor image_2d = image.view({B * S, vision_embed_dim});
        Tensor out_2d   = backend->alloc(Shape({B * S, hidden_dim}), dt, stream);
        backend->gemm(image_2d, proj_weight, out_2d, 1.f, 0.f, false, true, stream);
        backend->add(out_2d, proj_bias.view({1, hidden_dim}), out_2d, stream);
        return out_2d.view({B, S, hidden_dim});
    }
    // Full ViT forward: not implemented in Phase 1.
    // Phase 1 requires pre-computed embeddings as input.
    throw std::runtime_error(
        "VisionEncoder: full ViT forward not implemented in Phase 1. "
        "Provide pre-computed SigLIP embeddings via VLAInput::images.");
}

Tensor RDT1BModel::LanguageProjection::forward(const Tensor& lang_emb,
                                                BackendPtr backend,
                                                StreamHandle stream) const {
    int64_t B = lang_emb.shape()[0];
    int64_t S = lang_emb.shape()[1];
    DType dt  = proj_weight.dtype();

    Tensor in_2d  = lang_emb.view({B * S, lang_dim});
    Tensor out_2d = backend->alloc(Shape({B * S, hidden_dim}), dt, stream);
    backend->gemm(in_2d, proj_weight, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, proj_bias.view({1, hidden_dim}), out_2d, stream);
    return out_2d.view({B, S, hidden_dim});
}

Tensor RDT1BModel::ActionInputProj::forward(const Tensor& action,
                                             BackendPtr backend,
                                             StreamHandle stream) const {
    int64_t B = action.shape()[0];
    int64_t T = action.shape()[1];
    int64_t A = action.shape()[2];
    DType dt  = weight.dtype();

    Tensor a_2d   = action.view({B * T, A});
    Tensor out_2d = backend->alloc(Shape({B * T, hidden_dim}), dt, stream);
    backend->gemm(a_2d, weight, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, bias.view({1, hidden_dim}), out_2d, stream);
    return out_2d.view({B, T, hidden_dim});
}

Tensor RDT1BModel::StateTokeniser::forward(const Tensor& state,
                                            BackendPtr backend,
                                            StreamHandle stream) const {
    int64_t B = state.shape()[0];
    DType dt  = weight.dtype();

    Tensor out_2d = backend->alloc(Shape({B, hidden_dim}), dt, stream);
    backend->gemm(state, weight, out_2d, 1.f, 0.f, false, true, stream);
    backend->add(out_2d, bias.view({1, hidden_dim}), out_2d, stream);
    return out_2d.view({B, 1, hidden_dim});  // single token per sample
}

// ─────────────────────────────────────────────────────────────────────────────
// RDT1BModel
// ─────────────────────────────────────────────────────────────────────────────
RDT1BModel::RDT1BModel(RDT1BConfig config, BackendPtr backend)
    : cfg_(std::move(config)), backend_(backend) {
    cfg_.validate();
    dit_blocks_.resize(cfg_.num_layers);
}

void RDT1BModel::preallocate_work_buffers(StreamHandle stream) {
    DType dt  = cfg_.compute_dtype;
    int64_t T = cfg_.action_horizon;
    int64_t S = cfg_.total_seq_len();
    int64_t D = cfg_.hidden_dim;

    buf_time_emb_    = backend_->alloc(Shape({1, cfg_.time_embed_dim}), dt, stream);
    buf_action_proj_ = backend_->alloc(Shape({1, T, D}), dt, stream);
    buf_seq_         = backend_->alloc(Shape({1, S, D}), dt, stream);
    buf_block_out_   = backend_->alloc(Shape({1, S, D}), dt, stream);
}

void RDT1BModel::load_weights(const std::string& checkpoint_dir,
                               StreamHandle stream) {
    if (!fs::is_directory(checkpoint_dir))
        throw std::runtime_error("Checkpoint directory not found: " + checkpoint_dir);

    // ServoFlow checkpoints are stored as one or more safetensors shards.
    // The converter (tools/convert/hf_to_servoflow.py) produces:
    //   model.safetensors       (or model-00001-of-NNNNN.safetensors)
    //   config.json
    //   action_norm.safetensors  (mean/std for un-normalisation)
    WeightMap weights;
    for (auto& entry : fs::directory_iterator(checkpoint_dir)) {
        if (entry.path().extension() == ".safetensors") {
            auto shard = SafeTensorsLoader::load(entry.path().string());
            weights.merge(shard);
        }
    }

    // Validate that we have a minimum set of expected weights.
    if (weights.empty())
        throw std::runtime_error("No safetensors files found in: " + checkpoint_dir);

    StreamHandle s = stream ? stream : backend_->create_stream();

    // ── Timestep embedding ────────────────────────────────────────────────
    time_embed_.load(weights, "time_embed.", cfg_, backend_, s);

    // ── Vision encoder projection ─────────────────────────────────────────
    DType dt = cfg_.compute_dtype;
    vision_enc_.hidden_dim       = cfg_.hidden_dim;
    vision_enc_.vision_embed_dim = cfg_.vision_embed_dim;
    vision_enc_.use_precomputed_embeddings = true;  // Phase 1
    vision_enc_.proj_weight = load_weight_from_map(
        weights, "vision_proj.weight", dt, backend_, s);
    vision_enc_.proj_bias   = load_weight_from_map(
        weights, "vision_proj.bias",   dt, backend_, s);

    // ── Language projection ───────────────────────────────────────────────
    lang_proj_.lang_dim   = cfg_.lang_embed_dim;
    lang_proj_.hidden_dim = cfg_.hidden_dim;
    lang_proj_.proj_weight = load_weight_from_map(
        weights, "lang_proj.weight", dt, backend_, s);
    lang_proj_.proj_bias   = load_weight_from_map(
        weights, "lang_proj.bias",   dt, backend_, s);

    // ── Action input projection ───────────────────────────────────────────
    action_in_proj_.hidden_dim = cfg_.hidden_dim;
    action_in_proj_.weight = load_weight_from_map(
        weights, "action_in_proj.weight", dt, backend_, s);
    action_in_proj_.bias   = load_weight_from_map(
        weights, "action_in_proj.bias",   dt, backend_, s);

    // ── State tokeniser ───────────────────────────────────────────────────
    state_tok_.hidden_dim = cfg_.hidden_dim;
    state_tok_.weight = load_weight_from_map(
        weights, "state_tok.weight", dt, backend_, s);
    state_tok_.bias   = load_weight_from_map(
        weights, "state_tok.bias",   dt, backend_, s);

    // ── DiT blocks ────────────────────────────────────────────────────────
    load_dit_weights(weights, s);

    // ── Final layer ───────────────────────────────────────────────────────
    final_layer_.load(weights, "final_layer.", cfg_, backend_, s);

    // ── Action normalisation statistics ───────────────────────────────────
    if (weights.count("action_norm.mean")) {
        action_norm_.mean = load_weight_from_map(
            weights, "action_norm.mean", DType::Float32, backend_, s);
        action_norm_.std  = load_weight_from_map(
            weights, "action_norm.std",  DType::Float32, backend_, s);
    }

    backend_->sync_stream(s);
    if (!stream) backend_->destroy_stream(s);

    preallocate_work_buffers(stream);
}

void RDT1BModel::load_dit_weights(const WeightMap& weights,
                                    StreamHandle stream) {
    for (int i = 0; i < cfg_.num_layers; ++i) {
        std::string prefix = "dit.blocks." + std::to_string(i) + ".";
        dit_blocks_[i].load(weights, prefix, cfg_, backend_, stream);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IVLAModel::encode_condition
// ─────────────────────────────────────────────────────────────────────────────
Tensor RDT1BModel::encode_condition(const VLAInput& input,
                                     BackendPtr backend,
                                     StreamHandle stream) {
    DType dt = cfg_.compute_dtype;

    // ── 1. Encode images ──────────────────────────────────────────────────
    // Each camera image is expected as a pre-computed SigLIP embedding:
    //   [1, num_image_tokens, vision_embed_dim]
    std::vector<Tensor> cam_tokens;
    cam_tokens.reserve(input.images.size());
    for (auto& img_emb : input.images) {
        Tensor proj = vision_enc_.forward(img_emb, backend, stream);
        cam_tokens.push_back(proj);
    }

    // Concat camera tokens along seq dim: [1, N_cam * S_v, D]
    Tensor vision_tokens;
    if (cam_tokens.size() == 1) {
        vision_tokens = cam_tokens[0];
    } else {
        vision_tokens = backend->alloc(
            Shape({1, cfg_.num_cameras * cfg_.num_image_tokens, cfg_.hidden_dim}),
            dt, stream);
        backend->cat(cam_tokens, vision_tokens, /*dim=*/1, stream);
    }

    // ── 2. Project language embedding ─────────────────────────────────────
    // input.language_tokens is expected as a pre-computed T5 embedding:
    //   [1, S_l, lang_embed_dim]
    Tensor lang_tokens;
    if (input.language_tokens.is_valid()) {
        lang_tokens = lang_proj_.forward(input.language_tokens, backend, stream);
    } else {
        // No language input: use zero tokens.
        lang_tokens = backend->alloc(
            Shape({1, cfg_.lang_max_tokens, cfg_.hidden_dim}), dt, stream);
        backend->fill(lang_tokens, 0.f, stream);
    }

    // ── 3. Tokenise robot state ───────────────────────────────────────────
    Tensor state_token;
    if (input.robot_state.is_valid()) {
        // Cast state to compute dtype before projection.
        Tensor state_dt = backend->alloc(input.robot_state.shape(), dt, stream);
        backend->cast(input.robot_state, state_dt, stream);
        state_token = state_tok_.forward(state_dt, backend, stream);
    } else {
        state_token = backend->alloc(Shape({1, 1, cfg_.hidden_dim}), dt, stream);
        backend->fill(state_token, 0.f, stream);
    }

    // ── 4. Concatenate condition sequence ─────────────────────────────────
    // Order: [vision_tokens | lang_tokens | state_token]
    // Shape: [1, N_cam*S_v + S_l + 1, D]
    Tensor condition = backend->alloc(
        Shape({1, cfg_.cond_seq_len(), cfg_.hidden_dim}), dt, stream);
    backend->cat({vision_tokens, lang_tokens, state_token},
                 condition, /*dim=*/1, stream);

    return condition;
}

// ─────────────────────────────────────────────────────────────────────────────
// IVLAModel::denoise_step
// ─────────────────────────────────────────────────────────────────────────────
void RDT1BModel::denoise_step(const Tensor& x_t, float t,
                               const Tensor& condition,
                               Tensor& velocity,
                               BackendPtr backend,
                               StreamHandle stream) {
    DType dt = cfg_.compute_dtype;
    int64_t B = x_t.shape()[0];    // batch (usually 1)
    int64_t T = cfg_.action_horizon;
    int64_t D = cfg_.hidden_dim;
    int64_t S_cond = cfg_.cond_seq_len();
    int64_t S_total = S_cond + T;

    // ── 1. Timestep embedding: scalar t → [B, time_embed_dim] ─────────────
    Tensor t_emb = time_embed_.forward(t, backend, stream);
    // Expand batch dim if B > 1.
    if (B > 1) {
        // Broadcast [1, D] → [B, D].
        // Simple: copy B times (rarely needed in robot inference where B=1).
        Tensor t_emb_b = backend->alloc(Shape({B, D}), dt, stream);
        for (int64_t b = 0; b < B; ++b) {
            // TODO: implement a proper expand/broadcast op for efficiency.
            backend->copy(t_emb_b.slice(b, b + 1), t_emb, stream);
        }
        t_emb = t_emb_b;
    }

    // ── 2. Project noisy action to token space ────────────────────────────
    // x_t: [B, T, action_dim] → action_proj: [B, T, D]
    Tensor action_proj = action_in_proj_.forward(x_t, backend, stream);

    // ── 3. Concatenate condition + action tokens ───────────────────────────
    // seq: [B, S_cond + T, D]
    Tensor seq = backend->alloc(Shape({B, S_total, D}), dt, stream);
    backend->cat({condition, action_proj}, seq, /*dim=*/1, stream);

    // ── 4. DiT forward: N_layers transformer blocks ───────────────────────
    // Alternate between seq (input) and buf_block_out_ (output).
    // Condition embedding c = t_emb used by all blocks' adaLN.
    Tensor cur = seq;
    for (auto& block : dit_blocks_) {
        Tensor nxt = backend->alloc(Shape({B, S_total, D}), dt, stream);
        block.forward(cur, t_emb, S_cond, nxt, backend, stream);
        cur = nxt;
    }

    // ── 5. Final layer → velocity [B, T, action_dim] ──────────────────────
    final_layer_.forward(cur, t_emb, T, velocity, backend, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// IVLAModel::decode_action
// ─────────────────────────────────────────────────────────────────────────────
Tensor RDT1BModel::decode_action(const Tensor& raw,
                                  BackendPtr backend,
                                  StreamHandle stream) {
    if (!action_norm_.mean.is_valid()) return raw;

    // Un-normalise: action = raw * std + mean
    DType dt = DType::Float32;
    Tensor out = backend->alloc(raw.shape(), dt, stream);
    backend->cast(raw, out, stream);

    int64_t B = out.shape()[0];
    int64_t T = out.shape()[1];
    Tensor std_b = action_norm_.std.view({1, 1, cfg_.action_dim});
    Tensor mean_b = action_norm_.mean.view({1, 1, cfg_.action_dim});

    backend->mul(out, std_b, out, stream);
    backend->add(out, mean_b, out, stream);

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<RDT1BModel> load_rdt1b(const std::string& checkpoint_dir,
                                        Device device) {
    BackendPtr backend = get_backend(device);

    // Load config.json from the checkpoint directory.
    auto cfg_path = std::filesystem::path(checkpoint_dir) / "config.json";
    RDT1BConfig cfg = std::filesystem::exists(cfg_path)
                    ? RDT1BConfig::from_json(cfg_path.string())
                    : RDT1BConfig{};  // defaults match published model

    auto model = std::make_shared<RDT1BModel>(cfg, backend);
    model->load_weights(checkpoint_dir);
    return model;
}

}  // namespace rdt1b
}  // namespace sf
