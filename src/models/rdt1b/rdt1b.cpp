// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/loader/safetensors.h"
#include <iostream>

#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace fs = std::filesystem;

namespace sf {
namespace rdt1b {

// ─────────────────────────────────────────────────────────────────────────────
// Utility: load weight from WeightMap with dtype cast and device upload.
// ─────────────────────────────────────────────────────────────────────────────
static Tensor lw(const WeightMap& weights, const std::string& key,
                 DType dt, BackendPtr backend, StreamHandle stream) {
    return load_weight_from_map(weights, key, dt, backend, stream);
}

// ─────────────────────────────────────────────────────────────────────────────
// MlpAdaptor::load
//
// mlp2x_gelu: Sequential(Linear(in, hidden), GELU, Linear(hidden, hidden))
//   weight keys: "0.weight", "0.bias", "2.weight", "2.bias"
//
// mlp3x_gelu: Sequential(Linear(in, h), GELU, Linear(h, h), GELU, Linear(h, h))
//   weight keys: "0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"
// ─────────────────────────────────────────────────────────────────────────────
void RDT1BModel::MlpAdaptor::load(const WeightMap& weights,
                                    const std::string& prefix,
                                    int64_t in_dim, int64_t hidden_dim,
                                    int64_t depth,
                                    DType dt, BackendPtr backend,
                                    StreamHandle stream) {
    in_dim_     = in_dim;
    hidden_dim_ = hidden_dim;
    depth_      = static_cast<int>(depth);

    fc0_weight_ = lw(weights, prefix + "0.weight", dt, backend, stream);
    fc0_bias_   = lw(weights, prefix + "0.bias",   dt, backend, stream);
    fc2_weight_ = lw(weights, prefix + "2.weight", dt, backend, stream);
    fc2_bias_   = lw(weights, prefix + "2.bias",   dt, backend, stream);

    if (depth_ >= 3) {
        fc4_weight_ = lw(weights, prefix + "4.weight", dt, backend, stream);
        fc4_bias_   = lw(weights, prefix + "4.bias",   dt, backend, stream);
    }
}

Tensor RDT1BModel::MlpAdaptor::forward(const Tensor& x,
                                         BackendPtr backend,
                                         StreamHandle stream) const {
    DType   dt  = x.dtype();
    int64_t B   = x.shape()[0];
    int64_t S   = x.shape()[1];

    // fc0: [B*S, in_dim] → [B*S, hidden_dim]
    Tensor x_2d = x.view({B * S, in_dim_});
    Tensor h    = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
    backend->gemm(x_2d, fc0_weight_, h, 1.f, 0.f, false, true, stream);
    backend->add(h, fc0_bias_.view({1, hidden_dim_}), h, stream);
    backend->gelu(h, h, stream);

    // fc2: [B*S, hidden_dim] → [B*S, hidden_dim]
    Tensor h2 = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
    backend->gemm(h, fc2_weight_, h2, 1.f, 0.f, false, true, stream);
    backend->add(h2, fc2_bias_.view({1, hidden_dim_}), h2, stream);

    if (depth_ >= 3) {
        // gelu + fc4
        backend->gelu(h2, h2, stream);
        Tensor h3 = backend->alloc(Shape({B * S, hidden_dim_}), dt, stream);
        backend->gemm(h2, fc4_weight_, h3, 1.f, 0.f, false, true, stream);
        backend->add(h3, fc4_bias_.view({1, hidden_dim_}), h3, stream);
        return h3.view({B, S, hidden_dim_});
    }

    return h2.view({B, S, hidden_dim_});
}

// ─────────────────────────────────────────────────────────────────────────────
// RDT1BModel constructor
// ─────────────────────────────────────────────────────────────────────────────
RDT1BModel::RDT1BModel(RDT1BConfig config, BackendPtr backend)
    : cfg_(std::move(config)), backend_(backend) {
    cfg_.validate();
    blocks_.resize(cfg_.num_layers);
}

// ─────────────────────────────────────────────────────────────────────────────
// load_weights
// ─────────────────────────────────────────────────────────────────────────────
void RDT1BModel::load_weights(const std::string& checkpoint_dir,
                               StreamHandle stream) {
    if (!fs::is_directory(checkpoint_dir))
        throw std::runtime_error("Checkpoint directory not found: " + checkpoint_dir);

    WeightMap weights;
    for (auto& entry : fs::directory_iterator(checkpoint_dir)) {
        if (entry.path().extension() == ".safetensors") {
            auto shard = SafeTensorsLoader::load(entry.path().string());
            weights.merge(shard);
        }
    }
    if (weights.empty())
        throw std::runtime_error("No safetensors files found in: " + checkpoint_dir);

    StreamHandle s = stream ? stream : backend_->create_stream();
    DType dt = cfg_.compute_dtype;

    // ── Adaptors ──────────────────────────────────────────────────────────
    // lang_adaptor: mlp2x_gelu  (4096 → D → D)
    lang_adaptor_.load(weights, "lang_adaptor.",
                       cfg_.lang_token_dim, cfg_.hidden_dim, 2, dt, backend_, s);
    // img_adaptor: mlp2x_gelu  (1152 → D → D)
    img_adaptor_.load(weights, "img_adaptor.",
                      cfg_.img_token_dim, cfg_.hidden_dim, 2, dt, backend_, s);
    // state_adaptor: mlp3x_gelu (2*state_dim → D → D → D)
    state_adaptor_.load(weights, "state_adaptor.",
                        cfg_.state_token_dim * 2, cfg_.hidden_dim, 3, dt, backend_, s);

    // ── Timestep / freq embedders ─────────────────────────────────────────
    t_embedder_.load(weights, "t_embedder.", cfg_, backend_, s);
    freq_embedder_.load(weights, "freq_embedder.", cfg_, backend_, s);

    // ── Learnable position embeddings ─────────────────────────────────────
    x_pos_embed_         = lw(weights, "x_pos_embed",         dt, backend_, s);
    lang_cond_pos_embed_ = lw(weights, "lang_cond_pos_embed", dt, backend_, s);
    img_cond_pos_embed_  = lw(weights, "img_cond_pos_embed",  dt, backend_, s);

    // ── 28 RDT blocks ─────────────────────────────────────────────────────
    load_dit_weights(weights, s);

    // ── Final layer ───────────────────────────────────────────────────────
    final_layer_.load(weights, "final_layer.", cfg_, backend_, s);

    // ── Optional action normalisation ─────────────────────────────────────
    if (weights.count("action_norm.mean")) {
        action_norm_.mean = lw(weights, "action_norm.mean", DType::Float32, backend_, s);
        action_norm_.std  = lw(weights, "action_norm.std",  DType::Float32, backend_, s);
    }

    backend_->sync_stream(s);
    if (!stream) backend_->destroy_stream(s);
}

void RDT1BModel::load_dit_weights(const WeightMap& weights, StreamHandle stream) {
    std::cerr << "[fw] blocks\n" << std::flush;
    for (int i = 0; i < cfg_.num_layers; ++i) {
        std::string prefix = "blocks." + std::to_string(i) + ".";
        blocks_[i].load(weights, prefix, cfg_, backend_, stream);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// encode_condition
//
// Packs: [state_traj (1), lang_cond (L), img_cond (I)]
// Returns [B, 1+L+I, D]
// ─────────────────────────────────────────────────────────────────────────────
Tensor RDT1BModel::encode_condition(const VLAInput& input,
                                     BackendPtr backend,
                                     StreamHandle stream) {
    DType dt = cfg_.compute_dtype;
    int64_t D = cfg_.hidden_dim;

    // ── Language tokens ─────────────────────────────────────────────────
    // input.language_tokens: [B, L, lang_token_dim] (pre-computed T5 embeddings)
    Tensor lang_tokens_dt;
    if (input.language_tokens.is_valid()) {
        lang_tokens_dt = backend->alloc(input.language_tokens.shape(), dt, stream);
        backend->cast(input.language_tokens, lang_tokens_dt, stream);
    } else {
        // Zero padding for missing language.
        lang_tokens_dt = backend->alloc(
            Shape({1, cfg_.max_lang_cond_len, cfg_.lang_token_dim}), dt, stream);
        backend->fill(lang_tokens_dt, 0.f, stream);
    }
    Tensor lang_cond = lang_adaptor_.forward(lang_tokens_dt, backend, stream);
    // [B, L, D]

    // ── Image tokens ─────────────────────────────────────────────────────
    // input.images: one tensor [B, img_cond_len, img_token_dim]
    Tensor img_tokens_dt;
    if (!input.images.empty() && input.images[0].is_valid()) {
        img_tokens_dt = backend->alloc(input.images[0].shape(), dt, stream);
        backend->cast(input.images[0], img_tokens_dt, stream);
    } else {
        img_tokens_dt = backend->alloc(
            Shape({1, cfg_.img_cond_len, cfg_.img_token_dim}), dt, stream);
        backend->fill(img_tokens_dt, 0.f, stream);
    }
    Tensor img_cond = img_adaptor_.forward(img_tokens_dt, backend, stream);
    // [B, I, D]

    // ── Robot state token ─────────────────────────────────────────────────
    // input.robot_state: [B, state_dim]
    // state_adaptor input = cat(state, action_mask) = [B, 1, 2*state_dim]
    // For encode_condition (not per-step), we use zero action mask.
    int64_t B = lang_cond.shape()[0];
    Tensor state_2d = backend->alloc(Shape({B, 1, cfg_.state_token_dim * 2}), dt, stream);
    backend->fill(state_2d, 0.f, stream);

    if (input.robot_state.is_valid()) {
        // Copy state into first state_dim elements, mask stays zero.
        Tensor state_dt = backend->alloc(input.robot_state.shape(), dt, stream);
        backend->cast(input.robot_state, state_dt, stream);
        // state_dt: [B, state_dim] → first half of state_2d [B, 1, state_dim]
        // Reuse state_2d's first state_dim columns.
        // Simple workaround: allocate [B, 1, 2*D] buffer, fill state part manually.
        // We rely on the fact that cat of [state, zeros] gives the right input.
        Tensor state_expanded = state_dt.view({B, 1, cfg_.state_token_dim});
        Tensor mask_zeros     = backend->alloc(
            Shape({B, 1, cfg_.state_token_dim}), dt, stream);
        backend->fill(mask_zeros, 0.f, stream);
        backend->cat({state_expanded, mask_zeros}, state_2d, /*dim=*/2, stream);
    }
    Tensor state_traj = state_adaptor_.forward(state_2d, backend, stream);
    // [B, 1, D]

    // ── Pack: [state_traj | lang_cond | img_cond] ─────────────────────────
    int64_t L = lang_cond.shape()[1];
    int64_t I = img_cond.shape()[1];
    Tensor packed = backend->alloc(Shape({B, 1 + L + I, D}), dt, stream);
    backend->cat({state_traj, lang_cond, img_cond}, packed, /*dim=*/1, stream);

    return packed;
}

// ─────────────────────────────────────────────────────────────────────────────
// denoise_step
//
// Mirrors RDTRunner.conditional_sample (single step):
//   action_traj = state_adaptor(cat(x_t, action_mask))   [B, T, D]
//   x = cat(state_traj, action_traj)                     [B, T+1, D]
//   x = cat(t_emb, freq_emb, x)                         [B, T+3, D]
//   x = x + x_pos_embed
//   lang_c = lang_cond + lang_pos_embed[:L]
//   img_c  = img_cond  + img_pos_embed
//   for i, block in blocks: x = block(x, lang_c, img_c, i)
//   out = final_layer(x)[:, -T:]                         [B, T, action_dim]
// ─────────────────────────────────────────────────────────────────────────────
void RDT1BModel::denoise_step(const Tensor& x_t, float t,
                               const Tensor& condition,
                               Tensor& velocity,
                               BackendPtr backend,
                               StreamHandle stream) {
    DType dt    = cfg_.compute_dtype;
    int64_t B   = x_t.shape()[0];
    int64_t T   = cfg_.action_horizon;
    int64_t D   = cfg_.hidden_dim;
    int64_t I   = cfg_.img_cond_len;
    int64_t L   = condition.shape()[1] - 1 - I;  // L = total - state(1) - img(I)
    int64_t A   = cfg_.action_dim;

    // ── Unpack condition ──────────────────────────────────────────────────
    Tensor state_traj = condition.slice(0, 1);       // [B, 1, D]
    Tensor lang_cond  = condition.slice(1, 1 + L);   // [B, L, D]
    Tensor img_cond   = condition.slice(1 + L, 1 + L + I);  // [B, I, D]

    // ── Action tokens via state_adaptor ───────────────────────────────────
    // Concatenate noisy action with all-ones action mask (indicating valid dims).
    Tensor x_t_dt = backend->alloc(x_t.shape(), dt, stream);
    backend->cast(x_t, x_t_dt, stream);

    Tensor action_mask = backend->alloc(Shape({B, T, A}), dt, stream);
    backend->fill(action_mask, 1.f, stream);

    // action_input: [B, T, 2*A]
    Tensor action_input = backend->alloc(Shape({B, T, 2 * A}), dt, stream);
    backend->cat({x_t_dt, action_mask}, action_input, /*dim=*/2, stream);

    Tensor action_traj = state_adaptor_.forward(action_input, backend, stream);
    // [B, T, D]

    // ── Build x sequence ──────────────────────────────────────────────────
    // x = cat(state_traj [B,1,D], action_traj [B,T,D]) → [B, T+1, D]
    Tensor x_seq = backend->alloc(Shape({B, T + 1, D}), dt, stream);
    backend->cat({state_traj, action_traj}, x_seq, /*dim=*/1, stream);

    // ── Timestep and control-freq embeddings ─────────────────────────────
    // t is in [0, 1] from the sampler; map to integer [0, num_train_timesteps-1]
    int64_t t_int = static_cast<int64_t>(std::round(t * (cfg_.num_train_timesteps - 1)));
    t_int = std::max(int64_t(0), std::min(t_int, cfg_.num_train_timesteps - 1));

    Tensor t_emb    = t_embedder_.forward(t_int, backend, stream);     // [1, D]
    Tensor freq_emb = freq_embedder_.forward(cfg_.ctrl_freq, backend, stream); // [1, D]

    // Expand to [B, 1, D] if B > 1.
    auto expand_to_batch = [&](Tensor emb) -> Tensor {
        // emb: [1, D] → [B, 1, D]
        Tensor expanded = backend->alloc(Shape({B, 1, D}), dt, stream);
        // Broadcast: copy B times.
        for (int64_t b = 0; b < B; ++b) {
            Tensor row = expanded.slice(b, b + 1);  // [1, 1, D]
            // emb is [1, D]; row is [1, 1, D] → view as [1, D] for copy
            Tensor row_2d = row.view({1, D});
            backend->copy(row_2d, emb, stream);
        }
        return expanded;
    };
    Tensor t_token    = expand_to_batch(t_emb);     // [B, 1, D]
    Tensor freq_token = expand_to_batch(freq_emb);  // [B, 1, D]

    // ── Prepend t_token, freq_token to x: [B, T+3, D] ────────────────────
    Tensor x_full = backend->alloc(Shape({B, T + 3, D}), dt, stream);
    backend->cat({t_token, freq_token, x_seq}, x_full, /*dim=*/1, stream);

    // ── Add position embeddings ───────────────────────────────────────────
    // x_pos_embed: [1, T+3, D] (T+3 = 67)
    backend->add(x_full, x_pos_embed_, x_full, stream);

    // lang_cond + lang_pos_embed[:L]
    Tensor lang_pos = lang_cond_pos_embed_.slice(0, L);  // [1, L, D]
    Tensor lang_c = backend->alloc(lang_cond.shape(), dt, stream);
    backend->add(lang_cond, lang_pos, lang_c, stream);

    // img_cond + img_pos_embed
    Tensor img_c = backend->alloc(img_cond.shape(), dt, stream);
    backend->add(img_cond, img_cond_pos_embed_, img_c, stream);

    // ── 28 RDT blocks ─────────────────────────────────────────────────────
    Tensor x_cur = x_full;
    for (int i = 0; i < cfg_.num_layers; ++i) {
        x_cur = blocks_[i].forward(x_cur, lang_c, img_c, i, backend, stream);
    }

    // ── FinalLayer + slice last T tokens ─────────────────────────────────
    // final_layer output: [B, T+3, action_dim]
    Tensor final_out = final_layer_.forward(x_cur, backend, stream);

    // Take last T tokens: [B, T, action_dim]
    velocity = final_out.slice(3, T + 3);  // [B, T, action_dim]
}

// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<RDT1BModel> load_rdt1b(const std::string& checkpoint_dir,
                                        Device device) {
    BackendPtr backend = get_backend(device);

    auto cfg_path = fs::path(checkpoint_dir) / "config.json";
    RDT1BConfig cfg = fs::exists(cfg_path)
                    ? RDT1BConfig::from_json(cfg_path.string())
                    : RDT1BConfig{};

    auto model = std::make_shared<RDT1BModel>(cfg, backend);
    model->load_weights(checkpoint_dir);
    return model;
}


// ─────────────────────────────────────────────────────────────────────────────
// RDT1BModel::forward_raw — E2E alignment test forward pass (B=1)
//
// Exactly matches the PyTorch reference in test_rdt1b_alignment.py:
//   lang_c  = lang_adaptor(lang_features) + lang_pos[:L]
//   img_c   = img_adaptor(img_features)  + img_pos
//   traj_in = cat([cat([state, noisy], dim=1), ones_mask], dim=2)
//   x_seq   = state_adaptor(traj_in)
//   x       = cat([t_emb, freq_emb, x_seq], dim=1) + x_pos_embed
//   x       = RDTBlocks(x, lang_c, img_c, alternating)
//   out     = final_layer(x)[:, -T:]
// ─────────────────────────────────────────────────────────────────────────────
Tensor RDT1BModel::forward_raw(
    const Tensor& lang_features,
    const Tensor& img_features,
    const Tensor& state_tok,
    const Tensor& noisy_action,
    int64_t t,
    int64_t freq,
    StreamHandle stream) const
{
    StreamHandle s = stream ? stream : backend_->create_stream();
    bool own_stream = !stream;

    DType dt = cfg_.compute_dtype;
    int64_t D = cfg_.hidden_dim;
    int64_t A = cfg_.action_dim;
    int64_t L = lang_features.shape()[1];
    int64_t I = img_features.shape()[1];
    int64_t T = noisy_action.shape()[1];
    int64_t S = T + 3;   // t_emb + freq_emb + 1(state) + T(actions) = T+3 = 67

    // Cast input to compute dtype if needed.
    auto cast_to = [&](const Tensor& x) -> Tensor {
        if (x.dtype() == dt) return x;
        Tensor y = backend_->alloc(x.shape(), dt, s);
        backend_->cast(x, y, s);
        return y;
    };

    Tensor lf = cast_to(lang_features);   // [1, L, lang_dim]
    Tensor imf = cast_to(img_features);   // [1, I, img_dim]
    Tensor st  = cast_to(state_tok);      // [1, 1, A]
    Tensor na  = cast_to(noisy_action);   // [1, T, A]

    // ── 1. Adapt language and image features ──────────────────────────────
    std::cerr << "[fw] lang_adaptor\n" << std::flush;
    Tensor lang_c = lang_adaptor_.forward(lf,  backend_, s);   // [1, L, D]
    std::cerr << "[fw] img_adaptor\n" << std::flush;
    Tensor img_c  = img_adaptor_.forward(imf,  backend_, s);   // [1, I, D]

    // ── 2. Build state+action trajectory for state_adaptor ────────────────
    // state_action = cat([state_tok, noisy_action], dim=1) → [T+1, A] (B=1)
    // mask = ones [T+1, A]
    // traj_in = cat([state_action, mask], dim=1 of 2D) → [T+1, 2A]

    Tensor st_2d = st.view({1, A});             // [1, A]
    Tensor na_2d = na.view({T, A});             // [T, A]
    Tensor state_action_2d = backend_->alloc(Shape({T+1, A}), dt, s);
    backend_->cat({st_2d, na_2d}, state_action_2d, 0, s);   // [T+1, A]

    Tensor mask_full = backend_->alloc(Shape({T+1, A}), dt, s);
    backend_->fill(mask_full, 1.0f, s);         // action_mask = all ones

    Tensor traj_2d = backend_->alloc(Shape({T+1, 2*A}), dt, s);
    backend_->cat({state_action_2d, mask_full}, traj_2d, 1, s);  // [T+1, 2A]

    std::cerr << "[fw] state_adaptor\n" << std::flush;
    Tensor x_seq = state_adaptor_.forward(traj_2d.view({1, T+1, 2*A}), backend_, s); // [1, T+1, D]

    // ── 3. Timestep and frequency embeddings ──────────────────────────────
    std::cerr << "[fw] t_embedder\n" << std::flush;
    Tensor t_emb   = t_embedder_.forward(t, backend_, s);      // [1, D]
    std::cerr << "[fw] freq_embedder\n" << std::flush;
    Tensor f_emb   = freq_embedder_.forward(freq, backend_, s);   // [1, D]
    std::cerr << "[fw] cat_x\n" << std::flush;

    // ── 4. Concatenate: [t_emb, freq_emb, x_seq] along dim=1 ─────────────
    // For B=1, view as 2D [*, D] and cat along dim 0.
    Tensor t_2d    = t_emb.view({1, D});
    Tensor f_2d    = f_emb.view({1, D});
    Tensor xseq_2d = x_seq.view({T+1, D});

    Tensor x_2d = backend_->alloc(Shape({S, D}), dt, s);  // [S=67, D]
    backend_->cat({t_2d, f_2d, xseq_2d}, x_2d, 0, s);
    std::cerr << "[fw] cat_done, add_pos\n" << std::flush;

    Tensor x = x_2d.view({1, S, D});   // [1, 67, D]

    // ── 5. Add x_pos_embed ────────────────────────────────────────────────
    // x_pos_embed_: [1, 67, D]  (contiguous, flat add works)
    Tensor x_flat   = x.view({S * D});
    Tensor pos_flat = x_pos_embed_.view({S * D});
    Tensor x_sum    = backend_->alloc(Shape({S * D}), dt, s);
    backend_->add(x_flat, pos_flat, x_sum, s);
    x = x_sum.view({1, S, D});

    // ── 6. Add position embeddings to lang_c and img_c ────────────────────
    // lang_c: [1, L, D], lang_cond_pos_embed_: [1, max_lang, D]
    // lang_c += lang_cond_pos_embed_[:, :L, :]
    Tensor lc_2d      = lang_c.view({L * D});
    Tensor lpos_slice = lang_cond_pos_embed_.view({cfg_.max_lang_cond_len, D}).slice(0, L).view({L * D});
    Tensor lc_sum     = backend_->alloc(Shape({L * D}), dt, s);
    backend_->add(lc_2d, lpos_slice, lc_sum, s);
    lang_c = lc_sum.view({1, L, D});

    // img_c: [1, I, D], img_cond_pos_embed_: [1, I, D]
    Tensor ic_2d  = img_c.view({I * D});
    Tensor ipos   = img_cond_pos_embed_.view({I * D});
    Tensor ic_sum = backend_->alloc(Shape({I * D}), dt, s);
    backend_->add(ic_2d, ipos, ic_sum, s);
    img_c = ic_sum.view({1, I, D});

    // ── 7. Run 28 RDT blocks ──────────────────────────────────────────────
    // SF_DEBUG_NUM_BLOCKS=N  – limit to N transformer blocks (default: all)
    // SF_DEBUG_SAVE_X=path   – save x_before_blocks tensor to path
    int max_blocks = cfg_.num_layers;
    if (const char* e = std::getenv("SF_DEBUG_NUM_BLOCKS")) {
        max_blocks = std::min(max_blocks, std::atoi(e));
        std::cerr << "[fw] DEBUG: running " << max_blocks
                  << "/" << cfg_.num_layers << " blocks\n" << std::flush;
    }
    if (const char* sp = std::getenv("SF_DEBUG_SAVE_X")) {
        Tensor xf = backend_->alloc(x.shape(), DType::Float32, s);
        backend_->cast(x, xf, s);
        Tensor xp = backend_->alloc_pinned(x.shape(), DType::Float32);
        backend_->copy(xp, xf, s);
        backend_->sync_stream(s);
        size_t n = 1;
        for (int i = 0; i < x.ndim(); ++i) n *= static_cast<size_t>(x.shape()[i]);
        std::ofstream dbg(sp, std::ios::binary);
        uint32_t magic = 0xAF12BF16u, ndim = static_cast<uint32_t>(x.ndim());
        dbg.write(reinterpret_cast<const char*>(&magic), 4);
        dbg.write(reinterpret_cast<const char*>(&ndim), 4);
        for (int i = 0; i < x.ndim(); ++i) {
            int32_t d = static_cast<int32_t>(x.shape()[i]);
            dbg.write(reinterpret_cast<const char*>(&d), 4);
        }
        const float* ptr = static_cast<const float*>(xp.raw_data_ptr());
        dbg.write(reinterpret_cast<const char*>(ptr), n * sizeof(float));
        std::cerr << "[fw] DEBUG: saved x_before_blocks[" << n << "] to " << sp << "\n" << std::flush;
    }
    for (int i = 0; i < max_blocks; ++i) {
        x = blocks_[i].forward(x, lang_c, img_c, i, backend_, s);
    }

    std::cerr << "[fw] blocks_done\n" << std::flush;
    backend_->sync_stream(s);  // flush pending ops before final layer
    backend_->sync_device();   // catch any async CUDA errors
    std::cerr << "[fw] final_layer\n" << std::flush;
    // ── 8. Final layer → [1, S, A] ────────────────────────────────────────
    Tensor out = final_layer_.forward(x, backend_, s);   // [1, 67, A]

    // ── 9. Return last T tokens (skip t_emb, freq_emb, state) ─────────────
    // For B=1: view [S, A], slice(skip, S), view [1, T, A]
    int64_t skip = S - T;   // = 3
    Tensor out_2d     = out.view({S, A});
    Tensor action_2d  = out_2d.slice(skip, S);   // [T, A]
    Tensor result     = action_2d.view({1, T, A});

    if (own_stream) {
        backend_->sync_stream(s);
        backend_->destroy_stream(s);
    }
    return result;
}
}  // namespace rdt1b
}  // namespace sf
