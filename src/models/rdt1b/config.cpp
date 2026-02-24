// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/rdt1b/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>

namespace sf {
namespace rdt1b {

RDT1BConfig RDT1BConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("RDT1BConfig: cannot open " + path);

    nlohmann::json j;
    f >> j;

    RDT1BConfig cfg;

    auto get = [&]<typename T>(const char* key, T& field) {
        if (j.contains(key)) field = j[key].get<T>();
    };

    // ── Common top-level fields (present in both HF and ServoFlow formats) ──
    get("action_dim",        cfg.action_dim);
    get("pred_horizon",      cfg.action_horizon);   // HF key
    get("img_cond_len",      cfg.img_cond_len);
    get("img_token_dim",     cfg.img_token_dim);
    get("lang_token_dim",    cfg.lang_token_dim);
    get("max_lang_cond_len", cfg.max_lang_cond_len);
    get("state_token_dim",   cfg.state_token_dim);

    // ── ServoFlow flat format (written by hf_to_servoflow.py) ────────────
    // These keys are at top-level in the converted checkpoint.
    get("hidden_size",             cfg.hidden_dim);
    get("num_hidden_layers",       cfg.num_layers);
    get("num_attention_heads",     cfg.num_heads);
    get("num_train_timesteps",     cfg.num_train_timesteps);
    get("num_inference_timesteps", cfg.num_inference_timesteps);

    // ── HuggingFace nested format (for loading directly from HF config) ──
    if (j.contains("rdt")) {
        auto& r = j["rdt"];
        if (r.contains("hidden_size")) cfg.hidden_dim = r["hidden_size"].get<int64_t>();
        if (r.contains("depth"))       cfg.num_layers = r["depth"].get<int64_t>();
        if (r.contains("num_heads"))   cfg.num_heads  = r["num_heads"].get<int64_t>();
    }
    if (j.contains("noise_scheduler")) {
        auto& ns = j["noise_scheduler"];
        if (ns.contains("num_train_timesteps"))
            cfg.num_train_timesteps = ns["num_train_timesteps"].get<int64_t>();
        if (ns.contains("num_inference_timesteps"))
            cfg.num_inference_timesteps = ns["num_inference_timesteps"].get<int64_t>();
    }

    // ── ServoFlow-specific overrides ──────────────────────────────────────
    get("freq_dim",     cfg.freq_dim);
    get("rms_norm_eps", cfg.rms_norm_eps);

    // compute_dtype (ServoFlow format: "float16", "bfloat16", "float32")
    if (j.contains("compute_dtype")) {
        cfg.compute_dtype = dtype_from_string(j["compute_dtype"].get<std::string>());
    }

    // Derived.
    cfg.head_dim       = cfg.hidden_dim / cfg.num_heads;
    cfg.time_embed_dim = cfg.hidden_dim;

    cfg.validate();
    return cfg;
}

void RDT1BConfig::validate() const {
    if (hidden_dim % num_heads != 0)
        throw std::invalid_argument("hidden_dim must be divisible by num_heads");
    if (hidden_dim <= 0 || num_layers <= 0)
        throw std::invalid_argument("hidden_dim and num_layers must be > 0");
    if (action_dim <= 0 || action_horizon <= 0)
        throw std::invalid_argument("action_dim and action_horizon must be > 0");
}

}  // namespace rdt1b
}  // namespace sf
