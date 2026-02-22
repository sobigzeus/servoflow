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

    get("hidden_size",        cfg.hidden_dim);
    get("num_hidden_layers",  cfg.num_layers);
    get("num_attention_heads",cfg.num_heads);
    get("mlp_ratio",          cfg.mlp_ratio);
    get("action_dim",         cfg.action_dim);
    get("action_horizon",     cfg.action_horizon);
    get("freq_dim",           cfg.freq_dim);
    get("vision_embed_dim",   cfg.vision_embed_dim);
    get("num_image_tokens",   cfg.num_image_tokens);
    get("num_cameras",        cfg.num_cameras);
    get("lang_max_tokens",    cfg.lang_max_tokens);
    get("state_dim",          cfg.state_dim);
    get("layer_norm_eps",     cfg.layer_norm_eps);

    // Compute derived fields.
    cfg.head_dim           = cfg.hidden_dim / cfg.num_heads;
    cfg.time_embed_dim     = cfg.hidden_dim;
    cfg.projected_vision_dim = cfg.hidden_dim;
    cfg.projected_lang_dim   = cfg.hidden_dim;

    cfg.validate();
    return cfg;
}

void RDT1BConfig::validate() const {
    if (hidden_dim % num_heads != 0)
        throw std::invalid_argument(
            "hidden_dim must be divisible by num_heads");
    if (hidden_dim <= 0 || num_layers <= 0)
        throw std::invalid_argument("hidden_dim and num_layers must be > 0");
    if (action_dim <= 0 || action_horizon <= 0)
        throw std::invalid_argument("action_dim and action_horizon must be > 0");
}

}  // namespace rdt1b
}  // namespace sf
