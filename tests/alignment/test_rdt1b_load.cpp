// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <filesystem>
#include <iostream>

#include "servoflow/models/rdt1b/config.h"
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/backend/backend.h"
#include "servoflow/core/dtype.h"

namespace fs = std::filesystem;
using namespace sf;
using namespace sf::rdt1b;

static std::string checkpoint_dir() {
    const char* env = std::getenv("SF_RDT1B_CHECKPOINT");
    if (env) return std::string(env);
    return "/workspace/servoflow/tests/alignment/sf_checkpoint";
}

static bool has_checkpoint() {
    fs::path p(checkpoint_dir());
    return fs::is_directory(p) &&
           fs::exists(p / "config.json") &&
           fs::exists(p / "model.safetensors");
}

TEST(RDT1BLoad, ConfigParsed) {
    if (!has_checkpoint()) {
        GTEST_SKIP() << "Checkpoint not found at " << checkpoint_dir();
    }
    auto cfg = RDT1BConfig::from_json(checkpoint_dir() + "/config.json");

    EXPECT_EQ(cfg.hidden_dim,    2048);
    EXPECT_EQ(cfg.num_layers,    28);
    EXPECT_EQ(cfg.num_heads,     32);
    EXPECT_EQ(cfg.head_dim,      64);
    EXPECT_EQ(cfg.action_dim,    128);
    EXPECT_EQ(cfg.action_horizon, 64);
    EXPECT_EQ(cfg.img_cond_len,  4374);
    EXPECT_EQ(cfg.compute_dtype, DType::BFloat16);

    std::cout << "  Config OK: depth=" << cfg.num_layers
              << " hidden=" << cfg.hidden_dim
              << " dtype=" << dtype_name(cfg.compute_dtype) << "\n";
}

TEST(RDT1BLoad, WeightsLoaded) {
    if (!has_checkpoint()) {
        GTEST_SKIP() << "Checkpoint not found at " << checkpoint_dir();
    }
    auto cfg     = RDT1BConfig::from_json(checkpoint_dir() + "/config.json");
    auto backend = get_backend(DeviceType::CUDA, 0);

    ASSERT_NO_THROW({
        RDT1BModel model(cfg, backend);
        model.load_weights(checkpoint_dir());
        std::cout << "  Weights loaded from " << checkpoint_dir() << "\n";
    });
}
