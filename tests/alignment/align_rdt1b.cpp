// SPDX-License-Identifier: Apache-2.0
// align_rdt1b — E2E alignment binary for ServoFlow RDT-1B.
//
// Usage:
//   align_rdt1b <checkpoint_dir> <input.safetensors> <output.bin>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "servoflow/models/rdt1b/config.h"
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/loader/safetensors.h"
#include "servoflow/backend/backend.h"
#include "servoflow/core/dtype.h"

namespace fs = std::filesystem;
using namespace sf;
using namespace sf::rdt1b;

// ── Write raw float32 binary ──────────────────────────────────────────────────
// Format: 4B magic (0xAF12BF16), 4B ndim, ndim×4B shape, then float32 data.
void write_f32_bin(const std::string& path, const std::vector<float>& data,
                   const std::vector<int32_t>& shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for writing: " + path);
    uint32_t magic = 0xAF12BF16u;
    uint32_t ndim  = static_cast<uint32_t>(shape.size());
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&ndim),  4);
    f.write(reinterpret_cast<const char*>(shape.data()), ndim * 4);
    f.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
    std::cout << "  Wrote " << data.size() << " floats to " << path << "\n";
}

// ── Copy GPU tensor → CPU float32 ────────────────────────────────────────────
std::vector<float> gpu_to_host_f32(const Tensor& t, BackendPtr backend) {
    size_t n = 1;
    for (int i = 0; i < t.ndim(); ++i) n *= static_cast<size_t>(t.shape()[i]);

    // Cast to float32 on device, then copy to pinned host.
    Tensor t_f32 = backend->alloc(t.shape(), DType::Float32);
    backend->cast(t, t_f32);

    Tensor host = backend->alloc_pinned(t.shape(), DType::Float32);
    backend->copy(host, t_f32);
    backend->sync_device();

    const float* ptr = static_cast<const float*>(host.raw_data_ptr());
    return std::vector<float>(ptr, ptr + n);
}

// ── Upload CPU float32 → GPU tensor with dtype cast ──────────────────────────
Tensor host_f32_to_gpu(const void* host_data, const Shape& shape,
                        DType dtype, BackendPtr backend) {
    // Wrap raw host data as pinned tensor.
    Tensor host_f32 = backend->alloc_pinned(shape, DType::Float32);
    size_t nbytes = 1;
    for (int i = 0; i < shape.ndim(); ++i) nbytes *= static_cast<size_t>(shape[i]);
    nbytes *= dtype_size(DType::Float32);
    std::memcpy(host_f32.raw_data_ptr(), host_data, nbytes);

    // Copy to GPU.
    Tensor gpu_f32 = backend->alloc(shape, DType::Float32);
    backend->copy(gpu_f32, host_f32);

    if (dtype == DType::Float32) return gpu_f32;

    Tensor gpu = backend->alloc(shape, dtype);
    backend->cast(gpu_f32, gpu);
    return gpu;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: align_rdt1b <checkpoint_dir> <input.safetensors> <output.bin> <t> <freq>\n";
        return 1;
    }
    const std::string ckpt_dir   = argv[1];
    const std::string input_sf   = argv[2];
    const std::string output_bin = argv[3];
    int64_t t_arg   = (argc > 4) ? std::stoll(argv[4]) : 500LL;
    int64_t freq_arg = (argc > 5) ? std::stoll(argv[5]) : 25LL;

    std::cout << "=== ServoFlow RDT-1B E2E Alignment Binary ===\n";
    std::cout << "  checkpoint: " << ckpt_dir   << "\n";
    std::cout << "  inputs:     " << input_sf   << "\n";
    std::cout << "  output:     " << output_bin << "\n\n";

    // ── Load model ────────────────────────────────────────────────────────
    std::cout << "[1/3] Loading model...\n";
    auto backend = get_backend(DeviceType::CUDA, 0);
    auto cfg     = RDT1BConfig::from_json(ckpt_dir + "/config.json");
    std::cout << "  Config: depth=" << cfg.num_layers
              << " hidden=" << cfg.hidden_dim
              << " dtype=" << dtype_name(cfg.compute_dtype) << "\n";

    RDT1BModel model(cfg, backend);
    model.load_weights(ckpt_dir);
    std::cout << "  Weights loaded.\n\n";

    // ── Load inputs ───────────────────────────────────────────────────────
    std::cout << "[2/3] Loading inputs...\n";
    auto inputs = SafeTensorsLoader::load(input_sf);

    DType dt = cfg.compute_dtype;

    auto get_tensor = [&](const std::string& key) -> Tensor {
        auto it = inputs.find(key);
        if (it == inputs.end())
            throw std::runtime_error("Input tensor not found: " + key);
        const Tensor& cpu_t = it->second;
        std::cout << "  " << key << ": [";
        for (int i = 0; i < cpu_t.ndim(); ++i)
            std::cout << (i?",":"") << cpu_t.shape()[i];
        std::cout << "] " << dtype_name(cpu_t.dtype()) << "\n";
        // cpu_t is on CPU (float32). Upload to GPU with dtype cast.
        return host_f32_to_gpu(cpu_t.raw_data_ptr(), cpu_t.shape(), dt, backend);
    };

    Tensor lang_tok  = get_tensor("lang_tok");
    Tensor img_tok   = get_tensor("img_tok");
    Tensor state_tok = get_tensor("state_tok");
    Tensor noisy_act = get_tensor("noisy_act");

    int64_t t_step = t_arg;
    int64_t freq   = freq_arg;
    std::cout << "  t=" << t_step << "  freq=" << freq << "\n\n";

    // ── Run forward pass ──────────────────────────────────────────────────
    std::cout << "[3/3] Running forward_raw...\n";
    Tensor out = model.forward_raw(lang_tok, img_tok, state_tok,
                                    noisy_act, t_step, freq);

    std::cout << "  Output: [";
    for (int i = 0; i < out.ndim(); ++i)
        std::cout << (i?",":"") << out.shape()[i];
    std::cout << "] " << dtype_name(out.dtype()) << "\n";

    // Write float32 binary.
    auto out_data = gpu_to_host_f32(out, backend);
    std::vector<int32_t> shape;
    for (int i = 0; i < out.ndim(); ++i)
        shape.push_back(static_cast<int32_t>(out.shape()[i]));
    write_f32_bin(output_bin, out_data, shape);

    std::cout << "\n=== Done ===\n";
    return 0;
}
