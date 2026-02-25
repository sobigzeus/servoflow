// SPDX-License-Identifier: Apache-2.0
// bench_rdt1b_kernel — measure pure forward_raw kernel time for RDT-1B.
//
// Usage:
//   bench_rdt1b_kernel <checkpoint_dir> <input.safetensors> [warmup] [iters]
//
// - Loads the ServoFlow RDT-1B model once.
// - Uploads inputs once.
// - Runs `warmup` forwards (not timed).
// - Runs `iters` forwards, timing only the forward_raw kernel with CUDA events.
// - Prints average / min / max latency in milliseconds.

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "servoflow/models/rdt1b/config.h"
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/loader/safetensors.h"
#include "servoflow/backend/backend.h"
#include "servoflow/core/dtype.h"

namespace fs = std::filesystem;
using namespace sf;
using namespace sf::rdt1b;

// Upload CPU float32 tensor to GPU with cast to cfg.compute_dtype.
static Tensor host_f32_to_gpu(const Tensor& cpu_t, DType dtype,
                              BackendPtr backend) {
    // Wrap CPU tensor as pinned, then copy to device.
    Tensor host_f32 = backend->alloc_pinned(cpu_t.shape(), DType::Float32);
    assert(cpu_t.dtype() == DType::Float32);
    std::memcpy(host_f32.raw_data_ptr(),
                cpu_t.raw_data_ptr(),
                cpu_t.nbytes());

    Tensor gpu_f32 = backend->alloc(cpu_t.shape(), DType::Float32);
    backend->copy(gpu_f32, host_f32);

    if (dtype == DType::Float32) return gpu_f32;

    Tensor gpu = backend->alloc(cpu_t.shape(), dtype);
    backend->cast(gpu_f32, gpu);
    return gpu;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_rdt1b_kernel <checkpoint_dir> <input.safetensors> "
                     "[warmup=5] [iters=50]\n";
        return 1;
    }
    const std::string ckpt_dir = argv[1];
    const std::string input_sf = argv[2];
    int warmup = (argc > 3) ? std::stoi(argv[3]) : 5;
    int iters  = (argc > 4) ? std::stoi(argv[4]) : 50;

    std::cout << "=== ServoFlow RDT-1B kernel benchmark ===\n";
    std::cout << "  checkpoint: " << ckpt_dir << "\n";
    std::cout << "  inputs    : " << input_sf << "\n";
    std::cout << "  warmup    : " << warmup << "\n";
    std::cout << "  iters     : " << iters << "\n\n";

    // Backend on CUDA device 0 (outside may restrict visible devices).
    auto backend = get_backend(DeviceType::CUDA, 0);

    // Load config and model.
    auto cfg_path = fs::path(ckpt_dir) / "config.json";
    auto cfg = RDT1BConfig::from_json(cfg_path.string());
    std::cout << "  Config: depth=" << cfg.num_layers
              << " hidden=" << cfg.hidden_dim
              << " dtype=" << dtype_name(cfg.compute_dtype) << "\n";

    RDT1BModel model(cfg, backend);
    model.load_weights(ckpt_dir);
    std::cout << "  Weights loaded.\n";

    // Load inputs.
    auto inputs = SafeTensorsLoader::load(input_sf);
    auto get = [&](const std::string& key) -> const Tensor& {
        auto it = inputs.find(key);
        if (it == inputs.end())
            throw std::runtime_error("Missing input tensor: " + key);
        return it->second;
    };

    DType dt = cfg.compute_dtype;
    const Tensor& lang_tok_cpu  = get("lang_tok");
    const Tensor& img_tok_cpu   = get("img_tok");
    const Tensor& state_tok_cpu = get("state_tok");
    const Tensor& noisy_act_cpu = get("noisy_act");

    Tensor lang_tok  = host_f32_to_gpu(lang_tok_cpu,  dt, backend);
    Tensor img_tok   = host_f32_to_gpu(img_tok_cpu,   dt, backend);
    Tensor state_tok = host_f32_to_gpu(state_tok_cpu, dt, backend);
    Tensor noisy_act = host_f32_to_gpu(noisy_act_cpu, dt, backend);

    // We don't save t/freq in safetensors; use typical values.
    int64_t t_step = 500;
    int64_t freq   = 25;

    std::cout << "  Inputs uploaded to GPU.\n\n";

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        Tensor out = model.forward_raw(lang_tok, img_tok, state_tok,
                                       noisy_act, t_step, freq);
        (void)out;
    }
    backend->sync_device();
    std::cout << "  Warmup done.\n";

    // Benchmark with CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times_ms;
    times_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start, /*stream=*/nullptr);
        Tensor out = model.forward_raw(lang_tok, img_tok, state_tok,
                                       noisy_act, t_step, freq);
        (void)out;
        backend->sync_device();
        cudaEventRecord(stop, /*stream=*/nullptr);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        times_ms.push_back(ms);
        if (i == 0) {
            std::cout << "  iter 0: " << ms << " ms\n";
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double sum = 0.0;
    float tmin = times_ms.front();
    float tmax = times_ms.front();
    for (float v : times_ms) {
        sum += v;
        if (v < tmin) tmin = v;
        if (v > tmax) tmax = v;
    }
    double avg = sum / static_cast<double>(times_ms.size());

    std::cout << "\nKernel-only forward_raw latency over " << times_ms.size() << " iters:\n";
    std::cout << "  avg = " << avg << " ms\n";
    std::cout << "  min = " << tmin << " ms\n";
    std::cout << "  max = " << tmax << " ms\n";
    std::cout << "=== Done ===\n";
    return 0;
}

