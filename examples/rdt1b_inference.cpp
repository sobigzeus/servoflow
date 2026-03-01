// SPDX-License-Identifier: Apache-2.0
// ServoFlow — RDT-1B Inference Benchmark
//
// Demonstrates how to load RDT-1B and run inference with detailed performance profiling.
//
// Usage:
//   ./rdt1b_inference <checkpoint_dir> [num_steps]

#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/rdt1b/rdt1b.h"
#include "servoflow/sampling/sampler.h"
#include "servoflow/backend/cuda/cuda_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace sf;

// Helper for formatted time printing
std::string format_duration(double ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << ms << " ms";
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint_dir> [num_steps]\n";
        return 1;
    }
    std::string checkpoint_dir = argv[1];
    int num_steps = (argc > 2) ? std::atoi(argv[2]) : 50; // Default to 50 steps for stable stats

    // 1. Initialize Backend (CUDA device 0)
    auto backend = std::make_shared<cuda::CUDABackend>(0);
    std::cout << "======================================================================\n";
    std::cout << "  ServoFlow RDT-1B Benchmark\n";
    std::cout << "======================================================================\n";
    std::cout << "  Backend    : " << backend->caps().device_name << "\n";

    // 2. Configure Engine
    EngineConfig config;
    config.device = Device(DeviceType::CUDA, 0);
    config.compute_dtype = DType::Float16; // FP16 for speed
    config.num_denoise_steps = 10;        // Standard for RDT-1B
    config.use_cuda_graph = true;         // Enable CUDA Graph for 1.6x speedup
    config.pinned_output = true;          // Fast D2H transfer
    config.cache_condition = true;        // Enable condition caching

    std::cout << "  Steps      : " << config.num_denoise_steps << "\n";
    std::cout << "  Precision  : FP16\n";
    std::cout << "  CUDA Graph : " << (config.use_cuda_graph ? "Enabled" : "Disabled") << "\n";
    std::cout << "  Cond Cache : " << (config.cache_condition ? "Enabled" : "Disabled") << "\n";

    // 3. Load Model
    auto t_load_start = std::chrono::high_resolution_clock::now();
    std::cout << "  Loading model from " << checkpoint_dir << "...\n";
    auto model = rdt1b::load_rdt1b(checkpoint_dir, backend, config.device);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "  Model loaded in " << format_duration(load_time) << "\n";
    
    // 4. Create Sampler (Flow Matching Euler)
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/true);

    // 5. Create Inference Engine
    InferenceEngine engine(model, sampler, config);

    // 6. Prepare Inputs
    // a) Language Embeddings (T5-XXL): [1, 32, 4096]
    int64_t lang_len = 32;
    Tensor lang_embeds = backend->alloc({1, lang_len, 4096}, DType::Float16);
    backend->fill(lang_embeds, 0.01f);

    // b) Image Embeddings (SigLIP): [1, 4374, 1152]
    int64_t img_len = 4374;
    Tensor img_embeds = backend->alloc({1, img_len, 1152}, DType::Float16);
    backend->fill(img_embeds, 0.02f);

    // c) Robot State (Proprioception): [1, 14] -> 内部投影到 [1, 128]
    // Note: Assuming model handles 14-dim input correctly or we provide matched dim.
    // For benchmark stability we use 128 to match test weights expectation if any.
    int64_t state_dim = 128; 
    Tensor robot_state = backend->alloc({1, state_dim}, DType::Float16);
    backend->fill(robot_state, 0.0f);

    VLAInput input;
    input.language_tokens = lang_embeds;
    input.images = {img_embeds};
    input.robot_state = robot_state;

    // 7. Warmup
    std::cout << "  Warmup (3 iterations)...\n";
    for(int i=0; i<3; ++i) {
        engine.infer(input);
    }
    backend->sync_device();

    // 8. Inference Loop
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Running " << num_steps << " inference steps...\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Step | Latency (Engine) | Latency (Wall) | FPS\n";

    std::vector<double> latencies;
    latencies.reserve(num_steps);
    
    auto t_bench_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_steps; ++i) {
        auto t_step_start = std::chrono::high_resolution_clock::now();

        // Execute inference
        VLAOutput out = engine.infer(input);
        
        // Ensure GPU sync for wall clock measurement
        // (Note: engine.infer already syncs D2H stream, but we want full system sync for benchmark)
        // backend->sync_device(); 

        auto t_step_end = std::chrono::high_resolution_clock::now();
        double wall_latency = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        
        latencies.push_back(out.latency_ms); // Use engine reported latency (GPU compute time)

        if (i < 10 || i % 10 == 0) {
            std::cout << "  " << std::setw(4) << i 
                      << " | " << std::setw(16) << format_duration(out.latency_ms)
                      << " | " << std::setw(14) << format_duration(wall_latency)
                      << " | " << std::fixed << std::setprecision(1) << (1000.0 / out.latency_ms) << "\n";
        }
    }
    auto t_bench_end = std::chrono::high_resolution_clock::now();

    // 9. Statistics
    double min_lat = *std::min_element(latencies.begin(), latencies.end());
    double max_lat = *std::max_element(latencies.begin(), latencies.end());
    double avg_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    
    // Sort for percentiles
    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    double p50 = sorted_lat[static_cast<size_t>(latencies.size() * 0.50)];
    double p95 = sorted_lat[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = sorted_lat[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "======================================================================\n";
    std::cout << "  BENCHMARK RESULTS\n";
    std::cout << "======================================================================\n";
    std::cout << "  Samples      : " << num_steps << "\n";
    std::cout << "  Avg Latency  : " << format_duration(avg_lat) << "\n";
    std::cout << "  Min Latency  : " << format_duration(min_lat) << "\n";
    std::cout << "  Max Latency  : " << format_duration(max_lat) << "\n";
    std::cout << "  P50 Latency  : " << format_duration(p50) << "\n";
    std::cout << "  P95 Latency  : " << format_duration(p95) << "\n";
    std::cout << "  P99 Latency  : " << format_duration(p99) << "\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Throughput   : " << std::fixed << std::setprecision(2) << (1000.0 / avg_lat) << " Hz (Control Freq)\n";
    std::cout << "======================================================================\n";

    // Ensure all GPU work is done before destruction
    backend->sync_device();
    
    return 0;
}
