#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/rdt1b/rdt1b.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace sf;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [steps=10] [iters=20]" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    int steps = (argc > 2) ? std::atoi(argv[2]) : 10;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::cout << "Benchmarking ServoFlow C++ RDT-1B" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Steps: " << steps << ", Iters: " << iters << std::endl;

    // 1. Initialize Backend
    // Use device 0 by default
    auto backend = sf::get_backend(DeviceType::CUDA, 0);
    std::cout << "  Backend initialized." << std::endl;

    // 2. Load Model
    std::cout << "  Loading model..." << std::endl;
    auto model = rdt1b::load_rdt1b(model_path, backend);
    std::cout << "  Model loaded." << std::endl;

    // 3. Setup Engine
    EngineConfig config;
    config.device = Device{DeviceType::CUDA, 0};
    config.compute_dtype = DType::Float16;
    config.num_denoise_steps = steps;
    config.use_cuda_graph = false;
    config.pinned_output = true;
    config.cache_condition = true; 
    
    // Create sampler (Flow Matching for RDT-1B)
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/false);

    InferenceEngine engine(model, sampler, config);
    std::cout << "  Engine initialized." << std::endl;

    // 4. Create Dummy Input
    // We match the dimensions from bench_manual_rdt1b.py and config
    // lang: [1, 32, 4096] (using 32 for consistency with python benchmark)
    // img:  [1, 4374, 1152]
    // state: [1, 128]
    
    int64_t lang_len = 32; 
    int64_t img_len = 4374; 
    int64_t lang_dim = 4096;
    int64_t img_dim = 1152;
    int64_t state_dim = 128;

    Tensor lang_tok = backend->alloc(Shape({1, lang_len, lang_dim}), DType::Float16);
    Tensor img_tok = backend->alloc(Shape({1, img_len, img_dim}), DType::Float16);
    Tensor state_tok = backend->alloc(Shape({1, state_dim}), DType::Float32); 

    backend->fill(lang_tok, 0.1f);
    backend->fill(img_tok, 0.1f);
    backend->fill(state_tok, 0.1f);

    VLAInput input;
    input.language_tokens = lang_tok;
    input.images = {img_tok};
    input.robot_state = state_tok;

    // 5. Warmup
    std::cout << "  Warmup..." << std::endl;
    for (int i = 0; i < 5; ++i) {
        engine.infer(input);
    }
    backend->sync_device();

    // 6. Benchmark
    std::cout << "  Benchmarking..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        engine.infer(input);
    }
    backend->sync_device();
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double mean_ms = total_ms / iters;

    std::cout << "  Mean latency: " << mean_ms << " ms" << std::endl;
    std::cout << "  Control Hz: " << 1000.0 / mean_ms << " Hz" << std::endl;
    
    // Output JSON-like format for parser if needed, or just plain text
    // The previous run_comparison.sh regexed "Mean latency:\s+([\d\.]+)\s+ms"
    
    return 0;
}
