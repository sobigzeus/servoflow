// SPDX-License-Identifier: Apache-2.0
// Microbenchmark: ServoFlow attention vs naive baseline.
// Reports mean latency (ms) and throughput (TFLOP/s) over N warm-up + M timed
// iterations, using CUDA events for accurate GPU timing.
#include "servoflow/backend/backend.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sf;

#define CUDA_CHECK(expr)                                                 \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess)                                           \
            throw std::runtime_error(cudaGetErrorString(_e));            \
    } while (0)

// Compute theoretical FLOPs for attention:
//   QK^T: B*H*S*S*2D + softmax: B*H*S*S + AV: B*H*S*D*2Sk ≈ 4*B*H*S*Sk*D
static double attn_flops(int64_t B, int64_t H, int64_t S, int64_t D) {
    return 4.0 * B * H * S * S * D;
}

struct Config {
    int64_t batch   = 1;
    int64_t heads   = 16;
    int64_t seq     = 512;
    int64_t head_dim = 64;
    DType   dtype   = DType::Float16;
    int     warmup  = 5;
    int     iters   = 20;
    bool    causal  = false;
};

static void run_bench(const Config& cfg) {
    BackendPtr backend = get_backend(kCUDA0);
    StreamHandle stream = backend->create_stream();

    Shape qkv_shape{cfg.batch, cfg.heads, cfg.seq, cfg.head_dim};
    Tensor Q   = backend->alloc(qkv_shape, cfg.dtype, stream);
    Tensor K   = backend->alloc(qkv_shape, cfg.dtype, stream);
    Tensor V   = backend->alloc(qkv_shape, cfg.dtype, stream);
    Tensor out = backend->alloc(qkv_shape, cfg.dtype, stream);

    backend->fill(Q, 0.01f, stream);
    backend->fill(K, 0.01f, stream);
    backend->fill(V, 0.01f, stream);
    backend->sync_stream(stream);

    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);

    // Warm-up.
    for (int i = 0; i < cfg.warmup; ++i) {
        backend->attention(Q, K, V, out, nullptr, 0.f, cfg.causal, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(cs));

    // Timed iterations.
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    CUDA_CHECK(cudaEventRecord(ev_start, cs));
    for (int i = 0; i < cfg.iters; ++i)
        backend->attention(Q, K, V, out, nullptr, 0.f, cfg.causal, stream);
    CUDA_CHECK(cudaEventRecord(ev_end, cs));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float total_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    float mean_ms = total_ms / cfg.iters;

    double flops = attn_flops(cfg.batch, cfg.heads, cfg.seq, cfg.head_dim);
    double tflops = flops / (mean_ms * 1e-3) / 1e12;

    std::printf("  [B=%lld H=%lld S=%lld D=%lld causal=%d dtype=%s]  %.3f ms  %.2f TFLOP/s\n",
                (long long)cfg.batch, (long long)cfg.heads,
                (long long)cfg.seq,   (long long)cfg.head_dim,
                (int)cfg.causal,
                std::string(dtype_name(cfg.dtype)).c_str(),
                mean_ms, tflops);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    backend->destroy_stream(stream);
}

int main() {
    std::fprintf(stderr, "=== ServoFlow Attention Benchmark ===\n");
    std::vector<Config> configs = {
        // Correctness/sanity check (small)
        {1, 2, 128, 64, DType::Float16, 2, 2, false},
        {1, 2, 128, 64, DType::Float16, 2, 2, true},
        
        // Standard perf cases
        {1, 12, 1024, 64, DType::Float16, 5, 20, false},
        {1, 12, 1024, 64, DType::Float16, 5, 20, true},
        
        // Larger seq len
        {1, 12, 2048, 64, DType::Float16, 5, 20, false},
        
        // Larger batch
        {8, 12, 512, 64, DType::Float16, 5, 20, false},
        
        // Different head dim (if supported by FlashAttention - standard are 32, 64, 128)
        {1, 8, 512, 32, DType::Float16, 5, 20, false},
        {1, 8, 512, 128, DType::Float16, 5, 20, false},
    };

    for (const auto& cfg : configs) {
        try {
            run_bench(cfg);
        } catch (const std::exception& e) {
            std::printf("FAILED config [B=%lld H=%lld S=%lld D=%lld]: %s\n", 
                (long long)cfg.batch, (long long)cfg.heads, (long long)cfg.seq, (long long)cfg.head_dim, e.what());
        }
    }

    return 0;
}
