# ServoFlow

**High-performance C++/CUDA inference runtime for Vision-Language-Action (VLA) models.**

ServoFlow targets real-time robot control at **50 Hz** — something Python-based frameworks cannot reliably achieve due to interpreter and GIL overhead. It is the foundation for a full-stack (model + software + hardware) robotics VLA deployment solution.

> **Status**: Phase 1 — framework core + CUDA backend + RDT-1B support.

---

## Key Design Goals

| Goal | How |
|---|---|
| Real-time (≥50 Hz) | CUDA Graph loop capture, condition caching, static memory pool |
| Hardware-agnostic | `IBackend` abstraction; CUDA now, ROCm / Metal / TensorRT planned |
| Open architecture | Clean layered design; model Zoo and hardware backends are plug-in |
| Production quality | Typed APIs, zero hidden allocation in hot path, comprehensive tests |

---

## Architecture

```
Python bindings (optional)    ← pybind11
C API (servoflow.h)           ← stable ABI
─────────────────────────────────────────
InferenceEngine               ← condition cache · CUDA Graph · async D2H
  ├── FlowMatchingSampler     ← Euler ODE (RDT-1B / π0)
  │   └── DDIMSampler         ← DDIM (DDPM-trained models)
  └── IVLAModel               ← RDT-1B · (more planned)
─────────────────────────────────────────
Operator Library
  Attention (FlashAttention v2 on CUDA) · GEMM (cuBLAS) · LayerNorm · RMSNorm
  GELU · SiLU · Embedding · Cast · Cat · Softmax
─────────────────────────────────────────
IBackend
  ├── CUDABackend  (Phase 1)  ← memory pool · streams · CUDA Graph
  ├── ROCmBackend  (planned)
  ├── MetalBackend (planned)
  └── TensorRTBackend (planned)
─────────────────────────────────────────
Core: Tensor · Shape · DType · Device · Storage
```

---

## Performance Optimisations

1. **Condition cache** — vision + language encoding runs once per scene, not once per step. Saves ~60–80 % of total compute.
2. **CUDA Graph capture** — the entire denoising loop (N steps × DiT forward) is captured as a CUDA Graph on the first call and replayed on subsequent calls, eliminating CPU kernel-launch overhead.
3. **Static memory pool** — all intermediate tensors are pre-allocated at engine init; zero `cudaMalloc` / `cudaFree` in the hot path.
4. **FlashAttention v2** — O(S) memory, 2–4× faster attention vs standard SDPA on Ampere+.
5. **Multi-stream overlap** — vision encoding and denoising run on separate CUDA streams, synchronised via a lightweight CUDA event.
6. **Pinned host output** — action result is transferred from GPU to pinned host memory for minimum D2H latency.

---

## Supported Models

| Model | Sampler | Status |
|---|---|---|
| RDT-1B | Flow Matching (Euler) | Phase 1 target |
| OpenVLA | DDIM | Planned |

---

## Building

### Prerequisites

- CMake ≥ 3.22
- CUDA Toolkit ≥ 12.0 (for CUDA backend)
- GCC ≥ 11 or Clang ≥ 14
- (Optional) [FlashAttention v2](https://github.com/Dao-AILab/flash-attention) for best attention performance

```bash
git clone https://github.com/your-org/servoflow.git
cd servoflow
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DSF_CUDA_ARCHS="86"          # RTX 3090 = sm_86
cmake --build build -j$(nproc)
```

With FlashAttention:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DSF_USE_FLASH_ATTN=ON \
      -DFLASH_ATTN_ROOT=/path/to/flash-attention
cmake --build build -j$(nproc)
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

### Benchmarks

```bash
# Attention microbenchmark
./build/benchmarks/bench_attention

# End-to-end pipeline benchmark (stub model, N denoising steps)
./build/benchmarks/bench_pipeline 10
```

---

## Benchmarks

### RDT-1B Performance (RTX 3090)

ServoFlow achieves **1.61x speedup** over optimized PyTorch (FP16) for RDT-1B inference, thanks to aggressive operator fusion, zero-overhead memory management, and CUDA Graph execution.

| Metric | PyTorch (FP16) | ServoFlow (FP16) | Speedup |
| :--- | :--- | :--- | :--- |
| **Loop Latency (10 steps)** | 551.48 ms | **342.95 ms** | **1.61x** |
| **Per-step Latency** | 55.15 ms | **34.30 ms** | **1.61x** |
| **Control Freq** | 1.81 Hz | **2.92 Hz** | **1.61x** |

**Alignment Accuracy:**
- **Max Error**: 1.95e-03 (FP16)
- **Cosine Similarity**: 1.000001
- **Status**: Verified against HuggingFace `rdt-1b` PyTorch implementation.

**Key Optimizations:**
1. **CUDA Graph**: Captures the entire denoising loop (10 steps × 28 blocks) into a single graph launch, eliminating CPU overhead.
2. **Memory Pool**: Custom `cudaMallocAsync`-based memory pool ensures zero allocation overhead during inference.
3. **Operator Fusion**: Fused `Add+RMSNorm` and `GEMM+Bias+Act` kernels minimize memory bandwidth usage.
4. **FlashAttention**: Zero-allocation integration of FlashAttention v2.

To run benchmarks:
```bash
./run_gpu_comparison.sh
```

---

## Project Layout

```
servoflow/
├── include/servoflow/    # Public headers (stable API)
│   ├── core/             # Tensor, Shape, DType, Device, Storage
│   ├── backend/          # IBackend interface + CUDA header
│   ├── ops/              # Operator declarations
│   ├── models/           # IVLAModel + model configs
│   ├── sampling/         # ISampler, FlowMatchingSampler, DDIMSampler
│   └── engine/           # InferenceEngine, VLAInput/Output, EngineConfig
├── src/                  # Implementations
│   ├── backend/cuda/     # CUDABackend + CUDA kernels
│   ├── sampling/         # Sampler implementations
│   └── engine/           # InferenceEngine orchestration
├── tests/                # Unit + integration tests (GoogleTest)
├── benchmarks/           # GPU microbenchmarks + pipeline benchmark
├── examples/             # Usage examples
└── tools/convert/        # HuggingFace → ServoFlow weight converter
```

---

## Roadmap

**Phase 1 (current)**
- [x] Core tensor abstraction + CUDA backend
- [x] FlashAttention integration
- [x] Flow Matching sampler with CUDA Graph capture
- [x] InferenceEngine with condition cache
- [x] RDT-1B model weight loader (safetensors)
- [x] RDT-1B DiT block implementation
- [x] Benchmark vs diffusers + TensorRT pipeline (PyTorch baseline)

**Phase 2**
- [ ] Model distillation (fewer denoising steps)
- [ ] INT8 / INT4 quantisation
- [ ] ROCm backend
- [ ] Jetson / edge hardware optimisation
- [ ] TensorRT backend

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
