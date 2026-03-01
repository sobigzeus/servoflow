// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "servoflow/core/device.h"
#include "servoflow/core/dtype.h"
#include "servoflow/core/tensor.h"

namespace sf {

// Opaque handle for backend-specific stream objects.
// CUDA: cudaStream_t, ROCm: hipStream_t, CPU: nullptr
using StreamHandle = void*;

// Capabilities reported by a backend.
struct BackendCaps {
    bool supports_fp16   = false;
    bool supports_bf16   = false;
    bool supports_int8   = false;
    bool supports_int4   = false;
    bool supports_graph  = false;   // CUDA Graph / Metal Command Buffer reuse
    size_t max_shared_mem_bytes = 0;
    size_t total_device_mem_bytes = 0;
    std::string name;               // e.g. "CUDA", "ROCm", "CPU"
    std::string device_name;        // e.g. "NVIDIA GeForce RTX 3090"
};

// ─────────────────────────────────────────────────────────────────────────────
// IBackend: pure-virtual interface every hardware backend must implement.
//
// Design rules:
//   1. All methods are thread-compatible (callers must not share a stream
//      across threads without explicit synchronization).
//   2. Tensor memory allocated by this backend must be freed by the same
//      backend instance (via the Storage deleter set at alloc time).
//   3. Hot-path operator calls (gemm, attention, etc.) take a StreamHandle
//      to allow async execution without extra overhead.
// ─────────────────────────────────────────────────────────────────────────────
class IBackend {
public:
    virtual ~IBackend() = default;

    // ── Identity ──────────────────────────────────────────────────────────
    virtual DeviceType   device_type() const = 0;
    virtual BackendCaps  caps()        const = 0;

    // ── Memory management ─────────────────────────────────────────────────
    // Allocate a Tensor with uninitialised data on the backend device.
    virtual Tensor alloc(Shape shape, DType dtype,
                         StreamHandle stream = nullptr) = 0;

    // Allocate pinned host memory (for fast H<->D transfers).
    // Returns a CPU Tensor backed by pinned memory.
    virtual Tensor alloc_pinned(Shape shape, DType dtype) = 0;

    // Release all cached memory back to the OS/driver.
    virtual void empty_cache() = 0;

    // ── Data transfer ─────────────────────────────────────────────────────
    // Copies src → dst; handles H2D, D2H, D2D.
    // dst must already be allocated with compatible shape/dtype.
    virtual void copy(Tensor& dst, const Tensor& src,
                      StreamHandle stream = nullptr) = 0;

    // Convenience: fill a tensor with a scalar value (fp32 interpreted
    // according to dst.dtype()).
    virtual void fill(Tensor& dst, float value,
                      StreamHandle stream = nullptr) = 0;

    // ── Stream management ─────────────────────────────────────────────────
    virtual StreamHandle create_stream()               = 0;
    virtual void         destroy_stream(StreamHandle)  = 0;
    virtual void         sync_stream(StreamHandle)     = 0;
    virtual void         sync_device()                 = 0;

    // ── Operator dispatch ─────────────────────────────────────────────────
    // All operators are expressed as methods so the backend can fuse or
    // delegate to specialised libraries (cuBLAS, FlashAttention, etc.).
    //
    // Semantics follow standard deep-learning conventions:
    //   - All Tensors must reside on this backend's device.
    //   - Output Tensor must be pre-allocated by the caller unless noted.
    //   - In-place operations are explicitly marked.

    // General Matrix Multiply: C = alpha * A @ B + beta * C
    // Shapes: A [M, K], B [K, N], C [M, N]
    virtual void gemm(const Tensor& A, const Tensor& B, Tensor& C,
                      float alpha = 1.f, float beta = 0.f,
                      bool trans_a = false, bool trans_b = false,
                      StreamHandle stream = nullptr) = 0;

    // Batched GEMM: A [B, M, K], B [B, K, N], C [B, M, N]
    virtual void batched_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                              float alpha = 1.f, float beta = 0.f,
                              bool trans_a = false, bool trans_b = false,
                              StreamHandle stream = nullptr) = 0;

    // Scaled Dot-Product Attention (FlashAttention on CUDA backend).
    // Q [B, H, S, D], K [B, H, Sk, D], V [B, H, Sk, D] → out [B, H, S, D]
    // mask: optional causal or padding mask [B, 1, S, Sk] or nullptr.
    virtual void attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                           Tensor& out,
                           const Tensor* mask     = nullptr,
                           float         scale    = 0.f,    // 0 → 1/sqrt(D)
                           bool          is_causal = false,
                           StreamHandle  stream   = nullptr) = 0;

    // Layer normalisation: out = (x - mean) / sqrt(var + eps) * gamma + beta
    // x, out: [..., C]; gamma, beta: [C]
    virtual void layer_norm(const Tensor& x,
                            const Tensor& gamma, const Tensor& beta,
                            Tensor& out, float eps = 1e-5f,
                            StreamHandle stream = nullptr) = 0;

    // RMS normalisation (no mean subtraction): out = x / rms(x) * gamma
    virtual void rms_norm(const Tensor& x, const Tensor& gamma,
                          Tensor& out, float eps = 1e-6f,
                          StreamHandle stream = nullptr) = 0;

    // Fused Add + RMSNorm
    // out = RMSNorm(input + residual)
    // residual = input + residual (in-place)
    virtual void fused_add_rms_norm(const Tensor& input, Tensor& residual,
                                    const Tensor& gamma, Tensor& out,
                                    float eps = 1e-6f,
                                    StreamHandle stream = nullptr) {
        // Fallback
        add(input, residual, residual, stream);
        rms_norm(residual, gamma, out, eps, stream);
    }

    // Element-wise: out = a + b  (broadcast along leading dims supported)
    virtual void add(const Tensor& a, const Tensor& b, Tensor& out,
                     StreamHandle stream = nullptr) = 0;

    // Element-wise: out = a * b
    virtual void mul(const Tensor& a, const Tensor& b, Tensor& out,
                     StreamHandle stream = nullptr) = 0;

    // Element-wise: out = a * scalar
    virtual void scale(const Tensor& a, float scalar, Tensor& out,
                       StreamHandle stream = nullptr) = 0;

    // Activation types for fused kernels
    enum class ActivationType {
        None,
        GELU,
        SiLU,
        ReLU
    };

    // Fused GEMM + Bias + Activation
    // C = Act(alpha * A @ B + bias)
    // bias is [N] (broadcasted to [M, N])
    virtual void gemm_bias_act(const Tensor& A, const Tensor& B, 
                               const Tensor& bias, Tensor& C,
                               ActivationType act = ActivationType::None,
                               float alpha = 1.f, float beta = 0.f,
                               bool trans_a = false, bool trans_b = false,
                               StreamHandle stream = nullptr) {
        // Default fallback implementation
        gemm(A, B, C, alpha, beta, trans_a, trans_b, stream);
        // We assume bias is broadcastable. If C is [M, N] and bias is [N], 
        // add() should handle it if implemented robustly, or we reshape.
        // For safety here, we rely on the backend's add() to handle broadcasting
        // or the caller to provide correct shapes.
        // In ServoFlow, bias is usually [N], C is [M, N]. 
        // Standard add() might expect matching shapes or explicit broadcast.
        // Let's rely on specific backends to override this for performance,
        // and this fallback is just a functional placeholder.
        add(C, bias, C, stream);
        
        switch (act) {
            case ActivationType::GELU: gelu(C, C, stream); break;
            case ActivationType::SiLU: silu(C, C, stream); break;
            // case ActivationType::ReLU: relu(C, C, stream); break; // Need relu() interface if we support it
            default: break;
        }
    }

    // Activation functions (in-place: out may alias x).
    virtual void gelu(const Tensor& x, Tensor& out,
                      StreamHandle stream = nullptr) = 0;
    virtual void silu(const Tensor& x, Tensor& out,
                      StreamHandle stream = nullptr) = 0;
    virtual void softmax(const Tensor& x, Tensor& out, int64_t dim = -1,
                         StreamHandle stream = nullptr) = 0;

    // Embedding lookup: indices [B, S] → out [B, S, D]
    virtual void embedding(const Tensor& weight, const Tensor& indices,
                           Tensor& out, StreamHandle stream = nullptr) = 0;

    // Cast dtype: out must be pre-allocated with target dtype.
    virtual void cast(const Tensor& src, Tensor& dst,
                      StreamHandle stream = nullptr) = 0;

    // Dequantize INT8 tensor to FP16 using scale.
    // input: [N, K] (row-major) or [K, N] (col-major) depending on layout.
    // scale: [N] (per-channel) or [1] (per-tensor).
    // output: same shape as input, FP16.
    virtual void dequantize(const Tensor& input, const Tensor& scale, Tensor& output,
                            StreamHandle stream = nullptr) = 0;

    // Concatenate tensors along dim (all inputs on same device).
    virtual void cat(const std::vector<Tensor>& inputs, Tensor& out,
                     int64_t dim = 0, StreamHandle stream = nullptr) = 0;

    // ── Graph capture (optional, no-op on backends that don't support it) ─
    virtual void graph_begin_capture(StreamHandle) {}
    virtual void graph_end_capture(StreamHandle)   {}
    virtual void graph_launch(StreamHandle)        {}

protected:
    IBackend() = default;
};

using BackendPtr = std::shared_ptr<IBackend>;

// ─────────────────────────────────────────────────────────────────────────────
// BackendRegistry: singleton that maps DeviceType → backend factory.
// Backends register themselves at library load time via static initialisers.
// ─────────────────────────────────────────────────────────────────────────────
class BackendRegistry {
public:
    using Factory = std::function<BackendPtr(int device_index)>;

    static BackendRegistry& instance();

    void       register_backend(DeviceType type, Factory factory);
    BackendPtr get(Device device);
    bool       has(DeviceType type) const;

private:
    BackendRegistry() = default;
    std::unordered_map<uint8_t, Factory> factories_;
    std::unordered_map<std::string, BackendPtr> cache_;  // "type:index" → backend
};

// Helper to obtain (and cache) a backend for a given device.
BackendPtr get_backend(Device device);
BackendPtr get_backend(DeviceType type, int index = 0);

}  // namespace sf
