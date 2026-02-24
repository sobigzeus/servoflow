// SPDX-License-Identifier: Apache-2.0
#pragma once
// Utility kernels for tensor splitting and adaLN application.
// These are VLA-specific patterns that general-purpose backends don't cover.

#include "servoflow/core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sf {
namespace cuda_ops {

// ─────────────────────────────────────────────────────────────────────────────
// split_last_dim: split [B, 2*D] → [B, D] + [B, D]  (contiguous copy)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void split_last_dim_kernel(const T* __restrict__ src,
                                       T* __restrict__ out0,
                                       T* __restrict__ out1,
                                       int64_t rows, int64_t half_cols) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = rows * half_cols;
    if (idx >= total) return;
    int64_t row = idx / half_cols;
    int64_t col = idx % half_cols;
    out0[idx]   = src[row * 2 * half_cols + col];
    out1[idx]   = src[row * 2 * half_cols + half_cols + col];
}

inline void split_last_dim(const Tensor& src, Tensor& out0, Tensor& out1,
                            BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t rows      = src.numel() / src.shape()[src.ndim() - 1];
    int64_t half_cols = src.shape()[src.ndim() - 1] / 2;
    int64_t total     = rows * half_cols;
    int64_t grid      = (total + 255) / 256;

    switch (src.dtype()) {
        case DType::Float32:
            split_last_dim_kernel<float><<<grid, 256, 0, cs>>>(
                src.data_ptr<float>(), out0.data_ptr<float>(),
                out1.data_ptr<float>(), rows, half_cols); break;
        case DType::Float16:
            split_last_dim_kernel<__half><<<grid, 256, 0, cs>>>(
                src.data_ptr<__half>(), out0.data_ptr<__half>(),
                out1.data_ptr<__half>(), rows, half_cols); break;
        case DType::BFloat16:
            split_last_dim_kernel<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                src.data_ptr<__nv_bfloat16>(), out0.data_ptr<__nv_bfloat16>(),
                out1.data_ptr<__nv_bfloat16>(), rows, half_cols); break;
        default:
            throw std::runtime_error("split_last_dim: unsupported dtype");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// split_qkv_kernel: [B, S, 3*H*D] → Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]
// Interleaved layout: for each token, Q heads then K heads then V heads.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void split_qkv_kernel_impl(const T* __restrict__ qkv,
                                       T* __restrict__ Q,
                                       T* __restrict__ K,
                                       T* __restrict__ V,
                                       int64_t B, int64_t S,
                                       int64_t H, int64_t D) {
    // Each thread copies one element.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * S * D;
    if (idx >= total) return;

    int64_t b = idx / (H * S * D);
    int64_t h = (idx / (S * D)) % H;
    int64_t s = (idx / D) % S;
    int64_t d = idx % D;

    // src index in [B, S, 3*H*D] layout:  batch_offset + seq_offset + qkv_offset
    int64_t src_base = b * S * 3 * H * D + s * 3 * H * D;

    Q[idx] = qkv[src_base + (0 * H + h) * D + d];
    K[idx] = qkv[src_base + (1 * H + h) * D + d];
    V[idx] = qkv[src_base + (2 * H + h) * D + d];
}

inline void split_qkv_kernel(const Tensor& qkv,
                               Tensor& Q, Tensor& K, Tensor& V,
                               int64_t H, int64_t D,
                               BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t B  = qkv.shape()[0];
    int64_t S  = qkv.shape()[1];
    int64_t total = B * H * S * D;
    int64_t grid = (total + 255) / 256;

    switch (qkv.dtype()) {
        case DType::Float32:
            split_qkv_kernel_impl<float><<<grid, 256, 0, cs>>>(
                qkv.data_ptr<float>(),
                Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                B, S, H, D); break;
        case DType::Float16:
            split_qkv_kernel_impl<__half><<<grid, 256, 0, cs>>>(
                qkv.data_ptr<__half>(),
                Q.data_ptr<__half>(), K.data_ptr<__half>(), V.data_ptr<__half>(),
                B, S, H, D); break;
        case DType::BFloat16:
            split_qkv_kernel_impl<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                qkv.data_ptr<__nv_bfloat16>(),
                Q.data_ptr<__nv_bfloat16>(), K.data_ptr<__nv_bfloat16>(),
                V.data_ptr<__nv_bfloat16>(),
                B, S, H, D); break;
        default:
            throw std::runtime_error("split_qkv: unsupported dtype");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// seq_to_head: [B, S, H*D] → [B, H, S, D]
// Physically permutes from sequence-major to head-major layout.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void seq_to_head_kernel_impl(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         int64_t B, int64_t S, int64_t H, int64_t D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * S * D;
    if (idx >= total) return;

    int64_t b = idx / (H * S * D);
    int64_t h = (idx / (S * D)) % H;
    int64_t s = (idx / D) % S;
    int64_t d = idx % D;

    // src layout: [B, S, H*D]  → src[b, s, h*D + d]
    dst[idx] = src[b * S * H * D + s * H * D + h * D + d];
}

inline void seq_to_head(const Tensor& src, Tensor& dst,
                         int64_t H, int64_t D,
                         BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t B = src.shape()[0];
    int64_t S = src.shape()[1];
    int64_t total = B * H * S * D;
    int64_t grid  = (total + 255) / 256;

    switch (src.dtype()) {
        case DType::Float32:
            seq_to_head_kernel_impl<float><<<grid, 256, 0, cs>>>(
                src.data_ptr<float>(), dst.data_ptr<float>(), B, S, H, D); break;
        case DType::Float16:
            seq_to_head_kernel_impl<__half><<<grid, 256, 0, cs>>>(
                src.data_ptr<__half>(), dst.data_ptr<__half>(), B, S, H, D); break;
        case DType::BFloat16:
            seq_to_head_kernel_impl<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                src.data_ptr<__nv_bfloat16>(), dst.data_ptr<__nv_bfloat16>(), B, S, H, D); break;
        default:
            throw std::runtime_error("seq_to_head: unsupported dtype");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// head_to_seq: [B, H, S, D] → [B, S, H*D]
// Inverse of seq_to_head. Converts attention output back to sequence-major.
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void head_to_seq_kernel_impl(const T* __restrict__ src,
                                         T* __restrict__ dst,
                                         int64_t B, int64_t H, int64_t S, int64_t D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * S * D;
    if (idx >= total) return;

    int64_t b = idx / (H * S * D);
    int64_t h = (idx / (S * D)) % H;
    int64_t s = (idx / D) % S;
    int64_t d = idx % D;

    // dst layout: [B, S, H*D]  → dst[b, s, h*D + d]
    dst[b * S * H * D + s * H * D + h * D + d] = src[idx];
}

inline void head_to_seq(const Tensor& src, Tensor& dst,
                         int64_t H, int64_t D,
                         BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t B = src.shape()[0];
    int64_t S = src.shape()[2];
    int64_t total = B * H * S * D;
    int64_t grid  = (total + 255) / 256;

    switch (src.dtype()) {
        case DType::Float32:
            head_to_seq_kernel_impl<float><<<grid, 256, 0, cs>>>(
                src.data_ptr<float>(), dst.data_ptr<float>(), B, H, S, D); break;
        case DType::Float16:
            head_to_seq_kernel_impl<__half><<<grid, 256, 0, cs>>>(
                src.data_ptr<__half>(), dst.data_ptr<__half>(), B, H, S, D); break;
        case DType::BFloat16:
            head_to_seq_kernel_impl<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                src.data_ptr<__nv_bfloat16>(), dst.data_ptr<__nv_bfloat16>(), B, H, S, D); break;
        default:
            throw std::runtime_error("head_to_seq: unsupported dtype");
    }
}

// split_kv: [B, S, 2*H*D] → K[B,H,S,D], V[B,H,S,D]
template<typename T>
__global__ void split_kv_kernel_impl(const T* __restrict__ kv,
                                      T* __restrict__ K, T* __restrict__ V,
                                      int64_t B, int64_t S,
                                      int64_t H, int64_t D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = B * H * S * D;
    if (idx >= total) return;

    int64_t b = idx / (H * S * D);
    int64_t h = (idx / (S * D)) % H;
    int64_t s = (idx / D) % S;
    int64_t d = idx % D;

    int64_t src_base = b * S * 2 * H * D + s * 2 * H * D;
    K[idx] = kv[src_base + (0 * H + h) * D + d];
    V[idx] = kv[src_base + (1 * H + h) * D + d];
}

inline void split_kv_kernel(const Tensor& kv,
                              Tensor& K, Tensor& V,
                              int64_t H, int64_t D,
                              BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t B  = kv.shape()[0];
    int64_t S  = kv.shape()[1];
    int64_t total = B * H * S * D;
    int64_t grid = (total + 255) / 256;

    switch (kv.dtype()) {
        case DType::Float32:
            split_kv_kernel_impl<float><<<grid, 256, 0, cs>>>(
                kv.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                B, S, H, D); break;
        case DType::Float16:
            split_kv_kernel_impl<__half><<<grid, 256, 0, cs>>>(
                kv.data_ptr<__half>(), K.data_ptr<__half>(), V.data_ptr<__half>(),
                B, S, H, D); break;
        case DType::BFloat16:
            split_kv_kernel_impl<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                kv.data_ptr<__nv_bfloat16>(), K.data_ptr<__nv_bfloat16>(),
                V.data_ptr<__nv_bfloat16>(),
                B, S, H, D); break;
        default:
            throw std::runtime_error("split_kv: unsupported dtype");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_adaln: x = (1 + scale) * x + shift
// x: [B, S, D],  scale/shift: [B, D]  (broadcast over S)
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void apply_adaln_kernel(T* __restrict__ x,
                                    const T* __restrict__ scale,
                                    const T* __restrict__ shift,
                                    int64_t B, int64_t S, int64_t D) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= B * S * D) return;
    int64_t b = idx / (S * D);
    int64_t d = idx % D;
    float v  = static_cast<float>(x[idx]);
    float sc = static_cast<float>(scale[b * D + d]);
    float sh = static_cast<float>(shift[b * D + d]);
    x[idx]   = static_cast<T>((1.f + sc) * v + sh);
}

inline void apply_adaln(Tensor& x, const Tensor& scale, const Tensor& shift,
                         BackendPtr /*backend*/, StreamHandle stream) {
    cudaStream_t cs = reinterpret_cast<cudaStream_t>(stream);
    int64_t B = x.shape()[0];
    int64_t S = x.shape()[1];
    int64_t D = x.shape()[2];
    int64_t total = B * S * D;
    int64_t grid = (total + 255) / 256;

    switch (x.dtype()) {
        case DType::Float32:
            apply_adaln_kernel<float><<<grid, 256, 0, cs>>>(
                x.data_ptr<float>(), scale.data_ptr<float>(),
                shift.data_ptr<float>(), B, S, D); break;
        case DType::Float16:
            apply_adaln_kernel<__half><<<grid, 256, 0, cs>>>(
                x.data_ptr<__half>(), scale.data_ptr<__half>(),
                shift.data_ptr<__half>(), B, S, D); break;
        case DType::BFloat16:
            apply_adaln_kernel<__nv_bfloat16><<<grid, 256, 0, cs>>>(
                x.data_ptr<__nv_bfloat16>(), scale.data_ptr<__nv_bfloat16>(),
                shift.data_ptr<__nv_bfloat16>(), B, S, D); break;
        default:
            throw std::runtime_error("apply_adaln: unsupported dtype");
    }
}

}  // namespace cuda_ops
}  // namespace sf
