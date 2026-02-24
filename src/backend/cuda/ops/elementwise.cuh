// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace sf {
namespace cuda_ops {

// ─────────────────────────────────────────────────────────────────────────────
// Generic vectorised element-wise dispatch.
// Uses float4 / half2 loads where possible for memory bandwidth.
// ─────────────────────────────────────────────────────────────────────────────

// Operation tags (stateless functors for template dispatch).
struct AddOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ __half operator()(__half a, __half b) const { return __hadd(a, b); }
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __hadd(a, b);
    }
};

struct MulOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a * b; }
    __device__ __forceinline__ __half operator()(__half a, __half b) const { return __hmul(a, b); }
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __hmul(a, b);
    }
};

struct GeluOp {
    // tanh-approximation GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    static constexpr float kAlpha = 0.7978845608028654f;   // sqrt(2/pi)
    static constexpr float kBeta  = 0.044715f;

    __device__ __forceinline__ float operator()(float x) const {
        float t = tanhf(kAlpha * (x + kBeta * x * x * x));
        return 0.5f * x * (1.f + t);
    }

    __device__ __forceinline__ __half operator()(__half x) const {
        float fx = __half2float(x);
        return __float2half((*this)(fx));
    }

    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        return __float2bfloat16((*this)(fx));
    }
};

struct SiluOp {
    __device__ __forceinline__ float operator()(float x) const {
        return x / (1.f + expf(-x));
    }

    __device__ __forceinline__ __half operator()(__half x) const {
        return __float2half((*this)(__half2float(x)));
    }

    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        return __float2bfloat16((*this)(__bfloat162float(x)));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Kernels
// ─────────────────────────────────────────────────────────────────────────────

template<typename T, typename Op>
__global__ void binary_kernel(const T* __restrict__ a,
                               const T* __restrict__ b,
                               T* __restrict__ out,
                               int64_t n, Op op) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride)
        out[idx] = op(a[idx], b[idx]);
}

// Broadcast version: b has b_n elements; b[idx % b_n] is used (e.g. bias add).
template<typename T, typename Op>
__global__ void binary_kernel_bcast(const T* __restrict__ a,
                                     const T* __restrict__ b,
                                     T* __restrict__ out,
                                     int64_t n, int64_t b_n, Op op) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride)
        out[idx] = op(a[idx], b[idx % b_n]);
}

template<typename T, typename Op>
__global__ void unary_kernel(const T* __restrict__ x,
                              T* __restrict__ out,
                              int64_t n, Op op) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride)
        out[idx] = op(x[idx]);
}

template<typename T>
__global__ void scale_kernel_impl(const T* __restrict__ x,
                                   T* __restrict__ out,
                                   float scalar, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride)
        out[idx] = static_cast<T>(static_cast<float>(x[idx]) * scalar);
}

template<typename T>
__global__ void fill_kernel_impl(T* __restrict__ out, T val, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; idx < n; idx += stride)
        out[idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int kBlockSize = 256;

inline int64_t grid_size(int64_t n) {
    return std::min((n + kBlockSize - 1) / kBlockSize, static_cast<int64_t>(65535));
}

template<typename Op>
void elementwise_binary(const Tensor& a, const Tensor& b, Tensor& out,
                        cudaStream_t stream) {
    int64_t n   = a.numel();
    int64_t b_n = b.numel();
    // Use broadcast kernel when b is smaller (e.g. bias add: [N,D] + [D]).
    if (b_n == n) {
        switch (a.dtype()) {
            case DType::Float32:
                binary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<float>(), b.data_ptr<float>(),
                    out.data_ptr<float>(), n, Op{});
                break;
            case DType::Float16:
                binary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<__half>(), b.data_ptr<__half>(),
                    out.data_ptr<__half>(), n, Op{});
                break;
            case DType::BFloat16:
                binary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<__nv_bfloat16>(), b.data_ptr<__nv_bfloat16>(),
                    out.data_ptr<__nv_bfloat16>(), n, Op{});
                break;
            default:
                throw std::runtime_error("elementwise_binary: unsupported dtype");
        }
    } else {
        // Broadcast b over a: requires n % b_n == 0.
        switch (a.dtype()) {
            case DType::Float32:
                binary_kernel_bcast<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<float>(), b.data_ptr<float>(),
                    out.data_ptr<float>(), n, b_n, Op{});
                break;
            case DType::Float16:
                binary_kernel_bcast<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<__half>(), b.data_ptr<__half>(),
                    out.data_ptr<__half>(), n, b_n, Op{});
                break;
            case DType::BFloat16:
                binary_kernel_bcast<<<grid_size(n), kBlockSize, 0, stream>>>(
                    a.data_ptr<__nv_bfloat16>(), b.data_ptr<__nv_bfloat16>(),
                    out.data_ptr<__nv_bfloat16>(), n, b_n, Op{});
                break;
            default:
                throw std::runtime_error("elementwise_binary: unsupported dtype");
        }
    }
}

template<typename Op>
void activation_kernel(const Tensor& x, Tensor& out, cudaStream_t stream) {
    int64_t n = x.numel();
    switch (x.dtype()) {
        case DType::Float32:
            unary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), n, Op{});
            break;
        case DType::Float16:
            unary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                x.data_ptr<__half>(), out.data_ptr<__half>(), n, Op{});
            break;
        case DType::BFloat16:
            unary_kernel<<<grid_size(n), kBlockSize, 0, stream>>>(
                x.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(), n, Op{});
            break;
        default:
            throw std::runtime_error("activation: unsupported dtype");
    }
}

inline void scale_kernel(const Tensor& a, float scalar, Tensor& out,
                         cudaStream_t stream) {
    int64_t n = a.numel();
    switch (a.dtype()) {
        case DType::Float32:
            scale_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                a.data_ptr<float>(), out.data_ptr<float>(), scalar, n);
            break;
        case DType::Float16:
            scale_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                a.data_ptr<__half>(), out.data_ptr<__half>(), scalar, n);
            break;
        case DType::BFloat16:
            scale_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                a.data_ptr<__nv_bfloat16>(), out.data_ptr<__nv_bfloat16>(), scalar, n);
            break;
        default:
            throw std::runtime_error("scale: unsupported dtype");
    }
}

inline void fill_kernel(Tensor& dst, float value, cudaStream_t stream) {
    int64_t n = dst.numel();
    switch (dst.dtype()) {
        case DType::Float32:
            fill_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                dst.data_ptr<float>(), value, n);
            break;
        case DType::Float16:
            fill_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                dst.data_ptr<__half>(), __float2half(value), n);
            break;
        case DType::BFloat16:
            fill_kernel_impl<<<grid_size(n), kBlockSize, 0, stream>>>(
                dst.data_ptr<__nv_bfloat16>(), __float2bfloat16(value), n);
            break;
        default:
            throw std::runtime_error("fill: unsupported dtype");
    }
}

// Cast, embedding, cat, softmax declarations (implemented in separate .cu files).
void cast_kernel(const Tensor& src, Tensor& dst, cudaStream_t stream);
void embedding_lookup(const Tensor& weight, const Tensor& indices,
                      Tensor& out, cudaStream_t stream);
void cat_kernel(const std::vector<Tensor>& inputs, Tensor& out,
                int64_t dim, cudaStream_t stream);
void softmax_kernel(const Tensor& x, Tensor& out, int64_t dim,
                    cudaStream_t stream);

}  // namespace cuda_ops
}  // namespace sf
