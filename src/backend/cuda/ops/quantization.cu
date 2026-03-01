// SPDX-License-Identifier: Apache-2.0
#include "quantization.cuh"
#include "servoflow/core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>

namespace sf {
namespace cuda_ops {

// ─────────────────────────────────────────────────────────────────────────────
// Dequantize Kernel (INT8 -> FP16)
// ─────────────────────────────────────────────────────────────────────────────
// Supports per-tensor scaling (scale=[1]) or per-channel (scale=[N]).
// Assumes scale broadcasts along dimension 0 if scale.ndim > 0.

__global__ void dequantize_int8_fp16_kernel(
    const int8_t* __restrict__ input,
    const __half* __restrict__ scale,
    __half* __restrict__ output,
    int64_t K,
    int64_t num_elements,
    bool per_channel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int8_t val = input[idx];
    __half s;
    if (per_channel) {
        // Assume input is [N, K], scale is [N]
        // Row-major index: idx = row * K + col
        // Scale index = row
        int64_t row = idx / K;
        s = scale[row];
    } else {
        s = scale[0];
    }
    
    // FP16 multiplication
    // Convert int8 to half directly
#if __CUDA_ARCH__ >= 530
    output[idx] = __hmul(__int2half_rn(val), s);
#else
    // Fallback for older architectures
    float fval = static_cast<float>(val);
    float fscale = __half2float(s);
    output[idx] = __float2half(fval * fscale);
#endif
}

void dequantize_int8(const Tensor& input, const Tensor& scale, Tensor& output, cudaStream_t stream) {
    int64_t numel = input.numel();
    if (numel == 0) return;

    if (input.dtype() != DType::Int8) {
        throw std::invalid_argument("dequantize_int8: input must be Int8");
    }
    if (output.dtype() != DType::Float16) {
        throw std::invalid_argument("dequantize_int8: output must be Float16 (BF16/FP32 not implemented yet)");
    }
    if (scale.dtype() != DType::Float16) {
        throw std::invalid_argument("dequantize_int8: scale must be Float16");
    }

    // Determine K for broadcasting
    // If scale is scalar (numel=1), per-tensor.
    // If scale is vector (numel > 1), assume per-channel along dim 0.
    // input: [N, K] -> scale must be [N] or [N, 1].
    
    bool per_channel = (scale.numel() > 1);
    int64_t K = 1;
    if (per_channel) {
        if (scale.shape()[0] != input.shape()[0]) {
            throw std::runtime_error("dequantize_int8: scale dim 0 mismatch for per-channel quantization");
        }
        K = numel / input.shape()[0];
    }

    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(numel) + block.x - 1) / block.x);

    const int8_t* in_ptr = static_cast<const int8_t*>(input.raw_data_ptr());
    const __half* sc_ptr = static_cast<const __half*>(scale.raw_data_ptr());
    __half* out_ptr      = static_cast<__half*>(output.raw_data_ptr());
    
    dequantize_int8_fp16_kernel<<<grid, block, 0, stream>>>(
        in_ptr, sc_ptr, out_ptr, K, numel, per_channel);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("dequantize_int8 launch failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace cuda_ops
}  // namespace sf
