// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include <cuda_runtime.h>

namespace sf {
namespace cuda_ops {

// Dequantize INT8 tensor to FP16/BF16 using scale.
// Supports per-channel (axis=0) or per-tensor scaling.
// input: [N, K] (row-major) or [K, N] (col-major) depending on layout.
// scale: [N] or [1].
// output: same shape as input.
void dequantize_int8(const Tensor& input, const Tensor& scale, Tensor& output, cudaStream_t stream);

}  // namespace cuda_ops
}  // namespace sf
