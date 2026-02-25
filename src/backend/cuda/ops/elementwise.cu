// SPDX-License-Identifier: Apache-2.0
// Implementations for ops declared in elementwise.cuh that require their
// own compilation unit (cast, embedding, cat, softmax).
#include "elementwise.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <string>

namespace sf {
namespace cuda_ops {

// ─────────────────────────────────────────────────────────────────────────────
// cast_kernel: type-convert every element  src → dst
// ─────────────────────────────────────────────────────────────────────────────
template<typename Src, typename Dst>
__global__ void cast_kernel_impl(const Src* __restrict__ src,
                                  Dst* __restrict__ dst,
                                  int64_t n) {
    int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x)  * blockDim.x;
    for (; idx < n; idx += stride)
        dst[idx] = static_cast<Dst>(static_cast<float>(src[idx]));
}

void cast_kernel(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    int64_t n = src.numel();
    int64_t grid = grid_size(n);

#define CAST_CASE(SrcT, DstT)                                                  \
    cast_kernel_impl<SrcT, DstT><<<grid, kBlockSize, 0, stream>>>(             \
        src.data_ptr<SrcT>(), dst.data_ptr<DstT>(), n)

    if      (src.dtype() == DType::Float32  && dst.dtype() == DType::Float16)
        CAST_CASE(float,           __half);
    else if (src.dtype() == DType::Float32  && dst.dtype() == DType::BFloat16)
        CAST_CASE(float,           __nv_bfloat16);
    else if (src.dtype() == DType::Float16  && dst.dtype() == DType::Float32)
        CAST_CASE(__half,          float);
    else if (src.dtype() == DType::BFloat16 && dst.dtype() == DType::Float32)
        CAST_CASE(__nv_bfloat16,   float);
    else if (src.dtype() == DType::Float16  && dst.dtype() == DType::BFloat16)
        CAST_CASE(__half,          __nv_bfloat16);
    else if (src.dtype() == DType::BFloat16 && dst.dtype() == DType::Float16)
        CAST_CASE(__nv_bfloat16,   __half);
    else if (src.dtype() == dst.dtype()) {
        // Same dtype — plain memcpy.
        cudaMemcpyAsync(dst.raw_data_ptr(), src.raw_data_ptr(),
                        src.nbytes(), cudaMemcpyDeviceToDevice, stream);
    } else {
        throw std::runtime_error("cast_kernel: unsupported dtype pair");
    }
#undef CAST_CASE
}

// ─────────────────────────────────────────────────────────────────────────────
// embedding_lookup: out[i] = weight[indices[i]]
//   weight:  [vocab_size, embed_dim]  float32 or float16
//   indices: [B, seq_len]             int32
//   out:     [B, seq_len, embed_dim]
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void embedding_kernel(const T* __restrict__  weight,
                                  const int*  __restrict__ indices,
                                  T* __restrict__          out,
                                  int64_t num_tokens,
                                  int64_t embed_dim,
                                  int64_t vocab_size) {
    int64_t tok_idx = static_cast<int64_t>(blockIdx.x);
    int64_t dim_idx = static_cast<int64_t>(threadIdx.x)
                    + static_cast<int64_t>(blockIdx.y) * blockDim.x;
    if (tok_idx >= num_tokens || dim_idx >= embed_dim) return;

    int token_id = indices[tok_idx];
    if (token_id < 0 || token_id >= static_cast<int>(vocab_size)) {
        out[tok_idx * embed_dim + dim_idx] = static_cast<T>(0);
        return;
    }
    out[tok_idx * embed_dim + dim_idx] = weight[token_id * embed_dim + dim_idx];
}

void embedding_lookup(const Tensor& weight, const Tensor& indices,
                      Tensor& out, cudaStream_t stream) {
    int64_t vocab_size = weight.shape()[0];
    int64_t embed_dim  = weight.shape()[1];
    int64_t num_tokens = indices.numel();  // B * seq_len

    // One block per token; threads cover embed_dim.
    const int threads = 256;
    dim3 grid(static_cast<unsigned>(num_tokens),
              static_cast<unsigned>((embed_dim + threads - 1) / threads));

    switch (weight.dtype()) {
        case DType::Float32:
            embedding_kernel<float><<<grid, threads, 0, stream>>>(
                weight.data_ptr<float>(), indices.data_ptr<int>(),
                out.data_ptr<float>(), num_tokens, embed_dim, vocab_size);
            break;
        case DType::Float16:
            embedding_kernel<__half><<<grid, threads, 0, stream>>>(
                weight.data_ptr<__half>(), indices.data_ptr<int>(),
                out.data_ptr<__half>(), num_tokens, embed_dim, vocab_size);
            break;
        default:
            throw std::runtime_error("embedding_lookup: unsupported weight dtype");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// cat_kernel: concatenate tensors along dim 0 only (most common case in VLA).
// All inputs must have the same dtype and shape[1..].
// ─────────────────────────────────────────────────────────────────────────────
void cat_kernel(const std::vector<Tensor>& inputs, Tensor& out,
                int64_t dim, cudaStream_t stream) {
    char* dst = reinterpret_cast<char*>(out.raw_data_ptr());
    int elem_size = static_cast<int>(dtype_size(out.dtype()));

    if (dim == 0) {
        // dim=0: inputs are stacked along the first dimension.
        // Each input is a contiguous block — just copy sequentially.
        int64_t offset_bytes = 0;
        for (const auto& t : inputs) {
            size_t bytes = t.nbytes();
            cudaMemcpyAsync(dst + offset_bytes, t.raw_data_ptr(),
                            bytes, cudaMemcpyDeviceToDevice, stream);
            offset_bytes += static_cast<int64_t>(bytes);
        }
    } else if (dim == 1) {
        // dim=1: inputs are concatenated along the second dimension.
        // Each input has shape [rows, cols_i]; output has shape [rows, sum(cols_i)].
        // We interleave the rows: for each row r, copy row r from each input
        // sequentially into the output row.
        int64_t rows = inputs[0].shape()[0];  // all inputs must have same rows

        // Compute per-input "row" widths in bytes (bytes to copy per outer-dim index).
        // For shape [d0, d1, ...], we copy d1*...*dn elements.
        // This handles [B, N] (2D) and [B, N, D] (3D) correctly if we cat along dim=1
        // ONLY IF the trailing dimensions match.
        std::vector<int64_t> col_bytes;
        for (const auto& t : inputs) {
             // stride(0) in bytes = numel / shape[0] * elem_size
             int64_t row_elems = t.numel() / t.shape()[0];
             col_bytes.push_back(row_elems * elem_size);
        }

        int64_t out_row_bytes = 0;
        for (auto cb : col_bytes) out_row_bytes += cb;

        for (int64_t r = 0; r < rows; ++r) {
            int64_t dst_col_offset = 0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                const char* src = reinterpret_cast<const char*>(inputs[i].raw_data_ptr());
                src += r * col_bytes[i];  // row r of input i
                cudaMemcpyAsync(dst + r * out_row_bytes + dst_col_offset,
                                src, col_bytes[i],
                                cudaMemcpyDeviceToDevice, stream);
                dst_col_offset += col_bytes[i];
            }
        }
    } else {
        throw std::runtime_error("cat_kernel: only dim=0 and dim=1 are supported. Got dim=" + std::to_string(dim));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// softmax_kernel: numerically stable row-wise softmax over last dim.
// Supports float32 and float16 (fp16 is computed in fp32 for stability).
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
__global__ void softmax_kernel_impl(const T* __restrict__ x,
                                     T* __restrict__ out,
                                     int64_t rows, int64_t cols) {
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    const T* row_x   = x   + row * cols;
    T*       row_out = out + row * cols;

    // Find max for numerical stability.
    float max_val = -1e38f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        max_val = fmaxf(max_val, static_cast<float>(row_x[c]));
    // Warp reduce max.
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));

    // Sum of exp.
    float sum = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        sum += expf(static_cast<float>(row_x[c]) - max_val);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    float inv_sum = 1.f / (sum + 1e-12f);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
        row_out[c] = static_cast<T>(expf(static_cast<float>(row_x[c]) - max_val) * inv_sum);
}

void softmax_kernel(const Tensor& x, Tensor& out, int64_t dim,
                    cudaStream_t stream) {
    // Only support softmax over the last dimension.
    if (dim != -1 && dim != x.ndim() - 1)
        throw std::runtime_error("softmax_kernel: only last-dim softmax supported");

    int64_t cols = x.shape()[x.ndim() - 1];
    int64_t rows = x.numel() / cols;

    // 32 threads per row (one warp), multiple rows per block.
    const int rows_per_block = 4;
    dim3 block(32, rows_per_block);
    dim3 grid(static_cast<unsigned>((rows + rows_per_block - 1) / rows_per_block));

    switch (x.dtype()) {
        case DType::Float32:
            softmax_kernel_impl<float><<<grid, block, 0, stream>>>(
                x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
            break;
        case DType::Float16:
            softmax_kernel_impl<__half><<<grid, block, 0, stream>>>(
                x.data_ptr<__half>(), out.data_ptr<__half>(), rows, cols);
            break;
        default:
            throw std::runtime_error("softmax_kernel: unsupported dtype");
    }
}

}  // namespace cuda_ops
}  // namespace sf
