// SPDX-License-Identifier: Apache-2.0
#include "servoflow/backend/cuda/cuda_backend.h"
#include "backend/cuda/ops/attention.h"
#include "backend/cuda/ops/elementwise.cuh"
#include "backend/cuda/ops/norm.cuh"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

namespace sf {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error-checking helpers
// ─────────────────────────────────────────────────────────────────────────────
#define SF_CUDA_CHECK(expr)                                                    \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("CUDA error at " __FILE__ ":")                     \
                + std::to_string(__LINE__) + ": "                              \
                + cudaGetErrorString(_e));                                      \
    } while (0)

#define SF_CUBLAS_CHECK(expr)                                                  \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS)                                       \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error at " __FILE__ ":")                   \
                + std::to_string(__LINE__)                                     \
                + " code=" + std::to_string(static_cast<int>(_s)));            \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// CUDAMemoryPool
// ─────────────────────────────────────────────────────────────────────────────
CUDAMemoryPool::CUDAMemoryPool(int device_index) : device_index_(device_index) {}

CUDAMemoryPool::~CUDAMemoryPool() {
    try { empty_cache(); } catch (...) {}
}

size_t CUDAMemoryPool::round_up(size_t bytes) {
    // Round up to the next power of two (minimum 512 bytes).
    if (bytes <= 512) return 512;
    size_t p = 512;
    while (p < bytes) p <<= 1;
    return p;
}

void* CUDAMemoryPool::alloc(size_t bytes) {
    size_t bucket = round_up(bytes);
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto& list = free_blocks_[bucket];
        if (!list.empty()) {
            void* ptr = list.back().ptr;
            list.pop_back();
            cached_bytes_    -= bucket;
            allocated_bytes_ += bucket;
            return ptr;
        }
    }
    // No cached block; allocate from CUDA.
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    void* ptr = nullptr;
    SF_CUDA_CHECK(cudaMalloc(&ptr, bucket));
    {
        std::lock_guard<std::mutex> lk(mu_);
        allocated_bytes_ += bucket;
    }
    return ptr;
}

void CUDAMemoryPool::free(void* ptr, size_t bytes) {
    size_t bucket = round_up(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    free_blocks_[bucket].push_back({ptr, bucket});
    cached_bytes_    += bucket;
    allocated_bytes_ -= bucket;
}

void CUDAMemoryPool::empty_cache() {
    std::lock_guard<std::mutex> lk(mu_);
    cudaError_t set_err = cudaSetDevice(device_index_);
    // cudaErrorCudartUnloading means the process is exiting and the driver
    // has already started teardown — all device memory will be reclaimed
    // automatically; nothing for us to do.
    if (set_err == cudaErrorCudartUnloading) return;
    SF_CUDA_CHECK(set_err);
    for (auto& [size, blocks] : free_blocks_) {
        for (auto& b : blocks) {
            cudaFree(b.ptr);
            cached_bytes_ -= size;
        }
        blocks.clear();
    }
}

size_t CUDAMemoryPool::cached_bytes()    const { return cached_bytes_;    }
size_t CUDAMemoryPool::allocated_bytes() const { return allocated_bytes_; }

// ─────────────────────────────────────────────────────────────────────────────
// CUDAGraph
// ─────────────────────────────────────────────────────────────────────────────
CUDAGraph::~CUDAGraph() {
    if (exec_)  cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
}

void CUDAGraph::begin_capture(cudaStream_t stream) {
    if (exec_) {
        // Re-capture: destroy previous graph.
        cudaGraphExecDestroy(exec_);
        cudaGraphDestroy(graph_);
        exec_  = nullptr;
        graph_ = nullptr;
    }
    SF_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
}

void CUDAGraph::end_capture(cudaStream_t stream) {
    SF_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    SF_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
}

void CUDAGraph::launch(cudaStream_t stream) {
    if (!exec_)
        throw std::runtime_error("CUDAGraph::launch: no captured graph");
    SF_CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDABackend
// ─────────────────────────────────────────────────────────────────────────────
CUDABackend::CUDABackend(int device_index)
    : device_index_(device_index), pool_(device_index) {
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    SF_CUBLAS_CHECK(cublasCreate(&cublas_));
    // Enable Tensor Cores for fp16/bf16 workloads.
    SF_CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH));
}

CUDABackend::~CUDABackend() {
    // cublasDestroy may fail if the CUDA driver is already shutting down.
    if (cublas_) {
        cudaError_t e = cudaSetDevice(device_index_);
        if (e != cudaErrorCudartUnloading)
            cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
}

BackendCaps CUDABackend::caps() const {
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    cudaDeviceProp prop{};
    SF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index_));

    BackendCaps c;
    c.name                   = "CUDA";
    c.device_name            = prop.name;
    c.max_shared_mem_bytes   = prop.sharedMemPerBlock;
    c.total_device_mem_bytes = prop.totalGlobalMem;
    c.supports_fp16          = (prop.major >= 6);   // Pascal+
    c.supports_bf16          = (prop.major >= 8);   // Ampere+
    c.supports_int8          = (prop.major >= 6);
    c.supports_int4          = (prop.major >= 7);   // Turing+
    c.supports_graph         = true;
    return c;
}

void CUDABackend::check_device(const Tensor& t, const char* arg_name) const {
    if (!t.device().is_cuda() || t.device().index != device_index_)
        throw std::invalid_argument(
            std::string(arg_name) + " must reside on cuda:"
            + std::to_string(device_index_));
}

Tensor CUDABackend::alloc(Shape shape, DType dtype, StreamHandle stream) {
    size_t bytes = shape.nbytes(dtype_size(dtype));
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (cuda_stream != nullptr) {
        cudaError_t err = cudaStreamIsCapturing(cuda_stream, &capture_status);
        if (err != cudaSuccess) {
            // If checking capture status fails, assume no capture and clear error?
            // Or just proceed.
            cudaGetLastError(); // clear error
            capture_status = cudaStreamCaptureStatusNone;
        }
    }

    // If capturing, use cudaMallocAsync/cudaFreeAsync to be graph-friendly.
    if (capture_status == cudaStreamCaptureStatusActive) {
        void* ptr = nullptr;
        SF_CUDA_CHECK(cudaMallocAsync(&ptr, bytes, cuda_stream));
        
        // Deleter must use the same stream (or compatible) for async free.
        // Capturing the stream by value in the lambda.
        auto deleter = [ptr, cuda_stream](void*) {
            cudaFreeAsync(ptr, cuda_stream);
        };
        
        Device dev(DeviceType::CUDA, device_index_);
        auto storage = std::make_shared<Storage>(ptr, bytes, dev, deleter);
        return Tensor(std::move(storage), std::move(shape), dtype);
    }

    // Otherwise use our manual memory pool (synchronous alloc).
    void*  ptr   = pool_.alloc(bytes);
    Device dev(DeviceType::CUDA, device_index_);

    // Capture pool reference for the deleter (lambda captures by value).
    CUDAMemoryPool* pool_ptr = &pool_;
    auto deleter = [pool_ptr, bytes](void* p) { pool_ptr->free(p, bytes); };
    auto storage = std::make_shared<Storage>(ptr, bytes, dev, deleter);
    return Tensor(std::move(storage), std::move(shape), dtype);
}

Tensor CUDABackend::alloc_pinned(Shape shape, DType dtype) {
    size_t bytes = shape.nbytes(dtype_size(dtype));
    void*  ptr   = nullptr;
    SF_CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    auto deleter = [](void* p) { cudaFreeHost(p); };
    auto storage = std::make_shared<Storage>(ptr, bytes, kCPU, deleter);
    return Tensor(std::move(storage), std::move(shape), dtype);
}

void CUDABackend::empty_cache() {
    pool_.empty_cache();
}

void CUDABackend::copy(Tensor& dst, const Tensor& src, StreamHandle stream) {
    if (dst.nbytes() != src.nbytes())
        throw std::invalid_argument("copy: size mismatch");
    auto cuda_stream = to_stream(stream);

    bool src_gpu = src.device().is_cuda();
    bool dst_gpu = dst.device().is_cuda();

    cudaMemcpyKind kind;
    if      ( src_gpu &&  dst_gpu) kind = cudaMemcpyDeviceToDevice;
    else if (!src_gpu &&  dst_gpu) kind = cudaMemcpyHostToDevice;
    else if ( src_gpu && !dst_gpu) kind = cudaMemcpyDeviceToHost;
    else                           kind = cudaMemcpyHostToHost;

    if (cuda_stream) {
        SF_CUDA_CHECK(cudaMemcpyAsync(dst.raw_data_ptr(),
                                      src.raw_data_ptr(),
                                      src.nbytes(), kind, cuda_stream));
    } else {
        SF_CUDA_CHECK(cudaMemcpy(dst.raw_data_ptr(),
                                  src.raw_data_ptr(),
                                  src.nbytes(), kind));
    }
}

void CUDABackend::fill(Tensor& dst, float value, StreamHandle stream) {
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    // Use cudaMemsetAsync for zero; for non-zero we launch a kernel.
    cuda_ops::fill_kernel(dst, value, to_stream(stream));
}

StreamHandle CUDABackend::create_stream() {
    cudaStream_t s = nullptr;
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    SF_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    return reinterpret_cast<StreamHandle>(s);
}

void CUDABackend::destroy_stream(StreamHandle h) {
    if (h) SF_CUDA_CHECK(cudaStreamDestroy(to_stream(h)));
}

void CUDABackend::sync_stream(StreamHandle h) {
    SF_CUDA_CHECK(cudaStreamSynchronize(to_stream(h)));
}

void CUDABackend::sync_device() {
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    SF_CUDA_CHECK(cudaDeviceSynchronize());
}

// ── GEMM ─────────────────────────────────────────────────────────────────────
void CUDABackend::gemm_impl(const Tensor& A, const Tensor& B, Tensor& C,
                            float alpha, float beta,
                            bool trans_a, bool trans_b,
                            cudaStream_t stream) {
    SF_CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    int M = static_cast<int>(trans_a ? A.shape()[1] : A.shape()[0]);
    int N = static_cast<int>(trans_b ? B.shape()[0] : B.shape()[1]);
    int K = static_cast<int>(trans_a ? A.shape()[0] : A.shape()[1]);

    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cuBLAS is column-major; we compute C^T = B^T @ A^T.
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        cudaDataType_t dt = (A.dtype() == DType::Float16) ? CUDA_R_16F : CUDA_R_16BF;
        cublasComputeType_t ct = CUBLAS_COMPUTE_32F;
        SF_CUBLAS_CHECK(cublasGemmEx(
            cublas_,
            op_b, op_a,
            N, M, K,
            &alpha,
            B.raw_data_ptr(), dt, trans_b ? K : N,
            A.raw_data_ptr(), dt, trans_a ? M : K,
            &beta,
            C.raw_data_ptr(), dt, N,
            ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // Float32 fallback.
        SF_CUBLAS_CHECK(cublasSgemm(
            cublas_,
            op_b, op_a,
            N, M, K,
            &alpha,
            static_cast<const float*>(B.raw_data_ptr()), trans_b ? K : N,
            static_cast<const float*>(A.raw_data_ptr()), trans_a ? M : K,
            &beta,
            static_cast<float*>(C.raw_data_ptr()), N));
    }
}

void CUDABackend::gemm(const Tensor& A, const Tensor& B, Tensor& C,
                       float alpha, float beta, bool trans_a, bool trans_b,
                       StreamHandle stream) {
    check_device(A, "A"); check_device(B, "B"); check_device(C, "C");
    gemm_impl(A, B, C, alpha, beta, trans_a, trans_b, to_stream(stream));
}

void CUDABackend::batched_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                               float alpha, float beta, bool trans_a, bool trans_b,
                               StreamHandle stream) {
    check_device(A, "A"); check_device(B, "B"); check_device(C, "C");
    SF_CUBLAS_CHECK(cublasSetStream(cublas_, to_stream(stream)));

    int batch = static_cast<int>(A.shape()[0]);
    int M = static_cast<int>(trans_a ? A.shape()[2] : A.shape()[1]);
    int N = static_cast<int>(trans_b ? B.shape()[1] : B.shape()[2]);
    int K = static_cast<int>(trans_a ? A.shape()[1] : A.shape()[2]);
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    long long stride_a = M * K;
    long long stride_b = K * N;
    long long stride_c = M * N;

    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        cudaDataType_t dt = (A.dtype() == DType::Float16) ? CUDA_R_16F : CUDA_R_16BF;
        SF_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_,
            op_b, op_a,
            N, M, K,
            &alpha,
            B.raw_data_ptr(), dt, trans_b ? K : N, stride_b,
            A.raw_data_ptr(), dt, trans_a ? M : K, stride_a,
            &beta,
            C.raw_data_ptr(), dt, N, stride_c,
            batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        SF_CUBLAS_CHECK(cublasSgemmStridedBatched(
            cublas_,
            op_b, op_a,
            N, M, K,
            &alpha,
            static_cast<const float*>(B.raw_data_ptr()), trans_b ? K : N, stride_b,
            static_cast<const float*>(A.raw_data_ptr()), trans_a ? M : K, stride_a,
            &beta,
            static_cast<float*>(C.raw_data_ptr()), N, stride_c,
            batch));
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────
void CUDABackend::attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                            Tensor& out, const Tensor* mask,
                            float scale, bool is_causal,
                            StreamHandle stream) {
    check_device(Q, "Q"); check_device(K, "K");
    check_device(V, "V"); check_device(out, "out");
    cuda_ops::flash_attention(Q, K, V, out, mask, scale, is_causal,
                              to_stream(stream));
}

// ── Normalization ─────────────────────────────────────────────────────────────
void CUDABackend::layer_norm(const Tensor& x, const Tensor& gamma,
                             const Tensor& beta, Tensor& out, float eps,
                             StreamHandle stream) {
    check_device(x, "x");
    cuda_ops::layer_norm_kernel(x, gamma, beta, out, eps, to_stream(stream));
}

void CUDABackend::rms_norm(const Tensor& x, const Tensor& gamma,
                           Tensor& out, float eps, StreamHandle stream) {
    check_device(x, "x");
    cuda_ops::rms_norm_kernel(x, gamma, out, eps, to_stream(stream));
}

// ── Element-wise ──────────────────────────────────────────────────────────────
void CUDABackend::add(const Tensor& a, const Tensor& b, Tensor& out,
                      StreamHandle stream) {
    cuda_ops::elementwise_binary<cuda_ops::AddOp>(a, b, out, to_stream(stream));
}

void CUDABackend::mul(const Tensor& a, const Tensor& b, Tensor& out,
                      StreamHandle stream) {
    cuda_ops::elementwise_binary<cuda_ops::MulOp>(a, b, out, to_stream(stream));
}

void CUDABackend::scale(const Tensor& a, float scalar, Tensor& out,
                        StreamHandle stream) {
    cuda_ops::scale_kernel(a, scalar, out, to_stream(stream));
}

void CUDABackend::gelu(const Tensor& x, Tensor& out, StreamHandle stream) {
    cuda_ops::activation_kernel<cuda_ops::GeluOp>(x, out, to_stream(stream));
}

void CUDABackend::silu(const Tensor& x, Tensor& out, StreamHandle stream) {
    cuda_ops::activation_kernel<cuda_ops::SiluOp>(x, out, to_stream(stream));
}

void CUDABackend::softmax(const Tensor& x, Tensor& out, int64_t dim,
                          StreamHandle stream) {
    cuda_ops::softmax_kernel(x, out, dim, to_stream(stream));
}

// ── Embedding ─────────────────────────────────────────────────────────────────
void CUDABackend::embedding(const Tensor& weight, const Tensor& indices,
                            Tensor& out, StreamHandle stream) {
    cuda_ops::embedding_lookup(weight, indices, out, to_stream(stream));
}

// ── Cast ──────────────────────────────────────────────────────────────────────
void CUDABackend::cast(const Tensor& src, Tensor& dst, StreamHandle stream) {
    cuda_ops::cast_kernel(src, dst, to_stream(stream));
}

// ── Cat ───────────────────────────────────────────────────────────────────────
void CUDABackend::cat(const std::vector<Tensor>& inputs, Tensor& out,
                      int64_t dim, StreamHandle stream) {
    cuda_ops::cat_kernel(inputs, out, dim, to_stream(stream));
}

// ── Graph capture ─────────────────────────────────────────────────────────────
void CUDABackend::graph_begin_capture(StreamHandle stream) {
    auto cs = to_stream(stream);
    std::lock_guard<std::mutex> lk(graphs_mu_);
    graphs_[cs].begin_capture(cs);
}

void CUDABackend::graph_end_capture(StreamHandle stream) {
    auto cs = to_stream(stream);
    std::lock_guard<std::mutex> lk(graphs_mu_);
    graphs_[cs].end_capture(cs);
}

void CUDABackend::graph_launch(StreamHandle stream) {
    auto cs = to_stream(stream);
    std::lock_guard<std::mutex> lk(graphs_mu_);
    graphs_[cs].launch(cs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Static registration
// ─────────────────────────────────────────────────────────────────────────────
CUDABackendRegistrar::CUDABackendRegistrar() {
    BackendRegistry::instance().register_backend(
        DeviceType::CUDA,
        [](int index) -> BackendPtr {
            return std::make_shared<CUDABackend>(index);
        });
}

// Trigger registration at static-init time.
static CUDABackendRegistrar _cuda_registrar;

}  // namespace cuda
}  // namespace sf
