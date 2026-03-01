// SPDX-License-Identifier: Apache-2.0
#include "servoflow/backend/cuda/cuda_backend.h"
#include "backend/cuda/ops/attention.h"
#include "backend/cuda/ops/elementwise.cuh"
#include "backend/cuda/ops/norm.cuh"
#include "backend/cuda/ops/quantization.cuh"

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
    // No explicit cleanup needed for async pool (driver manages it)
}

void* CUDAMemoryPool::alloc(size_t bytes, cudaStream_t stream) {
    SF_CUDA_CHECK(cudaSetDevice(device_index_));
    void* ptr = nullptr;
    // Use cudaMallocAsync for graph-safe allocation
    cudaError_t err = cudaMallocAsync(&ptr, bytes, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDAMemoryPool::alloc failed: bytes=%zu, stream=%p, error=%s\n",
                bytes, (void*)stream, cudaGetErrorString(err));
        throw std::runtime_error("cudaMallocAsync failed");
    }
    return ptr;
}

void CUDAMemoryPool::free(void* ptr, size_t bytes, cudaStream_t stream) {
    (void)bytes; // Not needed for cudaFreeAsync
    if (ptr) {
        // Use cudaFreeAsync for graph-safe deallocation
        cudaFreeAsync(ptr, stream);
    }
}

void CUDAMemoryPool::empty_cache() {
    // cudaMallocAsync pool is managed by the driver.
    // We can hint to release memory using cudaDeviceSetLimit?
    // For now, do nothing or just synchronize.
}

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
    
    // Initialize cublasLt
    SF_CUBLAS_CHECK(cublasLtCreate(&cublas_lt_));
}

CUDABackend::~CUDABackend() {
    // cublasDestroy may fail if the CUDA driver is already shutting down.
    if (cublas_) {
        cudaError_t e = cudaSetDevice(device_index_);
        if (e != cudaErrorCudartUnloading) {
            cublasDestroy(cublas_);
            if (cublas_lt_) cublasLtDestroy(cublas_lt_);
            if (attention_workspace_) {
                cudaFree(attention_workspace_);
            }
        }
        cublas_ = nullptr;
        cublas_lt_ = nullptr;
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
    
    // Always use our async-aware memory pool
    void* ptr = pool_.alloc(bytes, cuda_stream);
    Device dev(DeviceType::CUDA, device_index_);

    // Capture pool reference and stream for the deleter
    CUDAMemoryPool* pool_ptr = &pool_;
    auto deleter = [pool_ptr, bytes, cuda_stream](void* p) { 
        pool_ptr->free(p, bytes, cuda_stream); 
    };
    
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

void CUDABackend::gemm_bias_act(const Tensor& A, const Tensor& B, 
                                const Tensor& bias, Tensor& C,
                                ActivationType act,
                                float alpha, float beta,
                                bool trans_a, bool trans_b,
                                StreamHandle stream) {
    check_device(A, "A"); check_device(B, "B"); check_device(C, "C");
    if (bias.numel() > 0) check_device(bias, "bias");

    // Only support float16/bfloat16 for cublasLt acceleration
    bool is_fp16 = (A.dtype() == DType::Float16);
    bool is_bf16 = (A.dtype() == DType::BFloat16);
    
    if (!is_fp16 && !is_bf16) {
        // Fallback to base implementation
        IBackend::gemm_bias_act(A, B, bias, C, act, alpha, beta, trans_a, trans_b, stream);
        return;
    }

    cudaStream_t cuda_stream = to_stream(stream);
    
    // Create descriptors
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    try {
        cudaDataType_t dt = is_fp16 ? CUDA_R_16F : CUDA_R_16BF;
        cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
        cudaDataType_t scaleType = CUDA_R_32F;

        SF_CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, computeType, scaleType));
        
        // We use the standard trick: C_row = A_row * B_row  <=>  C_col^T = B_col^T * A_col^T
        // We swap A and B, and use Col Major (default).
        // cublasLtMatmul(B, A) -> C.
        
        cublasOperation_t opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        
        SF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        SF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        // Epilogue setup
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        if (bias.numel() > 0) {
            if (act == ActivationType::GELU) epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
            else if (act == ActivationType::ReLU) epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
            else epilogue = CUBLASLT_EPILOGUE_BIAS; 
        } else {
            if (act == ActivationType::GELU) epilogue = CUBLASLT_EPILOGUE_GELU;
            else if (act == ActivationType::ReLU) epilogue = CUBLASLT_EPILOGUE_RELU;
        }
        SF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        
        if (bias.numel() > 0) {
            const void* bias_ptr = bias.raw_data_ptr();
            SF_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
        }

        // Layouts (Col Major logic on Row Major data)
        // Rows/Cols are swapped. ld is the original columns (stride).
        
        // Matrix A (for cublas) is B (original).
        int rowsA = B.shape()[1]; // Cols of B
        int colsA = B.shape()[0]; // Rows of B
        int ldA   = B.shape()[1]; // Stride of B (cols)
        SF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, dt, rowsA, colsA, ldA));

        // Matrix B (for cublas) is A (original).
        int rowsB = A.shape()[1];
        int colsB = A.shape()[0];
        int ldB   = A.shape()[1];
        SF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, dt, rowsB, colsB, ldB));

        // Matrix C (for cublas) is C (original).
        int rowsC = C.shape()[1];
        int colsC = C.shape()[0];
        int ldC   = C.shape()[1];
        SF_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, dt, rowsC, colsC, ldC));

        // Heuristic
        SF_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        
        // Use existing workspace if possible
        void* workspace = nullptr;
        size_t workspaceSize = 0;
        {
            std::lock_guard<std::mutex> lk(attention_mu_);
            if (attention_workspace_size_ > 0) {
                workspace = attention_workspace_;
                workspaceSize = attention_workspace_size_;
            }
        }
        SF_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        int returnedResults = 0;
        SF_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(cublas_lt_, opDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

        if (returnedResults == 0) {
            // Fallback
            IBackend::gemm_bias_act(A, B, bias, C, act, alpha, beta, trans_a, trans_b, stream);
        } else {
            // Swap A and B in the call
            SF_CUBLAS_CHECK(cublasLtMatmul(cublas_lt_, opDesc, 
                &alpha, B.raw_data_ptr(), Adesc, 
                A.raw_data_ptr(), Bdesc, 
                &beta, C.raw_data_ptr(), Cdesc, 
                C.raw_data_ptr(), Cdesc, 
                &heuristicResult.algo, workspace, workspaceSize, cuda_stream));
                
            if (act == ActivationType::SiLU) {
                 silu(C, C, stream);
            }
        }
    } catch (...) {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (opDesc) cublasLtMatmulDescDestroy(opDesc);
        throw;
    }

    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (opDesc) cublasLtMatmulDescDestroy(opDesc);
}

// ── Attention ─────────────────────────────────────────────────────────────────
void CUDABackend::attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                            Tensor& out,
                            const Tensor* mask,
                            float scale, bool is_causal,
                            StreamHandle stream) {
    check_device(Q, "Q"); check_device(K, "K");
    check_device(V, "V"); check_device(out, "out");

    int64_t B  = Q.shape()[0];
    int64_t H  = Q.shape()[1];
    int64_t Sq = Q.shape()[2];
    
    // For FlashAttention (fp16/bf16), we need a workspace for softmax_lse.
    size_t lse_bytes = static_cast<size_t>(B) * H * Sq * sizeof(float);
    size_t padding = 128 * 1024; 
    size_t required = lse_bytes + padding;

    // Use cudaMallocAsync for graph-safe workspace allocation.
    // The driver handles memory reuse, so we don't need a manual persistent buffer.
    void* ws_ptr = nullptr;
    SF_CUDA_CHECK(cudaMallocAsync(&ws_ptr, required, to_stream(stream)));

    try {
        cuda_ops::flash_attention(Q, K, V, out, mask, scale, is_causal,
                                  ws_ptr, required,
                                  to_stream(stream));
    } catch (...) {
        cudaFreeAsync(ws_ptr, to_stream(stream));
        throw;
    }
    SF_CUDA_CHECK(cudaFreeAsync(ws_ptr, to_stream(stream)));
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

void CUDABackend::fused_add_rms_norm(const Tensor& input, Tensor& residual,
                                     const Tensor& gamma, Tensor& out,
                                     float eps, StreamHandle stream) {
    check_device(input, "input"); check_device(residual, "residual");
    cuda_ops::fused_add_rms_norm_kernel(input, residual, gamma, out, eps, to_stream(stream));
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

// ── Dequantize ────────────────────────────────────────────────────────────────
void CUDABackend::dequantize(const Tensor& input, const Tensor& scale,
                             Tensor& output, StreamHandle stream) {
    check_device(input, "input"); check_device(scale, "scale"); check_device(output, "output");
    cuda_ops::dequantize_int8(input, scale, output, to_stream(stream));
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
