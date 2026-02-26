#include "flash_attn/flash_api.h"
#include "flash_attn/static_switch.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace flash_attn {

void mha_fwd(const void* q_ptr, const void* k_ptr, const void* v_ptr,
             void* o_ptr, void* softmax_lse_ptr,
             int B, int Sq, int Sk, int H, int H_k, int D,
             float scale, bool is_causal,
             int window_size_left, int window_size_right,
             float softcap,
             bool is_bf16,
             cudaStream_t stream,
             bool is_BSHD)
{
    FLASH_NAMESPACE::Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;

    params.q_ptr = const_cast<void*>(q_ptr);
    params.k_ptr = const_cast<void*>(k_ptr);
    params.v_ptr = const_cast<void*>(v_ptr);
    params.o_ptr = o_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.b = B;
    params.h = H;
    params.h_k = H_k;
    params.h_h_k_ratio = H / H_k;
    params.seqlen_q = Sq;
    params.seqlen_k = Sk;
    params.seqlen_q_rounded = Sq;
    params.seqlen_k_rounded = Sk;
    params.d = D;
    params.d_rounded = D;
    
    // ServoFlow: [B, H, S, D]
    // FlashAttention params: strides are in elements
    // If input is [B, H, S, D] (is_BSHD=false):
    // - D is contiguous (stride 1)
    // - S stride = D
    // - H stride = S * D
    // - B stride = H * S * D
    // If input is [B, S, H, D] (is_BSHD=true):
    // - D is contiguous (stride 1)
    // - H stride = D
    // - S stride = H * D
    // - B stride = S * H * D
    
    if (is_BSHD) {
        params.q_row_stride = H * D;
        params.k_row_stride = H_k * D;
        params.v_row_stride = H_k * D;
        params.o_row_stride = H * D;

        params.q_head_stride = D;
        params.k_head_stride = D;
        params.v_head_stride = D;
        params.o_head_stride = D;

        params.q_batch_stride = Sq * H * D;
        params.k_batch_stride = Sk * H_k * D;
        params.v_batch_stride = Sk * H_k * D;
        params.o_batch_stride = Sq * H * D;
    } else {
        params.q_row_stride = D;
        params.k_row_stride = D;
        params.v_row_stride = D;
        params.o_row_stride = D;

        params.q_head_stride = Sq * D;
        params.k_head_stride = Sk * D;
        params.v_head_stride = Sk * D;
        params.o_head_stride = Sq * D;

        params.q_batch_stride = H * Sq * D;
        params.k_batch_stride = H_k * Sk * D;
        params.v_batch_stride = H_k * Sk * D;
        params.o_batch_stride = H * Sq * D;
    }

    /*
    printf("DEBUG: FlashAttention params:\n");
    printf("  is_bf16: %d\n", is_bf16);
    printf("  B=%d H=%d Sq=%d Sk=%d D=%d\n", B, H, Sq, Sk, D);
    printf("  q_row_stride: %ld\n", (long)params.q_row_stride);
    printf("  q_head_stride: %ld\n", (long)params.q_head_stride);
    printf("  q_batch_stride: %ld\n", (long)params.q_batch_stride);
    printf("  k_row_stride: %ld\n", (long)params.k_row_stride);
    printf("  k_head_stride: %ld\n", (long)params.k_head_stride);
    printf("  k_batch_stride: %ld\n", (long)params.k_batch_stride);
    printf("  v_row_stride: %ld\n", (long)params.v_row_stride);
    printf("  v_head_stride: %ld\n", (long)params.v_head_stride);
    printf("  v_batch_stride: %ld\n", (long)params.v_batch_stride);
    printf("  o_row_stride: %ld\n", (long)params.o_row_stride);
    printf("  o_head_stride: %ld\n", (long)params.o_head_stride);
    printf("  o_batch_stride: %ld\n", (long)params.o_batch_stride);
    printf("  is_BSHD: %d\n", is_BSHD);
    printf("  is_causal: %d\n", is_causal);
    printf("  window_size_left: %d\n", window_size_left);
    printf("  window_size_right: %d\n", window_size_right);
    printf("  softcap: %f\n", softcap);
    printf("  softmax_lse_ptr: %p\n", params.softmax_lse_ptr);
    printf("  q_ptr: %p\n", params.q_ptr);
    printf("  k_ptr: %p\n", params.k_ptr);
    printf("  v_ptr: %p\n", params.v_ptr);
    printf("  o_ptr: %p\n", params.o_ptr);
    */

    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * M_LOG2E;

    params.is_causal = is_causal;
    if (is_causal) {
        window_size_right = 0;
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    
    params.softcap = softcap;
    params.p_dropout = 0.0f;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Dispatch
    BOOL_SWITCH(is_causal, Is_causal_const, [&] {
        if (!is_bf16) {
            if (D == 32) {
                FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 32, Is_causal_const>(params, stream);
            } else if (D == 64) {
                FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 64, Is_causal_const>(params, stream);
            } else if (D == 128) {
                FLASH_NAMESPACE::run_mha_fwd_<cutlass::half_t, 128, Is_causal_const>(params, stream);
            } else {
                fprintf(stderr, "Unsupported HeadDim %d for dev build (only 32, 64, 128 supported)\n", D);
            }
        } else {
             fprintf(stderr, "Unsupported BF16 for dev build\n");
        }
        /*
        FP16_SWITCH(!is_bf16, [&] {
             HEADDIM_SWITCH(D, [&] {
                 FLASH_NAMESPACE::run_mha_fwd_<elem_type, kHeadDim, Is_causal_const>(params, stream);
             });
        });
        */
    });
}

} // namespace flash_attn
