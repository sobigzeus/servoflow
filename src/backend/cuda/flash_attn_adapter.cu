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
             cudaStream_t stream)
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
    // If input is [B, H, S, D]:
    // - D is contiguous (stride 1)
    // - S stride = D
    // - H stride = S * D
    // - B stride = H * S * D
    
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
        FP16_SWITCH(!is_bf16, [&] {
             HEADDIM_SWITCH(D, [&] {
                 FLASH_NAMESPACE::run_mha_fwd_<elem_type, kHeadDim, Is_causal_const>(params, stream);
             });
        });
    });
}

} // namespace flash_attn
