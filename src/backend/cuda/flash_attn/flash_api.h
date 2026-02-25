#pragma once
#include "flash.h"
#include "cutlass/numeric_types.h"

namespace flash_attn {
    void mha_fwd(const void* q_ptr, const void* k_ptr, const void* v_ptr,
                 void* o_ptr, void* softmax_lse_ptr,
                 int B, int Sq, int Sk, int H, int H_k, int D,
                 float scale, bool is_causal,
                 int window_size_left, int window_size_right,
                 float softcap,
                 bool is_bf16,
                 cudaStream_t stream);
}
