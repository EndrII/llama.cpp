#pragma once
#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_op_conv_cluster(ggml_backend_cuda_context & ctx, const ggml_tensor * gather, const ggml_tensor * concat, const ggml_tensor * const * cpys, int n_cpy,
                               const ggml_tensor * conv, const ggml_tensor * silu);

// decode (one token, one sequence) conv cluster folded into the epilogue of the matvec that produces its input: per channel c, with x = {state[0..2], value}:
//   out[c] = silu(sum_j x[j] * w[c][j]),  and the new state windows x[start .. start+2] go to the cache rows
struct ggml_cuda_conv_fold {
    const float * state_base; const int32_t * ids; int64_t state_row_stride; const float * w; int64_t w_stride; float * out;
    float * dst[4]; int start[4]; int n;
};
bool ggml_cuda_conv_fold_args(const ggml_tensor * gather, const ggml_tensor * concat, const ggml_tensor * const * cpys, int n_cpy,
                              const ggml_tensor * conv, const ggml_tensor * silu, ggml_cuda_conv_fold * f);
