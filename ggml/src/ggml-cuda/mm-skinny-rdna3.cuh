#pragma once

#include "common.cuh"

// one-kernel bf16 x f32 -> f32 matmul for skinny weights (M <= 64) on RDNA3 prompt processing; false when not applicable
bool ggml_cuda_mul_mat_skinny_rdna3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                    int epi = 0, const float * bias = nullptr, const float * scl = nullptr);   // epi: 1 sigmoid, 2 softplus(x + bias) * scl

bool ggml_cuda_mul_mat_skinny_pair_rdna3(ggml_backend_cuda_context & ctx, const ggml_tensor * w0, const ggml_tensor * w1, const ggml_tensor * src1,
                                         ggml_tensor * dst0, int epi0, const float * bias0, const float * scl0, ggml_tensor * dst1, int epi1);
