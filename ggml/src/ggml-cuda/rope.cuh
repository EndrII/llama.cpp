#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// decode attention prep in one launch (see rope.cu); launch == false only checks that the nodes match
bool ggml_cuda_op_attn_prep(ggml_backend_cuda_context & ctx, bool launch,
        const ggml_tensor * rms_q, const ggml_tensor * mul_q, const ggml_tensor * rope_q,
        const ggml_tensor * rms_k, const ggml_tensor * mul_k, const ggml_tensor * rope_k,
        const ggml_tensor * sr_k, const ggml_tensor * sr_v, const float * kx_ov = nullptr, const float * vx_ov = nullptr);
