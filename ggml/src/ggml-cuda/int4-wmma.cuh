#pragma once
// INT4 activations x symmetric-INT4 weights on RDNA3 WMMA (opt-in, experimental).  See int4-wmma.cu.
#include "common.cuh"

// true if src0 is one of the rotated Q4_0 weight tensors (GGML_INT4 >= 1): its activations must be rotated before ANY matmul with it
bool ggml_cuda_int4_rotated(const ggml_tensor * src0);

// fp32 copy of src1 rotated with x' = (H D) x over 256-blocks of each row; dst has room for ggml_nelements(src1) floats
void ggml_cuda_int4_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float * dst);

// INT4 WMMA prefill matmul (GGML_INT4 >= 2).  Returns false when not applicable (caller falls back to rotate + regular path).
bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// decode-sized batches: rotate + quantize to Q8_1 in one kernel (used inside mul_mat_vec_q for rotated weights).
// ggml_cuda_int4_fused_ok tells whether src1 qualifies; the caller must then use ggml_cuda_mul_mat_vec_q directly.
bool ggml_cuda_int4_fused_ok(const ggml_tensor * src1);
void ggml_cuda_int4_rot_q8_1(const ggml_tensor * src1, void * q8_1, cudaStream_t stream);
