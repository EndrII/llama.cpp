#pragma once
// INT4 activations x symmetric-INT4 weights on RDNA3 WMMA (opt-in, experimental).  See int4-wmma.cu.
#include "common.cuh"

// true if src0 is one of the rotated Q4_0 weight tensors (GGML_INT4 >= 1): its activations must be rotated before ANY matmul with it
bool ggml_cuda_int4_rotated(const ggml_tensor * src0);

// fp32 copy of src1 rotated with x' = (H D) x over 256-blocks of each row; dst has room for ggml_nelements(src1) floats
void ggml_cuda_int4_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float * dst);

// INT4 WMMA prefill matmul (GGML_INT4 >= 2).  Returns false when not applicable (caller falls back to rotate + regular path).
// res: added to the output (a residual ADD folded into the GEMM epilogue; dst is then the ADD node)
// glu_x / glu_y: pack silu(glu_x) * glu_y instead of src1 (whose producer was skipped; src1 only supplies the shape)
bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const float * glu_x = nullptr, const float * glu_y = nullptr, const float * res = nullptr);

// decode-sized batches: rotate + quantize to Q8_1 in one kernel (used inside mul_mat_vec_q for rotated weights).
// ggml_cuda_int4_fused_ok tells whether src1 qualifies; the caller must then use ggml_cuda_mul_mat_vec_q directly.
bool ggml_cuda_int4_fused_ok(const ggml_tensor * src1);
void ggml_cuda_int4_rot_q8_1(const ggml_tensor * src1, void * q8_1, cudaStream_t stream);

// ---- single-copy weights (GGML_INT4_INPLACE, default on) --------------------------------------------------------------------
// The repacked layout (int4 K-slices + fixed-point u16 scales) has exactly the byte size of the Q4_0 tensor, so a rotated weight is
// converted IN PLACE the first time a graph that uses it is computed (ggml_cuda_int4_prepare).  A converted tensor is no longer valid
// Q4_0: every matmul with it must go through ggml_cuda_int4_mul_mat (batches > 8) or ggml_cuda_int4_matvec (decode, <= 8 tokens).
// RMS norm * weight fused with the decode rotation+quantization (fills the activation cache for `mul`); false when not applicable
bool ggml_cuda_int4_rmsnorm_rot(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul);

bool ggml_cuda_int4_converted(const ggml_tensor * src0);
void ggml_cuda_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);
// decode-sized matmul (<= 8 tokens) on converted weights: fused rotate+Q8 quantisation, then a matvec on the repacked layout; supports the
// gate/bias/GLU fusion of mul_mat_vec_q
// glu_x / glu_y: the matvec's input is built from these instead of src1 (rotation prologue, one token), by `mode`:
//   1: silu(glu_x) * glu_y     2: glu_x * sigmoid(glu_y), glu_y read with ystride floats between 256-blocks     3: silu(glu_x) * (RMS norm over 128 of glu_y * nw)
void ggml_cuda_int4_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion,
                            const float * glu_x = nullptr, const float * glu_y = nullptr, int mode = 1, int ystride = 256, const float * nw = nullptr, float eps = 0.f);
// the device buffer [ptr, ptr+size) is going away: forget converted tensors inside it
void ggml_cuda_int4_forget(const void * ptr, size_t size);
