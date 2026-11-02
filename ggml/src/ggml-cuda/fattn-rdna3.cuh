#pragma once

#include "common.cuh"

// WMMA flash attention (head size 256, f16 K/V) for RDNA3 prompt processing; returns false when the case is not supported
bool ggml_cuda_flash_attn_ext_rdna3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
