// Skinny bf16 matmul for RDNA3 prompt processing: dst[n][m] = sum_k src1[n][k] * src0[m][k] with bf16 weights (M <= 64, e.g. the GDN
// alpha/beta projections), f32 activations and f32 output.  Replaces f32->bf16 convert + hipBLASLt gemm + bf16->f32 convert (three
// launches) by one WMMA kernel; the activations are rounded to bf16 exactly as the convert did, but the result stays in f32.
#include "mm-skinny-rdna3.cuh"

#if defined(GGML_USE_HIP)

#include <cstdlib>

#if defined(__GFX11__) || !defined(__HIP_DEVICE_COMPILE__)
#define SK_OK 1
#endif

typedef float  sk_f8   __attribute__((ext_vector_type(8)));
typedef __bf16 sk_bf16 __attribute__((ext_vector_type(16)));

namespace {

constexpr int SK_MAXMT = 4;     // M <= 64

__device__ __forceinline__ uint32_t sk_bf16_bits(float f) {      // round to nearest even
    const uint32_t b = __float_as_uint(f);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}

__global__ void __launch_bounds__(128) sk_kernel(const __bf16 * __restrict__ W, const float * __restrict__ X, float * __restrict__ out, int M, int N, int K) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, hh = lane >> 4, lc = lane & 15;
    const int mt_n = M / 16;
    const int64_t t0 = (int64_t) blockIdx.x * 16;
    const int kq = K / 4;
    __shared__ float red[4][16][SK_MAXMT * 16 + 1];

    sk_f8 acc[SK_MAXMT];
#pragma unroll
    for (int mt = 0; mt < SK_MAXMT; mt++) { acc[mt] = sk_f8{0, 0, 0, 0, 0, 0, 0, 0}; }

    const int64_t tok = t0 + lc;
    const float * xr = X + (tok < N ? tok : 0) * (int64_t) K;
    for (int k = w * kq; k < (w + 1) * kq; k += 16) {
        uint32_t ab[8];
        {
            const float4 * xp = (const float4 *) (xr + k);
            const float4 f0 = xp[0], f1 = xp[1], f2 = xp[2], f3 = xp[3];
            const float fv[16] = { f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w, f2.x, f2.y, f2.z, f2.w, f3.x, f3.y, f3.z, f3.w };
#pragma unroll
            for (int i = 0; i < 8; i++) {
                ab[i] = tok < N ? (sk_bf16_bits(fv[2 * i]) | (sk_bf16_bits(fv[2 * i + 1]) << 16)) : 0u;
            }
        }
        const sk_bf16 a = __builtin_bit_cast(sk_bf16, ab);
#pragma unroll
        for (int mt = 0; mt < SK_MAXMT; mt++) {
            if (mt < mt_n) {
                const uint4 * wp = (const uint4 *) (W + (int64_t) (16 * mt + lc) * K + k);
                struct { uint4 x, y; } t = { wp[0], wp[1] };
                const sk_bf16 b = __builtin_bit_cast(sk_bf16, t);
#ifdef SK_OK
                acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[mt]);
#endif
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < SK_MAXMT; mt++) {
        if (mt < mt_n) {
#pragma unroll
            for (int i = 0; i < 8; i++) { red[w][2 * i + hh][16 * mt + lc] = acc[mt][i]; }
        }
    }
    __syncthreads();
    for (int idx = tid; idx < 16 * M; idx += 128) {
        const int r = idx / M, c = idx % M;
        const int64_t t = t0 + r;
        if (t < N) { out[t * M + c] = (red[0][r][c] + red[1][r][c]) + (red[2][r][c] + red[3][r][c]); }
    }
}

} // namespace

bool ggml_cuda_mul_mat_skinny_rdna3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool enabled = !getenv("GGML_MM_SKINNY") || atoi(getenv("GGML_MM_SKINNY")) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1] * src1->ne[2] * src1->ne[3];
    if (M > 16 * SK_MAXMT || M % 16 || K % 64 || N <= 8 || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != K) { return false; }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) || dst->ne[0] != M || ggml_nelements(dst) != N * M) { return false; }
    if (((uintptr_t) src0->data & 15) || ((uintptr_t) src1->data & 15)) { return false; }
    sk_kernel<<<dim3((unsigned) ((N + 15) / 16)), 128, 0, ctx.stream()>>>((const __bf16 *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) N, (int) K);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mul_mat_skinny_rdna3(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }

#endif // GGML_USE_HIP
