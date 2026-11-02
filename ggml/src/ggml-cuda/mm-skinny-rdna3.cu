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

constexpr int SK_MAXMT = 2;     // M <= 32
constexpr int SK_W = 16;        // waves per block, each takes a slice of K

__device__ __forceinline__ uint32_t sk_bf16_bits(float f) {      // round to nearest even
    const uint32_t b = __float_as_uint(f);
    return (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;
}

__global__ void __launch_bounds__(SK_W * 32) sk_kernel(const __bf16 * __restrict__ W, const float * __restrict__ X, float * __restrict__ out, int M, int N, int K,
                                                   int epi, const float * __restrict__ bias, const float * __restrict__ scl) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, hh = lane >> 4, lc = lane & 15;
    const int mt_n = M / 16;
    const int64_t t0 = (int64_t) blockIdx.x * 16;
    const int kq = K / SK_W;
    __shared__ float red[SK_W][16][SK_MAXMT * 16 + 1];

    sk_f8 acc[SK_MAXMT];
#pragma unroll
    for (int mt = 0; mt < SK_MAXMT; mt++) { acc[mt] = sk_f8{0, 0, 0, 0, 0, 0, 0, 0}; }

    const int64_t tok = t0 + lc;
    const float * xr = X + (tok < N ? tok : 0) * (int64_t) K;
    constexpr int U = 5;                                  // k-steps with their loads in flight together
    for (int k = w * kq; k < (w + 1) * kq; k += 16 * U) {
        float4 xf[U][4];
        uint4 wf[U][SK_MAXMT][2];
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (k + 16 * u < (w + 1) * kq) {
                const float4 * xp = (const float4 *) (xr + k + 16 * u);
                xf[u][0] = xp[0]; xf[u][1] = xp[1]; xf[u][2] = xp[2]; xf[u][3] = xp[3];
#pragma unroll
                for (int mt = 0; mt < SK_MAXMT; mt++) {
                    if (mt < mt_n) {
                        const uint4 * wp = (const uint4 *) (W + (int64_t) (16 * mt + lc) * K + k + 16 * u);
                        wf[u][mt][0] = wp[0]; wf[u][mt][1] = wp[1];
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            if (k + 16 * u < (w + 1) * kq) {
                const float fv[16] = { xf[u][0].x, xf[u][0].y, xf[u][0].z, xf[u][0].w, xf[u][1].x, xf[u][1].y, xf[u][1].z, xf[u][1].w,
                                       xf[u][2].x, xf[u][2].y, xf[u][2].z, xf[u][2].w, xf[u][3].x, xf[u][3].y, xf[u][3].z, xf[u][3].w };
                uint32_t ab[8];
#pragma unroll
                for (int i = 0; i < 8; i++) { ab[i] = tok < N ? (sk_bf16_bits(fv[2 * i]) | (sk_bf16_bits(fv[2 * i + 1]) << 16)) : 0u; }
                const sk_bf16 a = __builtin_bit_cast(sk_bf16, ab);
#pragma unroll
                for (int mt = 0; mt < SK_MAXMT; mt++) {
                    if (mt < mt_n) {
                        struct { uint4 x, y; } t = { wf[u][mt][0], wf[u][mt][1] };
                        const sk_bf16 b = __builtin_bit_cast(sk_bf16, t);
#ifdef SK_OK
                        acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[mt]);
#endif
                    }
                }
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
    for (int idx = tid; idx < 16 * M; idx += SK_W * 32) {
        const int r = idx / M, c = idx % M;
        const int64_t t = t0 + r;
        if (t < N) {
            float sum = 0.f;
#pragma unroll
            for (int q = 0; q < SK_W; q++) { sum += red[q][r][c]; }
            if (epi == 1) { sum = 1.0f / (1.0f + expf(-sum)); }                                           // sigmoid
            else if (epi == 2) { const float xb = sum + bias[c]; sum = (xb > 20.0f ? xb : logf(1.0f + expf(xb))) * scl[c]; }   // softplus(x + bias) * scale
            out[t * M + c] = sum;
        }
    }
}



// decode (1..8 tokens): one 128-thread block per output, f32 dot product over K (bf16 weights, f32 activations, like mul_mat_vec_f)
__global__ void __launch_bounds__(128) sk_decode_kernel(const __bf16 * __restrict__ W, const float * __restrict__ X, float * __restrict__ out, int M, int N, int K,
                                                         int epi, const float * __restrict__ bias, const float * __restrict__ scl) {
    const int c = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    float acc[8];
#pragma unroll
    for (int n = 0; n < 8; n++) { acc[n] = 0.f; }
    for (int k = tid * 8; k < K; k += 128 * 8) {
        const uint4 u = *(const uint4 *) (W + (int64_t) c * K + k);
        const uint32_t uw[4] = { u.x, u.y, u.z, u.w };
        float wv[8];
#pragma unroll
        for (int i = 0; i < 4; i++) { wv[2 * i] = __uint_as_float(uw[i] << 16); wv[2 * i + 1] = __uint_as_float(uw[i] & 0xFFFF0000u); }
#pragma unroll
        for (int n = 0; n < 8; n++) {
            if (n < N) {
                const float4 x0 = *(const float4 *) (X + (int64_t) n * K + k), x1 = *(const float4 *) (X + (int64_t) n * K + k + 4);
                acc[n] = fmaf(wv[0], x0.x, acc[n]); acc[n] = fmaf(wv[1], x0.y, acc[n]); acc[n] = fmaf(wv[2], x0.z, acc[n]); acc[n] = fmaf(wv[3], x0.w, acc[n]);
                acc[n] = fmaf(wv[4], x1.x, acc[n]); acc[n] = fmaf(wv[5], x1.y, acc[n]); acc[n] = fmaf(wv[6], x1.z, acc[n]); acc[n] = fmaf(wv[7], x1.w, acc[n]);
            }
        }
    }
    __shared__ float red[4][8];
#pragma unroll
    for (int n = 0; n < 8; n++) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor(v, o, 32); }
        if (lane == 0) { red[w][n] = v; }
    }
    __syncthreads();
    if (tid < N) {
        float v = (red[0][tid] + red[1][tid]) + (red[2][tid] + red[3][tid]);
        if (epi == 1) { v = 1.0f / (1.0f + expf(-v)); }
        else if (epi == 2) { const float xb = v + bias[c]; v = (xb > 20.0f ? xb : logf(1.0f + expf(xb))) * scl[c]; }
        out[(int64_t) tid * M + c] = v;
    }
}

} // namespace

bool ggml_cuda_mul_mat_skinny_rdna3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                    int epi, const float * bias, const float * scl) {
    static const bool enabled = !getenv("GGML_MM_SKINNY") || atoi(getenv("GGML_MM_SKINNY")) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    if (src0->type != GGML_TYPE_BF16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) { return false; }
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const bool decode = N >= 1 && N <= 8;
    if (M > 16 * SK_MAXMT || (decode ? (K % 8) : (M % 16 || K % (16 * SK_W) || N <= 8)) || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != K) { return false; }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) || (epi == 0 && dst->ne[0] != M) || ggml_nelements(dst) != N * M) { return false; }
    if (((uintptr_t) src0->data & 15) || ((uintptr_t) src1->data & 15)) { return false; }
    if (decode) {
        sk_decode_kernel<<<(unsigned) M, 128, 0, ctx.stream()>>>((const __bf16 *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) N, (int) K, epi, bias, scl);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    sk_kernel<<<dim3((unsigned) ((N + 15) / 16)), SK_W * 32, 0, ctx.stream()>>>((const __bf16 *) src0->data, (const float *) src1->data, (float *) dst->data, (int) M, (int) N, (int) K, epi, bias, scl);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mul_mat_skinny_rdna3(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, int, const float *, const float *) { return false; }

#endif // GGML_USE_HIP
