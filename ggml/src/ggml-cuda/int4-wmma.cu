// INT4 activations x symmetric-INT4 weights on RDNA3 WMMA (v_wmma_i32_16x16x16_iu4), opt-in and experimental.
//
//   GGML_INT4=1   Q4_0 "blk.*" weights are stored rotated (see scripts/rotate_quantize.py): every matmul with them first rotates the
//                 activations, x' = (H D) x per 256-block, then runs the regular path.  (correctness / baseline)
//   GGML_INT4=2   additionally, prefill-sized matmuls (>= GGML_INT4_MIN_N tokens) run the INT4 WMMA kernel below: rotate+quantize the
//                 activations to int4 (one scale per token per 256-block, expressed as a 4-bit multiplier of a per-token reference),
//                 then an int4 x int4 WMMA GEMM with integer sub-scales.
//   GGML_INT4=3   as 2 but with 8-bit activations (two int4 WMMA passes: q = 16*hi + lo) - the quality mode, any weight group size.
//   GGML_INT4_PREFILL=quality|balanced|fast  activation width per tensor kind in mode 3 (default quality = 8-bit everywhere; fast = 4-bit everywhere);
//                 GGML_INT4_A4=<name substrings> overrides the list, GGML_INT4_CLIP (default 0.85) is the 4-bit activation clip factor.
//   GGML_INT4_INPLACE  1 (default): the repacked layout (same byte size as Q4_0) overwrites the Q4_0 tensor the first time a graph uses it, so the weights
//                 take ~1x VRAM, and decode (<= 8 tokens) runs a matvec on that layout (k_mv).  0: keep the Q4_0 tensor and a repacked copy (~2x), stock mmvq for decode.
//   GGML_INT4_GW  K-groups of 32 that share one weight scale (1, 2 or 4; default 4) - must match how the model was quantized.
//
// The rotation follows Meta's HadaCore idea (H256 = H16 (x) H16 as two 16x16 matmuls); this is an independent RDNA3 implementation.
#include "int4-wmma.cuh"
#include "unary.cuh"

#if defined(GGML_USE_HIP)

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__GFX11__) || !defined(__HIP_DEVICE_COMPILE__)
#define INT4W_DEVICE_OK 1
#endif

namespace {

typedef int   int8v  __attribute__((ext_vector_type(8)));
typedef int   int4v  __attribute__((ext_vector_type(4)));
typedef int   int2v  __attribute__((ext_vector_type(2)));
typedef float float8v __attribute__((ext_vector_type(8)));
typedef _Float16 half16v __attribute__((ext_vector_type(16)));

constexpr int SH = 10;                 // fixed-point bits of the integer sub-scales (4-bit activations)
constexpr int SH8 = 12;                // same for the 8-bit activation kernel (GGML_INT4=3)
constexpr int MI = 2, NJ = 2, WAVES_N = 8;
constexpr int TILE_M = MI * 16;        // features per block
constexpr int TILE_N = WAVES_N * NJ * 16;  // tokens per block

#define INT4W_RSRC(p) __builtin_amdgcn_make_buffer_rsrc((void *)(p), 0, 0x7fffffff, 0x31004000)
__device__ __forceinline__ int4v bl128(__amdgpu_buffer_rsrc_t r, int voff, int soff) { return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0); }
__device__ __forceinline__ int2v bl64(__amdgpu_buffer_rsrc_t r, int voff, int soff) { return __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0); }
#define INT4W_LO(v) __builtin_shufflevector((v), (v), 0, 1)
#define INT4W_HI(v) __builtin_shufflevector((v), (v), 2, 3)
__device__ __forceinline__ void mul24a(int & out, int a, int b) { asm volatile("v_mul_i32_i24 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b)); }
__device__ __forceinline__ void mad24(int & acc, int a, int b) { asm volatile("v_mad_i32_i24 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b)); }

// slot s of a 16-nibble K-slice holds group-relative element P(s)
__host__ __device__ constexpr int P(int s) {
    return (s < 8) ? ((s & 1) ? 4 + (s >> 1) : (s >> 1)) : 8 + ((s - 8) & 1 ? 4 + ((s - 8) >> 1) : ((s - 8) >> 1));
}
// sign pattern D of the rotation (must match scripts/rotate_quantize.py)
__device__ __forceinline__ float sign256(int j) { return (((uint32_t) j * 2654435761u) >> 31) ? -1.0f : 1.0f; }
__device__ __forceinline__ _Float16 h16(int r, int c) { return (__builtin_popcount(r & c) & 1) ? (_Float16) -1.0f : (_Float16) 1.0f; }

// ---------------------------------------------------------------------------------------------------------------- weight repack
__global__ void k_row_scale(const block_q4_0 * __restrict__ W, int M, int nb, float * __restrict__ S) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x; if (m >= M) return;
    float mx = 0.f;
    for (int b = 0; b < nb; b++) mx = fmaxf(mx, fabsf(__half2float(W[(size_t) m * nb + b].d)));
    S[m] = mx;
}
// scale-group size of a Q4_0 tensor: bit0 set if some aligned pair of 32-groups has different scales, bit1 if some aligned quad does
__global__ void k_detect_gw(const block_q4_0 * __restrict__ W, int M, int nb, int * __restrict__ flags) {
    const int m = blockIdx.x;
    int f = 0;
    for (int b = threadIdx.x * 4; b < nb; b += blockDim.x * 4) {
        const uint16_t d0 = __half_as_ushort(W[(size_t) m * nb + b].d), d1 = __half_as_ushort(W[(size_t) m * nb + b + 1].d);
        const uint16_t d2 = __half_as_ushort(W[(size_t) m * nb + b + 2].d), d3 = __half_as_ushort(W[(size_t) m * nb + b + 3].d);
        if (d0 != d1 || d2 != d3) f |= 1;
        if (d0 != d2) f |= 2;
    }
    if (f) atomicOr(flags, f);
}

// Q4_0 -> operand-ready int4 K-slices (signed, permuted nibble order) + fixed-point u16 scale per 32-group
__global__ void k_repack(const block_q4_0 * __restrict__ W, int K, int nsb, const float * __restrict__ S, uint8_t * __restrict__ Wp, uint8_t * __restrict__ Wh, int sh) {
    const int m = blockIdx.x, sb = blockIdx.y, g = threadIdx.x;     // 8 threads: one per 32-group
    const int ft = m >> 4, r = m & 15, nb = K / 32;
    const block_q4_0 & b = W[(size_t) m * nb + sb * 8 + g];
    const float d = __half2float(b.d);
    const float sgn = d < 0.f ? -1.f : 1.f;                         // keep scales positive: a negative d flips the codes
    uint32_t w[4];
    for (int sl = 0; sl < 2; sl++)
        for (int dw = 0; dw < 2; dw++) {
            uint32_t word = 0;
            for (int s2 = 0; s2 < 8; s2++) {
                const int e = 16 * sl + P(dw * 8 + s2);              // element 0..31 inside the block
                int nib = e < 16 ? (b.qs[e] & 0xF) : (b.qs[e - 16] >> 4);
                int q = nib - 8;
                if (sgn < 0.f) q = max(-7, min(7, -q));
                word |= (uint32_t) (q & 15) << (4 * s2);
            }
            w[sl * 2 + dw] = word;
        }
    *(uint4 *) (Wp + ((size_t) ((ft * nsb + sb) * 8 + g)) * 256 + r * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    const float s = S[m] > 0.f ? S[m] : 1.f;
    ((uint16_t *) (Wh + ((size_t) ((ft * nsb + sb) * 16 + r)) * 16))[g] = (uint16_t) lrintf(fabsf(d) / s * (float) (1 << sh));
}

// ---------------------------------------------------------------------------------------------------------------- rotation
// H16 (x) H16 on a 16x16 view of s.*x via two f16 WMMAs; one wave per (row, 256-block)
__device__ __forceinline__ float8v rotate_block(const float * __restrict__ xrow, int lane, const half16v & b1, const half16v & a2) {
#ifdef INT4W_DEVICE_OK
    const int h = lane >> 4, lc = lane & 15;
    half16v a1;
    const float4 * xp = (const float4 *) (xrow + lc * 16);
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const float4 v = xp[q];
        a1[4 * q + 0] = (_Float16) (v.x * sign256(lc * 16 + 4 * q + 0)); a1[4 * q + 1] = (_Float16) (v.y * sign256(lc * 16 + 4 * q + 1));
        a1[4 * q + 2] = (_Float16) (v.z * sign256(lc * 16 + 4 * q + 2)); a1[4 * q + 3] = (_Float16) (v.w * sign256(lc * 16 + 4 * q + 3));
    }
    float8v z = {0, 0, 0, 0, 0, 0, 0, 0};
    const float8v p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, z);
    uint32_t own[4], oth[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        _Float16 lo = (_Float16) p[2 * i], hi = (_Float16) p[2 * i + 1];
        const uint32_t w = (uint32_t) __builtin_bit_cast(uint16_t, lo) | ((uint32_t) __builtin_bit_cast(uint16_t, hi) << 16);
        own[i] = w; oth[i] = __shfl_xor((int) w, 16, 32);
    }
    half16v b2;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t ev = h ? oth[i] : own[i], od = h ? own[i] : oth[i];
        b2[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (ev & 0xFFFF)); b2[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (ev >> 16));
        b2[8 + 2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (od & 0xFFFF)); b2[8 + 2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (od >> 16));
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, z);     // element i of lane (h,lc) = y[16*(2i+h) + lc] * 16
#else
    return float8v{0, 0, 0, 0, 0, 0, 0, 0};
#endif
}

__global__ __launch_bounds__(32) void k_rot_f32(const float * __restrict__ X, float * __restrict__ Y, int K) {
    const int sb = blockIdx.x, n = blockIdx.y, lane = threadIdx.x, h = lane >> 4, lc = lane & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
#pragma unroll
    for (int i = 0; i < 8; i++) Y[(size_t) n * K + sb * 256 + 16 * (2 * i + h) + lc] = w[i] * 0.0625f;
}

// fused rotate + quantize + pack: one 256-thread block per token (rotated fp16 row stays in LDS)
__global__ __launch_bounds__(256) void k_rot_pack(const float * __restrict__ X, int K, uint8_t * __restrict__ X4p, uint8_t * __restrict__ XMp, float * __restrict__ XRp, float clip) {
    extern __shared__ __align__(16) unsigned char fsm[];
    __half * xr = (__half *) fsm;
    float * amx = (float *) (fsm + (size_t) K * 2);
    const int nsb = K / 256, n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int h = lane >> 4, lc = lane & 15, tt = n >> 4, tl = n & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    for (int sb = wave; sb < nsb; sb += 8) {
        const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const __half hv = __float2half(w[i] * 0.0625f);
            xr[sb * 256 + 16 * (2 * i + h) + lc] = hv;
            mx = fmaxf(mx, fabsf(__half2float(hv)));
        }
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
        if (lane == 0) amx[sb] = mx;
    }
    __syncthreads();
    float dref = 0.f; for (int i = 0; i < nsb; i++) dref = fmaxf(dref, amx[i]); dref /= 7.f;
    for (int idx = tid; idx < nsb * 8; idx += 256) {
        const int sb = idx >> 3, g = idx & 7;
        int mm = 1;
        if (dref > 0.f) mm = max(1, min(16, (int) ceilf(16.f * clip * (amx[sb] / 7.f) / dref - 1e-4f)));
        const float dsb = dref * mm / 16.f, inv = dsb > 0.f ? 1.f / dsb : 0.f;
        int q[32];
#pragma unroll
        for (int i = 0; i < 32; i++) q[i] = max(-7, min(7, (int) roundf(__half2float(xr[sb * 256 + g * 32 + i]) * inv)));
        uint32_t wv[4];
#pragma unroll
        for (int sl = 0; sl < 2; sl++)
#pragma unroll
            for (int dw = 0; dw < 2; dw++) {
                uint32_t word = 0;
#pragma unroll
                for (int s2 = 0; s2 < 8; s2++) word |= (uint32_t) (q[16 * sl + P(dw * 8 + s2)] & 15) << (4 * s2);
                wv[sl * 2 + dw] = word;
            }
        *(uint4 *) (X4p + ((size_t) ((tt * nsb + sb) * 8 + g)) * 256 + tl * 16) = make_uint4(wv[0], wv[1], wv[2], wv[3]);
        if (g == 0) XMp[(size_t) (tt * nsb + sb) * 16 + (tl & 1) * 8 + (tl >> 1)] = (uint8_t) mm;
    }
    if (tid == 0) XRp[(size_t) tt * 16 + (tl & 1) * 8 + (tl >> 1)] = dref;
}

// ---------------------------------------------------------------------------------------------------------------- GEMM
template <int GW>
__global__ __launch_bounds__(WAVES_N * 32) void k_gemm(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ WS,
        const uint8_t * __restrict__ X4p, const uint8_t * __restrict__ XMp, const float * __restrict__ XRp, float * __restrict__ out, int M, int N, int K) {
#ifdef INT4W_DEVICE_OK
    const int lane = threadIdx.x & 31;
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int h = lane >> 4, lc = lane & 15;
    const int nsb = K / 256;
    const int ft0 = blockIdx.x * MI;
    const int tt0 = (blockIdx.y * WAVES_N + wn) * NJ;
    const __amdgpu_buffer_rsrc_t rWp = INT4W_RSRC(Wp), rWh = INT4W_RSRC(Wh), rX4 = INT4W_RSRC(X4p), rXM = INT4W_RSRC(XMp);

    int itot[MI][NJ][8];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NJ; b++)
#pragma unroll
            for (int i = 0; i < 8; i++) itot[a][b][i] = 0;

    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][4];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int4v h0 = bl128(rWh, lc * 16, (((ft0 + mi) * nsb + sb) * 16) * 16);
            scw[mi][0] = h0.x; scw[mi][1] = h0.y; scw[mi][2] = h0.z; scw[mi][3] = h0.w;
        }
        int2v mmw[NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) mmw[nj] = bl64(rXM, h * 8, ((tt0 + nj) * nsb + sb) * 16);
        int isum[MI][NJ][8];

        constexpr int NUNIT = 8 / GW;
        constexpr int UC = (NUNIT >= 2) ? 2 : 1;
#pragma unroll
        for (int u0 = 0; u0 < NUNIT; u0 += UC) {
            int4v wfc[UC][MI], xfc[UC][NJ], wfn[UC][MI], xfn[UC][NJ];
#pragma unroll
            for (int uc = 0; uc < UC; uc++) {
                const int g = (u0 + uc) * GW;
#pragma unroll
                for (int mi = 0; mi < MI; mi++) wfc[uc][mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + g) * 256);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) xfc[uc][nj] = bl128(rX4, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + g) * 256);
            }
            int8v ci[UC][MI][NJ];
#pragma unroll
            for (int q = 0; q < GW; q++) {
                if (q + 1 < GW) {
#pragma unroll
                    for (int uc = 0; uc < UC; uc++) {
                        const int g = (u0 + uc) * GW + q + 1;
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wfn[uc][mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + g) * 256);
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xfn[uc][nj] = bl128(rX4, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + g) * 256);
                    }
                }
#pragma unroll
                for (int sl = 0; sl < 2; sl++) {
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int uc = 0; uc < UC; uc++)
#pragma unroll
                        for (int mi = 0; mi < MI; mi++)
#pragma unroll
                            for (int nj = 0; nj < NJ; nj++) {
                                const int2v xa = sl ? INT4W_HI(xfc[uc][nj]) : INT4W_LO(xfc[uc][nj]), wb = sl ? INT4W_HI(wfc[uc][mi]) : INT4W_LO(wfc[uc][mi]);
                                int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
                                ci[uc][mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, xa, true, wb, (q == 0 && sl == 0) ? z : ci[uc][mi][nj], false);
                            }
                }
                if (q + 1 < GW) {
#pragma unroll
                    for (int uc = 0; uc < UC; uc++) {
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wfc[uc][mi] = wfn[uc][mi];
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xfc[uc][nj] = xfn[uc][nj];
                    }
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int uc = 0; uc < UC; uc++)
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    const int g = (u0 + uc) * GW;
                    const int scg = (int) ((scw[mi][g >> 1] >> (16 * (g & 1))) & 0xFFFF);
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                        for (int i = 0; i < 8; i++) { if (u0 == 0 && uc == 0) mul24a(isum[mi][nj][i], scg, ci[uc][mi][nj][i]); else mad24(isum[mi][nj][i], scg, ci[uc][mi][nj][i]); }
                }
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            int mv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mv[i] = (int) (((uint32_t) mmw[nj][i >> 2] >> (8 * (i & 3))) & 0xFF);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int i = 0; i < 8; i++) mad24(itot[mi][nj][i], isum[mi][nj][i], mv[i]);
        }
    }
    const float inv = 1.f / (float) ((1 << SH) * 16);
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        float dr[8];
        { const float4 a = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8), b = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8 + 4);
          dr[0] = a.x; dr[1] = a.y; dr[2] = a.z; dr[3] = a.w; dr[4] = b.x; dr[5] = b.y; dr[6] = b.z; dr[7] = b.w; }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int feat = (ft0 + mi) * 16 + lc;
            const float sf = WS[feat] * inv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int tok = (tt0 + nj) * 16 + 2 * i + h;
                if (tok < N) out[(size_t) tok * M + feat] = (float) itot[mi][nj][i] * (sf * dr[i]);
            }
        }
    }
#endif
}

// 8-bit activations: q = 16*hi + lo with hi in [-8,7] (signed int4 plane) and lo in [0,15] (unsigned int4 plane); two WMMA passes
__global__ __launch_bounds__(256) void k_rot_pack8(const float * __restrict__ X, int K, uint8_t * __restrict__ Xhp, uint8_t * __restrict__ Xlp, uint8_t * __restrict__ XMp, float * __restrict__ XRp) {
    extern __shared__ __align__(16) unsigned char fsm[];
    __half * xr = (__half *) fsm;
    float * amx = (float *) (fsm + (size_t) K * 2);
    const int nsb = K / 256, n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int h = lane >> 4, lc = lane & 15, tt = n >> 4, tl = n & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    for (int sb = wave; sb < nsb; sb += 8) {
        const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const __half hv = __float2half(w[i] * 0.0625f);
            xr[sb * 256 + 16 * (2 * i + h) + lc] = hv;
            mx = fmaxf(mx, fabsf(__half2float(hv)));
        }
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
        if (lane == 0) amx[sb] = mx;
    }
    __syncthreads();
    float dref = 0.f; for (int i = 0; i < nsb; i++) dref = fmaxf(dref, amx[i]); dref /= 127.f;
    for (int idx = tid; idx < nsb * 8; idx += 256) {
        const int sb = idx >> 3, g = idx & 7;
        int mm = 1;
        if (dref > 0.f) mm = max(1, min(16, (int) ceilf(16.f * (amx[sb] / 127.f) / dref - 1e-4f)));
        const float dsb = dref * mm / 16.f, inv = dsb > 0.f ? 1.f / dsb : 0.f;
        int q[32];
#pragma unroll
        for (int i = 0; i < 32; i++) q[i] = max(-127, min(127, (int) roundf(__half2float(xr[sb * 256 + g * 32 + i]) * inv)));
        uint32_t wh[4], wl[4];
#pragma unroll
        for (int sl = 0; sl < 2; sl++)
#pragma unroll
            for (int dw = 0; dw < 2; dw++) {
                uint32_t hw = 0, lw = 0;
#pragma unroll
                for (int s2 = 0; s2 < 8; s2++) {
                    const int v = q[16 * sl + P(dw * 8 + s2)];
                    hw |= (uint32_t) ((v >> 4) & 15) << (4 * s2);
                    lw |= (uint32_t) (v & 15) << (4 * s2);
                }
                wh[sl * 2 + dw] = hw; wl[sl * 2 + dw] = lw;
            }
        const size_t o = ((size_t) ((tt * nsb + sb) * 8 + g)) * 256 + tl * 16;
        *(uint4 *) (Xhp + o) = make_uint4(wh[0], wh[1], wh[2], wh[3]);
        *(uint4 *) (Xlp + o) = make_uint4(wl[0], wl[1], wl[2], wl[3]);
        if (g == 0) XMp[(size_t) (tt * nsb + sb) * 16 + (tl & 1) * 8 + (tl >> 1)] = (uint8_t) mm;
    }
    if (tid == 0) XRp[(size_t) tt * 16 + (tl & 1) * 8 + (tl >> 1)] = dref;
}

// one scale group (32 K-values) per unit: hi-plane WMMAs, x16, lo-plane WMMAs on the same accumulator, then the integer weight scale
__global__ __launch_bounds__(WAVES_N * 32) void k_gemm8(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ WS,
        const uint8_t * __restrict__ Xhp, const uint8_t * __restrict__ Xlp, const uint8_t * __restrict__ XMp, const float * __restrict__ XRp, float * __restrict__ out, int M, int N, int K) {
#ifdef INT4W_DEVICE_OK
    const int lane = threadIdx.x & 31;
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int h = lane >> 4, lc = lane & 15;
    const int nsb = K / 256;
    const int ft0 = blockIdx.x * MI;
    const int tt0 = (blockIdx.y * WAVES_N + wn) * NJ;
    const __amdgpu_buffer_rsrc_t rWp = INT4W_RSRC(Wp), rWh = INT4W_RSRC(Wh), rXh = INT4W_RSRC(Xhp), rXl = INT4W_RSRC(Xlp), rXM = INT4W_RSRC(XMp);

    float tot[MI][NJ][8];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NJ; b++)
#pragma unroll
            for (int i = 0; i < 8; i++) tot[a][b][i] = 0.f;

    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][4];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int4v h0 = bl128(rWh, lc * 16, (((ft0 + mi) * nsb + sb) * 16) * 16);
            scw[mi][0] = h0.x; scw[mi][1] = h0.y; scw[mi][2] = h0.z; scw[mi][3] = h0.w;
        }
        int2v mmw[NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) mmw[nj] = bl64(rXM, h * 8, ((tt0 + nj) * nsb + sb) * 16);
        int isum[MI][NJ][8];
        int4v wfc[MI], xhc[NJ], xlc[NJ], wfn[MI], xhn[NJ], xln[NJ];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) wfc[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8) * 256);
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            xhc[nj] = bl128(rXh, lc * 16, (((tt0 + nj) * nsb + sb) * 8) * 256);
            xlc[nj] = bl128(rXl, lc * 16, (((tt0 + nj) * nsb + sb) * 8) * 256);
        }
#pragma unroll
        for (int u = 0; u < 8; u++) {
            if (u + 1 < 8) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) wfn[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u + 1) * 256);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
                    xhn[nj] = bl128(rXh, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u + 1) * 256);
                    xln[nj] = bl128(rXl, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u + 1) * 256);
                }
            }
            int8v ci[MI][NJ];
            const int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int sl = 0; sl < 2; sl++)
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) {
                        const int2v xa = sl ? INT4W_HI(xhc[nj]) : INT4W_LO(xhc[nj]), wb = sl ? INT4W_HI(wfc[mi]) : INT4W_LO(wfc[mi]);
                        ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, xa, true, wb, sl == 0 ? z : ci[mi][nj], false);
                    }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) ci[mi][nj] = ci[mi][nj] << 4;
#pragma unroll
            for (int sl = 0; sl < 2; sl++)
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int nj = 0; nj < NJ; nj++) {
                        const int2v xa = sl ? INT4W_HI(xlc[nj]) : INT4W_LO(xlc[nj]), wb = sl ? INT4W_HI(wfc[mi]) : INT4W_LO(wfc[mi]);
                        ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, xa, true, wb, ci[mi][nj], false);
                    }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int scg = (int) ((scw[mi][u >> 1] >> (16 * (u & 1))) & 0xFFFF);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int i = 0; i < 8; i++) { if (u == 0) mul24a(isum[mi][nj][i], scg, ci[mi][nj][i]); else mad24(isum[mi][nj][i], scg, ci[mi][nj][i]); }
            }
            if (u + 1 < 8) {
#pragma unroll
                for (int mi = 0; mi < MI; mi++) wfc[mi] = wfn[mi];
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) { xhc[nj] = xhn[nj]; xlc[nj] = xln[nj]; }
            }
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            float mv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mv[i] = (float) (int) (((uint32_t) mmw[nj][i >> 2] >> (8 * (i & 3))) & 0xFF);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int i = 0; i < 8; i++) tot[mi][nj][i] = fmaf((float) isum[mi][nj][i], mv[i], tot[mi][nj][i]);
        }
    }
    const float inv = 1.f / (float) ((1 << SH8) * 16);
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        float dr[8];
        { const float4 a = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8), b = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8 + 4);
          dr[0] = a.x; dr[1] = a.y; dr[2] = a.z; dr[3] = a.w; dr[4] = b.x; dr[5] = b.y; dr[6] = b.z; dr[7] = b.w; }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int feat = (ft0 + mi) * 16 + lc;
            const float sf = WS[feat] * inv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int tok = (tt0 + nj) * 16 + 2 * i + h;
                if (tok < N) out[(size_t) tok * M + feat] = tot[mi][nj][i] * (sf * dr[i]);
            }
        }
    }
#endif
}

// as k_gemm8 for weights whose scale is shared by GW (2 or 4) consecutive 32-groups: the hi pass and the lo pass each chain GW groups
// on one accumulator, so the integer scale is applied once per GW groups
template <int GW>
__global__ __launch_bounds__(WAVES_N * 32) void k_gemm8w(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ WS,
        const uint8_t * __restrict__ Xhp, const uint8_t * __restrict__ Xlp, const uint8_t * __restrict__ XMp, const float * __restrict__ XRp, float * __restrict__ out, int M, int N, int K) {
#ifdef INT4W_DEVICE_OK
    const int lane = threadIdx.x & 31;
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int h = lane >> 4, lc = lane & 15;
    const int nsb = K / 256;
    const int ft0 = blockIdx.x * MI;
    const int tt0 = (blockIdx.y * WAVES_N + wn) * NJ;
    const __amdgpu_buffer_rsrc_t rWp = INT4W_RSRC(Wp), rWh = INT4W_RSRC(Wh), rXh = INT4W_RSRC(Xhp), rXl = INT4W_RSRC(Xlp), rXM = INT4W_RSRC(XMp);

    float tot[MI][NJ][8];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NJ; b++)
#pragma unroll
            for (int i = 0; i < 8; i++) tot[a][b][i] = 0.f;

    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][4];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int4v h0 = bl128(rWh, lc * 16, (((ft0 + mi) * nsb + sb) * 16) * 16);
            scw[mi][0] = h0.x; scw[mi][1] = h0.y; scw[mi][2] = h0.z; scw[mi][3] = h0.w;
        }
        int2v mmw[NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) mmw[nj] = bl64(rXM, h * 8, ((tt0 + nj) * nsb + sb) * 16);
        int isum[MI][NJ][8];
#pragma unroll
        for (int u0 = 0; u0 < 8; u0 += GW) {
            int8v ci[MI][NJ];
            const int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                const __amdgpu_buffer_rsrc_t rX = pass ? rXl : rXh;
                int4v wc[MI], xc[NJ], wn_[MI], xn[NJ];
#pragma unroll
                for (int mi = 0; mi < MI; mi++) wc[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0) * 256);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) xc[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0) * 256);
#pragma unroll
                for (int q = 0; q < GW; q++) {
                    if (q + 1 < GW) {
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wn_[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0 + q + 1) * 256);
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xn[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0 + q + 1) * 256);
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int sl = 0; sl < 2; sl++)
#pragma unroll
                        for (int mi = 0; mi < MI; mi++)
#pragma unroll
                            for (int nj = 0; nj < NJ; nj++) {
                                const int2v xa = sl ? INT4W_HI(xc[nj]) : INT4W_LO(xc[nj]), wb = sl ? INT4W_HI(wc[mi]) : INT4W_LO(wc[mi]);
                                const int8v cin = (q == 0 && sl == 0 && pass == 0) ? z : ci[mi][nj];
                                if (pass == 0) ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, xa, true, wb, cin, false);
                                else           ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, xa, true, wb, cin, false);
                            }
                    if (q + 1 < GW) {
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wc[mi] = wn_[mi];
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xc[nj] = xn[nj];
                    }
                }
                if (pass == 0) {
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) ci[mi][nj] = ci[mi][nj] << 4;
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int scg = (int) ((scw[mi][u0 >> 1] >> (16 * (u0 & 1))) & 0xFFFF);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int i = 0; i < 8; i++) { if (u0 == 0) mul24a(isum[mi][nj][i], scg, ci[mi][nj][i]); else mad24(isum[mi][nj][i], scg, ci[mi][nj][i]); }
            }
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            float mv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mv[i] = (float) (int) (((uint32_t) mmw[nj][i >> 2] >> (8 * (i & 3))) & 0xFF);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int i = 0; i < 8; i++) tot[mi][nj][i] = fmaf((float) isum[mi][nj][i], mv[i], tot[mi][nj][i]);
        }
    }
    const float inv = 1.f / (float) ((1 << SH8) * 16);
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        float dr[8];
        { const float4 a = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8), b = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8 + 4);
          dr[0] = a.x; dr[1] = a.y; dr[2] = a.z; dr[3] = a.w; dr[4] = b.x; dr[5] = b.y; dr[6] = b.z; dr[7] = b.w; }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int feat = (ft0 + mi) * 16 + lc;
            const float sf = WS[feat] * inv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int tok = (tt0 + nj) * 16 + 2 * i + h;
                if (tok < N) out[(size_t) tok * M + feat] = tot[mi][nj][i] * (sf * dr[i]);
            }
        }
    }
#endif
}

// rotate + Q8_1 quantize in one pass: one wave per (row, 256-block); the 8 rows pairs of the 16x16 view are exactly the 8 Q8_1 blocks
__global__ __launch_bounds__(32) void k_rot_q8_1(const float * __restrict__ X, block_q8_1 * __restrict__ Y, int K) {
    const int sb = blockIdx.x, n = blockIdx.y, lane = threadIdx.x, h = lane >> 4, lc = lane & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float x = w[i] * 0.0625f;
        float amax = fabsf(x), sum = x;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor(amax, o, 32)); sum += __shfl_xor(sum, o, 32); }
        const float d = amax / 127.0f;
        block_q8_1 & y = Y[(size_t) n * (K / 32) + sb * 8 + i];
        y.qs[16 * h + lc] = amax == 0.0f ? 0 : (int8_t) roundf(x / d);
        if (lane == 0) y.ds = make_half2(d, sum);
    }
}

// rotate + 8-bit quantize for the decode matvec: int8 codes [n][K] and per 32-group (d, d*sum(q)) so the matvec can use the exact code sum
__global__ __launch_bounds__(32) void k_rot_q8x(const float * __restrict__ X, int8_t * __restrict__ Q, float2 * __restrict__ D, int K) {
    const int sb = blockIdx.x, n = blockIdx.y, lane = threadIdx.x, h = lane >> 4, lc = lane & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float x = w[i] * 0.0625f;
        float amax = fabsf(x);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        const float d = amax / 127.0f;
        const int q = amax == 0.0f ? 0 : (int) roundf(x / d);
        float sq = (float) q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor(sq, o, 32);
        Q[(size_t) n * K + sb * 256 + i * 32 + 16 * h + lc] = (int8_t) q;
        if (lane == 0) D[(size_t) n * (K / 32) + sb * 8 + i] = make_float2(d, d * sq);
    }
}

// decode matvec on the repacked layout: one block = 16 weight rows (a feature tile) x NW waves splitting the 256-blocks of K.
// lane = (row r = lane & 15, half h = lane >> 4); half h owns groups 4h..4h+3 of every 256-block.  A repacked 32-group is 16 bytes per row:
// word (sl*2+dw) holds elements 16*sl+8*dw+{0..3} in its low nibbles and +{4..7} in its high nibbles, so a dot4 against 4 consecutive
// activation bytes needs no shuffling.  Codes are two's complement nibbles; ^8 turns them into q+8 and the sum of the activation codes
// (ds.y) removes the offset, as in ggml's Q4_0 x Q8_1 dot.
constexpr int MV_NW = 4;
__device__ __forceinline__ int sdot4(int a, int b, int c) { return ggml_cuda_dp4a(a, b, c); }

template <int NC, bool GATE>
__global__ __launch_bounds__(32 * MV_NW) void k_mv(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ S, float inv,
        const uint8_t * __restrict__ Wp2, const uint8_t * __restrict__ Wh2, const float * __restrict__ S2, float inv2,
        const int8_t * __restrict__ Xq, const float2 * __restrict__ Xd, float * __restrict__ dst, const float * __restrict__ xb, const float * __restrict__ gb,
        int glu, int M, int K, int stride_dst) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y, ft = blockIdx.x, nsb = K / 256;
    float acc[NC], accg[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) { acc[j] = 0.f; accg[j] = 0.f; }
    for (int sb = wv; sb < nsb; sb += MV_NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 16 + h * 8;
        uint4 w[4], wg[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = *(const uint4 *) (Wp + wo + g * 256);
        const uint2 sc = *(const uint2 *) (Wh + so);
        uint2 scg = make_uint2(0, 0);
        if (GATE) {
#pragma unroll
            for (int g = 0; g < 4; g++) wg[g] = *(const uint4 *) (Wp2 + wo + g * 256);
            scg = *(const uint2 *) (Wh2 + so);
        }
#pragma unroll
        for (int j = 0; j < NC; j++) {
            const uint4 * xq = (const uint4 *) (Xq + (size_t) j * K + sb * 256 + h * 128);
            const float4 * xd = (const float4 *) (Xd + (size_t) j * (K / 32) + sb * 8 + h * 4);
            uint4 xv[8]; float4 dv[2];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = xq[i];
            dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
                const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                const uint32_t sw = g < 2 ? sc.x : sc.y;
                acc[j] += (float) ((sw >> (16 * (g & 1))) & 0xFFFF) * (ds.x * (float) si - 8.f * ds.y);
                if (GATE) {
                    const uint32_t wwg[4] = {wg[g].x, wg[g].y, wg[g].z, wg[g].w};
                    int sg = 0;
#pragma unroll
                    for (int wi = 0; wi < 4; wi++) {
                        sg = sdot4((int) ((wwg[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], sg);
                        sg = sdot4((int) (((wwg[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], sg);
                    }
                    const uint32_t swg = g < 2 ? scg.x : scg.y;
                    accg[j] += (float) ((swg >> (16 * (g & 1))) & 0xFFFF) * (ds.x * (float) sg - 8.f * ds.y);
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) { acc[j] += __shfl_xor(acc[j], 16, 32); accg[j] += __shfl_xor(accg[j], 16, 32); }
    __shared__ float red[MV_NW][NC][2][16];
    if (h == 0) {
#pragma unroll
        for (int j = 0; j < NC; j++) { red[wv][j][0][r] = acc[j]; red[wv][j][1][r] = accg[j]; }
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
        const int row = ft * 16 + r;
        const float sf = S[row] * inv, sf2 = GATE ? S2[row] * inv2 : 0.f;
#pragma unroll
        for (int j = 0; j < NC; j++) {
            float a = 0.f, g = 0.f;
#pragma unroll
            for (int q = 0; q < MV_NW; q++) { a += red[q][j][0][r]; g += red[q][j][1][r]; }
            float res = a * sf;
            if (xb) res += xb[(size_t) j * stride_dst + row];
            if (GATE) {
                float gv = g * sf2;
                if (gb) gv += gb[(size_t) j * stride_dst + row];
                switch (glu) {
                    case GGML_GLU_OP_SWIGLU: res *= ggml_cuda_op_silu_single(gv); break;
                    case GGML_GLU_OP_GEGLU:  res *= ggml_cuda_op_gelu_single(gv); break;
                    default:                 res *= gv; break;
                }
            }
            dst[(size_t) j * stride_dst + row] = res;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------- host glue
int int4_mode() {
    static const int mode = [] { const char * e = getenv("GGML_INT4"); return e ? atoi(e) : 0; }();
    return mode;
}
int int4_gw() {
    static const int gw = [] { const char * e = getenv("GGML_INT4_GW"); int v = e ? atoi(e) : 4; return (v == 1 || v == 2 || v == 4) ? v : 4; }();
    return gw;
}
// per-tensor choice of the activation width in mode 3: GGML_INT4_A4=ffn_gate,ffn_up,... (substrings of the tensor name) selects 4-bit activations
bool int4_use_a8(const ggml_tensor * src0) {
    if (int4_mode() < 3) return false;
    static const std::vector<std::string> a4 = [] {
        std::vector<std::string> v; const char * e = getenv("GGML_INT4_A4");
        if (!e) {      // presets: GGML_INT4_PREFILL=quality (default, 8-bit activations everywhere) | balanced | fast (4-bit activations everywhere)
            const char * p = getenv("GGML_INT4_PREFILL");
            if (p && !strcmp(p, "fast")) e = "attn_,ffn_,ssm_";
            else if (p && !strcmp(p, "balanced")) e = "attn_q.weight,attn_k,attn_v,attn_qkv,attn_gate,ffn_gate,ffn_up";
        }
        if (e) { std::string s(e), cur; for (char c : s) { if (c == ',') { if (!cur.empty()) v.push_back(cur); cur.clear(); } else cur += c; } if (!cur.empty()) v.push_back(cur); }
        return v;
    }();
    for (const auto & p : a4) { if (strstr(src0->name, p.c_str())) return false; }
    return true;
}
float int4_clip() {
    static const float c = [] { const char * e = getenv("GGML_INT4_CLIP"); return e ? (float) atof(e) : 0.85f; }();
    return c;
}
int int4_min_n() {
    static const int n = [] { const char * e = getenv("GGML_INT4_MIN_N"); return e ? atoi(e) : 64; }();
    return n;
}

struct Int4Weights { uint8_t * Wp = nullptr; uint8_t * Wh = nullptr; float * S = nullptr; int gw = 1; bool inplace = false; int sh = SH; };
std::mutex g_mu;
std::unordered_map<const void *, Int4Weights> g_w;

// GGML_INT4_INPLACE=0 keeps the old layout (Q4_0 original + a repacked copy, ~2x weight VRAM) for A/B testing
bool int4_inplace() {
    static const bool v = [] { const char * e = getenv("GGML_INT4_INPLACE"); return !e || atoi(e) != 0; }();
    return v;
}

bool inplace_eligible(const ggml_tensor * src0) {
    return int4_mode() >= 2 && int4_inplace() && src0->type == GGML_TYPE_Q4_0 && (src0->ne[0] % 256) == 0 && (src0->ne[1] % TILE_M) == 0 &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && !src0->view_src && ggml_is_contiguous(src0) && strncmp(src0->name, "blk.", 4) == 0;
}

Int4Weights get_weights(cudaStream_t stream, const ggml_tensor * src0) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_w.find(src0->data);
    if (it != g_w.end()) return it->second;
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], nsb = K / 256;
    Int4Weights w;
    const bool ip = inplace_eligible(src0);
    const int sh = int4_use_a8(src0) ? SH8 : SH;
    uint8_t * src = (uint8_t *) src0->data;
    uint8_t * tmp = nullptr;
    if (ip) {
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        CUDA_CHECK(hipStreamIsCapturing(stream, &cs));
        GGML_ASSERT(cs == hipStreamCaptureStatusNone && "int4: in-place weight conversion must happen before graph capture");
        // the repack reads the original blocks while writing the new layout over them: work from a temporary copy (one tensor at a time)
        CUDA_CHECK(cudaMalloc((void **) &tmp, ggml_nbytes(src0)));
        CUDA_CHECK(cudaMemcpyAsync(tmp, src, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, stream));
        src = tmp;
        w.Wp = (uint8_t *) src0->data;
        w.Wh = (uint8_t *) src0->data + (size_t) M * K / 2;     // M*K/2 + M*nsb*16 == M*K/32*18 == ggml_nbytes(src0)
    } else {
        CUDA_CHECK(cudaMalloc((void **) &w.Wp, (size_t) M * K / 2));
        CUDA_CHECK(cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 16));
    }
    CUDA_CHECK(cudaMalloc((void **) &w.S, (size_t) M * sizeof(float)));
    {
        int * flags; CUDA_CHECK(cudaMalloc((void **) &flags, sizeof(int))); CUDA_CHECK(cudaMemsetAsync(flags, 0, sizeof(int), stream));
        k_detect_gw<<<M, 64, 0, stream>>>((const block_q4_0 *) src, M, K / 32, flags);
        int f = 0; CUDA_CHECK(cudaMemcpyAsync(&f, flags, sizeof(int), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(flags));
        if (f & 1) w.gw = 1; else if (f & 2) w.gw = 2; else w.gw = 4;
    }
    k_row_scale<<<(M + 127) / 128, 128, 0, stream>>>((const block_q4_0 *) src, M, K / 32, w.S);
    k_repack<<<dim3(M, nsb), 8, 0, stream>>>((const block_q4_0 *) src, K, nsb, w.S, w.Wp, w.Wh, sh);
    CUDA_CHECK(cudaGetLastError());
    if (ip) { CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaFree(tmp)); }
    w.inplace = ip; w.sh = sh;
    g_w.emplace(src0->data, w);
    return w;
}

bool contiguous_rows(const ggml_tensor * t) { return t->nb[0] == ggml_type_size(t->type) && t->nb[1] == t->ne[0] * ggml_type_size(t->type); }

} // namespace

bool ggml_cuda_int4_rotated(const ggml_tensor * src0) {
    return int4_mode() > 0 && src0->type == GGML_TYPE_Q4_0 && (src0->ne[0] % 256) == 0 && strncmp(src0->name, "blk.", 4) == 0;
}

void ggml_cuda_int4_rotate(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && contiguous_rows(src1) && (src1->ne[0] % 256) == 0);
    const int K = (int) src1->ne[0];
    const int64_t rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    GGML_ASSERT(ggml_is_contiguous(src1));
    k_rot_f32<<<dim3(K / 256, (unsigned) rows), 32, 0, ctx.stream()>>>((const float *) src1->data, dst, K);
    CUDA_CHECK(cudaGetLastError());
}


bool ggml_cuda_int4_fused_ok(const ggml_tensor * src1) {
    return src1->type == GGML_TYPE_F32 && src1->ne[2] == 1 && src1->ne[3] == 1 && (src1->ne[0] % 512) == 0 && contiguous_rows(src1) && ggml_is_contiguous(src1);
}

void ggml_cuda_int4_rot_q8_1(const ggml_tensor * src1, void * q8_1, cudaStream_t stream) {
    const int K = (int) src1->ne[0];
    k_rot_q8_1<<<dim3(K / 256, (unsigned) src1->ne[1]), 32, 0, stream>>>((const float *) src1->data, (block_q8_1 *) q8_1, K);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (int4_mode() < 2) return false;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) return false;
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], N = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);    // batch dims are just more token rows
    if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != K || !ggml_is_contiguous(src1)) return false;
    const bool ip = inplace_eligible(src0);          // converted weights are not Q4_0 any more: no fallback exists
    if ((K % 256) != 0 || (M % TILE_M) != 0 || (N < int4_min_n() && !ip)) return false;
    if (!contiguous_rows(src1) || !ggml_is_contiguous(dst) || dst->ne[0] != M || (int) ggml_nelements(dst) != N * M) return false;

    cudaStream_t stream = ctx.stream();
    const Int4Weights w = get_weights(stream, src0);
    const int nsb = K / 256, Npad = (N + TILE_N - 1) / TILE_N * TILE_N;
    const bool a8 = int4_use_a8(src0);
    const size_t plane = (size_t) Npad * K / 2;
    ggml_cuda_pool_alloc<uint8_t> x4(ctx.pool(), a8 ? 2 * plane : plane);
    ggml_cuda_pool_alloc<uint8_t> xm(ctx.pool(), (size_t) (Npad / 16) * nsb * 16);
    ggml_cuda_pool_alloc<float>   xr(ctx.pool(), (size_t) Npad);
    CUDA_CHECK(cudaMemsetAsync(x4.get(), 0, a8 ? 2 * plane : plane, stream));
    CUDA_CHECK(cudaMemsetAsync(xm.get(), 0, (size_t) (Npad / 16) * nsb * 16, stream));
    CUDA_CHECK(cudaMemsetAsync(xr.get(), 0, (size_t) Npad * sizeof(float), stream));
    const size_t smem = (size_t) K * 2 + (size_t) nsb * 4;
    const dim3 grid(M / TILE_M, Npad / TILE_N);
    if (a8) {
        k_rot_pack8<<<N, 256, smem, stream>>>((const float *) src1->data, K, x4.get(), x4.get() + plane, xm.get(), xr.get());
        switch (w.gw) {
            case 2:  k_gemm8w<2><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), x4.get() + plane, xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
            case 4:  k_gemm8w<4><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), x4.get() + plane, xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
            default: k_gemm8<<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), x4.get() + plane, xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
        }
    } else {
        k_rot_pack<<<N, 256, smem, stream>>>((const float *) src1->data, K, x4.get(), xm.get(), xr.get(), int4_clip());
        switch (w.gw) {
            case 1:  k_gemm<1><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
            case 2:  k_gemm<2><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
            default: k_gemm<4><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}


bool ggml_cuda_int4_converted(const ggml_tensor * src0) {
    if (src0->type != GGML_TYPE_Q4_0 || !int4_inplace()) return false;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_w.find(src0->data);
    return it != g_w.end() && it->second.inplace;
}

void ggml_cuda_int4_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (int4_mode() < 2 || !int4_inplace()) return;
    if (!GGML_CUDA_CC_IS_RDNA3_0(ggml_cuda_info().devices[ctx.device].cc)) return;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_MUL_MAT) continue;
        const ggml_tensor * w = n->src[0];
        if (!ggml_cuda_int4_rotated(w) || !inplace_eligible(w)) continue;
        get_weights(ctx.stream(), w);          // converts on first sight
    }
}

void ggml_cuda_int4_forget(const void * ptr, size_t size) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto it = g_w.begin(); it != g_w.end();) {
        if (it->first >= ptr && (const char *) it->first < (const char *) ptr + size) {
            (void) cudaFree(it->second.S);
            if (!it->second.inplace) { (void) cudaFree(it->second.Wp); (void) cudaFree(it->second.Wh); }   // in-place: the repacked data lives inside the dying weight buffer
            it = g_w.erase(it);
        } else ++it;
    }
}

void ggml_cuda_int4_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], N = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    GGML_ASSERT(N >= 1 && N <= 8 && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[0] == K);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == M && ggml_is_contiguous(dst));
    cudaStream_t stream = ctx.stream();
    const Int4Weights w = get_weights(stream, src0);
    GGML_ASSERT(w.inplace);
    Int4Weights g{};
    const float * xb = nullptr, * gb = nullptr;
    int glu = 0;
    bool gate = false;
    if (fusion) {
        GGML_ASSERT(fusion->x_scale == nullptr && fusion->gate_scale == nullptr);
        if (fusion->x_bias) xb = (const float *) fusion->x_bias->data;
        if (fusion->gate_bias) gb = (const float *) fusion->gate_bias->data;
        if (fusion->gate) { g = get_weights(stream, fusion->gate); GGML_ASSERT(g.inplace && fusion->gate->ne[1] == M && fusion->gate->ne[0] == K); gate = true; glu = (int) fusion->glu_op; }
    }
    ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), (size_t) N * K);
    ggml_cuda_pool_alloc<float2> xd(ctx.pool(), (size_t) N * (K / 32));
    k_rot_q8x<<<dim3(K / 256, N), 32, 0, stream>>>((const float *) src1->data, xq.get(), xd.get(), K);
    const float inv = 1.f / (float) (1 << w.sh), inv2 = gate ? 1.f / (float) (1 << g.sh) : 0.f;
    const int stride_dst = M;
    const dim3 grid(M / 16), blk(32, MV_NW);
#define MV_CASE(NCV) \
    case NCV: if (gate) k_mv<NCV, true><<<grid, blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, g.Wp, g.Wh, g.S, inv2, xq.get(), xd.get(), (float *) dst->data, xb, gb, glu, M, K, stride_dst); \
              else      k_mv<NCV, false><<<grid, blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, nullptr, nullptr, nullptr, 0.f, xq.get(), xd.get(), (float *) dst->data, xb, gb, glu, M, K, stride_dst); break;
    switch (N) { MV_CASE(1) MV_CASE(2) MV_CASE(3) MV_CASE(4) MV_CASE(5) MV_CASE(6) MV_CASE(7) MV_CASE(8) }
#undef MV_CASE
    CUDA_CHECK(cudaGetLastError());
}

#else  // not a HIP build: the feature is unavailable

bool ggml_cuda_int4_rotated(const ggml_tensor *) { return false; }
void ggml_cuda_int4_rotate(ggml_backend_cuda_context &, const ggml_tensor *, float *) {}
bool ggml_cuda_int4_fused_ok(const ggml_tensor *) { return false; }
void ggml_cuda_int4_rot_q8_1(const ggml_tensor *, void *, cudaStream_t) {}
bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_int4_converted(const ggml_tensor *) { return false; }
void ggml_cuda_int4_prepare(ggml_backend_cuda_context &, const ggml_cgraph *) {}
void ggml_cuda_int4_matvec(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {}
void ggml_cuda_int4_forget(const void *, size_t) {}

#endif
