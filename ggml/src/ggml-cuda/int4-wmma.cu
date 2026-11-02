// INT4 activations x symmetric-INT4 weights on RDNA3 WMMA (v_wmma_i32_16x16x16_iu4), opt-in and experimental.
//
//   GGML_INT4=1   Q4_0 "blk.*" weights are stored rotated (see scripts/rotate_quantize.py): every matmul with them first rotates the
//                 activations, x' = (H D) x per 256-block, then runs the regular path.  (correctness / baseline)
//   GGML_INT4=2   additionally, prefill-sized matmuls (>= GGML_INT4_MIN_N tokens) run the INT4 WMMA kernel below: rotate+quantize the
//                 activations to int4 (one scale per token per 256-block, expressed as a 4-bit multiplier of a per-token reference),
//                 then an int4 x int4 WMMA GEMM with integer sub-scales.
//   GGML_INT4_GW  K-groups of 32 that share one weight scale (1, 2 or 4; default 4) - must match how the model was quantized.
//
// The rotation follows Meta's HadaCore idea (H256 = H16 (x) H16 as two 16x16 matmuls); this is an independent RDNA3 implementation.
#include "int4-wmma.cuh"

#if defined(GGML_USE_HIP)

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if defined(__GFX11__) || !defined(__HIP_DEVICE_COMPILE__)
#define INT4W_DEVICE_OK 1
#endif

namespace {

typedef int   int8v  __attribute__((ext_vector_type(8)));
typedef int   int4v  __attribute__((ext_vector_type(4)));
typedef int   int2v  __attribute__((ext_vector_type(2)));
typedef float float8v __attribute__((ext_vector_type(8)));
typedef _Float16 half16v __attribute__((ext_vector_type(16)));

constexpr int SH = 10;                 // fixed-point bits of the integer sub-scales
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
// Q4_0 -> operand-ready int4 K-slices (signed, permuted nibble order) + fixed-point u16 scale per 32-group
__global__ void k_repack(const block_q4_0 * __restrict__ W, int K, int nsb, const float * __restrict__ S, uint8_t * __restrict__ Wp, uint8_t * __restrict__ Wh) {
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
    ((uint16_t *) (Wh + ((size_t) ((ft * nsb + sb) * 16 + r)) * 16))[g] = (uint16_t) lrintf(fabsf(d) / s * (float) (1 << SH));
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
__global__ __launch_bounds__(256) void k_rot_pack(const float * __restrict__ X, int K, uint8_t * __restrict__ X4p, uint8_t * __restrict__ XMp, float * __restrict__ XRp) {
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
        if (dref > 0.f) mm = max(1, min(16, (int) ceilf(16.f * (amx[sb] / 7.f) / dref - 1e-4f)));
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

// ---------------------------------------------------------------------------------------------------------------- host glue
int int4_mode() {
    static const int mode = [] { const char * e = getenv("GGML_INT4"); return e ? atoi(e) : 0; }();
    return mode;
}
int int4_gw() {
    static const int gw = [] { const char * e = getenv("GGML_INT4_GW"); int v = e ? atoi(e) : 4; return (v == 1 || v == 2 || v == 4) ? v : 4; }();
    return gw;
}
int int4_min_n() {
    static const int n = [] { const char * e = getenv("GGML_INT4_MIN_N"); return e ? atoi(e) : 64; }();
    return n;
}

// the fused/graph paths bypass ggml_cuda_mul_mat and would see rotated weights with unrotated activations
__attribute__((constructor)) void int4_env_init() {
    if (getenv("GGML_INT4") && atoi(getenv("GGML_INT4")) > 0) {
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    }
}

struct Int4Weights { uint8_t * Wp = nullptr; uint8_t * Wh = nullptr; float * S = nullptr; };
std::mutex g_mu;
std::unordered_map<const void *, Int4Weights> g_w;

Int4Weights get_weights(cudaStream_t stream, const ggml_tensor * src0) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_w.find(src0->data);
    if (it != g_w.end()) return it->second;
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], nsb = K / 256;
    Int4Weights w;
    CUDA_CHECK(cudaMalloc((void **) &w.Wp, (size_t) M * K / 2));
    CUDA_CHECK(cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 16));
    CUDA_CHECK(cudaMalloc((void **) &w.S, (size_t) M * sizeof(float)));
    k_row_scale<<<(M + 127) / 128, 128, 0, stream>>>((const block_q4_0 *) src0->data, M, K / 32, w.S);
    k_repack<<<dim3(M, nsb), 8, 0, stream>>>((const block_q4_0 *) src0->data, K, nsb, w.S, w.Wp, w.Wh);
    CUDA_CHECK(cudaGetLastError());
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

bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (int4_mode() < 2) return false;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) return false;
    const int K = (int) src0->ne[0], M = (int) src0->ne[1], N = (int) src1->ne[1];
    if (src0->type != GGML_TYPE_Q4_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[0] != K) return false;
    if ((K % 256) != 0 || (M % TILE_M) != 0 || N < int4_min_n()) return false;
    if (!contiguous_rows(src1) || !ggml_is_contiguous(dst) || dst->ne[0] != M) return false;

    cudaStream_t stream = ctx.stream();
    const Int4Weights w = get_weights(stream, src0);
    const int nsb = K / 256, Npad = (N + TILE_N - 1) / TILE_N * TILE_N;
    ggml_cuda_pool_alloc<uint8_t> x4(ctx.pool(), (size_t) Npad * K / 2);
    ggml_cuda_pool_alloc<uint8_t> xm(ctx.pool(), (size_t) (Npad / 16) * nsb * 16);
    ggml_cuda_pool_alloc<float>   xr(ctx.pool(), (size_t) Npad);
    CUDA_CHECK(cudaMemsetAsync(x4.get(), 0, (size_t) Npad * K / 2, stream));
    CUDA_CHECK(cudaMemsetAsync(xm.get(), 0, (size_t) (Npad / 16) * nsb * 16, stream));
    CUDA_CHECK(cudaMemsetAsync(xr.get(), 0, (size_t) Npad * sizeof(float), stream));
    const size_t smem = (size_t) K * 2 + (size_t) nsb * 4;
    k_rot_pack<<<N, 256, smem, stream>>>((const float *) src1->data, K, x4.get(), xm.get(), xr.get());
    const dim3 grid(M / TILE_M, Npad / TILE_N);
    switch (int4_gw()) {
        case 1:  k_gemm<1><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
        case 2:  k_gemm<2><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
        default: k_gemm<4><<<grid, WAVES_N * 32, 0, stream>>>(w.Wp, w.Wh, w.S, x4.get(), xm.get(), xr.get(), (float *) dst->data, M, N, K); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else  // not a HIP build: the feature is unavailable

bool ggml_cuda_int4_rotated(const ggml_tensor *) { return false; }
void ggml_cuda_int4_rotate(ggml_backend_cuda_context &, const ggml_tensor *, float *) {}
bool ggml_cuda_int4_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }

#endif
