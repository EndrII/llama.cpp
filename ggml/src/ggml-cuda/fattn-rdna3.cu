// Flash attention for head size 256 on RDNA3 WMMA (v_wmma_f32_16x16x16_f16), prompt processing only.
//
// One 4-wave block owns 64 query rows = (64 / gqa) tokens x gqa heads that share one K/V head, so every K/V tile read from memory
// serves all gqa heads.  Each wave owns 16 rows.  Per tile of 32 K/V positions:
//   S  = Q K^T          (Q f16 in LDS, K tile in LDS, f32 accumulators)
//   online softmax      (row statistics live in the lanes that hold the row's accumulator elements)
//   O += P V            (P goes through a small per-wave LDS buffer to become an A operand, V is stored transposed in LDS)
// K/V tiles that the mask hides completely (causal prefix of later tokens, sliding windows) are skipped.
// Supported: Q f32, K/V f16, DKQ == DV == 256, f16 mask (or none), no ALiBi / softcap / sinks, K/V length % 32 == 0.  Anything else
// returns false and goes through the regular kernels.
#include "fattn-rdna3.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP)

#if defined(__GFX11__) || !defined(__HIP_DEVICE_COMPILE__)
#define FA3_OK 1
#endif

typedef float    fa3_f8  __attribute__((ext_vector_type(8)));
typedef _Float16 fa3_h16 __attribute__((ext_vector_type(16)));

namespace {

constexpr int FA_D  = 256;
constexpr int FA_R  = 64;               // query rows per block
constexpr int FA_KT = 32;               // K/V positions per tile
constexpr int FA_DP = FA_D + 8;         // LDS row stride (halves) of Q and K tiles
constexpr int FA_VP = FA_KT + 8;        // LDS row stride (halves) of the transposed V tile
constexpr int FA_PP = FA_KT + 8;        // per-wave P buffer row stride

__device__ __forceinline__ fa3_f8 fa3_wmma(const fa3_h16 & a, const fa3_h16 & b, const fa3_f8 & c) {
#ifdef FA3_OK
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    return c;
#endif
}

__device__ __forceinline__ fa3_h16 fa3_ld16(const _Float16 * p) {
    const uint4 * u = (const uint4 *) p;
    struct { uint4 x, y; } t = { u[0], u[1] };
    return __builtin_bit_cast(fa3_h16, t);
}

struct fa3_params {
    const char * q;  const char * k;  const char * v;  const char * mask;  char * dst;
    int64_t n_tokens, n_kv, n_head, n_head_kv, gqa;
    size_t q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3, v_nb1, v_nb2, v_nb3, m_nb1, m_nb3, d_nb1, d_nb2, d_nb3;
    int64_t m_ne3;
    float scale;
};

__global__ void __launch_bounds__(128, 1) fa3_kernel(const fa3_params p) {
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, hh = lane >> 4, lc = lane & 15;
    const int gqa = (int) p.gqa;
    const int tok_per_blk = FA_R / gqa;
    const int64_t tok0 = (int64_t) blockIdx.x * tok_per_blk;
    const int g = blockIdx.y, seq = blockIdx.z;                 // kv head, sequence

    extern __shared__ __align__(16) unsigned char fa3_smem[];
    _Float16 * qs = (_Float16 *) fa3_smem;                      // [64][FA_DP]
    _Float16 * kv = qs + FA_R * FA_DP;                          // K tile [32][FA_DP], later V^T [256][FA_VP]
    _Float16 * pb = kv + (FA_D * FA_VP > FA_KT * FA_DP ? FA_D * FA_VP : FA_KT * FA_DP);   // [4][16][FA_PP]
    int * kv_range = (int *) (pb + 4 * 16 * FA_PP);             // [2]

    // ---- Q tile: f32 -> f16, row r = (token, head) with r = token_in_block * gqa + head_in_group
    for (int idx = tid; idx < FA_R * (FA_D / 4); idx += 128) {
        const int r = idx / (FA_D / 4), c4 = idx % (FA_D / 4);
        const int64_t tk = tok0 + r / gqa;
        const int head = g * gqa + r % gqa;
        float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tk < p.n_tokens) { f = *(const float4 *) (p.q + seq * p.q_nb3 + head * p.q_nb2 + tk * p.q_nb1 + c4 * 16); }
        _Float16 * d = qs + r * FA_DP + c4 * 4;
        d[0] = (_Float16) f.x; d[1] = (_Float16) f.y; d[2] = (_Float16) f.z; d[3] = (_Float16) f.w;
    }

    // ---- visible K/V range of this block from the mask
    if (tid == 0) { kv_range[0] = 1 << 30; kv_range[1] = -1; }
    __syncthreads();
    const int n_tiles_all = (int) (p.n_kv / FA_KT);
    if (p.mask) {
        const int ntok = (int) min((int64_t) tok_per_blk, p.n_tokens - tok0);
        const int per_row = (int) (p.n_kv / 8);
        const char * mb = p.mask + (seq % p.m_ne3) * p.m_nb3;
        int lo = 1 << 30, hi = -1;
        for (int idx = tid; idx < ntok * per_row; idx += 128) {
            const int r = idx / per_row, c8 = idx % per_row;
            const uint4 u = *(const uint4 *) (mb + (tok0 + r) * p.m_nb1 + (size_t) c8 * 16);
            const uint32_t ws[4] = { u.x, u.y, u.z, u.w };
            bool vis = false;
#pragma unroll
            for (int i = 0; i < 4; i++) { vis |= ((ws[i] & 0xFFFF) != 0xFC00) || ((ws[i] >> 16) != 0xFC00); }
            if (vis) { const int t = (c8 * 8) / FA_KT; lo = min(lo, t); hi = max(hi, t); }
        }
        for (int o = 16; o > 0; o >>= 1) { lo = min(lo, __shfl_xor(lo, o, 32)); hi = max(hi, __shfl_xor(hi, o, 32)); }
        if (lane == 0) { atomicMin(&kv_range[0], lo); atomicMax(&kv_range[1], hi); }
        __syncthreads();
    } else if (tid == 0) { kv_range[0] = 0; kv_range[1] = n_tiles_all - 1; }
    __syncthreads();
    const int t_lo = kv_range[0], t_hi = kv_range[1];

    fa3_f8 o[16];
#pragma unroll
    for (int j = 0; j < 16; j++) { o[j] = fa3_f8{0, 0, 0, 0, 0, 0, 0, 0}; }
    float m_run[8], l_run[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { m_run[i] = -INFINITY; l_run[i] = 0.f; }

    const char * kb = p.k + seq * p.k_nb3 + (size_t) g * p.k_nb2;
    const char * vb = p.v + seq * p.v_nb3 + (size_t) g * p.v_nb2;
    const char * mb = p.mask ? p.mask + (seq % p.m_ne3) * p.m_nb3 : nullptr;
    _Float16 * pw = pb + w * 16 * FA_PP;

    uint4 kr[8], vr[8];
    _Float16 mr[8][2];
    auto load_k = [&](int64_t kv0) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = tid + 128 * e, r = idx >> 5, c8 = idx & 31;
            kr[e] = *(const uint4 *) (kb + (kv0 + r) * p.k_nb1 + c8 * 16);
        }
    };
    if (t_lo <= t_hi) { load_k((int64_t) t_lo * FA_KT); }

    for (int t = t_lo; t <= t_hi; t++) {
        const int64_t kv0 = (int64_t) t * FA_KT;
        __syncthreads();                                    // previous tile's P V has finished reading the tile buffer
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = tid + 128 * e, r = idx >> 5, c8 = idx & 31;
            *(uint4 *) (kv + r * FA_DP + c8 * 8) = kr[e];
        }
        // this tile's V rows (lane = K/V position, wave = quarter of the head dim) and mask values, in flight during S
#pragma unroll
        for (int q = 0; q < 8; q++) { vr[q] = *(const uint4 *) (vb + (kv0 + lane) * p.v_nb1 + (64 * w + 8 * q) * 2); }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int row = 16 * w + 2 * i + hh;
            const int64_t tk = min(tok0 + row / gqa, p.n_tokens - 1);
            if (mb) {
                const _Float16 * mrow = (const _Float16 *) (mb + tk * p.m_nb1) + kv0;
                mr[i][0] = mrow[lc]; mr[i][1] = mrow[16 + lc];
            } else { mr[i][0] = (_Float16) 0.f; mr[i][1] = (_Float16) 0.f; }
        }
        __syncthreads();

        // ---- S = Q K^T
        fa3_f8 s[2] = { fa3_f8{0, 0, 0, 0, 0, 0, 0, 0}, fa3_f8{0, 0, 0, 0, 0, 0, 0, 0} };
#pragma unroll
        for (int ks = 0; ks < 16; ks++) {
            const fa3_h16 a = fa3_ld16(qs + (16 * w + lc) * FA_DP + 16 * ks);
#pragma unroll
            for (int nt = 0; nt < 2; nt++) { s[nt] = fa3_wmma(a, fa3_ld16(kv + (16 * nt + lc) * FA_DP + 16 * ks), s[nt]); }
        }

        // ---- scale, mask, online softmax
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float x0 = s[0][i] * p.scale + (float) mr[i][0], x1 = s[1][i] * p.scale + (float) mr[i][1];
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (int o2 = 8; o2 > 0; o2 >>= 1) { mx = fmaxf(mx, __shfl_xor(mx, o2, 32)); }
            const float m_new = fmaxf(m_run[i], mx);
            const float a_i = m_new == -INFINITY ? 1.f : __expf(m_run[i] - m_new);
            const float p0 = m_new == -INFINITY ? 0.f : __expf(x0 - m_new), p1 = m_new == -INFINITY ? 0.f : __expf(x1 - m_new);
#pragma unroll
            for (int j = 0; j < 16; j++) { o[j][i] *= a_i; }
            l_run[i] = l_run[i] * a_i + p0 + p1;
            m_run[i] = m_new;
            pw[(2 * i + hh) * FA_PP + lc]      = (_Float16) p0;
            pw[(2 * i + hh) * FA_PP + 16 + lc] = (_Float16) p1;
        }
        const fa3_h16 pa0 = fa3_ld16(pw + lc * FA_PP), pa1 = fa3_ld16(pw + lc * FA_PP + 16);

        // ---- V^T into the tile buffer: Vt[d][position], lanes contiguous in position
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const _Float16 * uh = (const _Float16 *) &vr[q];
#pragma unroll
            for (int m = 0; m < 8; m++) { kv[(64 * w + 8 * q + m) * FA_VP + lane] = uh[m]; }
        }
        __syncthreads();
        if (t < t_hi) { load_k((int64_t) (t + 1) * FA_KT); }        // next K tile in flight during P V

        // ---- O += P V
#pragma unroll
        for (int j = 0; j < 16; j++) {
            o[j] = fa3_wmma(pa0, fa3_ld16(kv + (16 * j + lc) * FA_VP), o[j]);
            o[j] = fa3_wmma(pa1, fa3_ld16(kv + (16 * j + lc) * FA_VP + 16), o[j]);
        }
    }

    // ---- normalise and store
#pragma unroll
    for (int i = 0; i < 8; i++) {
        float l = l_run[i];
#pragma unroll
        for (int o2 = 8; o2 > 0; o2 >>= 1) { l += __shfl_xor(l, o2, 32); }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        const int row = 16 * w + 2 * i + hh;
        const int64_t tk = tok0 + row / gqa;
        if (tk >= p.n_tokens) { continue; }
        const int head = g * gqa + row % gqa;
        float * d = (float *) (p.dst + seq * p.d_nb3 + tk * p.d_nb2 + (size_t) head * p.d_nb1);
#pragma unroll
        for (int j = 0; j < 16; j++) { d[16 * j + lc] = o[j][i] * inv; }
    }
}

} // namespace

bool ggml_cuda_flash_attn_ext_rdna3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = !getenv("GGML_FA_RDNA3") || atoi(getenv("GGML_FA_RDNA3")) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * sinks = dst->src[4];
    if (sinks || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) { return false; }
    if (Q->ne[0] != FA_D || K->ne[0] != FA_D || V->ne[0] != FA_D || Q->ne[1] < 16) { return false; }
    float scale, max_bias, softcap;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.f || softcap != 0.f) { return false; }
    if (K->ne[1] % FA_KT != 0 || K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) { return false; }
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    if (gqa < 1 || FA_R % gqa != 0) { return false; }
    if (Q->nb[0] != 4 || K->nb[0] != 2 || V->nb[0] != 2 || (Q->nb[1] % 16) || (K->nb[1] % 16) || (V->nb[1] % 16)) { return false; }
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || (mask->nb[1] % 16) || mask->nb[0] != 2)) { return false; }

    fa3_params p;
    p.q = (const char *) Q->data; p.k = (const char *) K->data; p.v = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr; p.dst = (char *) dst->data;
    p.n_tokens = Q->ne[1]; p.n_kv = K->ne[1]; p.n_head = Q->ne[2]; p.n_head_kv = K->ne[2]; p.gqa = gqa;
    p.q_nb1 = Q->nb[1]; p.q_nb2 = Q->nb[2]; p.q_nb3 = Q->nb[3];
    p.k_nb1 = K->nb[1]; p.k_nb2 = K->nb[2]; p.k_nb3 = K->nb[3];
    p.v_nb1 = V->nb[1]; p.v_nb2 = V->nb[2]; p.v_nb3 = V->nb[3];
    p.m_nb1 = mask ? mask->nb[1] : 0; p.m_nb3 = mask ? mask->nb[3] : 0; p.m_ne3 = mask ? mask->ne[3] : 1;
    p.d_nb1 = dst->nb[1]; p.d_nb2 = dst->nb[2]; p.d_nb3 = dst->nb[3];
    p.scale = scale;

    const size_t smem = ((size_t) FA_R * FA_DP + (size_t) std::max(FA_D * FA_VP, FA_KT * FA_DP) + 4 * 16 * FA_PP) * sizeof(_Float16) + 2 * sizeof(int);
    const int tok_per_blk = FA_R / gqa;
    const dim3 grid((unsigned) ((Q->ne[1] + tok_per_blk - 1) / tok_per_blk), (unsigned) K->ne[2], (unsigned) Q->ne[3]);
    fa3_kernel<<<grid, 128, smem, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_flash_attn_ext_rdna3(ggml_backend_cuda_context &, ggml_tensor *) { return false; }

#endif // GGML_USE_HIP
