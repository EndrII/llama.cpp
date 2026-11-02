#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include <algorithm>
#include <type_traits>
#include <cstdlib>

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

#if defined(GGML_USE_HIP)
// RDNA (wave32), S_v == 128, scalar gate: a state column is split over two lanes (64 rows each), so a row reduction is one
// lane exchange instead of five, and a wave owns 16 columns.  k/q of a token are staged per wave in LDS (double buffered).
template <bool keep_rs_t>
__global__ void __launch_bounds__(128, 2)
gated_delta_net_cols(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
                     const float * __restrict__ g, const float * __restrict__ beta, const float * __restrict__ curr_state,
                     float * __restrict__ dst, float * __restrict__ state,
                     int64_t H, int64_t n_tokens, int64_t n_seqs,
                     int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                     int64_t sb1, int64_t sb2, int64_t sb3,
                     const uint3 neqk1_magic, const uint3 rq3_magic, float scale, int64_t state_slot_stride, int K) {
    constexpr int S_v = 128, R = 64;
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x, half = lane >> 4, lc = lane & 15;
    const int col  = blockIdx.z * 64 + threadIdx.y * 16 + lc;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;
    const float * st_in = curr_state + sequence * H * S_v * S_v + h_idx * S_v * S_v + col * S_v + half * R;

    float s[R];
#pragma unroll
    for (int i = 0; i < R; i += 4) {
        const float4 x = *(const float4 *) (st_in + i);
        s[i] = x.x; s[i + 1] = x.y; s[i + 2] = x.z; s[i + 3] = x.w;
    }

    __shared__ __align__(16) float ks[4][2][S_v];
    __shared__ __align__(16) float qs[4][2][S_v];
    float (*kw)[S_v] = ks[threadIdx.y];
    float (*qw)[S_v] = qs[threadIdx.y];
    const float * kq0 = k + iq3 * sq3 + iq1 * sq1 + lane * 4;
    const float * qq0 = q + iq3 * sq3 + iq1 * sq1 + lane * 4;
    float4 kn = make_float4(0.f, 0.f, 0.f, 0.f), qn = kn;
    if (n_tokens > 0) { kn = *(const float4 *) kq0; qn = *(const float4 *) qq0; }
    *(float4 *) (kw[0] + lane * 4) = kn;
    *(float4 *) (qw[0] + lane * 4) = qn;
    asm volatile("" ::: "memory");

    float v_nx = 0.f, beta_nx = 0.f, g_nx = 0.f;
    if (n_tokens > 0) {
        const int64_t gb0 = sequence * sb3 + h_idx * sb1;
        v_nx = v[sequence * sv3 + h_idx * sv1 + col]; beta_nx = beta[gb0]; g_nx = g[gb0];
    }

    for (int t = 0; t < n_tokens; t++) {
        const int cur = t & 1;
        const float beta_val = beta_nx, g_val = expf(g_nx), v_val = v_nx;
        if (t + 1 < n_tokens) {
            kn = *(const float4 *) (kq0 + (t + 1) * sq2);
            qn = *(const float4 *) (qq0 + (t + 1) * sq2);
            const int64_t gb1 = sequence * sb3 + (t + 1) * sb2 + h_idx * sb1;
            v_nx = v[sequence * sv3 + (t + 1) * sv2 + h_idx * sv1 + col]; beta_nx = beta[gb1]; g_nx = g[gb1];
        }
        const float * k_t = kw[cur] + half * R;
        const float * q_t = qw[cur] + half * R;

        float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
#pragma unroll
        for (int i = 0; i < R; i += 4) {
            const float4 kk = *(const float4 *) (k_t + i);
            kv0 = fmaf(s[i], kk.x, kv0); kv1 = fmaf(s[i + 1], kk.y, kv1);
            kv2 = fmaf(s[i + 2], kk.z, kv2); kv3 = fmaf(s[i + 3], kk.w, kv3);
        }
        float kv = (kv0 + kv1) + (kv2 + kv3);
        kv += __shfl_xor(kv, 16, 32);
        const float delta = (v_val - g_val * kv) * beta_val;

        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int i = 0; i < R; i += 4) {
            const float4 kk = *(const float4 *) (k_t + i), qq = *(const float4 *) (q_t + i);
            s[i]     = fmaf(g_val, s[i],     kk.x * delta); a0 = fmaf(s[i],     qq.x, a0);
            s[i + 1] = fmaf(g_val, s[i + 1], kk.y * delta); a1 = fmaf(s[i + 1], qq.y, a1);
            s[i + 2] = fmaf(g_val, s[i + 2], kk.z * delta); a2 = fmaf(s[i + 2], qq.z, a2);
            s[i + 3] = fmaf(g_val, s[i + 3], kk.w * delta); a3 = fmaf(s[i + 3], qq.w, a3);
        }
        float att = (a0 + a1) + (a2 + a3);
        att += __shfl_xor(att, 16, 32);
        if (half == 0) { attn_data[col] = att * scale; }
        attn_data += S_v * H;

        *(float4 *) (kw[cur ^ 1] + lane * 4) = kn;
        *(float4 *) (qw[cur ^ 1] + lane * 4) = qn;
        asm volatile("" ::: "memory");

        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * cs = state + target_slot * state_slot_stride + col * S_v + half * R;
#pragma unroll
                for (int i = 0; i < R; i++) { cs[i] = s[i]; }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < R; i += 4) {
            *(float4 *) (state + col * S_v + half * R + i) = make_float4(s[i], s[i + 1], s[i + 2], s[i + 3]);
        }
    }
}
#endif // GGML_USE_HIP

#if defined(GGML_USE_HIP)
// ---------------------------------------------------------------------------------------------------------------------------------
// Chunked (WY form) gated delta net for RDNA3 WMMA, scalar gate, S_v == 128.  Chunks of C = 64 tokens.
//
//   gdn_chunk_prep (parallel over chunk x head x seq):  A = tril(beta_j exp(g_j - g_i) k_j.k_i);  X = (I + A)^-1 [beta v | beta exp(g) k]
//       (forward substitution in f32), giving  U = T beta v  and  Wm = T beta exp(g) k;  plus Qg = exp(g) q,  Kd = exp(g_C - g) k  and
//       P = tril(exp(g_t - g_j) q_t.k_j)  (the k.k / q.k products on f16 WMMA).
//   gdn_chunk_scan (sequential over the chunks of a head, parallel over head x seq x value-column block):  with the state S0 [128 x NB]
//       delta = U - Wm S0,   out = scale (Qg S0 + P delta),   S <- exp(g_C) S + Kd^T delta,   all on f16 WMMA with f32 accumulation;
//       the f32 state lives in the WMMA accumulators, an f16 copy (plus an optional f16 residual) in LDS for the B operand.
// ---------------------------------------------------------------------------------------------------------------------------------
#if defined(__GFX11__) || !defined(__HIP_DEVICE_COMPILE__)
#define GDN_WMMA_OK 1
#endif

typedef float    gdn_f8  __attribute__((ext_vector_type(8)));
typedef _Float16 gdn_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 gdn_h8  __attribute__((ext_vector_type(8)));

namespace {

constexpr int GDN_C = 64;
constexpr int GDN_D = 128;
constexpr int GDN_KP = GDN_D + 8;      // LDS row stride (halves) of the K / Q tiles and of the state
constexpr int GDN_AP = GDN_C + 4;      // LDS row stride (floats) of the A matrix
constexpr int GDN_DP = GDN_C + 8;      // LDS row stride (halves) of delta^T

__device__ __forceinline__ gdn_f8 gdn_wmma(const gdn_h16 & a, const gdn_h16 & b, const gdn_f8 & c) {
#ifdef GDN_WMMA_OK
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    return c;
#endif
}

__device__ __forceinline__ gdn_h16 gdn_ld16(const _Float16 * p) {
    const uint4 * u = (const uint4 *) p;
    const uint4 a = u[0], b = u[1];
    struct { uint4 x, y; } t = { a, b };
    return __builtin_bit_cast(gdn_h16, t);
}

__global__ void __launch_bounds__(256, 1)
gdn_chunk_prep(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
               const float * __restrict__ g, const float * __restrict__ beta,
               _Float16 * __restrict__ wm, float * __restrict__ uu, _Float16 * __restrict__ qg,
               _Float16 * __restrict__ kdt, _Float16 * __restrict__ pp, float * __restrict__ gc,
               int64_t H, int64_t n_tokens, int64_t NC,
               int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
               int64_t sb1, int64_t sb2, int64_t sb3, const uint3 neqk1_magic, const uint3 rq3_magic) {
    constexpr int C = GDN_C, D = GDN_D, KP = GDN_KP, AP = GDN_AP, BTP = GDN_C + 8, RP = 132;
    const int c = blockIdx.x, h = blockIdx.y, seq = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, hh = lane >> 4, lc = lane & 15;
    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);
    const int64_t id = ((int64_t) seq * H + h) * NC + c;

    __shared__ __align__(16) _Float16 buf[2 * C * KP];      // K | Q tiles, later the transposed panel solutions [128][BTP]
    __shared__ __align__(16) float am[C * AP];
    __shared__ __align__(16) float rb[16 * RP];
    __shared__ float gam[C], bet[C], eg[C], ek[C];
    _Float16 * kh = buf, * qh = buf + C * KP, * bt = buf;

    const float * kb = k + iq3 * sq3 + iq1 * sq1;
    const float * qb = q + iq3 * sq3 + iq1 * sq1;
    const float * vb = v + seq * sv3 + (int64_t) h * sv1;
    const int64_t tok0 = (int64_t) c * C;

    // operand columns of both panel passes (value / key column 16 w + lc of every token), requested up front to hide their latency
    float lv[2][4][8];
#pragma unroll
    for (int p = 0; p < 4; p++)
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int64_t tk = std::min(tok0 + 16 * p + 2 * i + hh, n_tokens - 1);
            lv[0][p][i] = vb[tk * sv2 + 16 * w + lc];
            lv[1][p][i] = kb[tk * sq2 + 16 * w + lc];
        }

    {
        float4 kv[8], qv[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = tid + 256 * e, row = idx >> 5, c4 = idx & 31;
            const int64_t tk = std::min(tok0 + row, n_tokens - 1);
            kv[e] = *(const float4 *) (kb + tk * sq2 + c4 * 4);
            qv[e] = *(const float4 *) (qb + tk * sq2 + c4 * 4);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int idx = tid + 256 * e, row = idx >> 5, c4 = idx & 31;
            const float m = tok0 + row < n_tokens ? 1.f : 0.f;
            _Float16 * kd = kh + row * KP + c4 * 4, * qd = qh + row * KP + c4 * 4;
            kd[0] = (_Float16) (m * kv[e].x); kd[1] = (_Float16) (m * kv[e].y); kd[2] = (_Float16) (m * kv[e].z); kd[3] = (_Float16) (m * kv[e].w);
            qd[0] = (_Float16) (m * qv[e].x); qd[1] = (_Float16) (m * qv[e].y); qd[2] = (_Float16) (m * qv[e].z); qd[3] = (_Float16) (m * qv[e].w);
        }
    }
    if (tid < C) {
        float gv = 0.f, bv = 0.f;
        if (tok0 + tid < n_tokens) {
            const int64_t o = (int64_t) seq * sb3 + (tok0 + tid) * sb2 + (int64_t) h * sb1;
            gv = g[o]; bv = beta[o];
        }
        float x = gv;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const float y = __shfl_up(x, d, 32);
            if ((tid & 31) >= d) { x += y; }
        }
        gam[tid] = x; bet[tid] = bv;
    }
    __syncthreads();
    if (tid >= 32 && tid < C) { gam[tid] += gam[31]; }
    __syncthreads();
    if (tid < C) { eg[tid] = expf(gam[tid]); ek[tid] = expf(gam[C - 1] - gam[tid]); }

    // k.k^T (-> A) and q.k^T (-> P) on the lower-triangular 16x16 tiles
    {
        const int lt_it[10] = { 0, 1, 1, 2, 2, 2, 3, 3, 3, 3 }, lt_jt[10] = { 0, 0, 1, 0, 1, 2, 0, 1, 2, 3 };
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int lt = w + 8 * r;
            if (lt >= 10) { break; }
            const int it = lt_it[lt], jt = lt_jt[lt];
            gdn_f8 cg = {0, 0, 0, 0, 0, 0, 0, 0}, cq = cg;
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                const gdn_h16 a  = gdn_ld16(kh + (16 * it + lc) * KP + 16 * ks);
                const gdn_h16 aq = gdn_ld16(qh + (16 * it + lc) * KP + 16 * ks);
                const gdn_h16 b  = gdn_ld16(kh + (16 * jt + lc) * KP + 16 * ks);
                cg = gdn_wmma(a, b, cg);
                cq = gdn_wmma(aq, b, cq);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int row = 16 * it + 2 * i + hh, col = 16 * jt + lc;
                const float dec = col <= row ? expf(gam[row] - gam[col]) : 0.f;
                if (row > col) { am[row * AP + col] = bet[row] * dec * cg[i]; }
                pp[id * C * C + row * C + col] = (_Float16) (dec * cq[i]);
            }
        }
    }
    __syncthreads();

    // Qg = exp(g) q (row major) and Kd^T = (exp(g_C - g) k)^T from the f16 tiles
#pragma unroll
    for (int e = 0; e < 4; e++) {
        const int item = tid + 256 * e;
        {
            const int j = item >> 4, d0 = (item & 15) * 8;
            const gdn_h8 u = *(const gdn_h8 *) (qh + j * KP + d0);
            gdn_h8 o;
#pragma unroll
            for (int m = 0; m < 8; m++) { o[m] = (_Float16) (eg[j] * (float) u[m]); }
            *(gdn_h8 *) (qg + id * C * D + j * D + d0) = o;
        }
        {
            const int dk = item >> 3, j0 = (item & 7) * 8;
            gdn_h8 o;
#pragma unroll
            for (int m = 0; m < 8; m++) { o[m] = (_Float16) (ek[j0 + m] * (float) kh[(j0 + m) * KP + dk]); }
            *(gdn_h8 *) (kdt + id * C * D + dk * C + j0) = o;
        }
    }
    __syncthreads();      // kh / qh are dead from here on

    // X = (I + A)^-1 [beta v | beta exp(g) k] by 16-row panels: the off-diagonal updates on WMMA, the 16x16 triangular solves in f32.
    // pass 0: U (value columns), pass 1: Wm (key columns, stored negated)
    auto run_pass = [&](auto tag) {
        constexpr bool isw = decltype(tag)::value;
        __shared__ float rsc[C];
        if (tid < C) { rsc[tid] = tok0 + tid < n_tokens ? bet[tid] * (isw ? eg[tid] : 1.f) : 0.f; }
        __syncthreads();
        gdn_f8 bacc[4];
#pragma unroll
        for (int p = 0; p < 4; p++)
#pragma unroll
            for (int i = 0; i < 8; i++) { bacc[p][i] = lv[isw ? 1 : 0][p][i] * rsc[16 * p + 2 * i + hh]; }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            gdn_f8 acc = bacc[p];
#pragma unroll
            for (int kk = 0; kk < p; kk++) {
                const float4 * ar = (const float4 *) (am + (16 * p + lc) * AP + 16 * kk);
                gdn_h16 a;
#pragma unroll
                for (int q4 = 0; q4 < 4; q4++) {
                    const float4 f = ar[q4];
                    a[4 * q4] = (_Float16) -f.x; a[4 * q4 + 1] = (_Float16) -f.y; a[4 * q4 + 2] = (_Float16) -f.z; a[4 * q4 + 3] = (_Float16) -f.w;
                }
                acc = gdn_wmma(a, gdn_ld16(bt + (16 * w + lc) * BTP + 16 * kk), acc);
            }
#pragma unroll
            for (int i = 0; i < 8; i++) { rb[(2 * i + hh) * RP + 16 * w + lc] = acc[i]; }
            __syncthreads();
            if (tid < 128) {
                float xv[16];
#pragma unroll
                for (int m = 0; m < 16; m++) {
                    float sacc = rb[m * RP + tid];
                    const float4 * ar = (const float4 *) (am + (16 * p + m) * AP + 16 * p);
#pragma unroll
                    for (int q4 = 0; q4 < m / 4; q4++) {
                        const float4 f = ar[q4];
                        sacc = fmaf(-f.x, xv[4 * q4], sacc); sacc = fmaf(-f.y, xv[4 * q4 + 1], sacc);
                        sacc = fmaf(-f.z, xv[4 * q4 + 2], sacc); sacc = fmaf(-f.w, xv[4 * q4 + 3], sacc);
                    }
#pragma unroll
                    for (int q = 4 * (m / 4); q < m; q++) { sacc = fmaf(-am[(16 * p + m) * AP + 16 * p + q], xv[q], sacc); }
                    xv[m] = sacc;
                }
#pragma unroll
                for (int m = 0; m < 16; m++) {
                    if constexpr (!isw) { uu[id * C * D + (16 * p + m) * D + tid] = xv[m]; }
                    else                { wm[id * C * D + (16 * p + m) * D + tid] = (_Float16) -xv[m]; }
                }
                if (p < 3) {
                    gdn_h8 o0, o1;
#pragma unroll
                    for (int m = 0; m < 8; m++) { o0[m] = (_Float16) xv[m]; o1[m] = (_Float16) xv[8 + m]; }
                    *(gdn_h8 *) (bt + tid * BTP + 16 * p)     = o0;
                    *(gdn_h8 *) (bt + tid * BTP + 16 * p + 8) = o1;
                }
            }
            __syncthreads();
        }
    };
    run_pass(std::false_type());
    run_pass(std::true_type());
    if (tid == 0) { gc[id] = eg[C - 1]; }
}

template <int NT, bool SPLIT>
__global__ void __launch_bounds__(128, 2)
gdn_chunk_scan(const _Float16 * __restrict__ wm, const float * __restrict__ uu, const _Float16 * __restrict__ qg,
               const _Float16 * __restrict__ kdt, const _Float16 * __restrict__ pp, const float * __restrict__ gc,
               const float * __restrict__ s_in, float * __restrict__ out, float * __restrict__ s_out,
               int64_t H, int64_t n_tokens, int64_t NC, float scale) {
    constexpr int C = GDN_C, D = GDN_D, NB = NT * 16, SP = GDN_KP, DP = GDN_DP;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, hh = lane >> 4, lc = lane & 15;
    const int h = blockIdx.y, seq = blockIdx.z, col0 = blockIdx.x * NB;

    __shared__ __align__(16) _Float16 st[2][NB][SP];
    __shared__ __align__(16) _Float16 stl[SPLIT ? 2 : 1][SPLIT ? NB : 1][SP];
    __shared__ __align__(16) _Float16 dt[NB][DP];
    __shared__ __align__(16) _Float16 dtl[SPLIT ? NB : 1][SPLIT ? DP : 1];

    const int64_t soff = ((int64_t) seq * H + h) * D * D + (int64_t) col0 * D;
    gdn_f8 sacc[2][NT];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int dk = 32 * w + 16 * mt + 2 * i + hh, n = 16 * nt + lc;
                const float x = s_in[soff + (int64_t) n * D + dk];
                sacc[mt][nt][i] = x;
                const _Float16 xh = (_Float16) x;
                st[0][n][dk] = xh;
                if constexpr (SPLIT) { stl[0][n][dk] = (_Float16) (x - (float) xh); }
            }
    __syncthreads();

    for (int c = 0; c < NC; c++) {
        const int cur = c & 1;
        const int64_t id = ((int64_t) seq * H + h) * NC + c;
        const _Float16 * wmc = wm + id * C * D, * qgc = qg + id * C * D, * kdc = kdt + id * C * D, * ppc = pp + id * C * C;
        const float * uc = uu + id * C * D;

        // delta = U - Wm S0   (Wm is stored negated), and the state half of the output, Qg S0, from the same S0 fragments
        gdn_f8 ad[NT], ao[NT];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            ao[nt] = gdn_f8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int i = 0; i < 8; i++) { ad[nt][i] = uc[(16 * w + 2 * i + hh) * D + col0 + 16 * nt + lc]; }
        }
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const gdn_h16 a  = gdn_ld16(wmc + (16 * w + lc) * D + 16 * ks);
            const gdn_h16 aq = gdn_ld16(qgc + (16 * w + lc) * D + 16 * ks);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const gdn_h16 b = gdn_ld16(&st[cur][16 * nt + lc][16 * ks]);
                ad[nt] = gdn_wmma(a, b, ad[nt]);
                ao[nt] = gdn_wmma(aq, b, ao[nt]);
                if constexpr (SPLIT) {
                    const gdn_h16 bl = gdn_ld16(&stl[cur][16 * nt + lc][16 * ks]);
                    ad[nt] = gdn_wmma(a, bl, ad[nt]);
                    ao[nt] = gdn_wmma(aq, bl, ao[nt]);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const _Float16 hv = (_Float16) ad[nt][i];
                dt[16 * nt + lc][16 * w + 2 * i + hh] = hv;
                if constexpr (SPLIT) { dtl[16 * nt + lc][16 * w + 2 * i + hh] = (_Float16) (ad[nt][i] - (float) hv); }
            }
        __syncthreads();

        // out = scale (Qg S0 + P delta)
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            if (ks > w) { break; }
            const gdn_h16 a = gdn_ld16(ppc + (16 * w + lc) * C + 16 * ks);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                ao[nt] = gdn_wmma(a, gdn_ld16(&dt[16 * nt + lc][16 * ks]), ao[nt]);
                if constexpr (SPLIT) { ao[nt] = gdn_wmma(a, gdn_ld16(&dtl[16 * nt + lc][16 * ks]), ao[nt]); }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int64_t tok = (int64_t) c * C + 16 * w + 2 * i + hh;
                if (tok < n_tokens) { out[(((int64_t) seq * n_tokens + tok) * H + h) * D + col0 + 16 * nt + lc] = ao[nt][i] * scale; }
            }

        // S <- exp(g_C) S + Kd^T delta
        const float gC = gc[id];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) { sacc[mt][nt] *= gC; }
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                const gdn_h16 a = gdn_ld16(kdc + (32 * w + 16 * mt + lc) * C + 16 * ks);
#pragma unroll
                for (int nt = 0; nt < NT; nt++) {
                    sacc[mt][nt] = gdn_wmma(a, gdn_ld16(&dt[16 * nt + lc][16 * ks]), sacc[mt][nt]);
                    if constexpr (SPLIT) { sacc[mt][nt] = gdn_wmma(a, gdn_ld16(&dtl[16 * nt + lc][16 * ks]), sacc[mt][nt]); }
                }
            }
        }
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int dk = 32 * w + 16 * mt + 2 * i + hh, n = 16 * nt + lc;
                    const _Float16 xh = (_Float16) sacc[mt][nt][i];
                    st[cur ^ 1][n][dk] = xh;
                    if constexpr (SPLIT) { stl[cur ^ 1][n][dk] = (_Float16) (sacc[mt][nt][i] - (float) xh); }
                }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int dk = 32 * w + 16 * mt + 2 * i + hh, n = 16 * nt + lc;
                s_out[soff + (int64_t) n * D + dk] = sacc[mt][nt][i];
            }
}

} // namespace

// returns false when the chunked path does not apply (needs RDNA3 WMMA, S_v == 128, enough tokens)
static bool gdn_chunk_launch(ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        const float * s_d, float * dst_d, float * state_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, cudaStream_t stream) {
    static const int mode    = getenv("GGML_GDN_CHUNK") ? atoi(getenv("GGML_GDN_CHUNK")) : 1;       // 0 off, 1 f16, 2 f16 + residual
    static const int min_tok = getenv("GGML_GDN_CHUNK_MIN") ? atoi(getenv("GGML_GDN_CHUNK_MIN")) : 64;
    static const int nt_sel  = getenv("GGML_GDN_NT") ? atoi(getenv("GGML_GDN_NT")) : 1;
    if (mode == 0 || S_v != GDN_D || n_tokens < min_tok || !GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    constexpr int C = GDN_C, D = GDN_D;
    constexpr int64_t SEG = 1024;
    const int64_t ncm = SEG / C;
    const int64_t nid = n_seqs * H * ncm;
    ggml_cuda_pool_alloc<_Float16> wm(ctx.pool(), nid * C * D), qg(ctx.pool(), nid * C * D), kdt(ctx.pool(), nid * C * D), pp(ctx.pool(), nid * C * C);
    ggml_cuda_pool_alloc<float> uu(ctx.pool(), nid * C * D), gc(ctx.pool(), nid);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    for (int64_t t0 = 0; t0 < n_tokens; t0 += SEG) {
        const int64_t nt = std::min(SEG, n_tokens - t0);
        const int64_t NC = (nt + C - 1) / C;
        gdn_chunk_prep<<<dim3(NC, H, n_seqs), 256, 0, stream>>>(q_d + t0 * sq2, k_d + t0 * sq2, v_d + t0 * sv2, g_d + t0 * sb2, b_d + t0 * sb2,
            wm.get(), uu.get(), qg.get(), kdt.get(), pp.get(), gc.get(), H, nt, NC,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);
        const float * s_src = t0 == 0 ? s_d : state_d;
        float * out = dst_d + t0 * S_v * H;
        if (nt_sel == 1) {
            if (mode == 2) { gdn_chunk_scan<1, true ><<<dim3(D / 16, H, n_seqs), 128, 0, stream>>>(wm.get(), uu.get(), qg.get(), kdt.get(), pp.get(), gc.get(), s_src, out, state_d, H, nt, NC, scale); }
            else           { gdn_chunk_scan<1, false><<<dim3(D / 16, H, n_seqs), 128, 0, stream>>>(wm.get(), uu.get(), qg.get(), kdt.get(), pp.get(), gc.get(), s_src, out, state_d, H, nt, NC, scale); }
        } else {
            if (mode == 2) { gdn_chunk_scan<2, true ><<<dim3(D / 32, H, n_seqs), 128, 0, stream>>>(wm.get(), uu.get(), qg.get(), kdt.get(), pp.get(), gc.get(), s_src, out, state_d, H, nt, NC, scale); }
            else           { gdn_chunk_scan<2, false><<<dim3(D / 32, H, n_seqs), 128, 0, stream>>>(wm.get(), uu.get(), qg.get(), kdt.get(), pp.get(), gc.get(), s_src, out, state_d, H, nt, NC, scale); }
        }
    }
    return true;
}
#endif // GGML_USE_HIP

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
#if defined(GGML_USE_HIP)
            if constexpr (!KDA) {
                if (warp_size == 32) {
                    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 2), dim3(32, 4, 1), 0, stream);
                    ggml_cuda_kernel_launch(gated_delta_net_cols<keep_rs_t>, lp,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                    break;
                }
            }
#endif
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

#if defined(GGML_USE_HIP)
    if (!kda && !keep_rs && gdn_chunk_launch(ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, stream)) {
        return;
    }
#endif

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
