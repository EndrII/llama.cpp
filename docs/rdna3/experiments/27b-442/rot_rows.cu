// Several output-row tiles share one fused activation rotation.
template <int ROWS, int NW = 10>
__global__ __launch_bounds__(32 * NW) void k_mv_rot_rows(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ S, float inv,
        const float * __restrict__ X, const float * __restrict__ Y, float * __restrict__ out, const float * __restrict__ residual,
        int M, int K, int mode, int ystride, const float * __restrict__ norm, float eps) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    const int nsb = K / 256, ft0 = blockIdx.x * ROWS;
    __shared__ __align__(16) int8_t rq[NW][256];
    __shared__ __align__(16) float2 rd[NW][8];
    __shared__ float red[ROWS][NW][16];
    half16v rb1, ra2;
#pragma unroll
    for (int k = 0; k < 16; k++) { rb1[k] = h16(k, r); ra2[k] = h16(r, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    float acc[ROWS], scales[ROWS], res[ROWS];
#pragma unroll
    for (int ri = 0; ri < ROWS; ri++) {
        acc[ri] = 0.f;
        scales[ri] = S[(ft0 + ri) * 16 + r];
        res[ri] = residual && h == 0 ? residual[(ft0 + ri) * 16 + r] : 0.f;
    }
    for (int sb = wv; sb < nsb; sb += NW) {
        uint4 weights[ROWS][4];
        uint2 subscales[ROWS];
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {
            const size_t wo = ((size_t) ((ft0 + ri) * nsb + sb) * 8 + h * 4) * 256 + r * 16;
            const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 16 + h * 8;
#pragma unroll
            for (int g = 0; g < 4; g++) { weights[ri][g] = *(const uint4 *) (Wp + wo + g * 256); }
            subscales[ri] = *(const uint2 *) (Wh + so);
        }
        const float8v wr = rotate_block(X + sb * 256, lane, rb1, ra2, Y ? Y + (size_t) sb * ystride : nullptr, mode, norm, eps);
        quant_rot8(wr, lane, &rq[wv][0], &rd[wv][0]);
        __builtin_amdgcn_wave_barrier();
        const uint4 * xq = (const uint4 *) &rq[wv][h * 128];
        const float4 * xd = (const float4 *) &rd[wv][h * 4];
        uint4 xv[8];
        float4 dv[2];
#pragma unroll
        for (int i = 0; i < 8; i++) { xv[i] = xq[i]; }
        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2*g].x,xv[2*g].y,xv[2*g].z,xv[2*g].w,xv[2*g+1].x,xv[2*g+1].y,xv[2*g+1].z,xv[2*g+1].w};
                const uint4 w = weights[ri][g];
                const uint32_t ww[4] = {w.x,w.y,w.z,w.w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2*wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2*wi+1], si);
                }
                const uint32_t sw = g < 2 ? subscales[ri].x : subscales[ri].y;
                acc[ri] += (float) ((sw >> (16 * (g & 1))) & 0xFFFF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
    }
#pragma unroll
    for (int ri = 0; ri < ROWS; ri++) {
        acc[ri] += swap_rows(acc[ri]);
        if (h == 0) { red[ri][wv][r] = acc[ri]; }
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {
            float sum = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) { sum += red[ri][q][r]; }
            out[(ft0 + ri) * 16 + r] = sum * (scales[ri] * inv) + res[ri];
        }
    }
}
