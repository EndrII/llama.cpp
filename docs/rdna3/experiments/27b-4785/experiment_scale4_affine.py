import json
from pathlib import Path
r=Path('/home/svt/tuning27b50');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'scale4.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:70],s.count(old));s=s.replace(old,new)
sub('''    float mx = 0.f;
    for (int b = 0; b < nb; b++) mx = fmaxf(mx, fabsf(__half2float(W[(size_t) m * nb + b].d)));
    S[m] = mx * ((float) (1 << sh) / 15.f);''','''    float mx = 0.f, mn = INFINITY;
    for (int b = 0; b < nb; b++) {
        const float x = fabsf(__half2float(W[(size_t) m * nb + b].d));
        mx = fmaxf(mx, x); mn = fminf(mn, x);
    }
    const int base = mx > 0.f ? min(255, max(0, (int) lrintf(mn / mx * 255.f))) : 0;
    const int step = max(1, (int) lrintf((255.f - base) / 15.f));
    const uint32_t packed = __half_as_ushort(__float2half(mx)) | (base << 16) | (step << 24);
    S[m] = __uint_as_float(packed);
    (void) sh;''')
sub('// Q4_0 -> operand-ready int4 K-slices','''__device__ __forceinline__ float row_scale(float packed) {
    return __half2float(__ushort_as_half((uint16_t) __float_as_uint(packed))) * (1.f / 255.f);
}
__device__ __forceinline__ uint32_t row_code(uint32_t packed, uint32_t code) {
    return ((packed >> 16) & 255) + (packed >> 24) * code;
}

// Q4_0 -> operand-ready int4 K-slices''')
sub('const float s = S[m] > 0.f ? S[m] : 1.f;','''const float sx = row_scale(S[m]);
    const float s = sx > 0.f ? sx : 1.f;
    const uint32_t packed = __float_as_uint(S[m]);
    const float base = (float) ((packed >> 16) & 255), step = (float) (packed >> 24);
    (void) sh;''')
sub('lrintf(fabsf(d) / s * (float) (1 << sh))','lrintf((fabsf(d) / s - base) / step)')
sub('lrintf(d1 / s * (float) (1 << sh))','lrintf((d1 / s - base) / step)')
# Three prefill kernels load the compact row metadata once, outside K.
sub('''    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][1];''','''    uint32_t row_meta[MI];
#pragma unroll
    for (int mi = 0; mi < MI; mi++) row_meta[mi] = __float_as_uint(WS[(ft0 + mi) * 16 + lc]);
    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][1];''',3)
for v in ['g','u','u0']:
 sub(f'(int) ((scw[mi][0] >> (4 * {v})) & 0xF)',f'(int) row_code(row_meta[mi], ((scw[mi][0] >> (4 * {v})) & 0xF))')
sub('1.f / (float) ((1 << SH) * 16)','1.f / 16.f')
sub('1.f / (float) ((1 << SH8) * 16)','1.f / 16.f',2)
sub('WS[feat] * inv','row_scale(WS[feat]) * inv',3)
sub('pS = S[prow]','pS = row_scale(S[prow])')
sub('''    float acc[NC], accg[NC];''','''    const uint32_t row_meta = __float_as_uint(S[prow]);
    const uint32_t row_meta2 = GATE ? __float_as_uint(S2[prow]) : 0;
    float acc[NC], accg[NC];''')
sub('acc[j] += (float) ((sw >> (4 * g)) & 0xF)','acc[j] += (float) row_code(row_meta, ((sw >> (4 * g)) & 0xF))')
sub('(float) ((swg >> (4 * g)) & 0xF)','(float) row_code(row_meta2, ((swg >> (4 * g)) & 0xF))')
sub('? pS : S[row]', '? pS : row_scale(S[row])')
sub('GATE ? S2[row] * inv2','GATE ? row_scale(S2[row]) * inv2')
sub('w.inplace = ip; w.sh = sh;','w.inplace = ip; w.sh = 0;')
sub('float acc[ROWS], scales[ROWS], res[ROWS];','float acc[ROWS], scales[ROWS], res[ROWS];\n    uint32_t row_meta[ROWS];')
sub('scales[ri] = S[(ft0 + ri) * 16 + r];','''scales[ri] = row_scale(S[(ft0 + ri) * 16 + r]);
        row_meta[ri] = __float_as_uint(S[(ft0 + ri) * 16 + r]);''')
sub('acc[ri] += (float) ((sw >> (4 * g)) & 0xF)','acc[ri] += (float) row_code(row_meta[ri], ((sw >> (4 * g)) & 0xF))')
(d/'int4-wmma.cu').write_text(s);(r/'scale4_affine.cu').write_text(s)
for f in ['ggml-cuda.cu','int4-wmma.cuh']:(d/f).write_text((r/'original'/f).read_text())
