import json
from pathlib import Path
root=Path('/home/svt/tuning27b50')
p=Path(json.loads((root/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda/int4-wmma.cu'
s=(root/'original/int4-wmma.cu').read_text()
def sub(old,new,count=1):
 global s
 assert s.count(old)==count,(old,s.count(old),count)
 s=s.replace(old,new)
sub('int nb, float * __restrict__ S) {','int nb, float * __restrict__ S, int sh) {')
sub('S[m] = mx;','S[m] = mx * ((float) (1 << sh) / 255.f);')
sub('((uint16_t *) (Wh + ((size_t) ((ft * nsb + sb) * 16 + r)) * 16))[g] = (uint16_t) lrintf(fabsf(d) / s * (float) (1 << sh));','Wh[((size_t) ((ft * nsb + sb) * 16 + r)) * 8 + g] = (uint8_t) min(255, max(0, (int) lrintf(fabsf(d) / s * (float) (1 << sh))));')
sub('uint32_t scw[MI][4];','uint32_t scw[MI][2];',3)
sub('const int4v h0 = bl128(rWh, lc * 16, (((ft0 + mi) * nsb + sb) * 16) * 16);','const int2v h0 = bl64(rWh, lc * 8, (((ft0 + mi) * nsb + sb) * 16) * 8);',3)
sub('scw[mi][0] = h0.x; scw[mi][1] = h0.y; scw[mi][2] = h0.z; scw[mi][3] = h0.w;','scw[mi][0] = h0.x; scw[mi][1] = h0.y;',3)
for v in ['g','u','u0']:
 sub(f'((scw[mi][{v} >> 1] >> (16 * ({v} & 1))) & 0xFFFF)',f'((scw[mi][{v} >> 2] >> (8 * ({v} & 3))) & 0xFF)')
sub('so = ((size_t) (ft * nsb + sb) * 16 + r) * 16 + h * 8;','so = ((size_t) (ft * nsb + sb) * 16 + r) * 8 + h * 4;')
sub('const uint2 sc = *(const uint2 *) (Wh + so);','const uint32_t sc = *(const uint32_t *) (Wh + so);')
sub('uint2 scg = make_uint2(0, 0);','uint32_t scg = 0;')
sub('scg = *(const uint2 *) (Wh2 + so);','scg = *(const uint32_t *) (Wh2 + so);')
sub('const uint32_t sw = g < 2 ? sc.x : sc.y;','const uint32_t sw = sc;')
sub('const uint32_t swg = g < 2 ? scg.x : scg.y;','const uint32_t swg = scg;')
sub('((sw >> (16 * (g & 1))) & 0xFFFF)','((sw >> (8 * g)) & 0xFF)',2)
sub('((swg >> (16 * (g & 1))) & 0xFFFF)','((swg >> (8 * g)) & 0xFF)')
sub('cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 16)','cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 8)')
sub('((const block_q4_0 *) src, M, K / 32, w.S);','((const block_q4_0 *) src, M, K / 32, w.S, sh);')
sub('uint2 subscales[ROWS];','uint32_t subscales[ROWS];')
sub('const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 16 + h * 8;','const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 8 + h * 4;')
sub('subscales[ri] = *(const uint2 *) (Wh + so);','subscales[ri] = *(const uint32_t *) (Wh + so);')
sub('const uint32_t sw = g < 2 ? subscales[ri].x : subscales[ri].y;','const uint32_t sw = subscales[ri];')
p.write_text(s)
print(p)
