import json
from pathlib import Path
r=Path('/home/svt/tuning27b50');s=(r/'scale8.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:80],s.count(old));s=s.replace(old,new)
sub('/ 255.f','/ 15.f')
sub('Wh[((size_t) ((ft * nsb + sb) * 16 + r)) * 8 + g] = (uint8_t) min(255, max(0, (int) lrintf(fabsf(d) / s * (float) (1 << sh))));','''if ((g & 1) == 0) {
        const float d1 = fabsf(__half2float(W[(size_t) m * nb + sb * 8 + g + 1].d));
        const int q0 = min(15, max(0, (int) lrintf(fabsf(d) / s * (float) (1 << sh))));
        const int q1 = min(15, max(0, (int) lrintf(d1 / s * (float) (1 << sh))));
        Wh[((size_t) ((ft * nsb + sb) * 16 + r)) * 4 + g / 2] = q0 | (q1 << 4);
    }''')
sub('uint32_t scw[MI][2];','uint32_t scw[MI][1];',3)
sub('const int2v h0 = bl64(rWh, lc * 8, (((ft0 + mi) * nsb + sb) * 16) * 8);','const uint32_t h0 = *(const uint32_t *) (Wh + (((ft0 + mi) * nsb + sb) * 16 + lc) * 4);',3)
sub('scw[mi][0] = h0.x; scw[mi][1] = h0.y;','scw[mi][0] = h0;',3)
for v in ['g','u','u0']:sub(f'((scw[mi][{v} >> 2] >> (8 * ({v} & 3))) & 0xFF)',f'((scw[mi][0] >> (4 * {v})) & 0xF)')
sub('so = ((size_t) (ft * nsb + sb) * 16 + r) * 8 + h * 4;','so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;')
sub('*(const uint32_t *) (Wh + so)','*(const uint16_t *) (Wh + so)',2)
sub('*(const uint32_t *) (Wh2 + so)','*(const uint16_t *) (Wh2 + so)')
sub('((sw >> (8 * g)) & 0xFF)','((sw >> (4 * g)) & 0xF)',2)
sub('((swg >> (8 * g)) & 0xFF)','((swg >> (4 * g)) & 0xF)')
sub('cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 8)','cudaMalloc((void **) &w.Wh, (size_t) M * nsb * 4)')
sub('const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 8 + h * 4;','const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 4 + h * 2;')
p=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda/int4-wmma.cu';p.write_text(s)
(r/'scale4.cu').write_text(s)
