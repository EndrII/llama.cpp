from pathlib import Path
import json,shutil
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
# Preserve the producer-fusion experiment for controlled off/on runs.
for f in ['gated_delta_net.cu','gated_delta_net.cuh','ggml-cuda.cu']:shutil.copy2(d/f,r/('gdn_out_'+f))
p=d/'int4-wmma.cu';s=(r/'original/int4-wmma.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:80],s.count(old));s=s.replace(old,new)
sub('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, bool RAW = false, bool LATE = false>')
old='''        for (int ri = 0; ri < ROWS; ri++) {
            const size_t wo = ((size_t) ((ft0 + ri) * nsb + sb) * 8 + h * 4) * 256 + r * 16;
            const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 4 + h * 2;
#pragma unroll
            for (int g = 0; g < 4; g++) { weights[ri][g] = nt128(Wp, wo + g * 256); }
            subscales[ri] = *(const uint16_t *) (Wh + so);
        }'''
assert old in s
loader=old
sub(old,'''        if (!LATE) {
#pragma unroll
'''+loader+'''        }''')
sub('''Y ? Y + (size_t) sb * ystride : nullptr, mode, norm, eps);''','''RAW ? nullptr : (Y ? Y + (size_t) sb * ystride : nullptr), mode, norm, eps);''')
old='''        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {'''
sub(old,'''        dv[0] = xd[0]; dv[1] = xd[1];
        if (LATE) {
#pragma unroll
'''+loader+'''        }
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {''')
old='''                k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
new='''                const int specialize = getenv("GGML_MV_RAW_SPECIALIZE") ? atoi(getenv("GGML_MV_RAW_SPECIALIZE")) : 0;
                if (!glu_y && specialize == 1) k_mv_rot_rows<ROWS,10,true><<<dim3(M/(16*ROWS)),blk,0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps);
                else if (!glu_y && specialize == 2) k_mv_rot_rows<ROWS,10,true,true><<<dim3(M/(16*ROWS)),blk,0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps);
                else k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
sub(old,new);p.write_text(s);(r/'raw.cu').write_text(s)
