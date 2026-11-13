import json,runpy
from pathlib import Path
r=Path('/home/svt/tuning27b50')
s=(r/'experiment_rms_mv.py').read_text().replace("(r/'scale4.cu').read_text()","(r/'best.cu').read_text()")
exec(compile(s,str(r/'experiment_rms_mv.py'),'exec'))
d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
p=d/'int4-wmma.cu';s=p.read_text()
s=s.replace('struct Int4Weights {',(r/'gate_rows_kernel.txt').read_text()+'\nstruct Int4Weights {')
old='''        k_mv<1, true, true, false, false, false, true><<<dim3(M / 16), dim3(32, MV_NW), 0, stream>>>(w.Wp, w.Wh, w.S, 1.f / (1 << w.sh), g.Wp, g.Wh, g.S, 1.f / (1 << g.sh), nullptr, nullptr, (float *) dst->data, xb, gb, glu, M, K, M, n.x, n.w, 4, 256, nullptr, n.eps);'''
new='''        const int rows = getenv("GGML_INT4_RMS_MV_ROWS") ? atoi(getenv("GGML_INT4_RMS_MV_ROWS")) : 1;
        if (rows == 2) k_gate_norm_rows<2><<<dim3(M/32),dim3(32,10),0,stream>>>(w.Wp,w.Wh,w.S,g.Wp,g.Wh,g.S,n.x,n.w,n.eps,(float *)dst->data,xb,gb,glu);
        else if (rows == 4) k_gate_norm_rows<4><<<dim3(M/64),dim3(32,10),0,stream>>>(w.Wp,w.Wh,w.S,g.Wp,g.Wh,g.S,n.x,n.w,n.eps,(float *)dst->data,xb,gb,glu);
        else k_mv<1, true, true, false, false, false, true><<<dim3(M / 16), dim3(32, MV_NW), 0, stream>>>(w.Wp, w.Wh, w.S, 1.f / (1 << w.sh), g.Wp, g.Wh, g.S, 1.f / (1 << g.sh), nullptr, nullptr, (float *) dst->data, xb, gb, glu, M, K, M, n.x, n.w, 4, 256, nullptr, n.eps);'''
assert s.count(old)==1;p.write_text(s.replace(old,new))
