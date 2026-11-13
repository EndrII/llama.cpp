from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');p=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda/int4-wmma.cu'
s=(r/'original/int4-wmma.cu').read_text()
s=s.replace('struct Int4Weights {',(r/'mv_wmma.txt').read_text()+'\nstruct Int4Weights {')
old='''    switch (N) { MV_CASE(1) MV_CASE(2) MV_CASE(3) MV_CASE(4) MV_CASE(5) MV_CASE(6) MV_CASE(7) MV_CASE(8) }'''
new='''    if(N==1&&gate&&M==17408&&K==5120&&getenv("GGML_INT4_DECODE_WMMA")&&atoi(getenv("GGML_INT4_DECODE_WMMA"))) {
        k_mv_wmma<true><<<grid,blk,0,stream>>>(w.Wp,w.Wh,w.S,inv,g.Wp,g.Wh,g.S,inv2,xq_ptr,xd_ptr,(float*)dst->data,xb,gb,glu,K);
        CUDA_CHECK(cudaGetLastError());return;
    }
'''+old
assert s.count(old)==1;s=s.replace(old,new)
p.write_text(s)
