from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'raw.cu').read_text()
s=s.replace('// Several output-row tiles share one fused activation rotation.',(r/'fht.txt').read_text()+'\n// Several output-row tiles share one fused activation rotation.')
s=s.replace('bool RAW = false, bool LATE = false>','bool RAW = false, bool LATE = false, bool FHT = false>')
old='''const float8v wr = rotate_block(X + sb * 256, lane, rb1, ra2, RAW ? nullptr : (Y ? Y + (size_t) sb * ystride : nullptr), mode, norm, eps);'''
new='''const float8v wr = FHT ? rotate_fht(X + sb * 256, lane) : rotate_block(X + sb * 256, lane, rb1, ra2, RAW ? nullptr : (Y ? Y + (size_t) sb * ystride : nullptr), mode, norm, eps);'''
assert s.count(old)==1;s=s.replace(old,new)
a=s.index('                const int specialize = getenv("GGML_MV_RAW_SPECIALIZE")');b=s.index('\n            };',a)
s=s[:a]+'''                const int specialize = getenv("GGML_MV_RAW_SPECIALIZE") ? atoi(getenv("GGML_MV_RAW_SPECIALIZE")) : 0;
                const bool active = !glu_y && ((specialize != 3 && specialize != 4 && specialize != 6 && specialize != 8) || K == 6144);
#define RUN_ROT(RAW,LATE,FHT) k_mv_rot_rows<ROWS,10,RAW,LATE,FHT><<<dim3(M/(16*ROWS)),blk,0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps)
                if (active && (specialize == 1 || specialize == 3)) { RUN_ROT(true,false,false); }
                else if (active && (specialize == 2 || specialize == 4)) { RUN_ROT(true,true,false); }
                else if (active && (specialize == 5 || specialize == 6)) { RUN_ROT(true,false,true); }
                else if (active && (specialize == 7 || specialize == 8)) { RUN_ROT(true,true,true); }
                else { RUN_ROT(false,false,false); }
#undef RUN_ROT'''+s[b:]
(r/'fht.cu').write_text(s)
# Add the separately selectable head experiment in the same build.
s=s.replace('#include <cstdlib>','#include <cstdlib>\n#include <algorithm>')
s=s.replace('struct Int4Weights {',(r/'head_requant.txt').read_text()+'\nstruct Int4Weights {')
old='''        const ggml_tensor * w = n->src[0];
        if (!ggml_cuda_int4_rotated(w)'''
new='''        const ggml_tensor * w = n->src[0];
        maybe_requant_head(ctx.stream(), w);
        if (!ggml_cuda_int4_rotated(w)'''
assert s.count(old)==1;s=s.replace(old,new)
(d/'int4-wmma.cu').write_text(s)
