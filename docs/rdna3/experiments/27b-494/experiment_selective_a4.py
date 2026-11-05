from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'selected_int4.cu').read_text()
# Reuse the exact DPP packing permutation from the previously checked native DOT8 path.
dot=(r/'dot8.cu').read_text();a=dot.index('template <bool DOT8 = false>\n__device__ __forceinline__ void quant_rot8');b=dot.index('    if (DOT8) {',a)
prefix=dot[a:b].replace('template <bool DOT8 = false>','template <int CLIP>').replace('void quant_rot8','void quant_rot4').replace('d[i] = am[i] / 127.0f;','d[i] = am[i] * ((float)CLIP / 700.f);').replace('(int) roundf(x[i] / d[i])','max(-8,min(7,(int)roundf(x[i]/d[i])))')
a=b+len('    if (DOT8) {');b=dot.index('    } else {',a)
pack=dot[a:b]
pack=pack.replace('            uint32_t hi = ((uint32_t) (q[i] >> 4) & 15u) << shift;\n','')
pack='\n'.join(line for line in pack.split('\n') if 'hi |=' not in line and '= hi;' not in line)
helper=prefix+pack+'}\n\n'
pos=s.index('// Several output-row tiles share one fused activation rotation.');s=s[:pos]+helper+s[pos:]
s=s.replace('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, int A4 = 0>')
# Only change quantization inside this output kernel; other producers retain Q8 activations.
a=s.index('__global__ __launch_bounds__(32 * NW) void k_mv_rot_rows');b=s.index('\n\n} // namespace',a)
part=s[a:b]
old='''        quant_rot8(wr, lane, &rq[wv][0], &rd[wv][0]);''';assert part.count(old)==1
part=part.replace(old,'''        if constexpr(A4) quant_rot4<A4>(wr,lane,&rq[wv][0],&rd[wv][0]);
        else quant_rot8(wr,lane,&rq[wv][0],&rd[wv][0]);''')
a2=part.index('                int si = 0;');b2=part.index('\n            }\n        }',a2)
orig=part[a2:b2]
new='''                if constexpr(A4){
                    int si=0;
#pragma unroll
                    for(int wi=0;wi<4;wi++)si=__builtin_amdgcn_sudot8(true,(int)ww[wi],true,(int)xw[wi],si,false);
                    acc[ri]+=(float)((subscales[ri]>>(4*g))&15)*(ds.x*(float)si);
                }else{
'''+orig+'''
                }'''
part=part[:a2]+new+part[b2:];s=s[:a]+part+s[b:]
old='''                k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
new='''                const char * selected=getenv("GGML_INT4_DECODE_A4");
                const bool a4=selected&&(!strcmp(selected,"both")||(K==6144&&!strcmp(selected,"gdn"))||(K==17408&&!strcmp(selected,"down")));
                const int clip=getenv("GGML_INT4_DECODE_CLIP")?atoi(getenv("GGML_INT4_DECODE_CLIP")):85;
#define RUN_A4(C) k_mv_rot_rows<ROWS,10,C><<<dim3(M/(16*ROWS)),blk,0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps)
                if(a4&&clip==100){RUN_A4(100);}
                else if(a4){RUN_A4(85);}
                else{RUN_A4(0);}
#undef RUN_A4'''
assert s.count(old)==1;s=s.replace(old,new)
(d/'int4-wmma.cu').write_text(s);(r/'selective_a4.cu').write_text(s)
