from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next'); d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
for f in ['gated_delta_net.cu','gated_delta_net.cuh','int4-wmma.cuh']:(d/f).write_text((r/'original'/f).read_text())
(d/'ggml-cuda.cu').write_text((r/'batch_cuda_base.cu').read_text())
s=(r/'head_base.cu').read_text()
a=s.index('template <int NC, bool GATE'); b=s.index('// Several output-row tiles',a) if '// Several output-row tiles' in s[a:] else s.index('struct ActCache',a)
# Scope just the kernel to parameterize its wave count.
b=s.index('\n\n}',s.index('            } else {\n                dst[',a))+4
part=s[a:b]; assert part.count('MV_NW')>=5
part=part.replace('MV_NW','NW').replace('template <int NC,','template <int NW, int NC,')
s=s[:a]+part+s[b:]
s=s.replace('k_mv<','k_mv<10,')
a=s.index('    if (jobs) {',s.index('void ggml_cuda_int4_matvec'))
b=s.index('    if (N == 8 && gate)',a)
body=s[a:b].replace('k_mv<10,','k_mv<NW,')
insert='''    if (N == 1 && !rot) {
        const auto launch_single = [&](auto waves_c) {
            constexpr int NW = decltype(waves_c)::value;
            const dim3 grid(M / 16), blk(32, NW);
'''+body+'''
            if (gate) k_mv<NW, 1, true><<<grid, blk, 0, stream>>>(w.Wp,w.Wh,w.S,inv,g.Wp,g.Wh,g.S,inv2,xq_ptr,xd_ptr,(float*)dst->data,xb,gb,glu,M,K,stride_dst);
            else k_mv<NW, 1, false><<<grid, blk, 0, stream>>>(w.Wp,w.Wh,w.S,inv,nullptr,nullptr,nullptr,0.f,xq_ptr,xd_ptr,(float*)dst->data,xb,gb,glu,M,K,stride_dst);
            CUDA_CHECK(cudaGetLastError());
        };
        const char * opt=getenv(gate ? "GGML_MV_WAVES_GATE" : "GGML_MV_WAVES_OTHER");
        if (!opt) opt=getenv("GGML_MV_WAVES");
        const int nw = opt ? atoi(opt) : 10;
        if (nw == 5) launch_single(std::integral_constant<int,5>{});
        else if (nw == 8) launch_single(std::integral_constant<int,8>{});
        else if (nw == 20) launch_single(std::integral_constant<int,20>{});
        else launch_single(std::integral_constant<int,10>{});
        return;
    }
'''
s=s[:a]+insert+s[a:]
old='''                k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
new='''                const char * opt = getenv(K == 6144 ? "GGML_MV_ROT_WAVES_GDN" : "GGML_MV_ROT_WAVES_DOWN");
                const int waves = opt ? atoi(opt) : 10;
#define RUN_R(NW) k_mv_rot_rows<ROWS,NW><<<dim3(M/(16*ROWS)),dim3(32,NW),0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps)
                if(waves==8){RUN_R(8);}
                else if(waves==12){RUN_R(12);}
                else if(waves==17){RUN_R(17);}
                else if(waves==20){RUN_R(20);}
                else{RUN_R(10);}
#undef RUN_R'''
assert s.count(old)==1;s=s.replace(old,new)
(d/'int4-wmma.cu').write_text(s); (r/'nwaves.cu').write_text(s)
