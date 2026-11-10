from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'rows8.cu').read_text();a=s.index('__global__ __launch_bounds__(32 * NW) void k_mv_rot_rows');b=s.index('\n\n} // namespace',a);p=s[a:b]
p=p.replace('const int nsb = K / 256, ft0 = blockIdx.x * ROWS;','const int nsb = K / 256, ft0 = blockIdx.x * ROWS;\n    constexpr int CHUNK = ROWS > 4 ? 4 : ROWS;')
p=p.replace('uint4 weights[ROWS][4]','uint4 weights[CHUNK][4]').replace('uint32_t subscales[ROWS]','uint32_t subscales[CHUNK]')
key='''        for (int ri = 0; ri < ROWS; ri++) {
            const size_t wo''';assert p.count(key)==1;p=p.replace(key,key.replace('ri < ROWS','ri < CHUNK'))
a2=p.index('''#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {''',p.index('dv[0] = xd[0];'))
b2=p.index('\n    }\n#pragma unroll',a2)
body=p[a2:b2].replace('ri < ROWS','ri < CHUNK').replace('acc[ri]','acc[batch+ri]')
outer='''        uint32_t dependency=0;
#pragma unroll
        for(int batch=0;batch<ROWS;batch+=CHUNK){
            if(batch){
#pragma unroll
                for(int ri=0;ri<CHUNK;ri++){
                    const size_t wo=((size_t)((ft0+batch+ri)*nsb+sb)*8+h*4)*256+r*16+dependency;
                    const size_t so=((size_t)((ft0+batch+ri)*nsb+sb)*16+r)*4+h*2+dependency;
#pragma unroll
                    for(int g=0;g<4;g++)weights[ri][g]=nt128(Wp,wo+g*256);
                    subscales[ri]=*(const uint16_t*)(Wh+so);
                }
            }
'''+body+'''
            if constexpr(ROWS>4){
                // The next four rows reuse these registers after the current dots finish.
                asm volatile("v_mov_b32 %0, 0" : "=v"(dependency) : "v"(acc[batch]));
                __builtin_amdgcn_sched_barrier(0);
            }
        }'''
p=p[:a2]+outer+p[b2:];s=s[:a]+p+s[b:]
(d/'int4-wmma.cu').write_text(s);(r/'rows8_chunk.cu').write_text(s)
