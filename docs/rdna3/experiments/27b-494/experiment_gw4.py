from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next'); d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'nwaves.cu').read_text()
# Experiment intentionally validates every tensor's group size before using the tighter layout.
key='''    k_repack<<<dim3(M, nsb), 8, 0, stream>>>'''
assert s.count(key)==1;s=s.replace(key,'''    GGML_ASSERT(w.gw == 4);
'''+key)
a=s.index('    if ((g & 1) == 0) {',s.index('__global__ void k_repack'))
b=s.index('\n    }',a)+6
old=s[a:b]
new='''    if (g == 0) {
        const float d4 = fabsf(__half2float(W[(size_t)m*nb+sb*8+4].d));
        const int q0 = min(15,max(0,(int)lrintf(fabsf(d)/s*(float)(1<<sh))));
        const int q4 = min(15,max(0,(int)lrintf(d4/s*(float)(1<<sh))));
        Wh[(size_t)(ft*nsb+sb)*16+r] = q0 | (q4<<4);
    }'''
s=s[:a]+new+s[b:]
key='#define INT4W_LO(v)'
idx=s.index(key)
s=s[:idx]+'''// Exact packing for weights whose scale groups are 128 elements.
__device__ __forceinline__ uint32_t load_gw4_scales(const uint8_t * p, size_t index) {
    const uint32_t v=p[index];
    return (v&15)*0x1111u + (v>>4)*0x11110000u;
}
__device__ __forceinline__ uint32_t load_gw4_half(const uint8_t * p,size_t old_offset,int half) {
    return ((p[old_offset/4]>>(half*4))&15)*0x1111u;
}
'''+s[idx:]
old='''*(const uint32_t *) (Wh + (((ft0 + mi) * nsb + sb) * 16 + lc) * 4)'''
assert s.count(old)==3;s=s.replace(old,'''load_gw4_scales(Wh,((ft0+mi)*nsb+sb)*16+lc)''')
old='''*(const uint16_t *) (Wh + so)''';assert s.count(old)==2;s=s.replace(old,'''load_gw4_half(Wh,so,h)''')
old='''*(const uint16_t *) (Wh2 + so)''';assert s.count(old)==1;s=s.replace(old,'''load_gw4_half(Wh2,so,h)''')
(d/'int4-wmma.cu').write_text(s); (r/'gw4.cu').write_text(s)
