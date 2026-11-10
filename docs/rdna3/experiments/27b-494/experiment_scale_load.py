from pathlib import Path
import json,sys
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'selected_int4.cu').read_text();policy=int(sys.argv[1]);assert policy in [0,4,6]
idx=s.index('#define INT4W_LO')
s=s[:idx]+f'''__device__ __forceinline__ uint32_t raw_scale16(const uint8_t * p,size_t offset) {{
    const uint32_t bits=__builtin_amdgcn_raw_buffer_load_b32(INT4W_RSRC(p),(int)(offset&~(size_t)3),0,{policy});
    return (bits>>((offset&2)*8))&0xffff;
}}
'''+s[idx:]
for name in ['Wh','Wh2']:
 old=f'*(const uint16_t *) ({name} + so)';assert s.count(old)>0;s=s.replace(old,f'raw_scale16({name},so)')
(d/'int4-wmma.cu').write_text(s);(r/f'scale_load_{policy}.cu').write_text(s)
