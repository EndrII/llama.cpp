from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
# Restore the producer fusion, then add the independently validated input batch extension.
for f in ['gated_delta_net.cu','gated_delta_net.cuh','ggml-cuda.cu']:(d/f).write_text((r/('gdn_out_'+f)).read_text())
s=(d/'ggml-cuda.cu').read_text();batch=(r/'batch_cuda_base.cu').read_text()
a=batch.index('struct ggml_cuda_input_copy');b=batch.index('static void ggml_backend_cuda_buffer_set_tensor(',a);helper=batch[a:b]
s=s.replace('static void ggml_backend_cuda_buffer_set_tensor(',helper+'static void ggml_backend_cuda_buffer_set_tensor(',1)
a=batch.index('static void ggml_backend_cuda_buffer_set_tensor(');b=batch.index('\nstatic void ggml_backend_cuda_buffer_get_tensor(',a)
a2=s.index('static void ggml_backend_cuda_buffer_set_tensor(');b2=s.index('\nstatic void ggml_backend_cuda_buffer_get_tensor(',a2);s=s[:a2]+batch[a:b]+s[b2:]
key='static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {';s=s.replace(key,key+'\n    ggml_cuda_input_batch_flush();')
key='static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {'
s=s.replace(key,key+'''
    if (strcmp(name, "ggml_backend_cuda_input_batch_begin") == 0) return (void *)ggml_backend_cuda_input_batch_begin;
    if (strcmp(name, "ggml_backend_cuda_input_batch_end") == 0) return (void *)ggml_backend_cuda_input_batch_end;''')
(d/'ggml-cuda.cu').write_text(s)
(d/'int4-wmma.cuh').write_text((r/'original/int4-wmma.cuh').read_text())
s=(r/'head_base.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:90],s.count(old));s=s.replace(old,new)
# Make wave-uniform indices explicit; every launch has x=32.
sub('wv = threadIdx.y','wv = __builtin_amdgcn_readfirstlane((int) threadIdx.y)',2)
sub('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, int PREF = ROWS, bool RAW = false>')
old='''        for (int ri = 0; ri < ROWS; ri++) {
            const size_t wo = ((size_t) ((ft0 + ri) * nsb + sb) * 8 + h * 4) * 256 + r * 16;
            const size_t so = ((size_t) ((ft0 + ri) * nsb + sb) * 16 + r) * 4 + h * 2;
#pragma unroll
            for (int g = 0; g < 4; g++) { weights[ri][g] = nt128(Wp, wo + g * 256); }
            subscales[ri] = *(const uint16_t *) (Wh + so);
        }'''
sub(old,old.replace('ri < ROWS','ri < PREF'))
sub('''Y ? Y + (size_t) sb * ystride : nullptr, mode, norm, eps);''','''RAW ? nullptr : (Y ? Y + (size_t) sb * ystride : nullptr), mode, norm, eps);''')
key='''        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {'''
new='''        dv[0] = xd[0]; dv[1] = xd[1];
        if (PREF < ROWS) {
            asm volatile("" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
'''+old.replace('int ri = 0','int ri = PREF')+'''        }
#pragma unroll
        for (int ri = 0; ri < ROWS; ri++) {'''
sub(key,new)
old='''                k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
new='''                const int variant = getenv("GGML_MV_PREFETCH") ? atoi(getenv("GGML_MV_PREFETCH")) : 0;
#define RUN_P(NW,PF,RAW) k_mv_rot_rows<ROWS,NW,PF,RAW><<<dim3(M/(16*ROWS)),dim3(32,NW),0,stream>>>(w.Wp,w.Wh,w.S,inv,glu_x?glu_x:(const float*)src1->data,glu_y,(float*)dst->data,xb,K,mode,ystride,nw,eps)
                if (!glu_y && variant == 1) { RUN_P(8,0,true); }
                else if (!glu_y && variant == 2) { RUN_P(8,1,true); }
                else if (!glu_y && variant == 3) { RUN_P(10,0,true); }
                else if (!glu_y && variant == 4) { RUN_P(10,1,true); }
                else if (!glu_y && variant == 5) { RUN_P(8,ROWS,true); }
                else if (!glu_y && variant == 6) { RUN_P(10,ROWS,true); }
                else { RUN_P(10,ROWS,false); }
#undef RUN_P'''
sub(old,new)
(d/'int4-wmma.cu').write_text(s);(r/'prefetch.cu').write_text(s)
