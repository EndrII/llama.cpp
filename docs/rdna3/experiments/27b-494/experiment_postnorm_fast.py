from pathlib import Path
import json,shutil
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
keep=r/'best_493';(keep/'bin').mkdir(parents=True,exist_ok=True);(keep/'src').mkdir(exist_ok=True)
for p in (r/'bin').iterdir():
 q=keep/'bin'/p.name
 if q.exists() or q.is_symlink():continue
 if p.name in ['libggml-hip.so.0.18.0','libllama.so.0.0.0','libggml-base.so.0.18.0']:shutil.copy2(p,q)
 elif p.is_symlink() and not p.readlink().is_absolute():q.symlink_to(p.readlink())
 else:q.symlink_to(p.resolve())
for f in ['int4-wmma.cu','int4-wmma.cuh','ggml-cuda.cu','gated_delta_net.cu','gated_delta_net.cuh']:shutil.copy2(d/f,keep/'src'/f)
shutil.copy2(d.parent/'ggml-backend.cpp',keep/'src/ggml-backend.cpp')
# Keep input batching, but remove the discarded GDN producer experiment.
s=(d/'ggml-cuda.cu').read_text();a=s.index('            if (getenv("GGML_GDN_OUT_FOLD")');b=s.index('            ggml_cuda_op_gated_delta_net_fused_cache',a);s=s[:a]+s[b:]
s=s.replace('            if (hn_used && !hn.w) { xa = hn.x; xb = nullptr; }\n','').replace('hn_used && hn.w ? 3 : 1','hn_used ? 3 : 1')
(r/'batch_cuda_base.cu').write_text(s)
# Start with the selected compact-scale kernel and the validated IQ4_NL head.
s=(r/'original/int4-wmma.cu').read_text().replace('#include <cstdlib>','#include <cstdlib>\n#include <algorithm>')
s=s.replace('struct Int4Weights {',(r/'head_requant.txt').read_text()+'\nstruct Int4Weights {')
old='''        const ggml_tensor * w = n->src[0];
        if (!ggml_cuda_int4_rotated(w)'''
new='''        const ggml_tensor * w = n->src[0];
        maybe_requant_head(ctx.stream(), w);
        if (!ggml_cuda_int4_rotated(w)'''
assert s.count(old)==1;s=s.replace(old,new);(r/'head_base.cu').write_text(s)
script=Path('/home/svt/tuning27b50/experiment_postnorm.py').read_text().replace('/home/svt/tuning27b50','/home/svt/tuning27b-next')
script=script.replace("(r/'scale8.cu').read_text()","(r/'head_base.cu').read_text()")
script=script.replace("(r/'original/ggml-cuda.cu').read_text()","(r/'batch_cuda_base.cu').read_text()")
exec(compile(script,str(r/'adapted_postnorm.py'),'exec'))
p=d/'int4-wmma.cu';s=p.read_text()
old='''        __threadfence();
        __syncthreads();
        __shared__ int last;'''
new='''        // Only the wave that wrote output rows needs to publish stores.
        if (wv == 0 && h == 0) __threadfence();
        __syncthreads();
        __shared__ int last;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('''        if (last) {
            constexpr int DIM''','''        if (last) {
            // All reader waves acquire the output after the final arrival.
            __threadfence();
            constexpr int DIM''')
p.write_text(s)
