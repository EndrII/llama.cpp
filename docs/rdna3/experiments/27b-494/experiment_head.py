from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
# Use the raw-specialization file and keep the producer fusion independently selectable.
s=(r/'raw.cu').read_text()
s=s.replace('#include <cstdlib>','#include <cstdlib>\n#include <algorithm>')
s=s.replace('struct Int4Weights {',(r/'head_requant.txt').read_text()+'\nstruct Int4Weights {')
old='''        const ggml_tensor * w = n->src[0];
        if (!ggml_cuda_int4_rotated(w)'''
new='''        const ggml_tensor * w = n->src[0];
        maybe_requant_head(ctx.stream(), w);
        if (!ggml_cuda_int4_rotated(w)'''
assert s.count(old)==1;s=s.replace(old,new)
(d/'int4-wmma.cu').write_text(s)
