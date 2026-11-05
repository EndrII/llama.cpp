from pathlib import Path
import json,shutil
r=Path('/home/svt/tuning27b-next');src=Path(json.loads((r/'paths.json').read_text())['source']);d=src/'ggml/src/ggml-cuda'
# Input copies happen in the scheduler, after llama's CPU-side input setters.
(src/'src/llama-context.cpp').write_text((r/'original/llama-context.cpp').read_text())
p=src/'ggml/src/ggml-backend.cpp'
if not (r/'original/ggml-backend.cpp').exists():shutil.copy2(p,r/'original/ggml-backend.cpp')
s=(r/'original/ggml-backend.cpp').read_text()
context=(r/'original/llama-context.cpp').read_text()
# Reuse the RAII extension scope already compiled successfully in the frontend.
current_helper=Path('/home/svt/tuning27b-next/input_scope_helper.txt')
if not current_helper.exists():
 # The tested helper's source is generated literally by the prior script.
 code=(r/'experiment_input_batch.py').read_text();a=code.index("helper='''")+len("helper='''");b=code.index("'''",a)
 current_helper.write_text(code[a:b])
helper=current_helper.read_text().replace('llama_input_batch_scope','ggml_input_batch_scope')
s=s.replace('#include "ggml-backend-impl.h"','#include "ggml-backend-impl.h"\n#include <cstdlib>')
key='static enum ggml_status ggml_backend_sched_compute_splits('
assert s.count(key)==1;s=s.replace(key,helper+key)
old='''        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {'''
new='''        // Preserve user-input lifetimes and complete the upload transaction
        // before launching the split. Restrict the extension to small host inputs.
        {
        bool batchable = split->n_inputs > 0;
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            const ggml_tensor * t = split->inputs[input_id];
            const auto buffer = t->view_src ? t->view_src->buffer : t->buffer;
            batchable &= buffer && ggml_backend_buffer_is_host(buffer) && ggml_nbytes(t) <= 65536;
        }
        ggml_input_batch_scope upload_transaction(batchable);
        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {'''
assert s.count(old)==1;s=s.replace(old,new)
old='''        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async'''
new='''        }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
p=d/'ggml-cuda.cu';s=p.read_text().replace('size <= 65536 && (tensor->flags & GGML_TENSOR_FLAG_INPUT)','size <= 65536')
p.write_text(s)
p=r/'build.py';s=p.read_text()
if "'bin/libggml-base.so'" not in s:
 s+='''\nbase = (Path(paths['build'])/'bin/libggml-base.so').resolve()
dest = root/'bin'/base.name
if dest.is_symlink(): dest.unlink()
shutil.copy2(base, dest)
print('COPIED', base.name, flush=True)
''';p.write_text(s)
