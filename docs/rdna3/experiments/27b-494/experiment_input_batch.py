from pathlib import Path
import json,shutil
r=Path('/home/svt/tuning27b-next');src=Path(json.loads((r/'paths.json').read_text())['source']);d=src/'ggml/src/ggml-cuda'
p=src/'src/llama-context.cpp'
if not (r/'original/llama-context.cpp').exists():shutil.copy2(p,r/'original/llama-context.cpp')
s=(r/'original/llama-context.cpp').read_text()
helper='''// Opt-in backend extension: keep source storage alive while a decode input
// transaction uploads its tensors, then synchronize once before graph compute.
class llama_input_batch_scope {
    using hook = void (*)();
    using hooks = std::vector<std::pair<hook, hook>>;
    const hooks * active = nullptr;
public:
    explicit llama_input_batch_scope(bool decode) {
        static const bool requested = [] { const char * e = std::getenv("GGML_CUDA_BATCH_INPUTS"); return e && std::atoi(e); }();
        if (!decode || !requested) return;
        static const hooks available = [] {
            hooks result;
            for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
                const auto reg = ggml_backend_reg_get(i);
                const auto begin = reinterpret_cast<hook>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_input_batch_begin"));
                const auto end = reinterpret_cast<hook>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_input_batch_end"));
                if (begin && end) result.emplace_back(begin, end);
            }
            return result;
        }();
        active = &available;
        for (const auto & h : *active) h.first();
    }
    ~llama_input_batch_scope() {
        if (active) for (auto i = active->rbegin(); i != active->rend(); ++i) i->second();
    }
    llama_input_batch_scope(const llama_input_batch_scope &) = delete;
    llama_input_batch_scope & operator=(const llama_input_batch_scope &) = delete;
};

'''
s=s.replace('#include <cinttypes>','#include <cinttypes>\n#include <cstdlib>')
s=s.replace('static llm_graph_type ctx_type_to_graph_type',helper+'static llm_graph_type ctx_type_to_graph_type')
old='''        res->set_inputs(&ubatch);''';new='''        llama_input_batch_scope input_batch(ubatch.n_tokens == 1);
        res->set_inputs(&ubatch);'''
assert s.count(old)==2;s=s.replace(old,new,1);p.write_text(s)
s=(r/'gdn_out_ggml-cuda.cu').read_text()
helper='''struct ggml_cuda_input_copy { int device; std::vector<uint8_t> bytes; };
static thread_local int ggml_cuda_input_batch_depth = 0;
static thread_local std::vector<ggml_cuda_input_copy> ggml_cuda_input_copies;
static void ggml_cuda_input_batch_flush() {
    for (size_t i = 0; i < ggml_cuda_input_copies.size(); i++) {
        bool seen = false;
        for (size_t j = 0; j < i; j++) seen |= ggml_cuda_input_copies[j].device == ggml_cuda_input_copies[i].device;
        if (!seen) {
            ggml_cuda_set_device(ggml_cuda_input_copies[i].device);
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    ggml_cuda_input_copies.clear();
}
static void ggml_backend_cuda_input_batch_begin() { ++ggml_cuda_input_batch_depth; }
static void ggml_backend_cuda_input_batch_end() {
    GGML_ASSERT(ggml_cuda_input_batch_depth > 0);
    if (--ggml_cuda_input_batch_depth == 0) ggml_cuda_input_batch_flush();
}

'''
key='static void ggml_backend_cuda_buffer_set_tensor('
assert s.count(key)==1;s=s.replace(key,helper+key)
old='''    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));'''
new='''    const bool batched = ggml_cuda_input_batch_depth > 0 && size > 0 && size <= 65536 && (tensor->flags & GGML_TENSOR_FLAG_INPUT);
    const void * source = data;
    if (batched) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        ggml_cuda_input_copies.push_back({ctx->device, std::vector<uint8_t>(bytes, bytes + size)});
        source = ggml_cuda_input_copies.back().bytes.data();
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, source, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    if (!batched) CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));'''
assert s.count(old)==1;s=s.replace(old,new)
key='''static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {'''
assert s.count(key)==1;s=s.replace(key,key+'\n    ggml_cuda_input_batch_flush();')
key='''static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {'''
assert s.count(key)==1;s=s.replace(key,key+'''
    if (strcmp(name, "ggml_backend_cuda_input_batch_begin") == 0) return (void *)ggml_backend_cuda_input_batch_begin;
    if (strcmp(name, "ggml_backend_cuda_input_batch_end") == 0) return (void *)ggml_backend_cuda_input_batch_end;''')
(d/'ggml-cuda.cu').write_text(s)
# New frontend code requires its shared library as well as the HIP backend.
p=r/'build.py';s=p.read_text()
if 'libllama.so' not in s:
 s+='''\nfrontend = (Path(paths['build'])/'bin/libllama.so').resolve()
dest = root/'bin'/frontend.name
if dest.is_symlink(): dest.unlink()
shutil.copy2(frontend, dest)
print('COPIED', frontend.name, flush=True)
''';p.write_text(s)
print('input transaction source prepared')
