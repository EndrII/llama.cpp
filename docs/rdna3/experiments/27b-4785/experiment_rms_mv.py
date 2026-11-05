import json
from pathlib import Path
r=Path('/home/svt/tuning27b50');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
t=(r/'scale4.cu').read_text()
def sub(old,new,n=1):
 global t
 assert t.count(old)==n,(old[:70],t.count(old));t=t.replace(old,new)
sub('if (mode == 2) {','''if (mode == 4) {
                v[q].x = eps * v[q].x * y[q].x; v[q].y = eps * v[q].y * y[q].y;
                v[q].z = eps * v[q].z * y[q].z; v[q].w = eps * v[q].w * y[q].w;
            } else if (mode == 2) {''')
sub('bool MJ = false>','bool MJ = false, bool RMS = false>')
sub('''    // ROT: the activation is rotated + quantized here, per wave and per 256-block, instead of by a separate kernel (NC == 1)''','''    if (RMS) {
        __shared__ float part[32];
        for (int sb = wv; sb < nsb; sb += MV_NW) {
            float ss = 0.f;
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const float4 x = ((const float4 *) (Xf + sb * 256 + r * 16))[q];
                ss += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
            }
            ss = wave32_sum(ss) * 0.5f;
            if (lane == 0) part[sb] = ss;
        }
        __syncthreads();
        float ss = 0.f;
        for (int sb = 0; sb < nsb; sb++) ss += part[sb];
        neps = rsqrtf(ss / K + neps);
    }
    // ROT: the activation is rotated + quantized here, per wave and per 256-block, instead of by a separate kernel (NC == 1)''')
sub('std::unordered_map<cudaStream_t, MvCache> g_mc;','''std::unordered_map<cudaStream_t, MvCache> g_mc;
struct DeferredNorm { const float * x; const float * w; float eps; };
std::unordered_map<const ggml_tensor *, DeferredNorm> g_norm;''')
sub('for (auto & kv : g_mc) kv.second.src1 = nullptr;','for (auto & kv : g_mc) kv.second.src1 = nullptr; g_norm.clear();')
sub('''    cudaStream_t stream = ctx.stream();
    const Int4Weights w = get_weights(stream, src0);
    GGML_ASSERT(w.inplace);''','''    cudaStream_t stream = ctx.stream();
    const Int4Weights w = get_weights(stream, src0);
    GGML_ASSERT(w.inplace);''')
sub('''    // single-consumer inputs (the ones whose producers are not shared with another matvec): rotate + quantize inside the matvec''','''    auto nf = g_norm.find(src1);
    if (nf != g_norm.end()) {
        GGML_ASSERT(N == 1 && K == 5120 && gate);
        const auto & n = nf->second;
        k_mv<1, true, true, false, false, false, true><<<dim3(M / 16), dim3(32, MV_NW), 0, stream>>>(w.Wp, w.Wh, w.S, 1.f / (1 << w.sh), g.Wp, g.Wh, g.S, 1.f / (1 << g.sh), nullptr, nullptr, (float *) dst->data, xb, gb, glu, M, K, M, n.x, n.w, 4, 256, nullptr, n.eps);
        CUDA_CHECK(cudaGetLastError()); return;
    }
    // single-consumer inputs (the ones whose producers are not shared with another matvec): rotate + quantize inside the matvec''')
sub('bool ggml_cuda_int4_rmsnorm_rot(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul) {','''bool ggml_cuda_int4_defer_rmsnorm(const ggml_tensor * rms, const ggml_tensor * mul) {
    const ggml_tensor * x = rms->src[0], * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    if (ggml_nelements(x) != 5120 || x->ne[0] != 5120 || !ggml_is_contiguous(x) || !ggml_is_contiguous(mul) || !ggml_is_contiguous(w) || ggml_nelements(w) != 5120 ||
        x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || (((uintptr_t) x->data | (uintptr_t) w->data) % 16)) return false;
    float eps; memcpy(&eps, rms->op_params, sizeof(float));
    std::lock_guard<std::mutex> lk(g_mu); g_norm[mul] = {(const float *) x->data, (const float *) w->data, eps}; return true;
}

bool ggml_cuda_int4_rmsnorm_rot(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul) {''')
sub('bool ggml_cuda_int4_rmsnorm_rot(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *) { return false; }','''bool ggml_cuda_int4_defer_rmsnorm(const ggml_tensor *, const ggml_tensor *) { return false; }
bool ggml_cuda_int4_rmsnorm_rot(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *) { return false; }''')
(d/'int4-wmma.cu').write_text(t)
t=(r/'original/int4-wmma.cuh').read_text();t+='\nbool ggml_cuda_int4_defer_rmsnorm(const ggml_tensor * rms, const ggml_tensor * mul);\n';(d/'int4-wmma.cuh').write_text(t)
t=(r/'original/ggml-cuda.cu').read_text()
old='''        if (feeds_int4 && ggml_cuda_int4_rmsnorm_rot(*cuda_ctx, node, mulx)) {'''
new='''        if (feeds_int4 && getenv("GGML_INT4_RMS_MV") && atoi(getenv("GGML_INT4_RMS_MV"))) {
            bool ffn = false;
            for (int j = i + 2; j < cgraph->n_nodes && j < i + 6; j++) {
                const ggml_tensor * c = cgraph->nodes[j];
                if (c->op == GGML_OP_MUL_MAT && c->src[1] == mulx && strstr(c->src[0]->name, "ffn_gate")) ffn = true;
            }
            if (ffn && ggml_cuda_int4_defer_rmsnorm(node, mulx)) return 1;
        }
        if (feeds_int4 && ggml_cuda_int4_rmsnorm_rot(*cuda_ctx, node, mulx)) {'''
assert t.count(old)==1;(d/'ggml-cuda.cu').write_text(t.replace(old,new))
