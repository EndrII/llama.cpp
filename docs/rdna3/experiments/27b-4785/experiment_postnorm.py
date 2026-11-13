import json,shutil
from pathlib import Path
r=Path('/home/svt/tuning27b50');s=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
p=s/'int4-wmma.cu';t=(r/'scale8.cu').read_text()
def change(old,new,n=1):
 global t
 assert t.count(old)==n,(old[:100],t.count(old));t=t.replace(old,new)
change('// Several output-row tiles share one fused activation rotation.','''struct mv_postnorm_dev {
    const float * weight = nullptr;
    float * output = nullptr;
    int8_t * q = nullptr;
    float2 * d = nullptr;
    unsigned int * counter = nullptr;
    float eps = 0.f;
};

// Several output-row tiles share one fused activation rotation.''')
change('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, bool POST = false>')
change('int K, int mode, int ystride, const float * __restrict__ norm, float eps) {','int K, int mode, int ystride, const float * __restrict__ norm, float eps, mv_postnorm_dev pn = {}) {')
old='''            out[(ft0 + ri) * 16 + r] = sum * (scales[ri] * inv) + res[ri];
        }
    }
}'''
new='''            out[(ft0 + ri) * 16 + r] = sum * (scales[ri] * inv) + res[ri];
        }
    }
    if (POST) {
        // Publish each writer's output before its block contributes to the completion counter.
        __threadfence();
        __syncthreads();
        __shared__ int last;
        if (wv == 0 && lane == 0) last = atomicInc(pn.counter, gridDim.x - 1) == gridDim.x - 1;
        __syncthreads();
        if (last) {
            constexpr int DIM = 5120, NSB = DIM / 256;
            __shared__ float part[NSB];
            for (int sb = wv; sb < NSB; sb += NW) {
                float ss = 0.f;
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float4 v = ((const float4 *) (out + sb * 256 + r * 16))[q];
                    ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
                }
                ss = wave32_sum(ss) * 0.5f;
                if (lane == 0) part[sb] = ss;
            }
            __syncthreads();
            float total = 0.f;
            for (int sb = 0; sb < NSB; sb++) total += part[sb];
            const float scale = rsqrtf(total / DIM + pn.eps);
            for (int sb = wv; sb < NSB; sb += NW) {
                float4 v[4];
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const float4 x = ((const float4 *) (out + sb * 256 + r * 16))[q];
                    const float4 w = ((const float4 *) (pn.weight + sb * 256 + r * 16))[q];
                    v[q] = make_float4(scale*x.x*w.x,scale*x.y*w.y,scale*x.z*w.z,scale*x.w*w.w);
                    if (h == 0) ((float4 *) (pn.output + sb * 256 + r * 16))[q] = v[q];
                }
                const float8v wr = rotate_regs(v, lane, rb1, ra2);
                quant_rot8(wr, lane, pn.q + sb * 256, pn.d + sb * 8);
            }
        }
    }
}'''
change(old,new)
change('const ggml_cuda_conv_fold * cf, const ggml_tensor * dual_w, float * dual_dst, const ggml_cuda_mv_jobs * jobs) {','const ggml_cuda_conv_fold * cf, const ggml_tensor * dual_w, float * dual_dst, const ggml_cuda_mv_jobs * jobs, const ggml_tensor * post_rms, const ggml_tensor * post_mul) {')
old='''                k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);'''
new='''                mv_postnorm_dev pn;
                if (post_rms) {
                    GGML_ASSERT(post_mul && N == 1 && ggml_nelements(post_mul) == 5120);
                    const ggml_tensor * wt = post_mul->src[0] == post_rms ? post_mul->src[1] : post_mul->src[0];
                    pn.weight = (const float *) wt->data; pn.output = (float *) post_mul->data;
                    memcpy(&pn.eps, post_rms->op_params, sizeof(float));
                    std::lock_guard<std::mutex> lk(g_mu);
                    MvCache & mc = g_mc[stream];
                    ac_reserve(mc.xq, mc.xq_cap, (size_t) 5120); ac_reserve(mc.xd, mc.xd_cap, (size_t) 160 * sizeof(float2));
                    mc.src1 = post_mul; mc.K = 5120; mc.N = 1; pn.q = mc.xq; pn.d = mc.xd;
                    static std::unordered_map<cudaStream_t, unsigned int *> counters;
                    auto & counter = counters[stream];
                    if (!counter) { CUDA_CHECK(cudaMalloc((void **) &counter, sizeof(unsigned int))); CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(unsigned int), stream)); }
                    pn.counter = counter;
                    k_mv_rot_rows<ROWS, 10, true><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps, pn);
                } else {
                    k_mv_rot_rows<ROWS><<<dim3(M / (16 * ROWS)), blk, 0, stream>>>(w.Wp, w.Wh, w.S, inv, glu_x ? glu_x : (const float *) src1->data, glu_y, (float *) dst->data, xb, K, mode, ystride, nw, eps);
                }'''
change(old,new)
change('float *, const ggml_cuda_mv_jobs *) {}','float *, const ggml_cuda_mv_jobs *, const ggml_tensor *, const ggml_tensor *) {}')
p.write_text(t)
p=s/'int4-wmma.cuh';original=r/'original/int4-wmma.cuh'
if not original.exists():shutil.copy2(p,original)
t=original.read_text().replace('const ggml_cuda_mv_jobs * jobs = nullptr);','const ggml_cuda_mv_jobs * jobs = nullptr, const ggml_tensor * post_rms = nullptr, const ggml_tensor * post_mul = nullptr);');p.write_text(t)
p=s/'ggml-cuda.cu';t=(r/'original/ggml-cuda.cu').read_text()
helper='''// Adjacent normalization after a 27B residual projection can be done by its last completed block.
static int ggml_cuda_int4_postnorm(const ggml_cgraph * cg, int add_i, const ggml_tensor ** rms, const ggml_tensor ** mul) {
    if (!getenv("GGML_INT4_POSTNORM") || !atoi(getenv("GGML_INT4_POSTNORM")) || getenv("GGML_INT4_NOCACHE") || add_i + 2 >= cg->n_nodes) return 0;
    const int rows = getenv("GGML_INT4_27B_ROWS") ? atoi(getenv("GGML_INT4_27B_ROWS")) : 4;
    if (rows != 2 && rows != 4) return 0;
    const ggml_tensor * add = cg->nodes[add_i], * nr = cg->nodes[add_i + 1], * nm = cg->nodes[add_i + 2];
    if (add->type != GGML_TYPE_F32 || add->ne[0] != 5120 || ggml_nelements(add) != 5120 || !ggml_is_contiguous(add) ||
        nr->op != GGML_OP_RMS_NORM || nr->src[0] != add || !ggml_cuda_can_fuse(cg, add_i + 1, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) return 0;
    const ggml_tensor * wt = nm->src[0] == nr ? nm->src[1] : nm->src[0];
    if (nm->type != GGML_TYPE_F32 || wt->type != GGML_TYPE_F32 || ggml_nelements(wt) != 5120 || !ggml_is_contiguous(wt) || !ggml_is_contiguous(nm) ||
        ((uintptr_t) add->data | (uintptr_t) wt->data | (uintptr_t) nm->data) % 16) return 0;
    for (int j = add_i + 3; j < cg->n_nodes && j < add_i + 30; j++) {
        const ggml_tensor * c = cg->nodes[j];
        if (c->op == GGML_OP_MUL_MAT && c->src[1] == nm && ggml_cuda_int4_converted(c->src[0])) { *rms = nr; *mul = nm; return 2; }
    }
    return 0;
}

'''
change('static int ggml_cuda_int4_residual_add(',helper+'static int ggml_cuda_int4_residual_add(')
old='''            ggml_cuda_int4_matvec(*cuda_ctx, mm->src[0], mm->src[1], r ? add : cgraph->nodes[i + extra], r ? &fusion_args : nullptr, xa, xb,
                                  hn_used ? 3 : 1, 256, hn.w, hn.eps);
            return extra + r;'''
new='''            const ggml_tensor * post_rms = nullptr, * post_mul = nullptr;
            const int pn = r && mm->src[1]->ne[1] == 1 && mm->src[0]->ne[1] == 5120 && (mm->src[0]->ne[0] == 6144 || mm->src[0]->ne[0] == 17408) ? ggml_cuda_int4_postnorm(cgraph, i + extra + r, &post_rms, &post_mul) : 0;
            ggml_cuda_int4_matvec(*cuda_ctx, mm->src[0], mm->src[1], r ? add : cgraph->nodes[i + extra], r ? &fusion_args : nullptr, xa, xb,
                                  hn_used ? 3 : 1, 256, hn.w, hn.eps, nullptr, nullptr, nullptr, nullptr, post_rms, post_mul);
            return extra + r + pn;'''
change(old,new)
old='''            ggml_cuda_int4_matvec(*cuda_ctx, node->src[0], node->src[1], add, &fusion_args);
            return r;'''
new='''            const ggml_tensor * post_rms = nullptr, * post_mul = nullptr;
            const int pn = node->src[1]->ne[1] == 1 && node->src[0]->ne[1] == 5120 && (node->src[0]->ne[0] == 6144 || node->src[0]->ne[0] == 17408) ? ggml_cuda_int4_postnorm(cgraph, i + r, &post_rms, &post_mul) : 0;
            ggml_cuda_int4_matvec(*cuda_ctx, node->src[0], node->src[1], add, &fusion_args, nullptr, nullptr, 1, 256, nullptr, 0.f, nullptr, nullptr, nullptr, nullptr, post_rms, post_mul);
            return r + pn;'''
change(old,new);p.write_text(t)
print('postnorm source prepared')
