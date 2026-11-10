from pathlib import Path
import json,shutil
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
# Restore the selected 47.85 source before applying this independent experiment.
(d/'int4-wmma.cu').write_text((r/'original/int4-wmma.cu').read_text())
p=d/'gated_delta_net.cuh'
if not (r/'original/gated_delta_net.cuh').exists():shutil.copy2(p,r/'original/gated_delta_net.cuh')
p.write_text((r/'original/gated_delta_net.cuh').read_text()+'\nvoid ggml_cuda_gdn_fold_out(const ggml_tensor * gdn, const float * norm, const float * z, float eps);\n')
s=(r/'original/gated_delta_net.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:90],s.count(old));s=s.replace(old,new)
sub('#include "ggml-cuda/common.cuh"','#include "ggml-cuda/common.cuh"\n#include "unary.cuh"\nstruct gdn_out_args { const float * norm = nullptr; const float * z = nullptr; float eps = 0.f; };')
sub('''template <bool keep_rs_t>
__global__ void __launch_bounds__(128, 2)
gated_delta_net_cols''','''template <bool keep_rs_t, bool POST = false>
__global__ void __launch_bounds__((POST ? 256 : 128), 2)
gated_delta_net_cols''')
sub('float l2eps, const gdn_ab_args ab) {','float l2eps, const gdn_ab_args ab, const gdn_out_args pn = {}) {')
sub('constexpr int S_v = 128, R = 64;','constexpr int S_v = 128, R = 64, NW = POST ? 8 : 4;')
sub('blockIdx.z * 64 + threadIdx.y * 16 + lc','blockIdx.z * (16 * NW) + threadIdx.y * 16 + lc')
sub('float ks[4][2][S_v]','float ks[NW][2][S_v]')
sub('float qs[4][2][S_v]','float qs[NW][2][S_v]')
sub('const int tid = threadIdx.y * 32 + lane;','const int tid = (threadIdx.y & 3) * 32 + lane;')
sub('__shared__ float abr[4][2];','__shared__ float abr[NW][2];')
sub('''        if (half == 0) { attn_data[col] = att * scale; }''','''        float value = att * scale;
        if (POST) {
            __shared__ float sums[NW];
            const float ss = gdn_wave_sum(value * value) * 0.5f;
            if (lane == 0) sums[threadIdx.y] = ss;
            __syncthreads();
            float total = 0.f;
            for (int j = 0; j < NW; j++) total += sums[j];
            const float sc = rsqrtf(total / S_v + pn.eps);
            value = (sc * value * pn.norm[col]) * ggml_cuda_op_silu_single(pn.z[h_idx * S_v + col]);
        }
        if (half == 0) { attn_data[col] = value; }''')
sub('const gdn_ab_args ab = gdn_ab_args{}) {','const gdn_ab_args ab = gdn_ab_args{}, const gdn_out_args pn = {}) {')
old='''                    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 2), dim3(32, 4, 1), 0, stream);
                    ggml_cuda_kernel_launch(gated_delta_net_cols<keep_rs_t>, lp,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows_base, rows_ids, rows_stride, l2eps, ab);'''
new='''                    if (pn.norm) {
                        GGML_ASSERT(n_tokens == 1 && n_seqs == 1 && H == 48);
                        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 1), dim3(32, 8, 1), 0, stream);
                        ggml_cuda_kernel_launch(gated_delta_net_cols<keep_rs_t, true>, lp,
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows_base, rows_ids, rows_stride, l2eps, ab, pn);
                    } else {
                        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(H, n_seqs, 2), dim3(32, 4, 1), 0, stream);
                        ggml_cuda_kernel_launch(gated_delta_net_cols<keep_rs_t>, lp,
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows_base, rows_ids, rows_stride, l2eps, ab);
                    }'''
sub(old,new)
sub('static std::unordered_map<const ggml_tensor *, gdn_ab_fold> g_gdn_ab;','''static std::unordered_map<const ggml_tensor *, gdn_ab_fold> g_gdn_ab;
static std::unordered_map<const ggml_tensor *, gdn_out_args> g_gdn_out;
void ggml_cuda_gdn_fold_out(const ggml_tensor * gdn, const float * norm, const float * z, float eps) {
    std::lock_guard<std::mutex> lk(g_gdn_rows_mu); g_gdn_out[gdn] = {norm, z, eps};
}''')
sub('gdn_ab_args ab = {};','gdn_ab_args ab = {};\n    gdn_out_args pn;')
sub('''        auto ia = g_gdn_ab.find(dst);''','''        auto io = g_gdn_out.find(dst);
        if (io != g_gdn_out.end()) pn = io->second;
        auto ia = g_gdn_ab.find(dst);''')
sub('rows_stride, l2eps, ab);','rows_stride, l2eps, ab, pn);',3)
# The default-POST kernel accepts the extra pn argument as well; it ignores it.
sub('g_gdn_ab.clear();','g_gdn_ab.clear(); g_gdn_out.clear();')
(d/'gated_delta_net.cu').write_text(s)
s=(r/'original/ggml-cuda.cu').read_text()
old='''            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);'''
new='''            if (getenv("GGML_GDN_OUT_FOLD") && atoi(getenv("GGML_GDN_OUT_FOLD")) &&
                node->src[2]->ne[0] == 128 && node->src[2]->ne[1] == 48 && node->src[2]->ne[2] == 1 && node->src[2]->ne[3] == 1) {
                for (int ni = i + 1; ni + 1 < cgraph->n_nodes && ni < i + 24; ni++) {
                    const ggml_tensor * nr = cgraph->nodes[ni], * nm = cgraph->nodes[ni + 1];
                    if (nr->op != GGML_OP_RMS_NORM || nr->src[0]->data != node->data || nr->ne[0] != 128 || ggml_nelements(nr) != 6144 ||
                        !ggml_cuda_can_fuse(cgraph, ni, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) continue;
                    const ggml_tensor * wt = nm->src[0] == nr ? nm->src[1] : nm->src[0];
                    if (wt->type != GGML_TYPE_F32 || ggml_nelements(wt) != 128 || !ggml_is_contiguous(wt) || !ggml_is_contiguous(nm)) break;
                    for (int ui = ni + 2; ui + 2 < cgraph->n_nodes && ui < ni + 14; ui++) {
                        const ggml_tensor * un = cgraph->nodes[ui];
                        if (un->op != GGML_OP_UNARY || ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) continue;
                        if (!ggml_cuda_can_fuse(cgraph, ui, {GGML_OP_UNARY, GGML_OP_MUL}, {GGML_UNARY_OP_SILU})) break;
                        const ggml_tensor * mul = cgraph->nodes[ui + 1], * other = mul->src[0] == un ? mul->src[1] : mul->src[0];
                        const ggml_tensor * mm = cgraph->nodes[ui + 2];
                        if (mm->op == GGML_OP_RESHAPE && mm->src[0] == mul && ui + 3 < cgraph->n_nodes) mm = cgraph->nodes[ui + 3];
                        if (other != nm || mm->op != GGML_OP_MUL_MAT || !strstr(mm->src[0]->name,"ssm_out") || !ggml_cuda_int4_converted(mm->src[0]) ||
                            !ggml_node_has_n_uses(cgraph, ui + 1, 1) || ggml_nelements(un) != 6144) break;
                        const float * z = nullptr;
                        { std::lock_guard<std::mutex> lk(g_done_mu); auto iz = g_zredir.find(un->src[0]); if (iz != g_zredir.end()) z = iz->second; }
                        if (z) {
                            float eps; memcpy(&eps, nr->op_params, sizeof(float));
                            ggml_cuda_gdn_fold_out(node,(const float *)wt->data,z,eps);
                            std::lock_guard<std::mutex> lk(g_done_mu);
                            g_hnorm[nm] = {(const float *)node->data,nullptr,-1.f};
                            g_done_nodes.insert(nr);g_done_nodes.insert(nm);
                        }
                        break;
                    }
                    break;
                }
            }
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);'''
sub(old,new)
old='''            ggml_cuda_int4_matvec(*cuda_ctx, mm->src[0], mm->src[1], r ? add : cgraph->nodes[i + extra], r ? &fusion_args : nullptr, xa, xb,
                                  hn_used ? 3 : 1, 256, hn.w, hn.eps);'''
new='''            if (hn_used && !hn.w) { xa = hn.x; xb = nullptr; }
            ggml_cuda_int4_matvec(*cuda_ctx, mm->src[0], mm->src[1], r ? add : cgraph->nodes[i + extra], r ? &fusion_args : nullptr, xa, xb,
                                  hn_used && hn.w ? 3 : 1, 256, hn.w, hn.eps);'''
sub(old,new)
(d/'ggml-cuda.cu').write_text(s)
print('GDN output fusion source prepared')
