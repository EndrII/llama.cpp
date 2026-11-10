from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'original/gated_delta_net.cu').read_text()
old='template <bool keep_rs_t>\n__global__ void __launch_bounds__(128, 2)';new='template <bool keep_rs_t, bool ALG = false>\n__global__ void __launch_bounds__(128, 2)'
assert s.count(old)==1;s=s.replace(old,new)
a=s.index('        float kv0 = 0.f',s.index('gated_delta_net_cols('));b=s.index('        if (half == 0) { attn_data[col]',a)
old=s[a:b]
new='''        float kv0=0.f,kv1=0.f,kv2=0.f,kv3=0.f;
        float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
#pragma unroll
        for(int i=0;i<R;i+=4){
            const float4 kk=*(const float4*)(k_t+i),qq=*(const float4*)(q_t+i);
            kv0=fmaf(s[i],kk.x,kv0);kv1=fmaf(s[i+1],kk.y,kv1);
            kv2=fmaf(s[i+2],kk.z,kv2);kv3=fmaf(s[i+3],kk.w,kv3);
            if constexpr(ALG){
                a0=fmaf(s[i],qq.x,a0);a1=fmaf(s[i+1],qq.y,a1);
                a2=fmaf(s[i+2],qq.z,a2);a3=fmaf(s[i+3],qq.w,a3);
            }
        }
        float kv=(kv0+kv1)+(kv2+kv3);kv+=gdn_swap_rows(kv);
        const float delta=(v_val-g_val*kv)*beta_val;
#pragma unroll
        for(int i=0;i<R;i+=4){
            const float4 kk=*(const float4*)(k_t+i),qq=*(const float4*)(q_t+i);
            s[i]=fmaf(g_val,s[i],kk.x*delta);
            s[i+1]=fmaf(g_val,s[i+1],kk.y*delta);
            s[i+2]=fmaf(g_val,s[i+2],kk.z*delta);
            s[i+3]=fmaf(g_val,s[i+3],kk.w*delta);
            if constexpr(!ALG){
                a0=fmaf(s[i],qq.x,a0);a1=fmaf(s[i+1],qq.y,a1);
                a2=fmaf(s[i+2],qq.z,a2);a3=fmaf(s[i+3],qq.w,a3);
            }
        }
        float att=(a0+a1)+(a2+a3);att+=gdn_swap_rows(att);
        if constexpr(ALG){
            const float qk=gdn_wave_sum((kn.x*qn.x+kn.y*qn.y)+(kn.z*qn.z+kn.w*qn.w));
            att=fmaf(g_val,att,delta*qk);
        }
'''
s=s[:a]+new+s[b:]
old='''                    ggml_cuda_kernel_launch(gated_delta_net_cols<keep_rs_t>, lp,
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                        n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rows_base, rows_ids, rows_stride, l2eps, ab);'''
new='''                    if (H==48 && n_tokens==1 && getenv("GGML_GDN_ALGEBRA")) {
'''+old.replace('<keep_rs_t>','<keep_rs_t,true>')+'''
                    } else {
'''+old+'''
                    }'''
assert s.count(old)==1;s=s.replace(old,new);(d/'gated_delta_net.cu').write_text(s);(r/'algebra_gdn.cu').write_text(s)
