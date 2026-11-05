from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
(d/'gated_delta_net.cu').write_text((r/'original/gated_delta_net.cu').read_text())
s=(r/'head_blob.cu').read_text()
key='// Several output-row tiles share one fused activation rotation.'
assert key in s
s=s.replace(key,'''// Pack the normalized/gated GDN activation once, using the same rotation math.
__global__ __launch_bounds__(32) void k_gdn_rot_q8x(const float * X,const float * Y,const float * norm,float eps,int mode,int ystride,int8_t * Q,float2 * D) {
    const int sb=blockIdx.x,lane=threadIdx.x,r=lane&15;
    half16v rb1,ra2;
#pragma unroll
    for(int k=0;k<16;k++){rb1[k]=h16(k,r);ra2[k]=h16(r,k<8?2*k:2*(k-8)+1);}
    const float8v wr=rotate_block(X+sb*256,lane,rb1,ra2,Y?Y+(size_t)sb*ystride:nullptr,mode,norm,eps);
    quant_rot8(wr,lane,Q+sb*256,D+sb*8);
}

'''+key)
s=s.replace('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, bool PACKED = false>')
old='''        int K, int mode, int ystride, const float * __restrict__ norm, float eps) {'''
new='''        int K, int mode, int ystride, const float * __restrict__ norm, float eps, const int8_t * Xq=nullptr,const float2 * Xd=nullptr) {'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('int8_t rq[NW][256]','int8_t rq[PACKED?1:NW][256]').replace('float2 rd[NW][8]','float2 rd[PACKED?1:NW][8]')
old='''        const float8v wr = rotate_block(X + sb * 256, lane, rb1, ra2, Y ? Y + (size_t) sb * ystride : nullptr, mode, norm, eps);
        quant_rot8(wr, lane, &rq[wv][0], &rd[wv][0]);
        __builtin_amdgcn_wave_barrier();
        const uint4 * xq = (const uint4 *) &rq[wv][h * 128];
        const float4 * xd = (const float4 *) &rd[wv][h * 4];'''
new='''        if constexpr(!PACKED){
            const float8v wr=rotate_block(X+sb*256,lane,rb1,ra2,Y?Y+(size_t)sb*ystride:nullptr,mode,norm,eps);
            quant_rot8(wr,lane,&rq[wv][0],&rd[wv][0]);
            __builtin_amdgcn_wave_barrier();
        }
        const uint4 * xq=PACKED?(const uint4*)(Xq+sb*256+h*128):(const uint4*)&rq[wv][h*128];
        const float4 * xd=PACKED?(const float4*)(Xd+sb*8+h*4):(const float4*)&rd[wv][h*4];'''
assert s.count(old)==1;s=s.replace(old,new)
key='''    if (rot) {
        static const int rows_27b'''
new='''    if (rot) {
        const int prepack=K==6144&&M==5120&&getenv("GGML_GDN_PREPACK")?atoi(getenv("GGML_GDN_PREPACK")):0;
        if(prepack==1||prepack==2||prepack==4){
            int8_t * q;float2 * scales;
            {
                std::lock_guard<std::mutex> lk(g_mu);MvCache & mc=g_mc[stream];
                ac_reserve(mc.xq,mc.xq_cap,(size_t)K);ac_reserve(mc.xd,mc.xd_cap,(size_t)(K/32)*sizeof(float2));
                q=mc.xq;scales=mc.xd;mc.src1=nullptr;
            }
            k_gdn_rot_q8x<<<K/256,32,0,stream>>>(glu_x?glu_x:(const float*)src1->data,glu_y,nw,eps,mode,ystride,q,scales);
            const auto launch=[&](auto rc){
                constexpr int R=decltype(rc)::value;
                k_mv_rot_rows<R,10,true><<<dim3(M/(16*R)),dim3(32,10),0,stream>>>(w.Wp,w.Wh,w.S,inv,nullptr,nullptr,(float*)dst->data,xb,K,1,256,nullptr,0.f,q,scales);
            };
            if(prepack==1)launch(std::integral_constant<int,1>{});
            else if(prepack==2)launch(std::integral_constant<int,2>{});
            else launch(std::integral_constant<int,4>{});
            CUDA_CHECK(cudaGetLastError());return;
        }
        static const int rows_27b'''
assert s.count(key)==1;s=s.replace(key,new)
(d/'int4-wmma.cu').write_text(s);(r/'gdn_prepack.cu').write_text(s)
