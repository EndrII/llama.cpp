from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'original/mmvq.cu').read_text()
old='''template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)'''
new='''template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, int HEAD_NW = 0, int HEAD_ROWS = 0>
__launch_bounds__((HEAD_NW ? HEAD_NW : calc_nwarps(type,ncols_dst,get_device_table_id()))*ggml_cuda_get_physical_warp_size(),1)'''
assert s.count(old)==1;s=s.replace(old,new)
old='''    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);'''
new='''    constexpr int nwarps = HEAD_NW ? HEAD_NW : calc_nwarps(type,ncols_dst,table_id);
    constexpr int rows_per_cuda_block = HEAD_ROWS ? HEAD_ROWS : calc_rows_per_block(ncols_dst,table_id,small_k,nwarps);'''
assert s.count(old)==1;s=s.replace(old,new)
key='''    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;'''
pos=s.index(key,s.index('static void mul_mat_vec_q_switch_fusion('))+len(key)
insert='''
    if constexpr(c_ncols_dst==1&&(type==GGML_TYPE_IQ4_NL||type==GGML_TYPE_IQ4_XS)){
        const char * opt=getenv("GGML_IQ_HEAD_TILE");
        if(opt&&!has_fusion&&!ids&&ncols_x==5120&&stride_col_dst==248320&&block_nums.x==248320&&block_nums.y==1&&block_nums.z==1&&
           GGML_CUDA_CC_IS_RDNA3_0(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)){
            const auto launch=[&](auto nwc,auto rc){
                constexpr int NW=decltype(nwc)::value,R=decltype(rc)::value;
                const ggml_cuda_kernel_launch_params lp=ggml_cuda_kernel_launch_params(dim3(248320/R),dim3(32,NW),0,stream);
                ggml_cuda_kernel_launch(mul_mat_vec_q<type,1,false,false,NW,R>,lp,
                    vx,vy,ids,fusion,dst,ncols_x,nchannels_y,stride_row_x,stride_col_y,stride_col_dst,
                    channel_ratio,stride_channel_x,stride_channel_y,stride_channel_dst,
                    sample_ratio,stride_sample_x,stride_sample_y,stride_sample_dst,ids_stride);
            };
            const int tile=atoi(opt);
#define HEAD_TILE(NW,R) case NW*100+R:launch(std::integral_constant<int,NW>{},std::integral_constant<int,R>{});return
            switch(tile){HEAD_TILE(1,4);HEAD_TILE(2,4);HEAD_TILE(4,4);HEAD_TILE(2,8);HEAD_TILE(4,8);HEAD_TILE(8,8);HEAD_TILE(4,16);HEAD_TILE(8,16);}
#undef HEAD_TILE
        }
    }
'''
s=s[:pos]+insert+s[pos:];(d/'mmvq.cu').write_text(s);(r/'head_tile_mmvq.cu').write_text(s)
