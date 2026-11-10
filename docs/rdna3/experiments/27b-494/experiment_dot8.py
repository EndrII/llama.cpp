from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
# Return to the non-postnorm API and keep the validated input-upload batching.
(d/'int4-wmma.cuh').write_text((r/'original/int4-wmma.cuh').read_text())
(d/'ggml-cuda.cu').write_text((r/'batch_cuda_base.cu').read_text())
for f in ['gated_delta_net.cu','gated_delta_net.cuh']:(d/f).write_text((r/'original'/f).read_text())
s=(r/'head_base.cu').read_text()
def sub(old,new,n=1):
 global s
 assert s.count(old)==n,(old[:100],s.count(old));s=s.replace(old,new)
# The low/high nibble planes fit in the exact same K-byte activation buffer.
sub('__device__ __forceinline__ void quant_rot8(', 'template <bool DOT8 = false>\n__device__ __forceinline__ void quant_rot8(')
a=s.index('''#pragma unroll
    for (int i = 0; i < 8; i++) { sq[i] = dppx<0xB1>(sq[i])''') if False else -1
# Inspect and replace the complete sum/store tail using its stable anchors.
a=s.index('''#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] += dppx<0xB1>''')
b=s.index('\n}',a)
old=s[a:b]
new='''    if (DOT8) {
        const int shift = 4 * (2 * (lane & 3) + ((lane >> 2) & 1));
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t lo = ((uint32_t) q[i] & 15u) << shift;
            uint32_t hi = ((uint32_t) (q[i] >> 4) & 15u) << shift;
            lo |= __float_as_uint(dppx<0xB1>(__uint_as_float(lo)));
            hi |= __float_as_uint(dppx<0xB1>(__uint_as_float(hi)));
            lo |= __float_as_uint(dppx<0x4E>(__uint_as_float(lo)));
            hi |= __float_as_uint(dppx<0x4E>(__uint_as_float(hi)));
            lo |= __float_as_uint(dppx<0x141>(__uint_as_float(lo)));
            hi |= __float_as_uint(dppx<0x141>(__uint_as_float(hi)));
            if ((lane & 7) == 0) {
                ((uint32_t *) (qdst + i * 32))[lane >> 3] = lo;
                ((uint32_t *) (qdst + i * 32 + 16))[lane >> 3] = hi;
            }
            if (lane == 0) ddst[i] = make_float2(d[i], 0.f);
        }
    } else {
'''+old+'''\n    }'''
s=s[:a]+new+s[b:]
# Standalone packing and fused RMS packing use the same selected format.
sub('__global__ __launch_bounds__(32) void k_rot_q8x(', 'template <bool DOT8 = false>\n__global__ __launch_bounds__(32) void k_rot_q8x(')
a=s.index('#pragma unroll\n    for (int i = 0; i < 8; i++) {',s.index('void k_rot_q8x('));b=s.index('\n}',a)
s=s[:a]+'''    quant_rot8<DOT8>(w, lane, Q + (size_t) n * K + sb * 256, D + (size_t) n * (K / 32) + sb * 8);'''+s[b:]
sub('__global__ __launch_bounds__(1024) void k_rmsnorm_rot_q8x(', 'template <bool DOT8 = false>\n__global__ __launch_bounds__(1024) void k_rmsnorm_rot_q8x(')
sub('quant_rot8(w, lane, Q + (size_t) n * K + wave * 256, D + (size_t) n * (K / 32) + wave * 8);','quant_rot8<DOT8>(w, lane, Q + (size_t) n * K + wave * 256, D + (size_t) n * (K / 32) + wave * 8);')
sub('constexpr int MV_NW = 10;','''constexpr int MV_NW = 10;
static bool int4_dot8() { static const bool enabled = getenv("GGML_INT4_DOT8") && atoi(getenv("GGML_INT4_DOT8")); return enabled; }
__device__ __forceinline__ int dot8_signed(int a, int b, int c) {
#ifdef INT4W_DEVICE_OK
    return __builtin_amdgcn_sdot8(a, b, c, false);
#else
    return 0;
#endif
}
__device__ __forceinline__ int dot8_mixed(int a, int b, int c) {
#ifdef INT4W_DEVICE_OK
    return __builtin_amdgcn_sudot8(true, a, false, b, c, false);
#else
    return 0;
#endif
}''')
sub('bool MJ = false>','bool MJ = false, bool DOT8 = false>')
sub('template <int ROWS, int NW = 10>','template <int ROWS, int NW = 10, bool DOT8 = false>')
sub('quant_rot8(wr, lane, &rq[wv][0], &rd[wv][0]);','quant_rot8<DOT8>(wr, lane, &rq[wv][0], &rd[wv][0]);',2)
# Replace the three dot/reconstruction sites, retaining the old arithmetic in the disabled instantiation.
a=s.index('''                int si = 0;''',s.index('void k_mv('));b=s.index('''                if (GATE) {''',a)
old=s[a:b]
new='''                int si = 0;
                if (DOT8) {
#pragma unroll
                    for (int wi = 0; wi < 4; wi++) si = dot8_signed((int) ww[wi], (int) xw[4 + wi], si);
                    si *= 16;
#pragma unroll
                    for (int wi = 0; wi < 4; wi++) si = dot8_mixed((int) ww[wi], (int) xw[wi], si);
                    acc[j] += (float) ((sc >> (4 * g)) & 15) * (ds.x * (float) si);
                } else {
'''+old+'''                }
'''
s=s[:a]+new+s[b:]
a=s.index('''                    int sg = 0;''',s.index('void k_mv('));b=s.index('\n                }\n            }',a)
old=s[a:b]
new='''                    int sg = 0;
                    if (DOT8) {
#pragma unroll
                        for (int wi = 0; wi < 4; wi++) sg = dot8_signed((int) wwg[wi], (int) xw[4 + wi], sg);
                        sg *= 16;
#pragma unroll
                        for (int wi = 0; wi < 4; wi++) sg = dot8_mixed((int) wwg[wi], (int) xw[wi], sg);
                        accg[j] += (float) ((scg >> (4 * g)) & 15) * (ds.x * (float) sg);
                    } else {
'''+old+'''\n                    }'''
s=s[:a]+new+s[b:]
a=s.index('''                int si = 0;''',s.index('void k_mv_rot_rows('));b=s.index('\n            }\n        }',a)
old=s[a:b]
new='''                int si = 0;
                if (DOT8) {
#pragma unroll
                    for (int wi = 0; wi < 4; wi++) si = dot8_signed((int) ww[wi], (int) xw[4 + wi], si);
                    si *= 16;
#pragma unroll
                    for (int wi = 0; wi < 4; wi++) si = dot8_mixed((int) ww[wi], (int) xw[wi], si);
                    acc[ri] += (float) ((subscales[ri] >> (4 * g)) & 15) * (ds.x * (float) si);
                } else {
'''+old+'''\n                }'''
s=s[:a]+new+s[b:]
# Producer launches select the format before any consumer executes.
old='''    k_rmsnorm_rot_q8x<<<N, 32 * (K / 256), 0, stream>>>((const float *) x->data, (const float *) wt->data, (float *) mul->data, xq, xd, K, eps);'''
new='''    if (int4_dot8()) k_rmsnorm_rot_q8x<true><<<N, 32 * (K / 256), 0, stream>>>((const float *) x->data, (const float *) wt->data, (float *) mul->data, xq, xd, K, eps);
    else k_rmsnorm_rot_q8x<false><<<N, 32 * (K / 256), 0, stream>>>((const float *) x->data, (const float *) wt->data, (float *) mul->data, xq, xd, K, eps);'''
sub(old,new)
old='''if (!packed) k_rot_q8x<<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);'''
new='''if (!packed) {
            if (int4_dot8()) k_rot_q8x<true><<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);
            else k_rot_q8x<false><<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);
        }'''
sub(old,new)
old='''        k_rot_q8x<<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);'''
new='''        if (int4_dot8()) k_rot_q8x<true><<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);
        else k_rot_q8x<false><<<dim3(K / 256, N), 32, 0, stream>>>(glu_x ? glu_x : (const float *) src1->data, xq_ptr, xd_ptr, K, glu_y);'''
sub(old,new)
a=s.index('''    const float inv = 1.f /''',s.index('void ggml_cuda_int4_matvec('));b=s.index('\n}\n\n#else',a)
launch=s[a:b]
replacements={
'k_mv<NCV, true>':'k_mv<NCV, true, false, false, false, false, DOT8>',
'k_mv<NCV, false>':'k_mv<NCV, false, false, false, false, false, DOT8>',
'k_mv_rot_rows<ROWS>':'k_mv_rot_rows<ROWS, 10, DOT8>',
'k_mv<1, false, true>':'k_mv<1, false, true, false, false, false, DOT8>',
'k_mv<1, false, false, false, false, true>':'k_mv<1, false, false, false, false, true, DOT8>',
'k_mv<1, false, false, true, true>':'k_mv<1, false, false, true, true, false, DOT8>',
'k_mv<1, false, false, true>':'k_mv<1, false, false, true, false, false, DOT8>',
'k_mv<4, true>':'k_mv<4, true, false, false, false, false, DOT8>'}
for old,new in replacements.items():assert old in launch,old;launch=launch.replace(old,new)
s=s[:a]+'''    const auto launch = [&](auto dot8_c) {
    constexpr bool DOT8 = decltype(dot8_c)::value;
'''+launch+'''\n    };
    if (int4_dot8()) launch(std::true_type{}); else launch(std::false_type{});'''+s[b:]
(d/'int4-wmma.cu').write_text(s);(r/'dot8.cu').write_text(s)
print('native dot8 source prepared')
