from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
p=d/'int4-wmma.cu';s=(r/'prefetch.cu').read_text()
old='''        static const int rows_27b = getenv("GGML_INT4_27B_ROWS") ? atoi(getenv("GGML_INT4_27B_ROWS")) : 4;'''
new='''        static const int rows_default = getenv("GGML_INT4_27B_ROWS") ? atoi(getenv("GGML_INT4_27B_ROWS")) : 4;
        const int rows_27b = K == 6144 && !glu_y && getenv("GGML_INT4_GDN_ROWS") ? atoi(getenv("GGML_INT4_GDN_ROWS")) : rows_default;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('(rows_27b == 2 || rows_27b == 4)', '(rows_27b == 1 || rows_27b == 2 || rows_27b == 4)')
s=s.replace('if (rows_27b == 2) run_rows', 'if (rows_27b == 1) run_rows(std::integral_constant<int, 1>{}); else if (rows_27b == 2) run_rows')
old='''                const int variant = getenv("GGML_MV_PREFETCH") ? atoi(getenv("GGML_MV_PREFETCH")) : 0;'''
new='''                const char * selected = K == 6144 && !glu_y ? getenv("GGML_MV_PREFETCH_GDN") : nullptr;
                if (!selected) selected = getenv("GGML_MV_PREFETCH");
                const int variant = selected ? atoi(selected) : 0;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('''                else { RUN_P(10,ROWS,false); }''','''                else if (!glu_y && variant == 7) { RUN_P(4,ROWS,true); }
                else if (!glu_y && variant == 8) { RUN_P(6,ROWS,true); }
                else { RUN_P(10,ROWS,false); }''')
p.write_text(s);(r/'occupancy_int4.cu').write_text(s)
p=d/'gated_delta_net.cu';s=p.read_text()
old='''template <bool keep_rs_t, bool POST = false>''';new='''template <bool keep_rs_t, bool POST = false, int LIMIT = 0>'''
assert s.count(old)==1;s=s.replace(old,new)
old='''            kv2 = fmaf(s[i + 2], kk.z, kv2); kv3 = fmaf(s[i + 3], kk.w, kv3);
        }
        float kv'''
new='''            kv2 = fmaf(s[i + 2], kk.z, kv2); kv3 = fmaf(s[i + 3], kk.w, kv3);
            if (LIMIT && ((i + 4) % (LIMIT ? LIMIT : 64)) == 0) {
                asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0);
            }
        }
        if (LIMIT) { asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0); }
        float kv'''
assert s.count(old)==1;s=s.replace(old,new)
old='''            s[i + 3] = fmaf(g_val, s[i + 3], kk.w * delta); a3 = fmaf(s[i + 3], qq.w, a3);
        }
        float att'''
new='''            s[i + 3] = fmaf(g_val, s[i + 3], kk.w * delta); a3 = fmaf(s[i + 3], qq.w, a3);
            if (LIMIT && ((i + 4) % (LIMIT ? LIMIT : 64)) == 0) {
                asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0);
            }
        }
        float att'''
assert s.count(old)==1;s=s.replace(old,new)
# One compile-time limit controls register lifetimes without dynamic indexing of state.
key='''                if (warp_size == 32) {'''
assert s.count(key)==1
s=s.replace(key,key+'''
                    const bool eligible = H == 48 && n_tokens == 1 && n_seqs == 1;
                    const int limit = eligible && getenv("GGML_GDN_LDS_LIMIT") ? atoi(getenv("GGML_GDN_LDS_LIMIT")) : 0;
                    const auto launch_cols = [&](auto limit_c) {
                    constexpr int LIM = decltype(limit_c)::value;''')
s=s.replace('gated_delta_net_cols<keep_rs_t, true>', 'gated_delta_net_cols<keep_rs_t, true, LIM>')
s=s.replace('gated_delta_net_cols<keep_rs_t>,', 'gated_delta_net_cols<keep_rs_t, false, LIM>,')
old='''                    break;
                }
            }
#endif'''
new='''                    };
                    if (limit == 8) launch_cols(std::integral_constant<int,8>{});
                    else if (limit == 16) launch_cols(std::integral_constant<int,16>{});
                    else if (limit == 32) launch_cols(std::integral_constant<int,32>{});
                    else launch_cols(std::integral_constant<int,0>{});
                    break;
                }
            }
#endif'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s);(r/'occupancy_gdn.cu').write_text(s)
