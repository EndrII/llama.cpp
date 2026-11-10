from pathlib import Path
import json
r=Path('/home/svt/tuning27b-next');d=Path(json.loads((r/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda'
s=(r/'selective_a4.cu').read_text()
s=s.replace('(rows_27b == 2 || rows_27b == 4)','(rows_27b == 2 || rows_27b == 4 || rows_27b == 8)')
s=s.replace('if (rows_27b == 2) run_rows','if (rows_27b == 8) run_rows(std::integral_constant<int, 8>{}); else if (rows_27b == 2) run_rows')
a=s.index('                const char * selected=getenv("GGML_INT4_DECODE_A4")');b=s.index('\n#undef RUN_A4',a)+len('\n#undef RUN_A4')
old=s[a:b]
new=old.replace('#define RUN_A4(C) k_mv_rot_rows<ROWS,10,C>','#define RUN_A4(C) k_mv_rot_rows<ROWS,NW,C>').replace(')),blk,0,stream>>>', ')),dim3(32,NW),0,stream>>>')
pos=new.index('#define RUN_A4');new=new[:pos]+'''                const auto launch=[&](auto nwc){
                constexpr int NW=decltype(nwc)::value;
'''+new[pos:]+'''
                };
                if constexpr(ROWS==8){
                    const int choice=getenv("GGML_ROT_ROW8_NW")?atoi(getenv("GGML_ROT_ROW8_NW")):0;
                    if(choice==10)launch(std::integral_constant<int,10>{});
                    else if(choice==20)launch(std::integral_constant<int,20>{});
                    else if(K==6144)launch(std::integral_constant<int,24>{});
                    else launch(std::integral_constant<int,17>{});
                }else launch(std::integral_constant<int,10>{});'''
s=s[:a]+new+s[b:]
(d/'int4-wmma.cu').write_text(s);(r/'rows8.cu').write_text(s)
