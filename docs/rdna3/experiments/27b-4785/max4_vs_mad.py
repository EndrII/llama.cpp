import subprocess,shutil,sys
from pathlib import Path
from run import ROOT,PATHS,run
r=ROOT;d=Path(PATHS['source'])/'ggml/src/ggml-cuda'
for f in ['ggml-cuda.cu','int4-wmma.cuh']:(d/f).write_text((r/'original'/f).read_text())
base=(r/'scale4.cu').read_text()
best=(r/'best.cu').read_text()
a=best.index('__device__ __forceinline__ uint4 nt128(');b=best.index('#define INT4W_LO',a)
base=base.replace('#define INT4W_LO',best[a:b]+'#define INT4W_LO')
for w in ['Wp','Wp2']:base=base.replace(f'*(const uint4 *) ({w} + wo + g * 256)',f'nt128({w}, wo + g * 256)')
(r/'max4_nt.cu').write_text(base)
mad=best.replace('return ((packed >> 16) & 255) + (packed >> 24) * code;','''uint32_t out;
    asm("v_mad_u32_u24 %0, %1, %2, %3" : "=v"(out) : "v"(code), "v"(packed >> 24), "v"((packed >> 16) & 255));
    return out;''')
assert mad!=best;(r/'affine_mad.cu').write_text(mad)
for label,source in [('max4_nt',base),('affine_mad',mad)]:
 (d/'int4-wmma.cu').write_text(source)
 subprocess.run([sys.executable,str(r/'build.py'),label],check=True)
 run({'label':label+'_quality','args':['-p','512,2048','-n','128','-r','4','-o','json']})
 run({'label':label+'_fast','env':{'GGML_INT4_PREFILL':'fast'},'args':['-p','512,2048','-n','0','-r','4','-o','json']})
 shutil.copy2(r/'bin/libggml-hip.so.0.18.0',r/(label+'.so'))
 if label=='max4_nt':run({'label':'max4_fast_accuracy','env':{'GGML_INT4_PREFILL':'fast'},'tool':'llama-perplexity','args':['-f','/mnt/data/projects/ossicle-gguf/work/wiki.test.raw','-c','512','-b','512','--kl-divergence','--kl-divergence-base','/mnt/9100pro/ossicle-work/kld_base_qwen38_27b.bin']})
