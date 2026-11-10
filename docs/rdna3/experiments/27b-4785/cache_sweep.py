import sys,json,subprocess,shutil
from pathlib import Path
from run import run,ROOT,PATHS
p=Path(PATHS['source'])/'ggml/src/ggml-cuda/int4-wmma.cu'
s=(ROOT/'scale4_affine_nt.cu').read_text()
s=s.replace('uint4 nt128(const uint8_t * p)', 'uint4 nt128(const uint8_t * p, size_t offset)')
s=s.replace('const int4v v = __builtin_nontemporal_load((const int4v *) p);','const int4v v = __builtin_amdgcn_raw_buffer_load_b128(INT4W_RSRC(p), (int) offset, 0, CACHE_POLICY);')
for w in ['Wp','Wp2']:s=s.replace(f'nt128({w} + wo + g * 256)',f'nt128({w}, wo + g * 256)')
(ROOT/'cache_base.cu').write_text(s)
for policy in range(8):
 p.write_text(s.replace('CACHE_POLICY',str(policy)))
 subprocess.run([sys.executable,str(ROOT/'build.py'),'cache'+str(policy)],check=True)
 run({'label':'scale4_cache'+str(policy),'args':['-p','0','-n','128','-r','3','-o','json']})
 if policy in [0,2,4,6,7]:shutil.copy2(ROOT/'bin/libggml-hip.so.0.18.0',ROOT/('cache'+str(policy)+'.so'))
