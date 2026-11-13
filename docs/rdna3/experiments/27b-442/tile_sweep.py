import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'/home/svt/tuning27b')
from run import run

root=Path('/home/svt/tuning27b')
source=Path(json.loads((root/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda/int4-wmma.cu'
base=source.read_text()
(root/'prepack_source.cu').write_text(base)
for mi,nj in [(1,2),(2,1),(1,1),(4,1)]:
    label=f'mi{mi}_nj{nj}'
    text=base.replace('constexpr int MI = 2, NJ = 2, WAVES_N = 8;',f'constexpr int MI = {mi}, NJ = {nj}, WAVES_N = 8;')
    text=text.replace('constexpr int TILE_N = WAVES_N * NJ * 16;', 'constexpr int TILE_N = 256;')
    source.write_text(text)
    subprocess.run(['/usr/bin/python3',str(root/'build.py'),label],check=True)
    for preset in ['fast','quality']:
        run({'label':label+'_'+preset,'env':{'GGML_INT4_PREFILL':preset},'args':['-p','512,2048','-n','0','-r','4','-o','json']})
source.write_text(base)
