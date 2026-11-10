import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'/home/svt/tuning27b')
from run import run

root=Path('/home/svt/tuning27b')
source=Path(json.loads((root/'paths.json').read_text())['source'])/'ggml/src/ggml-cuda/int4-wmma.cu'
base=source.read_text()
(root/'rows_source.cu').write_text(base)
old='constexpr int UC = (NUNIT >= 2) ? 2 : 1;'
assert base.count(old)==1
for unroll in [1,4]:
    source.write_text(base.replace(old,f'constexpr int UC = NUNIT < {unroll} ? NUNIT : {unroll};'))
    subprocess.run(['/usr/bin/python3',str(root/'build.py'),'uc'+str(unroll)],check=True)
    run({'label':'unroll_'+str(unroll)+'_fast','env':{'GGML_INT4_PREFILL':'fast'},'args':['-p','512,2048','-n','0','-r','5','-o','json']})
source.write_text(base)
