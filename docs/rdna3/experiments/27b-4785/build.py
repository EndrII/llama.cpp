import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path('/home/svt/tuning27b50')
paths=json.loads((root/'paths.json').read_text())
sdk=Path('/mnt/data/venvs/rocm101/lib/python3.12/site-packages')
env={'HOME':'/home/svt','PATH':str(sdk/'_rocm_sdk_devel/bin')+':/usr/bin:/bin',
     'LD_LIBRARY_PATH':':'.join(map(str,[sdk/'_rocm_sdk_devel/lib',sdk/'_rocm_sdk_core/lib',sdk/'_rocm_sdk_libraries/lib']))}
label=sys.argv[1]
start=time.monotonic()
with (root/('build_'+label+'.log')).open('w') as log:
    p=subprocess.run(['cmake','--build',paths['build'],'--target','llama-bench','llama-perplexity','-j','8'],env=env,stdout=log,stderr=subprocess.STDOUT)
if p.returncode:
    print((root/('build_'+label+'.log')).read_text()[-5000:])
    raise SystemExit(p.returncode)
shutil.copy2(Path(paths['build'])/'bin/libggml-hip.so.0.18.0',root/'bin/libggml-hip.so.0.18.0')
print('BUILT',label,round(time.monotonic()-start,1),'seconds',flush=True)
