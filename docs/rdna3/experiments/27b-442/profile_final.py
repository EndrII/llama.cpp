import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,'/home/svt/tuning27b')
from run import environment,SDK,MODEL

root=Path('/home/svt/tuning27b')
dest=root/(sys.argv[1] if len(sys.argv)>1 else 'profiles')
dest.mkdir(exist_ok=True)
for label,preset,prompt,tokens,reps in [('decode','quality',0,128,2),('prefill_quality_512','quality',512,0,3),('prefill_fast_512','fast',512,0,3)]:
    path=dest/label
    path.mkdir()
    command=[str(SDK/'_rocm_sdk_core/bin/rocprofv3'),'--kernel-trace','--memory-copy-trace','-d',str(path),'-o','run','-f','csv','--',
             str(root/'bin/llama-bench'),'-m',MODEL,'-ngl','99','-t','12','-p',str(prompt),'-n',str(tokens),'-r',str(reps),'-o','json']
    (path/'command.json').write_text(json.dumps(command,indent=2))
    print('PROFILE '+label,flush=True)
    with (path/'stdout.json').open('w') as out,(path/'stderr.log').open('w') as err:
        subprocess.run(command,env=environment(root/'bin',{'GGML_INT4_PREFILL':preset}),stdout=out,stderr=err,timeout=300,check=True)
subprocess.run(['/usr/bin/python3','/home/svt/analyze_qwen27_profile.py',str(dest)],check=True)
