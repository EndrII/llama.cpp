import json,subprocess
from pathlib import Path
from run import ROOT,environment,SDK,MODEL
r=ROOT;top=r/'profile_selected';top.mkdir(exist_ok=True)
base={'GGML_CUDA_BATCH_INPUTS':'1','GGML_INT4_HEAD_REQUANT':'iq4_xs','GGML_INT4_HEAD_FILE':str(r/'head_iq4_xs.bin')}
for name,args,env in [('decode',['-p','0','-n','128','-r','2'],{}),('prefill_quality_512',['-p','512','-n','0','-r','3'],{}),('prefill_fast_512',['-p','512','-n','0','-r','3'],{'GGML_INT4_PREFILL':'fast'})]:
 d=top/name;d.mkdir()
 env=dict(base,**env)
 cmd=[str(SDK/'_rocm_sdk_core/bin/rocprofv3'),'--kernel-trace','--hip-runtime-trace','--memory-copy-trace','-d',str(d),'-o','run','-f','csv','--',str(r/'selected/bin/llama-bench'),'-m',MODEL,'-ngl','99','-t','12']+args+['-o','json']
 (d/'command.json').write_text(json.dumps({'command':cmd,'env':env},indent=2))
 print('PROFILE',name,flush=True)
 with (d/'stdout.json').open('w') as out,(d/'stderr.log').open('w') as err:subprocess.run(cmd,env=environment(r/'selected/bin',env),stdout=out,stderr=err,check=True,timeout=300)
print('PROFILE_COMPLETE',flush=True)
