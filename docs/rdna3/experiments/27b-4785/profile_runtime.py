import sys,subprocess,json
from pathlib import Path
sys.path.insert(0,"/home/svt/tuning27b50")
from run import environment,SDK,MODEL
r=Path("/home/svt/tuning27b50");d=r/"runtime_profile";d.mkdir()
cmd=[str(SDK/"_rocm_sdk_core/bin/rocprofv3"),"--kernel-trace","--hip-runtime-trace","--hip-event-trace","--hip-graph-trace","--memory-copy-trace","-d",str(d),"-o","run","-f","csv","json","--",str(r/"baseline/llama-bench"),"-m",MODEL,"-ngl","99","-t","12","-p","0","-n","128","-r","2","-o","json"]
(d/"command.json").write_text(json.dumps(cmd,indent=2))
with (d/"stdout.json").open("w") as out,(d/"stderr.log").open("w") as err:
 subprocess.run(cmd,env=environment(r/"baseline",{}),stdout=out,stderr=err,check=True,timeout=300)
print("PROFILE_COMPLETE",flush=True)
