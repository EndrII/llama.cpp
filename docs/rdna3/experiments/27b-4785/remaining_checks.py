import subprocess,json
from run import ROOT,SDK,environment,run
r=ROOT;d=r/'gap_micro_profile';d.mkdir()
env=environment(r/'bin',{})
for i in range(3):
 p=subprocess.run([str(r/'gap_microbench')],env=env,check=True,stdout=subprocess.PIPE,text=True)
 print('UNPROFILED',p.stdout.strip(),flush=True)
 with (d/('unprofiled'+str(i)+'.txt')).open('w') as f:f.write(p.stdout)
cmd=[str(SDK/'_rocm_sdk_core/bin/rocprofv3'),'--kernel-trace','-d',str(d),'-o','run','-f','csv','--',str(r/'gap_microbench')]
with (d/'stdout.txt').open('w') as o,(d/'stderr.txt').open('w') as e:subprocess.run(cmd,env=env,check=True,stdout=o,stderr=e)
print('PROFILED', (d/'stdout.txt').read_text().strip(),flush=True)
for n in [2,4]:run({'label':'gate_norm_rows'+str(n),'env':{'GGML_INT4_RMS_MV':'1','GGML_INT4_RMS_MV_ROWS':str(n)},'args':['-p','0','-n','128','-r','4','-o','json']})
for label,env in [('quality_all8',{}),('fast_all8',{'GGML_INT4_PREFILL':'fast'})]:
 run({'label':label,'env':dict(env,GGML_INT4_WN4_MIN='1000000000'),'args':['-p','512,2048','-n','0','-r','4','-o','json']})
