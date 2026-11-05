from pathlib import Path
import datetime,hashlib,json,os,shutil,statistics,subprocess,threading,time
ROOT=Path('/mnt/data/base27b-rocm-compare')
SDK=Path('/mnt/data/venvs/rocm101/lib/python3.12/site-packages')
MODEL='/mnt/9100pro/ossicle-work/qwen38-27b-Q4_K_M.gguf'
RUNTIMES={
 '7.2.0':{'bin':ROOT/'build72/bin','lib':['/opt/rocm-7.2.0/lib']},
 '10.1.0':{'bin':ROOT/'bin101','lib':[str(SDK/x/'lib') for x in ['_rocm_sdk_devel','_rocm_sdk_core','_rocm_sdk_libraries']]},
}

def environment(version):
 rt=RUNTIMES[version]
 return {'HOME':'/home/svt','PATH':'/usr/bin:/bin','HIP_VISIBLE_DEVICES':'0','LD_LIBRARY_PATH':':'.join([str(rt['bin'])]+rt['lib']),'TMPDIR':str(ROOT/'tmp')}

def run(version,label,reps):
 rt=RUNTIMES[version];dest=ROOT/'runs'/label;dest.mkdir(parents=True,exist_ok=False)
 env=environment(version)
 cmd=[str(rt['bin']/'llama-bench'),'-m',MODEL,'-ngl','99','-t','12','-b','2048','-ub','512','-fa','auto','-p','512,2048','-n','128','-r',str(reps),'-o','json']
 (dest/'command.json').write_text(json.dumps({'version':version,'command':cmd,'env':env,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
 memory=max(Path('/sys/class/drm').glob('card[0-9]/device/mem_info_vram_total'),key=lambda p:int(p.read_text())).with_name('mem_info_vram_used')
 samples=[];stop=threading.Event()
 def monitor():
  while not stop.wait(.02):
   try:samples.append(int(memory.read_text()))
   except OSError:pass
 thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.monotonic();print('START',label,flush=True)
 try:
  with (dest/'stdout.json').open('w') as out,(dest/'stderr.log').open('w') as err:p=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=1200)
 finally:stop.set();thread.join()
 if p.returncode:print((dest/'stderr.log').read_text()[-5000:],flush=True);raise RuntimeError(label+' failed')
 assert 'RX 7900 XT' in (dest/'stderr.log').read_text()
 data=json.loads((dest/'stdout.json').read_text());results=[]
 for row in data:
  ns=row['samples_ns'][1:];n=row['n_prompt'] or row['n_gen'];name='pp'+str(n) if row['n_prompt'] else 'tg'+str(n)
  results.append({'test':name,'median_tok_s':n*1e9/statistics.median(ns),'warm_samples_tok_s':[n*1e9/x for x in ns],'samples_ns':row['samples_ns']})
 summary={'version':version,'label':label,'returncode':p.returncode,'elapsed_s':time.monotonic()-start,'peak_whole_gpu_vram_bytes':max(samples),'results':results}
 (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 return summary

if __name__=='__main__':
 if not (ROOT/'bin101').exists():shutil.copytree('/tmp/claude-1000/up/build/bin',ROOT/'bin101',symlinks=True)
 for ver in RUNTIMES:
  env=environment(ver)
  p=subprocess.run(['ldd',str(RUNTIMES[ver]['bin']/'libggml-hip.so')],env=env,capture_output=True,text=True,check=True)
  (ROOT/('libraries_'+ver+'.txt')).write_text(p.stdout)
  print('LIBRARIES',ver, '\n'.join(x for x in p.stdout.splitlines() if any(s in x for s in ['amdhip','rocblas','hipblas','hsa-runtime'])),flush=True)
  assert 'not found' not in p.stdout
  if ver=='7.2.0':assert '/opt/rocm-7.2.0/lib/libamdhip64' in p.stdout
  else:assert '_rocm_sdk_' in p.stdout and '/opt/rocm' not in p.stdout
 all_results=[]
 for ver,label,reps in [('7.2.0','rocm72_A',5),('10.1.0','rocm101_A',5),('10.1.0','rocm101_B',4),('7.2.0','rocm72_B',4)]:
  dest=ROOT/'runs'/label
  if (dest/'summary.json').exists():all_results.append(json.loads((dest/'summary.json').read_text()));continue
  if (dest/'stdout.json').exists():
   data=json.loads((dest/'stdout.json').read_text());assert 'RX 7900 XT' in (dest/'stderr.log').read_text();results=[]
   for row in data:
    ns=row['samples_ns'][1:];n=row['n_prompt'] or row['n_gen'];name='pp'+str(n) if row['n_prompt'] else 'tg'+str(n)
    results.append({'test':name,'median_tok_s':n*1e9/statistics.median(ns),'warm_samples_tok_s':[n*1e9/x for x in ns],'samples_ns':row['samples_ns']})
   summary={'version':ver,'label':label,'returncode':0,'results':results,'note':'Recovered completed benchmark JSON after GPU name validation was corrected.'}
   (dest/'summary.json').write_text(json.dumps(summary,indent=2));all_results.append(summary);print('RECOVERED',json.dumps(summary),flush=True)
  else:all_results.append(run(ver,label,reps))
 combined={}
 for ver in RUNTIMES:
  combined[ver]={}
  for test in ['pp512','pp2048','tg128']:
   ns=[]
   for x in all_results:
    if x['version']==ver:
     for rr in x['results']:
      if rr['test']==test:ns+=rr['samples_ns'][1:]
   n=int(test[2:]);combined[ver][test]={'median_tok_s':n*1e9/statistics.median(ns),'warm_repetitions':len(ns)}
 ratios={test:combined['10.1.0'][test]['median_tok_s']/combined['7.2.0'][test]['median_tok_s'] for test in ['pp512','pp2048','tg128']}
 result={'model':MODEL,'source':'/tmp/claude-1000/up/src','settings':{'gpu':'RX 7900 XT','gpu_layers':99,'threads':12,'batch':2048,'ubatch':512,'flash_attention':'auto','mtp':False,'speculative_decode':False},'combined':combined,'speedup_101_over_72':ratios,'runs':all_results}
 (ROOT/'results.json').write_text(json.dumps(result,indent=2));print('COMBINED',json.dumps({'combined':combined,'speedup':ratios}),flush=True)
