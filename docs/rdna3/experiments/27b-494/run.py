import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import threading

ROOT=Path('/home/svt/tuning27b-next')
PATHS=json.loads((ROOT/'paths.json').read_text())
SDK=Path('/mnt/data/venvs/rocm101/lib/python3.12/site-packages')
MODEL='/mnt/data/work27b/qwen38-27b-rotq4-gptq.gguf'

def environment(binary, overrides):
    return dict({'HOME':'/home/svt','PATH':'/usr/bin:/bin','HIP_VISIBLE_DEVICES':'0',
                 'LD_LIBRARY_PATH':':'.join(map(str,[binary,SDK/'_rocm_sdk_devel/lib',SDK/'_rocm_sdk_core/lib',SDK/'_rocm_sdk_libraries/lib']))},**overrides)

def run(job):
    label=job['label']
    binary=Path(job['binary']) if job.get('binary') else (Path(PATHS['baseline']) if job.get('baseline') else ROOT/'bin')
    dest=ROOT/'runs'/label
    dest.mkdir(parents=True,exist_ok=False)
    env=environment(binary,job.get('env',{}))
    command=[str(binary/job.get('tool','llama-bench')),'-m',job.get('model',MODEL),'-ngl','99','-t','12']+job['args']
    (dest/'command.json').write_text(json.dumps({'command':command,'env':env},indent=2))
    start=time.monotonic()
    print('START '+label,flush=True)
    memory_samples=[]
    stop=threading.Event()
    memory_devices=list(Path('/sys/class/drm').glob('card[0-9]/device/mem_info_vram_total'))
    memory_file=max(memory_devices,key=lambda p:int(p.read_text())).with_name('mem_info_vram_used')
    def sample_memory():
        while not stop.wait(0.02):
            try:memory_samples.append(int(memory_file.read_text()))
            except OSError:pass
    monitor=threading.Thread(target=sample_memory,daemon=True)
    monitor.start()
    with (dest/'stdout').open('w') as out,(dest/'stderr').open('w') as err:
        try:p=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=job.get('timeout',1200))
        finally:
            stop.set()
            monitor.join()
    summary={'label':label,'returncode':p.returncode,'elapsed':time.monotonic()-start}
    if memory_samples:summary['peak_vram_bytes']=max(memory_samples)
    if p.returncode:
        print((dest/'stderr').read_text()[-2500:],flush=True)
        raise RuntimeError(label)
    if job.get('tool','llama-bench')=='llama-bench':
        rows=json.loads((dest/'stdout').read_text())
        summary['results']=[]
        for r in rows:
            ns=r['samples_ns'][1:]
            n=r['n_prompt'] or r['n_gen']
            summary['results'].append({'prompt':r['n_prompt'],'gen':r['n_gen'],
                                       'median_ts':n*1e9/statistics.median(ns),
                                       'samples_ts':r['samples_ts'],'latency_ms':statistics.median(ns)/1e6})
    else:
        output=(dest/'stdout').read_text()+'\n'+(dest/'stderr').read_text()
        summary['metrics']=[line for line in output.splitlines() if any(s in line for s in ['Mean    KLD','Mean PPL(Q)','Same top p:','Mean PPL(Q)/','perplexity ='])]
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    for job in json.loads(Path(sys.argv[1]).read_text()):run(job)
