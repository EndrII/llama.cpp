from pathlib import Path
import hashlib,json,subprocess
r=Path('/mnt/data/base27b-rocm-compare');result=json.loads((r/'results.json').read_text())

def cache(p):
 out={}
 for line in p.read_text().splitlines():
  if line.startswith('GGML_') and '=' in line and ':' in line:
   key,rest=line.split(':',1);typ,val=rest.split('=',1)
   if typ in ['BOOL','STRING'] and not key.endswith('_ADVANCED'):out[key]=val
 return out
old=cache(r/'build72/CMakeCache.txt');new=cache(Path('/tmp/claude-1000/up/build/CMakeCache.txt'))
diff={k:[old.get(k),new.get(k)] for k in sorted(old.keys()|new.keys()) if old.get(k)!=new.get(k)}
assert not diff,diff
common_fields=['cpu_info','backends','model_filename','model_type','model_size','model_n_params','n_batch','n_ubatch','n_threads','type_k','type_v','n_gpu_layers','no_kv_offload','flash_attn','n_depth']
rows=[row for p in (r/'runs').glob('*/stdout.json') for row in json.loads(p.read_text())]
for key in common_fields:assert len({str(x[key]) for x in rows})==1,(key,[x[key] for x in rows])
source=Path('/tmp/claude-1000/up/src');h=hashlib.sha256();count=0
for prefix in ['src','ggml/src','ggml/include','common','tools/llama-bench']:
 for p in sorted((source/prefix).rglob('*')):
  if p.is_file() and p.suffix in ['.h','.hpp','.c','.cpp','.cu','.cuh','.cmake']:
   h.update(str(p.relative_to(source)).encode()+b'\0'+p.read_bytes());count+=1
manifest={'source':str(source),'source_commit':'unknown (existing local source archive)','source_code_sha256':h.hexdigest(),'source_files_hashed':count,'ggml_build_option_differences':diff,'identical_benchmark_fields':{k:rows[0][k] for k in common_fields},'model_file_stat':{'bytes':Path(result['model']).stat().st_size,'mtime_ns':Path(result['model']).stat().st_mtime_ns},'downloads':False,'comparison':'Each SDK compiler plus its own HIP, HSA, hipBLAS, hipBLASLt and rocBLAS libraries. This is a complete SDK/toolchain comparison, not runtime-only substitution.'}
for ver,binpath in [('7.2.0',r/'build72/bin'),('10.1.0',r/'bin101')]:
 manifest[ver]={'binaries':{}}
 for name in ['llama-bench','libggml-hip.so','libllama-bench-impl.so','libllama.so']:
  p=binpath/name
  with p.open('rb') as f:manifest[ver]['binaries'][name]=hashlib.file_digest(f,'sha256').hexdigest()
(r/'manifest.json').write_text(json.dumps(manifest,indent=2))
lines=['# Stock Qwen3.8-27B: ROCm 7.2.0 vs 10.1.0','',
'Existing Qwen3.8-27B Q4_K_M GGUF on the RX 7900 XT (20 GB), using the same local stock llama.cpp source. Both builds use gfx1100, Release, HIP graphs, native CPU flags, and matching GGML options. No model or SDK was downloaded.','',
'| Test | ROCm 7.2.0 (tokens/s) | ROCm 10.1.0 (tokens/s) | 10.1 / 7.2 | Change |',
'|---|---:|---:|---:|---:|']
for test in ['pp512','pp2048','tg128']:
 a=result['combined']['7.2.0'][test]['median_tok_s'];b=result['combined']['10.1.0'][test]['median_tok_s'];ratio=b/a
 lines.append(f'| {test} | {a:.2f} | {b:.2f} | {ratio:.4f}x | {(ratio-1)*100:+.2f}% |')
lines+=['','The measured differences are below 0.5%, with no meaningful speedup in this stock-model benchmark. Individual run medians vary by a similar amount.','',
'Each result is the median of seven warmed repetitions pooled from two processes. The first measured repetition in each process was excluded for graph/cache warmup. Order: 7.2 A, 10.1 A, 10.1 B, 7.2 B. No GPU profiling or compilation ran during measurements.','',
'Settings: full GPU offload (`-ngl 99`), 12 CPU threads, batch 2048, microbatch 512, Flash Attention `auto`, F16 KV cache, depth 0, no MTP or speculative decoding. Each SDK supplies its compiler and runtime/library stack, so the comparison includes compiler changes as well as runtime/library changes.','',
'ROCm 7.2.0: `/opt/rocm-7.2.0`, Clang 22. ROCm 10.1.0 SDK: `/mnt/data/venvs/rocm101/lib/python3.12/site-packages`, Clang 24. The runtime library paths were checked with `ldd`; the corresponding outputs are saved alongside this report.','',
'Model: `/mnt/9100pro/ossicle-work/qwen38-27b-Q4_K_M.gguf`. Source: `/tmp/claude-1000/up/src` (existing archive; commit metadata unavailable). Source fingerprint and binary hashes are in `manifest.json`.','',
'Raw commands, environment, full JSON timing samples and stderr are in `runs/`. Aggregates are in `results.json`. `configure72.json` records the local-only 7.2 build setup; `compare.py` contains the benchmark runner.','']
(r/'README.md').write_text('\n'.join(lines))
print('\n'.join(lines[4:11]));print('Validated identical benchmark settings and matching GGML build options.')
