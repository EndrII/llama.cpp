from pathlib import Path
import urllib.request,json,time,statistics,sys
R=Path('/home/svt/tuning27b-next/service')/(sys.argv[1] if len(sys.argv)>1 else 'http_current');R.mkdir(exist_ok=True);BASE='http://127.0.0.1:8080'
def post(path,data,timeout=300):
 req=urllib.request.Request(BASE+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)
with urllib.request.urlopen(BASE+'/v1/models',timeout=10) as response:models=json.load(response)
assert models['data'][0]['id']=='qwen3.8-27b-int4',models
(R/'v1_models.json').write_text(json.dumps(models,indent=2))
chat=post('/v1/chat/completions',{'model':'qwen3.8-27b-int4','messages':[{'role':'system','content':'Reply with exactly INT4 ready and no other text.'},{'role':'user','content':'Confirm that you are ready.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}})
(R/'chat_smoke.json').write_text(json.dumps(chat,indent=2));print('CHAT',chat['choices'][0]['message'],flush=True)
assert chat['choices'][0]['message'].get('content','').strip(),'No chat content'
text=('The quick brown fox jumps over the lazy dog. This is a local inference throughput check. Read the text and continue naturally. '*500)
tokens=post('/tokenize',{'content':text,'add_special':False})['tokens'];assert len(tokens)>2048
out=[]
for n in [512,2048]:
 for rep in range(3):
  start=time.monotonic();res=post('/completion',{'prompt':tokens[:n],'n_predict':128,'temperature':0,'seed':1234,'ignore_eos':True,'cache_prompt':False,'stream':False});elapsed=time.monotonic()-start
  (R/f'http_pp{n}_tg128_r{rep}.json').write_text(json.dumps(res,indent=2))
  t=res['timings'];row={'prompt':n,'repeat':rep,'wall_s':elapsed,'timings':t,'tokens_cached':res.get('tokens_cached')};out.append(row);print(json.dumps(row),flush=True)
  assert t['prompt_n']>=n-1,(n,t)
  assert t['predicted_n']==128,t
summary={str(n):{'prefill_tps':statistics.median(x['timings']['prompt_per_second'] for x in out if x['prompt']==n and x['repeat']>0),'decode_tps':statistics.median(x['timings']['predicted_per_second'] for x in out if x['prompt']==n and x['repeat']>0)} for n in [512,2048]}
(R/'http_benchmark.json').write_text(json.dumps({'results':out,'warm_medians':summary,'cache_prompt':False,'temperature':0,'ignore_eos':True},indent=2));print('HTTP_MEDIANS',json.dumps(summary),flush=True)
