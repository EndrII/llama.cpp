import json,sys,statistics as st
t=sys.stdin.read(); d=json.loads(t[t.index('['):])
out={}
for x in d:
    k=('pp%d'%x['n_prompt']) if x['n_prompt'] and not x['n_gen'] else ('tg%d@d%d'%(x['n_gen'],x.get('n_depth',0)))
    s=x['samples_ts'][1:]
    out[k]=round(st.median(s),1)
print(json.dumps(out))
