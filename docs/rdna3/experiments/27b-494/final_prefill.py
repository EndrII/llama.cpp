from run import run
base={'GGML_CUDA_BATCH_INPUTS':'1','GGML_INT4_HEAD_REQUANT':'iq4_xs','GGML_INT4_HEAD_FILE':'/mnt/data/tuning27b-next/head_iq4_xs.bin'}
for mode in ['quality','fast']:
 run({'label':'final4785_pp_'+mode,'baseline':True,'env':{'GGML_INT4_PREFILL':mode},'args':['-p','512,2048','-n','0','-r','5','-o','json']})
for ub in [512,768,1024]:
 for mode in ['quality','fast']:
  run({'label':f'final_xs_ub{ub}_pp_{mode}','env':dict(base,GGML_INT4_PREFILL=mode),'args':['-p','512,2048','-n','0','-r','5','-ub',str(ub),'-o','json']})
