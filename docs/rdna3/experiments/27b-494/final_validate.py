from run import run
import subprocess,sys
from pathlib import Path
base={'GGML_CUDA_BATCH_INPUTS':'1','GGML_INT4_HEAD_REQUANT':'iq4_xs','GGML_INT4_HEAD_FILE':'/mnt/data/tuning27b-next/head_iq4_xs.bin'}
for label,baseline,reps in [('baseA',True,5),('newA',False,5),('baseB',True,4),('newB',False,4)]:
 run({'label':'final_decode_'+label,'baseline':baseline,**({} if baseline else {'binary':'/home/svt/tuning27b-next/selected/bin'}),'env':{} if baseline else base,'args':['-p','0','-n','128','-r',str(reps),'-o','json']})
run({'label':'final_decode512','binary':'/home/svt/tuning27b-next/selected/bin','env':base,'args':['-p','0','-n','512','-r','3','-o','json']})
subprocess.run([sys.executable,'/home/svt/tuning27b-next/final_profile.py'],check=True)
for mode in ['quality','fast','decode']:
 run({'label':'selected_ppl_'+mode,'binary':'/home/svt/tuning27b-next/selected/bin','tool':'llama-perplexity','env':dict(base,GGML_INT4_PREFILL='quality' if mode=='decode' else mode),'args':['-f','/mnt/data/projects/ossicle-gguf/work/wiki.test.raw','-c','512','-b','1' if mode=='decode' else '512']+(['-ub','1'] if mode=='decode' else [])+['--kl-divergence','--kl-divergence-base','/home/svt/tuning27b/decode_reference.bin' if mode=='decode' else '/mnt/9100pro/ossicle-work/kld_base_qwen38_27b.bin']})
