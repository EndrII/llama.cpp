import shutil
from run import run,ROOT,PATHS
from pathlib import Path
args=['-p','0','-n','128','-r','4','-o','json']
run({'label':'row_nw_control','args':args})
shutil.copy2(ROOT/'scale8.so',ROOT/'bin/libggml-hip.so.0.18.0')
shutil.copy2(ROOT/'scale8.cu',Path(PATHS['source'])/'ggml/src/ggml-cuda/int4-wmma.cu')
for label,env,base in [
 ('scale8_tg_control',{},False),
 ('baseline_no_graph',{'GGML_CUDA_DISABLE_GRAPHS':'1'},True),
 ('scale8_no_graph',{'GGML_CUDA_DISABLE_GRAPHS':'1'},False),
 ('scale8_classic',{'DEBUG_HIP_GRAPH_CLASSIC_PATH':'1'},False),
 ('scale8_one_queue',{'DEBUG_HIP_FORCE_GRAPH_QUEUES':'1'},False),
 ('scale8_barrier_off',{'DEBUG_CLR_AQL_BARRIER_OPT':'0'},False),
 ('scale8_batch1024',{'DEBUG_HIP_GRAPH_BATCH_SIZE':'1024'},False)]:
 run({'label':label,'args':args,'env':env,'baseline':base})
