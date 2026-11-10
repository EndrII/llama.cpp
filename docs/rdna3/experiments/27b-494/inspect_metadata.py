import json,subprocess,sys,yaml
from pathlib import Path
from run import ROOT,PATHS,SDK
r=ROOT;name=sys.argv[1];p=Path(PATHS['build'])/'ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda'/(name+'.cu.o');b=SDK/'_rocm_sdk_devel/lib/llvm/bin'
f=r/(name+'.fat');co=r/(name+'.co')
for q in [f,co]:
 if q.exists():q.unlink()
subprocess.run([str(b/'llvm-objcopy'),'--dump-section','.hip_fatbin='+str(f),str(p)],check=True)
subprocess.run([str(b/'clang-offload-bundler'),'--unbundle','--type=o','--targets=hip-amdgcn-amd-amdhsa--gfx1100','--input='+str(f),'--output='+str(co)],check=True)
s=subprocess.check_output([str(b/'llvm-readobj'),'--notes',str(co)],text=True);a=s.index('---\n');z=s.index('\n...',a)+4;ks=yaml.safe_load(s[a:z])['amdhsa.kernels'];ns=subprocess.check_output(['c++filt']+[k['.name'] for k in ks],text=True).splitlines()
rows=[dict(name=n,vgpr=k['.vgpr_count'],sgpr=k['.sgpr_count'],scratch=k['.private_segment_fixed_size'],lds=k['.group_segment_fixed_size']) for n,k in zip(ns,ks)]
(r/('metadata_'+name+'.json')).write_text(json.dumps(rows,indent=2))
for x in rows:
 if 'gated_delta_net_cols' in x['name'] or 'k_mv_rot_rows<1,' in x['name']:print(x['vgpr'],x['scratch'],x['name'].split('>(')[0]+'>')
