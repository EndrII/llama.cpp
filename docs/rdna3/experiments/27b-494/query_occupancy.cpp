#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){hipDeviceProp_t p{};auto e=hipGetDeviceProperties(&p,0);if(e)return e;std::printf("CUs=%d regs/block=%d regs/MP=%d threads/MP=%d\n",p.multiProcessorCount,p.regsPerBlock,p.regsPerMultiprocessor,p.maxThreadsPerMultiProcessor);hipModule_t module;e=hipModuleLoad(&module,argv[1]);if(e){std::printf("load: %s\n",hipGetErrorString(e));return e;}for(int i=2;i+1<argc;i+=2){hipFunction_t f;e=hipModuleGetFunction(&f,module,argv[i]);if(e){std::printf("function: %s\n",hipGetErrorString(e));continue;}int blocks=0;e=hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,std::atoi(argv[i+1]),0);std::printf("%s threads=%s blocks/MP=%d status=%d\n",argv[i],argv[i+1],blocks,e);}hipModuleUnload(module);}
