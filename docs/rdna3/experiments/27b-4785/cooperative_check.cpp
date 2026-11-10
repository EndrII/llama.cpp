#include <hip/hip_runtime_api.h>
#include <cstdio>
int main(){hipDeviceProp_t p{};auto e=hipGetDeviceProperties(&p,0);std::printf("status=%d name=%s CUs=%d cooperative=%d\n",e,p.name,p.multiProcessorCount,p.cooperativeLaunch);}
