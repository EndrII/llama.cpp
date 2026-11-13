#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#define CK(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));std::exit(1);} } while(0)
__global__ void dependent_step(unsigned * x){if(threadIdx.x==0) *x=*x+1;}
int main(){
 constexpr int N=512,R=20;
 unsigned *x;CK(hipMalloc(&x,4));CK(hipMemset(x,0,4));
 hipStream_t stream;CK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
 CK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));
 for(int i=0;i<N;i++)dependent_step<<<1,32,0,stream>>>(x);
 hipGraph_t graph;CK(hipStreamEndCapture(stream,&graph));
 hipGraphExec_t exec;CK(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 for(int i=0;i<3;i++)CK(hipGraphLaunch(exec,stream));CK(hipStreamSynchronize(stream));
 hipEvent_t start,end;CK(hipEventCreate(&start));CK(hipEventCreate(&end));
 CK(hipEventRecord(start,stream));auto t=std::chrono::steady_clock::now();
 for(int i=0;i<R;i++)CK(hipGraphLaunch(exec,stream));
 CK(hipEventRecord(end,stream));CK(hipEventSynchronize(end));
 auto wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
 float ms;CK(hipEventElapsedTime(&ms,start,end));
 unsigned out;CK(hipMemcpy(&out,x,4,hipMemcpyDeviceToHost));
 std::printf("steps=%d result=%u expected=%d event_us_per_step=%.4f wall_us_per_step=%.4f\n",N*R,out,N*(R+3),ms*1000/(N*R),wall/(N*R));
 CK(hipGraphExecDestroy(exec));CK(hipGraphDestroy(graph));CK(hipEventDestroy(start));CK(hipEventDestroy(end));CK(hipStreamDestroy(stream));CK(hipFree(x));
 return out==N*(R+3)?0:1;
}
