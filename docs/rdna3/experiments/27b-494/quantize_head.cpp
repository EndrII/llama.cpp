#include "ggml.h"
#include "ggml-quants.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#include <sys/prctl.h>
int main(int argc,char**argv){
 prctl(PR_SET_DUMPABLE,0);
 if(argc!=5)return 2;
 const int K=5120,M=248320;const off_t offset=atoll(argv[2]);const ggml_type type=(ggml_type)atoi(argv[3]);
 int in=open(argv[1],O_RDONLY),out=open(argv[4],O_CREAT|O_EXCL|O_RDWR,0644);if(in<0||out<0){perror("open");return 1;}
 ggml_quantize_init(type);const size_t row=ggml_row_size(type,K),bytes=row*M;
 if(ftruncate(out,bytes)){perror("ftruncate");return 1;}
 char* data=(char*)mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,out,0);if(data==MAP_FAILED)return 1;
 const bool needs=ggml_quantize_requires_imatrix(type);std::vector<float> importance(K,1.f);std::atomic<int> done(0);double start=omp_get_wtime();
 #pragma omp parallel num_threads(12)
 {
  constexpr int CHUNK=32;std::vector<block_q6_K> q(CHUNK*K/256);std::vector<float> f(CHUNK*K);
  #pragma omp for schedule(dynamic)
  for(int first=0;first<M;first+=CHUNK){
   int count=std::min(CHUNK,M-first);size_t nb=count*K/256*sizeof(block_q6_K);
   ssize_t n=pread(in,q.data(),nb,offset+(size_t)first*(K/256)*sizeof(block_q6_K));if(n!=(ssize_t)nb){perror("pread");abort();}
   dequantize_row_q6_K(q.data(),f.data(),count*K);
   size_t nw=ggml_quantize_chunk(type,f.data(),data+(size_t)first*row,0,count,K,needs?importance.data():nullptr);if(nw!=row*count)abort();
   int n_done=done.fetch_add(count)+count;if(n_done%16384==0)fprintf(stderr,"%d/%d %.1fs\n",n_done,M,omp_get_wtime()-start);
  }
 }
 msync(data,bytes,MS_SYNC);munmap(data,bytes);close(in);close(out);fprintf(stderr,"done %s %zu bytes %.1fs\n",ggml_type_name(type),bytes,omp_get_wtime()-start);
}
