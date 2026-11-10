#include <hsa/hsa_api_trace.h>
#include <atomic>
#include <cstdio>
static decltype(hsa_amd_queue_intercept_create)* create_queue;
static decltype(hsa_amd_queue_intercept_register)* register_queue;
static std::atomic<unsigned long long> counts[65536];
static decltype(hsa_amd_queue_create)* amd_create;
static decltype(hsa_amd_queue_set_priority)* set_priority;
static decltype(hsa_amd_queue_cu_set_mask)* set_mask;
static void intercept(const void* packets, uint64_t count, uint64_t, void*, hsa_amd_queue_intercept_packet_writer writer) {
 const auto * bytes=static_cast<const unsigned char*>(packets);
 for(uint64_t i=0;i<count;i++) {
  uint16_t h; __builtin_memcpy(&h,bytes+64*i,2);counts[h].fetch_add(1,std::memory_order_relaxed);
 }
 writer(packets,count);
}
static hsa_status_t queue_create(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
 void (*callback)(hsa_status_t,hsa_queue_t*,void*), void* data, uint32_t priv, uint32_t group,hsa_queue_t** queue) {
 auto status=create_queue(agent,size,type,callback,data,priv,group,queue);
 if(status==HSA_STATUS_SUCCESS) status=register_queue(*queue,intercept,nullptr);
 return status;
}
static hsa_status_t amd_queue_create(hsa_agent_t agent,hsa_amd_queue_create_desc_t* descs,uint32_t n) {
 for(uint32_t i=0;i<n;i++) if(descs[i].engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE) return amd_create(agent,descs,n);
 for(uint32_t i=0;i<n;i++) {
  auto& d=descs[i];
  auto status=queue_create(agent,d.queue_size_bytes/64,d.engine.compute.type,d.callback,d.callback_data,d.engine.compute.private_segment_size,UINT32_MAX,&d.queue);
  if(status!=HSA_STATUS_SUCCESS) return status;
  set_priority(d.queue,d.priority);
  if(d.engine.compute.cu_mask_count) set_mask(d.queue,d.engine.compute.cu_mask_count,d.engine.compute.cu_mask);
 }
 return HSA_STATUS_SUCCESS;
}
extern "C" bool OnLoad(HsaApiTable* table,uint64_t,uint64_t,const char*const*) {
 create_queue=table->amd_ext_->hsa_amd_queue_intercept_create_fn;
 register_queue=table->amd_ext_->hsa_amd_queue_intercept_register_fn;
 table->core_->hsa_queue_create_fn=queue_create;
 amd_create=table->amd_ext_->hsa_amd_queue_create_fn;
 set_priority=table->amd_ext_->hsa_amd_queue_set_priority_fn;
 set_mask=table->amd_ext_->hsa_amd_queue_cu_set_mask_fn;
 table->amd_ext_->hsa_amd_queue_create_fn=amd_queue_create;
 std::fprintf(stderr,"AQL_AUDIT enabled\n");return true;
}
extern "C" __attribute__((destructor)) void OnUnload() {
 for(unsigned h=0;h<65536;h++) if(auto n=counts[h].load())
  std::fprintf(stderr,"AQL_AUDIT header=%04x type=%u barrier=%u acquire=%u release=%u count=%llu\n",h,h&255,(h>>8)&1,(h>>9)&3,(h>>11)&3,n);
}
