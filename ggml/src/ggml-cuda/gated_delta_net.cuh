#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// the GET_ROWS that gathers the recurrent state of `state_view` was skipped: the decode kernel reads the cache row itself
void ggml_cuda_gdn_skip_gather(const ggml_tensor * state_view, const ggml_tensor * gather);

// the L2 norms of q and k feeding `gdn` were skipped: the decode kernel normalizes q_src / k_src itself
void ggml_cuda_gdn_fold_l2(const ggml_tensor * gdn, ggml_tensor * q_src, ggml_tensor * k_src, float eps);
bool ggml_cuda_gdn_has_rows(const ggml_tensor * state_view);
void ggml_cuda_gdn_reset();
