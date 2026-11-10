#!/bin/bash
set -euo pipefail
TUNING_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROCM_SITE=/mnt/data/venvs/rocm101/lib/python3.12/site-packages
TUNING_TOOL=${1:-llama-bench}
if [ "$#" -gt 0 ]; then shift; fi
case "$TUNING_TOOL" in
    llama-bench|llama-perplexity) ;;
    *) echo 'First argument must be llama-bench or llama-perplexity.' >&2; exit 2 ;;
esac
exec env -i HOME=/home/svt PATH=/usr/bin:/bin HIP_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH="$TUNING_ROOT/bin:$ROCM_SITE/_rocm_sdk_devel/lib:$ROCM_SITE/_rocm_sdk_core/lib:$ROCM_SITE/_rocm_sdk_libraries/lib" \
    GGML_INT4_PREFILL="${GGML_INT4_PREFILL:-quality}" \
    GGML_INT4_27B_ROWS="${GGML_INT4_27B_ROWS:-4}" \
    "$TUNING_ROOT/bin/$TUNING_TOOL" \
    -m /mnt/data/work27b/qwen38-27b-rotq4-gptq.gguf -ngl 99 -t 12 "$@"
