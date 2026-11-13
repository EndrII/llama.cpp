#!/usr/bin/env bash
set -euo pipefail
: "${INT4_BIN:?Set INT4_BIN to the directory containing the built llama tools}"
: "${INT4_MODEL:?Set INT4_MODEL to the rotated GPTQ GGUF}"
: "${INT4_HEAD:?Set INT4_HEAD to the generated IQ4_XS head cache}"
tool=${1:-llama-bench}
if (( $# )); then shift; fi
case "$tool" in
    llama-bench|llama-perplexity|llama-server) ;;
    *) echo "Unsupported tool: $tool" >&2; exit 2 ;;
esac
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export GGML_INT4=3 GGML_INT4_INPLACE=1 GGML_INT4_27B_ROWS=4
export GGML_INT4_PREFILL="${GGML_INT4_PREFILL:-fast}"
export GGML_INT4_HEAD_REQUANT=iq4_xs GGML_INT4_HEAD_FILE="$INT4_HEAD"
export GGML_CUDA_BATCH_INPUTS=1
exec "$INT4_BIN/$tool" -m "$INT4_MODEL" -ngl 99 -t 12 "$@"
