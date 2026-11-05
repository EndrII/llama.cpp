#!/usr/bin/env bash
set -euo pipefail
TUNING_ROOT=/home/svt/tuning27b-next/selected
ROCM_SITE=/mnt/data/venvs/rocm101/lib/python3.12/site-packages
export PATH="/home/svt/.local/bin:/usr/local/bin:/usr/bin:/bin"
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$TUNING_ROOT/bin:$ROCM_SITE/_rocm_sdk_devel/lib:$ROCM_SITE/_rocm_sdk_core/lib:$ROCM_SITE/_rocm_sdk_libraries/lib"
export GGML_INT4=3
export GGML_INT4_INPLACE=1
export GGML_INT4_PREFILL=fast
export GGML_INT4_27B_ROWS=4
export GGML_INT4_HEAD_REQUANT=iq4_xs
export GGML_INT4_HEAD_FILE=/mnt/data/tuning27b-next/head_iq4_xs.bin
export GGML_CUDA_BATCH_INPUTS=1
exec "$TUNING_ROOT/bin/llama-server" \
  -m /mnt/data/work27b/qwen38-27b-rotq4-gptq.gguf \
  --alias qwen3.8-27b-int4 \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -t 12 -tb 12 -b 2048 -ub 512 \
  -c 65536 -np 1 -ctk f16 -ctv f16 -fa on \
  --metrics --perf --timeout 3600 \
  --no-cache-idle-slots --cache-ram 0 --ctx-checkpoints 0 \
  --fit on --fit-target 664 \
  --spec-type none \
  --cors-origins 'https://obsidian.tail907083.ts.net' \
  --ui-config '{"systemMessage":"For every user message, immediately call omp-agent_use_omp. Put the complete user request plus any relevant conversation context into task; use cwd /home/svt unless the user identifies another directory. Do not solve, analyze, search, or answer the request yourself. After the tool returns, relay its result faithfully. If the result asks a question, ask the user; when they answer, call omp-agent_use_omp again with the original task, the question, and their answer."}' \
  --mcp-servers-config /home/svt/.llama-mcp.json \
  "$@"
