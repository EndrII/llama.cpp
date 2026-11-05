#!/usr/bin/env bash
set -euo pipefail
args=()
skip_next=0
model_next=0
legacy_model='llama.cpp//mnt/9100pro/models/qwen3.8-27b-ud3-iq4xs/Qwen3.8-27B-UD-IQ4_XS.gguf'
current_model='llama.cpp/qwen3.8-27b-int4'
for arg in "$@"; do
  if (( skip_next )); then skip_next=0; continue; fi
  if (( model_next )); then
    if [[ "$arg" == "$legacy_model" ]]; then arg="$current_model"; fi
    args+=("$arg")
    model_next=0
    continue
  fi
  case "$arg" in
    --thinking) skip_next=1 ;;
    --thinking=*) ;;
    --model) args+=("$arg"); model_next=1 ;;
    --model=*)
      value="${arg#--model=}"
      if [[ "$value" == "$legacy_model" ]]; then value="$current_model"; fi
      args+=("--model=$value")
      ;;
    *) args+=("$arg") ;;
  esac
done
exec /home/svt/.local/bin/omp "${args[@]}" --thinking medium
