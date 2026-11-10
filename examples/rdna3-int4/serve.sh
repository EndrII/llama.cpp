#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run.sh" llama-server \
    --alias qwen3.8-27b-int4 --host 127.0.0.1 --port 8080 \
    -tb 12 -b 2048 -ub 512 -c 65536 -np 1 -ctk f16 -ctv f16 -fa on \
    --metrics --perf --timeout 3600 --no-cache-idle-slots --cache-ram 0 \
    --ctx-checkpoints 0 --fit on --fit-target 664 --spec-type none "$@"
