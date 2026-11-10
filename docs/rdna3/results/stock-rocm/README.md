# Stock Qwen3.8-27B: ROCm 7.2.0 vs 10.1.0

Existing Qwen3.8-27B Q4_K_M GGUF on the RX 7900 XT (20 GB), using the same local stock llama.cpp source. Both builds use gfx1100, Release, HIP graphs, native CPU flags, and matching GGML options. No model or SDK was downloaded.

| Test | ROCm 7.2.0 (tokens/s) | ROCm 10.1.0 (tokens/s) | 10.1 / 7.2 | Change |
|---|---:|---:|---:|---:|
| pp512 | 890.81 | 887.06 | 0.9958x | -0.42% |
| pp2048 | 821.62 | 820.18 | 0.9982x | -0.18% |
| tg128 | 32.71 | 32.80 | 1.0026x | +0.26% |

The measured differences are below 0.5%, with no meaningful speedup in this stock-model benchmark. Individual run medians vary by a similar amount.

Each result is the median of seven warmed repetitions pooled from two processes. The first measured repetition in each process was excluded for graph/cache warmup. Order: 7.2 A, 10.1 A, 10.1 B, 7.2 B. No GPU profiling or compilation ran during measurements.

Settings: full GPU offload (`-ngl 99`), 12 CPU threads, batch 2048, microbatch 512, Flash Attention `auto`, F16 KV cache, depth 0, no MTP or speculative decoding. Each SDK supplies its compiler and runtime/library stack, so the comparison includes compiler changes as well as runtime/library changes.

ROCm 7.2.0: `/opt/rocm-7.2.0`, Clang 22. ROCm 10.1.0 SDK: `/mnt/data/venvs/rocm101/lib/python3.12/site-packages`, Clang 24. The runtime library paths were checked with `ldd`; the corresponding outputs are saved alongside this report.

Model: `/mnt/9100pro/ossicle-work/qwen38-27b-Q4_K_M.gguf`. Source: `/tmp/claude-1000/up/src` (existing archive; commit metadata unavailable). Source fingerprint and binary hashes are in `manifest.json`.

Raw commands, environment, full JSON timing samples and stderr are in `runs/`. Aggregates are in `results.json`. `configure72.json` records the local-only 7.2 build setup; `compare.py` contains the benchmark runner.
