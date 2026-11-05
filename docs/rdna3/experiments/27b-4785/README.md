# 27B plain-decode checkpoint

The 50+ tokens/s goal has not been reached. The selected experimental 27B build delivers approximately 47.85 tokens/s on the RX 7900 XT, with no MTP or speculative decoding. Quality prefill improved in the measured pair; fast prefill is still 0.18–0.40% below the baseline, so the complete requested constraint set is not satisfied.

This directory is an isolated experiment. The previous verified 44.2 tokens/s build at `/home/svt/tuning27b` and its repository worktree are preserved. The selected source is in `src/int4-wmma.cu`; the patch is relative to the 44.2 source. No commit or push was made. The model GGUF was not modified. This binary uses the compact scale format globally and is intended for the tested 27B model; it has not been validated as a general replacement for smaller models.

## Matched measurements

Four repetitions, first discarded, median of the remaining three. Full GPU offload, batch 2048, microbatch 512, 12 threads, F16 KV, empty initial context. Results in tokens/s.

| Workload | 44.2 baseline | Selected candidate | Change |
|---|---:|---:|---:|
| Decode tg128 | 44.13 | 47.85 | +8.44% |
| Quality pp512 | 1219.55 | 1226.01 | +0.53% |
| Quality pp2048 | 1117.34 | 1122.68 | +0.48% |
| Fast pp512 | 2012.68 | 2009.00 | -0.18% |
| Fast pp2048 | 1936.26 | 1928.57 | -0.40% |

Whole-GPU peak VRAM, sampled every 20 ms for the matched quality workload: 15.470 GiB baseline versus 15.180 GiB candidate. The internal weight allocation remains the original size; the compact metadata reduces bytes read, not the allocation. Sampled whole-GPU peaks vary with allocation state and are not an allocator proof of identical usage.

## Accuracy

Same 24 WikiText chunks, context 512, compared with the local BF16 reference. These are prefill-path PPL measurements.

| Preset | Baseline PPL | Candidate PPL | Baseline KLD | Candidate KLD |
|---|---:|---:|---:|---:|
| Quality | 5.958872 | 5.969431 | 0.018352 | 0.022534 |
| Fast | 6.152013 | 6.111169 | 0.056538 | 0.058457 |

Quality PPL increased 0.18%; fast PPL decreased 0.66%. KLD increased in both cases. Quality was measured on the same arithmetic before adding a cache-policy load hint; the hint does not change fetched values or mathematical operations. The selected max-scale build has not yet had a separate single-token PPL check. The affine alternative did pass a decode check, but that is a different scale encoding and is not evidence for this selected build.

## Selected changes

- Pack each group scale into 4 bits rather than 16, using one maximum-derived floating scale per output row. INT4 weight codes remain unchanged. The effective core read format is about 4.125 bits/weight plus row metadata.
- Stream decode weight loads through raw-buffer loads with cache-policy 6. The cache-policy sweep and non-temporal flat-load variant independently confirmed a gain.
- Keep the previous 48-head fusions and four-row-tile output matvec unchanged.

## Launch-gap investigation

The full runtime trace is in `runtime_profile`. It showed about 3.16 microseconds between dependent kernels and about 0.3 ms at the token boundary. AQL packet auditing showed device-scope acquire/release between kernels, not a system-scope flush on every dispatch. Graphs were already enabled.

A dependent-kernel microbenchmark measured about 2.69–2.73 microseconds per step without profiling and 4.39 with profiling. The profiled median kernel duration was 0.96 microseconds and the gap 3.36. Instrumentation itself therefore contributes about 1.7 microseconds per dispatch on that microbenchmark. This does not transfer exactly to the model, but it means the observed model gaps cannot all be counted as recoverable idle time.

Graph disabling, alternate graph scheduling, queue count, SDMA, kernel-argument placement, direct dispatch, batch size, and barrier-optimization settings did not improve model throughput. Fusing normalization into the producer using a completion counter was slower. Folding it into gate/up matvecs, including 32- and 64-row variants, also failed to beat the selected path. Wave-count and cache-policy results are retained in `runs`.

The earlier 44.2 profile remains useful for operation ranking: FFN gate/up, FFN down, GDN QKV/z, GDN output, then the Q6_K output head dominate. There is not yet a new complete operation profile of this selected compact build.

## Run

```bash
/home/svt/tuning27b50/run_27b.sh llama-bench -p 512,2048 -n 128 -r 4
GGML_INT4_PREFILL=fast /home/svt/tuning27b50/run_27b.sh llama-bench -p 512,2048 -n 128 -r 4
```

`runs` preserves full commands, environments, samples, accuracy logs, and summaries. `manifest.json` identifies the selected source and library. The 50+ decode target and full fast-prefill constraint remain open.
