# RX 7900 XT profiling

The selected 27B profile is in [summary.json](profiles/27b-494/profile_selected/summary.json). Decode is normalized per token, with 474 kernel launches per token. Instrumented wall time is 21.020 ms/token, kernel union is 19.266 ms, and the interval uncovered by kernels/copies is 1.711 ms. These trace timings include profiling overhead.

| Decode operation | GPU ms/token | GPU time share |
|---|---:|---:|
| FFN gate + up INT4 matvec | 8.001 | 41.5% |
| FFN down INT4 matvec + rotation | 3.867 | 20.1% |
| GDN QKV + z INT4 matvec | 2.690 | 14.0% |
| GDN output INT4 matvec + rotation | 1.444 | 7.5% |
| Vocabulary output head | 0.860 | 4.5% |
| Attention QKV INT4 matvec | 0.783 | 4.1% |
| GDN recurrent update | 0.554 | 2.9% |
| RMS norm + rotation + packing | 0.451 | 2.3% |
| Attention output matvec + rotation | 0.407 | 2.1% |
| Flash attention | 0.136 | 0.7% |

The IQ4_XS head reduces its trace time from 1.335 ms to 0.860 ms. All selected decode kernels report zero scratch bytes. See [latency-sorted decode groups](profiles/27b-494/profile_selected/decode/groups.csv) and [per-kernel detail](profiles/27b-494/profile_selected/decode/kernels.csv).

## Prefill, sorted by latency

For pp512 fast, instrumented kernel time totals 246.830 ms and wall time 256.770 ms. Quality totals 411.948 ms kernel time and 421.939 ms wall time. The largest fast operations are:

| Operation | GPU ms per pp512 |
|---|---:|
| FFN gate + up | 91.202 |
| FFN down | 48.676 |
| GDN QKV | 20.606 |
| GDN output | 13.406 |
| GDN z | 12.886 |
| Rotation + activation packing | 9.044 |
| Flash attention | 8.885 |

Full rankings: [fast groups](profiles/27b-494/profile_selected/prefill_fast_512/groups.csv), [quality groups](profiles/27b-494/profile_selected/prefill_quality_512/groups.csv). The analyzer is preserved as [analyze.py](experiments/27b-494/analyze.py).

## Launch gaps

The earlier trace showed roughly 3 microseconds between dependent launches. A 512-step dependent-kernel graph microbenchmark measured 2.69-2.73 microseconds per step without profiling versus 4.39 with profiling. Instrumentation adds roughly 1.7 microseconds per dispatch in that microbenchmark; the entire observed gap cannot be treated as removable GPU idle time.

AQL packet auditing found device-scope acquire/release on interior graph packets, with system scope at the graph boundaries. Graph scheduling, queue counts, direct dispatch, SDMA and barrier settings did not produce a useful improvement. The input-upload batching change reduces host synchronization at token boundaries while retaining source storage until copies finish.

The raw [runtime trace archive](archive/27b-4785-runtime_profile.tar.gz), [microbenchmark trace](archive/27b-4785-gap_micro_profile.tar.gz), [microbenchmark source](experiments/27b-4785/gap_microbench.hip) and [AQL auditor](experiments/27b-4785/aql_audit.cpp) preserve the investigation.

## Candidate kernels

WMMA decode, GDN producer fusion/prepacking, native A8 DOT8, tile/wave sweeps, post-normalization publication and chunked eight-row output tiles were tested. None replaced the selected source. The selective A4 decode candidate is also archived separately because it changes the accuracy tradeoff. Invalid group-scale packing trials are preserved as failed experiments, not benchmark results. Commands and outcomes remain in `results/27b-494` and `results/27b-4785`.
