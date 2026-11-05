# Qwen Rotated INT4 on AMD RDNA3

This fork contains the rotated GPTQ INT4 path developed for Qwen3.5-4B, tuned for Qwen3.5-9B, and extended to Qwen3.8-27B. All rates below use one RX 7900 XT, without MTP or speculative decoding.

| Model | Prefill pp512 (tokens/s) | Prefill pp2048 (tokens/s) | Decode (tokens/s) |
|---|---:|---:|---:|
| Qwen3.5-4B, fast | 11,012 | 10,624 | 200.6 |
| Qwen3.5-9B, fast | 6,581 | 6,558 | 125.8 |
| Qwen3.8-27B, fast | 2,013 | 1,930 | **50.1** |

The 27B decode figure is the observed peak. The 4B numbers are from Sachin Mahesh, *Qwen3.5-4B on the Radeon RX 7900 XT: 11,000 tokens/s INT4 Prefill and 200 tokens/s Decode*, section 4, table `tab:stock` ([PDF](papers/qwen35_int4_summary.pdf), [TeX](papers/qwen35_int4_summary.tex#L165)). Its GPU comparison table also includes the 9B and 27B follow-up. The 9B pp2048 comes from the [local comparison log](results/claude/qwen/base_xt.txt); 27B prefill comes from the [selected prefill run](results/27b-494/final_xs_ub512_pp_fast/summary.json).

## Paper comparisons

The paper's GPU table reports the following pp512/tg128 rates. Dashes are unavailable measurements. GPU results are comparisons across quantizations and sources, rather than controlled hardware substitutions.

| GPU | 4B pp512 | 4B tg128 | 9B pp512 | 9B tg128 | 27B pp512 | 27B decode |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 16,036 | 313.6 | 11,721 | 220.5 | 3,881 | 79 |
| RTX 4090 | 14,882 | 220.2 | 9,762 | 141.6 | 3,417 | 51 |
| RTX 5080 | 10,057 | 200.5 | 6,543 | 136.1 | - | 50 |
| RTX 5070 Ti | 8,889 | 187.6 | 5,872 | 126.0 | - | - |
| RTX 5070 | 7,449 | 154.5 | 4,652 | 101.3 | - | - |
| RTX 3090 | 7,366 | 179.1 | 4,632 | 122.1 | 1,507 | 43 |
| RTX 5060 Ti 16 GB | 4,776 | 107.6 | 2,955 | 70.0 | - | - |

Source: [summary paper, GPU comparison table](papers/qwen35_int4_summary.tex#L61). The 5080 27B entry is the reported IQ4_XS result without MTP/speculation; the other external rows are preserved in the [leaderboard CSVs](results/leaderboard). The XT's latest peak is shown in the first table.

## Quality and memory

The paper distinguishes its two prefill presets. For 4B, `quality` measures 7,140/7,013 tokens/s pp512/pp2048 and 200.7 tg128, with +1.92% perplexity versus BF16 (KLD 0.0241). `fast` measures 11,012/10,624 and 200.6, with +8.67% perplexity (KLD 0.0969). The Q4_K_M baseline is 5,477/5,387 and 145.2, with +2.00% perplexity. Peak VRAM is 3,358 MiB for rotated quality versus 3,621 MiB for Q4_K_M; the paper uses the quality measurement for fast's identical weight storage. See [section 4 and its table](papers/qwen35_int4_summary.tex#L165).

For the selected 27B path, 24 chunks of WikiText-2 at context 512 give PPL 5.975924 / KLD 0.024445 in quality mode and PPL 6.125678 / KLD 0.060487 in fast mode. Relative to the preceding compact-scale checkpoint, PPL changes by +0.11% / +0.24%. Full-logit evaluation's sampled whole-GPU peak falls from about 17.72 GiB to 15.07 GiB with the IQ4_XS head. These are [recorded evaluation results](results/27b-494/selected_ppl_quality/summary.json), [fast results](results/27b-494/selected_ppl_fast/summary.json), and [preceding checkpoint results](results/27b-4785). Short batch-1 decode evaluation uses a separate reference; its KLD must not be compared directly to the BF16 prefill reference.

## Source and reproduction

The active source is the preserved, deployed selection: compact 4-bit group scales; raw-buffer decode reads; four-row rotated output matvec; 48-head GDN gate fusion; the zero-spill skinny matmul; optional IQ4_XS vocabulary head; and batched small host-input uploads. The 9B four-wave GEMM and earlier attention-combine/prologue work are included in the branch history.

- [Build and run](REPRODUCE.md)
- [Decode and prefill profiles, sorted by latency](PROFILING.md)
- [ROCm 7.2 versus 10.1 stock comparison](results/stock-rocm/README.md)
- [OMP, server and Paseo setup](deployment/README.md)
- [Qwen paper](papers/qwen35_int4_summary.pdf), [source](papers/qwen35_int4_summary.tex), figures and earlier drafts in `papers/`

The paper's 4B/9B results describe the historical builds used for those measurements. The selected 27B source was compiled and validated before publication, including real HTTP inference and OMP/Paseo requests. Publication copies that source unchanged; it does not rerun GPU benchmarks against the live service.

## Work archive

`results/` contains commands, environments, samples and evaluation logs. `profiles/` contains readable summaries and latency-sorted CSVs; `archive/` contains compressed raw traces. `experiments/` preserves candidate kernels, scripts, patches, compiler metadata and unsuccessful trials. Those files are research records and are not built by CMake. Their original absolute paths remain intact for provenance. `quantization/` preserves the rotation/GPTQ tools, including the layerwise 27B workflow.

`artifact-origins.json` maps copied artifacts to their local sources. `SHA256SUMS` covers the publication artifacts. Model weights, generated head caches, reference logits, SDKs, datasets, binaries and personal application state are excluded. The head-cache generator and its expected hash are included.

## Additional model support

Also added INT4 acceleration for **Ossicle 4B**, a Qwen3.5-4B fine-tune: **10,934 tokens/s prefill (pp512)**, **10,606 tokens/s prefill (pp2048)** and **200.6 tokens/s decode (tg128)** on the RX 7900 XT. See [section 5 of the Qwen paper](papers/qwen35_int4_summary.tex#L194). The earlier Ossicle case study is preserved as a [work log](papers/ossicle_int4_report.pdf), with its [source](papers/ossicle_int4_report.tex).
