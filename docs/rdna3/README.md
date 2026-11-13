# Qwen Rotated INT4 on AMD RDNA3

This fork contains the rotated GPTQ INT4 path developed for Qwen3.5-4B, tuned for Qwen3.5-9B, and extended to Qwen3.8-27B. All rates below use one RX 7900 XT, without MTP or speculative decoding.

| Model | Prefill pp512 (tokens/s) | Prefill pp2048 (tokens/s) | Decode (tokens/s) |
|---|---:|---:|---:|
| Qwen3.5-4B, fast | 11,012 | 10,624 | 200.6 |
| Qwen3.5-9B, fast | 6,624 | 6,561 | 126.2 |
| Qwen3.8-27B, fast | 2,013 | 1,930 | **50.1** |

Source: Sachin Mahesh, *Rotated GPTQ-INT4 WMMA Inference on the Radeon RX 7900 XT in Llama.cpp*, November 3, 2026 ([latest merged report](papers/rdna3_int4_report.pdf), [LaTeX source](papers/rdna3_int4_report.tex)). Tables 4, 6 and 7 give the model results. The 27B decode figure is the observed peak. The copy here matches the report supplied by the author byte for byte.

## Paper comparisons

The following table reproduces the latest report's Table 1. Dashes are unavailable measurements.

| GPU | 4B pp512 | 4B tg128 | 9B pp512 | 9B tg128 | 27B pp512 | 27B decode |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 16,036 | 313.6 | 11,721 | 220.5 | 3,213 | 81.8 |
| RTX 4090 | 14,882 | 220.2 | 9,762 | 141.6 | 2,510 | 49.8 |
| **RX 7900 XT, this work** | **11,012** | **200.6** | **6,624** | **126.2** | **2,013** | **50.1** |
| RTX 5080 | 10,057 | 200.5 | 6,543 | 136.1 | 1,673 | 49.4 |
| RTX 5070 Ti | 8,889 | 187.6 | 5,872 | 126.0 | 1,276 | 42.1 |
| RTX 5070 | 7,449 | 154.5 | 4,652 | 101.3 | - | - |
| RTX 3090 | 7,366 | 179.1 | 4,632 | 122.1 | 1,042 | 43 |
| RTX 5060 Ti 16 GB | 4,776 | 107.6 | 2,955 | 70.0 | 815 | 27.6 |

Source: [latest report, Table 1](papers/rdna3_int4_report.pdf#page=4). External 27B results use IQ4_XS, without MTP/speculation.

## Quality and memory

The paper distinguishes its two prefill presets. For 4B, `quality` measures 7,140/7,013 tokens/s pp512/pp2048 and 200.7 tg128, with +1.92% perplexity versus BF16 (KLD 0.0241). `fast` measures 11,012/10,624 and 200.6, with +8.67% perplexity (KLD 0.0969). The Q4_K_M baseline is 5,477/5,387 and 145.2, with +2.00% perplexity. Peak VRAM is 3,358 MiB for rotated quality versus 3,621 MiB for Q4_K_M; the paper uses the quality measurement for fast's identical weight storage. See [section 5 and its table](papers/rdna3_int4_report.tex#L214).

For 9B, the report gives quality / fast PPL changes of +1.27% / +4.50% and KLD 0.0188 / 0.0713, with 5,463 MiB quality peak VRAM. For 27B, the corresponding PPL changes are +1.27% / +4.55% and KLD 0.0184 / 0.0565; peak VRAM is 15,893 MiB for quality and 15,853 MiB for fast. See [Table 6](papers/rdna3_int4_report.pdf#page=8) and [Table 7](papers/rdna3_int4_report.pdf#page=9). Separate later accuracy runs and their references remain in the raw archive.

## Source and reproduction

The active source is the preserved, deployed selection: compact 4-bit group scales; raw-buffer decode reads; four-row rotated output matvec; 48-head GDN gate fusion; the zero-spill skinny matmul; optional IQ4_XS vocabulary head; and batched small host-input uploads. The 9B four-wave GEMM and earlier attention-combine/prologue work are included in the branch history.

- [Build and run](REPRODUCE.md)
- [Decode and prefill profiles, sorted by latency](PROFILING.md)
- [ROCm 7.2 versus 10.1 stock comparison](results/stock-rocm/README.md)
- [Qwen paper](papers/rdna3_int4_report.pdf), [source](papers/rdna3_int4_report.tex), figures and earlier drafts in `papers/`

The paper's 4B/9B results describe the historical builds used for those measurements. The selected 27B source was compiled and validated before publication, including real HTTP inference. Publication copies that source unchanged; it does not rerun GPU benchmarks against the live service.

## Work archive

`results/` contains commands, environments, samples and evaluation logs. `profiles/` contains readable summaries and latency-sorted CSVs; `archive/` contains compressed raw traces. `experiments/` preserves candidate kernels, scripts, patches, compiler metadata and unsuccessful trials. Those files are research records and are not built by CMake. Their original absolute paths remain intact for provenance. `quantization/` preserves the rotation/GPTQ tools, including the layerwise 27B workflow.

`artifact-origins.json` maps copied artifacts to their local sources. `SHA256SUMS` covers the publication artifacts. Model weights, generated head caches, reference logits, SDKs, datasets, binaries and personal application state are excluded. The head-cache generator and its expected hash are included.

## Additional model support

Also added INT4 acceleration for **Ossicle 4B**, a Qwen3.5-4B fine-tune: **10,934 tokens/s prefill (pp512)**, **10,606 tokens/s prefill (pp2048)** and **200.6 tokens/s decode (tg128)** on the RX 7900 XT. See [section 5.1 of the report](papers/rdna3_int4_report.tex#L243). The earlier Ossicle case study is preserved as a [work log](papers/ossicle_int4_report.pdf), with its [source](papers/ossicle_int4_report.tex).
