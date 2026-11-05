# Reproduce the local INT4 path

Use an existing ROCm 10.1 SDK and the rotated GPTQ GGUF. No model or SDK is downloaded by these commands. The reference compiler is AMD clang 24, targeting gfx1100. [CMakeCache.reference.txt](results/selected/CMakeCache.reference.txt) records the original configuration. The new raw-buffer intrinsics were validated with this compiler; the stock ROCm 7.2 comparison is a separate build.

```bash
export ROCM_SITE=/path/to/existing/rocm101/site-packages
export PATH="$ROCM_SITE/_rocm_sdk_devel/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_SITE/_rocm_sdk_devel/lib:$ROCM_SITE/_rocm_sdk_core/lib:$ROCM_SITE/_rocm_sdk_libraries/lib"
cmake -S . -B build-rdna3 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER="$ROCM_SITE/_rocm_sdk_devel/bin/amdclang++" \
  -DCMAKE_PREFIX_PATH="$ROCM_SITE/_rocm_sdk_devel;$ROCM_SITE/_rocm_sdk_core;$ROCM_SITE/_rocm_sdk_libraries" \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_NATIVE=ON \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON
cmake --build build-rdna3 --target llama-bench llama-perplexity llama-server -j 8
```

## IQ4_XS output-head cache

The included [quantize_head.cpp](../../examples/rdna3-int4/quantize_head.cpp) reads the original Q6_K `output.weight` tensor and writes a raw IQ4_XS cache. Its dimensions and byte offset apply only to the measured GGUF: K=5120, M=248320, offset=14795624320. The generator creates the output exclusively and will refuse to overwrite an existing file.

```bash
c++ -O3 -fopenmp -Iggml/include -Iggml/src \
  examples/rdna3-int4/quantize_head.cpp -Lbuild-rdna3/bin -lggml-base \
  -Wl,-rpath,"$PWD/build-rdna3/bin" -o build-rdna3/bin/quantize-head
build-rdna3/bin/quantize-head /path/to/qwen38-27b-rotq4-gptq.gguf \
  14795624320 23 /path/to/head_iq4_xs.bin
sha256sum /path/to/head_iq4_xs.bin
```

Expected cache: 675430400 bytes, SHA-256 `65ea8897fb9dd7c46de9a4246a3be034547f018cdaa21e155387ced727493998`. Type 23 is IQ4_XS in this source. The GGUF body remains rotated GPTQ INT4; the output head uses IQ4_XS.

## Benchmark and serve

```bash
export INT4_BIN="$PWD/build-rdna3/bin"
export INT4_MODEL=/path/to/qwen38-27b-rotq4-gptq.gguf
export INT4_HEAD=/path/to/head_iq4_xs.bin
export LD_LIBRARY_PATH="$INT4_BIN:$LD_LIBRARY_PATH"
examples/rdna3-int4/run.sh llama-bench -p 512,2048 -n 0 -r 5 -o json
examples/rdna3-int4/run.sh llama-bench -p 0 -n 128 -r 5 -o json
examples/rdna3-int4/serve.sh
```

The launcher selects fast prefill, four-row 27B matvecs, the IQ4_XS head and batched input uploads. Set `GGML_INT4_PREFILL=quality` for the quality prefill preset. The reference machine uses 12 threads, batch 2048, microbatch 512, F16 KV and Flash Attention. For throughput summaries, discard each process's first cold repetition; exact commands and samples are retained in `results/`.

The server uses the model alias `qwen3.8-27b-int4`, context 65536, one slot, F16 K/V, and `--ctx-checkpoints 0`. Disabling recurrent-state checkpoints avoids slow small-batch prompt tails and preserves the measured 2k prefill. Edited prompts may need more recomputation. `--spec-type none` disables speculation. The full local deployment and validation artifacts are in [deployment](deployment/README.md).

## Quantization workflow

`quantization/gptq_layerwise.py` reads BF16 GGUF shards one layer at a time and emits rotated GPTQ blocks. `rotate_quantize_split.py` writes the rotated GGUF using those blocks. The archive also contains the original 4B/9B rotation/GPTQ tools. Their command-line help and original absolute paths are preserved; adjust the GGUF Python-module path and local calibration/model paths before reuse. No weights or calibration datasets are checked into this fork.
