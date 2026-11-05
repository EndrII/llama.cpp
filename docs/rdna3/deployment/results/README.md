# Qwen3.8-27B rotated INT4 inference with OMP

The enabled user services are `qwen38-server.service` and `omp-web.service`.

- OpenAI-compatible API: http://127.0.0.1:8080/v1
- Server chat UI: http://127.0.0.1:8080
- OMP web UI: http://127.0.0.1:9120
- API model ID and OMP default: `qwen3.8-27b-int4` / `llama.cpp/qwen3.8-27b-int4`

The launcher is `/home/svt/qwen38-server-agent.sh`. It uses the preserved, benchmarked INT4 GPU library in `/home/svt/tuning27b-next/selected/bin`, the rotated GPTQ checkpoint, the IQ4_XS output-head cache and ROCm 10.1.0 already on disk. The server launcher was linked locally against the existing compatible server implementation. No downloads were required.

Settings: fast INT4 prefill; batched input uploads; one slot; 64k context; F16 K/V; batch 2048; microbatch 512; 12 threads; Flash Attention on. MTP and speculative decoding are disabled. Existing OMP MCP registration and the server UI's OMP delegation prompt are retained.

## Actual HTTP throughput

| Request | Prefill tokens/s | Decode tokens/s |
|---|---:|---:|
| 1-token prompt, 128-token output | - | 49.33 |
| 512-token prompt, 128-token output | 2005.97 | 49.11 |
| 2048-token prompt, 128-token output | 1928.97 | 48.16 |

These are native server timing fields from real uncached HTTP completions, with temperature 0 and 128 output tokens. `timings.cache_n` was zero. Prompt cases use the last two of three repetitions; short-context decode uses the median of three. The measured decode rate is about 49 tokens/s, not a verified 50+ guarantee. Long prompts increase attention work.

The inherited 128k/Q4-KV configuration measured about 46 tokens/s decode. F16 KV restores the fused attention path; the context is therefore configured at 64k to fit the XT. Server context checkpoints are disabled: their small prompt-tail batches were reducing uncached prefill to 1.6-1.8k tokens/s. Disabling them restored the measured 2k prefill. Prompt edits that need recurrent-state rollback can require more prompt recomputation.

## Validation

- Direct chat returned `INT4 ready`.
- OMP CLI used its default model and returned `OMP_INT4_OK`.
- The OMP web frontend invoked its `read` tool and returned a random nonce that was present only in a local test file.
- The model path, alias, context, non-speculative settings, INT4 environment and loaded library paths were checked against the running process. Details are in `deployment.json`.

`http_64k_f16_no_checkpoints/` contains final request results. The other `http_*` directories preserve the configuration comparisons. `omp_web_smoke.json` records the frontend/tool test.

Original launcher, service unit and OMP settings backups: `/home/svt/tuning27b-next/service/backups/20261104_160511`. The OMP settings file changed only its default model role.

To restart: `systemctl --user restart qwen38-server.service omp-web.service`.

## Paseo

Paseo's OMP model catalog now selects `llama.cpp/qwen3.8-27b-int4`, displayed as **Qwen3.8 27B INT4 (RX 7900 XT)**. Its service depends on `qwen38-server.service`.

The existing `/home/svt/.local/bin/omp-auto` wrapper translates the former absolute-path IQ4_XS model ID to the new INT4 ID. This keeps saved chats and stale browser model selections usable without editing their conversation histories. The wrapper retains the original medium-thinking behavior.

Actual requests through Paseo succeeded with both the new model ID and the legacy ID. The temporary validation chats were archived. Configuration backups are identified in `paseo-backup-location.txt`.
