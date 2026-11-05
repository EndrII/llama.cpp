# Local server, OMP and Paseo

The deployed model is `qwen3.8-27b-int4` at `http://127.0.0.1:8080/v1`. The portable [server launcher](../../../examples/rdna3-int4/serve.sh) uses the selected INT4 path, fast prefill, IQ4_XS head, batched host inputs, 64k F16 KV, Flash Attention and no speculation.

`local-snapshot/` preserves the actual launcher and user service units, the OMP web frontend and MCP adapter, and the `omp-auto` compatibility wrapper. These snapshots contain machine-specific paths. The wrapper translates the previous absolute-path model ID to `llama.cpp/qwen3.8-27b-int4`, including both `--model value` and `--model=value`, and retains medium thinking.

OMP's default model role and Paseo's OMP model catalog use `llama.cpp/qwen3.8-27b-int4`. Paseo's service depends on `qwen38-server.service`. Existing MCP registration and the server UI's delegation prompt are preserved in the launcher snapshot. Personal OMP/Paseo configurations and conversations are excluded.

Real requests succeeded through the API, OMP CLI, OMP web tool execution, and Paseo using both the current and previous model IDs. Validation records, model properties and uncached HTTP timing samples are under `results/`. The earlier KV/checkpoint configurations are retained there as experiments.

The local services were already running before this publication. Publishing does not restart or reconfigure them.
