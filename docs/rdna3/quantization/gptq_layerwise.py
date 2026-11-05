"""Layer-by-layer GPTQ for the rotated symmetric-INT4 format, reading BF16 weights from a (split) GGUF.

Same maths as gptq_rotate.py (rotated-input Hessians, column-wise error feedback, one fp16 scale per --group columns, Q4_0 blocks keyed by GGUF tensor name),
but only one decoder layer is resident at a time, so it scales to models that do not fit in VRAM or host memory:

  for each layer i: build the HF decoder layer from the GGUF weights (inverting the converter's transformations), run all calibration sequences through it
  (hidden states of the whole calibration set stay on the GPU), accumulate the Hessians of its linears, quantize them straight from the GGUF weights
  (already in llama.cpp row / column order, so no re-ordering is needed), and pass the layer's output on.

    python gptq_layerwise.py --gguf shard1.gguf shard2.gguf --hf <dir with config.json + tokenizer> --out blocks.safetensors
    python gptq_layerwise.py ... --validate 4        # NLL of N calibration sequences through all layers with the FP weights (no quantization), for comparison with llama.cpp
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/mnt/data/projects/llama.cpp/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType as QT
from scipy.linalg import hadamard
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
from transformers.masking_utils import create_causal_mask

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", nargs="+", required=True)
ap.add_argument("--hf", required=True)
ap.add_argument("--out")
ap.add_argument("--group", type=int, default=32)
ap.add_argument("--calib", type=int, default=128); ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--calib-text", default="/mnt/data/projects/ossicle-gguf/work/wiki.train.raw")
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--layers", type=int, default=0, help="only the first N layers (testing)")
ap.add_argument("--validate", type=int, default=0, help="NLL of N sequences through all layers with FP weights, then exit")
a = ap.parse_args()
dev = "cuda"

# ------------------------------------------------------------------------------------------------------------------------ GGUF access
T = {}
for p in a.gguf:
    r = GGUFReader(p)
    for t in r.tensors: T[t.name] = t
print(f"{len(T)} tensors in {len(a.gguf)} file(s)", flush=True)

def bf16_to_f32(u16): return (u16.astype(np.uint32) << 16).view(np.float32)
def load(name, dtype=torch.float32):
    t = T[name]; d = np.asarray(t.data)
    if t.tensor_type == QT.BF16:
        x = bf16_to_f32(d.view(np.uint16))
        x = x.reshape(d.shape[0], -1) if d.ndim >= 2 else x
    elif t.tensor_type in (QT.F32,): x = d.astype(np.float32)
    elif t.tensor_type == QT.F16: x = d.astype(np.float32)
    else: raise ValueError((name, t.tensor_type))
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype)

# ------------------------------------------------------------------------------------------------------------------------ config
cfg_json = json.load(open(f"{a.hf}/config.json"))
if "text_config" in cfg_json: cfg_json = cfg_json["text_config"]
cfg_json.pop("architectures", None)
cfg = AutoConfig.for_model(**cfg_json); cfg._attn_implementation = "sdpa"
NK, NV = cfg_json["linear_num_key_heads"], cfg_json["linear_num_value_heads"]
HK, HV = cfg_json["linear_key_head_dim"], cfg_json["linear_value_head_dim"]
NL = cfg_json["num_hidden_layers"]; LT = cfg.layer_types
NVK = NV // NK

def reorder_v_heads(t, dim, nk, nvk, hd):               # llama.cpp conversion/qwen.py (_reorder_v_heads): grouped -> tiled
    shape = list(t.shape)
    if dim < 0: dim += len(shape)
    t = t.reshape(*(shape[:dim] + [nk, nvk, hd] + shape[dim + 1:]))
    perm = list(range(t.dim())); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)
def inv_v(t, dim, hd): return reorder_v_heads(t, dim, NVK, NK, hd)       # tiled -> grouped (the converter's inverse)

def hf_layer_state(i):
    """HF state dict of decoder layer i from the GGUF tensors (inverting conversion/qwen.py)"""
    b = f"blk.{i}."; bf = torch.bfloat16; sd = {}
    sd["input_layernorm.weight"] = (load(b + "attn_norm.weight") - 1).to(bf)
    sd["post_attention_layernorm.weight"] = (load(b + "post_attention_norm.weight") - 1).to(bf)
    for k, n in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")): sd[f"mlp.{k}.weight"] = load(b + n + ".weight").to(bf)
    if LT[i] == "linear_attention":
        qkv = load(b + "attn_qkv.weight"); q = HK * NK
        sd["linear_attn.in_proj_qkv.weight"] = torch.cat([qkv[:2 * q], inv_v(qkv[2 * q:], 0, HV)], 0).to(bf)
        sd["linear_attn.in_proj_z.weight"] = inv_v(load(b + "attn_gate.weight"), 0, HV).to(bf)
        sd["linear_attn.in_proj_a.weight"] = inv_v(load(b + "ssm_alpha.weight"), 0, 1).to(bf)
        sd["linear_attn.in_proj_b.weight"] = inv_v(load(b + "ssm_beta.weight"), 0, 1).to(bf)
        sd["linear_attn.out_proj.weight"] = inv_v(load(b + "ssm_out.weight"), 1, HV).to(bf)
        sd["linear_attn.A_log"] = torch.log(-inv_v(load(b + "ssm_a").reshape(-1, 1), 0, 1).reshape(-1))     # parameter dtype follows the module
        sd["linear_attn.dt_bias"] = inv_v(load(b + "ssm_dt.bias").reshape(-1, 1), 0, 1).reshape(-1)
        cv = load(b + "ssm_conv1d.weight").squeeze(); cqk = HK * NK * 2
        sd["linear_attn.conv1d.weight"] = torch.cat([cv[:cqk], inv_v(cv[cqk:], 0, HV)], 0).unsqueeze(1).to(bf)
        sd["linear_attn.norm.weight"] = load(b + "ssm_norm.weight")
    else:
        for k, n in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")): sd[f"self_attn.{k}.weight"] = load(b + n + ".weight").to(bf)
        sd["self_attn.q_norm.weight"] = (load(b + "attn_q_norm.weight") - 1).to(bf)
        sd["self_attn.k_norm.weight"] = (load(b + "attn_k_norm.weight") - 1).to(bf)
    return sd

# targets of a layer: (module path in the HF layer, GGUF kind, hessian source module)
def targets(i):
    if LT[i] == "linear_attention":
        t = [("linear_attn.in_proj_qkv", "attn_qkv", None), ("linear_attn.in_proj_z", "attn_gate", "linear_attn.in_proj_qkv"), ("linear_attn.out_proj", "ssm_out", None)]
    else:
        t = [("self_attn.q_proj", "attn_q", None), ("self_attn.k_proj", "attn_k", "self_attn.q_proj"), ("self_attn.v_proj", "attn_v", "self_attn.q_proj"), ("self_attn.o_proj", "attn_output", None)]
    return t + [("mlp.gate_proj", "ffn_gate", None), ("mlp.up_proj", "ffn_up", "mlp.gate_proj"), ("mlp.down_proj", "ffn_down", None)]

# ------------------------------------------------------------------------------------------------------------------------ rotation / GPTQ (from gptq_rotate.py)
def sign_matrix():
    j = torch.arange(256, dtype=torch.int64)
    sgn = torch.where(((j * 2654435761) & 0xFFFFFFFF) >> 31 > 0, -1.0, 1.0)
    return (torch.tensor(hadamard(256), dtype=torch.float32) / 16.0 * sgn[None, :]).to(dev)
R = sign_matrix()
CANDS = torch.linspace(0.55, 1.05, 26)
def pick_scale(grp):
    amax = grp.abs().amax(1, keepdim=True).clamp_min(1e-12)
    best, scale = None, None
    for c in CANDS.to(grp.device):
        s = (amax * c / 7).half().float().clamp_min(1e-8)
        e = ((grp - (grp / s).round().clamp(-8, 7) * s) ** 2).sum(1, keepdim=True)
        if best is None: best, scale = e, s
        else:
            m = e < best; best = torch.where(m, e, best); scale = torch.where(m, s, scale)
    return scale

@torch.no_grad()
def gptq(w, H, wg, blocksize=128):
    M, K = w.shape
    w = w.clone()
    codes = torch.zeros(M, K, dtype=torch.int8, device=w.device)
    scales = torch.zeros(M, K // wg, device=w.device)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; w[:, dead] = 0
    H += a.damp * torch.mean(torch.diag(H)) * torch.eye(K, device=w.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    del H
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); cnt = i2 - i1
        W1 = w[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if col % wg == 0:
                scale = pick_scale(W1[:, i:i + wg]); scales[:, col // wg] = scale[:, 0]
            wc = W1[:, i:i + 1]
            qi = (wc / scale).round().clamp(-8, 7)
            codes[:, col] = qi[:, 0].to(torch.int8)
            err = (wc - qi * scale) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        w[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return codes, scales

def pack_q4_0(codes, scales, wg):
    M, K = codes.shape
    nib = (codes.cpu().to(torch.int16) + 8).clamp(0, 15).to(torch.uint8).reshape(M, K // 32, 32)
    qs = nib[..., :16] | (nib[..., 16:] << 4)
    d = scales.cpu().half().repeat_interleave(wg // 32, dim=1).reshape(M, K // 32, 1)
    out = torch.empty(M, K // 32, 18, dtype=torch.uint8)
    out[..., 0:2] = d.view(torch.uint8).reshape(M, K // 32, 2)
    out[..., 2:] = qs
    return out.reshape(M, K // 32 * 18)

def col_perm(suffix, K):
    """HF input-feature index for each llama.cpp column (only linear_attn.out_proj has its columns re-ordered)"""
    if suffix == "linear_attn.out_proj": return reorder_v_heads(torch.arange(K)[None, :], 1, NK, NVK, HV)[0].to(dev)
    return None

# ------------------------------------------------------------------------------------------------------------------------ calibration data and embeddings
tok = AutoTokenizer.from_pretrained(a.hf)
ids = tok(open(a.calib_text).read(), return_tensors="pt").input_ids[0]
nseq = a.validate if a.validate else a.calib
assert ids.numel() >= nseq * a.ctx, (ids.numel(), nseq * a.ctx)
seqs = torch.stack([ids[i * a.ctx:(i + 1) * a.ctx] for i in range(nseq)])
emb = load("token_embd.weight", torch.bfloat16)                          # [vocab, hidden]
hid = emb[seqs].to(dev); del emb                                          # [nseq, ctx, hidden] bf16, resident on the GPU
print("hidden", tuple(hid.shape), flush=True)

rot = Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)
pos = torch.arange(a.ctx, device=dev).view(1, 1, -1).expand(4, a.bs, -1)
text_pos = pos[0]; pos3 = pos[1:]

def run_layer(layer, hs, hess=None):
    """hs [n, ctx, hidden] -> outputs (in place); hess: dict module-path -> (hook handle holder)"""
    for c in range(0, hs.shape[0], a.bs):
        x = hs[c:c + a.bs]; b = x.shape[0]
        tp, p3 = text_pos[:b], pos3[:, :b]
        pe = rot(x, p3)
        mask = create_causal_mask(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=tp)
        lm = None if LT[layer.layer_idx_] == "linear_attention" else mask
        hs[c:c + b] = layer(x, position_embeddings=pe, attention_mask=lm, position_ids=tp)

t0 = time.time()
out = {}; tot_err = 0.0; ntar = 0
nlayers = a.layers if a.layers else NL
with torch.no_grad():
    for i in range(nlayers):
        with torch.device("meta"): layer = Qwen3_5DecoderLayer(cfg, i)           # no random init: every parameter is overwritten from the GGUF below
        layer = layer.to_empty(device=dev).to(torch.bfloat16); layer.layer_idx_ = i
        sd = hf_layer_state(i)
        missing, unexpected = layer.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        assert not [m for m in missing if "rotary" not in m] and not unexpected, (i, missing, unexpected)
        tg = targets(i); mods = dict(layer.named_modules())
        HESS = {}
        hooks = []
        if not a.validate:
            def mk(path, suffix):
                perm = col_perm(suffix, mods[path].in_features)
                def hook(mod, inp):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    if perm is not None: x = x[:, perm]
                    K = x.shape[-1]
                    xr = (x.reshape(-1, K // 256, 256) @ R.T).reshape(-1, K)
                    h = xr.T @ xr
                    HESS[path] = HESS[path] + h if path in HESS else h
                return hook
            for path, kind, src in tg:
                if src is None: hooks.append(mods[path].register_forward_pre_hook(mk(path, path)))
        run_layer(layer, hid)
        for h in hooks: h.remove()
        if not a.validate:
            for path, kind, src in tg:
                w = load(f"blk.{i}.{kind}.weight").to(dev)                      # llama.cpp order, [M, K]
                M, K = w.shape
                wr = (w.reshape(M, K // 256, 256) @ R.T).reshape(M, K)
                H = HESS[src or path].clone()
                codes, scales = gptq(wr, H, a.group)
                deq = codes.float() * scales.repeat_interleave(a.group, dim=1)
                tot_err += ((deq - wr).norm() / wr.norm()).item(); ntar += 1
                out[f"blk.{i}.{kind}.weight"] = pack_q4_0(codes, scales, a.group)
                del w, wr, H, codes, scales, deq
        del layer, sd, HESS
        torch.cuda.empty_cache()
        print(f"layer {i + 1}/{nlayers} ({LT[i][:3]})  {time.time() - t0:.0f}s  mean rel.err so far {tot_err / max(ntar, 1):.4f}  vram {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

if a.validate:
    normw = (load("output_norm.weight") - 1).to(dev)
    head = load("output.weight", torch.bfloat16).to(dev)                    # [vocab, hidden]
    nll = 0.0; cnt = 0; half = a.ctx // 2
    for c in range(0, hid.shape[0], a.bs):
        x = hid[c:c + a.bs].float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * (1.0 + normw.float())
        lg = (x.to(torch.bfloat16) @ head.T).float()
        lp = torch.log_softmax(lg[:, half - 1:-1], -1)
        tgt = seqs[c:c + a.bs, half:].to(dev)
        nll += -lp.gather(-1, tgt.unsqueeze(-1)).sum().item(); cnt += tgt.numel()
    print(f"VALIDATE: {nseq} sequences, NLL over the second half of each ({cnt} tokens): PPL = {math.exp(nll / cnt):.4f}", flush=True)
else:
    print(f"mean relative weight error {tot_err / ntar:.4f} over {ntar} tensors", flush=True)
    save_file(out, a.out, metadata={"group": str(a.group)})
    print("saved", a.out)
