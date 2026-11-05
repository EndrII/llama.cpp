"""GPTQ for the rotated symmetric-int4 format of scripts/rotate_quantize.py.

Loads the Ossicle language model with transformers, collects the Hessian H = sum x'x'^T of every target linear's *rotated* input
(x' = (H D) x per 256-block, in llama.cpp's column order), then quantizes each rotated weight with column-wise GPTQ (error feedback) to
int4 codes in [-8, 7] with one fp16 scale per `--group` columns (scale searched per group for minimum error).  The result is stored
as ready-made ggml Q4_0 blocks keyed by GGUF tensor name, for `rotate_quantize.py --blocks`.

    python gptq_rotate.py --src stage_lm --out gptq_g32.safetensors --group 32 --calib 128
"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn
from scipy.linalg import hadamard
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/mnt/data/projects/ossicle-gguf/stage_lm", help="HF dir; an original Qwen3.5 checkpoint (model.language_model.* keys, nested text_config) or the Ossicle llm.* layout")
ap.add_argument("--out", required=True)
ap.add_argument("--group", type=int, default=32)
ap.add_argument("--calib", type=int, default=128); ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--calib-text", default="/mnt/data/projects/ossicle-gguf/work/wiki.train.raw")
ap.add_argument("--damp", type=float, default=0.01)
ap.add_argument("--rtn", action="store_true", help="skip error feedback (per-group MSE scale + round) for comparison")
a = ap.parse_args()
dev = "cuda"

KINDS = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
         "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"}

def reorder_v_heads(t, dim, nk, nvk, hd):                 # copied from llama.cpp conversion/qwen.py (_reorder_v_heads)
    shape = list(t.shape)
    if dim < 0: dim += len(shape)
    t = t.reshape(*(shape[:dim] + [nk, nvk, hd] + shape[dim + 1:]))
    perm = list(range(t.dim())); perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)

cfg_json = json.load(open(f"{a.src}/config.json"))
if "text_config" in cfg_json: cfg_json = cfg_json["text_config"]
cfg_json.pop("architectures", None)
cfg = AutoConfig.for_model(**cfg_json)
NK, NV = cfg_json["linear_num_key_heads"], cfg_json["linear_num_value_heads"]
HK, HV = cfg_json["linear_key_head_dim"], cfg_json["linear_value_head_dim"]
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
print(type(model).__name__, flush=True)
sd = {}
for shard in sorted(f for f in os.listdir(a.src) if f.startswith("model") and f.endswith(".safetensors")):
    with safe_open(os.path.join(a.src, shard), "pt") as f:
        for k in f.keys():
            if k.startswith("llm."): sd["model." + k[4:]] = f.get_tensor(k)
            elif k.startswith("model.language_model."): sd["model." + k[len("model.language_model."):]] = f.get_tensor(k)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", [m for m in missing if "lm_head" not in m][:5], "unexpected", unexpected[:5], flush=True)
model.tie_weights(); model.eval().to(dev)
del sd

targets = []
for n, m in model.named_modules():
    if isinstance(m, nn.Linear) and ".layers." in n:
        suffix = n.split(".layers.")[1].split(".", 1)[1]
        if suffix in KINDS: targets.append((n, m, int(n.split(".layers.")[1].split(".")[0]), KINDS[suffix], suffix))
print("targets", len(targets), flush=True)

def sign_matrix():
    j = torch.arange(256, dtype=torch.int64)
    sgn = torch.where(((j * 2654435761) & 0xFFFFFFFF) >> 31 > 0, -1.0, 1.0)
    return (torch.tensor(hadamard(256), dtype=torch.float32) / 16.0 * sgn[None, :]).to(dev)
R = sign_matrix()

def col_perm(suffix, K):
    """HF input-feature index for each llama.cpp column (only linear_attn.out_proj has its columns re-ordered)"""
    if suffix == "linear_attn.out_proj":
        return reorder_v_heads(torch.arange(K)[None, :], 1, NK, NV // NK, HV)[0].to(dev)
    return None

# ---------------------------------------------------------------------------------------------------------- Hessians
HESS = {}
tok = AutoTokenizer.from_pretrained(a.src)
ids = tok(open(a.calib_text).read(), return_tensors="pt").input_ids[0]
assert ids.numel() >= a.calib * a.ctx, (ids.numel(), a.calib * a.ctx)
perms = {n: col_perm(s, m.in_features) for n, m, _, _, s in targets}
def mk(n):
    def hook(mod, inp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
        if perms[n] is not None: x = x[:, perms[n]]
        K = x.shape[-1]
        xr = (x.reshape(-1, K // 256, 256) @ R.T).reshape(-1, K)
        h = (xr.T @ xr).cpu()
        if n in HESS: HESS[n] += h
        else: HESS[n] = h
    return hook
mods = dict(model.named_modules())
hooks = [mods[n].register_forward_pre_hook(mk(n)) for n, *_ in targets]
t0 = time.time()
with torch.no_grad():
    for c in range(0, a.calib, 8):
        model.model(torch.stack([ids[i * a.ctx:(i + 1) * a.ctx] for i in range(c, min(c + 8, a.calib))]).to(dev))
for h in hooks: h.remove()
print(f"hessians: {len(HESS)} in {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------------------------------------------------- GPTQ
CANDS = torch.linspace(0.55, 1.05, 26)
def pick_scale(grp):
    """grp [M, g] -> fp16-representable scale [M,1] minimising the squared error of codes in [-8, 7]"""
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
def gptq(w, Hs, wg, blocksize=128):
    M, K = w.shape
    w = w.clone()
    codes = torch.zeros(M, K, dtype=torch.int8, device=w.device)
    scales = torch.zeros(M, K // wg, device=w.device)
    if a.rtn:
        for g in range(K // wg):
            s = pick_scale(w[:, g * wg:(g + 1) * wg]); scales[:, g] = s[:, 0]
            codes[:, g * wg:(g + 1) * wg] = (w[:, g * wg:(g + 1) * wg] / s).round().clamp(-8, 7).to(torch.int8)
        return codes, scales
    H = Hs.to(w.device).float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; w[:, dead] = 0
    H += a.damp * torch.mean(torch.diag(H)) * torch.eye(K, device=w.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
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
    """codes [M,K] int8 in [-8,7], scales [M,K/wg] -> uint8 [M, K/32*18] (ggml block_q4_0: fp16 d, then 16 bytes: element j low / j+16 high)"""
    M, K = codes.shape
    nib = (codes.cpu().to(torch.int16) + 8).clamp(0, 15).to(torch.uint8).reshape(M, K // 32, 32)
    qs = nib[..., :16] | (nib[..., 16:] << 4)
    d = scales.cpu().half().repeat_interleave(wg // 32, dim=1).reshape(M, K // 32, 1)
    out = torch.empty(M, K // 32, 18, dtype=torch.uint8)
    out[..., 0:2] = d.view(torch.uint8).reshape(M, K // 32, 2)
    out[..., 2:] = qs
    return out.reshape(M, K // 32 * 18)

out, tot_err = {}, 0.0
t0 = time.time()
for n, m, layer, kind, suffix in targets:
    w = m.weight.data.float()
    M, K = w.shape
    if perms[n] is not None: w = w[:, perms[n]]
    wr = (w.reshape(M, K // 256, 256) @ R.T).reshape(M, K)
    codes, scales = gptq(wr, HESS.pop(n, None), a.group)
    deq = codes.float() * scales.repeat_interleave(a.group, dim=1)
    rel = ((deq - wr).norm() / wr.norm()).item(); tot_err += rel
    # llama.cpp row order
    if suffix == "linear_attn.in_proj_qkv":
        q = NK * HK
        def fix(t): return torch.cat([t[:2 * q], reorder_v_heads(t[2 * q:], 0, NK, NV // NK, HV)], 0)
        codes, scales = fix(codes), fix(scales)
    elif suffix == "linear_attn.in_proj_z":
        codes, scales = reorder_v_heads(codes, 0, NK, NV // NK, HV), reorder_v_heads(scales, 0, NK, NV // NK, HV)
    out[f"blk.{layer}.{kind}.weight"] = pack_q4_0(codes, scales, a.group)
    if len(out) % 20 == 0: print(f"  {len(out)}/{len(targets)}  last rel.err {rel:.4f}  ({time.time() - t0:.0f}s)", flush=True)
print(f"mean relative weight error {tot_err / len(targets):.4f}", flush=True)
save_file(out, a.out, metadata={"group": str(a.group)})
print("saved", a.out)
