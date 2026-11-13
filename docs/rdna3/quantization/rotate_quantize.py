"""Rotate + quantize a BF16 GGUF for the INT4-WMMA prefill path.

For every target linear weight W [M rows (out), K (in)]:  W' = W (H D)^T  applied per 256-block along K  (H = Sylvester Hadamard / 16,
D = diag(sign_j)).  At run time the backend applies x' = (H D) x to the activations, so  W' x' = W x.  Rotated weights are quantized
symmetrically to int4 (codes -7..7, one shared fp16 scale per 32*GW values) and stored as ordinary ggml Q4_0 blocks, so no new GGUF
type is needed.  Everything else (embeddings/head, norms, small tensors) is copied unchanged.

    python rotate_quantize.py in-bf16.gguf out.gguf [--gw 4]
"""
import argparse, re, sys
import numpy as np
sys.path.insert(0, "/mnt/data/projects/llama.cpp/gguf-py")
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT, GGUFValueType
from scipy.linalg import hadamard

TARGET = re.compile(r"^blk\.\d+\.(attn_qkv|attn_gate|ssm_out|ffn_down|ffn_gate|ffn_up|attn_k|attn_v|attn_output|attn_q)\.weight$")

def signs():
    j = np.arange(256, dtype=np.uint64)
    return np.where(((j * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(31), -1.0, 1.0).astype(np.float32)   # same as the C++ side

def rot_matrix():
    return (hadamard(256).astype(np.float32) / 16.0) * signs()[None, :]       # H D : x' = (H D) x

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize_q4_0(w, gw):
    """w [M, K] float32 -> Q4_0 blocks as uint8 [M, K/32*18], symmetric codes in [-7,7], scale shared over gw consecutive 32-blocks"""
    M, K = w.shape
    g = w.reshape(M, K // (32 * gw), 32 * gw)
    d = (np.abs(g).max(-1, keepdims=True) / 7.0).astype(np.float16)                 # scale rounded to fp16, as stored
    df = np.maximum(d.astype(np.float32), 1e-30)
    q = np.clip(np.rint(g / df), -7, 7).astype(np.int8)
    q = q.reshape(M, K // 32, 32)
    nib = (q + 8).astype(np.uint8)
    qs = nib[..., :16] | (nib[..., 16:] << 4)                                       # element j low nibble, element j+16 high nibble
    dd = np.repeat(d.reshape(M, K // (32 * gw), 1), gw, axis=1).reshape(M, K // 32).astype(np.float16)
    out = np.empty((M, K // 32, 18), dtype=np.uint8)
    out[..., 0:2] = dd.view(np.uint8).reshape(M, K // 32, 2)
    out[..., 2:] = qs
    return out.reshape(M, K // 32 * 18), None

def rel_err(blocks, w):
    """relative error of the dequantized Q4_0 blocks [M, K/32*18] against w [M, K]"""
    M, K = w.shape
    b = blocks.reshape(M, K // 32, 18)
    d = b[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(M, K // 32, 1)
    qs = b[..., 2:]
    q = np.concatenate([(qs & 15).astype(np.int8), (qs >> 4).astype(np.int8)], -1) - 8
    deq = (q * d).reshape(M, K)
    return float(np.linalg.norm(deq - w) / np.linalg.norm(w))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--gw", type=int, default=4)
    ap.add_argument("--embd", choices=["bf16", "q8_0", "q6_k"], default="bf16", help="storage of token_embd (also the tied output head)")
    ap.add_argument("--blocks", help="safetensors from gptq_rotate.py with ready-made Q4_0 blocks per tensor (replaces the RTN quantizer; --gw is then the file's group/32)")
    a = ap.parse_args()
    r = GGUFReader(a.src)
    arch = r.fields["general.architecture"].contents()
    w = GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY: w.add_array(name, f.contents())
        elif t == GGUFValueType.STRING: w.add_string(name, f.contents())
        elif t == GGUFValueType.BOOL: w.add_bool(name, bool(f.contents()))
        else: w.add_key_value(name, f.contents(), t)
    w.add_uint32("ossicle.int4_rotation_block", 256)
    w.add_uint32("ossicle.int4_scale_group", 32 * a.gw)
    H = rot_matrix()
    nrot = 0
    errs = []
    pre = None
    if a.blocks:
        from safetensors import safe_open
        pre = safe_open(a.blocks, 'np'); a.gw = max(1, int(pre.metadata()['group']) // 32)
    for t in r.tensors:
        if TARGET.match(t.name) and t.tensor_type == QT.BF16:
            wt = bf16_to_f32(np.asarray(t.data).view(np.uint16)).reshape(t.data.shape[0], -1)       # [M, K]
            M, K = wt.shape
            assert K % 256 == 0 and K % (32 * a.gw) == 0, (t.name, K)
            wr = (wt.reshape(M, K // 256, 256) @ H.T).reshape(M, K)                                    # W' = W (H D)^T per 256-block
            if pre is not None:
                blocks = pre.get_tensor(t.name)
                err = rel_err(blocks, wr)
                assert err < 0.25, (t.name, err)          # a wrong row/column order shows up as ~1.0
                errs.append(err)
            else:
                blocks, _ = quantize_q4_0(wr, a.gw)
            w.add_tensor(t.name, blocks, raw_dtype=QT.Q4_0)
            nrot += 1
        elif a.embd == "q6_k" and t.name in ("token_embd.weight", "output.weight") and t.tensor_type == QT.BF16:
            import ctypes
            lib = ctypes.CDLL("/mnt/data/projects/llama.cpp/build-rocm101/bin/libggml-base.so")
            lib.ggml_quantize_chunk.restype = ctypes.c_size_t
            lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            wt = np.ascontiguousarray(bf16_to_f32(np.asarray(t.data).view(np.uint16)).reshape(t.data.shape[0], -1))
            rows, cols = wt.shape
            out = np.empty((rows, cols // 256 * 210), dtype=np.uint8)
            n = lib.ggml_quantize_chunk(14, wt.ctypes.data, out.ctypes.data, 0, rows, cols, None)
            assert n == out.size, (n, out.size)
            w.add_tensor(t.name, out, raw_dtype=QT.Q6_K)
        elif a.embd == "q8_0" and t.name in ("token_embd.weight", "output.weight") and t.tensor_type == QT.BF16:
            from gguf import quants
            wt = bf16_to_f32(np.asarray(t.data).view(np.uint16)).reshape(t.data.shape[0], -1)
            w.add_tensor(t.name, quants.quantize(wt, QT.Q8_0), raw_dtype=QT.Q8_0)
        else:
            d = np.asarray(t.data)
            if t.tensor_type in (QT.F32, QT.F16):
                w.add_tensor(t.name, d)
            else:
                w.add_tensor(t.name, d.view(np.uint8).reshape(d.shape[0], -1) if d.ndim > 1 else d.view(np.uint8), raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"rotated + quantized {nrot} tensors -> {a.dst}" + (f"  (mean rel. error vs BF16 {np.mean(errs):.4f}, max {np.max(errs):.4f})" if errs else ""))

if __name__ == "__main__":
    main()
