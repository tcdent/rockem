#!/usr/bin/env python3
"""Run the GPU FFN half of one MoE layer (bf16_rows_norm for the shared gate|up, the two
expert launches, shared_down with the routed combine) and the head (bf16_rows_norm over the
LM head) on a real state dumped by `forward --dump-attn`, and diff against the CPU path.

  ffn.py <libamdhip64.so> <kernels.elf> <dense.bf16.bin> <experts.q4g128.bin> <vectors dir>"""
import ctypes as C, sys, struct, os

hip = C.CDLL(sys.argv[1], mode=C.RTLD_GLOBAL)
ELF = open(sys.argv[2], "rb").read()
DENSE, EXPERTS, VEC = sys.argv[3], sys.argv[4], sys.argv[5]
meta = dict(l.split() for l in open(os.path.join(VEC, "meta.txt")))
L, H, WG, S, VOCAB, TOP_K, D_FF, RECORD, LOFF, RN_LDS, SD_LDS, UP_T, DN_T, UP_LDS, DN_LDS = (int(meta[k]) for k in
    ("layer", "H", "wg", "SHARED_FF", "VOCAB", "TOP_K", "D_FF", "record_bytes", "layer_record_off", "rows_norm_lds", "shared_down_lds", "up_tiles", "down_tiles", "up_lds", "down_lds"))
def ck(r, what):
    if r != 0:
        hip.hipGetErrorString.restype = C.c_char_p
        raise RuntimeError(f"{what}: hip error {r} {hip.hipGetErrorString(r)}")
hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
hip.hipModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
hip.hipModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
hip.hipModuleLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
H2D, D2H = 1, 2
ck(hip.hipSetDevice(0), "setdevice")
mod = C.c_void_p(); ck(hip.hipModuleLoadData(C.byref(mod), ELF), "load")
def fn(n):
    f = C.c_void_p(); ck(hip.hipModuleGetFunction(C.byref(f), mod, n.encode()), n); return f
def dmalloc(n):
    p = C.c_void_p(); ck(hip.hipMalloc(C.byref(p), n), "malloc"); return p
def upload(b):
    p = dmalloc(len(b)); ck(hip.hipMemcpy(p, b, len(b), H2D), "h2d"); return p
def download(p, n):
    buf = C.create_string_buffer(n); ck(hip.hipMemcpy(buf, p, n, D2H), "d2h"); return buf.raw
def sync(): ck(hip.hipDeviceSynchronize(), "sync")
def launch(f, grid, block, shared, *args):
    vals = [a if isinstance(a, C._SimpleCData) else C.c_void_p(a.value) for a in args]
    params = (C.c_void_p * len(vals))(*[C.addressof(v) for v in vals])
    ck(hip.hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, None, params, None), "launch")
def vec(name): return open(os.path.join(VEC, name), "rb").read()
def f32s(b): return struct.unpack(f"<{len(b)//4}f", b)
def fread(path, off, n):
    with open(path, "rb") as f:
        f.seek(off); return f.read(n)
def diff(name, got, want, tol=1e-4):
    g, w = f32s(got), f32s(want)
    err = max(abs(a - b) for a, b in zip(g, w)); scale = max(abs(b) for b in w) or 1.0
    print(f"  {name:<36} max|Δ| {err:.3e}  rel {err/scale:.3e}  {'ok' if err/scale <= tol else 'OVER TOL'}"); return err / scale <= tol

rows_norm, shared_down, up, down = fn("bf16_rows_norm"), fn("shared_down"), fn("experts_up"), fn("experts_down")
def grid8(m): return (m + 7) // 8

if L < 1:
    # --- the dense layer's FFN half: rows_norm over h for gate|up, dense_down with the residual add ---
    I, DD_LDS = int(meta["INTERMEDIATE"]), int(meta["dense_down_lds"])
    dense_down = fn("dense_down")
    d_h = upload(vec("h_out.f32"))
    d_lout = dmalloc(H * 4)
    d_ln = upload(fread(DENSE, int(meta["post_ln_off"]), 2 * H))
    d_g = upload(fread(DENSE, int(meta["dense_gate_off"]), 2 * 2 * I * H))
    d_d = upload(fread(DENSE, int(meta["dense_down_off"]), 2 * H * I))
    d_gu = dmalloc(2 * I * 4)
    launch(rows_norm, grid8(2 * I), WG, RN_LDS, d_h, d_ln, d_g, d_gu, C.c_uint(2 * I))
    launch(dense_down, grid8(H), WG, DD_LDS, d_gu, d_d, d_h, d_lout); sync()
    print(f"layer {L}: the dense FFN half on the card vs the CPU path")
    ok = diff("dense: layer out (h)", download(d_h, H * 4), vec("ffn_h_out.f32"), tol=1e-3)
    sys.exit(0 if ok else 1)

# --- the MoE layer's FFN half: input is h after attention (h_out.f32), expected is the layer output ---
h_mid = vec("h_out.f32"); d_h = upload(h_mid); d_lout = dmalloc(H * 4)
idx = [int(v) for v in f32s(vec("ffn_idx.u32"))]; wt = vec("ffn_wt.f32")
d_ln = upload(fread(DENSE, int(meta["post_ln_off"]), 2 * H))
d_sg = upload(fread(DENSE, int(meta["shared_gate_off"]), 2 * 2 * S * H))
d_sd = upload(fread(DENSE, int(meta["shared_down_off"]), 2 * H * S))
d_gu = dmalloc(2 * S * 4)
# the six hit experts' records, packed back to back; offsets relative to that buffer
recs = b"".join(fread(EXPERTS, LOFF + e * RECORD, RECORD) for e in idx)
d_stream = upload(recs); d_off = upload(struct.pack(f"<{TOP_K}Q", *[k * RECORD for k in range(TOP_K)]))
# the experts' quantized input is the host's post-norm: recompute it here exactly as the CPU does
hn = f32s(vec("h_out.f32"))  # placeholder, replaced below by the real hn from the dump if present
d_wt = upload(wt)
d_egu, d_eout = dmalloc(TOP_K * 2 * D_FF * 4), dmalloc(TOP_K * H * 4)
if True:
    # the post-norm on the host, exactly as the CPU does it; the experts quantize it on the card
    ln = struct.unpack(f"<{H}H", fread(DENSE, int(meta["post_ln_off"]), 2 * H))
    ss = sum(v * v for v in f32s(h_mid)); r = (ss / H + 1e-6) ** -0.5
    import math
    hnv = [struct.unpack("<f", struct.pack("<I", w << 16))[0] * (x * r) for w, x in zip(ln, f32s(h_mid))]
    q, sc, gs = [], [], []
    for g in range(H // 128):
        seg = hnv[g * 128:(g + 1) * 128]; am = max(abs(v) for v in seg); s = am / 127.0; inv = 1.0 / s if s > 0 else 0.0
        qs = [max(-127, min(127, int(math.floor(abs(v * inv) + 0.5) * (1 if v >= 0 else -1)))) for v in seg]
        q += qs; sc.append(s); gs.append(sum(qs))
    xq_b, xs_b, xg_b = struct.pack(f"<{H}b", *q), struct.pack(f"<{H//128}f", *sc), struct.pack(f"<{H//128}i", *gs)
d_xq, d_xs, d_xg = upload(xq_b), upload(xs_b), upload(xg_b)
n = C.c_uint(TOP_K)
launch(up, (TOP_K * UP_T * 32 + WG - 1) // WG, WG, UP_LDS, d_stream, d_off, n, d_xq, d_xs, d_xg, d_egu)
launch(down, (TOP_K * DN_T * 32 + WG - 1) // WG, WG, DN_LDS, d_stream, d_off, n, d_egu, d_eout)
launch(rows_norm, grid8(2 * S), WG, RN_LDS, d_h, d_ln, d_sg, d_gu, C.c_uint(2 * S))
launch(shared_down, grid8(H), WG, SD_LDS, d_gu, d_sd, d_eout, d_wt, d_h, d_lout); sync()
print(f"layer {L}: the FFN half on the card vs the CPU path (experts {idx})")
ok = diff("moe: layer out (h)", download(d_h, H * 4), vec("ffn_h_out.f32"), tol=1e-3)

# --- the head over the final layer's output ---
d_hf = upload(vec("h_final.f32"))
d_norm = upload(fread(DENSE, int(meta["norm_off"]), 2 * H)); d_lm = upload(fread(DENSE, int(meta["lm_head_off"]), 2 * VOCAB * H))
d_logits = dmalloc(VOCAB * 4)
launch(rows_norm, grid8(VOCAB), WG, RN_LDS, d_hf, d_norm, d_lm, d_logits, C.c_uint(VOCAB)); sync()
got = f32s(download(d_logits, VOCAB * 4)); want = f32s(vec("logits.f32"))
ok &= diff("head: lm_head.out", download(d_logits, VOCAB * 4), vec("logits.f32"))
ga, wa = max(range(VOCAB), key=lambda i: got[i]), max(range(VOCAB), key=lambda i: want[i])
print(f"  argmax gpu {ga} cpu {wa} {'ok' if ga == wa else 'DIFFERS'}")
sys.exit(0 if ok else 1)
