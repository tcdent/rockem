#!/usr/bin/env python3
"""Run deepseek_v2_lite's GPU attention kernels (attn_proj, attn_mla, attn_merge, attn_out) on
one real layer state dumped by `forward --dump-attn`, diff every edge against the CPU
realization at tolerance (f32 summation order differs), and time the four launches.

  attn.py <libamdhip64.so> <kernels.elf> <dense.bf16.bin> <vectors dir> [iters=20]"""
import ctypes as C, sys, time, struct, os

hip = C.CDLL(sys.argv[1], mode=C.RTLD_GLOBAL)
ELF = open(sys.argv[2], "rb").read()
DENSE, VEC = sys.argv[3], sys.argv[4]
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
meta = dict(l.split() for l in open(os.path.join(VEC, "meta.txt")))
L, POS, H, Q_DIM, KVA_DIM, KV_ROW, R, HEADS, WG, PROJ_GRID, OUT_GRID, PROJ_LDS, OUT_LDS, MLA_LDS, MERGE_LDS, CHUNK_MAX, CHUNK_MIN, SPLITS, PART, CEIL = (int(meta[k]) for k in
    ("layer", "pos", "H", "Q_DIM", "KVA_DIM", "KV_ROW", "KV_LORA_RANK", "HEADS", "wg", "proj_grid", "out_grid", "proj_lds", "out_lds", "mla_lds", "merge_lds", "chunk_max", "chunk_min", "splits_max", "part", "vram_ceiling_gbps"))
splits = lambda n: max(max(1, min(SPLITS, -(-n // CHUNK_MIN))), -(-n // CHUNK_MAX))   # attn::gfx11_consts::mla_splits
SCALE = float(meta["scale"])
MAX_CTX = 2048

def ck(r, what):
    if r != 0:
        hip.hipGetErrorString.restype = C.c_char_p
        raise RuntimeError(f"{what}: hip error {r} {hip.hipGetErrorString(r)}")
hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
hip.hipMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
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
def dense(off, nbytes):
    with open(DENSE, "rb") as f:
        f.seek(off); return f.read(nbytes)

# weights for this layer, bf16, straight from the dense file at the offsets the dump named
d_ln = upload(dense(int(meta["input_ln_off"]), 2 * H))
d_q = upload(dense(int(meta["q_proj_off"]), 2 * Q_DIM * H))
d_kva = upload(dense(int(meta["kv_a_off"]), 2 * KVA_DIM * H))
d_kvaln = upload(dense(int(meta["kv_a_ln_off"]), 2 * R))
d_kvb = upload(dense(int(meta["kv_b_off"]), 2 * HEADS * 256 * R))
d_o = upload(dense(int(meta["o_proj_off"]), 2 * H * H))
# state
h_in = vec("h_in.f32"); d_h = upload(h_in)
cache = vec("cache.f32")
d_cache = dmalloc(MAX_CTX * KV_ROW * 4); ck(hip.hipMemset(d_cache, 0, MAX_CTX * KV_ROW * 4), "memset")
ck(hip.hipMemcpy(d_cache, cache, len(cache), H2D), "cache h2d")   # rows 0..pos (row pos gets rewritten by the kernel)
d_cos, d_sin = upload(vec("cos.f32")), upload(vec("sin.f32"))
d_qo, d_kvao, d_heads = dmalloc(Q_DIM * 4), dmalloc(KVA_DIM * 4), dmalloc(H * 4)
d_mirror = dmalloc(KV_ROW * 4)
d_part = dmalloc(HEADS * splits(MAX_CTX) * PART * 4)
proj, mla, merge, out = fn("attn_proj"), fn("attn_mla"), fn("attn_merge"), fn("attn_out")

def run(pos=POS):
    launch(proj, PROJ_GRID, WG, PROJ_LDS, d_h, d_ln, d_q, d_kva, d_qo, d_kvao)
    launch(mla, HEADS * splits(pos + 1), WG, MLA_LDS, d_qo, d_kvao, d_kvaln, d_kvb, d_cache, C.c_uint(pos), d_cos, d_sin, C.c_float(SCALE), d_part, d_mirror)
    launch(merge, HEADS, WG, MERGE_LDS, d_part, C.c_uint(splits(pos + 1)), d_kvb, d_heads)
    launch(out, OUT_GRID, WG, OUT_LDS, d_heads, d_o, d_h)

def diff(name, got, want, tol=1e-4):
    g, w = f32s(got), f32s(want)
    err = max(abs(a - b) for a, b in zip(g, w)); scale = max(abs(b) for b in w) or 1.0
    ok = err / scale <= tol
    print(f"  {name:<32} max|Δ| {err:.3e}  rel {err/scale:.3e}  {'ok' if ok else 'OVER TOL'}"); return ok

print(f"layer {L}, pos {POS}: attention on the card vs the CPU realization")
launch(proj, PROJ_GRID, WG, PROJ_LDS, d_h, d_ln, d_q, d_kva, d_qo, d_kvao); sync()
ok = diff("attn_proj: q_proj.out", download(d_qo, Q_DIM * 4), vec("q.f32"))
ok &= diff("attn_proj: kv_a_proj.out", download(d_kvao, KVA_DIM * 4), vec("kva.f32"))
launch(mla, HEADS * splits(POS + 1), WG, MLA_LDS, d_qo, d_kvao, d_kvaln, d_kvb, d_cache, C.c_uint(POS), d_cos, d_sin, C.c_float(SCALE), d_part, d_mirror); sync()
row = download(C.c_void_p(d_cache.value + POS * KV_ROW * 4), KV_ROW * 4)
ok &= diff("attn_mla: cache row (c | k_pe)", row, cache[POS * KV_ROW * 4:(POS + 1) * KV_ROW * 4])
ok &= diff("attn_mla: mirror row", download(d_mirror, KV_ROW * 4), cache[POS * KV_ROW * 4:(POS + 1) * KV_ROW * 4])
launch(merge, HEADS, WG, MERGE_LDS, d_part, C.c_uint(splits(POS + 1)), d_kvb, d_heads); sync()
ok &= diff("attn_merge: o_proj.in (heads)", download(d_heads, H * 4), vec("heads.f32"))
launch(out, OUT_GRID, WG, OUT_LDS, d_heads, d_o, d_h); sync()
ok &= diff("attn_out: h + o_proj.out", download(d_h, H * 4), vec("h_out.f32"))

# timing: the three launches back to back, BATCH times in one sync window; h is re-uploaded each iter
BATCH = 16
weights_bytes = 2 * (Q_DIM * H + KVA_DIM * H + HEADS * 256 * R + H * H)
for pos in (POS, 511, 2047):
    def batch():
        for _ in range(BATCH): run(pos)
    batch(); sync(); ts = []
    for _ in range(ITERS):
        sync(); a = time.perf_counter(); batch(); sync(); ts.append((time.perf_counter() - a) / BATCH)
    ts.sort(); t = ts[len(ts) // 2]
    kv_bytes = (pos + 1) * KV_ROW * 4   # the rows leave DRAM once; the 16 heads' rereads (scores, then output) hit L2
    print(f"  pos {pos:5d}: {t*1e6:7.1f} µs per layer (4 launches)  weights {weights_bytes/1e6:.1f} MB + cache reads {kv_bytes/1e6:.1f} MB -> {(weights_bytes+kv_bytes)/t/1e9:.0f} GB/s, {(weights_bytes+kv_bytes)/t/1e9/CEIL:.2f} of {CEIL}")
sys.exit(0 if ok else 1)
