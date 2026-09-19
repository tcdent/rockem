#!/usr/bin/env python3
"""Run deepseek_v2_lite's GPU expert kernels on gfx1100 through HIP via ctypes and check
them byte-for-byte against the scalar realization's outputs from `dump-expert`, then time
them against the 760 GB/s streaming ceiling.

  expert.py <libamdhip64.so> <deepseek_v2_lite_gpu.elf> <experts.q4g128.bin> <vectors dir> [iters=20]

No torch, no numpy, no compiler. Launch loops are bounded by count."""
import ctypes as C, sys, time, struct, os

hip = C.CDLL(sys.argv[1], mode=C.RTLD_GLOBAL)
ELF = open(sys.argv[2], "rb").read()
BUNDLE, VEC = sys.argv[3], sys.argv[4]
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
meta = dict(l.split() for l in open(os.path.join(VEC, "meta.txt")))
OFF, RECORD, NREC, H, D_FF, N, WG, UP_T, DN_T, UP_LDS, DN_LDS, CEIL = (int(meta[k]) for k in
    ("record_offset", "record_bytes", "layer_records", "H", "D_FF", "n", "wg", "up_tiles", "down_tiles", "up_lds", "down_lds", "vram_ceiling_gbps"))

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

# the whole layer's records resident, so timing streams from VRAM rather than the 80 MB
# infinity cache: the 6-expert timing rotates through every disjoint set of 6 in the layer
with open(BUNDLE, "rb") as f:
    f.seek(OFF); layer = f.read(NREC * RECORD)
assert len(layer) == NREC * RECORD
d_layer = upload(layer)
experts = list(struct.unpack(f"<{N}I", vec("experts.u32")))
d_off = upload(struct.pack(f"<{N}Q", *[e * RECORD for e in experts]))            # the real hits
d_off_all = upload(struct.pack(f"<{NREC}Q", *[e * RECORD for e in range(NREC)]))  # every record, for bandwidth
d_off_rot = [upload(struct.pack(f"<{N}Q", *[((k * N + i) % NREC) * RECORD for i in range(N)])) for k in range(NREC // N)]
d_xq, d_xs, d_xg = upload(vec("xq.i8")), upload(vec("xs.f32")), upload(vec("xg.i32"))
d_gu, d_down = dmalloc(NREC * 2 * D_FF * 4), dmalloc(NREC * H * 4)
up, down = fn("experts_up"), fn("experts_down")
def grid(n, tiles): return (n * tiles * 32 + WG - 1) // WG   # one wave per 4-row tile
def launch_up(n, off): launch(up, grid(n, UP_T), WG, UP_LDS, d_layer, off, C.c_uint(n), d_xq, d_xs, d_xg, d_gu)
def launch_dn(n, off): launch(down, grid(n, DN_T), WG, DN_LDS, d_layer, off, C.c_uint(n), d_gu, d_down)

def check(name, got, want):
    if got == want:
        print(f"  {name}: {len(want)//4} f32 bit-identical to the scalar realization"); return True
    g, w = f32s(got), f32s(want)
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(g, w)) if a != b]
    print(f"  {name}: {len(bad)} of {len(w)} differ; first: {bad[:4]}"); return False

print(f"code object {len(ELF)} B; record {RECORD} B; {NREC} records resident; experts {experts}")
launch_up(N, d_off); sync(); ok = check(f"experts_up (n={N}, gate|up)", download(d_gu, N * 2 * D_FF * 4), vec("gate_up.f32"))
launch_dn(N, d_off); sync(); ok &= check(f"experts_down (n={N}, swiglu + quantize on the card)", download(d_down, N * H * 4), vec("down.f32"))

golden = vec("golden_routed.f32")
if golden:
    wt = f32s(vec("weights.f32")); got = f32s(download(d_down, N * H * 4)); want = f32s(golden)
    mine = [sum(wt[k] * got[k * H + i] for k in range(N)) for i in range(H)]
    ref = [sum(wt[k] * want[k * H + i] for k in range(N)) for i in range(H)]
    err = max(abs(a - b) for a, b in zip(mine, ref)); scale = max(abs(b) for b in ref)
    print(f"  real token, layer {meta['layer']}: routed sum vs oracle f32  max|Δ| {err:.3e}  rel {err/scale:.3e}")

BATCH = 32
def timed(n, offs, label):
    """BATCH launch pairs back-to-back inside one sync window; `offs` rotate per pair."""
    def run():
        for b in range(BATCH): launch_up(n, offs[b % len(offs)]); launch_dn(n, offs[b % len(offs)])
    run(); sync()
    ts = []
    for _ in range(ITERS):
        sync(); a = time.perf_counter(); run(); sync(); ts.append((time.perf_counter() - a) / BATCH)
    ts.sort(); t = ts[len(ts) // 2]
    print(f"  {label:<34} {t*1e6:8.1f} µs per launch pair  = {t/n*1e6:6.1f} µs/expert  {n*RECORD/t/1e9:5.0f} GB/s  {n*RECORD/t/1e9/CEIL:.2f} of {CEIL}")
print(f"timing, median of {ITERS} × {BATCH} back-to-back pairs (record at {CEIL} GB/s = {RECORD/CEIL/1e9*1e6:.1f} µs; ~2.5 µs launch floor each):")
timed(1, [d_off], "1 expert, same record (cached)")
timed(N, [d_off], f"{N} experts, same set (cache-warm)")
timed(N, d_off_rot, f"{N} experts, rotating sets (VRAM)")
timed(NREC, [d_off_all], f"{NREC} experts (VRAM)")
sys.exit(0 if ok else 1)
