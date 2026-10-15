//! bench-kfd — the launcher alone: the expert pair on six records and the
//! sync round trip, timed; the outputs' checksum, to compare runs.
//!
//!   bench-kfd --bundle <dir> [--layer 10] [--iters 200]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
#[path = "../../../node/src/v2lite_kfd.rs"]
mod v2lite_kfd;

use common::q4::{QAct, GROUP};
use deepseek_v2_lite::layout::{self, RECORD};
use deepseek_v2_lite::{D_FF, H, TOP_K};
use harness::{arg, arg_required};

fn main() {
    let bundle = arg_required("--bundle");
    let layer: usize = arg("--layer").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(10);
    let iters: usize = arg("--iters").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(200);
    let elf = std::fs::read(bundle.join(layout::KERNELS_FILE)).unwrap();
    let file = std::fs::File::open(bundle.join(layout::EXPERTS_FILE)).unwrap();
    let stream = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let n = TOP_K;
    let first = layout::record(layer, 0).start;
    let records = &stream[first..first + n * RECORD];
    let offs: Vec<u64> = (0..n).map(|e| (e * RECORD) as u64).collect();

    let mut rng = common::testing::Rng::new(5);
    let x: Vec<f32> = (0..H).map(|_| rng.unit()).collect();
    let (mut xq, mut xs, mut xg) = (vec![0i8; H], vec![0f32; H / GROUP], vec![0i32; H / GROUP]);
    QAct { q: &mut xq, scale: &mut xs, gsum: &mut xg }.quantize(&x);
    let bytes = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) };

    let k = kfd::Kfd::new(&elf, true);
    let (up, down) = (k.kernel("experts_up"), k.kernel("experts_down"));
    let d_stream = k.vram_visible(records.len());
    k.upload(&d_stream, 0, records);
    let d_off = k.host(n * 8);
    k.upload(&d_off, 0, unsafe { std::slice::from_raw_parts(offs.as_ptr() as *const u8, n * 8) });
    let d_xq = k.host(H);
    k.upload(&d_xq, 0, unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, H) });
    let d_xs = k.host(4 * H / GROUP);
    k.upload(&d_xs, 0, bytes(&xs));
    let d_xg = k.host(4 * H / GROUP);
    k.upload(&d_xg, 0, unsafe { std::slice::from_raw_parts(xg.as_ptr() as *const u8, 4 * H / GROUP) });
    let d_gu = k.vram(n * 2 * D_FF * 4);
    let d_out = k.host(n * H * 4);
    let (grid_up, grid_dn) = ((n as u32 * deepseek_v2_lite::expert::gfx11_consts::UP_TILES * 32).div_ceil(kfd::WG), (n as u32 * deepseek_v2_lite::expert::gfx11_consts::DOWN_TILES * 32).div_ceil(kfd::WG));
    let pair = |k: &kfd::Kfd| {
        let mut a = kfd::Args::new();
        a.ptr(d_stream.va).ptr(d_off.va).u32(n as u32).ptr(d_xq.va).ptr(d_xs.va).ptr(d_xg.va).ptr(d_gu.va);
        k.launch(&up, grid_up, kfd::WG, v2lite_kfd::stage_bytes(H), &a);
        let mut b = kfd::Args::new();
        b.ptr(d_stream.va).ptr(d_off.va).u32(n as u32).ptr(d_gu.va).ptr(d_out.va);
        k.launch(&down, grid_dn, kfd::WG, v2lite_kfd::stage_bytes(D_FF), &b);
    };
    pair(&k);
    k.sync();
    let out: Vec<f32> = unsafe { std::slice::from_raw_parts(d_out.ptr() as *const f32, n * H).to_vec() };
    let t = std::time::Instant::now();
    for _ in 0..iters {
        pair(&k);
        k.sync();
    }
    let per_pair = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let (ts0, ts1) = k.last_ts();
    let t = std::time::Instant::now();
    for _ in 0..iters {
        k.sync();
    }
    let per_sync = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let sum: u64 = out.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3));
    println!("expert pair + sync {per_pair:.1} µs ({:.0} GB/s of records), empty sync {per_sync:.1} µs, barrier packet on the card {:.1} µs, outputs {sum:016x}", records.len() as f64 / per_pair / 1e3, ts1.wrapping_sub(ts0) as f64 / 100.0);
}
