//! bench-item — what in the node's expert item costs the time (099)? A warm item in the node (L3-resident lines) costs
//! ~12 µs; the bench's one 64-row `gemv` over the same bytes 4.6. One core (`--core`) runs warm items — a set it touched
//! just before, as the prefetch does — in shapes that add the node's item one element at a time:
//!
//! - `gemv64`: the bench's item, one 64-row `gemv` over one contiguous 136 KB stream;
//! - `gu`: the node's gate/up part (construct.rs `gu_f`): `q4::gemm` (one token: `gemv`) over 32 gate rows, then 32 up
//!   rows 128 KB further, their scales apart, and `swiglu_clamped` into `h` (the layout's item: `expert::Weights::gu`);
//! - `gu+q`: and every fourth part quantizes its group of `h` (`QAct::quantize`), as a group's last part does;
//! - `down`: the node's down part: `q4::gemm::<D_FF>` over 128 rows, its output stored, then `sfence`.
//!
//! `--dirty c`: before each rep a thread on core `c` writes the activations (`xq`, `hq`) and `h`, as the main thread and
//! the run's other cores do between layers — the item's reads of them then come from another core's cache (core 1, the
//! node's main thread, is on socket 0; the pool's cores are on both sockets). Per shape: median item µs, the core's clock
//! (APERF/MPERF), the item's lines by source (`common::pmc::FILLS`).
//!
//! `--cold-pct p --cold-run r`: after the touch, `p` % of every item's lines flushed (`clflush`) in runs of `r` lines
//! spread evenly over the item: an L3-fed item with part of its lines from DRAM, as the node's prefetched experts' are
//! (prefetch_use: 13–18 % of a complete expert's lines, 27–44 % of a partial's).
//!
//! `--bg cores`: streamers that run the production kernel over a pool past the caches throughout (the node's run: its
//! CCD's mates and the socket's other CCDs streaming cold items).
//!
//!   bench-item [--core 56] [--parts 10] [--reps 300] [--dirty c] [--cold-pct 0] [--cold-run 1] [--bg cores]
use common::q4::{pack, Act, QAct, GROUP};
use harness::arg;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const H: usize = 4096;
const D_FF: usize = 2048;
const GU_CHUNK: usize = 64;
const GU_ROWS: usize = 32;
const DOWN_ROWS: usize = 128;
const GU_W: usize = GU_CHUNK * H / 2;
const GU_S: usize = GU_CHUNK * (H / GROUP);
const GU_ITEM: usize = 2 * GU_W + 2 * 4 * GU_S;
const LIMIT: f32 = 7.0;

fn num(name: &str, d: usize) -> usize {
    arg(name).map_or(d, |v| v.to_str().unwrap().parse().unwrap())
}
fn msr(cpu: usize, reg: u64) -> u64 {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(format!("/dev/cpu/{cpu}/msr")).expect("msr");
    let mut b = [0u8; 8];
    f.read_exact_at(&mut b, reg).expect("msr read");
    u64::from_le_bytes(b)
}
fn touch(b: &[u8]) -> u64 {
    let mut s = 0u64;
    let mut i = 0;
    while i < b.len() {
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(b.as_ptr().add(i)) } as u64);
        i += 64;
    }
    s
}
fn f32s(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}
/// Bytes at a 2 MiB boundary (the node's experts sit in 2 MiB pages).
fn aligned(len: usize) -> &'static mut [u8] {
    let l = std::alloc::Layout::from_size_align(len.next_multiple_of(2 << 20), 2 << 20).unwrap();
    unsafe { std::slice::from_raw_parts_mut(std::alloc::alloc_zeroed(l), len) }
}

fn main() {
    let core = num("--core", 56);
    let parts = num("--parts", 10).next_multiple_of(4);
    let reps = num("--reps", 300);
    let dirty: Option<usize> = arg("--dirty").map(|v| v.to_str().unwrap().parse().unwrap());
    let cold_pct = num("--cold-pct", 0);
    let cold_run = num("--cold-run", 1).max(1);
    let bg: Vec<usize> = arg("--bg").map_or(Vec::new(), |p| {
        let mut v = Vec::new();
        for part in p.to_str().unwrap().split(',') {
            match part.split_once('-') {
                Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
                None => v.push(part.parse().unwrap()),
            }
        }
        v
    });
    let bg_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let bg_bytes = Arc::new(AtomicU64::new(0));
    for (i, &cpu) in bg.iter().enumerate() {
        let (stop, bytes) = (bg_stop.clone(), bg_bytes.clone());
        std::thread::spawn(move || {
            harness::bench::pin(cpu);
            let mut rng = common::testing::Rng::new(900 + i as u64);
            let wq: Vec<i8> = (0..GU_CHUNK * H).map(|_| rng.i4()).collect();
            let (pw, ps) = pack(GU_CHUNK, H, &wq, &vec![0.01; GU_CHUNK * H / GROUP]);
            let items = (96 << 20) / pw.len();
            let pool: Vec<u8> = (0..items).flat_map(|k| pw.iter().map(move |&b| b ^ (k as u8 & 0x11))).collect();
            let x: Vec<f32> = (0..H).map(|_| rng.unit()).collect();
            let (mut q, mut s, mut g) = (vec![0i8; H], vec![0f32; H / GROUP], vec![0i32; H / GROUP]);
            QAct { q: &mut q, scale: &mut s, gsum: &mut g }.quantize(&x);
            let a = Act { q: &q, scale: &s, gsum: &g };
            let mut out = vec![0f32; GU_CHUNK];
            let mut k = 0;
            while !stop.load(Ordering::Relaxed) {
                common::q4::gemv::<H>(GU_CHUNK, &pool[k * pw.len()..(k + 1) * pw.len()], &ps, &a, &mut out);
                bytes.fetch_add(pw.len() as u64, Ordering::Relaxed);
                k = (k + 1) % items;
            }
            std::hint::black_box(out);
        });
    }
    if !bg.is_empty() {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let (b0, t0) = (bg_bytes.load(Ordering::Relaxed), std::time::Instant::now());
        std::thread::sleep(std::time::Duration::from_millis(500));
        println!("background: {} streamers, {:.1} GB/s", bg.len(), (bg_bytes.load(Ordering::Relaxed) - b0) as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
    let pmc_ok = {
        let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
        common::pmc::Pmc::open(core).is_some_and(|p| p.program(&common::pmc::FILLS)) && common::pmc::allow_rdpmc()
    };
    harness::bench::pin(core);
    let mut rng = common::testing::Rng::new(5);
    // a layout gate/up item: [gate 64 rows | up 64 rows | gate scales | up scales]; part h of it is each's half
    let wq: Vec<i8> = (0..GU_CHUNK * H).map(|_| rng.i4()).collect();
    let (pw, ps) = pack(GU_CHUNK, H, &wq, &vec![0.01; GU_CHUNK * H / GROUP]);
    assert!(pw.len() == GU_W && ps.len() == GU_S);
    let items = parts / 2;
    let gu = aligned(items * GU_ITEM);
    for i in 0..items {
        let it = &mut gu[i * GU_ITEM..(i + 1) * GU_ITEM];
        for m in 0..2 {
            it[m * GU_W..(m + 1) * GU_W].iter_mut().zip(&pw).for_each(|(d, &s)| *d = s ^ (i as u8 & 0x11) ^ m as u8);
            let sb: Vec<u8> = ps.iter().flat_map(|v| v.to_le_bytes()).collect();
            it[2 * GU_W + m * 4 * GU_S..2 * GU_W + (m + 1) * 4 * GU_S].copy_from_slice(&sb);
        }
    }
    // the bench's item: 64 rows contiguous, its scales after (bench-interfere's)
    let g64 = aligned(parts * (GU_W + 4 * GU_S));
    for i in 0..parts {
        let it = &mut g64[i * (GU_W + 4 * GU_S)..(i + 1) * (GU_W + 4 * GU_S)];
        it[..GU_W].iter_mut().zip(&pw).for_each(|(d, &s)| *d = s ^ (i as u8 & 0x11));
        let sb: Vec<u8> = ps.iter().flat_map(|v| v.to_le_bytes()).collect();
        it[GU_W..].copy_from_slice(&sb);
    }
    // down parts: 128 rows of D_FF and their scales
    let dq: Vec<i8> = (0..DOWN_ROWS * D_FF).map(|_| rng.i4()).collect();
    let (dw, ds) = pack(DOWN_ROWS, D_FF, &dq, &vec![0.01; DOWN_ROWS * D_FF / GROUP]);
    let dbytes = dw.len() + 4 * ds.len();
    let down = aligned(parts * dbytes);
    for i in 0..parts {
        let it = &mut down[i * dbytes..(i + 1) * dbytes];
        it[..dw.len()].iter_mut().zip(&dw).for_each(|(d, &s)| *d = s ^ (i as u8 & 0x11));
        let sb: Vec<u8> = ds.iter().flat_map(|v| v.to_le_bytes()).collect();
        it[dw.len()..].copy_from_slice(&sb);
    }
    // the activations and the hidden state, in lines a dirtier rewrites each rep
    let xq = aligned(H);
    let xs = aligned(4 * H / GROUP);
    let xg = aligned(4 * H / GROUP);
    let hq = aligned(D_FF);
    let hs = aligned(4 * D_FF / GROUP);
    let hg = aligned(4 * D_FF / GROUP);
    let h = aligned(4 * D_FF);
    let out = aligned(4 * H);
    {
        let x: Vec<f32> = (0..H).map(|_| rng.unit()).collect();
        let (q, s, g) = (unsafe { &mut *(xq as *mut [u8] as *mut [i8]) }, unsafe { std::slice::from_raw_parts_mut(xs.as_mut_ptr() as *mut f32, H / GROUP) }, unsafe { std::slice::from_raw_parts_mut(xg.as_mut_ptr() as *mut i32, H / GROUP) });
        QAct { q, scale: s, gsum: g }.quantize(&x);
        let x: Vec<f32> = (0..D_FF).map(|_| rng.unit()).collect();
        let (q, s, g) = (unsafe { &mut *(hq as *mut [u8] as *mut [i8]) }, unsafe { std::slice::from_raw_parts_mut(hs.as_mut_ptr() as *mut f32, D_FF / GROUP) }, unsafe { std::slice::from_raw_parts_mut(hg.as_mut_ptr() as *mut i32, D_FF / GROUP) });
        QAct { q, scale: s, gsum: g }.quantize(&x);
    }
    let addrs = [xq.as_ptr() as usize, xq.len(), xs.as_ptr() as usize, xs.len(), xg.as_ptr() as usize, xg.len(), hq.as_ptr() as usize, hq.len(), hs.as_ptr() as usize, hs.len(), hg.as_ptr() as usize, hg.len(), h.as_ptr() as usize, h.len()];
    // the dirtier: on request r, rewrites every line of the activations and h with the bytes they hold (an RFO and a
    // dirty line in its cache, the values unchanged), then acks r
    let (req, ack) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    if let Some(c) = dirty {
        let (req, ack) = (req.clone(), ack.clone());
        std::thread::spawn(move || {
            harness::bench::pin(c);
            let mut seen = 0;
            loop {
                let r = req.load(Ordering::Acquire);
                if r == u64::MAX {
                    return;
                }
                if r != seen {
                    for k in (0..addrs.len()).step_by(2) {
                        let (p, n) = (addrs[k] as *mut u8, addrs[k + 1]);
                        let mut i = 0;
                        while i < n {
                            unsafe { std::ptr::write_volatile(p.add(i), std::ptr::read_volatile(p.add(i))) };
                            i += 64;
                        }
                    }
                    seen = r;
                    ack.store(r, Ordering::Release);
                }
                std::hint::spin_loop();
            }
        });
    }
    let act_x = || Act { q: unsafe { &*(xq as *const [u8] as *const [i8]) }, scale: f32s(xs), gsum: unsafe { std::slice::from_raw_parts(xg.as_ptr() as *const i32, H / GROUP) } };
    let act_h = || Act { q: unsafe { &*(hq as *const [u8] as *const [i8]) }, scale: f32s(hs), gsum: unsafe { std::slice::from_raw_parts(hg.as_ptr() as *const i32, D_FF / GROUP) } };
    let hp = h.as_mut_ptr() as *mut f32;
    let op = out.as_mut_ptr() as *mut f32;
    println!("bench-item: core {core}, {parts} warm parts a rep, {reps} reps, {cold_pct} % of lines cold in runs of {cold_run}{}; pmc {}", dirty.map_or(String::new(), |c| format!(", activations and h dirtied by core {c} each rep")), if pmc_ok { "on" } else { "off" });
    println!("{:<8} {:>9} {:>9} {:>7}   lines an item: {}", "shape", "item µs", "p90 µs", "GHz", common::pmc::FILL_NAMES.join(" / "));
    let mut r = 0u64;
    for shape in ["gemv64", "gu", "gu+q", "down"] {
        let set: &[u8] = match shape {
            "gemv64" => g64,
            "down" => down,
            _ => gu,
        };
        let mut times = Vec::with_capacity(reps * parts);
        let (mut ap, mut mp) = (0u64, 0u64);
        let mut fills = [0u64; 6];
        for _ in 0..reps {
            std::hint::black_box(touch(set));
            if cold_pct > 0 {
                // runs of `cold_run` lines, one every `cold_run · 100 / cold_pct` lines
                let lines = set.len() / 64;
                let period = (cold_run * 100 / cold_pct).max(cold_run);
                let mut l = 0;
                while l < lines {
                    for k in l..(l + cold_run).min(lines) {
                        unsafe { core::arch::x86_64::_mm_clflush(set.as_ptr().add(k * 64)) };
                    }
                    l += period;
                }
                unsafe { core::arch::x86_64::_mm_mfence() };
            }
            if dirty.is_some() {
                r += 1;
                req.store(r, Ordering::Release);
                while ack.load(Ordering::Acquire) != r {
                    std::hint::spin_loop();
                }
            }
            let (a0, m0) = (msr(core, 0xE8), msr(core, 0xE7));
            for p in 0..parts {
                let f0: [u64; 6] = if pmc_ok { std::array::from_fn(|i| common::pmc::rdpmc(i as u32)) } else { [0; 6] };
                let t = std::time::Instant::now();
                match shape {
                    "gemv64" => {
                        let it = &set[p * (GU_W + 4 * GU_S)..];
                        common::q4::gemv::<H>(GU_CHUNK, &it[..GU_W], f32s(&it[GU_W..GU_W + 4 * GU_S]), &act_x(), unsafe { std::slice::from_raw_parts_mut(op, GU_CHUNK) });
                    }
                    "gu" | "gu+q" => {
                        // expert::Weights::gu(c): part c of item c / 2, half c % 2 of each matrix
                        let it = &set[(p / 2) * GU_ITEM..(p / 2 + 1) * GU_ITEM];
                        let (hh, w, sc) = (p % 2, GU_W / 2, 4 * GU_S / 2);
                        let (gw, uw) = (&it[hh * w..(hh + 1) * w], &it[GU_W + hh * w..GU_W + (hh + 1) * w]);
                        let (gs, us) = (f32s(&it[2 * GU_W + hh * sc..2 * GU_W + (hh + 1) * sc]), f32s(&it[2 * GU_W + 4 * GU_S + hh * sc..2 * GU_W + 4 * GU_S + (hh + 1) * sc]));
                        let mut g = [[0f32; 2 * GU_ROWS]; 1];
                        let gp = g.as_mut_ptr() as *mut f32;
                        common::q4::gemm::<H>(GU_ROWS, gw, gs, 1, |_| act_x(), |i| unsafe { gp.add(i * 2 * GU_ROWS) });
                        common::q4::gemm::<H>(GU_ROWS, uw, us, 1, |_| act_x(), |i| unsafe { gp.add(i * 2 * GU_ROWS + GU_ROWS) });
                        let c = p % (D_FF / GU_ROWS);
                        common::swiglu::swiglu_clamped::<GU_ROWS>(&g[0], LIMIT, unsafe { std::slice::from_raw_parts_mut(hp.add(c * GU_ROWS), GU_ROWS) });
                        if shape == "gu+q" && p % 4 == 3 {
                            let gi = c / 4;
                            let x = unsafe { std::slice::from_raw_parts(hp.add(gi * GROUP), GROUP) };
                            let q = unsafe { std::slice::from_raw_parts_mut((hq.as_ptr() as *mut i8).add(gi * GROUP), GROUP) };
                            let s = unsafe { std::slice::from_raw_parts_mut((hs.as_ptr() as *mut f32).add(gi), 1) };
                            let g = unsafe { std::slice::from_raw_parts_mut((hg.as_ptr() as *mut i32).add(gi), 1) };
                            QAct { q, scale: s, gsum: g }.quantize(x);
                        }
                    }
                    _ => {
                        let it = &set[p * dbytes..(p + 1) * dbytes];
                        let r0 = (p % (H / DOWN_ROWS)) * DOWN_ROWS;
                        common::q4::gemm::<D_FF>(DOWN_ROWS, &it[..dw.len()], f32s(&it[dw.len()..]), 1, |_| act_h(), |_| unsafe { op.add(r0) });
                        unsafe { core::arch::x86_64::_mm_sfence() };
                    }
                }
                times.push(t.elapsed().as_nanos() as u64);
                if pmc_ok {
                    for i in 0..6 {
                        fills[i] += common::pmc::rdpmc(i as u32).wrapping_sub(f0[i]) & ((1 << 48) - 1);
                    }
                }
            }
            let (a1, m1) = (msr(core, 0xE8), msr(core, 0xE7));
            ap += a1 - a0;
            mp += m1 - m0;
        }
        times.sort_unstable();
        let n = times.len() as f64;
        let f: Vec<String> = fills.iter().map(|&v| format!("{:.0}", v as f64 / n)).collect();
        println!("{shape:<8} {:>9.2} {:>9.2} {:>7.2}   {}", times[times.len() / 2] as f64 / 1e3, times[times.len() * 9 / 10] as f64 / 1e3, 2.65 * ap as f64 / mp.max(1) as f64, f.join(" / "));
    }
    req.store(u64::MAX, Ordering::Release);
    bg_stop.store(true, Ordering::Relaxed);
}
