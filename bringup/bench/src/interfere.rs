//! bench-interfere — what slows an L3-fed expert item while other cores stream from DRAM (099)? One core (`--warm-core`)
//! runs warm items (the production one-token kernel over a set it touched, L3-resident: `bench-order`'s fill counters)
//! while `j` streamer cores run the same kernel over a pool far past the caches, j = 0 … the streamer list's length;
//! the streamers either share the warm core's CCD (`--same`, default 57-63) or sit on another (`--other`, default 65-71).
//! Reported per j: the warm item's median time, the warm core's effective clock (APERF/MPERF over its items), the
//! streamers' aggregate GB/s.
//!
//!   bench-interfere [--warm-core 56] [--same 57-63] [--other 65-71] [--warm 9] [--reps 300]
use harness::arg;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

const K: usize = 4096;
const ROWS: usize = 64;

fn list(spec: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in spec.split(',') {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(part.parse().unwrap()),
        }
    }
    v
}
fn msr(cpu: usize, reg: u64) -> u64 {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(format!("/dev/cpu/{cpu}/msr")).expect("msr");
    let mut b = [0u8; 8];
    f.read_exact_at(&mut b, reg).expect("msr read");
    u64::from_le_bytes(b)
}
struct Set {
    w: Vec<u8>,
    s: Vec<f32>,
    ib: usize,
    sb: usize,
    items: usize,
}
fn set(items: usize, seed: u64) -> Set {
    use common::q4::{pack, GROUP};
    let mut rng = common::testing::Rng::new(seed);
    let w: Vec<i8> = (0..ROWS * K).map(|_| rng.i4()).collect();
    let sc: Vec<f32> = vec![0.01; ROWS * K / GROUP];
    let (pw1, ps1) = pack(ROWS, K, &w, &sc);
    let (ib, sb) = (pw1.len(), ps1.len());
    Set { w: (0..items).flat_map(|i| pw1.iter().map(move |&b| b ^ (i as u8 & 0x11))).collect(), s: (0..items).flat_map(|_| ps1.iter().copied()).collect(), ib, sb, items }
}
fn act(seed: u64) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    use common::q4::{QAct, GROUP};
    let mut rng = common::testing::Rng::new(seed);
    let (mut q, mut s, mut g) = (vec![0i8; K], vec![0f32; K / GROUP], vec![0i32; K / GROUP]);
    let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
    QAct { q: &mut q, scale: &mut s, gsum: &mut g }.quantize(&x);
    (q, s, g)
}

fn main() {
    let warm_core: usize = arg("--warm-core").map_or(56, |v| v.to_str().unwrap().parse().unwrap());
    let same = list(arg("--same").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("57-63"));
    let other = list(arg("--other").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("65-71"));
    let nw: usize = arg("--warm").map_or(9, |v| v.to_str().unwrap().parse().unwrap());
    let reps: usize = arg("--reps").map_or(300, |v| v.to_str().unwrap().parse().unwrap());
    println!("bench-interfere: warm core {warm_core}, {nw} warm items of {ROWS} rows (one token); streamers on its CCD {same:?} or another {other:?}");
    println!("{:<6} {:>3} {:>14} {:>14} {:>16}", "where", "j", "warm item µs", "warm GHz", "streamers GB/s");
    for (place, cores) in [("same", &same), ("other", &other)] {
        for j in 0..=cores.len() {
            if place == "other" && j == 0 {
                continue;
            }
            let stop = Arc::new(AtomicBool::new(false));
            let bytes = Arc::new(AtomicU64::new(0));
            let ready = Arc::new(std::sync::Barrier::new(j + 1));
            let streamers: Vec<_> = cores[..j]
                .iter()
                .enumerate()
                .map(|(i, &cpu)| {
                    let (stop, bytes, ready) = (stop.clone(), bytes.clone(), ready.clone());
                    std::thread::spawn(move || {
                        harness::bench::pin(cpu);
                        let pool = set((96 << 20) / (ROWS * K / 2), 100 + i as u64);
                        let (q, s, g) = act(7 + i as u64);
                        let a = common::q4::Act { q: &q, scale: &s, gsum: &g };
                        let mut out = vec![0f32; ROWS];
                        ready.wait();
                        let mut k = 0;
                        while !stop.load(Ordering::Relaxed) {
                            unsafe { common::q4::avx2::gemv::<K>(ROWS, pool.w.as_ptr().add(k * pool.ib), pool.s.as_ptr().add(k * pool.sb), &a, out.as_mut_ptr()) };
                            bytes.fetch_add(pool.ib as u64, Ordering::Relaxed);
                            k = (k + 1) % pool.items;
                        }
                        std::hint::black_box(out);
                    })
                })
                .collect();
            // the warm core: touch its set, then its items, timed; APERF/MPERF over the items
            let (wt, ghz) = {
                harness::bench::pin(warm_core);
                let ws = set(nw, 1);
                let (q, s, g) = act(3);
                let a = common::q4::Act { q: &q, scale: &s, gsum: &g };
                let mut out = vec![0f32; ROWS];
                ready.wait();
                let t_stream = std::time::Instant::now();
                let b0 = bytes.load(Ordering::Relaxed);
                let mut times = Vec::with_capacity(reps * nw);
                let (mut ap, mut mp) = (0u64, 0u64);
                for _ in 0..reps {
                    let mut sink = 0u64;
                    let mut i = 0;
                    while i < ws.w.len() {
                        sink = sink.wrapping_add(unsafe { std::ptr::read_volatile(ws.w.as_ptr().add(i)) } as u64);
                        i += 64;
                    }
                    std::hint::black_box(sink);
                    let (a0, m0) = (msr(warm_core, 0xE8), msr(warm_core, 0xE7));
                    for k in 0..nw {
                        let t = std::time::Instant::now();
                        unsafe { common::q4::avx2::gemv::<K>(ROWS, ws.w.as_ptr().add(k * ws.ib), ws.s.as_ptr().add(k * ws.sb), &a, out.as_mut_ptr()) };
                        times.push(t.elapsed().as_nanos() as u64);
                    }
                    let (a1, m1) = (msr(warm_core, 0xE8), msr(warm_core, 0xE7));
                    ap += a1 - a0;
                    mp += m1 - m0;
                }
                let gbs = (bytes.load(Ordering::Relaxed) - b0) as f64 / t_stream.elapsed().as_secs_f64() / 1e9;
                times.sort();
                // the P0 reference clock MPERF counts at: the base clock, from the TSC rate (2.65 GHz on the 7R13) — the ratio is what matters
                let base = 2.65;
                ((times[times.len() / 2] as f64 / 1e3, base * ap as f64 / mp.max(1) as f64), gbs)
            };
            stop.store(true, Ordering::Relaxed);
            for h in streamers {
                h.join().unwrap();
            }
            println!("{place:<6} {j:>3} {:>14.2} {:>14.2} {:>16.1}", wt.0, wt.1, ghz);
        }
    }
}
