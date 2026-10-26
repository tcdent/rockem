//! bench-l3life — how Milan's L3 keeps and drops lines (099). Each core (`--cpus`) touches a fresh set of `--set-kb` KB (new
//! lines, last read a pool's length ago), then either waits `--gap-us` idle or touches `--filler-kb` of other fresh
//! lines (read once, never again), then reads its set again in touch order. Per decile of the set (in touch order): the
//! share of its lines that came from DRAM at the reread. One load a line throughout. All cores in step (a barrier at each
//! phase).
//!
//! `--sib-spin 1`: each core's SMT sibling (cpu + 96) spins throughout (`pause`), instead of halting. Reported also: the
//! first touch's rate a core (GB/s).
//!
//!   bench-l3life [--sib-spin 0] [--cpus 56] [--set-kb 8192] [--gap-us 0] [--filler-kb 0] [--reps 20] [--pool-mb 1024]
use harness::arg;
use std::sync::{Arc, Barrier, Mutex};

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
fn num(name: &str, d: usize) -> usize {
    arg(name).map_or(d, |v| v.to_str().unwrap().parse().unwrap())
}
#[inline(never)]
fn touch(p: *const u8, len: usize) -> u64 {
    let mut s = 0u64;
    let mut i = 0;
    while i < len {
        s = s.wrapping_add(unsafe { std::ptr::read_volatile(p.add(i)) } as u64);
        i += 64;
    }
    s
}
/// (lines from L3, lines from DRAM) since the last read: demand and L2-prefetch fills (`common::pmc::FILLS` 0, 1, 4, 5).
fn fills() -> [u64; 2] {
    let r = |i| common::pmc::rdpmc(i);
    [r(0).wrapping_add(r(4)), r(1).wrapping_add(r(2)).wrapping_add(r(5))]
}

fn main() {
    let cpus = list(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("56"));
    let set = num("--set-kb", 8192) << 10;
    let gap = num("--gap-us", 0) as u64;
    let filler = num("--filler-kb", 0) << 10;
    let reps = num("--reps", 20);
    let pool = num("--pool-mb", 1024) << 20;
    let sib_spin = num("--sib-spin", 0) == 1;
    let quit = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if sib_spin {
        for &c in &cpus {
            let q = quit.clone();
            std::thread::spawn(move || {
                harness::bench::pin(c + 96);
                while !q.load(std::sync::atomic::Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            });
        }
    }
    let touch_ns: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    // APERF / MPERF over the first touches, summed over the cores: their effective clock
    let perf: Arc<Mutex<[u64; 2]>> = Arc::new(Mutex::new([0; 2]));
    fn msr(cpu: usize, reg: u64) -> u64 {
        use std::os::unix::fs::FileExt;
        let f = std::fs::File::open(format!("/dev/cpu/{cpu}/msr")).expect("msr");
        let mut b = [0u8; 8];
        f.read_exact_at(&mut b, reg).expect("msr read");
        u64::from_le_bytes(b)
    }
    let t = cpus.len();
    let _ = std::fs::write("/proc/sys/kernel/nmi_watchdog", "0");
    assert!(cpus.iter().all(|&c| common::pmc::Pmc::open(c).is_some_and(|p| p.program(&common::pmc::FILLS))) && common::pmc::allow_rdpmc(), "pmc");
    // per decile: (L3, DRAM) lines summed over cores and reps; the first touch's and the filler's DRAM share
    let acc: Arc<Mutex<[[u64; 2]; 10]>> = Arc::new(Mutex::new([[0; 2]; 10]));
    let first: Arc<Mutex<[u64; 2]>> = Arc::new(Mutex::new([0; 2]));
    let bar = Arc::new(Barrier::new(t));
    let hs: Vec<_> = cpus
        .iter()
        .map(|&cpu| {
            let (acc, first, bar, touch_ns, perf) = (acc.clone(), first.clone(), bar.clone(), touch_ns.clone(), perf.clone());
            std::thread::spawn(move || {
                harness::bench::pin(cpu);
                let mem = vec![1u8; pool];
                let step = set + filler;
                assert!(3 * step <= pool, "pool too small for set + filler");
                let mut at = 0usize;
                let mut mine = [[0u64; 2]; 10];
                let mut f1 = [0u64; 2];
                for rep in 0..reps + 1 {
                    if at + step > pool {
                        at = 0;
                    }
                    let (sp, fp) = (unsafe { mem.as_ptr().add(at) }, unsafe { mem.as_ptr().add(at + set) });
                    at += step;
                    bar.wait();
                    let c0 = fills();
                    let (a0, m0) = (msr(cpu, 0xE8), msr(cpu, 0xE7));
                    let tt = std::time::Instant::now();
                    std::hint::black_box(touch(sp, set));
                    if rep > 0 {
                        *touch_ns.lock().unwrap() += tt.elapsed().as_nanos() as u64;
                        let mut p = perf.lock().unwrap();
                        p[0] += msr(cpu, 0xE8) - a0;
                        p[1] += msr(cpu, 0xE7) - m0;
                    }
                    let c1 = fills();
                    bar.wait();
                    if filler > 0 {
                        std::hint::black_box(touch(fp, filler));
                    }
                    if gap > 0 {
                        let g = std::time::Instant::now();
                        while g.elapsed().as_micros() < gap as u128 {
                            std::hint::spin_loop();
                        }
                    }
                    bar.wait();
                    let dec = set / 10 / 64 * 64;
                    let mut prev = fills();
                    for d in 0..10 {
                        std::hint::black_box(touch(unsafe { sp.add(d * dec) }, dec));
                        let now = fills();
                        if rep > 0 {
                            mine[d][0] += now[0].wrapping_sub(prev[0]) & ((1 << 48) - 1);
                            mine[d][1] += now[1].wrapping_sub(prev[1]) & ((1 << 48) - 1);
                        }
                        prev = now;
                    }
                    if rep > 0 {
                        f1[0] += c1[0].wrapping_sub(c0[0]) & ((1 << 48) - 1);
                        f1[1] += c1[1].wrapping_sub(c0[1]) & ((1 << 48) - 1);
                    }
                }
                let mut a = acc.lock().unwrap();
                for d in 0..10 {
                    a[d][0] += mine[d][0];
                    a[d][1] += mine[d][1];
                }
                let mut f = first.lock().unwrap();
                f[0] += f1[0];
                f[1] += f1[1];
                std::hint::black_box(mem);
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    quit.store(true, std::sync::atomic::Ordering::Relaxed);
    let a = acc.lock().unwrap();
    let (h, m): (u64, u64) = (a.iter().map(|x| x[0]).sum(), a.iter().map(|x| x[1]).sum());
    let f = first.lock().unwrap();
    let dec: Vec<String> = a.iter().map(|x| format!("{:.1}", 100.0 * x[1] as f64 / (x[0] + x[1]).max(1) as f64)).collect();
    println!(
        "{t} cores × {} KB (CCD total {:.1} MB), gap {gap} µs, filler {} KB a core{} | first touch {:.2} GB/s a core at {:.2} GHz, from DRAM {:.0} % | reread from DRAM {:.1} % | by decile of touch order: {}",
        set >> 10,
        (t * set) as f64 / (1 << 20) as f64,
        filler >> 10,
        if sib_spin { ", siblings spinning" } else { "" },
        (set as f64 * reps as f64 * t as f64) / *touch_ns.lock().unwrap() as f64,
        { let p = perf.lock().unwrap(); 2.65 * p[0] as f64 / p[1].max(1) as f64 },
        100.0 * f[1] as f64 / (f[0] + f[1]).max(1) as f64,
        100.0 * m as f64 / (h + m).max(1) as f64,
        dec.join(" ")
    );
}
