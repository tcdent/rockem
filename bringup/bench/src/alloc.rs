//! bench-alloc: what the node's per-layer allocations cost under a libc's malloc — the shapes the plan, the pool's job
//! and the prefetch allocate every CPU layer (construct.rs: `plan_run`, `Threads::build`, `prefetch`), allocated and
//! freed in a loop, on one thread and across two (built on one, freed on another, as a builder-core plan is).
//! Build for `x86_64-unknown-linux-musl` (the node's target) and for gnu to compare.
use std::hint::black_box;
use std::sync::atomic::AtomicU8;
use std::time::Instant;

struct Job(Vec<u8>, Vec<u64>, Vec<u64>, Vec<u32>, Vec<u8>);
struct Plan(Vec<usize>, Vec<usize>, Vec<(u16, u8)>, Vec<usize>, Vec<u8>, Vec<(u16, u16, u8)>, Vec<AtomicU8>, Vec<AtomicU8>, [Box<[u8; 32]>; 4]);
struct Pf(Vec<(usize, usize, u16, u16)>, Vec<u8>, Vec<u16>);

fn job() -> Job {
    Job(vec![0u8; 4096], vec![0u64; 88], vec![0u64; 88], vec![0u32; 900], vec![0u8; 900])
}
fn plan() -> Plan {
    Plan(vec![0; 257], vec![0; 257], vec![(0, 0); 384], vec![0; 13], vec![0; 624], vec![(0, 0, 0); 624], (0..312).map(|_| AtomicU8::new(0)).collect(), (0..256).map(|_| AtomicU8::new(0)).collect(), std::array::from_fn(|_| Box::new([0u8; 32])))
}
fn pf() -> Pf {
    let mut v = Vec::with_capacity(1248);
    v.push((0, 0, 0, 0));
    Pf(v, Vec::with_capacity(1248), vec![0; 624])
}

fn time<T>(name: &str, iters: usize, f: impl Fn() -> T) {
    let mut best = f64::MAX;
    let mut sum = 0.0;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let per = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
        best = best.min(per);
        sum += per;
    }
    println!("{name:28} {best:8.0} ns best   {:8.0} ns mean of 5 runs", sum / 5.0);
}

fn main() {
    let libc = if cfg!(target_env = "musl") { "musl" } else { "gnu" };
    println!("bench-alloc ({libc}), ns an alloc+free of each shape");
    let n = 20000;
    time("job (4 KB + 4 small)", n, || drop(job()));
    time("plan (8 vecs + 4 boxes)", n, || drop(plan()));
    time("prefetch (30 KB + 2)", n, || drop(pf()));
    time("layer (job x2 + plan + pf)", n, || drop((job(), job(), plan(), pf())));
    // across threads: built on one, dropped on another (a builder's plan, freed by the main thread)
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Job, Job, Plan)>(4);
    let t = Instant::now();
    let producer = std::thread::spawn(move || {
        for _ in 0..n {
            tx.send((job(), job(), plan())).unwrap();
        }
    });
    for _ in 0..n {
        drop(rx.recv().unwrap());
    }
    producer.join().unwrap();
    println!("{:28} {:8.0} ns a layer's set, built on one thread, freed on another", "cross-thread", t.elapsed().as_secs_f64() * 1e9 / n as f64);
}
