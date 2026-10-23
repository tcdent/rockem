//! bench-pool: the pinned pool under the two engines' call patterns in one process — Voxtral's plain and node-placed
//! runs, then GLM's prepared slotted jobs — to reproduce the two-model node's hang on the first GLM step
//! (2026-10-23) without the box. Prints each phase as it completes; a hang shows as the last line printed.
use common::pool::Pool;
use glm_5_3_flash::construct::{physical_cores, Threads};
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let cores = physical_cores();
    let pool: &'static Threads = Box::leak(Box::new(Threads::new(&cores)));
    println!("pool     : {} cores", cores.len());
    let hits = AtomicUsize::new(0);
    for round in 0..3 {
        // Voxtral's shapes: 96 items placed by node (one node here), 2048 items for the head, a plain run of 64
        pool.run_on(96, &|_| 0, &|c| c, &|_| { hits.fetch_add(1, Ordering::Relaxed); });
        pool.run_on(2048, &|c| (c / 512) % 1, &|c| c, &|_| { hits.fetch_add(1, Ordering::Relaxed); });
        pool.run(64, &|_| { hits.fetch_add(1, Ordering::Relaxed); });
        println!("voxtral  : round {round} done ({} items)", hits.load(Ordering::Relaxed));
    }
    pool.idle();
    println!("idle     : done");
    // GLM's shape: a prepared slotted job, announced and finished
    let n = 1000;
    let slots: Vec<u8> = (0..n).map(|i| pool.slot(0, i) as u8).collect();
    let done = std::sync::Arc::new(AtomicUsize::new(0));
    let d = done.clone();
    let job = pool.prepare_slotted(&slots, Box::new(move |_| { d.fetch_add(1, Ordering::Relaxed); }));
    println!("glm      : job prepared ({job})");
    pool.run_prepared(job);
    println!("glm      : job ran, {} items", done.load(Ordering::Relaxed));
    pool.idle();
    let d2 = done.clone();
    let job2 = pool.prepare_slotted(&slots, Box::new(move |_| { d2.fetch_add(1, Ordering::Relaxed); }));
    pool.run_prepared(job2);
    println!("glm      : second job ran, {} items", done.load(Ordering::Relaxed));
    pool.run_slotted(n, &|i| slots[i] as usize, &|_| { done.fetch_add(1, Ordering::Relaxed); });
    println!("glm      : run_slotted done, {} items", done.load(Ordering::Relaxed));
}
