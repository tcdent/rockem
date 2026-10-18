//! bench-p2p — the links between the cards and the host, measured for docs/queue/001 (peer VRAM
//! writes for the handoffs). Three measures, beside the serving node (a few pages of VRAM):
//!   1. card ↔ card ping-pong: a round trip through peer VRAM (each card polls its own word, the
//!      other writes it over the link) against one through host memory (today's handoffs), with a
//!      payload checked after its word (visibility: mismatches counted).
//!   2. host → card word, card → host word: the flag round trip from a core on each socket.
//!   3. the BAR write rate (an expert slot's 13 MB) into each card from a core on each socket.
//!
//!   bench-p2p --elf <bringup_kernels.elf> [--rounds 2000] [--words 16384]
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::arg;
use std::sync::atomic::{fence, AtomicU32, Ordering};
use std::time::Instant;

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}
fn pin(cpu: usize) {
    let mut mask = [0u64; 16];
    mask[cpu / 64] |= 1 << (cpu % 64);
    assert_eq!(unsafe { sched_setaffinity(0, 128, mask.as_ptr()) }, 0, "pin to cpu {cpu}");
}
/// The first cpu of each NUMA node, with the node.
fn node_cpus() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    for n in 0..8 {
        if let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/cpulist")) {
            if let Some(first) = s.trim().split([',', '-']).next().and_then(|x| x.parse().ok()) {
                v.push((n, first));
            }
        }
    }
    v
}
fn word(m: &kfd::Mem, off: usize) -> &AtomicU32 {
    unsafe { AtomicU32::from_ptr(m.at(off) as *mut u32) }
}

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let rounds: u32 = arg("--rounds").map_or(2000, |v| v.to_str().unwrap().parse().unwrap());
    let words: u32 = arg("--words").map_or(16384, |v| v.to_str().unwrap().parse().unwrap());
    let a = kfd::Kfd::open(&elf, false, 0);
    let b = kfd::Kfd::open(&elf, false, 1);
    let kern = a.kernel("pingpong");
    let kern_b = b.kernel("pingpong");
    const DATA: usize = 4096; // the word at 0, the payload from here
    let bytes = DATA + words as usize * 4;
    let errors = a.host_coherent(4096);
    let nodes = node_cpus();
    println!("bench-p2p: {rounds} rounds, payload {} KB; nodes {:?}", words * 4 / 1024, nodes);

    // 1. card ↔ card, through peer VRAM and through host memory
    for (name, ma, mb) in [("peer VRAM", a.vram_shared(bytes, &[&b]), b.vram_shared(bytes, &[&a])), ("host memory", a.host_shared(bytes, &[&b]), b.host_shared(bytes, &[&a]))] {
        let run = |n: u32| -> (f64, u32) {
            word(&ma, 0).store(0, Ordering::SeqCst);
            word(&mb, 0).store(0, Ordering::SeqCst);
            word(&errors, 0).store(0, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            // b waits, a opens
            let mut ab = kfd::Args::new();
            ab.ptr(mb.at(0)).ptr(ma.at(0)).ptr(mb.at(DATA)).ptr(ma.at(DATA)).u32(words).u32(n).u32(0).ptr(errors.at(0));
            let mut aa = kfd::Args::new();
            aa.ptr(ma.at(0)).ptr(mb.at(0)).ptr(ma.at(DATA)).ptr(mb.at(DATA)).u32(words).u32(n).u32(1).ptr(errors.at(0));
            b.launch(&kern_b, 1, 32, 0, &ab);
            b.fence();
            let t = Instant::now();
            a.launch(&kern, 1, 32, 0, &aa);
            a.sync();
            b.wait();
            (t.elapsed().as_secs_f64() * 1e6, word(&errors, 0).load(Ordering::SeqCst))
        };
        run(4);
        let (t1, _) = run(1);
        let mut per = Vec::new();
        let mut errs = 0;
        for _ in 0..5 {
            let (tn, e) = run(rounds);
            per.push((tn - t1) / (rounds - 1) as f64);
            errs += e;
        }
        per.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("card↔card {name:<12}: round trip {:7.2} µs (min {:.2}, max {:.2}); {} KB a message; payload mismatches {errs}", per[2], per[0], per[4], words * 4 / 1024);
    }

    // 2. host ↔ card word round trip from each socket; 3. BAR write rate from each socket
    let slab = vec![0x5au8; 13 << 20];
    for (ci, (card, kern)) in [(&a, &kern), (&b, &kern_b)].into_iter().enumerate() {
        let mine = card.vram_shared(4096, &[]); // the card's word, UC VRAM, host writes it through the BAR
        let theirs = card.host_coherent(4096); // the host's word, the card writes it
        let big = card.vram_visible(13 << 20);
        for &(node, cpu) in &nodes {
            pin(cpu);
            let run = |n: u32| -> f64 {
                word(&mine, 0).store(0, Ordering::SeqCst);
                word(&theirs, 0).store(0, Ordering::SeqCst);
                fence(Ordering::SeqCst);
                let mut args = kfd::Args::new();
                args.ptr(mine.at(0)).ptr(theirs.at(0)).ptr(mine.at(0)).ptr(theirs.at(0)).u32(0).u32(n).u32(0).ptr(errors.at(0));
                card.launch(kern, 1, 32, 0, &args);
                card.fence();
                std::thread::sleep(std::time::Duration::from_millis(20)); // the kernel is up and spinning
                let t = Instant::now();
                for r in 1..=n {
                    word(&mine, 0).store(r, Ordering::SeqCst);
                    while word(&theirs, 0).load(Ordering::SeqCst) != r {}
                }
                let us = t.elapsed().as_secs_f64() * 1e6;
                card.wait();
                us
            };
            run(4);
            let mut per = Vec::new();
            for _ in 0..5 {
                per.push(run(rounds) / rounds as f64);
            }
            per.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let mut rate = Vec::new();
            for _ in 0..5 {
                let t = Instant::now();
                card.upload(&big, 0, &slab);
                rate.push(slab.len() as f64 / t.elapsed().as_secs_f64() / 1e9);
            }
            rate.sort_by(|x, y| x.partial_cmp(y).unwrap());
            println!("card {ci} from node {node} (cpu {cpu}): host→card→host word round trip {:6.2} µs (min {:.2}); BAR write 13 MB {:5.2} GB/s (max {:.2})", per[2], per[0], rate[2], rate[4]);
        }
    }
}
