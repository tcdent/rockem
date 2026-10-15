//! bench-box — this box's ceilings, measured: what each NUMA node's memory
//! streams to its own cores, what both stream together, what interleaved
//! pages stream to every core (the loader's placement); and, with a card free
//! (no other process on it), what the card streams from its VRAM, what it
//! reads from host memory across the link (the data-wait path), and what the
//! host writes into and reads back from host-visible VRAM through the BAR.
//! The numbers the token's theoretical time is built from.
//!
//!   bench-box [--mb 1024] [--iters 3] [--no-gpu] [--elf <code object with `stream`>]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::{arg, flag};
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}

#[target_feature(enable = "avx2")]
unsafe fn stream_sum(p: *const u8, len: usize) -> u64 {
    use core::arch::x86_64::*;
    let (mut a, mut b, mut c, mut d) = (_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256());
    let mut i = 0;
    while i + 128 <= len {
        a = _mm256_add_epi64(a, _mm256_loadu_si256(p.add(i) as *const __m256i));
        b = _mm256_add_epi64(b, _mm256_loadu_si256(p.add(i + 32) as *const __m256i));
        c = _mm256_add_epi64(c, _mm256_loadu_si256(p.add(i + 64) as *const __m256i));
        d = _mm256_add_epi64(d, _mm256_loadu_si256(p.add(i + 96) as *const __m256i));
        i += 128;
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a, b), _mm256_add_epi64(c, d));
    let mut o = [0u64; 4];
    _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, s);
    o.iter().sum()
}

extern "C" {
    fn syscall(nr: i64, ...) -> i64;
}
/// MPOL_INTERLEAVE over every node for this thread's new pages, or back to default.
fn interleave(on: bool) {
    let mask: u64 = u64::MAX;
    unsafe { syscall(238, if on { 3 } else { 0 }, if on { &mask as *const u64 } else { std::ptr::null() }, if on { 64u64 } else { 0 }) };
}

/// The CPUs of each NUMA node, first SMT sibling only, from /sys.
fn nodes() -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for n in 0.. {
        let Ok(list) = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/cpulist")) else { break };
        let mut cpus = Vec::new();
        for part in list.trim().split(',') {
            let (a, b) = part.split_once('-').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())).unwrap_or_else(|| (part.parse().unwrap(), part.parse().unwrap()));
            for c in a..=b {
                let sib = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")).unwrap_or_default();
                let first: usize = sib.trim().split([',', '-']).next().and_then(|s| s.parse().ok()).unwrap_or(c);
                if first == c {
                    cpus.push(c);
                }
            }
        }
        out.push(cpus);
    }
    out
}

/// Threads pinned to `cpus`, each streaming its own buffer (`mb` MB, first-touched by itself
/// under the current policy), all starting together: aggregate GB/s, best of `iters`.
fn dram(cpus: &[usize], mb: usize, iters: usize, inter: bool) -> f64 {
    let n = mb << 20;
    let start = Barrier::new(cpus.len());
    let results: Vec<f64> = std::thread::scope(|sc| {
        let hs: Vec<_> = cpus
            .iter()
            .map(|&cpu| {
                let start = &start;
                sc.spawn(move || {
                    harness::bench::pin(cpu);
                    interleave(inter);
                    let mut buf = vec![0u8; n];
                    for (i, b) in buf.iter_mut().enumerate().step_by(4096) {
                        *b = i as u8;
                    }
                    interleave(false);
                    let mut best = f64::MAX;
                    for _ in 0..iters {
                        start.wait();
                        let t = std::time::Instant::now();
                        let s = unsafe { stream_sum(buf.as_ptr(), n) };
                        let dt = t.elapsed().as_secs_f64();
                        std::hint::black_box(s);
                        best = best.min(dt);
                    }
                    n as f64 / best / 1e9
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.iter().sum()
}

fn main() {
    let mb = num("--mb", 1024);
    let iters = num("--iters", 3);
    let nodes = nodes();
    println!("box      : {} NUMA nodes, {} physical cores", nodes.len(), nodes.iter().map(|n| n.len()).sum::<usize>());
    let per_thread_mb = (mb / nodes.iter().map(|n| n.len()).max().unwrap_or(1)).max(16);
    for (i, cpus) in nodes.iter().enumerate() {
        println!("dram     : node {i} local, {} threads: {:.0} GB/s", cpus.len(), dram(cpus, per_thread_mb, iters, false));
    }
    let all: Vec<usize> = nodes.iter().flatten().copied().collect();
    if nodes.len() > 1 {
        println!("dram     : every node local, {} threads: {:.0} GB/s", all.len(), dram(&all, per_thread_mb, iters, false));
        println!("dram     : interleaved pages, {} threads: {:.0} GB/s (the loader's placement)", all.len(), dram(&all, per_thread_mb, iters, true));
    }
    if flag("--no-gpu") {
        return;
    }
    let elf = std::fs::read(arg("--elf").expect("--elf <code object>")).unwrap();
    use glm_5_3_flash::probe::gfx11_consts::{WG, WORDS_PER_WG};
    let words = (mb << 20) / 4 / WORDS_PER_WG * WORDS_PER_WG;
    let grid = (words / WORDS_PER_WG) as u32;
    for c in 0..kfd::Kfd::cards().len() {
        let k = kfd::Kfd::open(&elf, false, c);
        let stream = k.kernel("stream");
        let out = k.vram_visible(grid as usize * WG as usize * 4);
        let host_src: Vec<u8> = (0..words * 4).map(|i| i as u8).collect();
        let best = |mut f: Box<dyn FnMut()>| -> f64 {
            let mut b = f64::MAX;
            for _ in 0..iters {
                let t = std::time::Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            (words * 4) as f64 / b / 1e9
        };
        // host → BAR (a weight upload), BAR → host (a readback)
        let vram = k.vram_visible(words * 4);
        let up = best(Box::new(|| k.upload(&vram, 0, &host_src)));
        let mut back = vec![0u8; words * 4];
        let down = best(Box::new(|| kfd::read(&vram, 0, &mut back[..])));
        // the card streaming its VRAM, and host memory across the link
        let mut a = kfd::Args::new();
        a.ptr(vram.va).ptr(out.va);
        let vr = best(Box::new(|| {
            k.launch(&stream, grid, WG, 0, &a);
            k.sync();
        }));
        let host = k.host_coherent(words * 4);
        k.upload(&host, 0, &host_src);
        let mut b = kfd::Args::new();
        b.ptr(host.va).ptr(out.va);
        let hr = best(Box::new(|| {
            k.launch(&stream, grid, WG, 0, &b);
            k.sync();
        }));
        println!("card {c}   : VRAM stream {vr:.0} GB/s; reads host memory over the link {hr:.1} GB/s; host writes the BAR {up:.1} GB/s, reads it back {down:.1} GB/s ({} MB each)", words * 4 >> 20);
    }
}
