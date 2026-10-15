//! The probe mode (`rockem.probe=1`): the node measures this box and reports
//! through its log, then idles — an alternate state of execution, not a
//! side tool, so the baseline a token's theoretical time is built from comes
//! from the same binary on the same boot (tcdent, 2026-10-15). Today: what
//! each NUMA node's memory streams to its own cores, what both stream
//! together, what interleaved pages stream to every core; the bf16, FP8 and
//! INT4 GEMV rates of one core and of the whole pool from DRAM; and, per card,
//! what it streams from VRAM, what it reads from host memory across the link,
//! and what the host writes into and reads back from the BAR. Add a probe
//! whenever a question is "how fast is X here".

use crate::kfd;
use glm_5_3_flash::construct::{physical_cores, Threads};
use std::sync::Barrier;

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
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}
fn interleave(on: bool) {
    let mask: u64 = u64::MAX;
    unsafe { syscall(238, if on { 3 } else { 0 }, if on { &mask as *const u64 } else { std::ptr::null() }, if on { 64u64 } else { 0 }) };
}
fn pin(core: usize) {
    let mut mask = [0u64; 16];
    mask[core / 64] |= 1 << (core % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}

/// The physical cores of each NUMA node, from /sys.
fn nodes() -> Vec<Vec<usize>> {
    let phys = physical_cores();
    let mut out = Vec::new();
    for n in 0.. {
        let Ok(list) = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/cpulist")) else { break };
        let mut cpus = Vec::new();
        for part in list.trim().split(',') {
            let (a, b) = part.split_once('-').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())).unwrap_or_else(|| (part.parse().unwrap(), part.parse().unwrap()));
            cpus.extend((a..=b).filter(|c| phys.contains(c)));
        }
        if !cpus.is_empty() {
            out.push(cpus);
        }
    }
    if out.is_empty() {
        out.push(phys);
    }
    out
}

/// Threads pinned to `cpus`, each streaming its own `mb` MB (first-touched by itself under
/// the policy), all starting together: aggregate GB/s, best of `iters`.
fn dram(cpus: &[usize], mb: usize, iters: usize, inter: bool) -> f64 {
    let n = mb << 20;
    let start = Barrier::new(cpus.len());
    std::thread::scope(|sc| {
        let hs: Vec<_> = cpus
            .iter()
            .map(|&cpu| {
                let start = &start;
                sc.spawn(move || {
                    pin(cpu);
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
                        std::hint::black_box(unsafe { stream_sum(buf.as_ptr(), n) });
                        best = best.min(t.elapsed().as_secs_f64());
                    }
                    n as f64 / best / 1e9
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    })
}

/// The least wall time of `iters` runs of `f`, after a warm-up.
fn best(iters: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut b = f64::MAX;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

/// The GEMV kernels over a DRAM-resident matrix (1 GB of bf16, interleaved over the nodes
/// as the loader places weights): GB/s of weights streamed, one core and the pool.
fn gemv_rates(pool: &Threads, iters: usize) -> String {
    use common::pool::{Pool, Serial};
    const M: usize = 131072;
    const K: usize = 4096;
    interleave(true);
    let w16: Vec<u16> = (0..M * K).map(|i| ((i * 2654435761) >> 7) as u16 & 0x3fff).collect();
    let w8: Vec<u8> = (0..M * K).map(|i| ((i * 2654435761) >> 11) as u8 & 0x7e).collect();
    let s8: Vec<f32> = vec![1e-3; M.div_ceil(128) * (K / 128)];
    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut out = vec![0f32; M];
    let bf16_1 = (M * K * 2) as f64 / best(iters, || common::gemv::bf16_par::<M, K>(&w16, &x, &mut out, &Serial)) / 1e9;
    let bf16_p = (M * K * 2) as f64 / best(iters, || common::gemv::bf16_par::<M, K>(&w16, &x, &mut out, pool)) / 1e9;
    let fp8_1 = (M * K) as f64 / best(iters, || common::gemv::fp8_par::<M, K>(&w8, &s8, &x, &mut out, &Serial)) / 1e9;
    let fp8_p = (M * K) as f64 / best(iters, || common::gemv::fp8_par::<M, K>(&w8, &s8, &x, &mut out, pool)) / 1e9;
    // INT4 as the experts' records pack it: tiles of four rows, a scale per group
    let (q, sc): (Vec<i8>, Vec<f32>) = ((0..M * K).map(|i| ((i * 2654435761) >> 9) as i8 % 8).collect(), vec![1e-3; M * K / 128]);
    let (pw, ps) = common::q4::pack(M, K, &q, &sc);
    interleave(false);
    let (mut xq, mut xs, mut xg) = (vec![0i8; K], vec![0f32; K / 128], vec![0i32; K / 128]);
    common::q4::QAct { q: &mut xq, scale: &mut xs, gsum: &mut xg }.quantize(&x);
    let act = common::q4::Act { q: &xq, scale: &xs, gsum: &xg };
    let q4_1 = (M * K / 2) as f64 / best(iters, || common::q4::gemv::<K>(M, &pw, &ps, &act, &mut out)) / 1e9;
    let out_p = out.as_mut_ptr() as usize;
    let (pw_r, ps_r) = (&pw, &ps);
    let q4_p = (M * K / 2) as f64
        / best(iters, || {
            pool.run(M / 64, &move |i| {
                let r0 = i * 64;
                let o = unsafe { std::slice::from_raw_parts_mut((out_p as *mut f32).add(r0), 64) };
                common::q4::gemv::<K>(64, &pw_r[r0 / 4 * 2 * K..(r0 + 64) / 4 * 2 * K], &ps_r[r0 / 4 * (K / 128) * 4..(r0 + 64) / 4 * (K / 128) * 4], &act, o);
            })
        })
        / 1e9;
    format!("gemv     : weights streamed from DRAM, one core / the pool — bf16 {bf16_1:.1} / {bf16_p:.0} GB/s, fp8 {fp8_1:.1} / {fp8_p:.0}, int4 {q4_1:.1} / {q4_p:.0}")
}

/// Every probe, printed; `elf` is a code object with the `stream` kernel for the cards' probes.
pub fn run(elf: Option<&[u8]>) {
    let iters = 3;
    let nodes = nodes();
    let all: Vec<usize> = nodes.iter().flatten().copied().collect();
    println!("probe    : {} NUMA nodes, {} physical cores", nodes.len(), all.len());
    let mb = (2048 / all.len().max(1)).max(16);
    for (i, cpus) in nodes.iter().enumerate() {
        println!("dram     : node {i} local, {} threads: {:.0} GB/s", cpus.len(), dram(cpus, mb, iters, false));
    }
    if nodes.len() > 1 {
        println!("dram     : every node local, {} threads: {:.0} GB/s", all.len(), dram(&all, mb, iters, false));
        println!("dram     : interleaved pages, {} threads: {:.0} GB/s (the loader's placement)", all.len(), dram(&all, mb, iters, true));
    }
    let pool = Threads::new(&all);
    println!("{}", gemv_rates(&pool, iters));
    pool.idle();
    let Some(elf) = elf else {
        println!("probe    : no code object: the cards' probes skipped");
        return;
    };
    use glm_5_3_flash::probe::gfx11_consts::{WG, WORDS_PER_WG};
    let words = (1usize << 30) / 4 / WORDS_PER_WG * WORDS_PER_WG;
    let grid = (words / WORDS_PER_WG) as u32;
    for c in 0..kfd::Kfd::cards().len() {
        let k = kfd::Kfd::open(elf, false, c);
        let stream = k.kernel("stream");
        let out = k.vram_visible(grid as usize * WG as usize * 4);
        let host_src: Vec<u8> = (0..words * 4).map(|i| i as u8).collect();
        let gbs = |secs: f64| (words * 4) as f64 / secs / 1e9;
        let vram = k.vram_visible(words * 4);
        let up = gbs(best(iters, || k.upload(&vram, 0, &host_src)));
        let mut back = vec![0u8; words * 4];
        let down = gbs(best(iters, || {
            kfd::read(&vram, 0, &mut back[..]);
            std::hint::black_box(&back);
        }));
        let mut a = kfd::Args::new();
        a.ptr(vram.va).ptr(out.va);
        let vr = gbs(best(iters, || {
            k.launch(&stream, grid, WG, 0, &a);
            k.sync();
        }));
        let host = k.host_coherent(words * 4);
        k.upload(&host, 0, &host_src);
        let mut b = kfd::Args::new();
        b.ptr(host.va).ptr(out.va);
        let hr = gbs(best(iters, || {
            k.launch(&stream, grid, WG, 0, &b);
            k.sync();
        }));
        println!("card {c}   : streams VRAM at {vr:.0} GB/s; reads host memory over the link at {hr:.1} GB/s; the host writes its BAR at {up:.1} GB/s and reads it back at {down:.1} GB/s (1 GB each)");
    }
}
