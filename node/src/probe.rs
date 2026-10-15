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
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
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

/// Anonymous memory of `bytes`, every 2 MiB bound to a node in turn when `nodes` is
/// more than one (`alternate`), else under the calling thread's policy; touched.
fn region(bytes: usize, alternate: usize) -> &'static mut [u8] {
    const PAGE: usize = glm_5_3_flash::construct::PAGE;
    let len = bytes.div_ceil(PAGE) * PAGE;
    let p = unsafe { mmap(std::ptr::null_mut(), len, 3, 0x22, -1, 0) };
    assert!(p as isize != -1, "mmap");
    if alternate > 1 {
        for c in 0..len / PAGE {
            let mask: u64 = 1 << (c % alternate);
            let r = unsafe { syscall(237, p.add(c * PAGE), PAGE as u64, 2i32, &mask as *const u64, 64u64, 0u32) };
            assert_eq!(r, 0, "mbind: {}", std::io::Error::last_os_error());
        }
    }
    let buf = unsafe { std::slice::from_raw_parts_mut(p, len) };
    for b in buf.iter_mut().step_by(4096) {
        *b = 1;
    }
    &mut buf[..bytes]
}
fn as_mut<T>(b: &'static mut [u8]) -> &'static mut [T] {
    unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut T, b.len() / std::mem::size_of::<T>()) }
}

/// The GEMV kernels over a DRAM-resident matrix (1 GB of bf16): GB/s of weights streamed, one
/// core and the pool — the matrices interleaved by 4 KiB page with the items on any core, or by
/// 2 MiB page with the pool placed and every item on its rows' node.
fn gemv_rates(pool: &Threads, iters: usize, nodes: usize, placed: bool) -> String {
    use common::pool::{Pool, Serial};
    const M: usize = 131072;
    const K: usize = glm_5_3_flash::H; // the model's row width: a row is 10 KB, no item stride a power of two
    let alloc = |bytes: usize| {
        interleave(!placed);
        let b = region(bytes, if placed { nodes } else { 1 });
        interleave(false);
        if placed {
            pool.place(b.as_ptr() as usize, b.len(), crate::loader::placement(b));
        }
        b
    };
    let w16: &mut [u16] = as_mut(alloc(M * K * 2));
    for (i, w) in w16.iter_mut().enumerate() {
        *w = ((i * 2654435761) >> 7) as u16 & 0x3fff;
    }
    let w8: &mut [u8] = alloc(M * K);
    for (i, w) in w8.iter_mut().enumerate() {
        *w = ((i * 2654435761) >> 11) as u8 & 0x7e;
    }
    let s8: Vec<f32> = vec![1e-3; M.div_ceil(128) * (K / 128)];
    let x: Vec<f32> = (0..K).map(|i| (i % 13) as f32 * 0.01).collect();
    let mut out = vec![0f32; M];
    let bf16_1 = (M * K * 2) as f64 / best(iters, || common::gemv::bf16_par::<M, K>(&w16, &x, &mut out, &Serial)) / 1e9;
    let bf16_p = (M * K * 2) as f64 / best(iters, || common::gemv::bf16_par::<M, K>(&w16, &x, &mut out, pool)) / 1e9;
    // the same rows in bigger items: what an item's claim costs against what it streams
    let (wp, xp, op) = (w16.as_ptr() as usize, x.as_ptr() as usize, out.as_mut_ptr() as usize);
    let by_rows: Vec<String> = [64usize, 256]
        .iter()
        .map(|&rows| {
            let r = (M * K * 2) as f64 / best(iters, || pool.run_at(M / rows, &move |i| (wp + i * rows * K * 2) as *const u8, &move |i| unsafe { common::gemv::avx2::bf16_rows::<K>(wp as *const u16, xp as *const f32, op as *mut f32, i * rows, (i + 1) * rows) })) / 1e9;
            format!("{rows} rows {r:.0}")
        })
        .collect();
    // the same kernel with no pool: a thread per core over its own rows, pages dealt to the cores of their node
    let static_rate = {
        let cores = physical_cores();
        let per_node: Vec<Vec<usize>> = (0..nodes).map(|d| cores.iter().copied().filter(|&c| glm_5_3_flash::construct::node_of_core(c) == d).collect()).collect();
        let page_rows = glm_5_3_flash::construct::PAGE / (K * 2);
        let pages = (M * K * 2).div_ceil(glm_5_3_flash::construct::PAGE);
        let start = Barrier::new(cores.len());
        let (wp, xp, op) = (w16.as_ptr() as usize, x.as_ptr() as usize, out.as_mut_ptr() as usize);
        let secs = std::thread::scope(|sc| {
            let hs: Vec<_> = per_node
                .iter()
                .enumerate()
                .flat_map(|(d, cs)| cs.iter().enumerate().map(move |(t, &core)| (d, t, cs.len(), core)))
                .map(|(d, t, nt, core)| {
                    let start = &start;
                    sc.spawn(move || {
                        pin(core);
                        // this thread's pages: every nt-th of its node's (page p is on node p % nodes when placed; dealt the same way when not)
                        let mine: Vec<usize> = (0..pages).filter(|&p| p % nodes == d).enumerate().filter(|(i, _)| i % nt == t).map(|(_, p)| p).collect();
                        let mut b = f64::MAX;
                        for _ in 0..iters + 1 {
                            start.wait();
                            let t0 = std::time::Instant::now();
                            for &p in &mine {
                                let (m0, m1) = (p * page_rows, ((p + 1) * page_rows).min(M));
                                unsafe { common::gemv::avx2::bf16_rows::<K>(wp as *const u16, xp as *const f32, op as *mut f32, m0, m1) };
                            }
                            b = b.min(t0.elapsed().as_secs_f64());
                        }
                        b
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).fold(0f64, f64::max)
        });
        (M * K * 2) as f64 / secs / 1e9
    };
    let fp8_1 = (M * K) as f64 / best(iters, || common::gemv::fp8_par::<M, K>(&w8, &s8, &x, &mut out, &Serial)) / 1e9;
    let fp8_p = (M * K) as f64 / best(iters, || common::gemv::fp8_par::<M, K>(&w8, &s8, &x, &mut out, pool)) / 1e9;
    // INT4 as the experts' records pack it: tiles of four rows, a scale per group
    let (q, sc): (Vec<i8>, Vec<f32>) = ((0..M * K).map(|i| ((i * 2654435761) >> 9) as i8 % 8).collect(), vec![1e-3; M * K / 128]);
    let (pw_v, ps) = common::q4::pack(M, K, &q, &sc);
    let pw: &mut [u8] = alloc(pw_v.len());
    pw.copy_from_slice(&pw_v);
    drop(pw_v);
    let (mut xq, mut xs, mut xg) = (vec![0i8; K], vec![0f32; K / 128], vec![0i32; K / 128]);
    common::q4::QAct { q: &mut xq, scale: &mut xs, gsum: &mut xg }.quantize(&x);
    let act = common::q4::Act { q: &xq, scale: &xs, gsum: &xg };
    let q4_1 = (M * K / 2) as f64 / best(iters, || common::q4::gemv::<K>(M, &pw, &ps, &act, &mut out)) / 1e9;
    let out_p = out.as_mut_ptr() as usize;
    let (pw_r, ps_r) = (&*pw, &ps);
    let q4_p = (M * K / 2) as f64
        / best(iters, || {
            pool.run_at(M / 64, &move |i| pw_r[i * 64 / 4 * 2 * K..].as_ptr(), &move |i| {
                let r0 = i * 64;
                let o = unsafe { std::slice::from_raw_parts_mut((out_p as *mut f32).add(r0), 64) };
                common::q4::gemv::<K>(64, &pw_r[r0 / 4 * 2 * K..(r0 + 64) / 4 * 2 * K], &ps_r[r0 / 4 * (K / 128) * 4..(r0 + 64) / 4 * (K / 128) * 4], &act, o);
            })
        })
        / 1e9;
    format!("gemv     : {}, one core / the pool — bf16 {bf16_1:.1} / {bf16_p:.0} GB/s (items of {}; static rows per thread {static_rate:.0}), fp8 {fp8_1:.1} / {fp8_p:.0}, int4 {q4_1:.1} / {q4_p:.0}", if placed { "2 MiB pages alternating nodes, items on their rows' node" } else { "4 KiB pages interleaved, items on any core" }, by_rows.join(", "))
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
    println!("{}", gemv_rates(&pool, iters, nodes.len(), false));
    if nodes.len() > 1 {
        println!("{}", gemv_rates(&pool, iters, nodes.len(), true));
    }
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
        let mut back = vec![0u8; 16 << 20];
        let down = (back.len() as f64 / 1e9)
            / best(iters, || {
                kfd::read(&vram, 0, &mut back[..]);
                std::hint::black_box(&back);
            });
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
        println!("card {c}   : streams VRAM at {vr:.0} GB/s; reads host memory over the link at {hr:.1} GB/s; the host writes its BAR at {up:.1} GB/s (1 GB) and reads it back at {down:.2} GB/s (16 MB)");
        // the KDA layer's launches at its shapes, with the pool idle and then spinning as it is
        // between a step's jobs: the projections' rate as one launch and as three, the output
        // projection into VRAM and into host memory, a launch's round trip, the block output's way back
        use common::pool::Pool;
        use glm_5_3_flash::kda::gfx11_consts as g;
        use glm_5_3_flash::{H, KDA_QKV};
        let (rows_h, rows_qkv) = (k.kernel("rows_h"), k.kernel("rows_qkv"));
        let w16: Vec<u16> = (0..3 * KDA_QKV * H).map(|i| ((i * 2654435761) >> 7) as u16 & 0x3fff).collect();
        let w = k.vram_visible(w16.len() * 2);
        k.upload(&w, 0, kfd::raw(&w16));
        let xf: Vec<f32> = (0..KDA_QKV).map(|i| (i % 13) as f32 * 0.01).collect();
        let x = k.vram_visible(H * 4);
        k.upload(&x, 0, kfd::raw(&xf[..H]));
        let xq = k.vram_visible(KDA_QKV * 4);
        k.upload(&xq, 0, kfd::raw(&xf));
        let ln = k.vram_visible(H * 2);
        k.upload(&ln, 0, kfd::raw(&vec![0x3f80u16; H]));
        let proj = k.vram_visible(3 * KDA_QKV * 4);
        let y = k.vram_visible(H * 4);
        let y_host = k.host(H * 4);
        let rows = |kernel: &kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, kw: usize| {
            let mut a = kfd::Args::new();
            a.ptr(x);
            if let Some(ln) = ln {
                a.ptr(ln);
            }
            a.ptr(w).ptr(out).u32(m as u32).u32(1);
            let lds = if ln.is_some() { g::normed_rows_lds(kw, 1) } else { g::rows_lds(kw, 1) } as u32;
            k.launch(kernel, g::rows_grid(m), g::WG, lds, &a);
        };
        let gbs = |bytes: usize, secs: f64| bytes as f64 / secs / 1e9;
        let mut back = vec![0f32; H];
        let mut measure = |state: &str| {
            let one = gbs(3 * KDA_QKV * H * 2, best(iters, || {
                rows(&rows_h, x.va, Some(ln.va), w.va, proj.va, 3 * KDA_QKV, H);
                k.sync();
            }));
            let three = gbs(3 * KDA_QKV * H * 2, best(iters, || {
                for p in 0..3 {
                    rows(&rows_h, x.va, Some(ln.va), w.at(p * KDA_QKV * H * 2), proj.at(p * KDA_QKV * 4), KDA_QKV, H);
                }
                k.sync();
            }));
            let o_vram = gbs(H * KDA_QKV * 2, best(iters, || {
                rows(&rows_qkv, xq.va, None, w.va, y.va, H, KDA_QKV);
                k.sync();
            }));
            let o_host = gbs(H * KDA_QKV * 2, best(iters, || {
                rows(&rows_qkv, xq.va, None, w.va, y_host.va, H, KDA_QKV);
                k.sync();
            }));
            let trip = best(iters, || {
                rows(&rows_h, x.va, Some(ln.va), w.va, proj.va, 8, H);
                k.sync();
            }) * 1e6;
            let bar = best(iters, || {
                kfd::read(&y, 0, &mut back[..]);
                std::hint::black_box(&back);
            }) * 1e6;
            let host = best(iters, || {
                kfd::read(&y_host, 0, &mut back[..]);
                std::hint::black_box(&back);
            }) * 1e6;
            println!("card {c}   : pool {state}: the KDA projections at {one:.0} GB/s as one launch, {three:.0} as three; the output projection at {o_vram:.0} GB/s into VRAM, {o_host:.0} into host memory; a launch's round trip {trip:.0} µs; the block output back in {bar:.0} µs through the BAR, {host:.1} µs from host memory");
        };
        measure("idle");
        pool.run(2, &|_| {}); // busy: the workers spin, as between a step's jobs
        measure("spinning");
        pool.idle();
        // the FFN's FP8 projections at their shapes: a gate|up launch over the normed input, the down over its SwiGLU
        use glm_5_3_flash::ffn::gfx11_consts::{DENSE_GU_ROWS, SHARED_GU_ROWS};
        use glm_5_3_flash::{D_FF, INTERMEDIATE_SIZE};
        let (gu_k, down_s, down_d) = (k.kernel("ffn_gu"), k.kernel("ffn_down_shared"), k.kernel("ffn_down_dense"));
        let w8: Vec<u8> = (0..DENSE_GU_ROWS * H).map(|i| ((i * 2654435761) >> 11) as u8 & 0x7e).collect();
        let w8m = k.vram_visible(w8.len());
        k.upload(&w8m, 0, &w8);
        let sc = k.vram_visible(DENSE_GU_ROWS / 128 * (INTERMEDIATE_SIZE / 128) * 4);
        k.upload(&sc, 0, kfd::raw(&vec![1e-3f32; sc.size / 4]));
        let gu_out = k.vram_visible(DENSE_GU_ROWS * 4);
        let fp8 = |kernel: &kfd::Kernel, x: usize, ln: Option<usize>, out: usize, m: usize, kw: usize| {
            let mut a = kfd::Args::new();
            a.ptr(x);
            if let Some(ln) = ln {
                a.ptr(ln);
            }
            a.ptr(w8m.va).ptr(sc.va).ptr(out).u32(m as u32).u32(1);
            k.launch(kernel, g::rows_grid(m), g::WG, g::fp8_rows_lds(kw, 1) as u32, &a);
        };
        let shared_gu = gbs(SHARED_GU_ROWS * H, best(iters, || {
            fp8(&gu_k, x.va, Some(ln.va), gu_out.va, SHARED_GU_ROWS, H);
            k.sync();
        }));
        let shared_down = gbs(H * D_FF, best(iters, || {
            fp8(&down_s, gu_out.va, None, y_host.va, H, D_FF);
            k.sync();
        }));
        let dense_gu = gbs(DENSE_GU_ROWS * H, best(iters, || {
            fp8(&gu_k, x.va, Some(ln.va), gu_out.va, DENSE_GU_ROWS, H);
            k.sync();
        }));
        let dense_down = gbs(H * INTERMEDIATE_SIZE, best(iters, || {
            fp8(&down_d, gu_out.va, None, y_host.va, H, INTERMEDIATE_SIZE);
            k.sync();
        }));
        // the head's shape: rows_h over the vocabulary, 1.6 GB — nothing of it fits a cache
        let head_w = k.vram_visible(glm_5_3_flash::VOCAB_SIZE * H * 2);
        for off in (0..head_w.size).step_by(w16.len() * 2) {
            k.upload(&head_w, off, &kfd::raw(&w16)[..(head_w.size - off).min(w16.len() * 2)]);
        }
        let head_out = k.host(glm_5_3_flash::VOCAB_SIZE * 4);
        let head = gbs(head_w.size, best(iters, || {
            rows(&rows_h, x.va, Some(ln.va), head_w.va, head_out.va, glm_5_3_flash::VOCAB_SIZE, H);
            k.sync();
        }));
        let mclk = std::fs::read_to_string(format!("/sys/class/drm/card{}/device/pp_dpm_mclk", c)).map(|s| s.lines().filter(|l| l.ends_with('*')).map(|l| l.trim().to_string()).collect::<Vec<_>>().join(" ")).unwrap_or_default();
        println!("card {c}   : the head's shape ({} MB of bf16 rows, normed) at {head:.0} GB/s into host memory; memory clock {mclk}", head_w.size / 1_000_000);
        println!("card {c}   : FP8 rows — the shared expert's gate|up ({} MB) at {shared_gu:.0} GB/s, its down at {shared_down:.0}; a dense layer's gate|up ({} MB) at {dense_gu:.0}, its down at {dense_down:.0} (one launch + sync each)", SHARED_GU_ROWS * H / 1_000_000, DENSE_GU_ROWS * H / 1_000_000);
    }
}
