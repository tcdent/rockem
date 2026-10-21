//! bench-power — what a kind of work costs this box in watts and in clock: every listed core runs
//! one kind of work at once (pinned, its own first-touched buffers), and over a fixed window the
//! bench reads each core's APERF/MPERF (the clock it actually held, `/dev/cpu/N/msr`), each
//! socket's power from the firmware (the `amd_hsmp_hwmon` hwmon's `power1_input`, sampled), and
//! each core's rate. The question it answers: how many watts a multiply, a load from L1/L2/L3 or a
//! byte from DRAM costs here, and what the clock does once a socket reaches its power cap (metatron:
//! 280 W a socket, 2026-10-20). A row a run:
//!
//!   bench-power --mode M --cpus 1-23,25-47 [--secs 5] [--n 32]
//!
//! modes: idle; chain (dependent scalar adds); vmadd (`vpmaddubsw` on registers only); l1 / l2 / l3
//! (AVX2 loads over 16 KB / 256 KB / 2 MB a core); dram (AVX2 loads over 256 MB a core, streamed);
//! gemm (the expert `q4::gemm`, 32 rows × 4096, `--n` tokens, cache-resident); gemm-dram (the same
//! over 16,384 rows, 33.6 MB a core, from DRAM).

use common::q4::{gemm, pack, Act, QAct, GROUP};
use harness::arg;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}

/// `1-23,25-47` → the cores.
fn cpus(spec: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for part in spec.split(',') {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(part.parse().unwrap()),
        }
    }
    v
}

/// Every hwmon of `name`, in order: their directories.
fn hwmons(name: &str) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir("/sys/class/hwmon").map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).collect()).unwrap_or_default();
    v.sort();
    v.into_iter().filter(|p| std::fs::read_to_string(p.join("name")).map(|s| s.trim() == name).unwrap_or(false)).collect()
}

fn read_num(p: &std::path::Path) -> f64 {
    std::fs::read_to_string(p).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(f64::NAN)
}

#[target_feature(enable = "avx2")]
unsafe fn loads(p: *const u8, len: usize) -> u64 {
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

/// One core's work: `unit()` does a fixed amount and returns how much (ops, bytes or MACs).
struct Work {
    unit: Box<dyn FnMut() -> f64>,
}

fn work(mode: &str, t: usize, n: usize) -> Work {
    let buf = |bytes: usize| {
        let mut b = vec![0u8; bytes];
        for (i, x) in b.iter_mut().enumerate().step_by(64) {
            *x = (i + t) as u8;
        }
        b
    };
    match mode {
        "idle" => Work { unit: Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            0.0
        }) },
        "chain" => Work { unit: Box::new(|| {
            let mut x = 0u64;
            unsafe { core::arch::asm!("2:", ".rept 16", "add {x}, 1", ".endr", "dec {c}", "jnz 2b", x = inout(reg) x, c = inout(reg) 1000u64 => _) };
            std::hint::black_box(x);
            16000.0
        }) },
        "vmadd" => Work { unit: Box::new(|| {
            unsafe {
                core::arch::asm!("vpxor ymm0, ymm0, ymm0", "vpxor ymm1, ymm1, ymm1", "vpxor ymm2, ymm2, ymm2", "vpxor ymm3, ymm3, ymm3", "2:",
                    "vpmaddubsw ymm4, ymm0, ymm1", "vpaddw ymm2, ymm2, ymm4", "vpmaddubsw ymm5, ymm1, ymm0", "vpaddw ymm3, ymm3, ymm5",
                    "vpmaddubsw ymm6, ymm0, ymm1", "vpaddw ymm2, ymm2, ymm6", "vpmaddubsw ymm7, ymm1, ymm0", "vpaddw ymm3, ymm3, ymm7",
                    "dec {c}", "jnz 2b", c = inout(reg) 2000u64 => _, out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _, out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _)
            };
            // 8000 vpmaddubsw of 32 byte-MACs each
            8000.0 * 32.0
        }) },
        "l1" | "l2" | "l3" | "dram" => {
            let bytes = match mode { "l1" => 16 << 10, "l2" => 256 << 10, "l3" => 2 << 20, _ => 256 << 20 };
            let b = buf(bytes);
            Work { unit: Box::new(move || {
                std::hint::black_box(unsafe { loads(b.as_ptr(), bytes) });
                bytes as f64
            }) }
        }
        "gemm" | "gemm-dram" => {
            const K: usize = 4096;
            let rows = if mode == "gemm" { 32 } else { 16384 };
            let mut rng = (t as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut r = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let w: Vec<i8> = (0..rows * K).map(|_| (r() % 16) as i8 - 8).collect();
            let sc: Vec<f32> = (0..rows * (K / GROUP)).map(|_| 0.01).collect();
            let (pw, ps) = pack(rows, K, &w, &sc);
            let (mut q, mut s, mut g) = (vec![0i8; n * K], vec![0f32; n * K / GROUP], vec![0i32; n * K / GROUP]);
            for i in 0..n {
                let x: Vec<f32> = (0..K).map(|_| (r() % 2000) as f32 / 1000.0 - 1.0).collect();
                QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &mut g[i * (K / GROUP)..(i + 1) * (K / GROUP)] }.quantize(&x);
            }
            let mut out = vec![0f32; n * rows];
            Work { unit: Box::new(move || {
                let op = out.as_mut_ptr() as usize;
                let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * (K / GROUP)..(i + 1) * (K / GROUP)], gsum: &g[i * (K / GROUP)..(i + 1) * (K / GROUP)] };
                gemm::<K>(rows, &pw, &ps, n, act, |i| unsafe { (op as *mut f32).add(i * rows) });
                (rows * K * n) as f64
            }) }
        }
        _ => panic!("mode {mode}: idle, chain, vmadd, l1, l2, l3, dram, gemm, gemm-dram"),
    }
}

fn main() {
    let mode = arg("--mode").map(|p| p.to_str().unwrap().to_string()).unwrap_or("idle".into());
    let list = cpus(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1"));
    let secs = num("--secs", 5) as f64;
    let n = num("--n", 32);
    let power = hwmons("amd_hsmp_hwmon");
    let temps = hwmons("k10temp");
    let ready = Barrier::new(list.len() + 1);
    let (go, stop) = (Barrier::new(list.len() + 1), AtomicBool::new(false));
    let measuring = AtomicBool::new(false);
    let rows = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for (t, &cpu) in list.iter().enumerate() {
            let (ready, go, stop, measuring, rows, mode) = (&ready, &go, &stop, &measuring, &rows, mode.as_str());
            sc.spawn(move || {
                harness::bench::pin(cpu);
                let mut w = work(mode, t, n);
                let msr = std::fs::File::open(format!("/dev/cpu/{cpu}/msr")).ok();
                let rd = |a: u64| -> u64 {
                    let mut b = [0u8; 8];
                    if let Some(f) = msr.as_ref() {
                        let _ = f.read_exact_at(&mut b, a);
                    }
                    u64::from_le_bytes(b)
                };
                ready.wait();
                go.wait();
                // warm up until the window opens (the main thread's second of settling), then count over it
                while !measuring.load(Ordering::Acquire) {
                    (w.unit)();
                }
                let (a0, m0, t0) = (rd(0xE8), rd(0xE7), std::time::Instant::now());
                let mut done = 0f64;
                while !stop.load(Ordering::Acquire) {
                    done += (w.unit)();
                }
                let dt = t0.elapsed().as_secs_f64();
                let (a1, m1) = (rd(0xE8), rd(0xE7));
                rows.lock().unwrap().push((cpu, done / dt, (a1 - a0) as f64 / dt / 1e9, (m1 - m0) as f64 / dt / 1e9));
            });
        }
        ready.wait();
        go.wait();
        std::thread::sleep(std::time::Duration::from_secs(1));
        measuring.store(true, Ordering::Release);
        // the sockets' power over the window, sampled every 100 ms
        let t0 = std::time::Instant::now();
        let mut samples = vec![Vec::new(); power.len()];
        while t0.elapsed().as_secs_f64() < secs {
            for (i, p) in power.iter().enumerate() {
                samples[i].push(read_num(&p.join("power1_input")) / 1e6);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        stop.store(true, Ordering::Release);
        let watts: Vec<String> = samples.iter().map(|s| format!("{:.0}", s.iter().sum::<f64>() / s.len().max(1) as f64)).collect();
        let deg: Vec<String> = temps.iter().map(|p| format!("{:.0}", read_num(&p.join("temp1_input")) / 1e3)).collect();
        println!("watts {} temp {}", watts.join("/"), deg.join("/"));
    });
    let mut r = rows.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let (rl, rm, rh) = med(r.iter().map(|x| x.1).collect());
    let (gl, gm, gh) = med(r.iter().map(|x| x.2).collect());
    let unit = match mode.as_str() { "chain" => "Gop/s", "vmadd" | "gemm" | "gemm-dram" => "GMAC/s", "idle" => "-", _ => "GB/s" };
    println!("mode {mode} cores {} n {n}: rate {:.2}/{:.2}/{:.2} {unit} a core (min/med/max), clock {:.3}/{:.3}/{:.3} GHz, ref {:.3} GHz", r.len(), rl / 1e9, rm / 1e9, rh / 1e9, gl, gm, gh, r[0].3);
}
