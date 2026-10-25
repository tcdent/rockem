//! bench-trellis — the decode-and-dot throughput of expert formats on this box's cores: the production INT4 kernel
//! (`common::q4::avx2`), a trellis whose codebook is a sum of four nibble tables over the overlapping window
//! (`lut4`: weight w's value T0[c_w] + T1[c_w+1] + T2[c_w+2] + T3[c_w+3], localai/expert-quantization-formats.md
//! 2026-10-25), and a V1s-shaped trellis (`hash16`: the same 16-bit window hashed in 16-bit lanes — x ^= x >> 7, two
//! chained multiplies, the bytes summed, a cubic stretch — then an int16 dot). Random codes, tables and activations:
//! throughput only, not quality. Every kernel is 4096-wide rows at a 4-bit stride (the trellis rows carry a chunk of
//! pad: the window reads the next chunk's first codes).
//!
//! Per kernel, token count (`--n`), regime and thread set: the rows cache-resident (32 rows, re-read) or streamed from
//! DRAM (`--mb` MB of rows a thread, read through `--passes` times), on one core (`--one`, default 1) and on every
//! core of `--cpus` at once (default the node's pool, 1-23,25-47,49-71,73-95). Reports GMAC/s a core, G weights/s in
//! all (decoded a second), and the code bytes a second that implies.
//!
//!   bench-trellis [--n 1,3] [--mb 96] [--passes 3] [--cpus 1-23,25-47,49-71,73-95] [--one 1]
use harness::arg;
use std::arch::x86_64::*;
use std::sync::{Arc, Barrier};

const K: usize = 4096;
const CHUNKS: usize = K / 32;
/// A trellis row's bytes: K codes at four bits, then one chunk of pad (the last chunk's window reads into it).
const ROW: usize = K / 2 + 16;

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
fn fill(seed: u64, b: &mut [u8]) {
    let mut x = seed | 1;
    for c in b.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let bytes = x.to_le_bytes();
        c.copy_from_slice(&bytes[..c.len()]);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s)
}
/// A chunk's 32 codes as bytes: the low nibbles of its 16 bytes (positions 0–15), then the high (16–31).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack(p: *const u8) -> __m256i {
    let v = _mm_loadu_si128(p as *const __m128i);
    let m = _mm_set1_epi8(0x0F);
    _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m), _mm_and_si128(v, m))
}
/// The chunk's codes and the next three of each weight's window: positions shifted by 1, 2, 3 across the chunk's end.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn windows(cur: __m256i, nxt: __m256i) -> (__m256i, __m256i, __m256i) {
    let nx = _mm256_permute2x128_si256(cur, nxt, 0x21);
    (_mm256_alignr_epi8(nx, cur, 1), _mm256_alignr_epi8(nx, cur, 2), _mm256_alignr_epi8(nx, cur, 3))
}

/// The additive-table trellis: per weight T0[c_w] + T1[c_w+1] + T2[c_w+2] + T3[c_w+3] (each 0–31, so a u8 0–124, the
/// signed value less 62), its int8 dot against each token's activations.
#[target_feature(enable = "avx2")]
unsafe fn lut4<const N: usize>(m: usize, w: *const u8, t: &[__m256i; 4], acts: &[*const i8; N], asum: &[i32; N], out: *mut f32) {
    let ones = _mm256_set1_epi16(1);
    for r in 0..m {
        let row = w.add(r * ROW);
        let mut acc = [_mm256_setzero_si256(); N];
        let mut cur = unpack(row);
        for c in 0..CHUNKS {
            let nxt = unpack(row.add((c + 1) * 16));
            let (s1, s2, s3) = windows(cur, nxt);
            let wv = _mm256_add_epi8(
                _mm256_add_epi8(_mm256_shuffle_epi8(t[0], cur), _mm256_shuffle_epi8(t[1], s1)),
                _mm256_add_epi8(_mm256_shuffle_epi8(t[2], s2), _mm256_shuffle_epi8(t[3], s3)),
            );
            let mut i = 0;
            while i < N {
                let a = _mm256_loadu_si256(acts[i].add(c * 32) as *const __m256i);
                acc[i] = _mm256_add_epi32(acc[i], _mm256_madd_epi16(_mm256_maddubs_epi16(wv, a), ones));
                i += 1;
            }
            cur = nxt;
        }
        let mut i = 0;
        while i < N {
            *out.add(i * m + r) = (hsum(acc[i]) - 62 * asum[i]) as f32 * 1e-3;
            i += 1;
        }
    }
}

/// The V1s-shaped trellis: the 16-bit window (c_w | c_w+1 << 4 | c_w+2 << 8 | c_w+3 << 12) hashed in 16-bit lanes —
/// x ^= x >> 7, two chained `vpmullw`, the two bytes summed by `vpmaddubsw`, centred, a cubic stretch by three
/// `vpmulhrsw` — then an int16 dot (`vpmaddwd`) against each token's 16-bit activations.
#[target_feature(enable = "avx2")]
unsafe fn hash16<const N: usize>(m: usize, w: *const u8, acts: &[*const i16; N], out: *mut f32) {
    let (c1, c2) = (_mm256_set1_epi16(0x6D2Bu16 as i16), _mm256_set1_epi16(0x3A8Fu16 as i16));
    let ones8 = _mm256_set1_epi8(1);
    let mid = _mm256_set1_epi16(255);
    let a3 = _mm256_set1_epi16(9000);
    for r in 0..m {
        let row = w.add(r * ROW);
        let mut acc = [_mm256_setzero_si256(); N];
        let mut cur = unpack(row);
        for c in 0..CHUNKS {
            let nxt = unpack(row.add((c + 1) * 16));
            let (s1, s2, s3) = windows(cur, nxt);
            let b0 = _mm256_or_si256(cur, _mm256_slli_epi16(s1, 4));
            let b1 = _mm256_or_si256(s2, _mm256_slli_epi16(s3, 4));
            for (h, x) in [_mm256_unpacklo_epi8(b0, b1), _mm256_unpackhi_epi8(b0, b1)].into_iter().enumerate() {
                let x = _mm256_xor_si256(x, _mm256_srli_epi16(x, 7));
                let x = _mm256_mullo_epi16(_mm256_mullo_epi16(x, c1), c2);
                let v = _mm256_sub_epi16(_mm256_maddubs_epi16(x, ones8), mid);
                let v2 = _mm256_mulhrs_epi16(v, v);
                let v3 = _mm256_mulhrs_epi16(v2, v);
                let v = _mm256_add_epi16(v, _mm256_mulhrs_epi16(v3, a3));
                let mut i = 0;
                while i < N {
                    let a = _mm256_loadu_si256(acts[i].add(c * 32 + h * 16) as *const __m256i);
                    acc[i] = _mm256_add_epi32(acc[i], _mm256_madd_epi16(v, a));
                    i += 1;
                }
            }
            cur = nxt;
        }
        let mut i = 0;
        while i < N {
            *out.add(i * m + r) = hsum(acc[i]) as f32 * 1e-6;
            i += 1;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    Q4,
    Lut4,
    Hash16,
}
impl Kernel {
    fn name(self) -> &'static str {
        match self {
            Kernel::Q4 => "q4 (production)",
            Kernel::Lut4 => "lut4 trellis",
            Kernel::Hash16 => "hash16 trellis",
        }
    }
}

/// One thread's run of `kernel` at `n` tokens over `rows` rows, `passes` times; seconds.
fn run(kernel: Kernel, n: usize, rows: usize, passes: usize, seed: u64, start: &Barrier) -> f64 {
    let mut rng = common::testing::Rng::new(seed);
    // the activations (n tokens): i8 for q4 and lut4, i16 for hash16
    let acts8: Vec<Vec<i8>> = (0..n).map(|_| (0..K + 32).map(|_| (rng.i4() * 9) as i8).collect()).collect();
    let acts16: Vec<Vec<i16>> = (0..n).map(|_| (0..K + 32).map(|_| (rng.i4() as i16) * 200).collect()).collect();
    let asum: Vec<i32> = acts8.iter().map(|a| a[..K].iter().map(|&x| x as i32).sum()).collect();
    let mut out = vec![0f32; n * rows];
    let dt = match kernel {
        Kernel::Q4 => {
            use common::q4::{pack, Act, QAct, GROUP};
            // q4's packed form: random 4-bit weights, one scale a group
            let w: Vec<i8> = (0..32 * K).map(|_| rng.i4()).collect();
            let sc: Vec<f32> = vec![0.01; 32 * K / GROUP];
            let (pw1, ps1) = pack(32, K, &w, &sc);
            let reps = rows / 32;
            let pw: Vec<u8> = (0..reps).flat_map(|i| pw1.iter().map(move |&b| b ^ (i as u8))).collect();
            let ps: Vec<f32> = (0..reps).flat_map(|_| ps1.iter().copied()).collect();
            let (mut q, mut s, mut g) = (vec![0i8; n * K], vec![0f32; n * K / GROUP], vec![0i32; n * K / GROUP]);
            for i in 0..n {
                let x: Vec<f32> = (0..K).map(|_| rng.unit()).collect();
                QAct { q: &mut q[i * K..(i + 1) * K], scale: &mut s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &mut g[i * K / GROUP..(i + 1) * K / GROUP] }.quantize(&x);
            }
            let act = |i: usize| Act { q: &q[i * K..(i + 1) * K], scale: &s[i * K / GROUP..(i + 1) * K / GROUP], gsum: &g[i * K / GROUP..(i + 1) * K / GROUP] };
            let op = out.as_mut_ptr() as usize;
            start.wait();
            let t0 = std::time::Instant::now();
            for _ in 0..passes {
                if n == 1 {
                    unsafe { common::q4::avx2::gemv::<K>(rows, pw.as_ptr(), ps.as_ptr(), &act(0), op as *mut f32) };
                } else {
                    unsafe { common::q4::avx2::gemm::<K>(rows, pw.as_ptr(), ps.as_ptr(), n, act, |i| (op as *mut f32).add(i * rows)) };
                }
            }
            t0.elapsed().as_secs_f64()
        }
        Kernel::Lut4 | Kernel::Hash16 => {
            let mut w = vec![0u8; rows * ROW];
            fill(seed, &mut w);
            let mut tb = [[0u8; 32]; 4];
            for t in tb.iter_mut() {
                for (j, x) in t.iter_mut().enumerate() {
                    *x = ((j * 7 + seed as usize) % 32) as u8;
                }
                let (a, b) = t.split_at_mut(16);
                b.copy_from_slice(a);
            }
            let tables: [__m256i; 4] = unsafe { std::array::from_fn(|i| _mm256_loadu_si256(tb[i].as_ptr() as *const __m256i)) };
            let p8: Vec<*const i8> = acts8.iter().map(|a| a.as_ptr()).collect();
            let p16: Vec<*const i16> = acts16.iter().map(|a| a.as_ptr()).collect();
            start.wait();
            let t0 = std::time::Instant::now();
            for _ in 0..passes {
                unsafe {
                    match (kernel, n) {
                        (Kernel::Lut4, 1) => lut4::<1>(rows, w.as_ptr(), &tables, &[p8[0]], &[asum[0]], out.as_mut_ptr()),
                        (Kernel::Lut4, 2) => lut4::<2>(rows, w.as_ptr(), &tables, &[p8[0], p8[1]], &[asum[0], asum[1]], out.as_mut_ptr()),
                        (Kernel::Lut4, 3) => lut4::<3>(rows, w.as_ptr(), &tables, &[p8[0], p8[1], p8[2]], &[asum[0], asum[1], asum[2]], out.as_mut_ptr()),
                        (Kernel::Lut4, 4) => lut4::<4>(rows, w.as_ptr(), &tables, &[p8[0], p8[1], p8[2], p8[3]], &[asum[0], asum[1], asum[2], asum[3]], out.as_mut_ptr()),
                        (Kernel::Hash16, 1) => hash16::<1>(rows, w.as_ptr(), &[p16[0]], out.as_mut_ptr()),
                        (Kernel::Hash16, 2) => hash16::<2>(rows, w.as_ptr(), &[p16[0], p16[1]], out.as_mut_ptr()),
                        (Kernel::Hash16, 3) => hash16::<3>(rows, w.as_ptr(), &[p16[0], p16[1], p16[2]], out.as_mut_ptr()),
                        (Kernel::Hash16, 4) => hash16::<4>(rows, w.as_ptr(), &[p16[0], p16[1], p16[2], p16[3]], out.as_mut_ptr()),
                        _ => panic!("n {n}: 1–4 tokens"),
                    }
                }
            }
            t0.elapsed().as_secs_f64()
        }
    };
    std::hint::black_box(&out);
    dt
}

fn main() {
    let ns = arg("--n").map_or(vec![1, 3], |v| v.to_str().unwrap().split(',').map(|s| s.parse().unwrap()).collect());
    let mb: usize = arg("--mb").map_or(96, |v| v.to_str().unwrap().parse().unwrap());
    let passes: usize = arg("--passes").map_or(3, |v| v.to_str().unwrap().parse().unwrap());
    let pool = list(arg("--cpus").map(|p| p.to_str().unwrap().to_string()).as_deref().unwrap_or("1-23,25-47,49-71,73-95"));
    let one: usize = arg("--one").map_or(1, |v| v.to_str().unwrap().parse().unwrap());
    // rows: 32 (64 KB of codes, cache-resident, re-read until ~0.2 s) or `mb` MB a thread from DRAM
    let big = (mb << 20) / (K / 2) / 32 * 32;
    println!("bench-trellis: {K}-wide rows at 4 bits a weight; cache-resident 32 rows, DRAM {big} rows ({mb} MB) a thread x {passes} passes; pool {} cores", pool.len());
    println!("{:<16} {:>2} {:<6} {:>7} {:>13} {:>14} {:>13}", "kernel", "n", "rows", "threads", "GMAC/s a core", "G weights/s", "code GB/s");
    for kernel in [Kernel::Q4, Kernel::Lut4, Kernel::Hash16] {
        for &n in &ns {
            for (regime, rows, reps) in [("cache", 32usize, 0usize), ("dram", big, passes)] {
                for cpus in [vec![one], pool.clone()] {
                    // cache: enough passes for ~0.2 s at ~50 GMAC/s; dram: the passes asked
                    let passes = if reps == 0 { (0.2 * 50e9 / (32.0 * K as f64 * n as f64)) as usize } else { reps };
                    let start = Arc::new(Barrier::new(cpus.len()));
                    let handles: Vec<_> = cpus
                        .iter()
                        .enumerate()
                        .map(|(i, &cpu)| {
                            let start = start.clone();
                            std::thread::spawn(move || {
                                harness::bench::pin(cpu);
                                run(kernel, n, rows, passes, 0x9E37_79B9 + i as u64 * 7919, &start)
                            })
                        })
                        .collect();
                    let times: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
                    let tmax = times.iter().cloned().fold(0.0, f64::max);
                    let weights = (rows * K * passes) as f64;
                    let per_core = times.iter().map(|&t| weights * n as f64 / t).sum::<f64>() / times.len() as f64;
                    let all = weights * cpus.len() as f64 / tmax;
                    println!("{:<16} {n:>2} {regime:<6} {:>7} {:>13.1} {:>14.1} {:>13.1}", kernel.name(), cpus.len(), per_core / 1e9, all / 1e9, all / 2.0 / 1e9);
                }
            }
        }
    }
}
