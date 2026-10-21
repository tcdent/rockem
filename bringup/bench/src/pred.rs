//! bench-pred — the learned expert predictor (`glm_5_3_flash::predict`) replayed causally over a decode dump: per step,
//! every layer pair predicts the next layer's experts for the step's tokens, the step's union is ranked (Σ softplus),
//! its recall of the next layer's picks counted at caps 8 / 13 / 16, then the pair learns from those picks. The parity
//! target is bringup/expert-cache/phase3/sim.py's run of the same configuration (`--terms cal,bias,co --lrb 1e-2 --lrc 3e-3`,
//! the `b_co` run, whose co-routing update is dense: 4,986 steps .4231 / .6135 / .6927, this .4226 / .6126 / .6917 on
//! 2026-10-21); the time a step is what a host core spends on it (2.0 ms, 40 pairs, one thread).
//!
//!   bench-pred [--dir bringup/expert-cache/data/phase3] [--steps N]
//!
//! The arrays are build.py's: LG (41 layers × tokens × 288 route-ahead logits for the next layer), PK (× 8 picks), n
//! (tokens a step), bias (64 × 288, the router's selection bias by layer); row j is layer 3 + j.

use glm_5_3_flash::predict::{rank, Predictor};
use glm_5_3_flash::{E, TOP_K};
use harness::arg;

/// An .npy file's data after its header.
fn npy(path: &std::path::Path) -> memmap2::Mmap {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    unsafe { memmap2::Mmap::map(&f).unwrap() }
}
fn body(m: &memmap2::Mmap) -> &[u8] {
    assert_eq!(&m[..6], b"\x93NUMPY");
    let (major, len) = (m[6], if m[6] == 1 { u16::from_le_bytes([m[8], m[9]]) as usize + 10 } else { u32::from_le_bytes([m[8], m[9], m[10], m[11]]) as usize + 12 });
    let _ = major;
    &m[len..]
}
fn f32s(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

fn main() {
    let dir = arg("--dir").unwrap_or_else(|| "bringup/expert-cache/data/phase3".into());
    let (lgm, pkm, nm, bm) = (npy(&dir.join("LG.npy")), npy(&dir.join("PK.npy")), npy(&dir.join("n.npy")), npy(&dir.join("bias.npy")));
    let (lg, bias) = (f32s(body(&lgm)), f32s(body(&bm)));
    let pk: &[i16] = { let b = body(&pkm); unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i16, b.len() / 2) } };
    let ns: &[i64] = { let b = body(&nm); unsafe { std::slice::from_raw_parts(b.as_ptr() as *const i64, b.len() / 8) } };
    const L: usize = 41;
    let t_all: usize = ns.iter().map(|&v| v as usize).sum();
    let steps = arg("--steps").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(ns.len()).min(ns.len());
    assert_eq!(lg.len(), L * t_all * E);
    // the router bias by layer, layers 3..=43 (the dump's), empty elsewhere
    let rb: Vec<Vec<f32>> = (0..44).map(|l| if l >= 3 { bias[l * E..(l + 1) * E].to_vec() } else { Vec::new() }).collect();
    let mut p = Predictor::new(rb);
    const KS: [usize; 3] = [8, 13, 16];
    let (mut hits, mut tot) = ([0f64; 3], 0f64);
    let mut z = vec![0f32; 64 * E];
    let mut out = Vec::new();
    let (mut t_pred, mut t_learn) = (0f64, 0f64);
    let mut t0 = 0usize;
    for s in 0..steps {
        let m = ns[s] as usize;
        let picks = |j: usize| -> Vec<[usize; TOP_K]> { (0..m).map(|i| std::array::from_fn(|k| pk[(j * t_all + t0 + i) * TOP_K + k] as usize)).collect() };
        for j in 0..L - 1 {
            let l = 3 + j;
            let ahead = &lg[(j * t_all + t0) * E..(j * t_all + t0 + m) * E];
            let (here, next) = (picks(j), picks(j + 1));
            let c = std::time::Instant::now();
            p.predict(l, m, ahead, &here, &mut z);
            t_pred += c.elapsed().as_secs_f64();
            let mut y = [false; E];
            next.iter().flatten().for_each(|&e| y[e] = true);
            tot += y.iter().filter(|&&v| v).count() as f64;
            for (k, &cap) in KS.iter().enumerate() {
                rank(&z, m, &|_| false, cap, &mut out);
                hits[k] += out.iter().filter(|&&e| y[e]).count() as f64;
            }
            let c = std::time::Instant::now();
            p.learn(l, &next);
            t_learn += c.elapsed().as_secs_f64();
        }
        t0 += m;
        if s % 1000 == 0 {
            println!("step {s}: recall@8/13/16 {:.4} {:.4} {:.4}", hits[0] / tot, hits[1] / tot, hits[2] / tot);
        }
    }
    println!("steps {steps}: recall@8/13/16 {:.4} {:.4} {:.4}; a step (40 pairs, one thread) predict {:.2} ms, learn {:.2} ms", hits[0] / tot, hits[1] / tot, hits[2] / tot, t_pred / steps as f64 * 1e3, t_learn / steps as f64 * 1e3);
}
