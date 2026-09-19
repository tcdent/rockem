//! bench-gptq — one expert's GPTQ, single-threaded, on synthetic data of the
//! real shapes: the Hessian factor, the sweep, and the sweep's block update
//! at a padded row stride, to see whether the 4 KB-congruent rows of `w`
//! and `u` (both stride k = 2048 floats) pay a store-forwarding penalty.
//!
//!   bench-gptq [m] [k]

use common::linalg::{axpy, chol_of_inverse, dot};
use common::q4::{gptq, GROUP};
use common::testing::Rng;
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let (m, k, n) = (*args.first().unwrap_or(&2816), *args.get(1).unwrap_or(&2048), 257);
    let mut rng = Rng::new(1);
    let w0: Vec<f32> = (0..m * k).map(|_| rng.unit() * 0.02).collect();
    let x: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();

    let t = Instant::now();
    let mut h = vec![0f32; k * k];
    for i in 0..k {
        for j in 0..=i {
            let v: f32 = (0..n).map(|r| x[r * k + i] * x[r * k + j]).sum();
            h[i * k + j] = v;
            h[j * k + i] = v;
        }
    }
    let mean = (0..k).map(|i| h[i * k + i]).sum::<f32>() / k as f32;
    for i in 0..k {
        h[i * k + i] += 0.01 * mean;
    }
    let t_h = t.elapsed().as_secs_f32();
    let t = Instant::now();
    let u = chol_of_inverse(k, &mut h);
    println!("hessian {t_h:.2}s, chol_of_inverse({k}) {:.2}s", t.elapsed().as_secs_f32());

    let mut w = w0.clone();
    let t = Instant::now();
    let (q, _) = gptq(m, k, &mut w, &u);
    let t_g = t.elapsed().as_secs_f32();
    println!("gptq {m}x{k}: {t_g:.2}s  ({:.1} GFLOP/s on the block updates)", (m * k * k) as f32 / t_g / 1e9);

    // the block update alone, at stride k and at a padded stride
    for pad in [0usize, 16] {
        let ks = k + pad;
        let mut wp = vec![0f32; m * ks];
        for r in 0..m {
            wp[r * ks..r * ks + k].copy_from_slice(&w0[r * k..(r + 1) * k]);
        }
        let err: Vec<f32> = (0..m * GROUP).map(|_| rng.unit()).collect();
        let t = Instant::now();
        let mut flops = 0usize;
        for g in 0..k / GROUP {
            let (i1, i2) = (g * GROUP, (g + 1) * GROUP);
            if i2 < k {
                for r in 0..m {
                    let wrow = &mut wp[r * ks + i2..r * ks + k];
                    for b in 0..GROUP {
                        axpy(wrow, -err[r * GROUP + b], &u[(i1 + b) * k + i2..(i1 + b + 1) * k]);
                    }
                    flops += 2 * GROUP * (k - i2);
                }
            }
        }
        let dt = t.elapsed().as_secs_f32();
        println!("block updates, row stride {ks}: {dt:.2}s = {:.1} GFLOP/s", flops as f32 / dt / 1e9);
    }
    // dot along rows at stride k (cholesky's shape)
    let t = Instant::now();
    let mut acc = 0f32;
    for i in 0..k {
        for j in 0..i {
            acc += dot(&u[i * k..i * k + j], &u[j * k..j * k + j]);
        }
    }
    let dt = t.elapsed().as_secs_f32();
    println!("row dots {k}²/2: {dt:.2}s = {:.1} GFLOP/s ({acc:e})", (k * k * k / 3) as f32 / dt / 1e9);
    std::hint::black_box(q);
}
