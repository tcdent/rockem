//! The RMS norm's backward: `y = w ⊙ x · r`, `r = rsqrt(mean(x²) + eps)` (`common::rmsnorm`);
//! given `dy`, `dx = r · (w ⊙ dy) − x · r³ / N · ⟨w ⊙ dy, x⟩`. The weight is frozen.

use common::math::rsqrt;

/// `dx[0..N) += …` from `dy`, with `r` recomputed from `x` as the forward computed it.
#[inline(always)]
pub fn backward<const N: usize>(x: &[f32], w: &[f32], eps: f32, dy: &[f32], dx: &mut [f32]) {
    let mut ss = 0f32;
    for &v in &x[..N] {
        ss += v * v;
    }
    let r = rsqrt(ss / N as f32 + eps);
    let mut dot = 0f32;
    for i in 0..N {
        dot += w[i] * dy[i] * x[i];
    }
    let c = r * r * r / N as f32 * dot;
    for i in 0..N {
        dx[i] += r * w[i] * dy[i] - x[i] * c;
    }
}

/// The same for the unweighted norm `y = x · r` (the hyper-connections' flattened streams).
#[inline(always)]
pub fn backward_unweighted<const N: usize>(x: &[f32], eps: f32, dy: &[f32], dx: &mut [f32]) {
    let mut ss = 0f32;
    for &v in &x[..N] {
        ss += v * v;
    }
    let r = rsqrt(ss / N as f32 + eps);
    let mut dot = 0f32;
    for i in 0..N {
        dot += dy[i] * x[i];
    }
    let c = r * r * r / N as f32 * dot;
    for i in 0..N {
        dx[i] += r * dy[i] - x[i] * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

    #[test]
    fn matches_finite_differences() {
        const N: usize = 16;
        let mut rnd = Rng::new(0x4453_0001);
        let x: Vec<f32> = (0..N).map(|_| rnd.unit()).collect();
        let w: Vec<f32> = (0..N).map(|_| 0.5 + rnd.unit()).collect();
        let dy: Vec<f32> = (0..N).map(|_| rnd.unit()).collect();
        let eps = 1e-5;
        let loss = |x: &[f32]| -> f64 {
            let mut y = vec![0f32; N];
            common::rmsnorm::rmsnorm::<N>(x, &w, eps, &mut y);
            y.iter().zip(&dy).map(|(a, b)| (*a as f64) * (*b as f64)).sum()
        };
        let mut dx = vec![0f32; N];
        backward::<N>(&x, &w, eps, &dy, &mut dx);
        let h = 1e-3f32;
        for i in 0..N {
            let mut xp = x.clone();
            xp[i] += h;
            let mut xm = x.clone();
            xm[i] -= h;
            let num = ((loss(&xp) - loss(&xm)) / (2.0 * h as f64)) as f32;
            assert!((num - dx[i]).abs() <= 2e-3 * (1.0 + num.abs()), "{i}: {num} vs {}", dx[i]);
        }
    }
}
