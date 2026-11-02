//! Adam over a slice, torch's form (`torch.optim.Adam`, the inference crate's predictor uses
//! the same): `m = β₁m + (1−β₁)g`, `v = β₂v + (1−β₂)g²`, the step
//! `p −= lr / (1 − β₁ᵗ) · m / (√v / √(1 − β₂ᵗ) + ε)`. Split in two so a caller can look at the
//! proposed step before applying it (a movement cap scales it): `moments` advances the moments
//! and writes the unscaled step `d` (so that `p −= d` is Adam's step); `apply` subtracts `s · d`.

#[derive(Clone, Copy, Debug)]
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for Adam {
    fn default() -> Self {
        Adam { lr: 1e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

impl Adam {
    /// Advance the moments with `g` for step `t` (1-based) and write the step into `d`.
    pub fn moments(&self, t: u32, g: &[f32], m: &mut [f32], v: &mut [f32], d: &mut [f32]) {
        let c1 = 1.0 - self.beta1.powi(t as i32);
        let c2 = (1.0 - self.beta2.powi(t as i32)).sqrt();
        let lr = self.lr / c1;
        for i in 0..g.len() {
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i];
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i] * g[i];
            d[i] = lr * m[i] / (v[i].sqrt() / c2 + self.eps);
        }
    }
}

/// `p −= s · d`.
pub fn apply(p: &mut [f32], d: &[f32], s: f32) {
    for i in 0..p.len() {
        p[i] -= s * d[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torch.optim.Adam on one parameter, step by step: p 1.0, g constant 0.5, lr 0.1 — the first
    /// step is exactly lr (m̂/√v̂ = 1), later steps shrink toward lr·0.5/0.5 = lr as the moments settle.
    #[test]
    fn first_step_is_the_learning_rate() {
        let a = Adam { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 0.0 };
        let (mut p, mut m, mut v, mut d) = ([1.0f32], [0.0f32], [0.0f32], [0.0f32]);
        a.moments(1, &[0.5], &mut m, &mut v, &mut d);
        apply(&mut p, &d, 1.0);
        assert!((p[0] - 0.9).abs() < 1e-6, "{}", p[0]);
        // torch's second step on the same gradient: m̂ = 0.5, v̂ = 0.25 → again lr
        a.moments(2, &[0.5], &mut m, &mut v, &mut d);
        assert!((d[0] - 0.1).abs() < 1e-6, "{}", d[0]);
    }
}
