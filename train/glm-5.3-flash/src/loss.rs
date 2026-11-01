//! The per-position loss against a teacher kept as its top-K: the teacher's K most probable ids
//! with their log-probabilities, as the serving path can store them when a session's future has
//! arrived (`sdpo.md` §7). With the teacher renormalized over its K ids (`q`), the loss is the
//! forward KL on that support,
//!
//! ```text
//! L = Σ_k q_k · (ln q_k − ln p_k)        p = softmax(student logits)
//! d L / d logit_v = p_v − Σ_k q_k · [v = id_k]
//! ```
//!
//! which is what a top-K store can compute exactly. `sdpo.md` §2 writes the objective the other
//! way round, `KL(student ‖ teacher)`; that direction needs the teacher's log-probability at
//! every id the student puts mass on, which a top-K store does not hold. Which to keep — this
//! form, a fuller store, or the sampled estimator — is open (2026-11-01); the gradient's plumbing
//! here is the same for any of them.

use common::math::{exp, ln};

/// A position's teacher: `ids[k]` with `logp[k] = ln π_t(ids[k])`, any K, unnormalized over K.
pub struct Teacher<'a> {
    pub ids: &'a [u32],
    pub logp: &'a [f32],
}

/// The loss at one position and its gradient at the student's logits (`dlogits` written, `[vocab]`).
pub fn forward_kl_topk(logits: &[f32], t: &Teacher, dlogits: &mut [f32]) -> f32 {
    debug_assert_eq!(t.ids.len(), t.logp.len());
    // the student's log-softmax, the sum in f64 as `softmax::log_prob` does
    let mut m = f32::NEG_INFINITY;
    for &v in logits {
        if v > m {
            m = v;
        }
    }
    let mut s = 0f64;
    for &v in logits {
        s += exp(v - m) as f64;
    }
    let lse = m + ln(s as f32);
    for (d, &v) in dlogits.iter_mut().zip(logits) {
        *d = exp(v - lse);
    }
    // the teacher renormalized over its K
    let mut tm = f32::NEG_INFINITY;
    for &v in t.logp {
        if v > tm {
            tm = v;
        }
    }
    let mut ts = 0f32;
    for &v in t.logp {
        ts += exp(v - tm);
    }
    let tlse = tm + ln(ts);
    let mut loss = 0f32;
    for k in 0..t.ids.len() {
        let lq = t.logp[k] - tlse;
        let q = exp(lq);
        let id = t.ids[k] as usize;
        loss += q * (lq - (logits[id] - lse));
        dlogits[id] -= q;
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::testing::Rng;

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rnd = Rng::new(0x1055_0001);
        let v = 40;
        let mut logits: Vec<f32> = (0..v).map(|_| rnd.unit() * 3.0).collect();
        let ids: Vec<u32> = vec![3, 17, 29, 0, 8];
        let logp: Vec<f32> = (0..ids.len()).map(|_| rnd.unit() * 2.0 - 3.0).collect();
        let t = Teacher { ids: &ids, logp: &logp };
        let mut d = vec![0f32; v];
        let l0 = forward_kl_topk(&logits, &t, &mut d);
        assert!(l0 >= 0.0);
        let eps = 1e-2f32;
        for i in 0..v {
            let base = logits[i];
            let mut scratch = vec![0f32; v];
            logits[i] = base + eps;
            let p = forward_kl_topk(&logits, &t, &mut scratch) as f64;
            logits[i] = base - eps;
            let m = forward_kl_topk(&logits, &t, &mut scratch) as f64;
            logits[i] = base;
            let num = ((p - m) / (2.0 * eps as f64)) as f32;
            assert!((num - d[i]).abs() <= 2e-3 * (1.0 + num.abs()), "d[{i}]: numeric {num} analytic {}", d[i]);
        }
        // a teacher equal to the student on its support: the loss is the floor −ln(the student's
        // mass on K) — the forward KL on a renormalized support is zero only when the student
        // puts all its mass there
        let mut same = vec![0f32; ids.len()];
        let mut scratch = vec![0f32; v];
        let lse = {
            let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            m + ln(logits.iter().map(|x| exp(x - m)).sum::<f32>())
        };
        let mut mass = 0f32;
        for (k, &id) in ids.iter().enumerate() {
            same[k] = logits[id as usize] - lse;
            mass += exp(same[k]);
        }
        let l = forward_kl_topk(&logits, &Teacher { ids: &ids, logp: &same }, &mut scratch);
        assert!((l + ln(mass)).abs() < 1e-5, "{l} vs floor {}", -ln(mass));
    }
}
