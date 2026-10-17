//! Sampling: logits → token, under a recipe and a seeded generator. The
//! recipe is the checkpoint's generation config as transformers (the version
//! it names) applies it, in its order: the logits over the temperature, then
//! the `top_k` largest kept (everything tied with the k-th included), then
//! the nucleus: candidates in descending order, kept while the mass above
//! them is under `top_p` of the candidates' mass (the token that crosses the
//! line is kept), and a draw over what is kept. llama.cpp keeps the crossing
//! token too. Temperature 0 is greedy.
//!
//! The candidates are ordered by value descending and index ascending, ties
//! included, and capped at `CAP`; the cap binds only when more than
//! `CAP - top_k` logits tie at the k-th value, where the reference's own
//! result is arbitrary. Both realizations — this one and the card's `sample`
//! kernel — compute the same candidate list, the same exponentials at the
//! same max, and the same f64 walk in the same order, so they draw the same
//! token from the same logits.
//!
//! The generator is xoshiro256** seeded through splitmix64 — tiny, no_std —
//! but the node draws with `uniform_at`, a function of the seed and the
//! position alone.

use super::math::exp;
use super::softmax::argmax;

pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *v = x ^ (x >> 31);
        }
        Rng(s)
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    /// Uniform in [0, 1) with 53 bits.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The uniform for position `pos` of a sequence under `seed`: a function of
/// the two alone (one splitmix64 step), so a position drawn after being
/// computed, after being served from a store, or after a restart draws the
/// same, and the sampler keeps no state a cache could get out of step with.
#[inline]
pub fn uniform_at(seed: u64, pos: u64) -> f64 {
    let mut x = seed.wrapping_add(pos.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// The recipe: the checkpoint's constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// 1..=CAP. Everything tied with the k-th largest logit is a candidate too.
    pub top_k: usize,
    pub top_p: f32,
}
impl Sampling {
    pub const GREEDY: Sampling = Sampling { temperature: 0.0, top_k: 1, top_p: 1.0 };
}
/// The most candidates a draw considers: the top-k and their ties, in index order.
pub const CAP: usize = 256;

/// The candidate list for `logits` under `top_k`: (value, index) pairs by
/// value descending, index ascending — the top_k-th largest value and
/// everything at or above it, the first `CAP` of those by index. Returns the
/// list's length; entries at or beyond it are unspecified.
pub fn candidates(logits: &[f32], top_k: usize, out: &mut [(f32, u32); CAP]) -> usize {
    let k = top_k.clamp(1, CAP);
    // θ, the k-th largest with multiplicity: a descending list of the k largest seen
    let mut kept = [f32::NEG_INFINITY; CAP];
    let mut n = 0;
    for &v in logits {
        if n < k || v > kept[k - 1] {
            let mut i = if n < k { n } else { k - 1 };
            while i > 0 && kept[i - 1] < v {
                kept[i] = kept[i - 1];
                i -= 1;
            }
            kept[i] = v;
            if n < k {
                n += 1;
            }
        }
    }
    let theta = kept[k - 1];
    let mut m = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v >= theta {
            if m < CAP {
                out[m] = (v, i as u32);
            }
            m += 1;
        }
    }
    let m = m.min(CAP);
    for i in 1..m {
        let e = out[i];
        let mut j = i;
        while j > 0 && before(e, out[j - 1]) {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = e;
    }
    m
}
#[inline(always)]
fn before(a: (f32, u32), b: (f32, u32)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// Draw a token from `logits` under `s` with the uniform `u`. The card's
/// `sample` kernel is this op's twin, step for step.
pub fn draw(logits: &[f32], s: &Sampling, u: f64) -> usize {
    if s.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut c = [(0f32, 0u32); CAP];
    let n = candidates(logits, s.top_k, &mut c);
    let k = s.top_k.clamp(1, CAP).min(n);
    // the candidates at or above the k-th of the list (its ties included), and their masses
    let theta = c[k - 1].0;
    let mut n2 = k;
    while n2 < n && c[n2].0 >= theta {
        n2 += 1;
    }
    let inv_t = 1.0 / s.temperature;
    let top = c[0].0;
    let mut mass = [0f32; CAP];
    let mut total = 0f64;
    for i in 0..n2 {
        mass[i] = exp((c[i].0 - top) * inv_t);
        total += mass[i] as f64;
    }
    // the nucleus: kept while the mass above is under top_p of the total
    let lim = s.top_p as f64 * total;
    let mut above = 0f64;
    let mut keep = 0;
    while keep < n2 && above < lim {
        above += mass[keep] as f64;
        keep += 1;
    }
    let target = u * above;
    let mut acc = 0f64;
    for i in 0..keep {
        acc += mass[i] as f64;
        if target < acc {
            return c[i].1 as usize;
        }
    }
    c[keep - 1].1 as usize // u landed past the rounded-off tail
}

#[cfg(any(feature = "std", test))]
/// The distribution `draw` samples from, as (id, probability) over its support: the top-k
/// candidates and their ties at temperature, cut to the nucleus — `draw`'s arithmetic, so a draw
/// with uniform `u` is the id where the running mass first exceeds `u`. Greedy (temperature 0):
/// the argmax alone.
pub fn dist(logits: &[f32], s: &Sampling) -> std::vec::Vec<(u32, f64)> {
    if s.temperature <= 0.0 {
        return std::vec![(argmax(logits) as u32, 1.0)];
    }
    let mut c = [(0f32, 0u32); CAP];
    let n = candidates(logits, s.top_k, &mut c);
    let k = s.top_k.clamp(1, CAP).min(n);
    let theta = c[k - 1].0;
    let mut n2 = k;
    while n2 < n && c[n2].0 >= theta {
        n2 += 1;
    }
    let inv_t = 1.0 / s.temperature;
    let top = c[0].0;
    let mut mass = [0f32; CAP];
    let mut total = 0f64;
    for i in 0..n2 {
        mass[i] = exp((c[i].0 - top) * inv_t);
        total += mass[i] as f64;
    }
    let lim = s.top_p as f64 * total;
    let mut above = 0f64;
    let mut keep = 0;
    while keep < n2 && above < lim {
        above += mass[keep] as f64;
        keep += 1;
    }
    (0..keep).map(|i| (c[i].1, mass[i] as f64 / above)).collect()
}
#[cfg(any(feature = "std", test))]
/// Speculative verification of a draft `x` drawn from `q` (the draft's logits) against the
/// model's `p` at the same position: accepted with probability `min(1, p(x) / q(x))` by the
/// uniform `r`; else a token from the residual `max(p − q, 0)` normalized, by the uniform `u`
/// (the model's distribution exactly, Leviathan et al. / Chen et al.). Returns (accepted, token).
pub fn verify(p_logits: &[f32], q_logits: &[f32], s: &Sampling, x: u32, r: f64, u: f64) -> (bool, u32) {
    let (p, q) = (dist(p_logits, s), dist(q_logits, s));
    let at = |d: &[(u32, f64)], id: u32| d.iter().find(|(i, _)| *i == id).map_or(0.0, |(_, v)| *v);
    let (px, qx) = (at(&p, x), at(&q, x));
    if qx > 0.0 && r * qx < px {
        return (true, x);
    }
    // the residual over p's support
    let mut res: std::vec::Vec<(u32, f64)> = p.iter().map(|&(id, pv)| (id, (pv - at(&q, id)).max(0.0))).filter(|(_, v)| *v > 0.0).collect();
    let total: f64 = res.iter().map(|(_, v)| v).sum();
    if res.is_empty() || total <= 0.0 {
        // p within q: the draw from p itself
        res = p;
    }
    let total: f64 = res.iter().map(|(_, v)| v).sum();
    let target = u * total;
    let mut acc = 0f64;
    for &(id, v) in &res {
        acc += v;
        if target < acc {
            return (false, id);
        }
    }
    (false, res[res.len() - 1].0)
}

pub fn sample(logits: &[f32], s: &Sampling, rng: &mut Rng) -> usize {
    let u = rng.uniform();
    draw(logits, s, u)
}

#[cfg(test)]
mod tests {
    use super::*;
    const OPEN: Sampling = Sampling { temperature: 1.0, top_k: CAP, top_p: 1.0 };

    #[test]
    fn zero_temperature_is_greedy() {
        let l = [0.1f32, 2.0, -1.0, 1.9];
        assert_eq!(draw(&l, &Sampling::GREEDY, 0.99), 1);
    }

    #[test]
    fn candidates_are_the_top_k_with_ties_in_canonical_order() {
        let l = [1.0f32, 3.0, 2.0, 2.0, 0.5, 2.0, 3.0];
        let mut c = [(0f32, 0u32); CAP];
        let n = candidates(&l, 2, &mut c);
        assert_eq!((n, &c[..n]), (2, &[(3.0, 1), (3.0, 6)][..]));
        let n = candidates(&l, 3, &mut c);
        assert_eq!(&c[..n], &[(3.0, 1), (3.0, 6), (2.0, 2), (2.0, 3), (2.0, 5)]);
        let n = candidates(&l, 100, &mut c);
        assert_eq!(n, 7);
        // the cap: the first CAP by index among the ties
        let big = [1.0f32; 300];
        let n = candidates(&big, 4, &mut c);
        assert_eq!(n, CAP);
        assert_eq!((c[0].1, c[CAP - 1].1), (0, CAP as u32 - 1));
    }

    #[test]
    fn nucleus_keeps_the_crossing_token() {
        // logits = ln p for p = 0.5, 0.3, 0.15, 0.05 (index order 1, 0, 3, 2)
        let l = [-1.2039728f32, -0.6931472, -2.9957323, -1.89712];
        let at = |top_k, top_p, u| draw(&l, &Sampling { temperature: 1.0, top_k, top_p }, u);
        // k=3 drops 0.05; p=0.9 of the 0.95 left is 0.855: 0.5, 0.8 are under it, so all three stay
        assert_eq!(at(3, 0.9, 0.9), 3); // target 0.855 lands in the third
        // p=0.8 of 0.95 is 0.76: 0.5 + 0.3 = 0.8 crosses at the second, the third is out
        assert_eq!(at(3, 0.8, 0.9), 0); // target 0.72 lands in the second
        assert_eq!(at(3, 0.8, 0.1), 1);
        // top_p tiny: only the top token
        assert_eq!(at(4, 0.01, 0.999), 1);
    }

    #[test]
    fn seed_fixes_the_stream() {
        let (mut a, mut b) = (Rng::new(42), Rng::new(42));
        for _ in 0..1000 {
            assert_eq!(a.next(), b.next());
        }
        assert_ne!(Rng::new(42).next(), Rng::new(43).next());
    }

    #[test]
    fn uniform_at_is_a_function_of_seed_and_position() {
        assert_eq!(uniform_at(7, 100), uniform_at(7, 100));
        assert_ne!(uniform_at(7, 100), uniform_at(7, 101));
        assert_ne!(uniform_at(7, 100), uniform_at(8, 100));
        let mut acc = 0f64;
        for p in 0..10000 {
            let u = uniform_at(3, p);
            assert!((0.0..1.0).contains(&u));
            acc += u;
        }
        assert!((acc / 10000.0 - 0.5).abs() < 0.02);
    }

    #[test]
    fn frequencies_follow_the_distribution() {
        // logits = ln p for p = 0.2, 0.3, 0.5; the open recipe is the plain softmax draw
        let mut rng = Rng::new(7);
        let mut counts = [0usize; 3];
        let n = 20000;
        let l = [-1.6094379f32, -1.2039728, -0.6931472];
        for _ in 0..n {
            counts[sample(&l, &OPEN, &mut rng)] += 1;
        }
        for (c, want) in counts.iter().zip([0.2, 0.3, 0.5]) {
            let got = *c as f64 / n as f64;
            assert!((got - want).abs() < 0.015, "{got} vs {want}");
        }
        // temperature sharpens: at 0.25 the top entry dominates
        let mut top = 0;
        for _ in 0..n {
            top += (sample(&l, &Sampling { temperature: 0.25, ..OPEN }, &mut rng) == 2) as usize;
        }
        assert!(top as f64 / n as f64 > 0.85);
    }
}
