//! Kimi delta attention over a run of positions: the forward is the inference crate's
//! `kda::step`, one position at a time, keeping what the backward needs; the backward runs the
//! run in reverse, each head's recurrence recomputed from the entering memory (so the
//! per-position states are never stored across the run — one head's `(n + 1)` states at a
//! time), to `d xn` and the gradient at the entering state.
//!
//! Per position, per head (`reference::kda_forward`'s arithmetic, `kda::head`'s order):
//!
//! ```text
//! mixed = [Wq·xn | Wk·xn | Wv·xn]       conv = silu(taps over the last CONV_TAPS mixed)
//! f = Wfb·(Wfa·xn)   β = σ(Wb·xn)   gate = Wgb·(Wga·xn)
//! q̂ = l2(q)/√d   k̂ = l2(k)   e_kk = exp(LOWER · σ(decay_h · (f_kk + dt_bias)))
//! S₁ = diag(e) S      mem = S₁ᵀ k̂      δ = (v − mem) · β      S₂ = S₁ + k̂ ⊗ δ      o = S₂ᵀ q̂
//! out_h = rmsnorm(o) ⊙ σ(gate)          y = Wo · [out_h]
//! ```
//!
//! Everything is frozen: every projection contributes its transposed product and no weight
//! gradient. The entering state (the window and the memory) gets a gradient, so a caller may
//! carry it into the previous run or drop it at the history's edge.

use common::math::{exp, sqrt};
use common::pool::Serial;
use common::trace::NoTrace;
use glm_5_3_flash::hc::sigmoid;
use glm_5_3_flash::kda::{self, HEAD_SCRATCH};
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::weights;
use glm_5_3_flash::{CONV_DIM, CONV_TAPS, H, KDA_DIM, KDA_HEADS, KDA_QKV, LINEAR_GATE_LOWER_BOUND, RMS_NORM_EPS};
use train_common::{bf16, rmsnorm, swiglu};

/// The inference step's scratch, owned.
pub struct Scratch {
    mixed: Vec<f32>,
    conv: Vec<f32>,
    fa: Vec<f32>,
    f: Vec<f32>,
    beta: Vec<f32>,
    ga: Vec<f32>,
    gate: Vec<f32>,
    normed: Vec<f32>,
    heads: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Scratch { mixed: vec![0.0; CONV_DIM], conv: vec![0.0; CONV_DIM], fa: vec![0.0; KDA_DIM], f: vec![0.0; KDA_QKV], beta: vec![0.0; KDA_HEADS], ga: vec![0.0; KDA_DIM], gate: vec![0.0; KDA_QKV], normed: vec![0.0; KDA_QKV], heads: vec![0.0; KDA_HEADS * HEAD_SCRATCH] }
    }
    fn view(&mut self) -> kda::Scratch<'_> {
        kda::Scratch { mixed: &mut self.mixed, conv: &mut self.conv, fa: &mut self.fa, f: &mut self.f, beta: &mut self.beta, ga: &mut self.ga, gate: &mut self.gate, normed: &mut self.normed, heads: &mut self.heads }
    }
}
impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// What a run's forward keeps: per position the step's intermediates, and the state it entered with.
pub struct Saved {
    pub n: usize,
    pub mixed: Vec<f32>,  // [n][CONV_DIM]
    pub conv: Vec<f32>,   // [n][CONV_DIM]
    pub f: Vec<f32>,      // [n][KDA_QKV]
    pub beta: Vec<f32>,   // [n][KDA_HEADS]
    pub gate: Vec<f32>,   // [n][KDA_QKV]
    pub heads: Vec<f32>,  // [n][KDA_HEADS][HEAD_SCRATCH]: q̂ | k̂ | δ | o | on
    pub normed: Vec<f32>, // [n][KDA_QKV]
    pub window_in: Vec<f32>,
    pub mem_in: Vec<f32>,
}

impl Saved {
    pub fn new(n: usize) -> Self {
        Saved { n, mixed: vec![0.0; n * CONV_DIM], conv: vec![0.0; n * CONV_DIM], f: vec![0.0; n * KDA_QKV], beta: vec![0.0; n * KDA_HEADS], gate: vec![0.0; n * KDA_QKV], heads: vec![0.0; n * KDA_HEADS * HEAD_SCRATCH], normed: vec![0.0; n * KDA_QKV], window_in: vec![0.0; KDA_WINDOW], mem_in: vec![0.0; KDA_MEMORY] }
    }
}

/// `n` positions of `xn` (`[n][H]`) through the layer, advancing `window` and `mem`; `y: [n][H]` written.
pub fn forward(l: usize, w: &weights::Kda, xn: &[f32], n: usize, window: &mut [f32], mem: &mut [f32], s: &mut Scratch, saved: &mut Saved, y: &mut [f32]) {
    debug_assert!(saved.n == n);
    saved.window_in.copy_from_slice(&window[..KDA_WINDOW]);
    saved.mem_in.copy_from_slice(&mem[..KDA_MEMORY]);
    for t in 0..n {
        kda::step(l, w, &xn[t * H..(t + 1) * H], window, mem, &mut s.view(), &Serial, &mut NoTrace, &mut y[t * H..(t + 1) * H]);
        saved.mixed[t * CONV_DIM..(t + 1) * CONV_DIM].copy_from_slice(&s.mixed);
        saved.conv[t * CONV_DIM..(t + 1) * CONV_DIM].copy_from_slice(&s.conv);
        saved.f[t * KDA_QKV..(t + 1) * KDA_QKV].copy_from_slice(&s.f);
        saved.beta[t * KDA_HEADS..(t + 1) * KDA_HEADS].copy_from_slice(&s.beta);
        saved.gate[t * KDA_QKV..(t + 1) * KDA_QKV].copy_from_slice(&s.gate);
        saved.heads[t * KDA_HEADS * HEAD_SCRATCH..(t + 1) * KDA_HEADS * HEAD_SCRATCH].copy_from_slice(&s.heads);
        saved.normed[t * KDA_QKV..(t + 1) * KDA_QKV].copy_from_slice(&s.normed);
    }
}

/// y = x · r, r = (Σx² + 1e-6)^-½ (`kda::l2norm`): dx = r·dy − x·r³·⟨dy, x⟩.
fn l2norm_backward(x: &[f32], dy: &[f32], dx: &mut [f32]) {
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let r = 1.0 / sqrt(ss + 1e-6);
    let mut dot = 0f32;
    for i in 0..x.len() {
        dot += dy[i] * x[i];
    }
    for i in 0..x.len() {
        dx[i] = r * dy[i] - x[i] * r * r * r * dot;
    }
}

/// Head `h`'s backward over the run: `ds` enters as the gradient at the memory after the run and
/// leaves as the gradient at the memory before it; the per-position gradients land in `dconv`
/// (the head's q, k, v slices), `df`, `dbeta`, `dgate`.
#[allow(clippy::too_many_arguments)]
fn head_backward(w: &weights::Kda, h: usize, sv: &Saved, dnormed: &[f32], ds: &mut [f32], states: &mut [f32], dconv: &mut [f32], df: &mut [f32], dbeta: &mut [f32], dgate: &mut [f32]) {
    const D: usize = KDA_DIM;
    const DD: usize = KDA_DIM * KDA_DIM;
    let n = sv.n;
    let decay = exp(w.a_log[h]);
    let head = |t: usize| &sv.heads[(t * KDA_HEADS + h) * HEAD_SCRATCH..(t * KDA_HEADS + h + 1) * HEAD_SCRATCH];
    // the forward recurrence again, from the entering memory: states[t] is S before position t,
    // states[n] after the run; e and the read mem per position alongside
    let mut e_all = vec![0f32; n * D];
    let mut mem_read = vec![0f32; n * D];
    states[..DD].copy_from_slice(&sv.mem_in[h * DD..(h + 1) * DD]);
    for t in 0..n {
        let hs = head(t);
        let (kh, delta) = (&hs[D..2 * D], &hs[2 * D..3 * D]);
        let (prev, next) = states[t * DD..(t + 2) * DD].split_at_mut(DD);
        next.copy_from_slice(prev);
        for kk in 0..D {
            let e = exp(LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (sv.f[t * KDA_QKV + h * D + kk] + w.dt_bias[h * D + kk])));
            e_all[t * D + kk] = e;
            let row = &mut next[kk * D..(kk + 1) * D];
            let k = kh[kk];
            for vv in 0..D {
                row[vv] *= e;
                mem_read[t * D + vv] += row[vv] * k;
            }
        }
        for kk in 0..D {
            let row = &mut next[kk * D..(kk + 1) * D];
            let k = kh[kk];
            for vv in 0..D {
                row[vv] += k * delta[vv];
            }
        }
    }
    let scale = 1.0 / sqrt(D as f32);
    let (mut don, mut dqh, mut dkh, mut dmem) = ([0f32; D], [0f32; D], [0f32; D], [0f32; D]);
    for t in (0..n).rev() {
        let hs = head(t);
        let (qh, kh, delta, o, on) = (&hs[..D], &hs[D..2 * D], &hs[2 * D..3 * D], &hs[3 * D..4 * D], &hs[4 * D..5 * D]);
        let conv = &sv.conv[t * CONV_DIM..(t + 1) * CONV_DIM];
        let vh = &conv[2 * KDA_QKV + h * D..2 * KDA_QKV + (h + 1) * D];
        let (s_prev, s2) = (&states[t * DD..(t + 1) * DD], &states[(t + 1) * DD..(t + 2) * DD]);
        let e = &e_all[t * D..(t + 1) * D];
        let mr = &mem_read[t * D..(t + 1) * D];
        let b = sigmoid(sv.beta[t * KDA_HEADS + h]);
        // out = on ⊙ σ(gate); on = rmsnorm(o)
        let dout = &dnormed[t * KDA_QKV + h * D..t * KDA_QKV + (h + 1) * D];
        for vv in 0..D {
            let sg = sigmoid(sv.gate[t * KDA_QKV + h * D + vv]);
            don[vv] = dout[vv] * sg;
            dgate[t * KDA_QKV + h * D + vv] = dout[vv] * on[vv] * sg * (1.0 - sg);
        }
        let mut do_ = [0f32; D];
        rmsnorm::backward::<D>(o, w.o_norm, RMS_NORM_EPS, &don, &mut do_);
        // o = S₂ᵀ q̂: dS += q̂ ⊗ do; d q̂ = S₂ · do
        for kk in 0..D {
            let mut acc = 0f32;
            let row = &mut ds[kk * D..(kk + 1) * D];
            let s2r = &s2[kk * D..(kk + 1) * D];
            for vv in 0..D {
                row[vv] += qh[kk] * do_[vv];
                acc += s2r[vv] * do_[vv];
            }
            dqh[kk] = acc;
        }
        // S₂ = S₁ + k̂ ⊗ δ: d k̂ = dS · δ; d δ = dSᵀ k̂
        let mut ddelta = [0f32; D];
        for kk in 0..D {
            let row = &ds[kk * D..(kk + 1) * D];
            let mut acc = 0f32;
            for vv in 0..D {
                acc += row[vv] * delta[vv];
                ddelta[vv] += row[vv] * kh[kk];
            }
            dkh[kk] = acc;
        }
        // δ = (v − mem) · β
        let mut db = 0f32;
        let dv = &mut dconv[t * CONV_DIM + 2 * KDA_QKV + h * D..t * CONV_DIM + 2 * KDA_QKV + (h + 1) * D];
        for vv in 0..D {
            dv[vv] = ddelta[vv] * b;
            dmem[vv] = -ddelta[vv] * b;
            db += ddelta[vv] * (vh[vv] - mr[vv]);
        }
        dbeta[t * KDA_HEADS + h] = db * b * (1.0 - b);
        // mem = S₁ᵀ k̂, S₁ = diag(e) S: dS₁ += k̂ ⊗ dmem; d k̂ += S₁ · dmem; then d e and dS = diag(e) dS₁
        for kk in 0..D {
            let row = &mut ds[kk * D..(kk + 1) * D];
            let sp = &s_prev[kk * D..(kk + 1) * D];
            let (k, ek) = (kh[kk], e[kk]);
            let (mut acc, mut de) = (0f32, 0f32);
            for vv in 0..D {
                row[vv] += k * dmem[vv];
                acc += ek * sp[vv] * dmem[vv];
                de += row[vv] * sp[vv];
                row[vv] *= ek;
            }
            dkh[kk] += acc;
            // e = exp(LOWER · σ(z)), z = decay · (f + dt_bias)
            let z = decay * (sv.f[t * KDA_QKV + h * D + kk] + w.dt_bias[h * D + kk]);
            let sg = sigmoid(z);
            df[t * KDA_QKV + h * D + kk] = de * ek * LINEAR_GATE_LOWER_BOUND * sg * (1.0 - sg) * decay;
        }
        // q̂ = l2(q) · scale, k̂ = l2(k), on the convolution's output
        for v in dqh.iter_mut() {
            *v *= scale;
        }
        l2norm_backward(&conv[h * D..(h + 1) * D], &dqh, &mut dconv[t * CONV_DIM + h * D..t * CONV_DIM + (h + 1) * D]);
        l2norm_backward(&conv[KDA_QKV + h * D..KDA_QKV + (h + 1) * D], &dkh, &mut dconv[t * CONV_DIM + KDA_QKV + h * D..t * CONV_DIM + KDA_QKV + (h + 1) * D]);
    }
}

/// Given `dy: [n][H]` and the gradient at the state after the run, accumulate `dxn: [n][H]`
/// and the gradient at the state before it (`dwindow_in`, `dmem_in`), all `+=`.
#[allow(clippy::too_many_arguments)]
pub fn backward(w: &weights::Kda, sv: &Saved, dy: &[f32], dwindow_out: &[f32], dmem_out: &[f32], dxn: &mut [f32], dwindow_in: &mut [f32], dmem_in: &mut [f32]) {
    let n = sv.n;
    // d normed = Woᵀ · dy
    let mut dnormed = vec![0f32; n * KDA_QKV];
    for t in 0..n {
        bf16::dense_t::<KDA_QKV>(H, w.o, &dy[t * H..(t + 1) * H], &mut dnormed[t * KDA_QKV..(t + 1) * KDA_QKV]);
    }
    // the heads
    let mut dconv = vec![0f32; n * CONV_DIM];
    let mut df = vec![0f32; n * KDA_QKV];
    let mut dbeta = vec![0f32; n * KDA_HEADS];
    let mut dgate = vec![0f32; n * KDA_QKV];
    let mut states = vec![0f32; (n + 1) * KDA_DIM * KDA_DIM];
    let mut ds = vec![0f32; KDA_DIM * KDA_DIM];
    for h in 0..KDA_HEADS {
        ds.copy_from_slice(&dmem_out[h * KDA_DIM * KDA_DIM..(h + 1) * KDA_DIM * KDA_DIM]);
        head_backward(w, h, sv, &dnormed, &mut ds, &mut states, &mut dconv, &mut df, &mut dbeta, &mut dgate);
        for (a, b) in dmem_in[h * KDA_DIM * KDA_DIM..(h + 1) * KDA_DIM * KDA_DIM].iter_mut().zip(&ds) {
            *a += b;
        }
    }
    // the convolution over the mixed timeline (the entering window, then the run); tap j reads
    // TAPS−1−j positions back; the out window is the timeline's last TAPS−1 entries
    let mut dtl = vec![0f32; (CONV_TAPS - 1 + n) * CONV_DIM];
    dtl[n * CONV_DIM..].copy_from_slice(&dwindow_out[..KDA_WINDOW]);
    let mut tl = sv.window_in.clone();
    tl.extend_from_slice(&sv.mixed);
    for t in 0..n {
        for part in 0..3 {
            let taps = w.conv[part];
            for cc in 0..KDA_QKV {
                let c = part * KDA_QKV + cc;
                let mut acc = 0f32;
                for j in 0..CONV_TAPS {
                    acc += common::gemv::widen(taps[cc * CONV_TAPS + j]) * tl[(t + j) * CONV_DIM + c];
                }
                let da = dconv[t * CONV_DIM + c] * swiglu::dsilu(acc);
                for j in 0..CONV_TAPS {
                    dtl[(t + j) * CONV_DIM + c] += da * common::gemv::widen(taps[cc * CONV_TAPS + j]);
                }
            }
        }
    }
    for (a, b) in dwindow_in[..KDA_WINDOW].iter_mut().zip(&dtl[..KDA_WINDOW]) {
        *a += b;
    }
    // the projections
    let mut dfa = vec![0f32; KDA_DIM];
    let mut dga = vec![0f32; KDA_DIM];
    for t in 0..n {
        let dm = &dtl[(CONV_TAPS - 1 + t) * CONV_DIM..(CONV_TAPS + t) * CONV_DIM];
        let dx = &mut dxn[t * H..(t + 1) * H];
        bf16::dense_t::<H>(KDA_QKV, w.q, &dm[..KDA_QKV], dx);
        bf16::dense_t::<H>(KDA_QKV, w.k, &dm[KDA_QKV..2 * KDA_QKV], dx);
        bf16::dense_t::<H>(KDA_QKV, w.v, &dm[2 * KDA_QKV..], dx);
        dfa.iter_mut().for_each(|v| *v = 0.0);
        bf16::dense_t::<KDA_DIM>(KDA_QKV, w.f_b, &df[t * KDA_QKV..(t + 1) * KDA_QKV], &mut dfa);
        bf16::dense_t::<H>(KDA_DIM, w.f_a, &dfa, dx);
        bf16::dense_t::<H>(KDA_HEADS, w.b, &dbeta[t * KDA_HEADS..(t + 1) * KDA_HEADS], dx);
        dga.iter_mut().for_each(|v| *v = 0.0);
        bf16::dense_t::<KDA_DIM>(KDA_QKV, w.g_b, &dgate[t * KDA_QKV..(t + 1) * KDA_QKV], &mut dga);
        bf16::dense_t::<H>(KDA_DIM, w.g_a, &dga, dx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use common::gemv::widen;
    use common::testing::Rng;

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(worst <= tol * scale, "{what}: worst |Δ| {worst} against a scale of {scale}");
    }

    struct W {
        q: Vec<u16>,
        k: Vec<u16>,
        v: Vec<u16>,
        conv: [Vec<u16>; 3],
        f_a: Vec<u16>,
        f_b: Vec<u16>,
        dt_bias: Vec<f32>,
        a_log: Vec<f32>,
        b: Vec<u16>,
        g_a: Vec<u16>,
        g_b: Vec<u16>,
        o_norm: Vec<f32>,
        o: Vec<u16>,
    }
    fn bf(m: usize, s: f32, rnd: &mut Rng) -> Vec<u16> {
        (0..m).map(|_| ((rnd.unit() * s).to_bits() >> 16) as u16).collect()
    }
    fn wide(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&b| widen(b)).collect()
    }

    /// The run's forward (the inference step) and backward against the dense reference at the real shape.
    #[test]
    fn matches_dense_reference() {
        let mut rnd = Rng::new(0x4DA0_0001);
        let n = 3;
        // weights at a scale that keeps the sigmoids and the memory in their working range
        let w = W { q: bf(KDA_QKV * H, 0.03, &mut rnd), k: bf(KDA_QKV * H, 0.03, &mut rnd), v: bf(KDA_QKV * H, 0.03, &mut rnd), conv: [bf(KDA_QKV * CONV_TAPS, 0.5, &mut rnd), bf(KDA_QKV * CONV_TAPS, 0.5, &mut rnd), bf(KDA_QKV * CONV_TAPS, 0.5, &mut rnd)], f_a: bf(KDA_DIM * H, 0.03, &mut rnd), f_b: bf(KDA_QKV * KDA_DIM, 0.1, &mut rnd), dt_bias: (0..KDA_QKV).map(|_| rnd.unit()).collect(), a_log: (0..KDA_HEADS).map(|_| rnd.unit() * 0.5).collect(), b: bf(KDA_HEADS * H, 0.03, &mut rnd), g_a: bf(KDA_DIM * H, 0.03, &mut rnd), g_b: bf(KDA_QKV * KDA_DIM, 0.1, &mut rnd), o_norm: (0..KDA_DIM).map(|_| 1.0 + rnd.unit() * 0.3).collect(), o: bf(H * KDA_QKV, 0.03, &mut rnd) };
        let wk = weights::Kda { q: &w.q, k: &w.k, v: &w.v, conv: [&w.conv[0], &w.conv[1], &w.conv[2]], f_a: &w.f_a, f_b: &w.f_b, dt_bias: &w.dt_bias, a_log: &w.a_log, b: &w.b, g_a: &w.g_a, g_b: &w.g_b, o_norm: &w.o_norm, o: &w.o };
        let xn: Vec<f32> = (0..n * H).map(|_| rnd.unit()).collect();
        let window0: Vec<f32> = (0..KDA_WINDOW).map(|_| rnd.unit()).collect();
        let mem0: Vec<f32> = (0..KDA_MEMORY).map(|_| rnd.unit() * 0.3).collect();
        let dy: Vec<f32> = (0..n * H).map(|_| rnd.unit()).collect();
        let dwindow_out: Vec<f32> = (0..KDA_WINDOW).map(|_| rnd.unit()).collect();
        let dmem_out: Vec<f32> = (0..KDA_MEMORY).map(|_| rnd.unit()).collect();

        let (mut window, mut mem) = (window0.clone(), mem0.clone());
        let mut s = Scratch::new();
        let mut saved = Saved::new(n);
        let mut y = vec![0f32; n * H];
        forward(0, &wk, &xn, n, &mut window, &mut mem, &mut s, &mut saved, &mut y);
        let (mut dxn, mut dwin, mut dmem) = (vec![0f32; n * H], vec![0f32; KDA_WINDOW], vec![0f32; KDA_MEMORY]);
        backward(&wk, &saved, &dy, &dwindow_out, &dmem_out, &mut dxn, &mut dwin, &mut dmem);

        // the reference: the conv taps as one [CONV_DIM][TAPS]
        let (q, k, v, f_a, f_b, b, g_a, g_b, o) = (wide(&w.q), wide(&w.k), wide(&w.v), wide(&w.f_a), wide(&w.f_b), wide(&w.b), wide(&w.g_a), wide(&w.g_b), wide(&w.o));
        let conv: Vec<f32> = [wide(&w.conv[0]), wide(&w.conv[1]), wide(&w.conv[2])].concat();
        let r = reference::KdaRef { nh: KDA_HEADS, d: KDA_DIM, h: H, taps: CONV_TAPS, lower: LINEAR_GATE_LOWER_BOUND, rms_eps: RMS_NORM_EPS, q: &q, k: &k, v: &v, conv: &conv, f_a: &f_a, f_b: &f_b, dt_bias: &w.dt_bias, a_log: &w.a_log, b: &b, g_a: &g_a, g_b: &g_b, o_norm: &w.o_norm, o: &o };
        let st0 = reference::KdaState { window: window0, mem: mem0 };
        let (ry, rsv, rst) = reference::kda_forward(&r, &xn, n, &st0);
        close(&y, &ry, 1e-4, "y");
        close(&window, &rst.window, 1e-4, "window after");
        close(&mem, &rst.mem, 1e-4, "mem after");
        let (rdx, rdst) = reference::kda_backward(&r, &xn, &rsv, &dy, &reference::KdaState { window: dwindow_out, mem: dmem_out });
        close(&dxn, &rdx, 1e-4, "dxn");
        close(&dwin, &rdst.window, 1e-4, "dwindow");
        close(&dmem, &rdst.mem, 1e-4, "dmem");
    }
}
