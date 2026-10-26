//! `analog-dump columns --ids <file> --dir <forward's dir> --out <dir> [--pairs 3-43] [--variants ...]`
//!
//! For each MoE pair l → l + 1, layer l + 1's router logits (`[n][E]` f32, `<out>/<column>/l<l>.f32`) over:
//! - `ra`: layer l's FFN input (the route-ahead, today's first margin);
//! - `early`: that plus l's shared output (the early plan's second margin, outside the hyper-connection as served);
//! - `rows`: that plus l's routed output (the rows plan's second margin);
//! - `exact`: layer l + 1's own FFN input (its router's real logits: the picks are their top 8 with the bias);
//! - where l + 1 is a KDA layer, the analogs `<source>-<weights>`: layer l's FFN output stood in by `shared` (l's
//!   shared output alone, ready before the slot) or `full` (shared + routed, at the slot) through l's FFN mix, then
//!   layer l + 1's attention with its q, k, v and o projections at `f32`, `q8`, `q4`, `q2` or `q1` (round to nearest
//!   over groups of 128 inputs; q1 = sign × the group's mean |w|), from the real recurrent state at the start of
//!   each step of `--step` tokens and carried through the step's own tokens, then its mixes, norm and router.
//!   `full-f32` is `exact` (the sanity check: equal up to the saved streams' bf16). `full-none`, `shared-none` (every
//!   layer): no attention at all, l + 1's FFN input from the stood-in streams through its mixes alone.

use crate::*;
use common::swiglu::silu;

const GROUP: usize = 128;
const CHUNK: usize = 64;

/// `hc.pre` of every position (`st: [n][HC][H]`), in parallel: the mixes and the collapsed inputs `[n][H]`.
pub fn pre_all(hc: &Hc, st: &[f32], n: usize) -> (Vec<Mix>, Vec<f32>) {
    let parts = par_map(n.div_ceil(CHUNK), |c| {
        let (t0, t1) = (c * CHUNK, (c * CHUNK + CHUNK).min(n));
        let mut x = vec![0f32; (t1 - t0) * H];
        let m: Vec<Mix> = (t0..t1).map(|t| hc.pre(&st[t * HC * H..(t + 1) * HC * H], &mut x[(t - t0) * H..(t - t0 + 1) * H])).collect();
        (m, x)
    });
    let (mut mixes, mut x) = (Vec::with_capacity(n), Vec::with_capacity(n * H));
    for (m, v) in parts {
        mixes.extend(m);
        x.extend(v);
    }
    (mixes, x)
}

/// `mix.post` of every position: the streams `[n][HC][H]` after adding `y: [n][H]` to `st`.
pub fn post_all(mixes: &[Mix], y: &[f32], st: &[f32], n: usize) -> Vec<f32> {
    par_map(n.div_ceil(CHUNK), |c| {
        let (t0, t1) = (c * CHUNK, (c * CHUNK + CHUNK).min(n));
        let mut out = vec![0f32; (t1 - t0) * HC * H];
        for t in t0..t1 {
            mixes[t].post(&y[t * H..(t + 1) * H], &st[t * HC * H..(t + 1) * HC * H], &mut out[(t - t0) * HC * H..(t - t0 + 1) * HC * H]);
        }
        out
    })
    .concat()
}

pub fn run() {
    let ids = ids();
    let n = ids.len();
    let dir = arg_required("--dir");
    let out = arg_required("--out");
    let step: usize = arg("--step").map_or(3, |v| v.to_str().unwrap().parse().unwrap());
    let (p0, p1) = arg("--pairs").map_or((FIRST_K_DENSE_REPLACE, NUM_HIDDEN_LAYERS - 2), |v| {
        let s = v.to_str().unwrap().to_string();
        let (a, b) = s.split_once('-').unwrap();
        (a.parse().unwrap(), b.parse().unwrap())
    });
    let variants: Vec<String> = arg("--variants").map_or("shared-q4,shared-q2,shared-q1,full-q2".into(), |v| v.to_string_lossy().into_owned()).split(',').map(String::from).collect();
    // the analogs only from this position (a multiple of the step); their logits before it are zero
    let from: usize = arg("--from").map_or(0, |v| v.to_str().unwrap().parse().unwrap());
    assert!(from % step == 0 && from < n);
    let ck = ckpt();
    let write = |col: &str, l: usize, v: &[f32]| {
        let d = out.join(col);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(format!("l{l}.f32")), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    };
    for l in p0..=p1 {
        let t0 = std::time::Instant::now();
        let (a, b) = (load(l, &|name| ck.f32s(name)), load(l + 1, &|name| ck.f32s(name)));
        let s = load_saved(&dir, l, n);
        let Ffn::Moe(moe) = &b.ffn else { unreachable!() };
        let route = |x: &[f32]| mm(&moe.router, E, H, &rmsnorm(x, H, &b.post_attention_layernorm, RMS_NORM_EPS), n);
        // layer l's FFN input, collapsed, and its mix
        let (mixes, xl) = pre_all(&a.ffn_hc, &s.mid, n);
        write("ra", l, &route(&xl));
        let early: Vec<f32> = (0..n * H).map(|i| xl[i] + s.shared[i]).collect();
        write("early", l, &route(&early));
        let rows: Vec<f32> = (0..n * H).map(|i| early[i] + s.routed[i]).collect();
        write("rows", l, &route(&rows));
        drop((early, rows));
        // the streams entering l + 1 with layer l's FFN output stood in by `y`; l + 1's attention input over them
        let enter = |y: &[f32]| post_all(&mixes, y, &s.mid, n);
        let full: Vec<f32> = (0..n * H).map(|i| s.shared[i] + s.routed[i]).collect();
        let st_full = enter(&full);
        drop(full);
        let st_shared = enter(&s.shared);
        // after l + 1's attention output `o` on streams `st`: its FFN input, routed
        let finish = |st: &[f32], o: &[f32]| -> Vec<f32> {
            let (m, _) = pre_all(&b.attn_hc, st, n);
            route(&pre_all(&b.ffn_hc, &post_all(&m, o, st, n), n).1)
        };
        let attn_in = |st: &[f32]| -> Vec<f32> { rmsnorm(&pre_all(&b.attn_hc, st, n).1, H, &b.input_layernorm, RMS_NORM_EPS) };
        // without l + 1's attention at all: its FFN input from the stood-in streams alone
        let zero = vec![0f32; n * H];
        write("full-none", l, &finish(&st_full, &zero));
        write("shared-none", l, &finish(&st_shared, &zero));
        drop(zero);
        let x_full = attn_in(&st_full);
        match &b.attn {
            Attn::Kda(k) => {
                let real = project(k, &x_full, n, None);
                // the exact column: l + 1's real attention (the recurrence on the real inputs throughout)
                let normed_real = recur(k, &real, None, n, step);
                let o = mm(&k.o, H, KDA_QKV, &normed_real, n);
                write("exact", l, &finish(&st_full, &o));
                let x_shared = attn_in(&st_shared);
                for v in &variants {
                    let (src, wq) = v.split_once('-').unwrap();
                    let (st, x) = match src {
                        "full" => (&st_full, &x_full),
                        "shared" => (&st_shared, &x_shared),
                        _ => panic!("source {src}"),
                    };
                    // `pc<r>`: the full weights over the input projected onto its top r principal directions (from the
                    // positions before `from`: deployable as Uᵀ `[r][H]` shared by q, k, v and W·U `[KDA_QKV][r]` each)
                    // `h[d][b]<K>[q<bits>|pc<r>]`: K heads (d: chosen by their measured push on the router over the
                    // positions before `from`; b: K/2 from each card's half), then rounded or over the principal input
                    let (hw, rest) = match wq.strip_prefix('h') {
                        Some(r) => {
                            let i = r.find(|c: char| c.is_ascii_digit()).unwrap();
                            let j = r[i..].find(|c: char| !c.is_ascii_digit()).map_or(r.len(), |j| i + j);
                            (Some(&r[..j]), &r[j..])
                        }
                        None => (None, wq),
                    };
                    let xin = match rest.strip_prefix("pc") {
                        Some(r) => principal(&x[..from * H], from, r.parse().unwrap(), &x[from * H..], n - from),
                        None => x[from * H..].to_vec(),
                    };
                    let bits = if rest.starts_with("pc") { "" } else { rest };
                    let q = (hw.is_some() || (!bits.is_empty() && bits != "f32")).then(|| weights(k, hw, bits, &moe.router, &b.post_attention_layernorm, &normed_real[..from * KDA_QKV]));
                    let pv = project(k, &xin, n - from, q.as_ref());
                    let normed = recur(k, &real, Some((&pv, from)), n, step);
                    let mut o = vec![0f32; from * H];
                    o.extend(mm(q.as_ref().map_or(&k.o, |q| &q.o), H, KDA_QKV, &normed[from * KDA_QKV..], n - from));
                    let write = |name: &str, mut z: Vec<f32>| {
                        z[..from * E].fill(0.0);
                        write(name, l, &z)
                    };
                    write(v, finish(st, &o));
                    // the router composed into the output side (deployable as R·diag(ln)·O, `[E][KDA_QKV]`): the FFN
                    // mix's coefficients from the streams without the attention's output; `-rc` with the exact norm,
                    // `-rn` with the norm of the input without it
                    let (ma, _) = pre_all(&b.attn_hc, st, n);
                    let st0 = post_all(&ma, &vec![0f32; n * H], st, n);
                    let (_, x0) = pre_all(&b.ffn_hc, &st0, n);
                    let c: Vec<f32> = par_map(n, |t| {
                        let flat = rmsnorm(&st0[t * HC * H..(t + 1) * HC * H], HC * H, &[], RMS_NORM_EPS);
                        (0..HC).map(|i| {
                            let ci = crate::fast::dot8(&b.ffn_hc.fn_[i * HC * H..(i + 1) * HC * H], &flat);
                            (sigmoid(ci * b.ffn_hc.scale[0] + b.ffn_hc.base[i]) + HC_EPS) * ma[t].post[i]
                        }).sum()
                    });
                    let xc: Vec<f32> = (0..n * H).map(|i| x0[i] + c[i / H] * o[i]).collect();
                    write(&format!("{v}-rc"), route(&xc));
                    let mut y = vec![0f32; n * H];
                    for t in 0..n {
                        let r = 1.0 / (x0[t * H..(t + 1) * H].iter().map(|v| v * v).sum::<f32>() / H as f32 + RMS_NORM_EPS).sqrt();
                        for d in 0..H {
                            y[t * H + d] = xc[t * H + d] * r * b.post_attention_layernorm[d];
                        }
                    }
                    write(&format!("{v}-rn"), mm(&moe.router, E, H, &y, n));
                }
            }
            Attn::Mla(m) => {
                let mut ms = MlaState::default();
                let o = crate::fast::mla(m, &x_full, n, &mut ms);
                write("exact", l, &finish(&st_full, &o));
            }
        }
        println!("pair {l}: {:.1} s", t0.elapsed().as_secs_f32());
    }
}

/// The analog's q, k, v and o: `q4`/`q2`/`q1` rounded; `h<K>` the real weights of only the K heads whose output
/// matters most to the router (by ‖R·diag(ln)·O_h‖, weights alone: no extra VRAM, the layer's own rows); `h<K>q<b>` both.
fn weights(k: &Kda, hw: Option<&str>, bits: &str, router: &[f32], ln: &[f32], cal: &[f32]) -> Quant {
    let (heads, data, balanced) = match hw {
        Some(r) => (r.trim_start_matches(|c: char| !c.is_ascii_digit()).parse::<usize>().unwrap(), r.starts_with('d') || r.starts_with("bd"), r.contains('b')),
        None => (KDA_HEADS, false, false),
    };
    let mut w = Quant { q: k.q.clone(), k: k.k.clone(), v: k.v.clone(), o: k.o.clone() };
    if heads < KDA_HEADS {
        // C = R·diag(ln)·O, `[E][KDA_QKV]`; a head's importance its block's squared norm
        let rl: Vec<f32> = (0..E * H).map(|i| router[i] * ln[i % H]).collect();
        let mut ot = vec![0f32; KDA_QKV * H];
        for d in 0..H {
            for c in 0..KDA_QKV {
                ot[c * H + d] = k.o[d * KDA_QKV + c];
            }
        }
        let c = mm(&ot, KDA_QKV, H, &rl, E);
        let mut imp: Vec<(f32, usize)> = if data {
            // the head's push on the router's logits over the calibration positions: Σ_t ‖C_h · u_{t,h}‖²
            let m = cal.len() / KDA_QKV;
            par_map(KDA_HEADS, |h| {
                let ch: Vec<f32> = (0..E).flat_map(|e| c[e * KDA_QKV + h * KDA_DIM..e * KDA_QKV + (h + 1) * KDA_DIM].iter().copied()).collect::<Vec<f32>>();
                let uh: Vec<f32> = (0..m).flat_map(|t| cal[t * KDA_QKV + h * KDA_DIM..t * KDA_QKV + (h + 1) * KDA_DIM].iter().copied()).collect::<Vec<f32>>();
                let mut acc = 0f32;
                for t in 0..m {
                    for e in 0..E {
                        let d: f32 = (0..KDA_DIM).map(|i| ch[e * KDA_DIM + i] * uh[t * KDA_DIM + i]).sum();
                        acc += d * d;
                    }
                }
                (acc, h)
            })
        } else {
            (0..KDA_HEADS).map(|h| ((0..E).map(|e| c[e * KDA_QKV + h * KDA_DIM..e * KDA_QKV + (h + 1) * KDA_DIM].iter().map(|v| v * v).sum::<f32>()).sum(), h)).collect()
        };
        imp.sort_by(|a, b| b.0.total_cmp(&a.0));
        let keep: Vec<usize> = if balanced {
            let half = KDA_HEADS / 2;
            let mut v: Vec<usize> = imp.iter().filter(|x| x.1 < half).take(heads / 2).map(|x| x.1).collect();
            v.extend(imp.iter().filter(|x| x.1 >= half).take(heads / 2).map(|x| x.1));
            v
        } else {
            imp[..heads].iter().map(|x| x.1).collect()
        };
        for h in (0..KDA_HEADS).filter(|h| !keep.contains(h)) {
            for m in [&mut w.q, &mut w.k, &mut w.v] {
                m[h * KDA_DIM * H..(h + 1) * KDA_DIM * H].fill(0.0);
            }
            for d in 0..H {
                w.o[d * KDA_QKV + h * KDA_DIM..d * KDA_QKV + (h + 1) * KDA_DIM].fill(0.0);
            }
        }
    }
    if !bits.is_empty() && bits != "f32" {
        w = Quant { q: quant(&w.q, KDA_QKV, H, bits), k: quant(&w.k, KDA_QKV, H, bits), v: quant(&w.v, KDA_QKV, H, bits), o: quant(&w.o, H, KDA_QKV, bits) };
    }
    w
}

/// `x: [n][H]` projected onto the top `r` eigenvectors of the covariance of `cal: [m][H]` (subspace iteration).
fn principal(cal: &[f32], m: usize, r: usize, x: &[f32], n: usize) -> Vec<f32> {
    let mut ct = vec![0f32; H * m];
    for t in 0..m {
        for d in 0..H {
            ct[d * m + t] = cal[t * H + d];
        }
    }
    let c = mm(&ct, H, m, &ct, H); // [H][H]
    drop(ct);
    // rows of `u` are the basis; a fixed pseudo-random start
    let mut u: Vec<f32> = (0..r * H).map(|i| ((i as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 40) as f32 / (1u64 << 24) as f32 - 0.5).collect();
    let orth = |u: &mut [f32]| {
        for i in 0..r {
            for j in 0..i {
                let d: f32 = (0..H).map(|e| u[i * H + e] * u[j * H + e]).sum();
                for e in 0..H {
                    u[i * H + e] -= d * u[j * H + e];
                }
            }
            let nrm = (0..H).map(|e| u[i * H + e] * u[i * H + e]).sum::<f32>().sqrt().max(1e-30);
            u[i * H..(i + 1) * H].iter_mut().for_each(|v| *v /= nrm);
        }
    };
    orth(&mut u);
    for _ in 0..12 {
        u = mm(&c, H, H, &u, r);
        orth(&mut u);
    }
    let a = mm(&u, r, H, x, n); // [n][r]
    let mut ut = vec![0f32; H * r];
    for i in 0..r {
        for d in 0..H {
            ut[d * r + i] = u[i * H + d];
        }
    }
    mm(&ut, H, r, &a, n)
}

pub struct Quant {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    o: Vec<f32>,
}

/// `w: [m][k]` rounded to `how` over groups of `GROUP` along k, back to f32.
fn quant(w: &[f32], m: usize, k: usize, how: &str) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    let rows = par_map(m, |r| {
        let mut out = vec![0f32; k];
        for g in (0..k).step_by(GROUP) {
            let (src, dst) = (&w[r * k + g..r * k + g + GROUP], &mut out[g..g + GROUP]);
            let amax = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
            match how {
                "q8" | "q4" => {
                    let top = if how == "q8" { 127.0 } else { 7.0 };
                    let s = amax / top;
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = if s > 0.0 { (v / s).round().clamp(-top - 1.0, top) * s } else { 0.0 };
                    }
                }
                "q2" => {
                    // levels ±s/2, ±3s/2; s the best of a few against the group's squared error
                    let mut best = (f32::MAX, 0f32);
                    for f in [0.3f32, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] {
                        let s = amax * f;
                        let e: f32 = src.iter().map(|&v| {
                            let q = ((v / s - 0.5).round().clamp(-2.0, 1.0) + 0.5) * s;
                            (q - v) * (q - v)
                        }).sum();
                        if e < best.0 {
                            best = (e, s);
                        }
                    }
                    let s = best.1;
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = if s > 0.0 { ((v / s - 0.5).round().clamp(-2.0, 1.0) + 0.5) * s } else { 0.0 };
                    }
                }
                "q1" => {
                    let s = src.iter().map(|v| v.abs()).sum::<f32>() / GROUP as f32;
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = if v >= 0.0 { s } else { -s };
                    }
                }
                _ => panic!("weights {how}"),
            }
        }
        out
    });
    rows.concat()
}

/// A KDA layer's projections over `x: [n][H]`: q, k, v `[n][KDA_QKV]` (before the convolution), the forget gate's
/// input `[n][KDA_QKV]`, beta `[n][KDA_HEADS]`, the output gate `[n][KDA_QKV]`.
pub struct Proj {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    f: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
}

pub fn project(a: &Kda, x: &[f32], n: usize, q: Option<&Quant>) -> Proj {
    let (wq, wk, wv) = q.map_or((&a.q, &a.k, &a.v), |q| (&q.q, &q.k, &q.v));
    Proj {
        q: mm(wq, KDA_QKV, H, x, n),
        k: mm(wk, KDA_QKV, H, x, n),
        v: mm(wv, KDA_QKV, H, x, n),
        f: mm(&a.f_b, KDA_QKV, KDA_DIM, &mm(&a.f_a, KDA_DIM, H, x, n), n),
        beta: mm(&a.b, KDA_HEADS, H, x, n),
        gate: mm(&a.g_b, KDA_QKV, KDA_DIM, &mm(&a.g_a, KDA_DIM, H, x, n), n),
    }
}

/// One head's state: the convolution's last inputs of its q, k, v channels `[TAPS-1][3][KDA_DIM]` and its memory.
#[derive(Clone)]
struct Head {
    window: Vec<f32>,
    s: Vec<f32>,
}

fn l2norm(x: &mut [f32]) {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
    x.iter_mut().for_each(|v| *v *= r);
}

/// Token `t` through head `h` (reference.rs `Kda::forward`'s arithmetic), its gated normed output into `out`.
fn token(a: &Kda, p: &Proj, h: usize, t: usize, st: &mut Head, out: &mut [f32]) {
    const D: usize = KDA_DIM;
    let mut c = [[0f32; D]; 3];
    for (j, src) in [&p.q, &p.k, &p.v].into_iter().enumerate() {
        for d in 0..D {
            let ch = j * KDA_QKV + h * D + d;
            let w = &a.conv[ch * CONV_TAPS..(ch + 1) * CONV_TAPS];
            let x = src[t * KDA_QKV + h * D + d];
            let mut acc = w[CONV_TAPS - 1] * x;
            for tap in 0..CONV_TAPS - 1 {
                acc += w[tap] * st.window[(tap * 3 + j) * D + d];
            }
            c[j][d] = silu(acc);
            for tap in 0..CONV_TAPS - 2 {
                st.window[(tap * 3 + j) * D + d] = st.window[((tap + 1) * 3 + j) * D + d];
            }
            st.window[((CONV_TAPS - 2) * 3 + j) * D + d] = x;
        }
    }
    let [mut qh, mut kh, vh] = c;
    l2norm(&mut qh);
    l2norm(&mut kh);
    let scale = 1.0 / (D as f32).sqrt();
    qh.iter_mut().for_each(|v| *v *= scale);
    let b = sigmoid(p.beta[t * KDA_HEADS + h]);
    let decay = a.a_log[h].exp();
    let ft = &p.f[t * KDA_QKV + h * D..t * KDA_QKV + (h + 1) * D];
    let s = &mut st.s;
    for kk in 0..D {
        let e = (LINEAR_GATE_LOWER_BOUND * sigmoid(decay * (ft[kk] + a.dt_bias[h * D + kk]))).exp();
        s[kk * D..(kk + 1) * D].iter_mut().for_each(|v| *v *= e);
    }
    let mut mem = [0f32; D];
    for kk in 0..D {
        for vv in 0..D {
            mem[vv] += s[kk * D + vv] * kh[kk];
        }
    }
    let delta: [f32; D] = core::array::from_fn(|vv| (vh[vv] - mem[vv]) * b);
    for kk in 0..D {
        for vv in 0..D {
            s[kk * D + vv] += kh[kk] * delta[vv];
        }
    }
    let mut o = [0f32; D];
    for kk in 0..D {
        for vv in 0..D {
            o[vv] += s[kk * D + vv] * qh[kk];
        }
    }
    let on = rmsnorm(&o, D, &a.o_norm, RMS_NORM_EPS);
    let gt = &p.gate[t * KDA_QKV + h * D..t * KDA_QKV + (h + 1) * D];
    for vv in 0..D {
        out[vv] = on[vv] * sigmoid(gt[vv]);
    }
}

/// The heads' gated outputs `[n][KDA_QKV]`: of `real` throughout when `analog` is None; else of `analog` (its
/// projections from position `from` on, indexed from there), each step of `step` tokens from `from` on from the real
/// state at the step's start and carried through the step's tokens (zero before `from`).
pub fn recur(a: &Kda, real: &Proj, analog: Option<(&Proj, usize)>, n: usize, step: usize) -> Vec<f32> {
    let heads = par_map(KDA_HEADS, |h| {
        let mut st = Head { window: vec![0.0; (CONV_TAPS - 1) * 3 * KDA_DIM], s: vec![0.0; KDA_DIM * KDA_DIM] };
        let mut out = vec![0f32; n * KDA_DIM];
        let mut scratch = [0f32; KDA_DIM];
        for t0 in (0..n).step_by(step) {
            let t1 = (t0 + step).min(n);
            if let Some((p, from)) = analog.filter(|&(_, from)| t0 >= from) {
                let mut sa = st.clone();
                for t in t0..t1 {
                    token(a, p, h, t - from, &mut sa, &mut out[t * KDA_DIM..(t + 1) * KDA_DIM]);
                }
            }
            for t in t0..t1 {
                let o = if analog.is_some() { &mut scratch[..] } else { &mut out[t * KDA_DIM..(t + 1) * KDA_DIM] };
                token(a, real, h, t, &mut st, o);
            }
        }
        out
    });
    let mut normed = vec![0f32; n * KDA_QKV];
    for (h, o) in heads.iter().enumerate() {
        for t in 0..n {
            normed[t * KDA_QKV + h * KDA_DIM..t * KDA_QKV + (h + 1) * KDA_DIM].copy_from_slice(&o[t * KDA_DIM..(t + 1) * KDA_DIM]);
        }
    }
    normed
}

/// `y[s][m] = Σ_k w[m][k] · x[s][k]` (reference.rs `matmul`'s contract) through a blocked GEMM (matrixmultiply, its
/// own threading caps at 4) over a grid of position × row tiles, so each thread packs only its tile's share of both
/// operands (a split over rows alone packed all of `x` in every thread: 47 % of the time): this tool's columns are not
/// a reference, and the reference's serial dot ran at ~35 GFLOP/s here.
pub fn mm(w: &[f32], m: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
    assert!(w.len() == m * k && x.len() == n * k);
    let mut y = vec![0f32; n * m];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    // about two tiles a thread, split along both sides in proportion to their lengths
    let tiles = 2 * threads;
    let tn = ((tiles as f64 * n as f64 / m as f64).sqrt().round() as usize).clamp(1, n.div_ceil(4).max(1));
    let tm = tiles.div_ceil(tn).clamp(1, m.div_ceil(16).max(1));
    let (pn, pm) = (n.div_ceil(tn), m.div_ceil(tm));
    let yp = y.as_mut_ptr() as usize;
    par_map(tn * tm, |i| {
        let (s0, m0) = ((i / tm) * pn, (i % tm) * pm);
        let (s1, m1) = ((s0 + pn).min(n), (m0 + pm).min(m));
        if s1 > s0 && m1 > m0 {
            // y[s0..s1][m0..m1] = x[s0..s1] · w[m0..m1]ᵀ; the tiles are disjoint
            unsafe { matrixmultiply::sgemm(s1 - s0, k, m1 - m0, 1.0, x.as_ptr().add(s0 * k), k as isize, 1, w.as_ptr().add(m0 * k), 1, k as isize, 0.0, (yp as *mut f32).add(s0 * m + m0), m as isize, 1) };
        }
    });
    y
}
