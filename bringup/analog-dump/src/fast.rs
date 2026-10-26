//! reference.rs's MLA attention with this tool's vectorized products (`columns::mm`, `dot8`): the reference's serial
//! dot made an MLA layer the bulk of a long passage (its indexer scores every visible pool for every position).

use crate::columns::mm;
use glm_5_3_flash::reference::*;
use glm_5_3_flash::*;

/// A dot product over eight lanes (the reference's is serial, so it does not vectorize).
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

fn select(p: usize, q: &[f32], w: &[f32], pools: &[f32]) -> Vec<usize> {
    let visible = (p + 1) / INDEX_KPOOL;
    let keep = INDEX_TOPK / INDEX_KPOOL;
    let mut chosen: Vec<usize> = if visible <= keep {
        (0..visible).collect()
    } else {
        let scale = 1.0 / (INDEX_HEAD_DIM as f32).sqrt();
        let mut scored: Vec<(f32, usize)> = (0..visible)
            .map(|j| {
                let key = &pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM];
                let mut sc = 0f32;
                for h in 0..INDEX_N_HEADS {
                    sc += w[h] * (dot8(&q[h * INDEX_HEAD_DIM..(h + 1) * INDEX_HEAD_DIM], key) * scale).max(0.0);
                }
                (sc, j)
            })
            .collect();
        scored.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        scored[..keep].iter().map(|&(_, j)| j).collect()
    };
    chosen.sort();
    let mut out: Vec<usize> = chosen.iter().flat_map(|&j| j * INDEX_KPOOL..(j + 1) * INDEX_KPOOL).collect();
    out.extend(visible * INDEX_KPOOL..=p);
    out
}

fn pools(ix: &Indexer, st: &MlaState, total: usize) -> Vec<f32> {
    let d = INDEX_HEAD_DIM;
    let mut out = vec![0f32; total / INDEX_KPOOL * d];
    for j in 0..total / INDEX_KPOOL {
        for c in 0..d {
            let logit: [f32; INDEX_KPOOL] = core::array::from_fn(|i| st.igate[(j * INDEX_KPOOL + i) * d + c] + ix.ape[i * d + c]);
            let mx = logit.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let e: [f32; INDEX_KPOOL] = core::array::from_fn(|i| (logit[i] - mx).exp());
            let sum: f32 = e.iter().sum();
            out[j * d + c] = (0..INDEX_KPOOL).map(|i| e[i] / sum * st.ikey[(j * INDEX_KPOOL + i) * d + c]).sum();
        }
    }
    out
}

/// `Mla::forward`.
pub fn mla(a: &Mla, x: &[f32], n: usize, st: &mut MlaState) -> Vec<f32> {
    let qr = rmsnorm(&mm(&a.q_a, Q_LORA_RANK, H, x, n), Q_LORA_RANK, &a.q_a_ln, RMS_NORM_EPS);
    let q = mm(&a.q_b, MLA_HEADS * Q_HEAD, Q_LORA_RANK, &qr, n);
    let past = st.latent.len() / KV_LORA_RANK;
    st.latent.extend(rmsnorm(&mm(&a.kv_a, KV_LORA_RANK, H, x, n), KV_LORA_RANK, &a.kv_a_ln, RMS_NORM_EPS));
    let ix = &a.indexer;
    let k = mm(&ix.wk, INDEX_HEAD_DIM, H, x, n);
    for r in k.chunks_exact(INDEX_HEAD_DIM) {
        let mean = r.iter().sum::<f32>() / INDEX_HEAD_DIM as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / INDEX_HEAD_DIM as f32;
        let rs = 1.0 / (var + 1e-6).sqrt();
        st.ikey.extend((0..INDEX_HEAD_DIM).map(|c| (r[c] - mean) * rs * ix.k_norm_w[c] + ix.k_norm_b[c]));
    }
    st.igate.extend(mm(&ix.gate, INDEX_HEAD_DIM, H, x, n));
    let iq = mm(&ix.wq_b, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK, &qr, n);
    let wscale = 1.0 / (INDEX_N_HEADS as f32).sqrt();
    let iw: Vec<f32> = mm(&ix.weights, INDEX_N_HEADS, H, x, n).iter().map(|v| v * wscale).collect();
    let total = past + n;
    let pl = pools(ix, st, total);
    let sel: Vec<Vec<usize>> = par_map(n, |t| select(past + t, &iq[t * INDEX_N_HEADS * INDEX_HEAD_DIM..(t + 1) * INDEX_N_HEADS * INDEX_HEAD_DIM], &iw[t * INDEX_N_HEADS..(t + 1) * INDEX_N_HEADS], &pl));
    // the absorbed form: a head's score q·(W_k latent) = (W_kᵀ q)·latent and its output W_v·(Σ p latent), so the
    // positions are read once a query as gathered latents instead of an expanded `[total][heads][KVB_HEAD]` cache
    // (whose strided reads ran ~4 GFLOP/s); equal to the reference's up to rounding
    const L: usize = KV_LORA_RANK;
    let mut qa = vec![0f32; n * MLA_HEADS * L]; // [n][heads][L]
    for h in 0..MLA_HEADS {
        let wk = &a.kv_b[h * KVB_HEAD * L..(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * L];
        // qa[:, h, :] = q[:, h, :256] · wk [256][L]
        unsafe { matrixmultiply::sgemm(n, QK_NOPE_HEAD_DIM, L, 1.0, q.as_ptr().add(h * Q_HEAD), (MLA_HEADS * Q_HEAD) as isize, 1, wk.as_ptr(), L as isize, 1, 0.0, qa.as_mut_ptr().add(h * L), (MLA_HEADS * L) as isize, 1) };
    }
    let scale = 1.0 / (Q_HEAD as f32).sqrt();
    let lat = &st.latent;
    let ctx = par_map(n, |t| {
        let js = &sel[t];
        let m = js.len();
        let mut g = vec![0f32; m * L];
        for (i, &j) in js.iter().enumerate() {
            g[i * L..(i + 1) * L].copy_from_slice(&lat[j * L..(j + 1) * L]);
        }
        let mut sc = vec![0f32; MLA_HEADS * m];
        unsafe { matrixmultiply::sgemm(MLA_HEADS, L, m, scale, qa.as_ptr().add(t * MLA_HEADS * L), L as isize, 1, g.as_ptr(), 1, L as isize, 0.0, sc.as_mut_ptr(), m as isize, 1) };
        for r in sc.chunks_exact_mut(m) {
            let mx = r.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
            let mut sum = 0f32;
            for v in r.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            r.iter_mut().for_each(|v| *v /= sum);
        }
        let mut c = vec![0f32; MLA_HEADS * L];
        unsafe { matrixmultiply::sgemm(MLA_HEADS, m, L, 1.0, sc.as_ptr(), m as isize, 1, g.as_ptr(), L as isize, 1, 0.0, c.as_mut_ptr(), L as isize, 1) };
        c
    })
    .concat(); // [n][heads][L]
    let mut rows = vec![0f32; n * MLA_HEADS * V_HEAD_DIM];
    for h in 0..MLA_HEADS {
        let wv = &a.kv_b[(h * KVB_HEAD + QK_NOPE_HEAD_DIM) * L..(h + 1) * KVB_HEAD * L]; // [256][L]
        unsafe { matrixmultiply::sgemm(n, L, V_HEAD_DIM, 1.0, ctx.as_ptr().add(h * L), (MLA_HEADS * L) as isize, 1, wv.as_ptr(), 1, L as isize, 0.0, rows.as_mut_ptr().add(h * V_HEAD_DIM), (MLA_HEADS * V_HEAD_DIM) as isize, 1) };
    }
    mm(&a.o, H, MLA_HEADS * V_HEAD_DIM, &rows, n)
}
