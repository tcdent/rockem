//! bench-exponents — what a lossless code of the KDA projections' bf16 weights can save (029): per projection, over every
//! KDA layer of the bundle's `dense.bin`, the distinct exponents, the exponent's entropy (the bits a per-matrix Huffman
//! code of it would take — DFloat11's figure is sign + mantissa + this), and for a fixed-length code the share of weights
//! whose exponent sits within 2^b − 1 values of its group's largest (`b` 3 and 4 bits, groups of 128 along k: what a
//! group-max-relative code covers without an escape). Read-only; the dev host.
//!
//! Then the format itself (`common::exp4`): every matrix encoded and decoded back, compared with the bundle's bf16
//! bit for bit, with its bytes against bf16's and the share of groups it stores whole.
//!
//!   bench-exponents --bundle <bundle dir>
use glm_5_3_flash::weights::{Attn, Weights};
use harness::arg;

fn main() {
    let dir = arg("--bundle").expect("--bundle").join("glm-5.3-flash");
    let f = std::fs::File::open(dir.join(glm_5_3_flash::layout::DENSE_FILE)).unwrap();
    let dense = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let w = Weights::from_dense(&dense);
    // per projection name: exponent histogram, and counts within the b-bit window of the group max
    let mut stats: std::collections::BTreeMap<&str, ([u64; 256], u64, [u64; 2], u64)> = Default::default();
    let mut layers = 0;
    for layer in &w.layers {
        let Attn::Kda(k) = &layer.attn else { continue };
        layers += 1;
        let k_h = glm_5_3_flash::H;
        let mats: [(&str, &[u16], usize); 8] = [("q", k.q, k_h), ("k", k.k, k_h), ("v", k.v, k_h), ("f_a", k.f_a, k_h), ("b", k.b, k_h), ("g_a", k.g_a, k_h), ("f_b+g_b", k.f_b, glm_5_3_flash::KDA_DIM), ("o", k.o, glm_5_3_flash::KDA_QKV)];
        for (name, m, kk) in mats {
            let e = stats.entry(name).or_insert(([0; 256], 0, [0; 2], 0));
            for row in m.chunks_exact(kk) {
                for g in row.chunks(128) {
                    let exps: Vec<u8> = g.iter().map(|&v| ((v >> 7) & 0xFF) as u8).collect();
                    let mx = *exps.iter().max().unwrap();
                    for &x in &exps {
                        e.0[x as usize] += 1;
                        // a zero exponent (zero or subnormal) needs no window: one code for it
                        for (bi, b) in [3u32, 4].iter().enumerate() {
                            if x == 0 || (mx as u32) - (x as u32) < (1 << b) - 1 {
                                e.2[bi] += 1;
                            }
                        }
                    }
                    e.1 += exps.len() as u64;
                }
            }
            e.3 += m.len() as u64 * 2;
        }
        // g_b beside f_b: the same shape, counted under one name
        let e = stats.get_mut("f_b+g_b").unwrap();
        for row in k.g_b.chunks_exact(glm_5_3_flash::KDA_DIM) {
            for g in row.chunks(128) {
                let exps: Vec<u8> = g.iter().map(|&v| ((v >> 7) & 0xFF) as u8).collect();
                let mx = *exps.iter().max().unwrap();
                for &x in &exps {
                    e.0[x as usize] += 1;
                    for (bi, b) in [3u32, 4].iter().enumerate() {
                        if x == 0 || (mx as u32) - (x as u32) < (1 << b) - 1 {
                            e.2[bi] += 1;
                        }
                    }
                }
                e.1 += exps.len() as u64;
            }
        }
        e.3 += k.g_b.len() as u64 * 2;
    }
    println!("bench-exponents: {layers} KDA layers of {}", dir.display());
    println!("{:>8} {:>9} {:>9} {:>10} {:>11} {:>14} {:>14}", "matrix", "MB", "distinct", "H(exp) bit", "Huffman b/w", "in 3-bit win %", "in 4-bit win %");
    let (mut tb, mut th, mut tw3, mut tw4, mut tn) = (0f64, 0f64, 0u64, 0u64, 0u64);
    for (name, (hist, n, win, bytes)) in &stats {
        let tot = *n as f64;
        let distinct = hist.iter().filter(|&&c| c > 0).count();
        let h: f64 = hist.iter().filter(|&&c| c > 0).map(|&c| { let p = c as f64 / tot; -p * p.log2() }).sum();
        println!("{name:>8} {:>9.1} {distinct:>9} {h:>10.3} {:>11.2} {:>14.4} {:>14.4}", *bytes as f64 / 1e6, 8.0 + h, 100.0 * win[0] as f64 / tot, 100.0 * win[1] as f64 / tot);
        tb += *bytes as f64;
        th += h * tot;
        tw3 += win[0];
        tw4 += win[1];
        tn += n;
    }
    // the format: every KDA projection of every layer round-tripped
    let (mut bf, mut ex, mut groups, mut slots, mut mats, mut wrong) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for layer in &w.layers {
        let Attn::Kda(k) = &layer.attn else { continue };
        let (h, d, q) = (glm_5_3_flash::H, glm_5_3_flash::KDA_DIM, glm_5_3_flash::KDA_QKV);
        for (m, kk) in [(k.q, h), (k.k, h), (k.v, h), (k.f_a, h), (k.b, h), (k.g_a, h), (k.f_b, d), (k.g_b, d), (k.o, q)] {
            let rows = m.len() / kk;
            let e = common::exp4::encode(m, rows, kk);
            if common::exp4::decode(&e, rows, kk) != m {
                wrong += 1;
            }
            bf += 2 * m.len();
            ex += common::exp4::bytes(rows, kk, e.slots());
            groups += e.head.len();
            slots += e.slots();
            mats += 1;
        }
    }
    println!("exp4: {mats} matrices round-tripped, {wrong} differ from the bundle's bf16; {:.3} GB against {:.3} GB bf16 ({:.1} %, {:.2} bits a weight); {slots} of {groups} groups stored whole ({:.3} %)", ex as f64 / 1e9, bf as f64 / 1e9, 100.0 * ex as f64 / bf as f64, 16.0 * ex as f64 / bf as f64, 100.0 * slots as f64 / groups as f64);
    let n = tn as f64;
    // the zero code's worth: exponent-0 weights (zeros, subnormals), and the groups a 16-binade window with no zero
    // code would store whole (every exponent within 16 of the group's largest, zeros included)
    let zeros: u64 = stats.values().map(|v| v.0[0]).sum();
    let mut g16 = (0usize, 0usize);
    for layer in &w.layers {
        let Attn::Kda(k) = &layer.attn else { continue };
        let (h, d, q) = (glm_5_3_flash::H, glm_5_3_flash::KDA_DIM, glm_5_3_flash::KDA_QKV);
        for (m, kk) in [(k.q, h), (k.k, h), (k.v, h), (k.f_a, h), (k.b, h), (k.g_a, h), (k.f_b, d), (k.g_b, d), (k.o, q)] {
            for row in m.chunks_exact(kk) {
                for gr in row.chunks(128) {
                    let ex: Vec<u8> = gr.iter().map(|&v| ((v >> 7) & 0xFF) as u8).collect();
                    let mx = *ex.iter().max().unwrap();
                    g16.1 += 1;
                    if ex.iter().any(|&x| mx - x >= 16) {
                        g16.0 += 1;
                    }
                }
            }
        }
    }
    println!("exponent-0 weights: {zeros} of {tn} ({:.5} %); a 16-binade window with no zero code stores {} of {} groups whole ({:.3} %)", 100.0 * zeros as f64 / n, g16.0, g16.1, 100.0 * g16.0 as f64 / g16.1 as f64);
    println!("all KDA projections: {:.2} GB bf16; Huffman exponent {:.3} bits → {:.2} bits a weight ({:.1} % of bf16); 3-bit window covers {:.4} %, 4-bit {:.4} %", tb / 1e9, th / n, 8.0 + th / n, 100.0 * (8.0 + th / n) / 16.0, 100.0 * tw3 as f64 / n, 100.0 * tw4 as f64 / n);
}
