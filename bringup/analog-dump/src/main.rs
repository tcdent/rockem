//! analog-dump — the data of the expert-predictor analog study (docs/queue/098).
//!
//!   analog-dump forward --ids <file> --out <dir> [--ckpt ~/hf-cache/glm-5.3-flash] [--bundle bundles/glm-5.3-flash-q4rtn/glm-5.3-flash]
//!               [--positions N] [--layers N] [--resume L]
//!       The f32 reference's arithmetic (models/glm-5.3-flash/src/reference.rs) with vectorized products, layer-major over the teacher-forced ids, the dense
//!       weights from the checkpoint and the routed experts from the bundle's INT4 (dequantized through f32, the
//!       oracle's q4 recipe). Per MoE layer l, `<dir>/l<l>.bin`, `[n]` rows of: the streams after l's attention
//!       (`[HC][H]` bf16, the FFN input's streams), l's shared output and routed output (`[H]` bf16 each), l's picks
//!       (`[TOP_K]` u16, ascending).
//!   analog-dump columns ... (the predictor columns over the saved layers; see `columns`)

use glm_5_3_flash::reference::*;
use glm_5_3_flash::*;
use harness::{arg, arg_required, Checkpoint};
use std::io::Write;
use std::path::{Path, PathBuf};
use common::swiglu::silu;

mod columns;
mod fast;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("forward") => forward(),
        Some("columns") => columns::run(),
        _ => panic!("analog-dump forward|columns ..."),
    }
}

pub fn ckpt() -> Checkpoint {
    Checkpoint::open(&arg("--ckpt").unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("hf-cache/glm-5.3-flash")))
}

pub fn ids() -> Vec<usize> {
    let f = arg_required("--ids");
    let ids: Vec<usize> = std::fs::read_to_string(&f).unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
    let positions = arg("--positions").map_or(usize::MAX, |v| v.to_str().unwrap().parse().unwrap());
    ids[..ids.len().min(positions)].to_vec()
}

pub fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

pub fn widen(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// One MoE layer's saved rows, `n` of them.
pub struct Saved {
    pub mid: Vec<f32>,    // [n][HC][H]
    pub shared: Vec<f32>, // [n][H]
    pub routed: Vec<f32>, // [n][H]
    pub picks: Vec<u16>,  // [n][TOP_K]
}

pub const ROW: usize = (HC * H + 2 * H) * 2 + TOP_K * 2;

pub fn load_saved(dir: &Path, l: usize, n: usize) -> Saved {
    let b = std::fs::read(dir.join(format!("l{l}.bin"))).unwrap();
    assert!(b.len() >= n * ROW, "l{l}.bin holds {} rows", b.len() / ROW);
    let u = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let mut s = Saved { mid: vec![0.0; n * HC * H], shared: vec![0.0; n * H], routed: vec![0.0; n * H], picks: vec![0; n * TOP_K] };
    for t in 0..n {
        let o = t * ROW;
        for i in 0..HC * H {
            s.mid[t * HC * H + i] = widen(u(o + 2 * i));
        }
        let o = o + HC * H * 2;
        for i in 0..H {
            s.shared[t * H + i] = widen(u(o + 2 * i));
            s.routed[t * H + i] = widen(u(o + 2 * (H + i)));
        }
        let o = o + 4 * H;
        for k in 0..TOP_K {
            s.picks[t * TOP_K + k] = u(o + 2 * k);
        }
    }
    s
}

fn forward() {
    let ids = ids();
    let n = ids.len();
    let out = arg_required("--out");
    std::fs::create_dir_all(&out).unwrap();
    let ck = ckpt();
    let bundle = arg("--bundle").unwrap_or_else(|| PathBuf::from("/home/slomp/Work/rockem/bundles/glm-5.3-flash-q4rtn/glm-5.3-flash"));
    let f = std::fs::File::open(bundle.join(layout::EXPERTS_FILE)).unwrap();
    let experts: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&f).unwrap() }));
    experts.advise(memmap2::Advice::Sequential).unwrap();
    let stream: &'static expert::Stream<'static> = Box::leak(Box::new(expert::Stream::file(&experts[..])));
    println!("{n} positions → {}", out.display());

    let emb = ck.bf16("model.language_model.embed_tokens.weight");
    let mut streams = vec![0f32; n * HC * H];
    for (t, &id) in ids.iter().enumerate() {
        for j in 0..HC {
            for d in 0..H {
                streams[(t * HC + j) * H + d] = widen(emb[id * H + d]);
            }
        }
    }
    let layers: usize = arg("--layers").map_or(NUM_HIDDEN_LAYERS, |v| v.to_str().unwrap().parse().unwrap());
    // --resume L: the streams entering layer L rebuilt from layer L - 1's saved rows (its FFN-input streams, shared and
    // routed outputs, bf16) through its FFN mix, and the layers from L on
    let first: usize = arg("--resume").map_or(0, |v| v.to_str().unwrap().parse().unwrap());
    if first > 0 {
        let prev = load(first - 1, &|name| ck.f32s(name));
        let s = load_saved(&out, first - 1, n);
        let (mixes, _) = columns::pre_all(&prev.ffn_hc, &s.mid, n);
        let y: Vec<f32> = s.shared.iter().zip(&s.routed).map(|(a, b)| a + b).collect();
        streams = columns::post_all(&mixes, &y, &s.mid, n);
    }
    for l in first..layers {
        let t0 = std::time::Instant::now();
        let layer = load(l, &|name| ck.f32s(name));
        let mut st = layer.state();
        let saved = layer_forward(&layer, &mut streams, n, &mut st, l, stream);
        if let Some((mid, shared, routed, idx)) = saved {
            let mut w = std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(out.join(format!("l{l}.bin"))).unwrap());
            let mut row = Vec::with_capacity(ROW);
            for t in 0..n {
                row.clear();
                row.extend(mid[t * HC * H..(t + 1) * HC * H].iter().flat_map(|&v| bf16(v).to_le_bytes()));
                row.extend(shared[t * H..(t + 1) * H].iter().flat_map(|&v| bf16(v).to_le_bytes()));
                row.extend(routed[t * H..(t + 1) * H].iter().flat_map(|&v| bf16(v).to_le_bytes()));
                row.extend(idx[t * TOP_K..(t + 1) * TOP_K].iter().flat_map(|&e| (e as u16).to_le_bytes()));
                w.write_all(&row).unwrap();
            }
            w.flush().unwrap();
        }
        println!("layer {l}: {:.1} s", t0.elapsed().as_secs_f32());
    }
}

/// `Layer::forward` (reference.rs) with this tool's vectorized products (`columns::mm`; the reference's serial dot
/// ran ~5 min a layer over 18k positions), keeping a MoE layer's FFN-input streams, shared and routed outputs and
/// routing.
fn layer_forward(ly: &Layer, streams: &mut [f32], n: usize, st: &mut LayerState, l: usize, stream: &expert::Stream) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>)> {
    use columns::{mm, project, recur};
    use columns::{post_all, pre_all};
    let (mixes, x) = pre_all(&ly.attn_hc, streams, n);
    let x = rmsnorm(&x, H, &ly.input_layernorm, RMS_NORM_EPS);
    let y = match (&ly.attn, st) {
        (Attn::Kda(a), LayerState::Kda(_)) => mm(&a.o, H, KDA_QKV, &recur(a, &project(a, &x, n, None), None, n, n), n),
        (Attn::Mla(a), LayerState::Mla(c)) => fast::mla(a, &x, n, c),
        _ => unreachable!(),
    };
    let s2 = post_all(&mixes, &y, streams, n);
    streams.copy_from_slice(&s2);
    let (mixes, x) = pre_all(&ly.ffn_hc, streams, n);
    let x = rmsnorm(&x, H, &ly.post_attention_layernorm, RMS_NORM_EPS);
    let mlp = |m: &Mlp, x: &[f32], n: usize| -> Vec<f32> {
        let (g, u) = (mm(&m.gate, m.ff, H, x, n), mm(&m.up, m.ff, H, x, n));
        let a: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu(g.min(SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)).collect();
        mm(&m.down, H, m.ff, &a, n)
    };
    let (y, saved) = match &ly.ffn {
        Ffn::Dense(m) => (mlp(m, &x, n), None),
        Ffn::Moe(m) => {
            // Moe::route's selection over this tool's logits
            let logits = mm(&m.router, E, H, &x, n);
            let mut idx = vec![0usize; n * TOP_K];
            let mut wt = vec![0f32; n * TOP_K];
            for t in 0..n {
                let s: Vec<f32> = logits[t * E..(t + 1) * E].iter().map(|&v| sigmoid(v)).collect();
                let mut order: Vec<usize> = (0..E).collect();
                order.sort_by(|&a, &b| (s[b] + m.bias[b]).total_cmp(&(s[a] + m.bias[a])));
                let mut pick = order[..TOP_K].to_vec();
                pick.sort();
                let sum: f32 = pick.iter().map(|&e| s[e]).sum::<f32>() + 1e-20;
                for (k, &e) in pick.iter().enumerate() {
                    idx[t * TOP_K + k] = e;
                    wt[t * TOP_K + k] = s[e] / sum * ROUTED_SCALING_FACTOR;
                }
            }
            // the touched experts dequantized a batch at a time in parallel, each run over its positions
            let mut routed = vec![0f32; n * H];
            let touched: Vec<usize> = (0..E).filter(|e| idx.contains(e)).collect();
            for batch in touched.chunks(32) {
                let ws = par_map(batch.len(), |i| dequant_expert(stream, l, batch[i]));
                for (&e, (gu, down)) in batch.iter().zip(&ws) {
                    let slots: Vec<usize> = (0..n * TOP_K).filter(|&i| idx[i] == e).collect();
                    let xs: Vec<f32> = slots.iter().flat_map(|&i| x[(i / TOP_K) * H..(i / TOP_K + 1) * H].iter().copied()).collect();
                    let g = mm(gu, 2 * D_FF, H, &xs, slots.len());
                    let mut a = vec![0f32; slots.len() * D_FF];
                    for (r, row) in g.chunks_exact(2 * D_FF).enumerate() {
                        for i in 0..D_FF {
                            a[r * D_FF + i] = silu(row[i].min(SWIGLU_LIMIT)) * row[D_FF + i].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
                        }
                    }
                    let yv = mm(down, H, D_FF, &a, slots.len());
                    for (r, &slot) in slots.iter().enumerate() {
                        let t = slot / TOP_K;
                        for d in 0..H {
                            routed[t * H + d] += wt[slot] * yv[r * H + d];
                        }
                    }
                }
            }
            let shared = mlp(&m.shared, &x, n);
            (routed.iter().zip(&shared).map(|(a, b)| a + b).collect::<Vec<f32>>(), Some((streams.to_vec(), shared, routed, idx)))
        }
    };
    let s2 = post_all(&mixes, &y, streams, n);
    streams.copy_from_slice(&s2);
    saved
}

/// One `q4g128` record's weights back to f32 (forward-glm's `dequant_expert`): integer × scale.
fn dequant_expert(stream: &expert::Stream, l: usize, e: usize) -> (Vec<f32>, Vec<f32>) {
    use common::q4::{self, GROUP};
    use expert::{Weights, DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS};
    let w = Weights::at(stream, l, e);
    let expand = |rows: usize, k: usize, iw: &[u8], is: &[f32], r_base: usize, out: &mut [f32]| {
        for r in 0..rows {
            for kk in 0..k {
                out[(r_base + r) * k + kk] = q4::unpack(k, iw, r, kk) as f32 * is[q4::scale_index(k, r, kk / GROUP)];
            }
        }
    };
    let mut gu = vec![0f32; 2 * D_FF * H];
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        expand(GU_ROWS, H, gw, gs, c * GU_ROWS, &mut gu);
        expand(GU_ROWS, H, uw, us, D_FF + c * GU_ROWS, &mut gu);
    }
    let mut down = vec![0f32; H * D_FF];
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        expand(DOWN_ROWS, D_FF, dw, ds, c * DOWN_ROWS, &mut down);
    }
    (gu, down)
}
