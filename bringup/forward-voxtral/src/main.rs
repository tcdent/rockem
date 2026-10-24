//! forward-voxtral — the shipped host decoder (`voxtral_small_24b::decoder`) over an oracle set, diffed
//! against the goldens (bringup/voxtral/oracle.py): the oracle's `embed` rows are the decoder's inputs at
//! every position (the audio rows included, so the encoder is out of the picture), the positions go through
//! in steps of `STEP`, and every layer's residual output and the logits are diffed. The decoder is INT4
//! with INT8 activations against an f32 reference on bf16 values, so the numbers to read are the layer
//! outputs' relative error growth and the logits' top-1 agreement over the transcript — the recipe's cost,
//! with a kernel error standing out as a jump.
//!
//!   forward-voxtral --goldens <bundles/goldens-voxtral/SET> --bundle <bundle dir> [--positions N] [--tol 1e-2]
//!                   [--serial]   the serial pool (one core), to separate the pool from the arithmetic
//!                   [--encoder]  the encoder and projector over the set's windows (`windows.f32`) instead: the encoder's
//!                                output and the rows diffed against `enc_out` and `rows` (f32 on bf16 both sides: tight)
//!                   [--transcribe <dir>]  end to end: the dir's `windows.f32` through the encoder, its `prompt.json` ids
//!                                through the decoder with the rows at the placeholders, then greedy decoding to EOS or
//!                                --new tokens; the transcript printed through the bundle's vocab.bin (the user's test)

use glm_5_3_flash::construct::{physical_cores, Threads};
use harness::{arg, arg_required, flag, Golden, Report};
use voxtral_small_24b::decoder::{Decoder, STEP};
use voxtral_small_24b::encoder::Encoder;
use voxtral_small_24b::layout;
use voxtral_small_24b::weights::{Dense, Stream};
use voxtral_small_24b::*;

fn main() {
    let bundle = arg_required("--bundle").join("voxtral-small-24b");
    let tol: f32 = arg("--tol").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(1e-2);
    if let Some(dir) = arg("--transcribe") {
        return transcribe(&bundle, &dir, arg("--new").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(64));
    }
    let gdir = arg_required("--goldens");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let prompt_len = meta["prompt_len"].as_u64().unwrap() as usize;
    let n_all = arg("--positions").map(|p| p.to_str().unwrap().parse::<usize>().unwrap()).unwrap_or(ids.len()).min(ids.len());
    let gold = Golden::load(&gdir.join("edges.safetensors"));
    let t0 = std::time::Instant::now();
    let dense_b = std::fs::read(bundle.join(layout::DENSE_FILE)).unwrap();
    let stream_f = std::fs::File::open(bundle.join(layout::STREAM_FILE)).unwrap();
    let stream_m = unsafe { memmap2::Mmap::map(&stream_f).unwrap() };
    let (dense, stream) = (Dense::new(&dense_b), Stream::new(&stream_m));
    let serial = common::pool::Serial;
    let threads;
    let pool: &dyn common::pool::Pool = if flag("--serial") {
        &serial
    } else {
        threads = Threads::new(&physical_cores());
        &threads
    };
    if flag("--encoder") {
        return encoder_gate(dense, pool, &gdir, &gold, &meta, tol);
    }
    if let Some(snap) = arg("--snapshot") {
        // the stream against the checkpoint: layer 0's q rows 0..64 dequantized (INT4 · scale) vs the bf16 rows
        let ck = harness::Checkpoint::open(&snap);
        // the INT4 recipe's cost per matrix kind: the quantized rows' error against the bf16 rows, relative to the rows' rms
        let check = |name: &str, which: layout::Which, k: usize, rows: usize| {
            let g = ck.f32s(&format!("language_model.model.layers.0.{name}.weight"));
            let it = stream.item(0, which, 0);
            let (mut se, mut sw) = (0f64, 0f64);
            for r in 0..rows {
                for kk in 0..k {
                    let w = common::q4::unpack(k, it.w, r, kk) as f32 * it.s[common::q4::scale_index(k, r, kk / common::q4::GROUP)];
                    let x = g[r * k + kk];
                    se += ((w - x) * (w - x)) as f64;
                    sw += (x * x) as f64;
                }
            }
            println!("stream check: layer 0 {name} item 0 ({rows} rows of {k}): error rms / weight rms = {:.3}", (se / sw).sqrt());
        };
        check("self_attn.q_proj", layout::Which::Qkv, H, 64);
        check("mlp.gate_proj", layout::Which::Gu, H, 64);
        check("mlp.down_proj", layout::Which::Down, D_FF, 64);
        // layer 0 at position 0 in f32 from the checkpoint (attention over one position is its value: the layer's
        // arithmetic alone), against the oracle's l0.out: the formula check, apart from any quantization
        {
            let g = |n: &str| ck.f32s(&format!("language_model.model.layers.0.{n}.weight"));
            let x = gold.row("embed", 0).unwrap();
            let mut xn = vec![0f32; H];
            common::rmsnorm::rmsnorm::<H>(x, &g("input_layernorm"), RMS_NORM_EPS, &mut xn);
            let matvec = |w: &[f32], m: usize, k: usize, v: &[f32]| -> Vec<f32> { (0..m).map(|r| common::linalg::dot(&w[r * k..(r + 1) * k], &v[..k])).collect() };
            let v = matvec(&g("self_attn.v_proj"), KV_DIM, H, &xn);
            // every query head attends the one position: its value head (GQA: head h takes kv head h / GROUP)
            let mut attn = vec![0f32; Q_DIM];
            for h in 0..HEADS {
                attn[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&v[(h / KV_GROUP) * HEAD_DIM..(h / KV_GROUP + 1) * HEAD_DIM]);
            }
            let o = matvec(&g("self_attn.o_proj"), H, Q_DIM, &attn);
            let mut h1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
            common::rmsnorm::rmsnorm::<H>(&h1, &g("post_attention_layernorm"), RMS_NORM_EPS, &mut xn);
            let gate = matvec(&g("mlp.gate_proj"), D_FF, H, &xn);
            let up = matvec(&g("mlp.up_proj"), D_FF, H, &xn);
            let hid: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| common::swiglu::silu(*a) * b).collect();
            let down = matvec(&g("mlp.down_proj"), H, D_FF, &hid);
            for i in 0..H {
                h1[i] += down[i];
            }
            let (x_f, attn_f, o_f, hid_f, down_f) = (x.to_vec(), attn.clone(), o.clone(), hid.clone(), down.clone());
            // the same layer again through freshly read matrices, as a check to run after each block below: a
            // difference from h1 after a block means that block corrupted memory
            let again = || -> f32 {
                let (wv2, wo2, wg2, wu2, wd2) = (g("self_attn.v_proj"), g("self_attn.o_proj"), g("mlp.gate_proj"), g("mlp.up_proj"), g("mlp.down_proj"));
                let mut xn2 = vec![0f32; H];
                common::rmsnorm::rmsnorm::<H>(x, &g("input_layernorm"), RMS_NORM_EPS, &mut xn2);
                let v2 = matvec(&wv2, KV_DIM, H, &xn2);
                let mut attn2 = vec![0f32; Q_DIM];
                for h in 0..HEADS {
                    attn2[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&v2[(h / KV_GROUP) * HEAD_DIM..(h / KV_GROUP + 1) * HEAD_DIM]);
                }
                let o2 = matvec(&wo2, H, Q_DIM, &attn2);
                let mut h4: Vec<f32> = x.iter().zip(&o2).map(|(a, b)| a + b).collect();
                common::rmsnorm::rmsnorm::<H>(&h4, &g("post_attention_layernorm"), RMS_NORM_EPS, &mut xn2);
                let gate2 = matvec(&wg2, D_FF, H, &xn2);
                let up2 = matvec(&wu2, D_FF, H, &xn2);
                let hid2: Vec<f32> = gate2.iter().zip(&up2).map(|(a, b)| common::swiglu::silu(*a) * b).collect();
                let down2 = matvec(&wd2, H, D_FF, &hid2);
                for i in 0..H {
                    h4[i] += down2[i];
                }
                harness::diff::err(&h4, &h1).0
            };
            println!("repeat check before any kernel: {:.3e}", again());
            // the q4 kernels on item 0 of layer 0's q|k|v against the dequantized rows times the dequantized INT8 input
            {
                use common::q4::{self, Act, QAct, GROUP};
                let it = stream.item(0, layout::Which::Qkv, 0);
                let (mut q, mut sc, mut gs) = (vec![0i8; H], vec![0f32; H / GROUP], vec![0i32; H / GROUP]);
                QAct { q: &mut q, scale: &mut sc, gsum: &mut gs }.quantize(&xn);
                let act = Act { q: &q, scale: &sc, gsum: &gs };
                let xq: Vec<f32> = (0..H).map(|k| q[k] as f32 * sc[k / GROUP]).collect();
                let want: Vec<f32> = (0..64).map(|r| (0..H).map(|k| q4::unpack(H, it.w, r, k) as f32 * it.s[q4::scale_index(H, r, k / GROUP)] * xq[k]).sum()).collect();
                let mut got = vec![0f32; 64];
                q4::gemv::<H>(64, it.w, it.s, &act, &mut got);
                let mut got2 = vec![0f32; 2 * 64];
                let o2 = got2.as_mut_ptr();
                q4::gemm::<H>(64, it.w, it.s, 2, |_| act, |t| unsafe { o2.add(t * 64) });
                let e = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                println!("kernel check: q4::gemv vs dequantized f32: max |Δ| {:.3e} (|want| max {:.3e}); q4::gemm n=2 vs gemv: {:.3e} / {:.3e}", e(&got, &want), want.iter().fold(0f32, |m, v| m.max(v.abs())), e(&got2[..64], &got), e(&got2[64..], &got));
            }
            println!("repeat check after the kernel block: {:.3e}", again());
            // the same layer with the stream's INT4 rows dequantized in f32 (the recipe's cost at this position, apart
            // from the INT8 activations and the kernels): against the oracle, and against the decoder's own output
            {
                use common::q4::{self, GROUP};
                let deq = |which: layout::Which, rows: usize, k: usize, x: &[f32], upper: bool| -> Vec<f32> {
                    let m = layout::mat(which);
                    let mut out = vec![0f32; rows];
                    let per = if which == layout::Which::Gu { m.chunk / 2 } else { m.chunk };
                    for c in 0..m.items() {
                        let (w, sc) = if which == layout::Which::Gu {
                            let (gw, uw, gs, us) = stream.gu(0, c);
                            if upper { (uw, us) } else { (gw, gs) }
                        } else {
                            let it = stream.item(0, which, c);
                            (it.w, it.s)
                        };
                        for r in 0..per {
                            let mut acc = 0f32;
                            for kk in 0..k {
                                acc += q4::unpack(k, w, r, kk) as f32 * sc[q4::scale_index(k, r, kk / GROUP)] * x[kk];
                            }
                            out[c * per + r] = acc;
                        }
                    }
                    out
                };
                let t = std::time::Instant::now();
                let mut xn = vec![0f32; H];
                common::rmsnorm::rmsnorm::<H>(x, &g("input_layernorm"), RMS_NORM_EPS, &mut xn);
                let qkv = deq(layout::Which::Qkv, layout::QKV.rows, H, &xn, false);
                let mut attn = vec![0f32; Q_DIM];
                for h in 0..HEADS {
                    attn[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&qkv[Q_DIM + KV_DIM + (h / KV_GROUP) * HEAD_DIM..Q_DIM + KV_DIM + (h / KV_GROUP + 1) * HEAD_DIM]);
                }
                let o = deq(layout::Which::O, H, Q_DIM, &attn, false);
                let mut h2: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
                common::rmsnorm::rmsnorm::<H>(&h2, &g("post_attention_layernorm"), RMS_NORM_EPS, &mut xn);
                let gate = deq(layout::Which::Gu, D_FF, H, &xn, false);
                let up = deq(layout::Which::Gu, D_FF, H, &xn, true);
                let hid: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| common::swiglu::silu(*a) * b).collect();
                let down = deq(layout::Which::Down, H, D_FF, &hid, false);
                for i in 0..H {
                    h2[i] += down[i];
                }
                let (a1, _) = harness::diff::err(&h2, gold.row("l0.out", 0).unwrap());
                let (a2, _) = harness::diff::err(&h2, &h1);
                println!("recipe check ({:.0}s): layer 0 at position 0 with the stream's INT4 rows in f32: vs the oracle max |Δ| {a1:.3e}, vs f32 weights {a2:.3e}; mine[..4] {:?}", t.elapsed().as_secs_f32(), &h2[..4]);
                println!("repeat check after the recipe block: {:.3e}", again());
                // which matrix: the layer with one matrix INT4 and the rest f32, at positions 0 and 3 (a text token, an audio row)
                let gm = |n: &str| g(n);
                let (wq, wk, wv, wo, wg, wu, wd) = (gm("self_attn.q_proj"), gm("self_attn.k_proj"), gm("self_attn.v_proj"), gm("self_attn.o_proj"), gm("mlp.gate_proj"), gm("mlp.up_proj"), gm("mlp.down_proj"));
                let _ = (&wq, &wk);
                for pos in [0usize, 3] {
                    let x = gold.row("embed", pos).unwrap();
                    let gold_out = gold.row("l0.out", pos).unwrap();
                    let gmax = gold_out.iter().fold(0f32, |m, v| m.max(v.abs()));
                    for which in ["none", "qkv", "o", "gu", "down"] {
                        let mut xn = vec![0f32; H];
                        common::rmsnorm::rmsnorm::<H>(x, &g("input_layernorm"), RMS_NORM_EPS, &mut xn);
                        // one position: attention is the value (the q and k matrices do not matter here)
                        let v = if which == "qkv" { deq(layout::Which::Qkv, layout::QKV.rows, H, &xn, false)[Q_DIM + KV_DIM..].to_vec() } else { matvec(&wv, KV_DIM, H, &xn) };
                        let mut attn = vec![0f32; Q_DIM];
                        for h in 0..HEADS {
                            attn[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&v[(h / KV_GROUP) * HEAD_DIM..(h / KV_GROUP + 1) * HEAD_DIM]);
                        }
                        let o = if which == "o" { deq(layout::Which::O, H, Q_DIM, &attn, false) } else { matvec(&wo, H, Q_DIM, &attn) };
                        let mut h3: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
                        common::rmsnorm::rmsnorm::<H>(&h3, &g("post_attention_layernorm"), RMS_NORM_EPS, &mut xn);
                        let (gate, up) = if which == "gu" { (deq(layout::Which::Gu, D_FF, H, &xn, false), deq(layout::Which::Gu, D_FF, H, &xn, true)) } else { (matvec(&wg, D_FF, H, &xn), matvec(&wu, D_FF, H, &xn)) };
                        let hid: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| common::swiglu::silu(*a) * b).collect();
                        let hmax = hid.iter().fold(0f32, |m, v| m.max(v.abs()));
                        let down = if which == "down" { deq(layout::Which::Down, H, D_FF, &hid, false) } else { matvec(&wd, H, D_FF, &hid) };
                        for i in 0..H {
                            h3[i] += down[i];
                        }
                        if which == "none" && pos == 0 {
                            // stage by stage against the formula check's values, recomputed here the same way
                            let mut xn1 = vec![0f32; H];
                            common::rmsnorm::rmsnorm::<H>(x, &g("input_layernorm"), RMS_NORM_EPS, &mut xn1);
                            let v1 = matvec(&g("self_attn.v_proj"), KV_DIM, H, &xn1);
                            println!("    stage v: {:.3e} (wv rows {} vs {})", harness::diff::err(&v, &v1).0, wv.len() / H, KV_DIM);
                            let o1 = matvec(&g("self_attn.o_proj"), H, Q_DIM, &attn);
                            println!("    stage o: {:.3e}", harness::diff::err(&o, &o1).0);
                            let g1 = matvec(&g("mlp.gate_proj"), D_FF, H, &xn);
                            println!("    stage gate: {:.3e}; up: {:.3e}", harness::diff::err(&gate, &g1).0, harness::diff::err(&up, &matvec(&g("mlp.up_proj"), D_FF, H, &xn)).0);
                            println!("    stage down: {:.3e}", harness::diff::err(&down, &matvec(&g("mlp.down_proj"), H, D_FF, &hid)).0);
                            println!("    x vs the formula's x: {:.3e}", harness::diff::err(x, &x_f).0);
                            println!("    vs the formula's: attn {:.3e} o {:.3e} hid {:.3e} down {:.3e}; x+o+down vs h1 {:.3e}", harness::diff::err(&attn, &attn_f).0, harness::diff::err(&o, &o_f).0, harness::diff::err(&hid, &hid_f).0, harness::diff::err(&down, &down_f).0, harness::diff::err(&(0..H).map(|i| x[i] + o[i] + down[i]).collect::<Vec<f32>>(), &h1).0);
                        }
                        let (a, _) = harness::diff::err(&h3, gold_out);
                        let (b, _) = harness::diff::err(&h3, &h1);
                        let at = (0..H).max_by(|&i, &j| (h3[i] - gold_out[i]).abs().partial_cmp(&(h3[j] - gold_out[j]).abs()).unwrap()).unwrap();
                        println!("  pos {pos} INT4 {which:>4}: max |Δ| {a:.3e} against the output's max {gmax:.3e} (vs the f32 formula at pos 0: {b:.3e}; worst dim {at}: mine {:.4} gold {:.4}; hidden max {hmax:.2e})", h3[at], gold_out[at]);
                    }
                }
            }
            let (abs, rel) = harness::diff::err(&h1, gold.row("l0.out", 0).unwrap());
            println!("formula check: layer 0 at position 0 in f32 from the checkpoint vs the oracle: max |Δ| {abs:.3e}, rel {rel:.3e}; mine[..4] {:?} gold[..4] {:?}", &h1[..4], &gold.row("l0.out", 0).unwrap()[..4]);
        }
    }
    let mut dec = Decoder::new(dense, stream, pool, 1, n_all.max(64));
    dec.trace = Some(Vec::new());
    println!("bundle   : {} in {:.1}s; {} positions of {} (prompt {prompt_len}), transcript {:?}", bundle.display(), t0.elapsed().as_secs_f32(), n_all, ids.len(), meta["transcript"].as_str().unwrap_or(""));
    let mut report = Report::new(tol);
    let (mut agree, mut scored) = (0usize, 0usize);
    let mut pos = 0;
    while pos < n_all {
        let n = (n_all - pos).min(STEP);
        let t1 = std::time::Instant::now();
        let rows: Vec<&[f32]> = (0..n).map(|t| gold.row("embed", pos + t).unwrap()).collect();
        dec.step(&ids[pos..pos + n], pos, |t| Some(rows[t]), true);
        let tr = dec.trace.as_ref().unwrap();
        for t in 0..n {
            for l in 0..NUM_HIDDEN_LAYERS {
                let key = format!("l{l}.out");
                if let Some(g) = gold.row(&key, pos + t) {
                    let (abs, rel) = report.record(&key, &tr[(l * n + t) * H..(l * n + t + 1) * H], g, &format!("pos {}", pos + t));
                    if l == 0 && pos + t < 6 {
                        println!("  l0.out pos {}: max|Δ| {abs:.3e} rel {rel:.3e}; mine[..4] {:?} gold[..4] {:?}", pos + t, &tr[t * H..t * H + 4], &g[..4]);
                    }
                    if pos + t == 40 && (l % 5 == 4 || l == 0 || l >= 30) {
                        println!("  pos 40 l{l}.out: max|Δ| {abs:.3e} rel {rel:.3e} (gold max {:.3e})", g.iter().fold(0f32, |m, v| m.max(v.abs())));
                    }
                }
            }
            let mine = &dec.logits[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE];
            if let Some(g) = gold.row("logits", pos + t) {
                report.record("logits", mine, g, &format!("pos {}", pos + t));
                if pos + t + 1 >= prompt_len && pos + t + 1 < ids.len() {
                    // a transcript position: the id the reference wrote next, against our argmax
                    scored += 1;
                    agree += (common::softmax::argmax(mine) as u32 == ids[pos + t + 1]) as usize;
                }
            }
        }
        println!("step at {pos} ({n} tokens): {:.2}s", t1.elapsed().as_secs_f32());
        pos += n;
    }
    report.print();
    println!("top-1 over the transcript: {agree}/{scored}; the reference's own {:.3}", meta["top1_agreement"].as_f64().unwrap_or(0.0));
}

fn encoder_gate(dense: Dense, pool: &dyn common::pool::Pool, gdir: &std::path::Path, gold: &Golden, meta: &serde_json::Value, tol: f32) {
    let windows = meta["windows"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(gdir.join("windows.f32")).unwrap();
    assert_eq!(bytes.len(), windows * NUM_MEL_BINS * MEL_FRAMES * 4, "windows.f32");
    let mel: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let t0 = std::time::Instant::now();
    let mut enc = Encoder::new(dense, pool);
    enc.trace = Some(Vec::new());
    println!("encoder  : built in {:.1}s; {windows} windows", t0.elapsed().as_secs_f32());
    let mut report = Report::new(tol);
    let mut rows = vec![0f32; ROWS_PER_WINDOW * H];
    for w in 0..windows {
        let t1 = std::time::Instant::now();
        enc.encode(&mel[w * NUM_MEL_BINS * MEL_FRAMES..(w + 1) * NUM_MEL_BINS * MEL_FRAMES], &mut rows);
        let tr = enc.trace.as_ref().unwrap();
        for p in 0..ENC_POSITIONS {
            if let Some(g) = gold.row("enc_out", w * ENC_POSITIONS + p) {
                report.record("enc_out", &tr[p * ENC_H..(p + 1) * ENC_H], g, &format!("window {w} pos {p}"));
            }
        }
        for r in 0..ROWS_PER_WINDOW {
            if let Some(g) = gold.row("rows", w * ROWS_PER_WINDOW + r) {
                report.record("rows", &rows[r * H..(r + 1) * H], g, &format!("window {w} row {r}"));
            }
        }
        let ph: Vec<String> = voxtral_small_24b::encoder::PHASES.iter().zip(enc.phases).map(|(n, s)| format!("{n} {s:.2}")).collect();
        println!("window {w}: {:.2}s ({})", t1.elapsed().as_secs_f32(), ph.join(", "));
    }
    report.print();
}

/// The bundle's vocabulary: byte offsets then bytes, by id (`vocab.bin`).
struct Vocab {
    offs: Vec<u32>,
    bytes: Vec<u8>,
}
impl Vocab {
    fn load(path: &std::path::Path) -> Self {
        let b = std::fs::read(path).unwrap();
        let n = VOCAB_SIZE + 1;
        let offs = b[..4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Vocab { offs, bytes: b[4 * n..].to_vec() }
    }
    fn token(&self, id: usize) -> &[u8] {
        &self.bytes[self.offs[id] as usize..self.offs[id + 1] as usize]
    }
}

fn transcribe(bundle: &std::path::Path, dir: &std::path::Path, max_new: usize) {
    let prompt: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("prompt.json")).unwrap()).unwrap();
    let ids: Vec<u32> = prompt["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let windows = prompt["windows"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(dir.join("windows.f32")).unwrap();
    assert_eq!(bytes.len(), windows * NUM_MEL_BINS * MEL_FRAMES * 4, "windows.f32");
    let mel: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let dense_b = std::fs::read(bundle.join(layout::DENSE_FILE)).unwrap();
    let stream_f = std::fs::File::open(bundle.join(layout::STREAM_FILE)).unwrap();
    let stream_m = unsafe { memmap2::Mmap::map(&stream_f).unwrap() };
    let (dense, stream) = (Dense::new(&dense_b), Stream::new(&stream_m));
    let vocab = Vocab::load(&bundle.join(layout::VOCAB_FILE));
    let threads = Threads::new(&physical_cores());
    let pool: &dyn common::pool::Pool = &threads;
    let t0 = std::time::Instant::now();
    let mut enc = Encoder::new(dense, pool);
    let mut rows = vec![0f32; windows * ROWS_PER_WINDOW * H];
    for w in 0..windows {
        enc.encode(&mel[w * NUM_MEL_BINS * MEL_FRAMES..(w + 1) * NUM_MEL_BINS * MEL_FRAMES], &mut rows[w * ROWS_PER_WINDOW * H..(w + 1) * ROWS_PER_WINDOW * H]);
    }
    let t_enc = t0.elapsed();
    drop(enc);
    let mut dec = Decoder::new(dense, stream, pool, 1, ids.len() + max_new + 8);
    // the prompt in steps, the rows at the placeholders in order
    let t1 = std::time::Instant::now();
    let mut taken = 0usize;
    let mut pos = 0;
    while pos < ids.len() {
        let n = (ids.len() - pos).min(STEP);
        let mut step_rows: Vec<Option<&[f32]>> = Vec::with_capacity(n);
        for t in 0..n {
            if ids[pos + t] == AUDIO_ID {
                step_rows.push(Some(&rows[taken * H..(taken + 1) * H]));
                taken += 1;
            } else {
                step_rows.push(None);
            }
        }
        let sr = &step_rows;
        dec.step(&ids[pos..pos + n], pos, |t| sr[t], false);
        pos += n;
    }
    assert_eq!(taken, windows * ROWS_PER_WINDOW, "placeholders vs rows");
    let t_prefill = t1.elapsed();
    // greedy
    let t2 = std::time::Instant::now();
    let mut out = Vec::new();
    let mut next = common::softmax::argmax(&dec.logits[(ids.len() - 1) % STEP * VOCAB_SIZE..][..VOCAB_SIZE]) as u32;
    // the last step's logits row is the step's last token's: row n - 1 of that step
    let last_n = ids.len() - (ids.len() - 1) / STEP * STEP;
    next = common::softmax::argmax(&dec.logits[(last_n - 1) * VOCAB_SIZE..last_n * VOCAB_SIZE]) as u32;
    while out.len() < max_new && !EOS_TOKENS.contains(&next) {
        out.push(next);
        dec.step(&[next], pos, |_| None, false);
        pos += 1;
        next = common::softmax::argmax(&dec.logits[..VOCAB_SIZE]) as u32;
    }
    let text: Vec<u8> = out.iter().flat_map(|&id| vocab.token(id as usize).to_vec()).collect();
    println!("transcript ({} ids, EOS {}): {}", out.len(), EOS_TOKENS.contains(&next), String::from_utf8_lossy(&text));
    println!("encoder {:.1}s for {windows} window(s); prefill {} ids {:.1}s; {} tokens at {:.0} ms each", t_enc.as_secs_f32(), ids.len(), t_prefill.as_secs_f32(), out.len(), t2.elapsed().as_secs_f64() * 1e3 / out.len().max(1) as f64);
}
