//! forward-glm — run the f32 transcription of GLM-5.3-Flash over an oracle
//! passage, layer by layer from the FP8 checkpoint (the oracle's own order: one
//! layer's weights resident at a time), and diff every edge the oracle saved.
//!
//!   forward-glm --goldens <bringup/goldens/glm/SET> [--ckpt ~/hf-cache/glm-5.3-flash] [--layers N] [--tol 1e-4]
//!               [--detail 1e-3]   also print every row past this relative error, and every position whose picks differ
//!
//! Edges per layer, each `[positions][H]`: attn_in (the normed block input), attn_out,
//! ffn_in, ffn_out, out (the mean of the four streams). The table is the worst
//! row per edge class; each layer also prints its own worst per edge, which is
//! where an error's growth shows. Edges a golden lacks are skipped.

use glm_5_3_flash::reference::*;
use glm_5_3_flash::*;
use harness::{arg, arg_required, Checkpoint, Golden, Report};
use std::path::PathBuf;

fn load_layer(ck: &Checkpoint, l: usize) -> Layer {
    let p = |s: &str| format!("model.language_model.layers.{l}.{s}");
    let w = |s: &str| ck.f32s(&p(s));
    let hc = |which: &str| Hc {
        fn_: w(&format!("hc_{which}_fn")),
        base: w(&format!("hc_{which}_base")),
        scale: w(&format!("hc_{which}_scale")).try_into().unwrap(),
    };
    let attn = if LINEAR_ATTENTION[l] {
        let conv = ["q", "k", "v"].iter().flat_map(|c| w(&format!("self_attn.{c}_conv1d.weight"))).collect();
        Attn::Kda(Kda {
            q: w("self_attn.q_proj.weight"),
            k: w("self_attn.k_proj.weight"),
            v: w("self_attn.v_proj.weight"),
            conv,
            f_a: w("self_attn.f_a_proj.weight"),
            f_b: w("self_attn.f_b_proj.weight"),
            dt_bias: w("self_attn.dt_bias"),
            a_log: w("self_attn.A_log"),
            b: w("self_attn.b_proj.weight"),
            g_a: w("self_attn.g_a_proj.weight"),
            g_b: w("self_attn.g_b_proj.weight"),
            o_norm: w("self_attn.o_norm.weight"),
            o: w("self_attn.o_proj.weight"),
        })
    } else {
        Attn::Mla(Mla {
            q_a: w("self_attn.q_a_proj.weight"),
            q_a_ln: w("self_attn.q_a_layernorm.weight"),
            q_b: w("self_attn.q_b_proj.weight"),
            kv_a: w("self_attn.kv_a_proj_with_mqa.weight"),
            kv_a_ln: w("self_attn.kv_a_layernorm.weight"),
            kv_b: w("self_attn.kv_b_proj.weight"),
            o: w("self_attn.o_proj.weight"),
            indexer: Indexer {
                wq_b: w("self_attn.indexer.wq_b.weight"),
                wk: w("self_attn.indexer.wk.weight"),
                k_norm_w: w("self_attn.indexer.k_norm.weight"),
                k_norm_b: w("self_attn.indexer.k_norm.bias"),
                weights: w("self_attn.indexer.weights_proj.weight"),
                ape: w("self_attn.indexer.index_kpool_compress_ape"),
                gate: w("self_attn.indexer.index_kpool_compress_gate"),
            },
        })
    };
    let mlp = |pre: &str, ff| Mlp { gate: w(&format!("{pre}gate_proj.weight")), up: w(&format!("{pre}up_proj.weight")), down: w(&format!("{pre}down_proj.weight")), ff };
    let ffn = if DENSE_MLP[l] {
        Ffn::Dense(mlp("mlp.", INTERMEDIATE_SIZE))
    } else {
        Ffn::Moe(Moe { router: w("mlp.gate.weight"), bias: w("mlp.gate.e_score_correction_bias"), shared: mlp("mlp.shared_experts.", D_FF * N_SHARED_EXPERTS) })
    };
    Layer { attn_hc: hc("attn"), ffn_hc: hc("ffn"), input_layernorm: w("input_layernorm.weight"), post_attention_layernorm: w("post_attention_layernorm.weight"), attn, ffn }
}

fn load_expert(ck: &Checkpoint, l: usize, e: usize) -> Expert {
    let p = |s: &str| ck.f32s(&format!("model.language_model.layers.{l}.mlp.experts.{e}.{s}_proj.weight"));
    let mut gate_up = p("gate");
    gate_up.extend(p("up"));
    Expert { gate_up, down: p("down") }
}

fn main() {
    let gdir = arg_required("--goldens");
    let ckpt = arg("--ckpt").unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("hf-cache/glm-5.3-flash"));
    let layers: usize = arg("--layers").map_or(NUM_HIDDEN_LAYERS, |v| v.to_str().unwrap().parse().unwrap());
    let detail: f32 = arg("--detail").map_or(0.0, |v| v.to_str().unwrap().parse().unwrap());
    let tol: f32 = arg("--tol").map_or(1e-4, |v| v.to_str().unwrap().parse().unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<usize> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let n = ids.len();
    let gold = Golden::load(&gdir.join("edges.safetensors"));
    let routing = Golden::load(&gdir.join("routing.safetensors"));
    let ck = Checkpoint::open(&ckpt);
    let mut report = Report::new(tol);
    println!("{n} positions, {layers} layers, goldens {}", gdir.display());

    let emb = ck.bf16("model.language_model.embed_tokens.weight");
    let mut streams = vec![0f32; n * HC * H];
    for (t, &id) in ids.iter().enumerate() {
        let row: Vec<f32> = emb[id * H..(id + 1) * H].iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();
        report.record("embed", &row, gold.row("embed", t).unwrap(), &format!("pos {t}"));
        for j in 0..HC {
            streams[(t * HC + j) * H..(t * HC + j + 1) * H].copy_from_slice(&row);
        }
    }

    for l in 0..layers {
        let t0 = std::time::Instant::now();
        let layer = load_layer(&ck, l);
        let tl = t0.elapsed().as_secs_f32();
        let mut st = layer.state();
        let mut line = Vec::new();
        let touched = std::sync::atomic::AtomicUsize::new(0);
        let r = layer.forward(&mut streams, n, &mut st, &mut |name, data| {
            let key = format!("l{l}.{name}");
            if gold.row(&key, 0).is_none() {
                return; // a golden run with fewer edges (--no-edges keeps each layer's out)
            }
            let mut worst = 0f32;
            for t in 0..n {
                let (_, rel) = report.record(&format!("model.layers.{l}.{name}"), &data[t * H..(t + 1) * H], gold.row(&key, t).unwrap(), &format!("pos {t}"));
                if detail > 0.0 && rel > detail {
                    println!("    l{l}.{name} pos {t}: rel {rel:.2e}");
                }
                worst = worst.max(rel);
            }
            line.push(format!("{name} {worst:.1e}"));
        }, &|e| {
            touched.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            load_expert(&ck, l, e)
        });
        let touched = touched.into_inner();
        if let (LayerState::Mla(m), Some(_)) = (&st, routing.irow(&format!("l{l}.index"), 0)) {
            // the indexer's choices against the oracle's, as sets of positions
            let (mut differ, mut first) = (0, None);
            for t in 0..n {
                let mut g: Vec<usize> = routing.irow(&format!("l{l}.index"), t).unwrap().iter().filter(|&&j| j >= 0).map(|&j| j as usize).collect();
                g.sort();
                g.dedup();
                if g != m.selected[t] {
                    differ += 1;
                    first.get_or_insert(t);
                    if detail > 0.0 {
                        let only_ours = m.selected[t].iter().filter(|j| g.binary_search(j).is_err()).count();
                        let only_theirs = g.iter().filter(|j| m.selected[t].binary_search(j).is_err()).count();
                        println!("    l{l} pos {t}: selection differs, {only_ours} only ours, {only_theirs} only the oracle's (of {})", g.len());
                    }
                }
            }
            if differ > 0 {
                report.fail(&format!("model.layers.{l}.index"), format!("{differ} selections differ at layer {l}, first at pos {}", first.unwrap()));
            }
            line.push(format!("index: {differ} of {n} selections differ"));
        }
        if let Some(r) = r {
            // the picks as sets (the reference's top-k is unsorted), the weights matched by expert
            let (mut flips, mut wworst) = (0, 0f32);
            for t in 0..n {
                let gi = routing.irow(&format!("l{l}.topk"), t).unwrap();
                let gw = routing.row(&format!("l{l}.weights"), t).unwrap();
                let mine = &r.idx[t * TOP_K..(t + 1) * TOP_K];
                let miss = mine.iter().filter(|e| !gi.contains(&(**e as i64))).count();
                if miss > 0 && detail > 0.0 {
                    println!("    l{l} pos {t}: {miss} picks differ, ours {mine:?} oracle {gi:?}");
                }
                flips += miss;
                for (k, &e) in mine.iter().enumerate() {
                    if let Some(j) = gi.iter().position(|&g| g == e as i64) {
                        wworst = wworst.max((r.weight[t * TOP_K + k] - gw[j]).abs() / gw[j].abs());
                    }
                }
            }
            if flips > 0 {
                report.fail(&format!("model.layers.{l}.picks"), format!("{flips} picks differ at layer {l}"));
            }
            line.push(format!("picks {} differ, weights {wworst:.1e}, {touched} experts", flips));
        }
        println!("layer {l:2} {} load {tl:.1}s run {:.1}s  {}", if LINEAR_ATTENTION[l] { "kda" } else { "mla" }, t0.elapsed().as_secs_f32() - tl, line.join("  "));
    }
    if layers == NUM_HIDDEN_LAYERS {
        // the head: the four streams' mean, norm, lm_head; then the logits against the oracle's and the text
        let mean: Vec<f32> = (0..n * H).map(|i| (0..HC).map(|j| streams[(i / H) * HC * H + j * H + i % H]).sum::<f32>() / HC as f32).collect();
        let x = rmsnorm(&mean, H, &ck.f32s("model.language_model.norm.weight"), RMS_NORM_EPS);
        let logits = matmul(&ck.f32s("lm_head.weight"), VOCAB_SIZE, H, &x, n);
        let (mut agree, mut agree_oracle, mut nll) = (0, 0, 0f64);
        for t in 0..n {
            let row = &logits[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE];
            let g = gold.row("logits", t).unwrap();
            report.record("logits", row, g, &format!("pos {t}"));
            let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree_oracle += (am(row) == am(g)) as usize;
            if t + 1 < n {
                agree += (am(row) == ids[t + 1]) as usize;
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|&v| ((v as f64) - mx).exp()).sum::<f64>().ln();
                nll += lse - row[ids[t + 1]] as f64;
            }
        }
        println!("head: top-1 agrees with the oracle's at {agree_oracle}/{n}; with the text {:.3} (oracle {:.3}); NLL {:.4} (oracle {:.4})",
            agree as f64 / (n - 1) as f64, meta["top1_agreement"].as_f64().unwrap_or(f64::NAN), nll / (n - 1) as f64, meta["nll"].as_f64().unwrap_or(f64::NAN));
    }
    report.print();
    std::process::exit(if report.failures() > 0 { 1 } else { 0 });
}
