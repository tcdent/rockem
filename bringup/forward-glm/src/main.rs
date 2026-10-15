//! forward-glm — run the f32 transcription of GLM-5.3-Flash over an oracle
//! passage, layer by layer from the FP8 checkpoint (the oracle's own order: one
//! layer's weights resident at a time), and diff every edge the oracle saved.
//!
//!   forward-glm --goldens <bringup/goldens/glm/SET> [--ckpt ~/hf-cache/glm-5.3-flash] [--layers N] [--tol 1e-4]
//!
//! Edges per layer, each `[positions][H]`: attn_in (the normed block input), attn_out,
//! ffn_in, ffn_out, out (the mean of the four streams). The table is the worst
//! row per edge class; each layer also prints its own worst per edge, which is
//! where an error's growth shows.

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
    assert!(LINEAR_ATTENTION[l], "layer {l}: MLA is not transcribed yet");
    assert!(DENSE_MLP[l], "layer {l}: the MoE is not transcribed yet");
    let conv = ["q", "k", "v"].iter().flat_map(|c| w(&format!("self_attn.{c}_conv1d.weight"))).collect();
    Layer {
        attn_hc: hc("attn"),
        ffn_hc: hc("ffn"),
        input_layernorm: w("input_layernorm.weight"),
        post_attention_layernorm: w("post_attention_layernorm.weight"),
        attn: Attn::Kda(Kda {
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
        }),
        ffn: Ffn::Dense(Mlp { gate: w("mlp.gate_proj.weight"), up: w("mlp.up_proj.weight"), down: w("mlp.down_proj.weight"), ff: INTERMEDIATE_SIZE }),
    }
}

fn main() {
    let gdir = arg_required("--goldens");
    let ckpt = arg("--ckpt").unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("hf-cache/glm-5.3-flash"));
    let layers: usize = arg("--layers").map_or(NUM_HIDDEN_LAYERS, |v| v.to_str().unwrap().parse().unwrap());
    let tol: f32 = arg("--tol").map_or(1e-4, |v| v.to_str().unwrap().parse().unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<usize> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let n = ids.len();
    let gold = Golden::load(&gdir.join("edges.safetensors"));
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
        layer.forward(&mut streams, n, &mut st, &mut |name, data| {
            let key = format!("l{l}.{name}");
            let mut worst = 0f32;
            for t in 0..n {
                let (_, rel) = report.record(&format!("model.layers.{l}.{name}"), &data[t * H..(t + 1) * H], gold.row(&key, t).unwrap_or_else(|| panic!("no golden {key}")), &format!("pos {t}"));
                worst = worst.max(rel);
            }
            line.push(format!("{name} {worst:.1e}"));
        });
        println!("layer {l:2} {} load {tl:.1}s run {:.1}s  {}", if LINEAR_ATTENTION[l] { "kda" } else { "mla" }, t0.elapsed().as_secs_f32() - tl, line.join("  "));
    }
    report.print();
    std::process::exit(if report.failures() > 0 { 1 } else { 0 });
}
