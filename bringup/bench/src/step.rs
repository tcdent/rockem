//! bench-step — the card's ops one at a time with a wait after each, in the
//! order a step runs them, printing as each returns: where a step hangs is
//! the last line printed. With `--tokens n` the step carries n tokens.
//!
//!   bench-step --bundle <dir> --elf <kernels.gfx1100.elf> [--tokens 1] [--pos 8]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
use deepseek_v2_lite::{attn, ffn, Attention, FeedForward, Head, Model, Rope, Weights, NUM_HIDDEN_LAYERS, QK_ROPE_HEAD_DIM, SAMPLING, VOCAB_SIZE};
use harness::{arg, arg_required};
use std::io::Write;

fn main() {
    let bundle = arg_required("--bundle");
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let n: usize = arg("--tokens").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(1);
    let pos: usize = arg("--pos").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(8);
    let dense_b = std::fs::read(bundle.join(deepseek_v2_lite::layout::DENSE_FILE)).unwrap();
    let experts = std::fs::read(bundle.join(deepseek_v2_lite::layout::EXPERTS_FILE)).unwrap();
    let (p, dense, q) = unsafe { dense_b.align_to::<u16>() };
    assert!(p.is_empty() && q.is_empty());
    let w = Weights::from_dense(dense);
    let max_ctx = 256;
    let rope = Rope::new(max_ctx);
    let k = kfd::Kfd::new(&elf);
    let mut m = kfd::KfdModel::new(k, dense, &experts[..deepseek_v2_lite::layout::STREAM], &rope, max_ctx);
    let mut rng = common::testing::Rng::new(1);
    let mut h: Vec<f32> = (0..n * deepseek_v2_lite::H).map(|_| rng.unit()).collect();
    let mut logits = vec![0f32; n * VOCAB_SIZE];
    let cos = &rope.cos[pos * QK_ROPE_HEAD_DIM..];
    let sin = &rope.sin[pos * QK_ROPE_HEAD_DIM..];
    let say = |s: &str| {
        print!("{s} ");
        std::io::stdout().flush().unwrap();
    };
    let mut per: std::collections::BTreeMap<&'static str, f64> = Default::default();
    let mut kernels: std::collections::BTreeMap<&'static str, (usize, f64)> = Default::default();
    for l in 0..NUM_HIDDEN_LAYERS {
        let lw = &w.layers[l];
        let aw = attn::Weights { input_ln: &lw.input_ln, q_proj: lw.q_proj, kv_a: lw.kv_a, kv_a_ln: &lw.kv_a_ln, kv_b: &lw.kv_b, o_proj: lw.o_proj };
        let t = std::time::Instant::now();
        m.step(l, &aw, &mut h, pos, n, cos, sin, rope.softmax_scale, &mut common::trace::NoTrace);
        m.sync();
        *per.entry("attention").or_default() += t.elapsed().as_secs_f64();
        for (k, a, b) in m.timeline() {
            let e = kernels.entry(k).or_default();
            e.0 += 1;
            e.1 += (b - a) as f64 / 100.0;
        }
        say(&format!("attn{l}"));
        let t = std::time::Instant::now();
        match &lw.ffn {
            deepseek_v2_lite::Ffn::Dense { gate, up, down } => m.dense(l, &ffn::Dense { post_ln: &lw.post_ln, gate, up, down }, &mut h, n, &mut common::trace::NoTrace),
            deepseek_v2_lite::Ffn::Moe { router, shared_gate, shared_up, shared_down } => m.moe(l, &ffn::Moe { post_ln: &lw.post_ln, router, shared_gate, shared_up, shared_down }, &mut h, n, &mut common::trace::NoTrace),
        }
        m.sync();
        *per.entry("ffn").or_default() += t.elapsed().as_secs_f64();
        for (k, a, b) in m.timeline() {
            let e = kernels.entry(k).or_default();
            e.0 += 1;
            e.1 += (b - a) as f64 / 100.0;
        }
        say(&format!("ffn{l}"));
    }
    let t = std::time::Instant::now();
    m.logits(&w.norm, w.lm_head, &h, n, &mut logits);
    m.sync();
    *per.entry("head").or_default() += t.elapsed().as_secs_f64();
    for (k, a, b) in m.timeline() {
        let e = kernels.entry(k).or_default();
        e.0 += 1;
        e.1 += (b - a) as f64 / 100.0;
    }
    say("head");
    println!();
    for (k, v) in &per {
        println!("  {k:<10} {:.2} ms for {n} tokens", v * 1e3);
    }
    for (k, (c, us)) in &kernels {
        println!("  {k:<14} ×{c:<3} inside {:>8.0} µs ({:>7.1} each)", us, us / *c as f64);
    }
    let t = m.draw(n - 1, &SAMPLING, 0.5, &logits);
    say(&format!("draw={t}"));
    let s = m.score(n - 1, t, &logits);
    println!("score={s:.4}");
    for (name, from, to) in m.timeline() {
        if name == "sample" || name == "score" {
            println!("  {name}: {} µs", (to - from) as f64 / 100.0);
        }
    }
}
