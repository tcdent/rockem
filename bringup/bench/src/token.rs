//! bench-token — one token on the card, by phase: attention for all layers,
//! the FFN half for all layers, the head, and the whole `decode`; each posted
//! and waited once, at a given position. Where the token's time goes.
//!
//!   bench-token --bundle <dir> [--elf <kernels.elf>] [--pos 64] [--iters 30]
//!
//! Then the card's own timeline of one token, from the signal every launch
//! carries: per kernel, the time inside it and the gap before it, for the
//! layers run interleaved (as a token runs them) and for each half alone.

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use deepseek_v2_lite::layout;
use deepseek_v2_lite::{attn, ffn, Attention, FeedForward, Head, Model, Rope, State, Weights, NUM_HIDDEN_LAYERS, SAMPLING, VOCAB_SIZE};
use harness::{arg, arg_required};

fn main() {
    let bundle = arg_required("--bundle");
    let pos: usize = arg("--pos").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(64);
    let iters: usize = arg("--iters").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(30);
    let elf = std::fs::read(arg("--elf").unwrap_or_else(|| bundle.join(layout::KERNELS_FILE))).unwrap();
    let f = std::fs::File::open(bundle.join(layout::DENSE_FILE)).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let (_, dense, _) = unsafe { mm[..].align_to::<u16>() };
    let ef = std::fs::File::open(bundle.join(layout::EXPERTS_FILE)).unwrap();
    let emm = unsafe { memmap2::Mmap::map(&ef).unwrap() };
    let w = Weights::from_dense(dense);
    let k = kfd::Kfd::new(&elf, true);
    let max_ctx = 2048;
    let rope = Rope::new(max_ctx);
    let mut m = kfd::KfdModel::new(k.clone(), dense, &emm[..layout::STREAM], &rope, max_ctx);
    let mut rng = common::testing::Rng::new(3);
    let rows: Vec<f32> = (0..(pos + 1) * layout::POSITION).map(|_| rng.unit() * 0.1).collect();
    m.seed(&rows);
    let mut h: Vec<f32> = (0..deepseek_v2_lite::H).map(|_| rng.unit()).collect();
    let mut logits = vec![0f32; VOCAB_SIZE];
    let (cos, sin) = (&rope.cos[pos * deepseek_v2_lite::QK_ROPE_HEAD_DIM..], &rope.sin[pos * deepseek_v2_lite::QK_ROPE_HEAD_DIM..]);
    let aw = |l: usize| {
        let lw = &w.layers[l];
        attn::Weights { input_ln: &lw.input_ln, q_proj: lw.q_proj, kv_a: lw.kv_a, kv_a_ln: &lw.kv_a_ln, kv_b: &lw.kv_b, o_proj: lw.o_proj }
    };
    let ms = |b: harness::Bench| b.median.as_secs_f64() * 1e3;

    let attention = |m: &mut kfd::KfdModel, h: &mut [f32]| {
        for l in 0..NUM_HIDDEN_LAYERS {
            m.step(l, &aw(l), h, pos, 1, cos, sin, rope.softmax_scale, &mut common::trace::NoTrace);
        }
        m.sync();
    };
    attention(&mut m, &mut h);
    let t_attn = ms(harness::bench(iters, || attention(&mut m, &mut h)));

    let ffn_all = |m: &mut kfd::KfdModel, h: &mut [f32]| {
        for (l, lw) in w.layers.iter().enumerate() {
            match &lw.ffn {
                deepseek_v2_lite::Ffn::Dense { gate, up, down } => m.dense(l, &ffn::Dense { post_ln: &lw.post_ln, gate, up, down }, h, 1, &mut common::trace::NoTrace),
                deepseek_v2_lite::Ffn::Moe { router, shared_gate, shared_up, shared_down } => m.moe(l, &ffn::Moe { post_ln: &lw.post_ln, router, shared_gate, shared_up, shared_down }, h, 1, &mut common::trace::NoTrace),
            }
        }
        m.sync();
    };
    ffn_all(&mut m, &mut h);
    let t_ffn = ms(harness::bench(iters, || ffn_all(&mut m, &mut h)));

    let head = |m: &mut kfd::KfdModel, logits: &mut [f32]| m.logits(&w.norm, w.lm_head, &h, 1, logits);
    head(&mut m, &mut logits);
    let t_head = ms(harness::bench(iters, || head(&mut m, &mut logits)));

    // the layers as decode runs them, attention and FFN interleaved, no head: one wait
    let layers = |m: &mut kfd::KfdModel, h: &mut [f32]| {
        for (l, lw) in w.layers.iter().enumerate() {
            m.step(l, &aw(l), h, pos, 1, cos, sin, rope.softmax_scale, &mut common::trace::NoTrace);
            match &lw.ffn {
                deepseek_v2_lite::Ffn::Dense { gate, up, down } => m.dense(l, &ffn::Dense { post_ln: &lw.post_ln, gate, up, down }, h, 1, &mut common::trace::NoTrace),
                deepseek_v2_lite::Ffn::Moe { router, shared_gate, shared_up, shared_down } => m.moe(l, &ffn::Moe { post_ln: &lw.post_ln, router, shared_gate, shared_up, shared_down }, h, 1, &mut common::trace::NoTrace),
            }
        }
        m.sync();
    };
    layers(&mut m, &mut h);
    let t_layers = ms(harness::bench(iters, || layers(&mut m, &mut h)));
    // the head and the draw, one wait
    let tail = |m: &mut kfd::KfdModel, logits: &mut [f32]| {
        m.logits(&w.norm, w.lm_head, &h, 1, logits);
        m.draw(0, &SAMPLING, 0.5, logits);
    };
    tail(&mut m, &mut logits);
    let t_tail = ms(harness::bench(iters, || tail(&mut m, &mut logits)));
    // the draw alone: its launch, the kernel, the wait
    let t_draw = ms(harness::bench(iters, || {
        m.draw(0, &SAMPLING, 0.5, &logits);
    }));

    // what posting costs the host: a token's worth of attention launches, timed before the wait
    let t = std::time::Instant::now();
    for l in 0..NUM_HIDDEN_LAYERS {
        m.step(l, &aw(l), &mut h, pos, 1, cos, sin, rope.softmax_scale, &mut common::trace::NoTrace);
    }
    let post_us = t.elapsed().as_secs_f64() * 1e6 / (4 * NUM_HIDDEN_LAYERS) as f64;
    m.sync();

    let mut st = State::new(max_ctx);
    st.decode(&w, 549, pos, &mut m, &SAMPLING, 0.5, &mut common::trace::NoTrace);
    let t_tok = ms(harness::bench(iters, || {
        st.decode(&w, 549, pos, &mut m, &SAMPLING, 0.5, &mut common::trace::NoTrace);
    }));
    let _ = t_head;
    let account = |title: &str, tl: &[(&'static str, u64, u64)]| {
        let mut per: std::collections::BTreeMap<&str, (usize, f64, f64)> = Default::default(); // n, inside µs, gap-before µs
        for (i, &(name, s, e)) in tl.iter().enumerate() {
            let en = per.entry(name).or_default();
            en.0 += 1;
            en.1 += (e - s) as f64 / 100.0;
            if i > 0 {
                en.2 += (s as i64 - tl[i - 1].2 as i64) as f64 / 100.0;
            }
        }
        let span = (tl.last().unwrap().2 - tl[0].1) as f64 / 100.0;
        let (inside, gaps): (f64, f64) = per.values().fold((0.0, 0.0), |a, v| (a.0 + v.1, a.1 + v.2));
        println!("{title}: {:.0} µs on the card = {inside:.0} inside {} kernels + {gaps:.0} in the gaps between them", span, tl.len());
        for (name, (n, ins, gap)) in &per {
            println!("  {name:<16} ×{n:<3} inside {ins:>7.0} µs ({:>5.1} each)  gap before {gap:>6.0} µs ({:>4.1} each)", ins / *n as f64, gap / *n as f64);
        }
    };
    layers(&mut m, &mut h);
    account("layers interleaved", &m.timeline());
    // the kernels' own statistics, one entry per wave, aggregated here: with a stats code object
    let st = m.stats();
    if !st.is_empty() {
        let mut per: std::collections::BTreeMap<(&str, usize), Vec<f32>> = Default::default();
        for (name, slots) in &st {
            for (i, v) in slots.iter().enumerate() {
                if !deepseek_v2_lite::stat_names(name)[i].is_empty() {
                    per.entry((name, i)).or_default().extend_from_slice(v);
                }
            }
        }
        // cycles → time: the shader clock inferred per kernel from the timeline (Σ phases per wave
        // against the launch's real-time span), printed so the assumption is visible
        let tl = m.timeline();
        let mut ghz: std::collections::BTreeMap<&str, f64> = Default::default();
        for ((name, _), _) in &per {
            let name: &str = name;
            let cyc: f64 = per.iter().filter(|((n, _), _)| *n == name).map(|(_, v)| v.iter().sum::<f32>() as f64 / v.len() as f64).sum();
            let secs: f64 = tl.iter().filter(|(n, _, _)| *n == name).map(|(_, s, e)| (e - s) as f64 / 1e8).sum::<f64>() / tl.iter().filter(|(n, _, _)| *n == name).count().max(1) as f64;
            if cyc > 0.0 && secs > 0.0 {
                ghz.insert(name, cyc / secs / 1e9);
            }
        }
        println!("kernel statistics, over every wave of every launch of the token (cycles; the clock inferred per kernel):");
        for ((name, i), v) in &per {
            let label = deepseek_v2_lite::stat_names(name)[*i];
            let (n, mean, max) = (v.len(), v.iter().sum::<f32>() / v.len() as f32, v.iter().cloned().fold(0f32, f32::max));
            if label.ends_with("_ticks") {
                let g = ghz.get(*name).copied().unwrap_or(2.5);
                println!("  {name:<12} {label:<16} waves {n:>6}  mean {:>8.0} cycles = {:>6.2} µs at {g:.2} GHz  max {:>8.0}", mean, mean as f64 / g / 1e3, max);
            } else {
                println!("  {name:<12} {label:<16} waves {n:>6}  mean {mean:>10.3}  max {max:>10.3}");
            }
        }
        // the sink detector, layer by layer: the largest residual any wave saw in each down kernel
        let mut l = 0;
        for (name, slots) in &st {
            if *name == "dense_down" || *name == "shared_down" {
                let mx = slots[0].iter().cloned().fold(0f32, f32::max);
                print!("  layer {l:>2} |h| max {mx:>8.1}{}", if l % 3 == 2 { "\n" } else { "   " });
                l += 1;
            }
        }
        println!();
    }
    attention(&mut m, &mut h);
    account("attention alone", &m.timeline());
    ffn_all(&mut m, &mut h);
    account("ffn alone", &m.timeline());
    m.logits(&w.norm, w.lm_head, &h, 1, &mut logits);
    m.draw(0, &SAMPLING, 0.5, &logits);
    account("head + draw alone", &m.timeline());
    println!("pos {pos}: attention {t_attn:.2} ms + ffn {t_ffn:.2} ms (interleaved as decode runs them: {t_layers:.2}); head + draw {t_tail:.2} ms (the draw alone {:.0} µs); decode {t_tok:.2} ms = {:.0} tok/s; posting a launch costs the host {post_us:.1} µs", t_draw * 1e3, 1e3 / t_tok);
}
