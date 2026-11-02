//! Where a round's time goes, on one core, on the random five-layer model (`fixtures`): the three
//! passes a session takes — the feed forward (states only), the recompute forward (everything
//! kept, logits where targets are) and the backward — timed whole and per layer kind, then
//! extrapolated to GLM-5.3-Flash's 45 layers. Cache-hot where a layer's weights fit (the KDA
//! layer's ~300 MB and the MoE's eight experts do not fit L3; they stream from DRAM as on the
//! box), so the per-token numbers are a dev-box floor for the CPU path's instruction and memory
//! cost, not metatron's.
//!
//!   cargo run --release --example round_rate [n_tokens] [rank]

use common::pool::Serial;
use common::testing::Rng;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::weights::{Attn, Ffn};
use glm_5_3_flash::{FIRST_K_DENSE_REPLACE, H, HC, NUM_HIDDEN_LAYERS, TOP_K};
use glm_5_3_flash_train::expert::Adapter;
use glm_5_3_flash_train::fixtures::{Bufs, Experts};
use glm_5_3_flash_train::model::{self, Adapters, Carry, Scratch, States};
use glm_5_3_flash_train::testing::unit;
use glm_5_3_flash_train::{ffn, hc, head, kda, mla};
use std::time::Instant;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let n: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(16);
    let r: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(8);
    let vocab = 4096;
    let mut rnd = Rng::new(0xBE4C_0002);
    let t = Instant::now();
    let bufs = Bufs::new(vocab, &mut rnd);
    let w = bufs.weights();
    let experts = Experts::new(&mut rnd);
    let stream = experts.stream();
    eprintln!("model built in {:.1} s", t.elapsed().as_secs_f64());
    let mut ad = Adapters::zeros(&w, r);
    for l in [FIRST_K_DENSE_REPLACE, FIRST_K_DENSE_REPLACE + 1] {
        for e in 0..TOP_K {
            let a = &mut ad.layers[l].as_mut().unwrap()[e];
            for v in a.ga.iter_mut().chain(a.gb.iter_mut()).chain(a.ua.iter_mut()).chain(a.ub.iter_mut()).chain(a.da.iter_mut()).chain(a.db.iter_mut()) {
                *v = rnd.unit() * 0.05;
            }
        }
    }
    let tokens: Vec<u32> = (0..n).map(|_| (rnd.next() % vocab as u64) as u32).collect();
    let at: Vec<usize> = (0..n).filter(|t| t % 2 == 1).collect();
    let max_ctx = n.max(16);

    // the three passes, whole
    let mut st = States::new(&w, max_ctx);
    let mut s = Scratch::new(r, max_ctx);
    let mut none = [0f32; 0];
    let t0 = Instant::now();
    let _ = model::forward(&w, &stream, &ad, &tokens, 0, vocab, &[], &mut st, &mut s, &mut none);
    let feed = ms(t0);
    st.reset();
    let mut logits = vec![0f32; at.len() * vocab];
    let t0 = Instant::now();
    let sv = model::forward(&w, &stream, &ad, &tokens, 0, vocab, &at, &mut st, &mut s, &mut logits);
    let recompute = ms(t0);
    let dlogits = unit(at.len() * vocab, 1e-3, &mut rnd);
    let mut grads = Adapters::zeros(&w, r);
    let mut carry = Carry::zeros(&w, max_ctx);
    let t0 = Instant::now();
    model::backward(&w, &stream, &ad, &sv, &st, vocab, &dlogits, &mut s, &mut grads, &mut carry);
    let backward = ms(t0);
    println!("five layers (3 dense KDA, MLA+MoE, KDA+MoE), {n} tokens, {} with targets, rank {r}, one core:", at.len());
    println!("  feed forward      {feed:8.1} ms   {:6.2} ms a token", feed / n as f64);
    println!("  recompute forward {recompute:8.1} ms   {:6.2} ms a token", recompute / n as f64);
    println!("  backward          {backward:8.1} ms   {:6.2} ms a token", backward / n as f64);

    // per layer kind: the attention over the run, the FFN half per token, the hc halves, the head
    let xn = unit(n * H, 1.0, &mut rnd);
    let dy = unit(n * H, 1.0, &mut rnd);
    let mut y = vec![0f32; n * H];
    let kda_w = match &w.layers[0].attn { Attn::Kda(a) => a, _ => unreachable!() };
    let mla_w = match &w.layers[FIRST_K_DENSE_REPLACE].attn { Attn::Mla(a) => a, _ => unreachable!() };
    let moe_w = match &w.layers[FIRST_K_DENSE_REPLACE].ffn { Ffn::Moe(m) => m, _ => unreachable!() };
    let dense_w = match &w.layers[0].ffn { Ffn::Dense(d) => d, _ => unreachable!() };

    let (mut window, mut mem) = (vec![0f32; KDA_WINDOW], vec![0f32; KDA_MEMORY]);
    let mut ks = kda::Scratch::new();
    let mut ksaved = kda::Saved::new(n);
    let t0 = Instant::now();
    kda::forward(0, kda_w, &xn, n, &mut window, &mut mem, &mut ks, &mut ksaved, &mut y);
    let kda_f = ms(t0);
    let (mut dxn, mut dwin, mut dmem) = (vec![0f32; n * H], vec![0f32; KDA_WINDOW], vec![0f32; KDA_MEMORY]);
    let t0 = Instant::now();
    kda::backward(kda_w, &ksaved, &dy, &dwin.clone(), &dmem.clone(), &mut dxn, &mut dwin, &mut dmem);
    let kda_b = ms(t0);

    let mut mst = mla::State::new(max_ctx);
    let mut mscr = mla::Scratch::new(max_ctx);
    let mut msaved = mla::Saved::new(0, n);
    let t0 = Instant::now();
    mla::forward(3, mla_w, &xn, 0, n, &mut mst, &mut mscr, &mut msaved, &mut y);
    let mla_f = ms(t0);
    let mut dlatent = vec![0f32; max_ctx * glm_5_3_flash::KV_LORA_RANK];
    dxn.iter_mut().for_each(|v| *v = 0.0);
    let t0 = Instant::now();
    mla::backward(mla_w, &mst, &msaved, &dy, &mut dlatent, &mut dxn);
    let mla_b = ms(t0);

    let mut fs = ffn::Scratch::new(r);
    let mut fsaved = ffn::Saved::new(r);
    let adl = ad.layers[FIRST_K_DENSE_REPLACE].as_ref().unwrap();
    let mut out = vec![0f32; H];
    let t0 = Instant::now();
    for t in 0..n {
        ffn::forward(FIRST_K_DENSE_REPLACE, moe_w, &stream, adl, &xn[t * H..(t + 1) * H], &mut fs, &mut fsaved, &mut out);
    }
    let moe_f = ms(t0) / n as f64;
    let mut g: Vec<Adapter> = adl.iter().map(|a| Adapter::zeros(a.r)).collect();
    let mut dx = vec![0f32; H];
    let t0 = Instant::now();
    for _ in 0..n {
        ffn::backward(FIRST_K_DENSE_REPLACE, moe_w, &stream, adl, &fsaved, &dy[..H], &mut fs, &mut g, &mut dx);
    }
    let moe_b = ms(t0) / n as f64;

    let (mut gu, mut h, mut eo, mut sh, mut sc) = (vec![0f32; 2 * glm_5_3_flash::INTERMEDIATE_SIZE], vec![0f32; glm_5_3_flash::INTERMEDIATE_SIZE], vec![0f32; TOP_K * H], vec![0f32; H], vec![0f32; glm_5_3_flash::E]);
    let t0 = Instant::now();
    for t in 0..n {
        let mut ds = glm_5_3_flash::ffn::Scratch { gu: &mut gu, h: &mut h, expert_out: &mut eo, shared: &mut sh, score: &mut sc };
        glm_5_3_flash::ffn::dense(0, dense_w, &xn[t * H..(t + 1) * H], &mut ds, &Serial, &mut common::trace::NoTrace, &mut out);
    }
    let dense_f = ms(t0) / n as f64;

    let streams = unit(HC * H, 1.0, &mut rnd);
    let (mut flat, mut coef) = (vec![0f32; HC * H], vec![0f32; glm_5_3_flash::HC_COEFS]);
    let mut x = vec![0f32; H];
    let t0 = Instant::now();
    let mut mix = None;
    for _ in 0..n {
        mix = Some(glm_5_3_flash::hc::pre(&w.layers[0].hc_attn, &streams, &mut glm_5_3_flash::hc::Scratch { flat: &mut flat, coef: &mut coef }, &Serial, &mut x));
    }
    let hc_f = ms(t0) / n as f64;
    let mix = mix.unwrap();
    let hsaved = hc::Saved::from_forward(&w.layers[0].hc_attn, &coef, mix);
    let mut ds = vec![0f32; HC * H];
    let dsout = unit(HC * H, 1.0, &mut rnd);
    let t0 = Instant::now();
    for _ in 0..n {
        let mut dyv = vec![0f32; H];
        let dm = hc::post_backward(&mix, &x, &streams, &dsout, &mut dyv, &mut ds);
        hc::pre_backward(&w.layers[0].hc_attn, &streams, &hsaved, &dyv, &dm, &mut ds);
    }
    let hc_b = ms(t0) / n as f64;

    let mut hsv = head::Saved::new();
    let mut lg = vec![0f32; vocab];
    let t0 = Instant::now();
    for _ in 0..n {
        head::forward(w.norm, w.lm_head, vocab, &streams, &mut hsv, &mut lg);
    }
    let head_f = ms(t0) / n as f64 * (glm_5_3_flash::VOCAB_SIZE as f64 / vocab as f64);
    let t0 = Instant::now();
    for _ in 0..n {
        head::backward(w.norm, w.lm_head, vocab, &hsv, &lg, &mut ds);
    }
    let head_b = ms(t0) / n as f64 * (glm_5_3_flash::VOCAB_SIZE as f64 / vocab as f64);

    println!("per layer kind, ms a token (attention timed over the {n}-token run):");
    println!("  KDA layer     forward {:6.2}  backward {:6.2}", kda_f / n as f64, kda_b / n as f64);
    println!("  MLA layer     forward {:6.2}  backward {:6.2}   (short context: {n} positions)", mla_f / n as f64, mla_b / n as f64);
    println!("  MoE half      forward {moe_f:6.2}  backward {moe_b:6.2}   (8 experts + shared; weights stream from DRAM)");
    println!("  dense FFN     forward {dense_f:6.2}");
    println!("  hc half       forward {hc_f:6.2}  backward {hc_b:6.2}");
    println!("  head (full vocabulary, scaled from {vocab} rows) forward {head_f:6.2}  backward {head_b:6.2}   per target position");

    // GLM-5.3-Flash: 34 KDA + 11 MLA layers, 42 MoE + 3 dense FFN halves, 90 hc halves
    let (kda_n, mla_n, moe_n, dense_n, hc_n) = (34.0, 11.0, (NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE) as f64, FIRST_K_DENSE_REPLACE as f64, 2.0 * NUM_HIDDEN_LAYERS as f64);
    let fwd = kda_n * kda_f / n as f64 + mla_n * mla_f / n as f64 + moe_n * moe_f + dense_n * dense_f + hc_n * hc_f;
    let bwd = kda_n * kda_b / n as f64 + mla_n * mla_b / n as f64 + moe_n * moe_b + hc_n * hc_b;
    let target_share = at.len() as f64 / n as f64;
    println!("extrapolated to the 45 layers, ms a session token, one core:");
    println!("  feed forward ≈ {fwd:7.1}   recompute ≈ {:7.1}   backward ≈ {bwd:7.1}   total ≈ {:7.1}", fwd + target_share * head_f, 2.0 * fwd + bwd + target_share * (head_f + head_b));
    println!("  (targets on {:.0} % of positions; the head counted there only)", 100.0 * target_share);
}
