//! bench-repeat — the same step, again and again, every result compared bit
//! for bit with the first: the forward's determinism on its own, no node, no
//! wire, no dumps. A prompt of `--positions` positions stepped `--tokens` at a
//! time from position 0 with the draw after each step, as the node runs a
//! prompt; `--repeat` runs, each position's per-layer residual traces, router
//! picks and logits against the first run's. Prints every run that differs
//! (the first position and layer where, the largest difference) and the tally. Found for: one step in ~15 gate runs
//! parting from its twin at one token, one layer, by ~1e-3 (2026-09-19).
//!
//!   bench-repeat --bundle <dir> --elf <kernels.gfx1100.elf> [--tokens 32] [--positions 54] [--repeat 200 | --against <k>] [--idle-ms 0] [--prompt <ids file>] [--ctx 256] [--mirror] [--dump <dir>]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
use deepseek_v2_lite::{attn, ffn, Attention, FeedForward, Head, Model, Rope, Weights, H, NUM_HIDDEN_LAYERS, QK_ROPE_HEAD_DIM, TOP_K, VOCAB_SIZE};
use harness::{arg, arg_required};

fn main() {
    let bundle = arg_required("--bundle");
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let n: usize = arg("--tokens").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(32);
    let repeat: usize = arg("--repeat").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(200);
    let dense_b = std::fs::read(bundle.join(deepseek_v2_lite::layout::DENSE_FILE)).unwrap();
    let experts = std::fs::read(bundle.join(deepseek_v2_lite::layout::EXPERTS_FILE)).unwrap();
    let (p, dense, q) = unsafe { dense_b.align_to::<u16>() };
    assert!(p.is_empty() && q.is_empty());
    let w = Weights::from_dense(dense);
    // `--ctx`: the attention cache's positions on the card (the node's default is 2048)
    let max_ctx: usize = arg("--ctx").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(256);
    // `--mirror`: read every token's mirrored cache rows after the step, as the node does for the store
    let mirror = std::env::args_os().any(|a| a == "--mirror");
    // `--dump <dir>`: every position's traces written to files as the node's rockem.debug does
    let dump = arg("--dump");
    let rope = Rope::new(max_ctx);
    let k = kfd::Kfd::new(&elf, false);
    let place = vec![if harness::arg("--experts").map(|p| p.to_str().unwrap() == "host").unwrap_or(false) { 1u32 } else { 0 }; deepseek_v2_lite::N_ROUTED_EXPERTS];
    let mut m = kfd::KfdModel::new(k, dense, &experts[..deepseek_v2_lite::layout::STREAM], &rope, max_ctx, &place);
    let mut rng = common::testing::Rng::new(1);
    let fill: Vec<f32> = (0..32 * H).map(|_| rng.unit() * 0.1).collect();
    let mut h: Vec<f32> = (0..n * H).map(|_| rng.unit() * 0.1).collect();
    let mut logits = vec![0f32; n * VOCAB_SIZE];
    let forward = |m: &mut kfd::KfdModel, h: &mut Vec<f32>, pos: usize, n: usize, logits: &mut Vec<f32>| {
        let cos = &rope.cos[pos * QK_ROPE_HEAD_DIM..];
        let sin = &rope.sin[pos * QK_ROPE_HEAD_DIM..];
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let aw = attn::Weights { input_ln: &lw.input_ln, q_proj: lw.q_proj, kv_a: lw.kv_a, kv_a_ln: &lw.kv_a_ln, kv_b: &lw.kv_b, o_proj: lw.o_proj };
            m.step(l, &aw, h, pos, n, cos, sin, rope.softmax_scale, &mut common::trace::NoTrace);
            match lw.ffn {
                deepseek_v2_lite::Ffn::Dense { gate, up, down } => m.dense(l, &ffn::Dense { post_ln: &lw.post_ln, gate, up, down }, h, n, &mut common::trace::NoTrace),
                deepseek_v2_lite::Ffn::Moe { router, shared_gate, shared_up, shared_down } => m.moe(l, &ffn::Moe { post_ln: &lw.post_ln, router, shared_gate, shared_up, shared_down }, h, n, &mut common::trace::NoTrace),
            }
        }
        m.logits(&w.norm, w.lm_head, h, n, logits);
        m.sync();
    };
    // a prompt of `positions` positions as the node steps it: `n` a step from position 0,
    // the draw after each step, every position's trace against the first run's
    let positions: usize = arg("--positions").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(54);
    let _ = (fill, &mut h, &mut logits);
    let mut all: Vec<f32> = (0..positions * H).map(|_| rng.unit() * 0.1).collect();
    let mut lg = vec![0f32; n * VOCAB_SIZE];
    // `--against k`: the prompt in steps of k against the same prompt in steps of `--tokens`,
    // once — every step size must leave every position the same bits (the gate's claim, for
    // the sizes the wire happens to batch); with it `--repeat` is ignored
    let against: Option<usize> = arg("--against").map(|p| p.to_str().unwrap().parse().unwrap());
    // `--idle-ms`: the card left idle that long between steps, as a client pacing a
    // sequential prompt leaves it (the clocks fall and ramp again)
    let idle_ms: u64 = arg("--idle-ms").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(0);
    // `--prompt <ids file>`: the node's own path on real ids — the sink's logits by a decode
    // at position 0, then per run the bundle's sink seeded into row 0 and the ids from
    // position 1 through `State::run` in steps of `--tokens`, the draw after each
    if let Some(pf) = arg("--prompt") {
        let ids: Vec<u32> = std::fs::read_to_string(pf).unwrap().split_whitespace().map(|t| t.parse().unwrap()).collect();
        let ids: Vec<u32> = if ids[0] as usize == deepseek_v2_lite::BOS_TOKEN { ids[1..].to_vec() } else { ids };
        let sink_b = std::fs::read(bundle.join(deepseek_v2_lite::layout::SINK_FILE)).unwrap();
        let sink: Vec<f32> = sink_b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(sink.len(), deepseek_v2_lite::layout::SINK_LEN);
        let mut st = deepseek_v2_lite::State::new(max_ctx);
        st.decode(&w, deepseek_v2_lite::BOS_TOKEN as u32, 0, &mut m, &deepseek_v2_lite::Sampling::GREEDY, 0.0, &mut common::trace::NoTrace);
        let positions = ids.len();
        let run = |m: &mut kfd::KfdModel, st: &mut deepseek_v2_lite::State| -> Vec<(Vec<f32>, Vec<u32>, Vec<f32>)> {
            m.seed(&sink);
            let mut out = Vec::with_capacity(positions);
            let (mut pos, mut at) = (1usize, 0usize);
            while at < positions {
                let k = n.min(positions - at);
                if idle_ms > 0 {
                    std::thread::sleep(std::time::Duration::from_millis(idle_ms));
                }
                st.run(&w, &ids[at..at + k], pos, m, &mut common::trace::NoTrace);
                let u = common::sample::uniform_at(1, (pos + k - 1) as u64);
                let _ = m.draw(k - 1, &deepseek_v2_lite::SAMPLING, u, &st.logits[(k - 1) * VOCAB_SIZE..k * VOCAB_SIZE]);
                if mirror {
                    let mut rows = vec![0f32; deepseek_v2_lite::layout::POSITION];
                    for i in 0..k {
                        m.rows(i, &mut rows);
                    }
                }
                for i in 0..k {
                    let (mut layers, mut topk, mut l) = (vec![0f32; NUM_HIDDEN_LAYERS * H], vec![0u32; NUM_HIDDEN_LAYERS * TOP_K], vec![0f32; VOCAB_SIZE]);
                    m.token_trace(i, &mut layers, &mut topk, &mut l);
                    if let Some(d) = &dump {
                        let raw = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }.to_vec();
                        std::fs::write(d.join(format!("pos{}.layers.f32", pos + i)), raw(&layers)).unwrap();
                        std::fs::write(d.join(format!("pos{}.f32", pos + i)), raw(&l)).unwrap();
                    }
                    out.push((layers, topk, l));
                }
                pos += k;
                at += k;
            }
            out
        };
        let first = run(&mut m, &mut st);
        let mut bad = 0;
        let t = std::time::Instant::now();
        for r in 1..repeat {
            let now = run(&mut m, &mut st);
            let mut report = None;
            'find: for i in 0..positions {
                for l in 0..NUM_HIDDEN_LAYERS {
                    let (a, b) = (&first[i].0[l * H..(l + 1) * H], &now[i].0[l * H..(l + 1) * H]);
                    let picks = first[i].1[l * TOP_K..(l + 1) * TOP_K] == now[i].1[l * TOP_K..(l + 1) * TOP_K];
                    if a != b || !picks {
                        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                        report = Some(format!("position {} layer {l}: residual max |Δ| {d:.3e}, picks {}", i + 1, if picks { "same" } else { "differ" }));
                        break 'find;
                    }
                }
                if first[i].2 != now[i].2 {
                    report = Some(format!("position {}: every layer the same, logits differ", i + 1));
                    break;
                }
            }
            if let Some(s) = report {
                bad += 1;
                println!("run {r}: {s}");
            }
        }
        println!("the prompt's {positions} ids in steps of {n} × {repeat} runs in {:.0} s: {bad} runs differ from the first", t.elapsed().as_secs_f64());
        return;
    }
    let run = |m: &mut kfd::KfdModel, all: &mut Vec<f32>, lg: &mut Vec<f32>, n: usize| -> Vec<(Vec<f32>, Vec<u32>, Vec<f32>)> {
        let mut out = Vec::with_capacity(positions);
        let mut pos = 0;
        while pos < positions {
            let k = n.min(positions - pos);
            if idle_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(idle_ms));
            }
            let mut h: Vec<f32> = all[pos * H..(pos + k) * H].to_vec();
            forward(m, &mut h, pos, k, lg);
            let u = common::sample::uniform_at(1, (pos + k - 1) as u64);
            let _ = m.draw(k - 1, &deepseek_v2_lite::SAMPLING, u, &lg[(k - 1) * VOCAB_SIZE..k * VOCAB_SIZE]);
            for i in 0..k {
                let (mut layers, mut topk, mut l) = (vec![0f32; NUM_HIDDEN_LAYERS * H], vec![0u32; NUM_HIDDEN_LAYERS * TOP_K], vec![0f32; VOCAB_SIZE]);
                m.token_trace(i, &mut layers, &mut topk, &mut l);
                out.push((layers, topk, l));
            }
            pos += k;
        }
        out
    };
    let mut lg_b = vec![0f32; against.unwrap_or(n) * VOCAB_SIZE];
    let first = run(&mut m, &mut all, &mut lg, n);
    let mut bad = 0;
    let t = std::time::Instant::now();
    let repeat = if against.is_some() { 2 } else { repeat };
    for r in 1..repeat {
        let now = match against {
            Some(k) => run(&mut m, &mut all, &mut lg_b, k),
            None => run(&mut m, &mut all, &mut lg, n),
        };
        let mut report = None;
        'find: for i in 0..positions {
            for l in 0..NUM_HIDDEN_LAYERS {
                let (a, b) = (&first[i].0[l * H..(l + 1) * H], &now[i].0[l * H..(l + 1) * H]);
                let picks = first[i].1[l * TOP_K..(l + 1) * TOP_K] == now[i].1[l * TOP_K..(l + 1) * TOP_K];
                if a != b || !picks {
                    let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                    report = Some(format!("position {i} layer {l}: residual max |Δ| {d:.3e}, picks {}", if picks { "same" } else { "differ" }));
                    break 'find;
                }
            }
            if first[i].2 != now[i].2 {
                report = Some(format!("position {i}: every layer the same, logits differ"));
                break;
            }
        }
        if let Some(s) = report {
            bad += 1;
            println!("run {r}: {s}");
        }
    }
    match against {
        Some(k) => println!("{positions} positions in steps of {k} against steps of {n}: {}", if bad == 0 { "the same bits" } else { "DIFFER" }),
        None => println!("{positions} positions in steps of {n} × {repeat} runs in {:.0} s: {bad} runs differ from the first", t.elapsed().as_secs_f64()),
    }
}
