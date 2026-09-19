//! bench-attn — per-token attention on the card at a given context length
//! (27 layers of proj/mla/out with a sync each, as the node runs them) and
//! the cost of seeding that many positions — the swap-in.
//!
//!   bench-attn --bundle <dir> --elf <kernels.elf> [--ctx 2048] [--iters 30]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use deepseek_v2_lite::layout;
use deepseek_v2_lite::{attn, Attention, Rope, Weights, H, NUM_HIDDEN_LAYERS};
use harness::{arg, arg_required};

fn main() {
    let bundle = arg_required("--bundle");
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let ctx: usize = arg("--ctx").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(2048);
    let iters: usize = arg("--iters").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(30);
    let f = std::fs::File::open(bundle.join(layout::DENSE_FILE)).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let (_, dense, _) = unsafe { mm[..].align_to::<u16>() };
    let w = Weights::from_dense(dense);
    let ef = std::fs::File::open(bundle.join(layout::EXPERTS_FILE)).unwrap();
    let emm = unsafe { memmap2::Mmap::map(&ef).unwrap() };
    let k = kfd::Kfd::new(&elf, true);
    let rope = Rope::new(ctx);
    let mut attention = kfd::KfdModel::new(k.clone(), dense, &emm[..layout::STREAM], &rope, ctx);

    let mut rng = common::testing::Rng::new(3);
    let rows: Vec<f32> = (0..ctx * layout::POSITION).map(|_| rng.unit() * 0.1).collect();
    let t = std::time::Instant::now();
    attention.seed(&rows);
    let seed_ms = t.elapsed().as_secs_f64() * 1e3;

    let pos = ctx - 1;
    let mut h: Vec<f32> = (0..H).map(|_| rng.unit()).collect();
    let mut tr = common::trace::NoTrace;
    // the card's realization reads its weights from `d_dense` and ignores this argument
    // 27 layers posted, one wait: the node's shape of a token's attention
    let mut token = |attention: &mut kfd::KfdModel, h: &mut [f32]| {
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let aw = attn::Weights { input_ln: &lw.input_ln, q_proj: lw.q_proj, kv_a: lw.kv_a, kv_a_ln: &lw.kv_a_ln, kv_b: &lw.kv_b, o_proj: lw.o_proj };
            attention.step(l, &aw, h, pos, 1, &rope.cos[pos * deepseek_v2_lite::QK_ROPE_HEAD_DIM..], &rope.sin[pos * deepseek_v2_lite::QK_ROPE_HEAD_DIM..], rope.softmax_scale, &mut tr);
        }
        attention.sync();
    };
    token(&mut attention, &mut h);
    let b = harness::bench(iters, || token(&mut attention, &mut h));
    let bytes = ctx as f64 * layout::POSITION as f64 * 4.0;
    println!("ctx {ctx}: attention per token {:.3} ms (min {:.3}) = {:.1} GB/s of cache rows; seed of {ctx} positions {seed_ms:.1} ms ({:.1} GB/s)", b.median.as_secs_f64() * 1e3, b.min.as_secs_f64() * 1e3, bytes / b.median.as_secs_f64() / 1e9, bytes / seed_ms / 1e6);
    // with a stats code object: the attention kernels' phases, cycles per wave (mean and max over waves), and the timeline per kernel
    token(&mut attention, &mut h);
    let tl = attention.timeline();
    let mut per: std::collections::BTreeMap<&str, (usize, f64)> = Default::default();
    for &(name, s, e) in &tl {
        let en = per.entry(name).or_default();
        en.0 += 1;
        en.1 += (e - s) as f64 / 100.0;
    }
    for (name, (n, us)) in &per {
        println!("  {name:<12} ×{n:<4} inside {us:>8.0} µs ({:>6.1} each)", us / *n as f64);
    }
    let st = attention.stats();
    if !st.is_empty() {
        let mut agg: std::collections::BTreeMap<(&str, usize), (f64, f32, usize)> = Default::default();
        for (name, slots) in &st {
            for (i, v) in slots.iter().enumerate() {
                if !deepseek_v2_lite::stat_names(name)[i].is_empty() && !v.is_empty() {
                    let e = agg.entry((name, i)).or_default();
                    e.0 += v.iter().map(|&x| x as f64).sum::<f64>();
                    e.1 = e.1.max(v.iter().cloned().fold(0f32, f32::max));
                    e.2 += v.len();
                }
            }
        }
        for ((name, i), (sum, max, n)) in &agg {
            println!("  {name:<12} {:<14} waves {n:>7}  mean {:>9.0} cycles  max {:>9.0}", deepseek_v2_lite::stat_names(name)[*i], sum / *n as f64, max);
        }
    }
}
