//! bench-sample — the draw over the vocabulary under the model's recipe
//! (`common::sample::draw`: top-k, top-p, the f64 walk), the argmax, and a
//! 400 KB copy, on vocabulary-sized f32s. With `--draw <f32 file> <u>` it is
//! the CPU twin as a command instead: the token drawn from those logits with
//! that uniform, for a checker in another language.
use deepseek_v2_lite::{SAMPLING, VOCAB_SIZE};

fn main() {
    if let Some(path) = harness::arg("--draw") {
        let raw = std::fs::read(&path).unwrap();
        let logits: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let u: f64 = std::env::args().last().unwrap().parse().unwrap();
        println!("{}", common::sample::draw(&logits, &SAMPLING, u));
        return;
    }
    let mut rng = common::testing::Rng::new(1);
    let logits: Vec<f32> = (0..VOCAB_SIZE).map(|_| rng.unit() * 10.0).collect();
    let mut srng = common::sample::Rng::new(7);
    let mut buf = logits.clone();
    let b = harness::bench(50, || { std::hint::black_box(common::sample::sample(&logits, &SAMPLING, &mut srng)); });
    println!("draw (top-{} of {VOCAB_SIZE}, top-p {}, the walk): {:.1} µs", SAMPLING.top_k, SAMPLING.top_p, b.median.as_secs_f64() * 1e6);
    let b = harness::bench(50, || { std::hint::black_box(common::softmax::argmax(&logits)); });
    println!("argmax: {:.1} µs", b.median.as_secs_f64() * 1e6);
    let b = harness::bench(50, || buf.copy_from_slice(&logits));
    println!("copy 400 KB: {:.1} µs", b.median.as_secs_f64() * 1e6);
}
