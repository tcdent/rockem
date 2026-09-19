//! bench-draw — the card's `sample` kernel against its CPU twin,
//! `common::sample::draw`, row by row: real logits from the goldens
//! (`--logits <f32 file>`, `[rows][VOCAB_SIZE]`) and the shapes real logits
//! never take — every logit equal, hundreds tied at the top, ties exactly at
//! the k-th — under the model's recipe, each row drawn with the uniform of its
//! position. Reports mismatches and the kernel's time inside from the timeline.
//!
//!   bench-draw --elf <kernels.gfx1100.elf> [--logits rows.f32]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
use deepseek_v2_lite::{ffn, SAMPLING, SEED, VOCAB_SIZE};
use harness::{arg, arg_required};

fn main() {
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let k = kfd::Kfd::new(&elf, true);
    let sample = k.kernel("sample");
    let d_logits = k.vram_visible(VOCAB_SIZE * 4);
    let d_out = k.host(64);
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    if let Some(p) = arg("--logits") {
        let raw = std::fs::read(p).unwrap();
        let all: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        for (i, r) in all.chunks_exact(VOCAB_SIZE).enumerate() {
            rows.push((format!("golden row {i}"), r.to_vec()));
        }
    }
    let mut rng = common::testing::Rng::new(3);
    let mut rand = |scale: f32| (0..VOCAB_SIZE).map(|_| rng.unit() * scale).collect::<Vec<f32>>();
    rows.push(("random 0..10".into(), rand(10.0)));
    rows.push(("random 0..1".into(), rand(1.0)));
    rows.push(("all equal".into(), vec![1.0; VOCAB_SIZE]));
    let mut v = rand(1.0);
    for i in 0..300 {
        v[i * 331 % VOCAB_SIZE] = 5.0;
    }
    rows.push(("300 tied at the top".into(), v));
    let mut v = rand(1.0);
    for i in 0..40 {
        v[i * 977 % VOCAB_SIZE] = 5.0 + i as f32;
    }
    for i in 0..30 {
        v[(i * 1237 + 5) % VOCAB_SIZE] = 4.5; // ties exactly at the 50th: 40 above, 30 at
    }
    rows.push(("ties at the k-th".into(), v));
    let mut v = rand(1.0);
    v[VOCAB_SIZE - 1] = 9.0;
    v[0] = 8.0;
    rows.push(("top at the ends".into(), v));
    let mut v = vec![-1e30f32; VOCAB_SIZE];
    v[77] = 0.0;
    rows.push(("one finite-ish".into(), v));

    let mut bad = 0;
    let mut inside = Vec::new();
    let mut phases: [Vec<f32>; 4] = Default::default();
    for (i, (name, row)) in rows.iter().enumerate() {
        let u = common::sample::uniform_at(SEED, i as u64);
        let want = common::sample::draw(row, &SAMPLING, u);
        k.upload(&d_logits, 0, unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len() * 4) });
        let mut a = kfd::Args::new();
        a.ptr(d_logits.va).ptr(d_out.va).f64(u).f32(SAMPLING.temperature).u32(SAMPLING.top_k as u32).f32(SAMPLING.top_p);
        k.launch(&sample, 1, 256, ffn::gfx11_consts::SAMPLE_LDS as u32, &a);
        k.sync();
        let mut got = [0u32; 1];
        kfd::read(&d_out, 0, &mut got);
        for (kname, from, to) in k.timeline() {
            if kname == "sample" {
                inside.push(to - from);
            }
        }
        for (kname, slots) in k.stats() {
            if kname == "sample" {
                for (j, v) in slots.iter().enumerate() {
                    phases[j].push(v.iter().cloned().fold(0f32, f32::max)); // the slowest wave: the critical path
                }
            }
        }
        if got[0] as usize != want {
            bad += 1;
            println!("MISMATCH {name}: card {} cpu {want} (u {u:.6})", got[0]);
        }
    }
    inside.sort();
    println!("{} rows, {bad} mismatches; sample kernel inside: median {:.0} µs, max {:.0} µs", rows.len(), inside[inside.len() / 2] as f64 / 100.0, inside[inside.len() - 1] as f64 / 100.0);
    if !phases[0].is_empty() {
        // with a stats code object: the phases in shader cycles, the slowest wave of each, median over rows
        for (j, name) in deepseek_v2_lite::stat_names("sample").iter().enumerate() {
            let mut v = phases[j].clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("  {name:<18} median {:>8.0} cycles  max {:>8.0}", v[v.len() / 2], v[v.len() - 1]);
        }
    }
}
