//! bench-wmma — the card's matrix and FMA rates with only registers in the way (bringup/kernels):
//! iu8 WMMA, bf16 WMMA, scalar f32 FMA, as TOPS/TFLOPS over a grid that fills the card.
//!
//!   bench-wmma --elf <bringup_kernels.elf> [--iters 4096] [--wgs 672]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::arg;

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let iters: u32 = arg("--iters").map_or(4096, |v| v.to_str().unwrap().parse().unwrap());
    let wgs: u32 = arg("--wgs").map_or(672, |v| v.to_str().unwrap().parse().unwrap());
    let k = kfd::Kfd::open(&elf, false, 0);
    let threads = wgs as usize * 256;
    let out = k.vram(threads * 4 * 32);
    let waves = threads as f64 / 32.0;
    for (name, ops_per_round_wave) in [("rate_wmma_iu8", 4.0 * 2.0 * 4096.0), ("rate_wmma_bf16", 4.0 * 2.0 * 4096.0), ("rate_fma_f32", 4.0 * 2.0 * 32.0)] {
        let kern = k.kernel(name);
        let run = |it: u32| {
            let mut a = kfd::Args::new();
            a.ptr(out.va).u32(it);
            k.launch(&kern, wgs, 256, 0, &a);
            k.sync();
        };
        run(16);
        let mut us: Vec<f64> = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            run(iters);
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ops = waves * iters as f64 * ops_per_round_wave;
        println!("{name:<16} {:8.1} µs  {:7.1} T{}/s  ({wgs} workgroups × 8 waves, {iters} rounds of 4 chains)", us[2], ops / us[2] / 1e6, if name.contains("fma") || name.contains("bf16") { "FLOP" } else { "OP" });
    }
}
