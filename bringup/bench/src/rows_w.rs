//! bench-rows-w — the WMMA rows kernels of the model's code object (`rows_h_w`: norm folded in; `rows_part_w`)
//! on synthetic data at a served shape, timed alone: the 009 efficiency round's loop (docs/queue/009). The
//! kernel's K is the model's H; `--m` is the rows, `--n` the tokens of the step.
//!
//!   bench-rows-w --elf <kernels.gfx1100.elf> --m 6144 [--n 512] [--iters 20]
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::H;
use harness::arg;

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let m: usize = arg("--m").map_or(6144, |v| v.to_str().unwrap().parse().unwrap());
    let n: usize = arg("--n").map_or(512, |v| v.to_str().unwrap().parse().unwrap());
    let iters: usize = arg("--iters").map_or(20, |v| v.to_str().unwrap().parse().unwrap());
    let k = kfd::Kfd::open(&elf, false, 0);
    let mut rng = common::testing::Rng::new(7);
    let x: Vec<f32> = (0..n * H).map(|_| rng.unit()).collect();
    let w: Vec<u16> = (0..m * H).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
    let ln: Vec<u16> = (0..H).map(|_| (1.0f32.to_bits() >> 16) as u16).collect();
    let raw = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let raw16 = |v: &[u16]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    let (bx, bw, bln, bout) = (k.vram_visible(n * H * 4), k.vram_visible(m * H * 2), k.vram_visible(H * 2), k.vram_visible(n * m * 4));
    k.upload(&bx, 0, raw(&x));
    k.upload(&bw, 0, raw16(&w));
    k.upload(&bln, 0, raw16(&ln));
    println!("bench-rows-w: m {m} × K {H} bf16 ({:.1} MB), n {n} tokens, grid {} × {} threads, LDS {} B", (m * H * 2) as f64 / 1e6, g::wmma_grid(m, n), g::WG, g::WMMA_LDS);
    for (name, norm) in [("rows_h_w", true), ("rows_part_w", false)] {
        let kern = k.kernel(name);
        let run = || {
            let mut a = kfd::Args::new();
            a.ptr(bx.va);
            if norm {
                a.ptr(bln.va);
            }
            a.ptr(bw.va).ptr(bout.va).u32(m as u32).u32(n as u32);
            k.launch(&kern, g::wmma_grid(m, n), g::WG, g::WMMA_LDS, &a);
            k.sync();
        };
        run();
        let mut us: Vec<f64> = Vec::new();
        for _ in 0..iters {
            let t = std::time::Instant::now();
            run();
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let flop = 2.0 * m as f64 * n as f64 * H as f64;
        println!("{name:<12} {:8.1} µs median (min {:.1})  {:6.2} TFLOP/s  weights {:5.1} GB/s", us[iters / 2], us[0], flop / us[iters / 2] / 1e6, (m * H * 2) as f64 / us[iters / 2] / 1e3);
    }
}
