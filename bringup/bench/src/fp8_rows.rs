//! bench-fp8-rows — the FP8 rows kernels of the model's code object (`ffn_gu`: the shared experts' gate|up
//! with the norm folded in; `ffn_down_shared`) alone, at 1, 2, 4, 8 and 16 tokens: the per-token cost curve
//! of the few-token pass (docs/queue/008 — a verify step's FP8 launches ran 1.7–2× a token's).
//!
//!   bench-fp8-rows --elf <kernels.gfx1100.elf> [--iters 40]
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::ffn::gfx11_consts::{ROWS_GU, SHARED_GU_ROWS};
use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::{D_FF, H, N_SHARED_EXPERTS};
use harness::arg;

const TOKENS: usize = 32; // a pass

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let iters: usize = arg("--iters").map_or(40, |v| v.to_str().unwrap().parse().unwrap());
    let k = kfd::Kfd::open(&elf, false, 0);
    let mut rng = common::testing::Rng::new(11);
    let kd = N_SHARED_EXPERTS * D_FF; // the down kernel's K; its input is the gate|up row, 2·kd a token
    let x: Vec<f32> = (0..TOKENS * 2 * kd.max(H)).map(|_| rng.unit()).collect();
    let bx = k.vram_visible(x.len() * 4);
    k.upload(&bx, 0, unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) });
    let ln: Vec<u16> = (0..H).map(|_| (1.0f32.to_bits() >> 16) as u16).collect();
    let bln = k.vram_visible(H * 2);
    k.upload(&bln, 0, unsafe { std::slice::from_raw_parts(ln.as_ptr() as *const u8, H * 2) });
    let mut fp8 = |m: usize, kk: usize| {
        let w: Vec<u8> = (0..m * kk).map(|_| (rng.unit() * 40.0) as u8 & 0x7f).collect();
        let s: Vec<f32> = (0..(m / 128 + 1) * (kk / 128 + 1)).map(|_| 0.01).collect();
        let (bw, bs) = (k.vram_visible(w.len()), k.vram_visible(s.len() * 4));
        k.upload(&bw, 0, &w);
        k.upload(&bs, 0, unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) });
        (bw, bs)
    };
    let (gu_w, gu_s) = fp8(SHARED_GU_ROWS, H);
    let (dn_w, dn_s) = fp8(H, kd);
    let out = k.vram_visible(TOKENS * SHARED_GU_ROWS.max(H) * 4);
    println!("bench-fp8-rows: ffn_gu {SHARED_GU_ROWS} × {H} ({:.1} MB), ffn_down_shared {H} × {kd} ({:.1} MB)", (SHARED_GU_ROWS * H) as f64 / 1e6, (H * kd) as f64 / 1e6);
    println!("{:>8} {:>22} {:>22}", "tokens", "ffn_gu µs (a token)", "ffn_down_shared µs (a token)");
    for n in [1usize, 2, 4, 8, 16, 32] {
        let time = |kern: &kfd::Kernel, norm: bool, w: &kfd::Mem, s: &kfd::Mem, m: usize, kw: usize, r: usize| -> f64 {
            let run = || {
                let mut a = kfd::Args::new();
                a.ptr(bx.va);
                if norm {
                    a.ptr(bln.va);
                }
                a.ptr(w.va).ptr(s.va).ptr(out.va).u32(m as u32).u32(n as u32);
                k.launch(kern, g::rows_grid(m, n, r), g::WG, g::fp8_rows_lds(kw, n) as u32, &a);
                k.sync();
            };
            run();
            let mut us: Vec<f64> = (0..iters).map(|_| {
                let t = std::time::Instant::now();
                run();
                t.elapsed().as_secs_f64() * 1e6
            }).collect();
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            us[iters / 2]
        };
        let gu = time(&k.kernel("ffn_gu"), true, &gu_w, &gu_s, SHARED_GU_ROWS, H, ROWS_GU);
        let dn = time(&k.kernel("ffn_down_shared"), false, &dn_w, &dn_s, H, kd, common::gemv::ROWS);
        println!("{n:>8} {gu:>12.1} ({:>6.1}) {dn:>14.1} ({:>6.1})", gu / n as f64, dn / n as f64);
    }
}
