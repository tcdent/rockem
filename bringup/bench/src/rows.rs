//! bench-rows — the bf16 rows kernel of the decode path (`rows_h`: the norm folded in), the KDA
//! projections' realization, at the served shapes and widths, timed alone: 060's occupancy audit
//! at the KDA projections' shape (m 12,608 × K 4,096, the gate/qkv launch; m 4,096, the output
//! projection), n 1–4 (a token, a verify step's widths). Each row prints the launch's geometry and
//! its rate against the bytes a stream would move, so the occupancy's cost is on the printout:
//!
//!   bench-rows --elf <kernels.gfx1100.elf> [--m 12608,4096] [--n 1,2,3,4] [--iters 20] [--chain 1] [--kernel rows_h1] [--check]
//!
//! `--kernel` benches another kernel of `rows_h`'s arguments (the narrow twin `rows_h1`, its grid at one row a wave);
//! `--check` compares its output with `rows_h`'s on the same input, bit for bit. [--kernel rows_h]
//!
//! `--kernel rows_h1` benches the narrow twin (060: one row a wave, the under-occupied grids' fix);
//!
//! `--chain N` times N back-to-back launches a round against one sync — the serving shape's launch
//! cadence (43 a card a verify step), whose tails the single-launch round does not see. The round
//! depth and the rows a wave are the kernel's own constants (`common::gemv::{FEW, FLIGHT, ROWS}`,
//! 2026-09-19, the one-token era): a variant of them is a rebuild, and this bench is how a variant
//! is judged.
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::H;
use harness::arg;

fn list(name: &str, default: &[usize]) -> Vec<usize> {
    arg(name).map_or(default.to_vec(), |v| {
        v.to_str().unwrap().split(',').map(|s| s.trim().parse().expect(name)).collect()
    })
}

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let ms = list("--m", &[12608, 4096]);
    let ns = list("--n", &[1, 2, 3, 4]);
    let iters: usize = arg("--iters").map_or(20, |v| v.to_str().unwrap().parse().unwrap());
    let chain: usize = arg("--chain").map_or(1, |v| v.to_str().unwrap().parse().unwrap());
    let k = kfd::Kfd::open(&elf, false, 0);
    let name = arg("--kernel").map_or("rows_h".to_owned(), |v| v.to_str().unwrap().to_owned());
    let kern = k.kernel(&name);
    // the rows a wave the kernel was built for: the narrow twins (`*1`, 060) take one, the others `ROWS`; the grid has to
    // match, or a narrow kernel launched on the wide grid computes a quarter of the rows and looks four times faster
    let r = if name.ends_with('1') { 1 } else { common::gemv::ROWS };
    // `--check`: the kernel's output against `rows_h`'s on the same input, bit for bit, at every (m, n)
    let check = harness::flag("--check");
    let base = if check { Some(k.kernel("rows_h")) } else { None };
    let mut rng = common::testing::Rng::new(7);
    let n_max = *ns.iter().max().unwrap();
    let m_max = *ms.iter().max().unwrap();
    let x: Vec<f32> = (0..n_max * H).map(|_| rng.unit()).collect();
    let w: Vec<u16> = (0..m_max * H).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
    let ln: Vec<u16> = (0..H).map(|_| (1.0f32.to_bits() >> 16) as u16).collect();
    let raw = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    let raw16 = |v: &[u16]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) };
    let (bx, bw, bln, bout) = (k.vram_visible(n_max * H * 4), k.vram_visible(m_max * H * 2), k.vram_visible(H * 2), k.vram_visible(n_max * m_max * 4));
    k.upload(&bx, 0, raw(&x));
    k.upload(&bw, 0, raw16(&w));
    k.upload(&bln, 0, raw16(&ln));
    println!("bench-rows: {name} (normed) × K {H} bf16, WG {} threads ({} waves of 32), {r} rows a wave past one token, FLIGHT {} steps a round at one token, FEW {} a verify step's width; chain {chain}", g::WG, g::WG / 32, common::gemv::FLIGHT, common::gemv::FEW);
    println!("{:>6} {:>3} {:>6} {:>7} {:>10} {:>10} {:>9} {:>8}", "m", "n", "wgs", "LDS kB", "median µs", "min µs", "GB/s", "TFLOP/s");
    for &m in &ms {
        for &n in &ns {
            // the served launch's geometry: the grid the node's `rows_grid` gives, the LDS it stages
            let per_wg = common::gemv::rows_per_wg(n, r);
            let wgs = m.div_ceil(per_wg);
            let lds = g::normed_rows_lds(H, n);
            let run = || {
                let mut a = kfd::Args::new();
                a.ptr(bx.va).ptr(bln.va).ptr(bw.va).ptr(bout.va).u32(m as u32).u32(n as u32);
                for _ in 0..chain {
                    k.launch(&kern, wgs as u32, g::WG, lds as u32, &a);
                }
                k.sync();
            };
            run();
            let mut us: Vec<f64> = Vec::new();
            for _ in 0..iters {
                let t = std::time::Instant::now();
                run();
                us.push(t.elapsed().as_secs_f64() * 1e6 / chain as f64);
            }
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut verdict = String::new();
            if let Some(bk) = &base {
                let mut mine = vec![0u32; n * m];
                kfd::read(&bout, 0, &mut mine);
                let mut a = kfd::Args::new();
                a.ptr(bx.va).ptr(bln.va).ptr(bw.va).ptr(bout.va).u32(m as u32).u32(n as u32);
                k.launch(bk, m.div_ceil(common::gemv::rows_per_wg(n, common::gemv::ROWS)) as u32, g::WG, lds as u32, &a);
                k.sync();
                let mut want = vec![0u32; n * m];
                kfd::read(&bout, 0, &mut want);
                let diff = mine.iter().zip(&want).filter(|(a, b)| a != b).count();
                verdict = if diff == 0 { "  bits = rows_h".to_owned() } else { format!("  {diff} of {} words differ from rows_h", n * m) };
            }
            let bytes = m * H * 2;
            let flop = 2.0 * m as f64 * n as f64 * H as f64;
            println!("{m:>6} {n:>3} {wgs:>6} {:>7.1} {:>10.1} {:>10.1} {:>9.0} {:>8.2}{verdict}", lds as f64 / 1e3, us[iters / 2], us[0], bytes as f64 / us[iters / 2] / 1e3, flop / us[iters / 2] / 1e6);
        }
    }
}
