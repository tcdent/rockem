//! bench-memtest — the card's memory through the BAR: host-visible VRAM taken
//! in 1 GB pieces up to `--gb`, each written with an address-derived pattern
//! and read back, then the inverse pattern, `--passes` times. Every mismatch
//! is printed with its address and the bits that turned. For: one word of a
//! token's residual wrong in ~1 of 15 gate runs (2026-09-19), reproduced by
//! nothing the benches run.
//!
//!   bench-memtest --elf <kernels.gfx1100.elf> [--gb 16] [--passes 2]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
use harness::{arg, arg_required};

fn main() {
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let gb: usize = arg("--gb").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(16);
    let passes: usize = arg("--passes").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(2);
    let k = kfd::Kfd::new(&elf, false);
    const PIECE: usize = 1 << 30;
    let pieces: Vec<kfd::Mem> = (0..gb).map(|_| k.vram_visible(PIECE)).collect();
    println!("{} GB of host-visible VRAM taken", pieces.len());
    let mut pattern = vec![0u64; PIECE / 8];
    let mut back = vec![0u64; PIECE / 8];
    let mut bad = 0usize;
    for pass in 0..passes {
        for (i, m) in pieces.iter().enumerate() {
            let base = (m.va as u64) ^ ((pass as u64) << 40);
            for (j, w) in pattern.iter_mut().enumerate() {
                let a = base.wrapping_add((j * 8) as u64);
                *w = if pass % 2 == 0 { a.wrapping_mul(0x9E3779B97F4A7C15) } else { !a.wrapping_mul(0x9E3779B97F4A7C15) };
            }
            let t = std::time::Instant::now();
            k.upload(m, 0, unsafe { std::slice::from_raw_parts(pattern.as_ptr() as *const u8, PIECE) });
            let tw = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            kfd::read(m, 0, &mut back);
            let tr = t.elapsed().as_secs_f64();
            let mut here = 0;
            for j in 0..pattern.len() {
                if back[j] != pattern[j] {
                    here += 1;
                    if here <= 8 {
                        println!("  pass {pass} piece {i} +{:#x}: wrote {:#018x} read {:#018x} (bits {:#018x})", j * 8, pattern[j], back[j], pattern[j] ^ back[j]);
                    }
                }
            }
            bad += here;
            println!("pass {pass} piece {i}: written at {:.1} GB/s, read at {:.2} GB/s, {here} words wrong", 1.0 / tw, 1.0 / tr);
        }
    }
    println!("memtest: {} GB × {passes} passes, {bad} words wrong", pieces.len());
}
