//! bench-mla-attend — the card's indexer scoring (`ix_score`), selection (`ix_select`) and sparse
//! attend (`mla_attend`) alone, at long positions, over a prefill step and a verify step: what
//! docs/queue/018 (attend on WMMA) and 019 (scoring on WMMA) would move. Synthetic latents, pools
//! and queries; the selection is the real kernel's, so the attend sees real list shapes.
//!
//!   bench-mla-attend --elf <kernels.gfx1100.elf> [--pos 18432,65536,262144] [--tokens 2,512] [--iters 10] [--ah 1]
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::mla::gfx11_consts::{IX_ROWS, SELECT_LDS};
use glm_5_3_flash::{DENSE_BELOW, INDEX_HEAD_DIM, INDEX_KPOOL, INDEX_N_HEADS, KVB_HEAD, KV_LORA_RANK, MLA_HEADS, Q_HEAD, V_HEAD_DIM};
use harness::arg;

fn list(name: &str, default: &[usize]) -> Vec<usize> {
    arg(name).map_or(default.to_vec(), |v| v.to_str().unwrap().split(',').map(|x| x.parse().unwrap()).collect())
}

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let positions = list("--pos", &[18432, 65536, 262144]);
    let tokens = list("--tokens", &[2, 512]);
    let iters: usize = arg("--iters").map_or(10, |v| v.to_str().unwrap().parse().unwrap());
    // the attend's heads a workgroup (`ATTEND_H` of the code object under test; the crate's is 1): grid and LDS follow
    let ah: usize = arg("--ah").map_or(1, |v| v.to_str().unwrap().parse().unwrap());
    let attend_grid = (MLA_HEADS / ah) as u32;
    let att_slots = DENSE_BELOW.div_ceil(256) * 256;
    let attend_lds = (4 * (ah * Q_HEAD + ah * KV_LORA_RANK + ah * att_slots + ah * 8)) as u32;
    let k = kfd::Kfd::open(&elf, false, 0);
    let mut rng = common::testing::Rng::new(3);
    let maxpos = *positions.iter().max().unwrap();
    let maxn = *tokens.iter().max().unwrap();
    let npools = (maxpos + maxn) / INDEX_KPOOL + 1;
    let upload_f32 = |m: &kfd::Mem, v: &[f32]| k.upload(m, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) });
    let upload_u16 = |m: &kfd::Mem, v: &[u16]| k.upload(m, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) });
    // the attend's inputs
    let q = k.vram_visible(maxn * MLA_HEADS * Q_HEAD * 4);
    upload_f32(&q, &(0..maxn * MLA_HEADS * Q_HEAD).map(|_| rng.unit()).collect::<Vec<f32>>());
    let kvb = k.vram_visible(MLA_HEADS * KVB_HEAD * KV_LORA_RANK * 2);
    upload_u16(&kvb, &(0..MLA_HEADS * KVB_HEAD * KV_LORA_RANK).map(|_| (rng.unit().to_bits() >> 16) as u16).collect::<Vec<u16>>());
    let lat = k.vram_visible((maxpos + maxn) * KV_LORA_RANK * 2);
    {
        let v: Vec<u16> = (0..(maxpos + maxn) * KV_LORA_RANK).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
        upload_u16(&lat, &v);
    }
    let sel = k.vram(maxn * DENSE_BELOW * 4);
    let counts = k.vram_visible(maxn * 4);
    let heads = k.vram(maxn * MLA_HEADS * V_HEAD_DIM * 4);
    // the indexer's
    let iq = k.vram_visible(maxn * INDEX_N_HEADS * INDEX_HEAD_DIM * 4);
    upload_f32(&iq, &(0..maxn * INDEX_N_HEADS * INDEX_HEAD_DIM).map(|_| rng.unit()).collect::<Vec<f32>>());
    let ix = k.vram_visible(maxn * IX_ROWS * 4);
    upload_f32(&ix, &(0..maxn * IX_ROWS).map(|_| rng.unit().abs()).collect::<Vec<f32>>());
    let pools = k.vram_visible(npools * INDEX_HEAD_DIM * 2);
    upload_u16(&pools, &(0..npools * INDEX_HEAD_DIM).map(|_| (rng.unit().to_bits() >> 16) as u16).collect::<Vec<u16>>());
    let score = k.vram(maxn * npools * 4);
    let (scale, wscale) = glm_5_3_flash::mla::score_scales();
    let (k_score, k_select, k_attend) = (k.kernel("ix_score"), k.kernel("ix_select"), k.kernel("mla_attend"));
    println!("bench-mla-attend: {MLA_HEADS} heads, latents {KV_LORA_RANK} bf16, {INDEX_N_HEADS}×{INDEX_HEAD_DIM} indexer, {npools} pools allocated; µs a launch (a token)");
    println!("{:>9} {:>6} {:>22} {:>22} {:>22}", "positions", "tokens", "ix_score", "ix_select", "mla_attend");
    for &pos in &positions {
        for &n in &tokens {
            let nb = ((pos + n) / INDEX_KPOOL).div_ceil(256);
            let time = |f: &dyn Fn()| -> f64 {
                f();
                k.sync();
                let mut us: Vec<f64> = (0..iters).map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    k.sync();
                    t.elapsed().as_secs_f64() * 1e6
                }).collect();
                us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                us[iters / 2]
            };
            let t_score = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(iq.va).ptr(ix.va).ptr(pools.va).ptr(score.va).u32(pos as u32).u32(nb as u32).u32(npools as u32).f32(scale).f32(wscale);
                k.launch(&k_score, (n * nb) as u32, g::WG, 0, &a);
            });
            let t_select = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(score.va).u32(npools as u32).u32(pos as u32).ptr(sel.va).ptr(counts.va);
                k.launch(&k_select, n as u32, g::WG, SELECT_LDS as u32, &a);
            });
            let t_attend = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(q.va).ptr(kvb.va).ptr(lat.va).ptr(sel.va).ptr(counts.va).ptr(heads.va);
                k.launch(&k_attend, attend_grid * n as u32, g::WG, attend_lds, &a);
            });
            let mut c = vec![0u32; n];
            unsafe { std::ptr::copy_nonoverlapping(counts.ptr() as *const u32, c.as_mut_ptr(), n) };
            println!("{pos:>9} {n:>6} {t_score:>12.1} ({:>7.2}) {t_select:>12.1} ({:>7.2}) {t_attend:>12.1} ({:>7.2})   selected {}", t_score / n as f64, t_select / n as f64, t_attend / n as f64, c[n - 1]);
        }
    }
}
