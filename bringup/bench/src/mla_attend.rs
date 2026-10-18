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
use glm_5_3_flash::mla::gfx11_consts::{aw_chunk, aw_chunks, ixw_grid, AW_HEAD_TILES, AW_LDS, AW_MERGE_LDS, AW_PART, AW_WG, IXW_LDS, IX_ROWS, SELECT_LDS};
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
    let sel = k.vram_visible(maxn * DENSE_BELOW * 4);
    let k_score_w = k.kernel("ix_score_w");
    let counts = k.vram_visible(maxn * 4);
    let heads = k.vram_visible(maxn * MLA_HEADS * V_HEAD_DIM * 4);
    let heads_w = k.vram_visible(maxn * MLA_HEADS * V_HEAD_DIM * 4);
    // the WMMA attend's chunking: derived from the step, or forced (`--chunk <positions>`; 0 = derived)
    let chunk_override: usize = arg("--chunk").map_or(0, |v| v.to_str().unwrap().parse().unwrap());
    let max_chunks = DENSE_BELOW.div_ceil(if chunk_override > 0 { chunk_override } else { aw_chunk(1).min(aw_chunk(512)) });
    // the partials exist only for chunked launches (a verify step's few tokens, or a forced chunk): sized so,
    // not for 512 tokens × 9 chunks (600 MB — beside the node's residency that evicted buffers under the CP)
    let part_tokens = if chunk_override > 0 { maxn } else { maxn.min(glm_5_3_flash::SPEC_TOKENS) };
    let part = k.vram_visible(part_tokens * AW_HEAD_TILES * max_chunks * AW_PART * 4);
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
    let (k_attend_w, k_merge, k_qabs, k_uv, k_uv_s) = (k.kernel("mla_attend_w"), k.kernel("mla_attend_merge"), k.kernel("mla_qabs"), k.kernel("mla_uv_w"), k.kernel("mla_uv"));
    let k_qabs_w = k.kernel("mla_qabs_w");
    let qabs = k.vram(maxn * MLA_HEADS * KV_LORA_RANK * 2);
    let olat = k.vram(maxn * MLA_HEADS * KV_LORA_RANK * 4);
    println!("bench-mla-attend: {MLA_HEADS} heads, latents {KV_LORA_RANK} bf16, {INDEX_N_HEADS}×{INDEX_HEAD_DIM} indexer, {npools} pools allocated; µs a launch (a token)");
    println!("{:>9} {:>6} {:>22} {:>22} {:>22} {:>26}", "positions", "tokens", "ix_score", "ix_select", "mla_attend", "qabs+attend_w(+merge)+uv_w");
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
            // the WMMA scorer: its time, then the selection it leads to against the scalar scorer's
            let t_score_w = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(iq.va).ptr(ix.va).ptr(pools.va).ptr(score.va).u32(n as u32).u32(((pos + n) / INDEX_KPOOL) as u32).u32(npools as u32).f32(scale).f32(wscale);
                k.launch(&k_score_w, ixw_grid(n, (pos + n) / INDEX_KPOOL), 256, IXW_LDS as u32, &a);
            });
            let select = || {
                let mut a = kfd::Args::new();
                a.ptr(score.va).u32(npools as u32).u32(pos as u32).ptr(sel.va).ptr(counts.va);
                k.launch(&k_select, n as u32, g::WG, SELECT_LDS as u32, &a);
                k.sync();
            };
            select();
            let sel_w: Vec<u32> = unsafe { std::slice::from_raw_parts(sel.ptr() as *const u32, n * DENSE_BELOW) }.to_vec();
            let cnt_w: Vec<u32> = unsafe { std::slice::from_raw_parts(counts.ptr() as *const u32, n) }.to_vec();
            let t_select = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(score.va).u32(npools as u32).u32(pos as u32).ptr(sel.va).ptr(counts.va);
                k.launch(&k_select, n as u32, g::WG, SELECT_LDS as u32, &a);
            });
            // the scalar scorer's selection (the one the attends below use)
            {
                let mut a = kfd::Args::new();
                a.ptr(iq.va).ptr(ix.va).ptr(pools.va).ptr(score.va).u32(pos as u32).u32(nb as u32).u32(npools as u32).f32(scale).f32(wscale);
                k.launch(&k_score, (n * nb) as u32, g::WG, 0, &a);
                select();
            }
            let sel_s = unsafe { std::slice::from_raw_parts(sel.ptr() as *const u32, n * DENSE_BELOW) };
            let cnt_s = unsafe { std::slice::from_raw_parts(counts.ptr() as *const u32, n) };
            let mut sel_diff = 0usize;
            for i in 0..n {
                let (a, b) = (&sel_s[i * DENSE_BELOW..i * DENSE_BELOW + cnt_s[i] as usize], &sel_w[i * DENSE_BELOW..i * DENSE_BELOW + cnt_w[i] as usize]);
                let bs: std::collections::HashSet<u32> = b.iter().copied().collect();
                sel_diff += a.iter().filter(|p| !bs.contains(p)).count();
            }
            let t_attend = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(q.va).ptr(kvb.va).ptr(lat.va).ptr(sel.va).ptr(counts.va).ptr(heads.va);
                k.launch(&k_attend, attend_grid * n as u32, g::WG, attend_lds, &a);
            });
            // the merge unrolls over the constant decode chunk count, so a forced chunk only shapes the one-chunk form
            let (chunk, chunks) = if chunk_override > 0 && DENSE_BELOW.div_ceil(chunk_override) == 1 { (chunk_override, 1) } else { (aw_chunk(n), aw_chunks(n)) };
            // the three launches timed apart (each after the one before it completed)
            let t_qabs = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(q.va).ptr(kvb.va).ptr(qabs.va);
                if n > glm_5_3_flash::SPEC_TOKENS {
                    a.u32(n as u32);
                    k.launch(&k_qabs_w, (MLA_HEADS as u32) * g::wmma_grid(KV_LORA_RANK, n), g::WG, g::WMMA_LDS, &a);
                } else {
                    k.launch(&k_qabs, (n * MLA_HEADS) as u32, 256, 0, &a);
                }
            });
            let t_aw = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(qabs.va).ptr(lat.va).ptr(sel.va).ptr(counts.va).ptr(olat.va).ptr(part.va).u32(chunks as u32).u32(chunk as u32);
                k.launch(&k_attend_w, (n * AW_HEAD_TILES * chunks) as u32, AW_WG, AW_LDS as u32, &a);
            });
            let t_merge = if chunks > 1 {
                time(&|| {
                    let mut a = kfd::Args::new();
                    a.ptr(part.va).ptr(olat.va);
                    k.launch(&k_merge, (n * AW_HEAD_TILES) as u32, AW_WG, AW_MERGE_LDS as u32, &a);
                })
            } else {
                0.0
            };
            let t_uv = time(&|| {
                let mut a = kfd::Args::new();
                a.ptr(olat.va).ptr(kvb.va).ptr(heads_w.va);
                if n > glm_5_3_flash::SPEC_TOKENS {
                    a.u32(n as u32);
                    k.launch(&k_uv, (MLA_HEADS as u32) * g::wmma_grid(V_HEAD_DIM, n), g::WG, g::WMMA_LDS, &a);
                } else {
                    k.launch(&k_uv_s, (n * MLA_HEADS) as u32, 256, (KV_LORA_RANK * 4) as u32, &a);
                }
            });
            let t_w = t_qabs + t_aw + t_merge + t_uv;
            // the two attends' outputs: the largest difference against the largest magnitude
            let (mut dmax, mut vmax, mut nans) = (0f32, 0f32, 0usize);
            {
                let a = unsafe { std::slice::from_raw_parts(heads.ptr() as *const f32, n * MLA_HEADS * V_HEAD_DIM) };
                let b = unsafe { std::slice::from_raw_parts(heads_w.ptr() as *const f32, n * MLA_HEADS * V_HEAD_DIM) };
                for (x, y) in a.iter().zip(b) {
                    if !y.is_finite() {
                        nans += 1;
                    } else {
                        dmax = dmax.max((x - y).abs());
                    }
                    vmax = vmax.max(x.abs());
                }
                if nans > 0 {
                    // where the non-finite values sit: by head and by channel eighth, token 0
                    let mut by_head = [0usize; MLA_HEADS];
                    let mut by_ch = [0usize; 8];
                    for h in 0..MLA_HEADS {
                        for r in 0..V_HEAD_DIM {
                            if !b[h * V_HEAD_DIM + r].is_finite() {
                                by_head[h] += 1;
                                by_ch[r / (V_HEAD_DIM / 8)] += 1;
                            }
                        }
                    }
                    println!("    non-finite by head {:?}", by_head);
                    println!("    non-finite by channel eighth {:?}", by_ch);
                }
                if nans > 0 || std::env::args().any(|a| a == "--show") {
                    println!("    scalar {:?}", &a[..4]);
                    println!("    wmma   {:?}", &b[..4]);
                    if chunks > 1 {
                        let pp = unsafe { std::slice::from_raw_parts(part.ptr() as *const f32, AW_PART) };
                        println!("    part0 m {:?} l {:?} o {:?}", &pp[..4], &pp[16..20], &pp[32..36]);
                    }
                }
            }
            let mut c = vec![0u32; n];
            unsafe { std::ptr::copy_nonoverlapping(counts.ptr() as *const u32, c.as_mut_ptr(), n) };
            println!("    ix_score_w {t_score_w:.1} µs ({:.2} a token); selection vs scalar: {sel_diff} positions differ of {}", t_score_w / n as f64, cnt_s.iter().map(|&c| c as usize).sum::<usize>());
            println!("{pos:>9} {n:>6} {t_score:>12.1} ({:>7.2}) {t_select:>12.1} ({:>7.2}) {t_attend:>12.1} ({:>7.2}) {t_w:>12.1} ({:>7.2}) ×{chunks:<2} = qabs {t_qabs:.0} + attend {t_aw:.0} + merge {t_merge:.0} + uv {t_uv:.0}  selected {}; |Δ|max {dmax:.2e} of {vmax:.2e}, non-finite {nans}", t_score / n as f64, t_select / n as f64, t_attend / n as f64, t_w / n as f64, c[n - 1]);
        }
    }
}
