//! bench-exp4 — the exp4 rows kernels (029) against the bf16 ones on the bundle's own KDA matrices, as the node launches
//! them: a part's `sq` (f_a, g_a, b, then its q, k, v rows: `[SMALL_ROWS + 3·PART_QKV][H]`, `rows_h` / `rows_h_e4`),
//! its half of the output projection (`[H][PART_QKV]`, `rows_part` / `rows_part_e4`) and its f_b | g_b rows
//! (`[2·PART_QKV][KDA_DIM]`, `rows_d2` / `rows_d2_e4`). Every (matrix, width) runs both kernels over the same activations
//! and compares the outputs bit for bit; then each is timed in chains that rotate over `--layers` layers' copies, so the
//! weights stream from VRAM and not the 80 MB Infinity Cache, as in serving (the node's grid and LDS for each launch).
//!
//! Then the prefill path: each exp4 matrix unpacked to bf16 scratch (`exp4_unpack`, checked equal to the bundle's bf16
//! byte for byte), the bf16 matrix kernels (`rows_h_w` over `sq`, `rows_part_w` over `o`) on the scratch at the prefill
//! widths `--wn`, against the same kernels on the bf16 matrix: the outputs' bits, and the unpack's time beside the
//! launch's (chains of 8: a launch is milliseconds).
//!
//!   bench-exp4 --elf <kernels.gfx1100.elf> --bundle <bundle dir> [--layers 3] [--n 1,2,3,4] [--wn 32,128,512] [--iters 20]
#![allow(clippy::missing_safety_doc)]
#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::weights::{Attn, Weights};
use glm_5_3_flash::{H, KDA_DIM, KDA_HEADS, KDA_QKV};
use harness::arg;

const SMALL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS;
const M_SMALL: usize = SMALL_ROWS + 3 * g::PART_QKV;

fn list(name: &str, default: &[usize]) -> Vec<usize> {
    arg(name).map_or(default.to_vec(), |v| v.to_str().unwrap().split(',').map(|s| s.trim().parse().expect(name)).collect())
}
fn raw16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
}
fn raw32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
struct Mat {
    name: &'static str,
    m: usize,
    k: usize,
    bf: Vec<kfd::Mem>,
    e4: Vec<kfd::Mem>,
    bytes_bf: usize,
    bytes_e4: usize,
}

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let dir = arg("--bundle").expect("--bundle").join("glm-5.3-flash");
    let layers: usize = arg("--layers").map_or(3, |v| v.to_str().unwrap().parse().unwrap());
    let ns = list("--n", &[1, 2, 3, 4]);
    let wns = list("--wn", &[32, 128, 512]);
    let iters: usize = arg("--iters").map_or(20, |v| v.to_str().unwrap().parse().unwrap());
    let f = std::fs::File::open(dir.join(glm_5_3_flash::layout::DENSE_FILE)).unwrap();
    let dense = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let w = Weights::from_dense(&dense);
    let k = kfd::Kfd::open(&elf, false, 0);
    let kern = |n: &str| k.kernel(n);
    let (rows_h, rows_h1, rows_h_e4, rows_h1_e4) = (kern("rows_h"), kern("rows_h1"), kern("rows_h_e4"), kern("rows_h1_e4"));
    let (rows_part, rows_part1, rows_part_e4, rows_part1_e4) = (kern("rows_part"), kern("rows_part1"), kern("rows_part_e4"), kern("rows_part1_e4"));
    let (rows_d2, rows_d2_e4) = (kern("rows_d2"), kern("rows_d2_e4"));
    let (rows_h_w, rows_part_w, unpack) = (kern("rows_h_w"), kern("rows_part_w"), kern("exp4_unpack"));
    // the first `layers` KDA layers, part 0 of each: the node's slicing
    let kda: Vec<usize> = w.layers.iter().enumerate().filter(|(_, l)| matches!(l.attn, Attn::Kda(_))).map(|(i, _)| i).take(layers).collect();
    let mut mats = vec![
        Mat { name: "sq", m: M_SMALL, k: H, bf: vec![], e4: vec![], bytes_bf: 0, bytes_e4: 0 },
        Mat { name: "o", m: H, k: g::PART_QKV, bf: vec![], e4: vec![], bytes_bf: 0, bytes_e4: 0 },
        Mat { name: "fg", m: 2 * g::PART_QKV, k: KDA_DIM, bf: vec![], e4: vec![], bytes_bf: 0, bytes_e4: 0 },
    ];
    let mut slots = 0usize;
    for &l in &kda {
        let Attn::Kda(kw) = &w.layers[l].attn else { unreachable!() };
        let rows = |m: &[u16], width: usize| -> Vec<u16> { m[..g::PART_QKV * width].to_vec() };
        let mut sq = Vec::with_capacity(M_SMALL * H);
        for part in [kw.f_a.to_vec(), kw.g_a.to_vec(), kw.b.to_vec(), rows(kw.q, H), rows(kw.k, H), rows(kw.v, H)] {
            sq.extend_from_slice(&part);
        }
        let mut o = Vec::with_capacity(H * g::PART_QKV);
        for r in 0..H {
            o.extend_from_slice(&kw.o[r * KDA_QKV..r * KDA_QKV + g::PART_QKV]);
        }
        let mut fg = rows(kw.f_b, KDA_DIM);
        fg.extend_from_slice(&rows(kw.g_b, KDA_DIM));
        for (mat, data) in mats.iter_mut().zip([sq, o, fg]) {
            assert_eq!(data.len(), mat.m * mat.k, "{}", mat.name);
            let e = common::exp4::encode(&data, mat.m, mat.k);
            assert_eq!(common::exp4::decode(&e, mat.m, mat.k), data, "{} layer {l}: round trip", mat.name);
            slots += e.slots();
            let pb = e.planes();
            let (mb, me) = (k.vram_visible(data.len() * 2), k.vram_visible(pb.len()));
            k.upload(&mb, 0, raw16(&data));
            k.upload(&me, 0, &pb);
            mat.bytes_bf = data.len() * 2;
            mat.bytes_e4 = pb.len();
            mat.bf.push(mb);
            mat.e4.push(me);
        }
    }
    println!("bench-exp4: {} KDA layers (part 0), {slots} raw groups in all; µs a launch, chains rotating over the layers", kda.len());
    let Attn::Kda(_) = &w.layers[kda[0]].attn else { unreachable!() };
    let ln: Vec<u16> = w.layers[kda[0]].input_ln.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
    let mut rng = common::testing::Rng::new(5);
    let n_max = (*ns.iter().max().unwrap()).max(*wns.iter().max().unwrap());
    let x: Vec<f32> = (0..n_max * H.max(g::PART_QKV)).map(|_| rng.unit()).collect();
    let (bx, bln) = (k.vram_visible(x.len() * 4), k.vram_visible(ln.len() * 2));
    k.upload(&bx, 0, raw32(&x));
    k.upload(&bln, 0, raw16(&ln));
    let (oa, ob) = (k.vram_visible(n_max * M_SMALL * 4), k.vram_visible(n_max * M_SMALL * 4));
    // the matrix path's activations: n_max tokens of H (sq) or PART_QKV (o) — x is long enough for both
    println!("{:>4} {:>3} {:>9} {:>9} {:>8} {:>8} {:>8}  {}", "mat", "n", "bf16 µs", "exp4 µs", "bf16 GB/s", "exp4 GB/s", "speedup", "bits");
    for mat in &mats {
        for &n in &ns {
            // the node's choice: the narrow twin below a workgroup a CU past one token (`rows_at`)
            let narrow = n > 1 && mat.m.div_ceil(common::gemv::rows_per_wg(n, common::gemv::ROWS)) < k.cu;
            let r = if narrow && mat.name != "fg" { 1 } else { common::gemv::ROWS };
            let (kb, ke) = match (mat.name, narrow) {
                ("sq", false) => (&rows_h, &rows_h_e4),
                ("sq", true) => (&rows_h1, &rows_h1_e4),
                ("o", false) => (&rows_part, &rows_part_e4),
                ("o", true) => (&rows_part1, &rows_part1_e4),
                _ => (&rows_d2, &rows_d2_e4),
            };
            let lds = match mat.name { "sq" => g::normed_rows_lds(H, n), "o" => g::rows_lds(g::PART_QKV, n), _ => g::rows_lds(KDA_DIM, n) } as u32;
            let grid = g::rows_grid(mat.m, n, r);
            let launch = |kernel: &kfd::Kernel, wv: usize, out: usize| {
                let mut a = kfd::Args::new();
                a.ptr(bx.va);
                if mat.name == "sq" {
                    a.ptr(bln.va);
                }
                a.ptr(wv).ptr(out).u32(mat.m as u32).u32(n as u32);
                if mat.name == "fg" {
                    // rows_d2: the split (f_b's rows read the first KDA_DIM of a token's input, g_b's the next) and the token stride
                    a.u32(g::PART_QKV as u32).u32(KDA_DIM as u32 * 2);
                }
                k.launch(kernel, grid, g::WG, lds, &a);
            };
            // the bits: every layer's copy, both kernels
            let mut differ = 0usize;
            for li in 0..mat.bf.len() {
                launch(kb, mat.bf[li].va, oa.va);
                launch(ke, mat.e4[li].va, ob.va);
                k.sync();
                let mut va = vec![0u32; n * mat.m];
                let mut vb = vec![0u32; n * mat.m];
                kfd::read(&oa, 0, &mut va);
                kfd::read(&ob, 0, &mut vb);
                differ += va.iter().zip(&vb).filter(|(a, b)| a != b).count();
            }
            // the time: chains of 43 launches over the layers' copies in turn, one sync a chain
            let time = |kernel: &kfd::Kernel, ws: &[kfd::Mem]| -> f64 {
                let chain = 43;
                let mut v: Vec<f64> = (0..iters)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        for c in 0..chain {
                            launch(kernel, ws[c % ws.len()].va, oa.va);
                        }
                        k.sync();
                        t.elapsed().as_secs_f64() * 1e6 / chain as f64
                    })
                    .collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[iters / 2]
            };
            let (tb, te) = (time(kb, &mat.bf), time(ke, &mat.e4));
            println!("{:>4} {n:>3} {tb:>9.1} {te:>9.1} {:>9.0} {:>9.0} {:>8.2}  {}", mat.name, mat.bytes_bf as f64 / tb / 1e3, mat.bytes_e4 as f64 / te / 1e3, tb / te, if differ == 0 { "equal".to_string() } else { format!("{differ} of {} words differ", n * mat.m * mat.bf.len()) });
        }
    }
    // the prefill path: unpack, then the bf16 matrix kernel on the scratch
    println!("prefill path (unpack once, then the bf16 matrix kernels):");
    println!("{:>4} {:>4} {:>9} {:>11} {:>10} {:>8}  {}", "mat", "n", "bf16 µs", "unpack µs", "total µs", "speedup", "bits");
    let scratch = k.vram_visible(M_SMALL * H * 2);
    for mat in mats.iter().filter(|m| m.name != "fg") {
        let launch_unpack = |li: usize| {
            let mut a = kfd::Args::new();
            a.ptr(mat.e4[li].va).ptr(scratch.va).u32(mat.m as u32).u32(mat.k as u32);
            k.launch(&unpack, (mat.m * mat.k / 8).div_ceil(256) as u32, 256, 0, &a);
        };
        // the unpack's bytes against the original
        let mut unpacked_ok = true;
        for li in 0..mat.bf.len() {
            launch_unpack(li);
            k.sync();
            let (mut got, mut want) = (vec![0u16; mat.m * mat.k], vec![0u16; mat.m * mat.k]);
            kfd::read(&scratch, 0, &mut got);
            kfd::read(&mat.bf[li], 0, &mut want);
            unpacked_ok &= got == want;
        }
        let tu = {
            let mut v: Vec<f64> = (0..iters).map(|_| { let t = std::time::Instant::now(); for c in 0..8 { launch_unpack(c % mat.bf.len()); } k.sync(); t.elapsed().as_secs_f64() * 1e6 / 8.0 }).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[iters / 2]
        };
        for &n in &wns {
            let kb = if mat.name == "sq" { &rows_h_w } else { &rows_part_w };
            let grid = g::wmma_grid(mat.m, n);
            let launch = |wv: usize, out: usize| {
                let mut a = kfd::Args::new();
                a.ptr(bx.va);
                if mat.name == "sq" {
                    a.ptr(bln.va);
                }
                a.ptr(wv).ptr(out).u32(mat.m as u32).u32(n as u32);
                k.launch(kb, grid, g::WG, g::WMMA_LDS, &a);
            };
            let mut differ = 0usize;
            for li in 0..mat.bf.len() {
                launch(mat.bf[li].va, oa.va);
                launch_unpack(li);
                launch(scratch.va, ob.va);
                k.sync();
                let mut va = vec![0u32; n * mat.m];
                let mut vb = vec![0u32; n * mat.m];
                kfd::read(&oa, 0, &mut va);
                kfd::read(&ob, 0, &mut vb);
                differ += va.iter().zip(&vb).filter(|(a, b)| a != b).count();
            }
            let tb = {
                let mut v: Vec<f64> = (0..iters).map(|_| { let t = std::time::Instant::now(); for c in 0..8 { launch(mat.bf[c % mat.bf.len()].va, oa.va); } k.sync(); t.elapsed().as_secs_f64() * 1e6 / 8.0 }).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[iters / 2]
            };
            let te = {
                let mut v: Vec<f64> = (0..iters).map(|_| { let t = std::time::Instant::now(); for c in 0..8 { launch_unpack(c % mat.bf.len()); launch(scratch.va, oa.va); } k.sync(); t.elapsed().as_secs_f64() * 1e6 / 8.0 }).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[iters / 2]
            };
            println!("{:>4} {n:>4} {tb:>9.1} {tu:>11.1} {te:>10.1} {:>8.2}  unpacked {}, outputs {}", mat.name, tb / te, if unpacked_ok { "equal" } else { "DIFFER" }, if differ == 0 { "equal".to_string() } else { format!("{differ} words differ") });
        }
    }
    let (b, e): (usize, usize) = mats.iter().map(|m| (m.bytes_bf, m.bytes_e4)).fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
    println!("a part's three matrices: {:.1} MB bf16, {:.1} MB exp4 ({:.1} %)", b as f64 / 1e6, e as f64 / 1e6, 100.0 * e as f64 / b as f64);
}
