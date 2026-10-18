//! bench-xpt — one routed expert of GLM-5.3-Flash on a card (`expert::gfx11`: `xpt_gu`, `xpt_down`)
//! against the CPU realization on the same bytes and activation: the bits must match, then the
//! card's time a job. Runs on the box (the bundle under /bundle).
//!
//!   bench-xpt --bundle <dir> --elf <kernels.gfx1100.elf> [--layer 10] [--expert 0] [--jobs 8] [--iters 20]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use common::q4::{self, Act, QAct, GROUP};
use common::swiglu::swiglu_clamped;
use glm_5_3_flash::expert::gfx11_consts::{DOWN_WGS, SLOT_BYTES, UP_WGS, XPT_LDS};
use glm_5_3_flash::expert::{Stream, Weights, DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS};
use glm_5_3_flash::layout::{DOWN_ITEMS, GU_ITEMS};
use glm_5_3_flash::{D_FF, H, SWIGLU_LIMIT};
use harness::{arg, arg_required};

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}

/// The CPU realization of one expert on `x` (`run_batch`'s arithmetic at one token: `q4::gemv` a part).
fn cpu(w: &Weights, x: &[f32]) -> Vec<f32> {
    const HG: usize = H / GROUP;
    const DG: usize = D_FF / GROUP;
    let (mut xq, mut xs, mut xg) = (vec![0i8; H], vec![0f32; HG], vec![0i32; HG]);
    QAct { q: &mut xq, scale: &mut xs, gsum: &mut xg }.quantize(x);
    let act = Act { q: &xq, scale: &xs, gsum: &xg };
    let mut gu = vec![0f32; 2 * D_FF];
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4::gemv::<H>(GU_ROWS, gw, gs, &act, &mut gu[c * GU_ROWS..(c + 1) * GU_ROWS]);
        q4::gemv::<H>(GU_ROWS, uw, us, &act, &mut gu[D_FF + c * GU_ROWS..D_FF + (c + 1) * GU_ROWS]);
    }
    let mut h = vec![0f32; D_FF];
    swiglu_clamped::<D_FF>(&gu, SWIGLU_LIMIT, &mut h);
    let (mut hq, mut hs, mut hg) = (vec![0i8; D_FF], vec![0f32; DG], vec![0i32; DG]);
    QAct { q: &mut hq, scale: &mut hs, gsum: &mut hg }.quantize(&h);
    let act2 = Act { q: &hq, scale: &hs, gsum: &hg };
    let mut out = vec![0f32; H];
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4::gemv::<D_FF>(DOWN_ROWS, dw, ds, &act2, &mut out[c * DOWN_ROWS..(c + 1) * DOWN_ROWS]);
    }
    out
}

fn main() {
    let bdir = arg_required("--bundle");
    let elf = std::fs::read(arg_required("--elf")).unwrap();
    let (l, e0, jobs, iters) = (num("--layer", 10), num("--expert", 0), num("--jobs", 8), num("--iters", 20));
    // --slots N: the slot buffer sized for N experts (the node's residency: one host-visible allocation of ~10 GB)
    let nslots = num("--slots", jobs).max(jobs);
    let f = std::fs::File::open(bdir.join(glm_5_3_flash::layout::EXPERTS_FILE)).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let stream = Stream::file(&mm[..]);
    let k = kfd::Kfd::open(&elf, false, 0);
    let (kgu, kdown) = (k.kernel("xpt_gu"), k.kernel("xpt_down"));
    // `jobs` experts e0.. in slots 0.., each a job on its own token (the same activation shifted)
    let slots = k.vram_visible(nslots * SLOT_BYTES);
    println!("slots: {nslots} ({:.1} GB host-visible), {:.1} GB allocated on the card", (nslots * SLOT_BYTES) as f64 / 1e9, k.allocated.get() as f64 / 1e9);
    for j in 0..jobs {
        let w = Weights::at(&stream, l, e0 + j);
        let mut bytes = Vec::with_capacity(SLOT_BYTES);
        for c in 0..GU_ITEMS {
            bytes.extend_from_slice(w.gu_bytes(c));
        }
        for c in 0..DOWN_ITEMS {
            bytes.extend_from_slice(w.down_bytes(c));
        }
        assert_eq!(bytes.len(), SLOT_BYTES);
        k.upload(&slots, j * SLOT_BYTES, &bytes);
    }
    const HG: usize = H / GROUP;
    let xs_all: Vec<Vec<f32>> = (0..jobs).map(|j| (0..H).map(|i| (((i + 13 * j) * 7919) % 2000) as f32 * 1e-3 - 1.0).collect()).collect();
    let (mut xq, mut xs, mut xg) = (vec![0i8; jobs * H], vec![0f32; jobs * HG], vec![0i32; jobs * HG]);
    for j in 0..jobs {
        QAct { q: &mut xq[j * H..(j + 1) * H], scale: &mut xs[j * HG..(j + 1) * HG], gsum: &mut xg[j * HG..(j + 1) * HG] }.quantize(&xs_all[j]);
    }
    let raw = |b: &[u8]| -> kfd::Mem {
        let m = k.vram_visible(b.len());
        k.upload(&m, 0, b);
        m
    };
    let mq = raw(unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) });
    let ms = raw(unsafe { std::slice::from_raw_parts(xs.as_ptr() as *const u8, xs.len() * 4) });
    let mg = raw(unsafe { std::slice::from_raw_parts(xg.as_ptr() as *const u8, xg.len() * 4) });
    let jl: Vec<u32> = (0..jobs as u32).flat_map(|j| [j, j]).collect();
    let mj = raw(unsafe { std::slice::from_raw_parts(jl.as_ptr() as *const u8, jl.len() * 4) });
    let gu = k.vram(jobs * 2 * D_FF * 4);
    let out = k.vram_visible(jobs * H * 4);
    let run = |n: usize| {
        let mut a = kfd::Args::new();
        a.ptr(slots.va).ptr(mj.va).ptr(mq.va).ptr(ms.va).ptr(mg.va).ptr(gu.va);
        k.launch(&kgu, (n * UP_WGS) as u32, 256, XPT_LDS as u32, &a);
        let mut b = kfd::Args::new();
        b.ptr(slots.va).ptr(mj.va).ptr(gu.va).ptr(out.va);
        k.launch(&kdown, (n * DOWN_WGS) as u32, 256, XPT_LDS as u32, &b);
        k.sync();
    };
    run(jobs);
    let card = unsafe { std::slice::from_raw_parts(out.ptr() as *const f32, jobs * H) };
    let mut same = 0usize;
    let mut worst = 0f32;
    for j in 0..jobs {
        let ref_ = cpu(&Weights::at(&stream, l, e0 + j), &xs_all[j]);
        for i in 0..H {
            if ref_[i].to_bits() == card[j * H + i].to_bits() {
                same += 1;
            }
            worst = worst.max((ref_[i] - card[j * H + i]).abs());
        }
    }
    println!("xpt vs cpu: {same} of {} outputs bit-identical over {jobs} experts of layer {l}, max |Δ| {worst:.3e}", jobs * H);
    for n in [1usize, 2, 4, jobs] {
        let n = n.min(jobs);
        let mut us: Vec<f64> = (0..iters).map(|_| {
            let t = std::time::Instant::now();
            run(n);
            t.elapsed().as_secs_f64() * 1e6
        }).collect();
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{n} jobs: {:7.1} µs median (gu + down + sync), {:6.1} µs a job, {:6.1} GB/s of expert bytes", us[iters / 2], us[iters / 2] / n as f64, (n * SLOT_BYTES) as f64 / us[iters / 2] / 1e3);
    }
}
