//! One expert's forward and backward on one core, over its own bytes: the backward's
//! transposed products against the forward's dot products on the same `q4g128` record,
//! and the adapter's share. Cache-hot (one expert is ~12.6 MB), so what it measures is the
//! instruction cost of each op, not the DRAM stream — the number that says whether the
//! backward stays memory-bound where the forward is.
//!
//!   cargo run --release --example expert_rate [rank]

use common::q4::{self, GROUP};
use common::testing::Rng;
use glm_5_3_flash::expert::{pack_items, ScratchBuf, Stream, Weights, ITEMS};
use glm_5_3_flash::layout::{self, DOWN_ITEMS, GU_ITEMS};
use glm_5_3_flash::{D_FF, E, FIRST_K_DENSE_REPLACE, H};
use glm_5_3_flash_train::expert::{backward, forward, Adapter, Grad, Saved};
use std::time::Instant;

fn main() {
    let r: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(16);
    let mut rnd = Rng::new(0xBE4C_0001);
    let mk = |m: usize, k: usize, rnd: &mut Rng| {
        let w: Vec<i8> = (0..m * k).map(|_| rnd.i4()).collect();
        let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.01 + (rnd.next() % 100) as f32 * 1e-4).collect();
        q4::pack(m, k, &w, &s)
    };
    let (guw, gus) = mk(2 * D_FF, H, &mut rnd);
    let (dw, ds) = mk(H, D_FF, &mut rnd);
    let mut block = vec![0u32; layout::XBLOCK_BYTES / 4];
    let bytes: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
    for (off, it) in pack_items(FIRST_K_DENSE_REPLACE, 0, &guw, &gus, &dw, &ds) {
        bytes[off..off + it.len()].copy_from_slice(&it);
    }
    let mut addr = Vec::with_capacity(layout::MOE_LAYERS * E * ITEMS);
    for l in FIRST_K_DENSE_REPLACE..layout::LAYERS_IN_FILE {
        for e in 0..E {
            for c in 0..GU_ITEMS {
                addr.push(layout::gu_item(l, e, c).start as u64);
            }
            for c in 0..DOWN_ITEMS {
                addr.push(layout::down_item(l, e, c).start as u64);
            }
        }
    }
    let bytes: &[u8] = bytes;
    let st = Stream::placed(bytes, addr, vec![0; layout::MOE_LAYERS * E * ITEMS], Vec::new());
    let w = Weights::at(&st, FIRST_K_DENSE_REPLACE, 0);
    let expert_bytes = (GU_ITEMS * layout::GU_ITEM + DOWN_ITEMS * layout::DOWN_ITEM) as f64;

    let x: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
    let dout: Vec<f32> = (0..H).map(|_| rnd.unit()).collect();
    let mut buf = ScratchBuf::new();
    let mut out = vec![0f32; H];
    let mut dx = vec![0f32; H];

    let time = |name: &str, bytes: f64, mut f: &mut dyn FnMut()| {
        for _ in 0..3 {
            f();
        }
        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!("{name:<28} {us:8.1} µs   {:6.1} GB/s of expert bytes", bytes / us * 1e-3);
        us
    };

    let base = time("forward (inference)", expert_bytes, &mut || glm_5_3_flash::expert::forward(&w, &x, &mut buf.scratch(), &mut out));
    for &rank in &[0usize, r] {
        let ad = Adapter::zeros(rank);
        let mut saved = Saved::new(rank);
        let mut grads = Adapter::zeros(rank);
        let mut gr = Grad::new(rank);
        let f = time(&format!("forward, adapter r={rank}"), expert_bytes, &mut || forward(&w, &ad, &x, &mut buf.scratch(), &mut saved, &mut out));
        let b = time(&format!("backward, adapter r={rank}"), expert_bytes, &mut || {
            dx.iter_mut().for_each(|v| *v = 0.0);
            backward(&w, &ad, &saved, &dout, &mut gr, &mut grads, &mut dx)
        });
        println!("  r={rank}: forward {:.2}× the inference forward, backward {:.2}×, adapter params {}", f / base, b / base, ad.params());
    }
}
