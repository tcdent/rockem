//! bench-hc — the hyper-connection kernels timed on the card, alone: hc_pre with and without
//! the router tail, hc_post, and rows_h over the q/k/v shape for reference, each launched
//! N times with a sync, the median µs a launch. Runs beside a serving node (its own queue).
//!
//!   bench-hc --elf <kernels.gfx1100.elf> [--n 200]

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use glm_5_3_flash::hc::gfx11_consts::{CHANNEL_GRID, PRE_COUNTS, PRE_GRID, PRE_LDS, PRE_PARTS, MIX};
use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::{E, H, HC, HC_COEFS};
use harness::arg;

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let n: usize = arg("--n").map_or(200, |v| v.to_str().unwrap().parse().unwrap());
    let k = kfd::Kfd::open(&elf, false, 0);
    let (pre, post, rows_h) = (k.kernel("hc_pre"), k.kernel("hc_post"), k.kernel("rows_h"));
    let f32s = |m: usize, f: &dyn Fn(usize) -> f32| -> Vec<u8> { (0..m).flat_map(|i| f(i).to_le_bytes()).collect() };
    let bf16s = |m: usize, f: &dyn Fn(usize) -> f32| -> Vec<u8> { (0..m).map(|i| (f(i).to_bits() >> 16) as u16).flat_map(|v| v.to_le_bytes()).collect() };
    let up = |bytes: &[u8]| -> kfd::Mem {
        let m = k.vram_visible(bytes.len());
        k.upload(&m, 0, bytes);
        m
    };
    let streams = up(&f32s(HC * H, &|i| ((i * 7919) % 1000) as f32 * 1e-3 - 0.5));
    let w = up(&bf16s(HC_COEFS * HC * H, &|i| ((i * 104729) % 2000) as f32 * 1e-4 - 0.1));
    let part = k.vram(PRE_PARTS * 4);
    let base = up(&f32s(HC_COEFS, &|_| 0.1));
    let scale = up(&f32s(3, &|_| 1.0));
    let mix = k.vram(MIX * 4);
    let x = k.vram(H * 4);
    let count = up(&vec![0u8; PRE_COUNTS * 4]);
    let ln = up(&bf16s(H, &|_| 1.0));
    let router = up(&bf16s(E * H, &|i| ((i * 31) % 1000) as f32 * 1e-4));
    let scores = k.host_coherent(E * 4);
    let word = k.host_coherent(4096);
    let go = up(&[0u8; 64]);
    let ya = k.host_coherent(H * 4);
    let median = |name: &str, launch: &dyn Fn(u32)| {
        let mut us: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let t = std::time::Instant::now();
            launch(i as u32 + 1);
            k.sync();
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name:<32} median {:6.1} µs  min {:6.1}  (launch + sync, {n} runs)", us[n / 2], us[0]);
    };
    median("sync alone", &|_| {});
    median("hc_pre, no router", &|v| {
        let mut a = kfd::Args::new();
        a.ptr(streams.va).ptr(w.va).ptr(part.va).ptr(base.va).ptr(scale.va).ptr(mix.va).ptr(x.va).ptr(0).ptr(count.va).u32(v).ptr(0).ptr(0).ptr(0).ptr(0);
        k.launch(&pre, PRE_GRID, g::WG, PRE_LDS, &a);
    });
    for (name, flag) in [("hc_pre, no mix (publish at once)", 1u32 << 31), ("hc_pre, no collapse", 1 << 30), ("hc_pre, no sinkhorn", 1 << 29), ("hc_pre, no dot", 1 << 28), ("hc_pre, no dot, no mix, no collapse", (1 << 31) | (1 << 30) | (1 << 28))] {
        median(name, &|v| {
            let mut a = kfd::Args::new();
            a.ptr(streams.va).ptr(w.va).ptr(part.va).ptr(base.va).ptr(scale.va).ptr(mix.va).ptr(x.va).ptr(0).ptr(count.va).u32(v | flag).ptr(0).ptr(0).ptr(0).ptr(0);
            k.launch(&pre, PRE_GRID, g::WG, PRE_LDS, &a);
        });
    }
    median("hc_pre, router + word", &|v| {
        let mut a = kfd::Args::new();
        a.ptr(streams.va).ptr(w.va).ptr(part.va).ptr(base.va).ptr(scale.va).ptr(mix.va).ptr(x.va).ptr(0).ptr(count.va).u32(v).ptr(ln.va).ptr(router.va).ptr(scores.va).ptr(word.va);
        k.launch(&pre, PRE_GRID, g::WG, PRE_LDS, &a);
    });
    median("hc_post, no handoff", &|v| {
        let mut a = kfd::Args::new();
        a.ptr(mix.va).ptr(ya.va).ptr(0).ptr(streams.va).ptr(0).ptr(0).u32(0).ptr(0).ptr(0).ptr(0).u32(v).ptr(go.va);
        k.launch(&post, CHANNEL_GRID, g::WG, 0, &a);
    });
    // the rows skeleton's rate against the row count and width: a `[m][K]` bf16 matrix over one vector
    let rows_part = k.kernel("rows_part");
    let big = k.vram(8192 * H * 2);
    let out = k.vram(8192 * 4);
    let xp = k.vram(g::PART_QKV * 4);
    for m in [512usize, 1024, 2048, 4096, 8192] {
        let mb = (m * H * 2) as f64 / 1e6;
        median(&format!("rows_h   m={m:<5} K={H} ({mb:.0} MB)"), &|_| {
            let mut a = kfd::Args::new();
            a.ptr(x.va).ptr(ln.va).ptr(big.va).ptr(out.va).u32(m as u32).u32(1);
            k.launch(&rows_h, g::rows_grid(m), g::WG, g::normed_rows_lds(H, 1) as u32, &a);
        });
    }
    for m in [1024usize, 2048, 4096, 8192] {
        let mb = (m * g::PART_QKV * 2) as f64 / 1e6;
        median(&format!("rows_part m={m:<5} K={} ({mb:.0} MB)", g::PART_QKV), &|_| {
            let mut a = kfd::Args::new();
            a.ptr(xp.va).ptr(big.va).ptr(out.va).u32(m as u32).u32(1);
            k.launch(&rows_part, g::rows_grid(m), g::WG, g::rows_lds(g::PART_QKV, 1) as u32, &a);
        });
    }
}
