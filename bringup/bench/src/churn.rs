//! bench-churn — a fresh process, a fresh queue on every gfx1100 card, one copy
//! launch and a sync on each, then exit: the first packet of a new queue. Run in a
//! loop to ask whether a queue ever fails to consume its first packet (the one hang
//! of 2026-10-15 19:38 read "packet 1 never completed (read index 0)": a queue that
//! had retired nothing). Prints one line per card; a hang trips the launcher's assert.
//!
//!   bench-churn --elf <kernels.gfx1100.elf> [--launches 1] [--clocks] [--upload-mb 0]
//!
//! `--clocks` forces every card's performance level high after the open, `--upload-mb`
//! streams that much through each card's BAR before its first launch — what the serving
//! boot does between opening the cards and its first packet.

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::arg;

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let launches: usize = arg("--launches").map_or(1, |v| v.to_str().unwrap().parse().unwrap());
    let upload_mb: usize = arg("--upload-mb").map_or(0, |v| v.to_str().unwrap().parse().unwrap());
    let t0 = std::time::Instant::now();
    let cards = kfd::Kfd::cards().len();
    let ks: Vec<_> = (0..cards).map(|i| kfd::Kfd::open(&elf, false, i)).collect();
    let topen = t0.elapsed();
    if harness::flag("--clocks") {
        kfd::Kfd::clocks_high();
    }
    for (i, k) in ks.iter().enumerate() {
        if upload_mb > 0 {
            let big = k.vram_visible(upload_mb << 20);
            let bytes = vec![0x5au8; 64 << 20];
            let mut off = 0;
            while off < big.size {
                let n = bytes.len().min(big.size - off);
                k.upload(&big, off, &bytes[..n]);
                off += n;
            }
        }
        let copy = k.kernel("copy_f32");
        let n = 4096usize;
        let src = k.vram_visible(n * 4);
        let dst = k.host(n * 4);
        let pattern: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + (i % 7) as f32).collect();
        k.upload(&src, 0, kfd::raw(&pattern));
        let t1 = std::time::Instant::now();
        for _ in 0..launches {
            let mut a = kfd::Args::new();
            a.ptr(src.va).ptr(dst.va).u32(n as u32);
            k.launch(&copy, n.div_ceil(256 * 16) as u32, 256, 0, &a);
            k.sync();
        }
        let mut back = vec![0f32; n];
        kfd::read(&dst, 0, &mut back);
        let ok = back == pattern;
        println!("card {i}: open {:.0} ms, {launches} launch{} + sync in {:.0} µs, copy {}", topen.as_secs_f64() * 1e3, if launches == 1 { "" } else { "es" }, t1.elapsed().as_secs_f64() * 1e6, if ok { "exact" } else { "WRONG" });
        assert!(ok, "card {i}: the copy came back wrong");
    }
}
