//! bench-wmma-layout — the wave32 bf16 WMMA fragment layout on gfx1100, solved empirically: A is fed
//! so that lane l's 16 values are 16·l + i, B so that lane l's are 1000·(16·l + i) — then each output
//! element identifies the A lane and B lane it multiplied. Prints, for each (lane, register) of the
//! product, which A lane (row) and B lane (column) it holds. The WMMA rows kernel is written from this.
//!
//!   bench-wmma-layout --elf <bringup_kernels.elf>

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;

use harness::arg;

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

fn main() {
    let elf = std::fs::read(arg("--elf").expect("--elf")).unwrap();
    let k = kfd::Kfd::open(&elf, false, 0);
    let kern = k.kernel("wmma_layout");
    let up = |v: &[u16]| -> kfd::Mem {
        let m = k.vram_visible(v.len() * 2);
        k.upload(&m, 0, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) });
        m
    };
    let out = k.vram_visible(32 * 8 * 4);
    // two probes: A[l][0] = l + 1 with B[l][0] = 1 names the A lane each output came from; the reverse
    // names the B lane (only k = 0 is nonzero, so a product is one lane's value)
    let probe = |a_lane: bool| -> Vec<i64> {
        let mut a = vec![0u16; 32 * 16];
        let mut b = vec![0u16; 32 * 16];
        for l in 0..32 {
            a[l * 16] = bf16(if a_lane { (l + 1) as f32 } else { 1.0 });
            b[l * 16] = bf16(if a_lane { 1.0 } else { (l + 1) as f32 });
        }
        let (ma, mb) = (up(&a), up(&b));
        let mut args = kfd::Args::new();
        args.ptr(ma.va).ptr(mb.va).ptr(out.va);
        k.launch(&kern, 1, 32, 0, &args);
        k.sync();
        let d = unsafe { std::slice::from_raw_parts(out.ptr() as *const f32, 32 * 8) };
        d.iter().map(|&v| v.round() as i64 - 1).collect()
    };
    let (ra, rb) = (probe(true), probe(false));
    println!("product lane l, register r  →  A lane (the row's source) / B lane (the column's source)");
    for l in 0..32 {
        let row: Vec<String> = (0..8).map(|r| format!("r{r}:A{:>2}B{:>2}", ra[l * 8 + r], rb[l * 8 + r])).collect();
        println!("lane {l:>2}: {}", row.join("  "));
    }
}
