//! dump-expert — test vectors for the GPU expert kernels, as raw little-endian
//! files the launcher compares byte for byte against the scalar realization.
//!
//!   dump-expert --bundle <dir> --out <dir> [--layer 1] [--expert 0]
//!   dump-expert --bundle <dir> --out <dir> --goldens <dir> --layer L --step s --token t
//!
//! The second form takes a real token: the layer's post-norm hidden state and
//! its top-k routing from the oracle goldens, runs every hit expert through
//! the scalar realization, and also writes the oracle's own routed outputs so
//! the launcher can show what the recipe costs on real activations.

use common::q4::{gemv_scalar, QAct, GROUP};
use common::swiglu::swiglu;
use deepseek_v2_lite::expert::Weights;
use deepseek_v2_lite::layout::{record, RECORD};
use deepseek_v2_lite::{D_FF, H, N_ROUTED_EXPERTS, TOP_K};
use harness::{arg, arg_required, Golden};
use std::fs;
use std::path::Path;

fn num(name: &str, default: usize) -> usize {
    arg(name).map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(default)
}
fn write<T: Copy>(path: &Path, v: &[T]) {
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    fs::write(path, bytes).unwrap();
}

/// The scalar expert on one record; returns (gate_up, h, h quantized, down).
fn scalar(w: &Weights, xa: &QAct) -> (Vec<f32>, Vec<f32>, Vec<i8>, Vec<f32>, Vec<i32>, Vec<f32>) {
    let mut gu = vec![0f32; 2 * D_FF];
    gemv_scalar(H, 2 * D_FF, w.gate_up_w, w.gate_up_s, &xa.act(), &mut gu);
    let mut h = vec![0f32; D_FF];
    swiglu::<D_FF>(&gu, &mut h);
    let (mut hq, mut hs, mut hg) = (vec![0i8; D_FF], vec![0f32; D_FF / GROUP], vec![0i32; D_FF / GROUP]);
    let mut ha = QAct { q: &mut hq, scale: &mut hs, gsum: &mut hg };
    ha.quantize(&h);
    let mut down = vec![0f32; H];
    gemv_scalar(D_FF, H, w.down_w, w.down_s, &ha.act(), &mut down);
    // (`expert::forward_scalar` opened up so its intermediates can be written out)
    (gu, h, hq, hs, hg, down)
}

fn main() {
    let bdir = arg_required("--bundle");
    let out = arg_required("--out");
    let l = num("--layer", 1);
    fs::create_dir_all(&out).unwrap();
    let f = fs::File::open(bdir.join("experts.q4g128.bin")).unwrap();
    let mm = unsafe { memmap2::Mmap::map(&f).unwrap() };

    // input and the experts to run: synthetic, or a real token from the goldens
    let (x, experts, weights, golden_routed): (Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) = match arg("--goldens") {
        None => ((0..H).map(|i| ((i * 7919) % 2000) as f32 * 1e-3 - 1.0).collect(), vec![num("--expert", 0)], vec![1.0], vec![]),
        Some(g) => {
            let (s, t) = (num("--step", 0), num("--token", 0));
            let g = Golden::load(&g.join(format!("step{s}.safetensors")));
            let x = g.row(&format!("model.layers.{l}.post_attention_layernorm.out"), t).unwrap().to_vec();
            let idx: Vec<usize> = g.irow(&format!("model.layers.{l}.mlp.gate.topk_idx"), t).unwrap().iter().map(|&v| v as usize).collect();
            let wt = g.row(&format!("model.layers.{l}.mlp.gate.topk_weight"), t).unwrap().to_vec();
            let routed = g.row(&format!("model.layers.{l}.mlp.routed.out"), t).unwrap().to_vec();
            (x, idx, wt, routed)
        }
    };
    let n = experts.len();
    assert!(n == 1 || n == TOP_K);

    let (mut xq, mut xs, mut xg) = (vec![0i8; H], vec![0f32; H / GROUP], vec![0i32; H / GROUP]);
    let mut xa = QAct { q: &mut xq, scale: &mut xs, gsum: &mut xg };
    xa.quantize(&x);
    let (mut gu_all, mut h_all, mut hq_all, mut hs_all, mut hg_all, mut down_all) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for &e in &experts {
        let w = Weights::from_record(&mm[record(l, e)]);
        let (gu, h, hq, hs, hg, down) = scalar(&w, &xa);
        gu_all.extend(gu);
        h_all.extend(h);
        hq_all.extend(hq);
        hs_all.extend(hs);
        hg_all.extend(hg);
        down_all.extend(down);
    }
    write(&out.join("x.f32"), &x);
    write(&out.join("xq.i8"), xa.q);
    write(&out.join("xs.f32"), xa.scale);
    write(&out.join("xg.i32"), xa.gsum);
    write(&out.join("gate_up.f32"), &gu_all);
    write(&out.join("h.f32"), &h_all);
    write(&out.join("hq.i8"), &hq_all);
    write(&out.join("hs.f32"), &hs_all);
    write(&out.join("hg.i32"), &hg_all);
    write(&out.join("down.f32"), &down_all);
    write(&out.join("experts.u32"), &experts.iter().map(|&e| e as u32).collect::<Vec<_>>());
    write(&out.join("weights.f32"), &weights);
    write(&out.join("golden_routed.f32"), &golden_routed);
    let layer_start = record(l, 0).start;
    // every constant the launcher needs, emitted here and never retyped in Python
    let stage = |k: usize| k + 8 * (k / GROUP);
    fs::write(
        out.join("meta.txt"),
        format!(
            "layer {l}\nn {n}\nrecord_offset {layer_start}\nrecord_bytes {RECORD}\nlayer_records {N_ROUTED_EXPERTS}\nH {H}\nD_FF {D_FF}\nwg 256\nup_tiles {}\ndown_tiles {}\nup_lds {}\ndown_lds {}\nvram_ceiling_gbps 760\n",
            2 * D_FF / 4, H / 4, stage(H), stage(D_FF)
        ),
    )
    .unwrap();
    println!("{}: layer {l}, experts {experts:?}", out.display());
}
