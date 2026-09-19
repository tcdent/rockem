//! forward — run the transcribed DeepSeek-V2-Lite forward token by token and
//! diff every traced edge against the oracle goldens.
//!
//!   forward --snapshot <hf snapshot dir> --goldens <goldens/v2lite> --bundle <dir> [--dense] [--tol 1e-4]
//!           [--q4-layers all|none|1,5,9]   which MoE layers run the q4 recipe (bf16 from the checkpoint elsewhere)
//!           [--q4-experts 3:5,17,40]       finer: within --q4-layers, only these experts of layer 3 (others bf16)
//!           [--usage <file>]               per-layer expert usage counts over all positions (TSV, [layer][expert])
//!           [--q4-skip-pos 0,1]            positions that always run bf16 experts, whatever the recipe
//!           [--no-sink]                    compute position 0 under the recipe instead of seeding it from the bundle's sink.f32
//!                                          (what the sink is worth; the default is the node's path, from FIRST_POS)
//!           [--noise L:norm[:fixed]]       gain probe: perturb layer L's routed expert outputs by a random vector of that L2 norm
//!                                          (`fixed`: the same vector at every position — a bias instead of noise)
//!           [--quiet]                      only the summary line: teacher-forced KL, top-1 agreement, margins
//!           [--stats <file>]               per-layer activation ranges (rms, absmax, absmax/rms) averaged over positions
//!
//! The routed experts run from the bundle's packed q4g128 records, so the
//! table shows the recipe against the f32 oracle. With `--dense` the
//! non-expert weights come from the bundle's dense file instead of the
//! checkpoint — the node's load path, checked against the same goldens. `FORWARD_DUMP=n` prints
//! every edge of the first n tokens and stops.

use harness::{arg, arg_required, flag, Checkpoint, Collect, Golden, Report};
use deepseek_v2_lite::layout::dense as dl;
use deepseek_v2_lite::*;
use deepseek_v2_lite::{CpuAttention, CpuExperts, CpuFfn, CpuModel, Routed, State, Weights, TOP_K};

/// The routed experts from the checkpoint's bf16 tensors, through the plain
/// bf16 GEMV: the recipe-free realization, for measuring what a recipe costs.
struct CheckpointExperts {
    ck: &'static Checkpoint,
    gu: Vec<f32>,
    h: Vec<f32>,
}
impl Routed for CheckpointExperts {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        use deepseek_v2_lite::{D_FF, H};
        for k in 0..TOP_K {
            let p = format!("model.layers.{l}.mlp.experts.{}.", idx[k]);
            common::gemv::bf16::<D_FF, H>(self.ck.bf16(&format!("{p}gate_proj.weight")), x, &mut self.gu[..D_FF]);
            common::gemv::bf16::<D_FF, H>(self.ck.bf16(&format!("{p}up_proj.weight")), x, &mut self.gu[D_FF..]);
            common::swiglu::swiglu::<D_FF>(&self.gu, &mut self.h);
            common::gemv::bf16::<H, D_FF>(self.ck.bf16(&format!("{p}down_proj.weight")), &self.h, &mut out[k * H..(k + 1) * H]);
        }
    }
}

/// Per-(layer, expert) recipe: q4 records where `use_q4[l][e]`, bf16 elsewhere.
struct MixedExperts {
    q4: CpuExperts<'static>,
    bf16: CheckpointExperts,
    use_q4: Vec<Vec<bool>>, // [layer][expert]
    /// Gain probe: add a random perturbation of this L2 norm to every routed
    /// expert output of this layer (seeded per call, so it is reproducible and
    /// independent of which experts fired).
    noise: Option<(usize, f32, bool)>,
    calls: u64,
    /// Positions that always run bf16 regardless of the recipe: position-level precision (analysis.md §9).
    skip_pos: Vec<usize>,
    pos: usize,
}
impl Routed for MixedExperts {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        use deepseek_v2_lite::H;
        for k in 0..TOP_K {
            let one = [idx[k]; TOP_K];
            let mut tmp = vec![0f32; TOP_K * H];
            let q = self.use_q4[l][idx[k]] && !self.skip_pos.contains(&self.pos);
            if q { self.q4.run(l, &one, x, &mut tmp) } else { self.bf16.run(l, &one, x, &mut tmp) }
            out[k * H..(k + 1) * H].copy_from_slice(&tmp[..H]);
        }
        if let Some((nl, norm, fixed)) = self.noise {
            if nl == l {
                for k in 0..TOP_K {
                    self.calls += 1;
                    let seed = if fixed { 7 } else { self.calls };
                    let mut rng = common::testing::Rng::new(0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
                    // approximately gaussian: sum of 4 uniforms, then scaled to the requested L2 norm
                    let v: Vec<f32> = (0..H).map(|_| (0..4).map(|_| rng.unit()).sum::<f32>()).collect();
                    let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
                    for i in 0..H {
                        out[k * H + i] += v[i] * (norm / n);
                    }
                }
            }
        }
    }
}
use std::fs;

fn main() {
    let snapshot = arg_required("--snapshot");
    let gdir = arg_required("--goldens");
    let tol: f32 = arg("--tol").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(1e-4);
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let dump: usize = std::env::var("FORWARD_DUMP").ok().map(|v| v.parse().unwrap_or(1)).unwrap_or(0);
    // --dump-attn <dir> --dump-layer L --dump-pos P: one layer's attention inputs, cache and traced
    // outputs at one position, for the GPU attention launcher (bringup/gpu/attn.py)
    let dump_attn = arg("--dump-attn");
    let num = |k: &str| arg(k).map(|p| p.to_str().unwrap().parse::<usize>().unwrap());
    let (dump_layer, dump_pos) = (num("--dump-layer").unwrap_or(1), num("--dump-pos").unwrap_or(3));

    let t0 = std::time::Instant::now();
    let b = arg_required("--bundle");
    let mmap = |name: &str| -> &'static [u8] {
        let f = fs::File::open(b.join(name)).unwrap();
        let mm: &'static memmap2::Mmap = Box::leak(Box::new(unsafe { memmap2::Mmap::map(&f).unwrap() }));
        &mm[..]
    };
    let experts = mmap(deepseek_v2_lite::layout::EXPERTS_FILE);
    let ck: &'static Checkpoint = Box::leak(Box::new(Checkpoint::open(&snapshot)));
    let q4_layers = arg("--q4-layers").map(|p| p.to_str().unwrap().to_string()).unwrap_or("all".into());
    let layer_q4: Vec<bool> = (0..deepseek_v2_lite::NUM_HIDDEN_LAYERS)
        .map(|l| match q4_layers.as_str() {
            "all" => true,
            "none" => false,
            list => list.split(',').any(|t| t.parse::<usize>().ok() == Some(l)),
        })
        .collect();
    let mut use_q4: Vec<Vec<bool>> = layer_q4.iter().map(|&b| vec![b; deepseek_v2_lite::N_ROUTED_EXPERTS]).collect();
    if let Some(spec) = arg("--q4-experts") {
        // "L:e1,e2" — restrict layer L's q4 set to those experts
        let spec = spec.to_str().unwrap().to_string();
        let (l, es) = spec.split_once(':').unwrap();
        let l: usize = l.parse().unwrap();
        let set: Vec<usize> = es.split(',').map(|t| t.parse().unwrap()).collect();
        for e in 0..deepseek_v2_lite::N_ROUTED_EXPERTS {
            use_q4[l][e] = set.contains(&e);
        }
    }
    let mut usage = vec![vec![0u32; deepseek_v2_lite::N_ROUTED_EXPERTS]; deepseek_v2_lite::NUM_HIDDEN_LAYERS];
    let quiet = flag("--quiet");
    let w = if flag("--dense") {
        let bytes = mmap(deepseek_v2_lite::layout::DENSE_FILE);
        let (p, dense, q) = unsafe { bytes.align_to::<u16>() };
        assert!(p.is_empty() && q.is_empty());
        eprintln!("weights from the bundle's dense file");
        Weights::from_dense(dense)
    } else {
        Weights::new(|name| ck.bf16(name))
    };
    let noise = arg("--noise").map(|p| {
        let s = p.to_str().unwrap().to_string();
        let parts: Vec<&str> = s.split(':').collect();
        (parts[0].parse::<usize>().unwrap(), parts[1].parse::<f32>().unwrap(), parts.get(2) == Some(&"fixed"))
    });
    let ffn = CpuFfn::new(MixedExperts {
        q4: CpuExperts::new(experts),
        bf16: CheckpointExperts { ck, gu: vec![0.0; 2 * deepseek_v2_lite::D_FF], h: vec![0.0; deepseek_v2_lite::D_FF] },
        use_q4,
        noise,
        calls: 0,
        skip_pos: arg("--q4-skip-pos").map(|p| p.to_str().unwrap().split(',').map(|t| t.parse().unwrap()).collect()).unwrap_or_default(),
        pos: 0,
    });
    // distributional closeness to the oracle under teacher forcing, and the routing margins
    let (mut kl_sum, mut nll_sum, mut agree, mut n_pos) = (0f64, 0f64, 0usize, 0usize);
    let mut margins: Vec<f32> = Vec::new(); // (p6 − p7) / p6 per routing decision
    // per-layer activation ranges: (edge class) -> [layer] -> (Σ rms, Σ absmax, n)
    let mut stats: std::collections::BTreeMap<String, Vec<(f64, f64, usize)>> = std::collections::BTreeMap::new();
    let mut attention = CpuAttention::new(256);
    let no_sink = flag("--no-sink");
    if !no_sink {
        let bytes = mmap(deepseek_v2_lite::layout::SINK_FILE);
        let (p, sink, q) = unsafe { bytes.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty());
        attention.seed(sink);
    }
    let mut st = State::new(256);
    let mut model = CpuModel { attention, ffn };
    eprintln!("weights mapped in {:.1}s; softmax_scale {}", t0.elapsed().as_secs_f32(), st.rope.softmax_scale);

    let mut rep = Report::new(tol);
    let mut pos = 0usize;
    let mut argmax_ok = true;
    for step in meta["steps"].as_array().unwrap() {
        let s = step["step"].as_u64().unwrap();
        let ids: Vec<u32> = step["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want = step["argmax"].as_u64().unwrap() as usize;
        let g = Golden::load(&gdir.join(format!("step{s}.safetensors")));
        let mut got = 0;
        for (t, &tok) in ids.iter().enumerate() {
            if !no_sink && pos < FIRST_POS {
                pos = FIRST_POS; // the sink: seeded, never computed
                continue;
            }
            let t1 = std::time::Instant::now();
            let mut tr = Collect::default();
            model.ffn.routed.pos = pos;
            got = st.decode(&w, tok, pos, &mut model, &Sampling::GREEDY, 0.0, &mut tr);
            let at = format!("step{s} t{t} pos{pos}");
            // teacher-forced metrics at this position
            if let (Some(gl), Some((_, ml))) = (g.row("lm_head.out", t), tr.edges.iter().find(|(k, _)| k == "lm_head.out")) {
                let lsm = |v: &[f32]| {
                    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let z: f64 = v.iter().map(|&x| ((x - m) as f64).exp()).sum();
                    (m as f64, z.ln())
                };
                let (gm, gz) = lsm(gl);
                let (mm, mz) = lsm(ml);
                let mut kl = 0f64;
                for i in 0..gl.len() {
                    let lp_g = (gl[i] as f64 - gm) - gz;
                    let lp_m = (ml[i] as f64 - mm) - mz;
                    kl += lp_g.exp() * (lp_g - lp_m);
                }
                let g_arg = gl.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                let m_arg = ml.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                kl_sum += kl;
                nll_sum -= (ml[g_arg] as f64 - mm) - mz;
                agree += (g_arg == m_arg) as usize;
                n_pos += 1;
            }
            for (key, mine) in &tr.edges {
                if let Some(rest) = key.strip_prefix("model.layers.") {
                    if let Some((ls, name)) = rest.split_once('.') {
                        if let Ok(l) = ls.parse::<usize>() {
                            if ["out", "post_attention_layernorm.out", "self_attn.o_proj.out", "mlp.out", "self_attn.kv_a_layernorm.out"].contains(&name) {
                                let rms = (mine.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / mine.len() as f64).sqrt();
                                let amax = mine.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
                                let e = stats.entry(name.to_string()).or_insert_with(|| vec![(0.0, 0.0, 0); deepseek_v2_lite::NUM_HIDDEN_LAYERS]);
                                e[l].0 += rms;
                                e[l].1 += amax;
                                e[l].2 += 1;
                            }
                        }
                    }
                }
                if key.ends_with("mlp.gate.probs") {
                    let mut p = mine.clone();
                    p.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    margins.push((p[TOP_K - 1] - p[TOP_K]) / p[TOP_K - 1]);
                    continue;
                }
                match g.row(key, t) {
                    Some(gold) => {
                        let (abs, rel) = rep.record(key, mine, gold, &at);
                        if dump > 0 {
                            eprintln!("  {key:<52} abs {abs:.3e} rel {rel:.3e}");
                        }
                    }
                    // a traced name the oracle never captured is a typo on one side or the other
                    None => rep.fail(key, format!("{key}: traced by the kernel but absent from the goldens")),
                }
            }
            for (l, (idx, wt)) in &tr.topk {
                for &e in idx {
                    usage[*l][e] += 1;
                }
                let gk = format!("model.layers.{l}.mlp.gate.topk_idx");
                let gidx: Vec<usize> = g.irow(&gk, t).unwrap().iter().map(|&v| v as usize).collect();
                let gwt = g.row(&format!("model.layers.{l}.mlp.gate.topk_weight"), t).unwrap();
                let rk = format!("model.layers.{l}.mlp.routed.out");
                let grouted = g.row(&rk, t).unwrap();
                let hdim = grouted.len() / gidx.len();
                let (mut a, mut b) = (idx.clone(), gidx.clone());
                a.sort();
                b.sort();
                if a != b {
                    rep.fail(&gk, format!("{gk} {at}: mine {a:?} gold {b:?}"));
                    continue;
                }
                let (mut wt_m, mut wt_g) = (Vec::new(), Vec::new());
                for (j, &e) in idx.iter().enumerate() {
                    let gj = gidx.iter().position(|&v| v == e).unwrap();
                    wt_m.push(wt[j]);
                    wt_g.push(gwt[gj]);
                    rep.record(&rk, &tr.routed[l][j], &grouted[gj * hdim..(gj + 1) * hdim], &format!("{at} expert{e}"));
                }
                rep.record(&format!("model.layers.{l}.mlp.gate.topk_weight"), &wt_m, &wt_g, &at);
            }
            if let (Some(dir), true) = (&dump_attn, pos == dump_pos) {
                let l = dump_layer;
                let edge = |name: &str| -> &[f32] { &tr.edges.iter().find(|(k, _)| k == name).unwrap().1 };
                let key = |n: &str| format!("model.layers.{l}.{n}");
                let h_in: &[f32] = if l == 0 { edge("model.embed_tokens.out") } else { edge(&format!("model.layers.{}.out", l - 1)) };
                let o = edge(&key("self_attn.o_proj.out"));
                let h_out: Vec<f32> = h_in.iter().zip(o).map(|(a, b)| a + b).collect();
                fs::create_dir_all(dir).unwrap();
                let w = |name: &str, v: &[f32]| fs::write(dir.join(name), unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }).unwrap();
                w("h_in.f32", h_in);
                w("cache.f32", &model.attention.cache[l][..(pos + 1) * KV_ROW]);
                w("cos.f32", &st.rope.cos[pos * QK_ROPE_HEAD_DIM..(pos + 1) * QK_ROPE_HEAD_DIM]);
                w("sin.f32", &st.rope.sin[pos * QK_ROPE_HEAD_DIM..(pos + 1) * QK_ROPE_HEAD_DIM]);
                w("q.f32", edge(&key("self_attn.q_proj.out")));
                w("kva.f32", edge(&key("self_attn.kv_a_proj_with_mqa.out")));
                w("c.f32", edge(&key("self_attn.kv_a_layernorm.out")));
                w("heads.f32", edge(&key("self_attn.o_proj.in")));
                w("o.f32", o);
                w("h_out.f32", &h_out);
                // the FFN half of the same layer and the head, for bringup/gpu/ffn.py
                w("ffn_h_out.f32", edge(&key("out")));
                if let Some((idx, wt)) = tr.topk.get(&l) {
                    w("ffn_wt.f32", wt);
                    w("ffn_idx.u32", &idx.iter().map(|&v| v as f32).collect::<Vec<_>>()); // as f32 for one writer; the launcher casts
                }
                w("logits.f32", edge("lm_head.out"));
                let h_final: &[f32] = edge(&format!("model.layers.{}.out", NUM_HIDDEN_LAYERS - 1));
                w("h_final.f32", h_final);
                let b = dl::layer(l);
                fs::write(dir.join("meta.txt"), format!(
                    "layer {l}\npos {pos}\nH {H}\nQ_DIM {Q_DIM}\nKVA_DIM {KVA_DIM}\nKV_ROW {KV_ROW}\nKV_LORA_RANK {KV_LORA_RANK}\nHEADS {HEADS}\nscale {}\nwg 256\nproj_grid {}\nout_grid {}\nproj_lds {}\nout_lds {}\nmla_lds {}\nmerge_lds {}\nchunk_max {}\nchunk_min {}\nsplits_max {}\npart {}\ninput_ln_off {}\nq_proj_off {}\nkv_a_off {}\nkv_a_ln_off {}\nkv_b_off {}\no_proj_off {}\nvram_ceiling_gbps 760\npost_ln_off {}\nshared_gate_off {}\nshared_down_off {}\nnorm_off {}\nlm_head_off {}\nSHARED_FF {}\nVOCAB {}\nTOP_K {}\nD_FF {}\nrecord_bytes {}\nlayer_record_off {}\nrows_norm_lds {}\nshared_down_lds {}\nup_tiles {}\ndown_tiles {}\nup_lds {}\ndown_lds {}\ndense_gate_off {}\ndense_down_off {}\nINTERMEDIATE {}\ndense_down_lds {}\n",
                    st.rope.softmax_scale, (Q_DIM + KVA_DIM).div_ceil(8), H / 8, 4 * (H + 8), 4 * O_IN_DIM, attn::gfx11_consts::MLA_LDS, attn::gfx11_consts::MERGE_LDS, attn::gfx11_consts::CHUNK_MAX, attn::gfx11_consts::CHUNK_MIN, attn::gfx11_consts::SPLITS, attn::gfx11_consts::PART,
                    2 * (b + dl::INPUT_LN), 2 * (b + dl::Q_PROJ), 2 * (b + dl::KV_A), 2 * (b + dl::KV_A_LN), 2 * (b + dl::KV_B), 2 * (b + dl::O_PROJ),
                    2 * (b + dl::POST_LN), 2 * (b + dl::SHARED_GATE), 2 * (b + dl::SHARED_DOWN), 2 * dl::NORM, 2 * dl::LM_HEAD, SHARED_FF, VOCAB_SIZE, TOP_K, D_FF,
                    deepseek_v2_lite::layout::RECORD, deepseek_v2_lite::layout::record(l.max(FIRST_K_DENSE_REPLACE), 0).start, 4 * (H + 8), 4 * SHARED_FF, 2 * D_FF / 4, H / 4, H + 8 * (H / 128), D_FF + 8 * (D_FF / 128),
                    2 * (b + dl::DENSE_GATE), 2 * (b + dl::DENSE_DOWN), INTERMEDIATE_SIZE, 4 * INTERMEDIATE_SIZE)).unwrap();
                eprintln!("attention state for layer {l} at pos {pos} written to {}", dir.display());
            }
            if !quiet {
                eprintln!("step{s} t{t} pos{pos} tok {tok:6} -> argmax {got:6}  ({:.2}s)", t1.elapsed().as_secs_f32());
            }
            pos += 1;
            if dump > 0 && pos >= dump {
                std::process::exit(0);
            }
        }
        let ok = got == want;
        argmax_ok &= ok;
        if !quiet {
            println!("step {s}: argmax {got} golden {want} {}", if ok { "OK" } else { "MISMATCH" });
        }
    }
    if !quiet {
        rep.print();
        println!("\n{} edge classes over tol {tol:.0e}; argmax {}", rep.failures(), if argmax_ok { "all match" } else { "MISMATCH" });
    }
    if let Some(path) = arg("--usage") {
        let mut out = String::new();
        for (l, row) in usage.iter().enumerate() {
            out += &format!("{l}\t{}\n", row.iter().map(|c| c.to_string()).collect::<Vec<_>>().join("\t"));
        }
        fs::write(path, out).unwrap();
    }
    if let Some(path) = arg("--stats") {
        let mut out = String::from("layer");
        for k in stats.keys() {
            out += &format!("\t{k}.rms\t{k}.absmax\t{k}.ratio");
        }
        out += "\n";
        for l in 0..deepseek_v2_lite::NUM_HIDDEN_LAYERS {
            out += &format!("{l}");
            for v in stats.values() {
                let (r, a, n) = v[l];
                let (r, a) = (r / n.max(1) as f64, a / n.max(1) as f64);
                out += &format!("\t{r:.4}\t{a:.3}\t{:.1}", a / r.max(1e-30));
            }
            out += "\n";
        }
        fs::write(path, out).unwrap();
    }
    margins.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let frac = |thr: f32| margins.iter().filter(|&&m| m < thr).count() as f64 / margins.len().max(1) as f64;
    println!(
        "q4-layers {q4_layers}{}{}: teacher-forced KL(oracle||ours) {:.4e} nats/pos, oracle-token NLL {:.4}, top-1 agreement {}/{n_pos}; routing margins (p6−p7)/p6 over {} decisions: min {:.2e}, <1e-2 {:.1}%, <1e-3 {:.1}%, <1e-4 {:.2}%",
        if no_sink { " no-sink" } else { "" },
        arg("--q4-experts").map(|p| format!(" q4-experts {}", p.to_str().unwrap())).unwrap_or_default() + &arg("--noise").map(|p| format!(" noise {}", p.to_str().unwrap())).unwrap_or_default() + &arg("--q4-skip-pos").map(|p| format!(" skip-pos {}", p.to_str().unwrap())).unwrap_or_default(),
        kl_sum / n_pos.max(1) as f64, nll_sum / n_pos.max(1) as f64, agree, margins.len(), margins.first().copied().unwrap_or(0.0), 100.0 * frac(1e-2), 100.0 * frac(1e-3), 100.0 * frac(1e-4)
    );
    if !argmax_ok {
        std::process::exit(1);
    }
}
