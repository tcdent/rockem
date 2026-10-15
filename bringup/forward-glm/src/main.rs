//! forward-glm — run the f32 transcription of GLM-5.3-Flash over an oracle
//! passage, layer by layer from the FP8 checkpoint (the oracle's own order: one
//! layer's weights resident at a time), and diff every edge the oracle saved.
//!
//!   forward-glm --goldens <bringup/goldens/glm/SET> [--ckpt ~/hf-cache/glm-5.3-flash] [--layers N] [--tol 1e-4]
//!               [--cpu <bundle dir>]  the shipped CPU realization (construct::CpuModel) from the bundle's dense, expert
//!                                     and prefix files instead of the f32 reference from the checkpoint: the kernel gate
//!               [--experts a8|q4f32|f32]  with --cpu: the routed experts as shipped (INT4 weights, INT8 activations), the
//!                                     bundle's INT4 bytes dequantized through f32 arithmetic (the oracle's q4 recipe), or
//!                                     the checkpoint's unquantized experts — what separates the recipe's cost from a kernel error
//!               [--positions N]       stop after N computed positions
//!               [--force-picks]   every MoE layer takes the oracle's expert picks (weights still its own): separates tie flips from the rest
//!               [--detail 1e-3]   also print every row past this relative error, and every position whose picks differ
//!
//! Edges per layer, each `[positions][H]`: attn_in (the normed block input), attn_out,
//! ffn_in, ffn_out, out (the mean of the four streams). The table is the worst
//! row per edge class; each layer also prints its own worst per edge, which is
//! where an error's growth shows. Edges a golden lacks are skipped.

use glm_5_3_flash::construct::{CpuExperts, CpuModel};
use glm_5_3_flash::reference::*;
use glm_5_3_flash::*;
use harness::{arg, arg_required, Checkpoint, Golden, Report};
use std::path::PathBuf;

fn main() {
    let gdir = arg_required("--goldens");
    let ckpt = arg("--ckpt").unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("hf-cache/glm-5.3-flash"));
    let layers: usize = arg("--layers").map_or(NUM_HIDDEN_LAYERS, |v| v.to_str().unwrap().parse().unwrap());
    let force = harness::flag("--force-picks");
    let detail: f32 = arg("--detail").map_or(0.0, |v| v.to_str().unwrap().parse().unwrap());
    let tol: f32 = arg("--tol").map_or(1e-4, |v| v.to_str().unwrap().parse().unwrap());
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<usize> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let n = ids.len();
    let gold = Golden::load(&gdir.join("edges.safetensors"));
    let routing = Golden::load(&gdir.join("routing.safetensors"));
    let mut report = Report::new(tol);
    let ck = if arg("--cpu").is_some() && arg("--experts").map_or(true, |v| v.to_str() != Some("f32")) { Checkpoint::open_lazy(&ckpt) } else { Checkpoint::open(&ckpt) };
    println!("{n} positions, {layers} layers, goldens {}", gdir.display());

    if let Some(bundle) = arg("--cpu") {
        let experts = arg("--experts").map_or("a8".to_string(), |v| v.to_string_lossy().into_owned());
        let positions = arg("--positions").map_or(usize::MAX, |v| v.to_str().unwrap().parse().unwrap());
        cpu_gate(&bundle, &ck, &experts, positions, &ids, layers, &gold, &routing, force, detail, &mut report, &meta);
        return;
    }
    let emb = ck.bf16("model.language_model.embed_tokens.weight");
    let mut streams = vec![0f32; n * HC * H];
    for (t, &id) in ids.iter().enumerate() {
        let row: Vec<f32> = emb[id * H..(id + 1) * H].iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();
        report.record("embed", &row, gold.row("embed", t).unwrap(), &format!("pos {t}"));
        for j in 0..HC {
            streams[(t * HC + j) * H..(t * HC + j + 1) * H].copy_from_slice(&row);
        }
    }

    for l in 0..layers {
        let t0 = std::time::Instant::now();
        let layer = load(l, &|n| ck.f32s(n));
        let tl = t0.elapsed().as_secs_f32();
        let mut st = layer.state();
        let mut line = Vec::new();
        let touched = std::sync::atomic::AtomicUsize::new(0);
        let forced: Option<Vec<usize>> = (force && !DENSE_MLP[l]).then(|| (0..n).flat_map(|t| routing.irow(&format!("l{l}.topk"), t).unwrap().iter().map(|&e| e as usize).collect::<Vec<_>>()).collect());
        let r = layer.forward(&mut streams, n, &mut st, &mut |name, data| {
            let key = format!("l{l}.{name}");
            if gold.row(&key, 0).is_none() {
                return; // a golden run with fewer edges (--no-edges keeps each layer's out)
            }
            let mut worst = 0f32;
            for t in 0..n {
                let (_, rel) = report.record(&format!("model.layers.{l}.{name}"), &data[t * H..(t + 1) * H], gold.row(&key, t).unwrap(), &format!("pos {t}"));
                if detail > 0.0 && rel > detail {
                    println!("    l{l}.{name} pos {t}: rel {rel:.2e}");
                }
                worst = worst.max(rel);
            }
            line.push(format!("{name} {worst:.1e}"));
        }, &|e| {
            touched.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            load_expert(l, e, &|n| ck.f32s(n))
        }, forced.as_deref());
        let touched = touched.into_inner();
        if let (glm_5_3_flash::reference::LayerState::Mla(m), Some(_)) = (&st, routing.irow(&format!("l{l}.index"), 0)) {
            // the indexer's choices against the oracle's, as sets of positions
            let (mut differ, mut first) = (0, None);
            for t in 0..n {
                let mut g: Vec<usize> = routing.irow(&format!("l{l}.index"), t).unwrap().iter().filter(|&&j| j >= 0).map(|&j| j as usize).collect();
                g.sort();
                g.dedup();
                if g != m.selected[t] {
                    differ += 1;
                    first.get_or_insert(t);
                    if detail > 0.0 {
                        let only_ours = m.selected[t].iter().filter(|j| g.binary_search(j).is_err()).count();
                        let only_theirs = g.iter().filter(|j| m.selected[t].binary_search(j).is_err()).count();
                        println!("    l{l} pos {t}: selection differs, {only_ours} only ours, {only_theirs} only the oracle's (of {})", g.len());
                    }
                }
            }
            if differ > 0 {
                report.fail(&format!("model.layers.{l}.index"), format!("{differ} selections differ at layer {l}, first at pos {}", first.unwrap()));
            }
            line.push(format!("index: {differ} of {n} selections differ"));
        }
        if let Some(r) = r {
            // the picks as sets (the reference's top-k is unsorted), the weights matched by expert
            let (mut flips, mut wworst) = (0, 0f32);
            for t in 0..n {
                let gi = routing.irow(&format!("l{l}.topk"), t).unwrap();
                let gw = routing.row(&format!("l{l}.weights"), t).unwrap();
                let mine = &r.idx[t * TOP_K..(t + 1) * TOP_K];
                let miss = mine.iter().filter(|e| !gi.contains(&(**e as i64))).count();
                if miss > 0 && detail > 0.0 {
                    println!("    l{l} pos {t}: {miss} picks differ, ours {mine:?} oracle {gi:?}");
                }
                flips += miss;
                for (k, &e) in mine.iter().enumerate() {
                    if let Some(j) = gi.iter().position(|&g| g == e as i64) {
                        wworst = wworst.max((r.weight[t * TOP_K + k] - gw[j]).abs() / gw[j].abs());
                    }
                }
            }
            if flips > 0 {
                report.fail(&format!("model.layers.{l}.picks"), format!("{flips} picks differ at layer {l}"));
            }
            line.push(format!("picks {} differ, weights {wworst:.1e}, {touched} experts", flips));
        }
        println!("layer {l:2} {} load {tl:.1}s run {:.1}s  {}", if LINEAR_ATTENTION[l] { "kda" } else { "mla" }, t0.elapsed().as_secs_f32() - tl, line.join("  "));
    }
    if layers == NUM_HIDDEN_LAYERS {
        // the head: the four streams' mean, norm, lm_head; then the logits against the oracle's and the text
        let mean: Vec<f32> = (0..n * H).map(|i| (0..HC).map(|j| streams[(i / H) * HC * H + j * H + i % H]).sum::<f32>() / HC as f32).collect();
        let x = rmsnorm(&mean, H, &ck.f32s("model.language_model.norm.weight"), RMS_NORM_EPS);
        let logits = matmul(&ck.f32s("lm_head.weight"), VOCAB_SIZE, H, &x, n);
        let (mut agree, mut agree_oracle, mut nll) = (0, 0, 0f64);
        for t in 0..n {
            let row = &logits[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE];
            let g = gold.row("logits", t).unwrap();
            report.record("logits", row, g, &format!("pos {t}"));
            let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
            agree_oracle += (am(row) == am(g)) as usize;
            if t + 1 < n {
                agree += (am(row) == ids[t + 1]) as usize;
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = mx + row.iter().map(|&v| ((v as f64) - mx).exp()).sum::<f64>().ln();
                nll += lse - row[ids[t + 1]] as f64;
            }
        }
        println!("head: top-1 agrees with the oracle's at {agree_oracle}/{n}; with the text {:.3} (oracle {:.3}); NLL {:.4} (oracle {:.4})",
            agree as f64 / (n - 1) as f64, meta["top1_agreement"].as_f64().unwrap_or(f64::NAN), nll / (n - 1) as f64, meta["nll"].as_f64().unwrap_or(f64::NAN));
    }
    report.print();
    std::process::exit(if report.failures() > 0 { 1 } else { 0 });
}

/// Every edge of one token, by (layer, name), and the MoE layers' picks.
#[derive(Default)]
struct Collect {
    edges: Vec<(usize, String, Vec<f32>)>,
    topk: Vec<(usize, Vec<usize>)>,
}
impl common::trace::Trace for Collect {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        self.edges.push((layer, name.to_string(), data.to_vec()));
    }
    fn topk(&mut self, layer: usize, idx: &[usize], _weight: &[f32]) {
        self.topk.push((layer, idx.to_vec()));
    }
    fn routed(&mut self, _: usize, _: usize, _: &[f32]) {}
}

/// The kernel gate: the shipped CPU realization from the bundle, token by token from the
/// prefix state, every edge the oracle saved diffed against it, the picks forced to the
/// oracle's when asked (what separates a tie flip from a kernel error).
#[allow(clippy::too_many_arguments)]
fn cpu_gate(bundle: &std::path::Path, ck: &Checkpoint, experts_mode: &str, positions: usize, ids: &[usize], layers: usize, gold: &Golden, routing: &Golden, force: bool, detail: f32, report: &mut Report, meta: &serde_json::Value) {
    use glm_5_3_flash::layout;
    assert_eq!(layers, NUM_HIDDEN_LAYERS, "--cpu runs every layer: the realization has no partial forward");
    let n = ids.len();
    let map = |name: &str| -> memmap2::Mmap {
        let f = std::fs::File::open(bundle.join(name)).unwrap_or_else(|e| panic!("{}: {e}", bundle.join(name).display()));
        unsafe { memmap2::Mmap::map(&f).unwrap() }
    };
    let (dense, experts) = (map(layout::DENSE_FILE), map(layout::EXPERTS_FILE));
    dense.advise(memmap2::Advice::WillNeed).unwrap();
    experts.advise(memmap2::Advice::Random).unwrap();
    let w = glm_5_3_flash::weights::Weights::from_dense(&dense[..]);
    let prefix_b = std::fs::read(bundle.join(layout::PREFIX_FILE)).unwrap();
    let prefix: Vec<f32> = prefix_b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let pool = glm_5_3_flash::construct::Threads::new(&glm_5_3_flash::construct::physical_cores());
    let routed: Box<dyn glm_5_3_flash::ffn::Routed> = match experts_mode {
        "a8" => Box::new(CpuExperts::new(&experts[..], &pool)),
        "q4f32" => Box::new(F32Experts { get: Box::new(move |l, e| dequant_record(&experts[glm_5_3_flash::layout::record(l, e)])) }),
        "f32" => Box::new(F32Experts { get: Box::new(move |l, e| { let x = load_expert(l, e, &|n| ck.f32s(n)); (x.gate_up, x.down) }) }),
        m => panic!("--experts {m}: a8, q4f32 or f32"),
    };
    let mut model = CpuModel::new(routed, &pool, n.max(FIRST_POS + 1));
    model.seed_prefix(&w, &prefix);
    for (t, &id) in ids.iter().enumerate().take(FIRST_POS) {
        assert_eq!(id as u32, PREFIX_IDS[t], "the passage does not start with the model's prefix");
    }
    let n = n.min(FIRST_POS.saturating_add(positions));
    println!("cpu gate : {} from {}, experts {experts_mode}, {} positions after the prefix{}", if force { "forced picks" } else { "the router's picks" }, bundle.display(), n - FIRST_POS, if gold.row("l0.attn_in", 0).is_some() { "" } else { " (a golden with layer outputs only)" });
    let names = ["attn_in", "attn_out", "ffn_in", "routed", "shared", "ffn_out", "out"];
    let mut flips = 0;
    let (mut agree, mut agree_oracle, mut nll) = (0usize, 0usize, 0f64);
    let t0 = std::time::Instant::now();
    for t in FIRST_POS..n {
        let forced: Vec<[usize; TOP_K]> = (FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS).map(|l| routing.irow(&format!("l{l}.topk"), t).unwrap().iter().map(|&e| e as usize).collect::<Vec<_>>().try_into().unwrap()).collect();
        let mut c = Collect::default();
        model.step(&w, ids[t] as u32, t, if force { Some(&forced) } else { None }, &mut c);
        for (l, name, data) in &c.edges {
            let key = if *l == common::trace::NONE { name.clone() } else { format!("l{l}.{name}") };
            if *l != common::trace::NONE && !names.contains(&name.as_str()) {
                continue;
            }
            let Some(g) = gold.row(&key, t) else { continue };
            let class = if *l == common::trace::NONE { name.clone() } else { format!("model.layers.{l}.{name}") };
            let (_, rel) = report.record(&class, data, g, &format!("pos {t}"));
            if detail > 0.0 && rel > detail {
                println!("    {key} pos {t}: rel {rel:.2e}");
            }
        }
        for (l, mine) in &c.topk {
            let gi = routing.irow(&format!("l{l}.topk"), t).unwrap();
            let miss = mine.iter().filter(|e| !gi.contains(&(**e as i64))).count();
            if miss > 0 && detail > 0.0 {
                println!("    l{l} pos {t}: {miss} picks differ, ours {mine:?} oracle {gi:?}");
            }
            flips += miss;
        }
        let lg = &model.logits;
        let am = |r: &[f32]| r.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0;
        if let Some(g) = gold.row("logits", t) {
            agree_oracle += (am(lg) == am(g)) as usize;
        }
        if t + 1 < n {
            agree += (am(lg) == ids[t + 1]) as usize;
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + lg.iter().map(|&v| ((v as f64) - mx).exp()).sum::<f64>().ln();
            nll += lse - lg[ids[t + 1]] as f64;
        }
        if (t - FIRST_POS + 1) % 8 == 0 || t + 1 == n {
            println!("  pos {t}: {:.1} s a token so far", t0.elapsed().as_secs_f32() / (t - FIRST_POS + 1) as f32);
        }
    }
    let scored = (n - FIRST_POS - 1) as f64;
    println!("head: top-1 agrees with the oracle's at {agree_oracle}/{}; with the text {:.3} (oracle {:.3}); NLL {:.4} (oracle {:.4}); picks differ {flips}", n - FIRST_POS, agree as f64 / scored, meta["top1_agreement"].as_f64().unwrap_or(f64::NAN), nll / scored, meta["nll"].as_f64().unwrap_or(f64::NAN));
    report.print();
    std::process::exit(if report.failures() > 0 { 1 } else { 0 });
}

/// Routed experts in f32 arithmetic with f32 activations (the oracle's recipe), from whatever
/// `get(l, e)` returns: `(gate_up [2·D_FF][H], down [H][D_FF])` widened.
struct F32Experts<'a> {
    get: Box<dyn Fn(usize, usize) -> (Vec<f32>, Vec<f32>) + 'a>,
}
impl glm_5_3_flash::ffn::Routed for F32Experts<'_> {
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        let mut act = vec![0f32; D_FF];
        for k in 0..TOP_K {
            let (gu, dn) = (self.get)(l, idx[k]);
            let g = matmul_serial(&gu, 2 * D_FF, H, x, 1);
            for i in 0..D_FF {
                act[i] = common::swiglu::silu(g[i].min(SWIGLU_LIMIT)) * g[D_FF + i].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
            }
            out[k * H..(k + 1) * H].copy_from_slice(&matmul_serial(&dn, H, D_FF, &act, 1));
        }
    }
}

/// One `q4g128` record's weights back to f32: integer × scale, the oracle's dequantized INT4.
fn dequant_record(rec: &[u8]) -> (Vec<f32>, Vec<f32>) {
    use common::q4::{self, GROUP};
    use glm_5_3_flash::layout::{DOWN_W, GATE_UP_S, GATE_UP_W};
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let (guw, rest) = rec.split_at(GATE_UP_W);
    let (gus, rest) = rest.split_at(4 * GATE_UP_S);
    let (dw, dsb) = rest.split_at(DOWN_W);
    let (gus, ds) = (f32s(gus), f32s(dsb));
    let expand = |m: usize, k: usize, w: &[u8], s: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; m * k];
        for r in 0..m {
            for kk in 0..k {
                out[r * k + kk] = q4::unpack(k, w, r, kk) as f32 * s[q4::scale_index(k, r, kk / GROUP)];
            }
        }
        out
    };
    (expand(2 * D_FF, H, guw, &gus), expand(H, D_FF, dw, &ds))
}
