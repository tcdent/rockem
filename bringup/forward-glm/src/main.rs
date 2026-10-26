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
//!               [--step N]            with --gpu: the positions again in steps of N tokens, bit for bit the single tokens'
//!               [--step-ids <file>]   the step gate over these ids (the model's prefix first, then any text's: longer than the goldens)
//!               [--elf <path>]        with --gpu: this code object instead of the bundle's (a kernel under test)
//!               [--timeline]          with --gpu: the launches stamped, every launch a step a prow zone (`zones`, lanes card0/card1);
//!                                     the run's `args` attribute names the --elf, so variants compare across runs
//!               [--routing-trace <file>] with --gpu: every MoE layer's picks a position over the --step-ids text, and the layer-ahead predictor scored
//!               [--mtp-accept]        with --gpu: the draft layer's acceptance over the --step-ids text (greedy, and under the model's sampler)
//!               [--gpu]               with --cpu: the blocks on the gfx1100 cards (node/src/glm_kfd.rs), the bundle's code object
//!               [--force-picks]   every MoE layer takes the oracle's expert picks (weights still its own): separates tie flips from the rest
//!               [--detail 1e-3]   also print every row past this relative error, and every position whose picks differ
//!
//! Edges per layer, each `[positions][H]`: attn_in (the normed block input), attn_out,
//! ffn_in, ffn_out, out (the mean of the four streams). The table is the worst
//! row per edge class; each layer also prints its own worst per edge, which is
//! where an error's growth shows. Edges a golden lacks are skipped.

#[path = "../../../node/src/kfd.rs"]
#[allow(dead_code)]
mod kfd;
#[path = "../../../node/src/glm_kfd.rs"]
mod glm_kfd;

use glm_5_3_flash::construct::{CpuExperts, CpuModel};
use glm_5_3_flash::reference::*;
use glm_5_3_flash::*;
use harness::{arg, arg_required, Checkpoint, Golden, Report};
use std::io::Write;
use std::path::PathBuf;

fn main() {
    prow::Run::new("rockem")
        .attrs(prow::build_attrs!())
        .server(std::env::var("PROW_SERVER").unwrap_or_else(|_| "192.168.88.3:7380".into()))
        .attr("tool", "forward-glm")
        .attr("args", std::env::args().skip(1).collect::<Vec<_>>().join(" "))
        .start();
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
    fn wants_edges(&self) -> bool {
        true
    }
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
    // --step-ids: the step gate's own ids, for steps longer than the goldens' positions
    let step_ids: Option<Vec<usize>> = arg("--step-ids").map(|f| std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("{}: {e}", f.to_string_lossy())).split_whitespace().map(|t| t.parse().unwrap()).collect());
    let ctx = n.max(step_ids.as_ref().map_or(0, Vec::len)).max(FIRST_POS + 1);
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
    let stream: &'static glm_5_3_flash::expert::Stream<'static> = Box::leak(Box::new(glm_5_3_flash::expert::Stream::file(Box::leak(Box::new(experts)))));
    let routed: Box<dyn glm_5_3_flash::ffn::Routed> = match experts_mode {
        "a8" => Box::new(CpuExperts::new(stream, &pool)),
        "q4f32" => Box::new(F32Experts { get: Box::new(move |l, e| dequant_expert(stream, l, e)) }),
        "f32" => Box::new(F32Experts { get: Box::new(move |l, e| { let x = load_expert(l, e, &|n| ck.f32s(n)); (x.gate_up, x.down) }) }),
        m => panic!("--experts {m}: a8, q4f32 or f32"),
    };
    let mut model = CpuModel::new(routed, &pool, ctx);
    if harness::flag("--gpu") {
        // the bundle's code object, or one under test (`--elf`); `--timeline` records the cards' launches as zones
        let elf_path = arg("--elf").unwrap_or_else(|| bundle.join(layout::KERNELS_FILE));
        let elf = std::fs::read(&elf_path).unwrap_or_else(|e| panic!("{}: {e}", elf_path.display()));
        model.placed = Some(Box::new(glm_kfd::Cards::new(&elf, &w, ctx, harness::flag("--timeline"), 1 << 30)));
        model.placed.as_mut().unwrap().xpt_loader(Box::new(move |l, e, out| glm_5_3_flash::expert::Weights::at(stream, l, e).slot_bytes(out)));
    }
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
        model.step(&w, ids[t] as u32, t, if force { Some(&forced) } else { None }, &mut c, true);
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
        let lg = &model.logits[..VOCAB_SIZE];
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
    // --step N: the same positions again as steps of N tokens on the cards, against a token at a
    // time — every position's rows, the last logits and the recurrent state must be bit-identical
    let step_gate = if let Some(st) = arg("--step") {
        let st: usize = st.to_str().unwrap().parse().unwrap();
        assert!(model.placed.is_some() && st > 1, "--step needs --gpu and a step of two or more");
        let (ids, n) = match &step_ids {
            Some(s) => {
                assert!(s.len() > FIRST_POS && s[..FIRST_POS].iter().zip(PREFIX_IDS).all(|(&a, b)| a as u32 == b), "--step-ids must start with the model's prefix");
                (&s[..], s.len())
            }
            None => (ids, n),
        };
        let position = glm_5_3_flash::layout::POSITION_W;
        let snap_len = glm_5_3_flash::layout::SNAPSHOT;
        // every position's logits (what SCORE reads), every position's rows, the recurrent state
        let run = |model: &mut CpuModel<_>, step: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            model.seed_prefix(&w, &prefix);
            let mut lg = vec![0f32; n * VOCAB_SIZE];
            let mut t = FIRST_POS;
            while t < n {
                let m = (n - t).min(step);
                if m == 1 {
                    model.step(&w, ids[t] as u32, t, None, &mut common::trace::NoTrace, true);
                } else {
                    let toks: Vec<u32> = ids[t..t + m].iter().map(|&i| i as u32).collect();
                    model.step_batch(&w, &toks, t, None, &mut common::trace::NoTrace, glm_5_3_flash::construct::Logits::All);
                }
                lg[t * VOCAB_SIZE..(t + m) * VOCAB_SIZE].copy_from_slice(&model.logits[..m * VOCAB_SIZE]);
                t += m;
            }
            let mut rows = vec![0f32; n * position];
            for p in FIRST_POS..n {
                model.rows(p, &mut rows[p * position..(p + 1) * position]);
            }
            let mut snap = vec![0f32; snap_len];
            model.snapshot(&mut snap);
            (lg, rows, snap)
        };
        let t0 = std::time::Instant::now();
        let (lg1, rows1, snap1) = run(&mut model, 1);
        let t1 = t0.elapsed().as_secs_f32();
        let t0 = std::time::Instant::now();
        let (lg2, rows2, snap2) = run(&mut model, st);
        let t2 = t0.elapsed().as_secs_f32();
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let (dl, dr, ds) = (diff(&lg1, &lg2), diff(&rows1, &rows2), diff(&snap1, &snap2));
        let maxd = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        // the numeric distance too (a prefill kernel pair is gated by it): top-1 agreement and the mean KL of
        // the step path's distributions from the token-at-a-time path's, over the positions computed
        // and each path's quality on the text itself — the mean −ln p of the next id and the top-1 hits —
        // which says whether two realizations merely differ (near-ties) or one is worse
        let (mut agree, mut kl, mut npos) = (0usize, 0f64, 0usize);
        let (mut nll1, mut nll2, mut hit1, mut hit2, mut scored) = (0f64, 0f64, 0usize, 0usize, 0usize);
        for t in FIRST_POS..n {
            let (a, b) = (&lg1[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE], &lg2[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE]);
            agree += (argmax(a) == argmax(b)) as usize;
            let (ma, mb) = (a.iter().cloned().fold(f32::NEG_INFINITY, f32::max), b.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            let (za, zb): (f64, f64) = (a.iter().map(|&v| ((v - ma) as f64).exp()).sum(), b.iter().map(|&v| ((v - mb) as f64).exp()).sum());
            kl += a.iter().zip(b).map(|(&x, &y)| { let p = ((x - ma) as f64).exp() / za; p * (((x - ma) as f64 - za.ln()) - ((y - mb) as f64 - zb.ln())) }).sum::<f64>();
            npos += 1;
            if t + 1 < n {
                let next = ids[t + 1];
                nll1 -= (a[next] - ma) as f64 - za.ln();
                nll2 -= (b[next] - mb) as f64 - zb.ln();
                hit1 += (argmax(a) == next) as usize;
                hit2 += (argmax(b) == next) as usize;
                scored += 1;
            }
        }
        println!("step gate: {} positions a token at a time in {t1:.1} s, in steps of {st} in {t2:.1} s ({:.1}×); differing f32s — logits {dl} of every position's (max {:.2e}), rows {dr} (max {:.2e}), state {ds} (max {:.2e}); top-1 agreement {:.4}, mean KL {:.3e}", n - FIRST_POS, t1 / t2, maxd(&lg1, &lg2), maxd(&rows1, &rows2), maxd(&snap1, &snap2), agree as f64 / npos as f64, kl / npos as f64);
        println!("step gate: on the text — a token at a time NLL {:.4} (top-1 hits {:.4}), in steps NLL {:.4} (top-1 hits {:.4}) over {scored} positions", nll1 / scored as f64, hit1 as f64 / scored as f64, nll2 / scored as f64, hit2 as f64 / scored as f64);
        if dr > 0 {
            // where it starts: the first position with a differing row, and how many of its floats differ
            let first = rows1.iter().zip(&rows2).position(|(x, y)| x.to_bits() != y.to_bits()).unwrap();
            let p = first / position;
            let in_row = rows1[p * position..(p + 1) * position].iter().zip(&rows2[p * position..(p + 1) * position]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let positions_differing = (FIRST_POS..n).filter(|&q| rows1[q * position..(q + 1) * position].iter().zip(&rows2[q * position..(q + 1) * position]).any(|(x, y)| x.to_bits() != y.to_bits())).count();
            println!("step gate: rows first differ at position {p} (float {} of the row, {in_row} of {position} in it); {positions_differing} positions differ", first % position);
        }
        dl + dr + ds == 0
    } else {
        true
    };
    // --mtp-accept: the draft layer a token behind the main model over the --step-ids text: at every
    // position its draft of the token after the next, against the main model's own distribution there —
    // top-1 agreement (greedy acceptance), the draft's top-1 hitting the text, and the expected
    // acceptance under the model's sampler (Σ min(p, q) over the two truncated distributions)
    if harness::flag("--mtp-accept") {
        let ids = step_ids.as_deref().expect("--mtp-accept needs --step-ids");
        let n = ids.len();
        assert!(model.mtp.is_some() && model.placed.is_some(), "--mtp-accept needs a bundle with the draft layer and --gpu");
        model.seed_prefix(&w, &prefix);
        let mut h = vec![0f32; H];
        let mut h_out = vec![0f32; H];
        let mut q_prev = vec![0f32; VOCAB_SIZE]; // the draft made at t-1 (the cards'): its guess of ids[t+1]
        let mut q_cpu = vec![0f32; VOCAB_SIZE];
        let mut have_q = false;
        let (mut agree, mut hit, mut n_acc, mut sum_acc, mut sum_acc_greedy_hit) = (0usize, 0usize, 0usize, 0f64, 0usize);
        // depth 2: the draft at t+1 from the draft's own hidden state at t and the text's token there (an
        // accepted first draft), its guess of ids[t+3] against the main model's logits at t+2
        // keyed by the position its guess is checked at: the second draft's logits, the first draft's confidence (its
        // top probability under the sampler) and that first draft's acceptance, filled in a position later (081)
        let mut q2: std::collections::HashMap<usize, (Vec<f32>, f64, f64, bool)> = std::collections::HashMap::new();
        let (mut by_q1, mut by_q2, mut by_c1) = (Bands::default(), Bands::default(), Bands::default());
        let (mut agree2, mut n_acc2, mut sum_acc2) = (0usize, 0usize, 0f64);
        let (mut n_draft, mut card_agree, mut card_dmax) = (0usize, 0usize, 0f32);
        let s = &glm_5_3_flash::SAMPLING;
        let t0 = std::time::Instant::now();
        // the draft at `pos` from hidden `h` and the next token `nt`: on the host and on the cards (the
        // served path); `q` receives the cards', the two compared
        let mut draft_both = |model: &mut CpuModel<_>, h: &[f32], nt: u32, pos: usize, q: &mut [f32], q_cpu: &mut [f32], h_out: &mut [f32]| {
            let (mut th, mut tc) = (Collect::default(), Collect::default());
            model.draft(&w, h, nt, pos, q_cpu, h_out, &mut th);
            model.draft_batch(&w, h, &[nt], pos, 1, q, &mut tc);
            let d = q.iter().zip(q_cpu.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            card_dmax = card_dmax.max(d);
            card_agree += (argmax(q) == argmax(q_cpu)) as usize;
            n_draft += 1;
            if pos < FIRST_POS + 4 || (d > 1.0 && pos < 64) {
                // the stages the cards report, each against the host's
                let stages: Vec<String> = tc.edges.iter().filter_map(|(_, name, c)| th.edges.iter().find(|(_, n2, _)| n2 == name).map(|(_, _, hv)| {
                    let dm = c.iter().zip(hv.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let hm = hv.iter().fold(0f32, |a, &x| a.max(x.abs()));
                    format!("{name} {dm:.2e} (host max {hm:.2e})")
                })).collect();
                // the cards' shared expert, derived: the FFN residual less the attention state and the routed sum
                let get = |t: &Collect, name: &str| t.edges.iter().find(|(_, n2, _)| n2 == name).map(|(_, _, v)| v.clone());
                let shared = match (get(&tc, "mtp.ffn"), get(&tc, "mtp.attn"), get(&tc, "routed"), get(&th, "mtp.shared")) {
                    (Some(f), Some(a), Some(r), Some(hs)) => {
                        let dm = (0..H).map(|i| (f[i] - a[i] - r[i] - hs[i]).abs()).fold(0f32, f32::max);
                        format!(", shared (derived) {dm:.2e}")
                    }
                    _ => String::new(),
                };
                println!("  draft at {pos}: cards' top-1 {} host's top-1 {} max |Δ logit| {d:.3e}; {}{shared}", argmax(q), argmax(q_cpu), stages.join(", "));
            }
        };
        // the prefix's draft rows: position 0 from the prefix's hidden state and the prefix's second id
        let (h0, h1) = (model.prefix_hidden(0).to_vec(), model.prefix_hidden(1).to_vec());
        draft_both(&mut model, &h0, ids[1] as u32, 0, &mut q_prev, &mut q_cpu, &mut h_out);
        for t in 1..n - 1 {
            if t >= FIRST_POS {
                model.step(&w, ids[t] as u32, t, None, &mut common::trace::NoTrace, true);
                model.hidden(&mut h);
            } else {
                h.copy_from_slice(&h1);
            }
            if let Some((q, c1, a1, g1)) = q2.remove(&t) {
                // the depth-2 draft made at t-1 (from the draft at t-2) and the main model's logits at t both predict ids[t+1]
                let p = &model.logits[..VOCAB_SIZE];
                let g2 = argmax(p) == argmax(&q);
                agree2 += g2 as usize;
                let a2 = overlap(p, &q, s);
                sum_acc2 += a2;
                n_acc2 += 1;
                by_q2.add(p, &q, s);
                // the third position's yield (both drafts accepted), by the first draft's confidence: the drafting gate
                by_c1.add_point(c1, a1 * a2, g1 && g2);
            }
            if have_q {
                // q_prev (drafted at t-1) and the main model's logits at t both predict ids[t+1]
                let p = &model.logits[..VOCAB_SIZE];
                let (ap, aq) = (argmax(p), argmax(&q_prev));
                agree += (ap == aq) as usize;
                hit += (aq == ids[t + 1]) as usize;
                let acc = overlap(p, &q_prev, s);
                sum_acc += acc;
                by_q1.add(p, &q_prev, s);
                if let Some(e) = q2.get_mut(&(t + 1)) {
                    (e.2, e.3) = (acc, ap == aq);
                }
                sum_acc_greedy_hit += (ap == ids[t + 1]) as usize;
                n_acc += 1;
                if t < FIRST_POS + 4 {
                    println!("  at {t}: text {} main top-1 {ap} draft top-1 {aq} acceptance {acc:.3}", ids[t + 1]);
                }
            }
            draft_both(&mut model, &h, ids[t + 1] as u32, t, &mut q_prev, &mut q_cpu, &mut h_out);
            have_q = t >= FIRST_POS;
            if t >= FIRST_POS && t + 2 < n {
                // the served path: the cards' draft at t+1 from the cards' draft output at t
                let mut q = vec![0f32; VOCAB_SIZE];
                let c1 = common::sample::dist(&q_prev, s)[0].1;
                model.draft_next(&w, ids[t + 2] as u32, t + 1, &mut q, &mut common::trace::NoTrace);
                q2.insert(t + 2, (q, c1, 0.0, false));
            }
            if t % 100 == 0 {
                println!("  mtp: {t} positions, {:.1} s", t0.elapsed().as_secs_f32());
            }
        }
        println!("mtp draft: cards vs host over {n_draft} positions — top-1 agree {:.3}, max |Δ logit| {card_dmax:.3e}", card_agree as f64 / n_draft as f64);
        println!("mtp accept, depth 2: {n_acc2} positions — draft top-1 = main top-1 {:.3}; expected acceptance under the sampler {:.3}", agree2 as f64 / n_acc2.max(1) as f64, sum_acc2 / n_acc2.max(1) as f64);
        by_q1.print("first draft by its token's probability q(x)");
        by_q2.print("second draft by its token's probability q(x)");
        by_c1.print("both drafts accepted, by the first draft's top probability");
        println!("mtp accept: {n_acc} positions — draft top-1 = main top-1 {:.3}; draft top-1 = text {:.3} (main top-1 = text {:.3}); expected acceptance under the sampler (T {} top-k {} top-p {}) {:.3}; {:.1} s", agree as f64 / n_acc as f64, hit as f64 / n_acc as f64, sum_acc_greedy_hit as f64 / n_acc as f64, s.temperature, s.top_k, s.top_p, sum_acc / n_acc as f64, t0.elapsed().as_secs_f32());
    }
    // --routing-trace: the --step-ids text a token at a time; every MoE layer's picks a position to the
    // file (`pos layer e0..e7`, for the cache replay), and the layer-ahead predictor scored here: layer
    // l+1's router over layer l's FFN input, against l+1's picks (today's predictor, layer l's router over
    // the attention block's input, from the profile)
    if let Some(out) = arg("--routing-trace") {
        use glm_5_3_flash::weights::Ffn;
        let ids = step_ids.as_deref().expect("--routing-trace needs --step-ids");
        assert!(model.placed.is_some(), "--routing-trace needs --gpu");
        model.seed_prefix(&w, &prefix);
        model.profile = Default::default();
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
        let mut logits = vec![0f32; E];
        let (mut ahead, mut ahead_n) = (0usize, 0usize);
        // the analog's first margin off the cards (098) against the same picks, where the next layer is KDA
        let mut an_logits = vec![0f32; E];
        let (mut an_hit, mut an_n, mut ra_kda) = (0usize, 0usize, 0usize);
        let t0 = std::time::Instant::now();
        for t in FIRST_POS..ids.len() {
            let mut tc = Collect::default();
            model.step(&w, ids[t] as u32, t, None, &mut tc, true);
            for l in 0..NUM_HIDDEN_LAYERS - 1 {
                if let (Ffn::Moe(next), Some((_, actual))) = (&w.layers[l + 1].ffn, tc.topk.iter().find(|(l2, _)| *l2 == l + 1)) {
                    if model.placed.as_mut().unwrap().analog_first(l, 1, &mut an_logits) {
                        let pred = glm_5_3_flash::ffn::picks(next, &mut an_logits);
                        an_hit += actual.iter().filter(|e| pred.contains(e)).count();
                        an_n += 1;
                        if let Some((_, _, xn)) = tc.edges.iter().find(|(l2, name, _)| *l2 == l && name == "ffn_in") {
                            common::gemv::bf16_par::<E, H>(next.router, xn, &mut logits, &pool);
                            let pred = glm_5_3_flash::ffn::picks(next, &mut logits);
                            ra_kda += actual.iter().filter(|e| pred.contains(e)).count();
                        }
                    }
                }
            }
            for (l, idx) in &tc.topk {
                writeln!(f, "{t} {l} {}", idx.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(" ")).unwrap();
            }
            for (l, name, xn) in &tc.edges {
                if name != "ffn_in" || l + 1 >= NUM_HIDDEN_LAYERS {
                    continue;
                }
                if let (Ffn::Moe(next), Some((_, actual))) = (&w.layers[l + 1].ffn, tc.topk.iter().find(|(l2, _)| *l2 == l + 1)) {
                    common::gemv::bf16_par::<E, H>(next.router, xn, &mut logits, &pool);
                    let pred = glm_5_3_flash::ffn::picks(next, &mut logits);
                    ahead += actual.iter().filter(|e| pred.contains(e)).count();
                    ahead_n += 1;
                }
            }
            if t % 200 == 0 {
                println!("  routing trace: {t} positions, {:.1} s", t0.elapsed().as_secs_f32());
            }
        }
        let p = &model.profile;
        println!("routing trace: {} positions to {}; layer-ahead predictor names {:.2} of {TOP_K} picks (layer l+1's router over layer l's FFN input); today's names {:.2} (layer l's router over the attention input); {:.1} s", ids.len() - FIRST_POS, out.display(), ahead as f64 / ahead_n.max(1) as f64, p.pre_agree as f64 / p.moe_layers.max(1) as f64, t0.elapsed().as_secs_f32());
        if an_n > 0 {
            println!("routing trace: the analog's first margin (098) names {:.2} of {TOP_K} picks over {an_n} KDA-next layer positions; the route-ahead names {:.2} on the same", an_hit as f64 / an_n as f64, ra_kda as f64 / an_n as f64);
        }
    }
    std::process::exit(if report.failures() > 0 || !step_gate { 1 } else { 0 });
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0
}
/// Expected acceptance of a draft from `q` verified against `p`: Σ_x min(p(x), q(x)).
/// Acceptance in bands of the draft's probability (081): a band's mass (drafts falling in it) and its accepted
/// mass under the sampler, and its greedy count and top-1 agreements. `add` spreads a position over the tokens the
/// draft could sample, each x weighted q(x) and accepted min(p(x), q(x)) — the expectation of rejection sampling.
#[derive(Default)]
struct Bands {
    mass: [f64; BANDS],
    acc: [f64; BANDS],
    n: [usize; BANDS],
    top1: [usize; BANDS],
}
const BANDS: usize = 10;
impl Bands {
    fn band(c: f64) -> usize {
        ((c * BANDS as f64) as usize).min(BANDS - 1)
    }
    fn add(&mut self, p: &[f32], q: &[f32], s: &common::sample::Sampling) {
        let (dp, dq) = (common::sample::dist(p, s), common::sample::dist(q, s));
        for (id, qx) in &dq {
            let px = dp.iter().find(|(i, _)| i == id).map_or(0.0, |x| x.1);
            let b = Self::band(*qx);
            self.mass[b] += qx;
            self.acc[b] += qx.min(px);
        }
        let b = Self::band(dq[0].1);
        self.n[b] += 1;
        self.top1[b] += (argmax(p) == argmax(q)) as usize;
    }
    fn add_point(&mut self, c: f64, acc: f64, top1: bool) {
        let b = Self::band(c);
        self.mass[b] += 1.0;
        self.acc[b] += acc;
        self.n[b] += 1;
        self.top1[b] += top1 as usize;
    }
    /// A line a band, and the gate's view: drafts kept at or above the band, their share and acceptance.
    fn print(&self, what: &str) {
        let (tm, tn) = (self.mass.iter().sum::<f64>(), self.n.iter().sum::<usize>());
        println!("mtp bands, {what} — band, share, acceptance (sampler) | greedy share, top-1 | kept at ≥ band: share, acceptance");
        let (mut km, mut ka) = (0f64, 0f64);
        for b in (0..BANDS).rev() {
            km += self.mass[b];
            ka += self.acc[b];
            if self.mass[b] == 0.0 && self.n[b] == 0 {
                continue;
            }
            println!("  {:.1}–{:.1}  {:.3}  {:.3} | {:.3}  {:.3} | {:.3}  {:.3}", b as f64 / BANDS as f64, (b + 1) as f64 / BANDS as f64, self.mass[b] / tm, self.acc[b] / self.mass[b].max(1e-12), self.n[b] as f64 / tn.max(1) as f64, self.top1[b] as f64 / self.n[b].max(1) as f64, km / tm, ka / km.max(1e-12));
        }
    }
}

fn overlap(p: &[f32], q: &[f32], s: &common::sample::Sampling) -> f64 {
    let (dp, dq) = (common::sample::dist(p, s), common::sample::dist(q, s));
    let mut acc = 0f64;
    for (id, pq) in &dq {
        if let Some((_, pp)) = dp.iter().find(|(i, _)| i == id) {
            acc += pq.min(*pp);
        }
    }
    acc
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
fn dequant_expert(stream: &glm_5_3_flash::expert::Stream, l: usize, e: usize) -> (Vec<f32>, Vec<f32>) {
    use common::q4::{self, GROUP};
    use glm_5_3_flash::expert::Weights;
    use glm_5_3_flash::expert::{DOWN_PARTS, DOWN_ROWS, GU_PARTS, GU_ROWS};
    let w = Weights::at(stream, l, e);
    // an item's rows expanded into their place in the full matrix (`[rows][k]`)
    let expand = |rows: usize, k: usize, iw: &[u8], is: &[f32], r_base: usize, out: &mut [f32]| {
        for r in 0..rows {
            for kk in 0..k {
                out[(r_base + r) * k + kk] = q4::unpack(k, iw, r, kk) as f32 * is[q4::scale_index(k, r, kk / GROUP)];
            }
        }
    };
    let mut gu = vec![0f32; 2 * D_FF * H];
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        expand(GU_ROWS, H, gw, gs, c * GU_ROWS, &mut gu);
        expand(GU_ROWS, H, uw, us, D_FF + c * GU_ROWS, &mut gu);
    }
    let mut down = vec![0f32; H * D_FF];
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        expand(DOWN_ROWS, D_FF, dw, ds, c * DOWN_ROWS, &mut down);
    }
    (gu, down)
}
