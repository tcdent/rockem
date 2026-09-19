//! bundle — build the DeepSeek-V2-Lite bundle: everything the node loads.
//!
//!   bundle --snapshot <hf snapshot dir> --out <dir> [--rtn | --calib <ids file>] [--dense-only] [--kernels <gfx1100 .elf>]
//!
//! Writes `experts.q4g128.bin` (one record per routed expert in (layer,
//! expert) order, the four streams `deepseek_v2_lite::expert` reads),
//! `dense.bf16.bin` (every non-expert tensor in `layout::dense` order),
//! `vocab.bin` (token bytes for the console), `sink.f32` (the BOS token's cache
//! row per layer, computed at full precision — analysis.md §5), the checkpoint's
//! `tokenizer.json` and `tokenizer_config.json` verbatim (the edge's half:
//! encoding and the chat template), and `manifest.toml` naming the checkpoint,
//! the recipe string, the wire's model constants and each file's sha256.
//!
//! Quantization is symmetric INT4, absmax/7 per group of 128, integers
//! chosen by GPTQ (`common::q4::gptq`, `RECIPE_GPTQ`): the model's
//! calibration sequences (`models/deepseek-v2-lite/calib.ids`, one per line,
//! BOS first; `--calib` names another file) run through the bf16 model, every
//! routed expert's input is kept per MoE layer, and each expert is quantized
//! against them — gate/up against the layer's input Hessian, down against its
//! own SwiGLU hidden states. Position 0 is included, as it was for the goldens
//! the reference stream was calibrated on (analysis.md §3, §5). `--rtn` is the
//! round-to-nearest twin (`RECIPE_RTN`), kept for pricing what calibration buys.

use common::linalg::{chol_of_inverse, dot};
use common::q4::{self, GROUP};
use deepseek_v2_lite::layout::{self, dense as d, RECIPE_GPTQ, RECIPE_RTN, RECORD};
use deepseek_v2_lite::*;
use harness::{arg_required as arg, flag, Checkpoint};
use std::fs;
use std::io::Write;

const D_MODEL: usize = H;

/// RTN: per (row, group) scale = absmax / 7; q = round(w / scale) in [-8, 7].
fn quantize(m: usize, k: usize, w: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let ng = k / GROUP;
    let mut q = vec![0i8; m * k];
    let mut s = vec![0f32; m * ng];
    for r in 0..m {
        for g in 0..ng {
            let seg = &w[r * k + g * GROUP..r * k + (g + 1) * GROUP];
            let amax = seg.iter().fold(0f32, |a, v| a.max(v.abs()));
            let scale = amax / 7.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, v) in seg.iter().enumerate() {
                q[r * k + g * GROUP + i] = (v * inv).round().clamp(-8.0, 7.0) as i8;
            }
            s[r * ng + g] = scale;
        }
    }
    (q, s)
}

/// What one MoE layer's calibration leaves behind: the expert inputs
/// `x` (`[n][H]`) and the upper factor of their damped Hessian's inverse.
struct Calib {
    x: Vec<f32>,
    u: Vec<f32>,
}
const DAMP: f32 = 0.01;

/// `Uᵀ·U = (XᵀX + damp·mean(diag)·I)⁻¹` for row-major `x: [n][k]`. The
/// Hessian is built from a transposed copy so both dots run along rows.
fn hessian_factor(k: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len() / k;
    let mut xt = vec![0f32; k * n];
    for r in 0..n {
        for c in 0..k {
            xt[c * n + r] = x[r * k + c];
        }
    }
    let mut h = vec![0f32; k * k];
    for i in 0..k {
        for j in 0..=i {
            let v = dot(&xt[i * n..(i + 1) * n], &xt[j * n..(j + 1) * n]);
            h[i * k + j] = v;
            h[j * k + i] = v;
        }
    }
    let mean = (0..k).map(|i| h[i * k + i]).sum::<f32>() / k as f32;
    for i in 0..k {
        h[i * k + i] += DAMP * mean;
    }
    chol_of_inverse(k, &mut h)
}

/// Run the calibration sequences through the bf16 model and keep every routed
/// expert's input, per MoE layer, in sequence order — one thread per sequence.
fn calibrate(ck: &Checkpoint, seqs: &[Vec<u32>]) -> Vec<Calib> {
    use common::trace::NoTrace;
    use deepseek_v2_lite::{Bf16Experts, CpuAttention, CpuFfn, CpuModel, Routed, State, Weights};
    struct Keep<R> {
        inner: R,
        x: Vec<Vec<f32>>,
    }
    impl<R: Routed> Routed for Keep<R> {
        fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
            self.x[l - FIRST_K_DENSE_REPLACE].extend_from_slice(x);
            self.inner.run(l, idx, x, out)
        }
    }
    let per_seq: Vec<Vec<Vec<f32>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = seqs
            .iter()
            .map(|seq| {
                sc.spawn(move || {
                    let w = Weights::new(|name| ck.bf16(name));
                    let experts = Bf16Experts::new(|l, e, which| ck.bf16(&format!("model.layers.{l}.mlp.experts.{e}.{which}.weight")));
                    let ffn = CpuFfn::new(Keep { inner: experts, x: vec![Vec::new(); NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE] });
                    let mut model = CpuModel { attention: CpuAttention::new(seq.len()), ffn };
                    let mut st = State::new(seq.len());
                    for (pos, &tok) in seq.iter().enumerate() {
                        st.decode(&w, tok, pos, &mut model, &Sampling::GREEDY, 0.0, &mut NoTrace);
                    }
                    model.ffn.routed.x
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let layers = NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE;
    let xs: Vec<Vec<f32>> = (0..layers).map(|i| per_seq.iter().flat_map(|p| p[i].iter().copied()).collect()).collect();
    std::thread::scope(|sc| {
        let hs: Vec<_> = xs.into_iter().map(|x| sc.spawn(move || Calib { u: hessian_factor(D_MODEL, &x), x })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// GPTQ for one expert: gate/up against the layer's factor, down against
/// the factor of its own hidden states (`swiglu(x·[gate;up]ᵀ)` on every
/// calibration row, through the same gemv the forward uses).
fn quantize_calibrated(ck: &Checkpoint, l: usize, e: usize, cal: &Calib, gate_up: &mut [f32], down: &mut [f32]) -> ((Vec<i8>, Vec<f32>), (Vec<i8>, Vec<f32>)) {
    let p = format!("model.layers.{l}.mlp.experts.{e}.");
    let (gate, up) = (ck.bf16(&format!("{p}gate_proj.weight")), ck.bf16(&format!("{p}up_proj.weight")));
    let n = cal.x.len() / D_MODEL;
    let mut hid = vec![0f32; n * D_FF];
    let mut gu = vec![0f32; 2 * D_FF];
    for r in 0..n {
        let x = &cal.x[r * D_MODEL..(r + 1) * D_MODEL];
        common::gemv::bf16::<D_FF, D_MODEL>(gate, x, &mut gu[..D_FF]);
        common::gemv::bf16::<D_FF, D_MODEL>(up, x, &mut gu[D_FF..]);
        common::swiglu::swiglu::<D_FF>(&gu, &mut hid[r * D_FF..(r + 1) * D_FF]);
    }
    let ud = hessian_factor(D_FF, &hid);
    (q4::gptq(2 * D_FF, D_MODEL, gate_up, &cal.u), q4::gptq(D_MODEL, D_FF, down, &ud))
}

/// One expert → one record: gate rows then up rows fused, then down.
fn pack_expert(ck: &Checkpoint, l: usize, e: usize, cal: Option<&Calib>) -> Vec<u8> {
    let p = format!("model.layers.{l}.mlp.experts.{e}.");
    let mut gate_up = ck.f32s(&format!("{p}gate_proj.weight"));
    gate_up.extend_from_slice(&ck.f32s(&format!("{p}up_proj.weight")));
    let mut down = ck.f32s(&format!("{p}down_proj.weight"));
    assert_eq!(gate_up.len(), 2 * D_FF * D_MODEL);
    assert_eq!(down.len(), D_MODEL * D_FF);
    let ((gq, gs), (dq, ds)) = match cal {
        Some(cal) => quantize_calibrated(ck, l, e, cal, &mut gate_up, &mut down),
        None => (quantize(2 * D_FF, D_MODEL, &gate_up), quantize(D_MODEL, D_FF, &down)),
    };
    let (guw, gus) = q4::pack(2 * D_FF, D_MODEL, &gq, &gs);
    let (dw, dsc) = q4::pack(D_MODEL, D_FF, &dq, &ds);
    let mut rec = Vec::with_capacity(RECORD);
    rec.extend_from_slice(&guw);
    rec.extend(gus.iter().flat_map(|v| v.to_le_bytes()));
    rec.extend_from_slice(&dw);
    rec.extend(dsc.iter().flat_map(|v| v.to_le_bytes()));
    assert_eq!(rec.len(), RECORD);
    rec
}

/// Every non-expert tensor, bf16, at its `layout::dense` offset.
fn write_dense(ck: &Checkpoint, path: &std::path::Path) {
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
    file.set_len(d::BYTES as u64).unwrap();
    let mut mm = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
    let out: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(mm.as_mut_ptr() as *mut u16, d::TOTAL) };
    let mut put = |off: usize, name: &str| {
        let t = ck.bf16(name);
        out[off..off + t.len()].copy_from_slice(t);
    };
    put(d::EMBED, "model.embed_tokens.weight");
    put(d::NORM, "model.norm.weight");
    put(d::LM_HEAD, "lm_head.weight");
    for l in 0..NUM_HIDDEN_LAYERS {
        let (b, p) = (d::layer(l), format!("model.layers.{l}."));
        put(b + d::INPUT_LN, &format!("{p}input_layernorm.weight"));
        put(b + d::POST_LN, &format!("{p}post_attention_layernorm.weight"));
        put(b + d::Q_PROJ, &format!("{p}self_attn.q_proj.weight"));
        put(b + d::KV_A, &format!("{p}self_attn.kv_a_proj_with_mqa.weight"));
        put(b + d::KV_A_LN, &format!("{p}self_attn.kv_a_layernorm.weight"));
        put(b + d::KV_B, &format!("{p}self_attn.kv_b_proj.weight"));
        put(b + d::O_PROJ, &format!("{p}self_attn.o_proj.weight"));
        if l < FIRST_K_DENSE_REPLACE {
            put(b + d::DENSE_GATE, &format!("{p}mlp.gate_proj.weight"));
            put(b + d::DENSE_UP, &format!("{p}mlp.up_proj.weight"));
            put(b + d::DENSE_DOWN, &format!("{p}mlp.down_proj.weight"));
        } else {
            put(b + d::ROUTER, &format!("{p}mlp.gate.weight"));
            put(b + d::SHARED_GATE, &format!("{p}mlp.shared_experts.gate_proj.weight"));
            put(b + d::SHARED_UP, &format!("{p}mlp.shared_experts.up_proj.weight"));
            put(b + d::SHARED_DOWN, &format!("{p}mlp.shared_experts.down_proj.weight"));
        }
    }
    mm.flush().unwrap();
}

/// The sink token's state: BOS through the bf16 model once, at full precision;
/// its cache row for every layer is what every sequence starts from.
fn write_sink(ck: &Checkpoint, path: &std::path::Path) {
    use common::trace::NoTrace;
    use deepseek_v2_lite::{Bf16Experts, CpuAttention, CpuFfn, CpuModel, State, Weights, BOS_TOKEN};
    let w = Weights::new(|name| ck.bf16(name));
    let ffn = CpuFfn::new(Bf16Experts::new(|l, e, which| ck.bf16(&format!("model.layers.{l}.mlp.experts.{e}.{which}.weight"))));
    let mut model = CpuModel { attention: CpuAttention::new(2), ffn };
    let mut st = State::new(2);
    st.decode(&w, BOS_TOKEN as u32, 0, &mut model, &Sampling::GREEDY, 0.0, &mut NoTrace);
    let mut sink = Vec::with_capacity(deepseek_v2_lite::layout::SINK_LEN);
    for c in &model.attention.cache {
        sink.extend_from_slice(&c[..KV_ROW]);
    }
    fs::write(path, unsafe { std::slice::from_raw_parts(sink.as_ptr() as *const u8, 4 * sink.len()) }).unwrap();
}

/// Token bytes by id from tokenizer.json: the BPE vocab is byte-level (the
/// GPT-2 byte→unicode alphabet), the added tokens are literal strings, and
/// ids past both are empty.
fn write_vocab(snapshot: &std::path::Path, path: &std::path::Path) {
    let t: serde_json::Value = serde_json::from_slice(&fs::read(snapshot.join("tokenizer.json")).unwrap()).unwrap();
    // GPT-2 bytes_to_unicode, inverted
    let mut bs: Vec<u32> = (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF).map(|b| b as u32).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut uni2byte = std::collections::HashMap::new();
    for (b, c) in bs.iter().zip(&cs) {
        uni2byte.insert(char::from_u32(*c).unwrap(), *b as u8);
    }
    let mut toks: Vec<Vec<u8>> = vec![Vec::new(); VOCAB_SIZE];
    for (s, id) in t["model"]["vocab"].as_object().unwrap() {
        toks[id.as_u64().unwrap() as usize] = s.chars().map(|c| uni2byte[&c]).collect();
    }
    for a in t["added_tokens"].as_array().unwrap() {
        toks[a["id"].as_u64().unwrap() as usize] = a["content"].as_str().unwrap().as_bytes().to_vec();
    }
    let mut offs: Vec<u32> = Vec::with_capacity(VOCAB_SIZE + 1);
    let mut bytes = Vec::new();
    for tk in &toks {
        offs.push(bytes.len() as u32);
        bytes.extend_from_slice(tk);
    }
    offs.push(bytes.len() as u32);
    let mut f = fs::File::create(path).unwrap();
    f.write_all(unsafe { std::slice::from_raw_parts(offs.as_ptr() as *const u8, offs.len() * 4) }).unwrap();
    f.write_all(&bytes).unwrap();
}

fn main() {
    let snapshot = arg("--snapshot");
    let out = arg("--out");
    fs::create_dir_all(&out).unwrap();
    let ck = Checkpoint::open(&snapshot);
    let layers: Vec<usize> = (FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS).collect();
    let jobs: Vec<(usize, usize)> = layers.iter().flat_map(|&l| (0..N_ROUTED_EXPERTS).map(move |e| (l, e))).collect();
    let t0 = std::time::Instant::now();

    let calib_file = if flag("--rtn") { None } else { Some(harness::arg("--calib").unwrap_or_else(|| std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/deepseek-v2-lite/calib.ids")))) };
    let calib: Option<Vec<Calib>> = calib_file.as_ref().filter(|_| !flag("--dense-only")).map(|f| {
        let text = fs::read_to_string(f).unwrap();
        let seqs: Vec<Vec<u32>> = text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect()).collect();
        assert!(seqs.iter().all(|s| s.first() == Some(&(BOS_TOKEN as u32))), "every calibration sequence starts with BOS");
        let n: usize = seqs.iter().map(|s| s.len()).sum();
        let cal = calibrate(&ck, &seqs);
        eprintln!("calibrated on {} sequences, {n} positions, in {:.1}s", seqs.len(), t0.elapsed().as_secs_f32());
        cal
    });
    let recipe = if calib_file.is_some() { RECIPE_GPTQ } else { RECIPE_RTN };

    let path = out.join(layout::EXPERTS_FILE);
    if flag("--dense-only") {
        // reuse the packed experts already in <out>
        assert!(path.exists());
    } else {
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).unwrap();
    file.set_len((jobs.len() * RECORD) as u64).unwrap();
    let mut mm = unsafe { memmap2::MmapMut::map_mut(&file).unwrap() };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(if calib.is_some() { 64 } else { 32 });
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out_slices: Vec<&mut [u8]> = mm.chunks_exact_mut(RECORD).collect();
    let out_slices = std::sync::Mutex::new(out_slices.into_iter().map(Some).collect::<Vec<_>>());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (l, e) = jobs[i];
                let rec = pack_expert(&ck, l, e, calib.as_ref().map(|c| &c[l - FIRST_K_DENSE_REPLACE]));
                let dst = out_slices.lock().unwrap()[i].take().unwrap();
                dst.copy_from_slice(&rec);
            });
        }
    });
    mm.flush().unwrap();
    drop(mm);
    eprintln!("{} experts packed in {:.1}s", jobs.len(), t0.elapsed().as_secs_f32());
    }

    let t1 = std::time::Instant::now();
    if !flag("--experts-only") {
        write_dense(&ck, &out.join(layout::DENSE_FILE));
        write_vocab(&snapshot, &out.join(layout::VOCAB_FILE));
        write_sink(&ck, &out.join(layout::SINK_FILE));
        for f in layout::TOKENIZER_FILES {
            fs::copy(snapshot.join(f), out.join(f)).unwrap();
        }
        eprintln!("dense + vocab + sink + tokenizer written in {:.1}s", t1.elapsed().as_secs_f32());
    }
    if let Some(k) = harness::arg("--kernels") {
        fs::copy(&k, out.join(layout::KERNELS_FILE)).unwrap();
    }
    let sha_of = |name: &str| sha256(&fs::read(out.join(name)).unwrap());
    let sha_k = if out.join(layout::KERNELS_FILE).exists() { sha_of(layout::KERNELS_FILE) } else { String::new() };
    let sha = sha_of(layout::EXPERTS_FILE);
    let (sha_d, sha_v, sha_s) = (sha_of(layout::DENSE_FILE), sha_of(layout::VOCAB_FILE), sha_of(layout::SINK_FILE));
    let sha_t = layout::TOKENIZER_FILES.map(|f| format!("\"{f}\" = \"sha256:{}\"", sha_of(f))).join("\n");
    let calib_line = calib_file.as_ref().map(|f| format!("calibration = \"{}\" # sha256:{}\n", f.file_name().unwrap().to_str().unwrap(), sha256(&fs::read(&f).unwrap()))).unwrap_or_default();
    let mut m = fs::File::create(out.join("manifest.toml")).unwrap();
    let ck_toml = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/deepseek-v2-lite/checkpoint.toml")).unwrap_or_default();
    // [model]: the model's name and the constants the wire assumes, so the edge speaks it without reading the crate
    writeln!(m, "recipe = \"{recipe}\"\n{calib_line}record_bytes = {RECORD}\nexperts = {}\nlayers = [{}, {}]\n\n[model]\nname = \"deepseek-v2-lite\"\nbos = {BOS_TOKEN}\neos = {EOS_TOKEN}\nvocab_size = {VOCAB_SIZE}\n\n[files]\n\"experts.q4g128.bin\" = \"sha256:{sha}\"\n\"dense.bf16.bin\" = \"sha256:{sha_d}\"\n\"vocab.bin\" = \"sha256:{sha_v}\"\n\"kernels.gfx1100.elf\" = \"sha256:{sha_k}\"\n\"sink.f32\" = \"sha256:{sha_s}\"\n{sha_t}\n\n# checkpoint.toml\n{}", jobs.len(), FIRST_K_DENSE_REPLACE, NUM_HIDDEN_LAYERS, ck_toml.lines().map(|l| format!("# {l}")).collect::<Vec<_>>().join("\n")).unwrap();
    println!("{}", out.join("manifest.toml").display());
}

/// Plain SHA-256; a hundred lines would be a dependency, thirty is fine.
fn sha256(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g; g = f; f = e; e = d.wrapping_add(t1); d = c; c = b; b = a; a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}
