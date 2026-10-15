//! The GLM-5.3-Flash bundle: the dense weights exactly as the checkpoint
//! stores them (bf16, f32, FP8 + block scales; `layout::dense`), the routed
//! experts INT4 round-to-nearest (`q4g128`; GPTQ does not beat RTN on this
//! model — analysis.md §2), the prefix state after `[gMASK]<sop>` computed by
//! the f32 reference, the console vocabulary, the checkpoint's tokenizer and
//! chat template, and the manifest.
//!
//!   bundle --model glm-5.3-flash --snapshot <dir> --out <dir> [--experts-only | --dense-only] [--kernels <gfx1100 .elf>]
//!
//! `--dense-only` reuses the experts file already in <out>; `--experts-only` writes
//! nothing else. The experts take the time: 12,096 records of 13.4 MB, each
//! widened from FP8, rounded and packed on its own thread, written in place.

use crate::shared::{quantize, sha256, sha256_file, write_vocab};
use common::q4;
use glm_5_3_flash::layout::{self, Kind};
use glm_5_3_flash::*;
use harness::{arg_required as arg, flag, Checkpoint};
use std::fs;
use std::io::Write;
use std::os::unix::fs::FileExt;

fn ckname(l: usize, name: &str) -> String {
    format!("model.language_model.layers.{l}.{name}")
}

/// One expert → one record: gate rows then up rows fused, then down.
fn pack_expert(ck: &Checkpoint, l: usize, e: usize) -> Vec<u8> {
    let p = |s: &str| ck.f32s(&ckname(l, &format!("mlp.experts.{e}.{s}_proj.weight")));
    let mut gate_up = p("gate");
    gate_up.extend_from_slice(&p("up"));
    let down = p("down");
    assert_eq!(gate_up.len(), 2 * D_FF * H);
    assert_eq!(down.len(), H * D_FF);
    let (gq, gs) = quantize(2 * D_FF, H, &gate_up);
    let (dq, ds) = quantize(H, D_FF, &down);
    let (guw, gus) = q4::pack(2 * D_FF, H, &gq, &gs);
    let (dw, dsc) = q4::pack(H, D_FF, &dq, &ds);
    let mut rec = Vec::with_capacity(layout::RECORD);
    rec.extend_from_slice(&guw);
    rec.extend(gus.iter().flat_map(|v| v.to_le_bytes()));
    rec.extend_from_slice(&dw);
    rec.extend(dsc.iter().flat_map(|v| v.to_le_bytes()));
    assert_eq!(rec.len(), layout::RECORD);
    rec
}

fn write_experts(ck: &Checkpoint, path: &std::path::Path) {
    let jobs: Vec<(usize, usize)> = (FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS).flat_map(|l| (0..E).map(move |e| (l, e))).collect();
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
    file.set_len(layout::STREAM as u64).unwrap();
    let t0 = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= jobs.len() {
                    break;
                }
                let (l, e) = jobs[i];
                let rec = pack_expert(ck, l, e);
                file.write_all_at(&rec, layout::record(l, e).start as u64).unwrap();
                let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if d % 1000 == 0 || d == jobs.len() {
                    eprintln!("experts  : {d}/{} packed, {:.0}s", jobs.len(), t0.elapsed().as_secs_f32());
                }
            });
        }
    });
    file.sync_all().unwrap();
    eprintln!("{} experts packed in {:.1}s", jobs.len(), t0.elapsed().as_secs_f32());
}

/// Every dense tensor at its slot, bytes copied as stored; the kind and size are checked against the table.
fn write_dense(ck: &Checkpoint, path: &std::path::Path) {
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
    file.set_len(layout::DENSE_BYTES as u64).unwrap();
    let put = |off: usize, t: &layout::Tensor, name: &str| {
        let want = match t.kind {
            Kind::Bf16 => "BF16",
            Kind::F32 => "F32",
            Kind::Fp8 => "F8_E4M3",
        };
        assert_eq!(ck.kind(name), want, "{name}");
        let w = ck.raw(name);
        assert_eq!(w.len(), t.weight_bytes(), "{name}: {} bytes, the table says {}", w.len(), t.weight_bytes());
        file.write_all_at(w, off as u64).unwrap();
        if t.kind == Kind::Fp8 {
            let s = ck.raw(&format!("{name}_scale_inv"));
            assert_eq!(s.len(), 4 * t.scales(), "{name}_scale_inv");
            file.write_all_at(s, (off + w.len()) as u64).unwrap();
        }
    };
    for (i, t) in layout::HEAD.iter().enumerate() {
        put(layout::within(&layout::HEAD, i), t, t.name);
    }
    for l in 0..NUM_HIDDEN_LAYERS {
        let mut off = layout::layer(l);
        for s in [&layout::COMMON[..], layout::attn(l), layout::ffn(l)] {
            for (i, t) in s.iter().enumerate() {
                put(off + layout::within(s, i), t, &ckname(l, t.name));
            }
            off += layout::section(s);
        }
    }
    file.sync_all().unwrap();
}

/// The prefix's state: `[gMASK]<sop>` through the f32 reference, every layer;
/// the MLA layers' rows at both positions, then the KDA layers' states.
fn write_prefix(ck: &Checkpoint, path: &std::path::Path) {
    use glm_5_3_flash::reference::*;
    let get = |n: &str| ck.f32s(n);
    let n = FIRST_POS;
    let emb = ck.bf16("model.language_model.embed_tokens.weight");
    let mut streams = vec![0f32; n * HC * H];
    for (t, &id) in PREFIX_IDS.iter().enumerate() {
        let id = id as usize;
        let row: Vec<f32> = emb[id * H..(id + 1) * H].iter().map(|&v| f32::from_bits((v as u32) << 16)).collect();
        for j in 0..HC {
            streams[(t * HC + j) * H..(t * HC + j + 1) * H].copy_from_slice(&row);
        }
    }
    let mut rows = vec![0f32; n * layout::POSITION];
    let mut snap: Vec<f32> = Vec::with_capacity(layout::SNAPSHOT);
    let mut mla_i = 0;
    let t0 = std::time::Instant::now();
    for l in 0..NUM_HIDDEN_LAYERS {
        let layer = load(l, &get);
        let mut st = layer.state();
        layer.forward(&mut streams, n, &mut st, &mut |_, _| {}, &|e| load_expert(l, e, &get), None);
        match st {
            LayerState::Kda(k) => {
                assert_eq!(k.window.len(), layout::KDA_WINDOW);
                assert_eq!(k.s.len(), layout::KDA_MEMORY);
                snap.extend_from_slice(&k.window);
                snap.extend_from_slice(&k.s);
            }
            LayerState::Mla(m) => {
                for p in 0..n {
                    let r = &mut rows[p * layout::POSITION + mla_i * layout::MLA_ROW..p * layout::POSITION + (mla_i + 1) * layout::MLA_ROW];
                    r[..KV_LORA_RANK].copy_from_slice(&m.latent[p * KV_LORA_RANK..(p + 1) * KV_LORA_RANK]);
                    r[KV_LORA_RANK..KV_LORA_RANK + INDEX_HEAD_DIM].copy_from_slice(&m.ikey[p * INDEX_HEAD_DIM..(p + 1) * INDEX_HEAD_DIM]);
                    r[KV_LORA_RANK + INDEX_HEAD_DIM..].copy_from_slice(&m.igate[p * INDEX_HEAD_DIM..(p + 1) * INDEX_HEAD_DIM]);
                }
                mla_i += 1;
            }
        }
        eprintln!("prefix   : layer {l} done, {:.0}s", t0.elapsed().as_secs_f32());
    }
    assert_eq!(snap.len(), layout::SNAPSHOT);
    rows.extend_from_slice(&snap);
    assert_eq!(rows.len(), layout::PREFIX_LEN);
    fs::write(path, unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, 4 * rows.len()) }).unwrap();
}

pub fn build() {
    let snapshot = arg("--snapshot");
    let out = arg("--out");
    fs::create_dir_all(&out).unwrap();
    // a bundle is a function of the checkpoint and the recipe: reproducible, so backups skip it
    fs::write(out.join("CACHEDIR.TAG"), "Signature: 8a477f597d28d172789f06886806bc55\n# rockem bundle: rebuilt from the checkpoint by `bundle --model glm-5.3-flash`\n").unwrap();
    let ck = Checkpoint::open(&snapshot);
    let t0 = std::time::Instant::now();
    let experts = out.join(layout::EXPERTS_FILE);
    if flag("--dense-only") {
        assert!(experts.exists(), "--dense-only: no {}", experts.display());
    } else {
        write_experts(&ck, &experts);
    }
    if !flag("--experts-only") {
        let t1 = std::time::Instant::now();
        write_dense(&ck, &out.join(layout::DENSE_FILE));
        eprintln!("dense    : {:.2} GB copied in {:.1}s", layout::DENSE_BYTES as f64 / 1e9, t1.elapsed().as_secs_f32());
        write_vocab(&snapshot, &out.join(layout::VOCAB_FILE), VOCAB_SIZE);
        write_prefix(&ck, &out.join(layout::PREFIX_FILE));
        for f in layout::TOKENIZER_FILES {
            fs::copy(snapshot.join(f), out.join(f)).unwrap();
        }
        fs::copy(snapshot.join(layout::CHAT_TEMPLATE_FILE), out.join(layout::CHAT_TEMPLATE_FILE)).unwrap();
        eprintln!("dense + vocab + prefix + tokenizer written in {:.1}s", t1.elapsed().as_secs_f32());
    }
    if let Some(k) = harness::arg("--kernels") {
        fs::copy(&k, out.join(layout::KERNELS_FILE)).unwrap();
    }
    let t2 = std::time::Instant::now();
    let mut files: Vec<&str> = vec![layout::EXPERTS_FILE, layout::DENSE_FILE, layout::VOCAB_FILE, layout::PREFIX_FILE];
    files.extend(layout::TOKENIZER_FILES);
    files.push(layout::CHAT_TEMPLATE_FILE);
    if out.join(layout::KERNELS_FILE).exists() {
        files.push(layout::KERNELS_FILE);
    }
    let shas: Vec<String> = files.iter().map(|f| format!("\"{f}\" = \"sha256:{}\"", sha256_file(&out.join(f)))).collect();
    eprintln!("hashed {} files in {:.0}s", files.len(), t2.elapsed().as_secs_f32());
    let ck_toml = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/glm-5.3-flash/checkpoint.toml")).unwrap_or_default();
    let mut m = fs::File::create(out.join("manifest.toml")).unwrap();
    // [model]: the model's name and the constants the wire assumes, so the edge speaks it without reading
    // the crate; `bos` is the prefix's first id and `eos` its first end id, with the full sets beside them
    writeln!(
        m,
        "recipe = \"{}\"\nrecord_bytes = {}\nexperts = {}\nlayers = [{}, {}]\n\n[model]\nname = \"glm-5.3-flash\"\nbos = {}\neos = {}\nprefix = [{}]\neos_tokens = [{}]\nvocab_size = {VOCAB_SIZE}\n\n[files]\n{}\n\n# checkpoint.toml\n{}",
        layout::RECIPE_RTN,
        layout::RECORD,
        layout::MOE_LAYERS * E,
        FIRST_K_DENSE_REPLACE,
        NUM_HIDDEN_LAYERS,
        PREFIX_IDS[0],
        EOS_TOKENS[0],
        PREFIX_IDS.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        EOS_TOKENS.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        shas.join("\n"),
        ck_toml.lines().map(|l| format!("# {l}")).collect::<Vec<_>>().join("\n")
    )
    .unwrap();
    let _ = sha256; // the one-shot form is the V2-Lite builder's
    eprintln!("bundle written in {:.0}s", t0.elapsed().as_secs_f32());
    println!("{}", out.join("manifest.toml").display());
}
