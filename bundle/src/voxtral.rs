//! The Voxtral-Small-24B bundle (docs/queue/094): the decoder's matrices INT4 round-to-nearest as the
//! `q4g128` stream `layout` describes (every layer's fused q|k|v, o, gate|up and down, then the lm_head, in
//! page runs), the token embedding, the norms, the audio encoder and the projector exactly as the
//! checkpoint stores them (bf16), the console vocabulary from the converted tokenizer, the edge's files
//! (`tokenizer.json`, `mel_filters.f32`; bringup/voxtral/edge_files.py makes them from `tekken.json`),
//! and the manifest with the `[model.audio]` table the edge and node agree on.
//!
//!   bundle --model voxtral-small-24b --snapshot <HF dir> --edge <edge files dir> --out <bundle dir> [--dense-only | --stream-only]
//!
//! The model's files go under `<out>/voxtral-small-24b/`; the manifest at `<out>` lists it (a bundle that
//! already carries another model keeps its entries: this builder appends its own).

use crate::shared::{quantize, sha256_file, write_vocab};
use common::q4::{self, GROUP, TILE};
use harness::{arg_required as arg, flag, Checkpoint};
use std::fs;
use std::io::Write;
use std::os::unix::fs::FileExt;
use voxtral_small_24b::layout::{self, Mat, Which};
use voxtral_small_24b::*;

pub const NAME: &str = "voxtral-small-24b";

fn lname(l: usize, name: &str) -> String {
    format!("language_model.model.layers.{l}.{name}")
}

/// A packed `[rows][k]` q4 matrix cut into `m`'s items: item `c` is rows `c · chunk ..` and their scales (for
/// `Gu`, gate rows then up rows, gate scales then up scales — `half` is the gate's row count).
fn items(m: &Mat, w: &[u8], s: &[f32], half: Option<usize>) -> Vec<Vec<u8>> {
    let k = m.k;
    assert_eq!(w.len(), m.rows * k / 2);
    assert_eq!(s.len(), m.rows * (k / GROUP));
    let le = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let rows = |r0: usize, n: usize| &w[r0 / TILE * k / 2 * TILE..(r0 + n) / TILE * k / 2 * TILE];
    let scales = |r0: usize, n: usize| &s[r0 / TILE * (k / GROUP) * TILE..(r0 + n) / TILE * (k / GROUP) * TILE];
    (0..m.items())
        .map(|c| {
            let mut it = Vec::with_capacity(m.item());
            match half {
                Some(h) => {
                    let n = m.chunk / 2;
                    it.extend_from_slice(rows(c * n, n));
                    it.extend_from_slice(rows(h + c * n, n));
                    it.extend_from_slice(&le(scales(c * n, n)));
                    it.extend_from_slice(&le(scales(h + c * n, n)));
                }
                None => {
                    it.extend_from_slice(rows(c * m.chunk, m.chunk));
                    it.extend_from_slice(&le(scales(c * m.chunk, m.chunk)));
                }
            }
            assert_eq!(it.len(), m.item());
            it
        })
        .collect()
}

/// A matrix widened, rounded to nearest INT4 per group of 128, packed.
fn q4_of(rows: usize, k: usize, w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(w.len(), rows * k);
    let (q, s) = quantize(rows, k, w);
    q4::pack(rows, k, &q, &s)
}

/// Layer `l`'s four matrices into the stream, at `layout::item`'s offsets.
fn write_layer(ck: &Checkpoint, file: &fs::File, l: usize) {
    let p = |n: &str| ck.f32s(&lname(l, n));
    let put = |w: Which, its: Vec<Vec<u8>>| {
        for (c, it) in its.into_iter().enumerate() {
            file.write_all_at(&it, layout::item(l, w, c).start as u64).unwrap();
        }
    };
    let mut qkv = p("self_attn.q_proj.weight");
    qkv.extend_from_slice(&p("self_attn.k_proj.weight"));
    qkv.extend_from_slice(&p("self_attn.v_proj.weight"));
    let (w, s) = q4_of(layout::QKV.rows, H, &qkv);
    drop(qkv);
    put(Which::Qkv, items(&layout::QKV, &w, &s, None));
    let (w, s) = q4_of(layout::O.rows, Q_DIM, &p("self_attn.o_proj.weight"));
    put(Which::O, items(&layout::O, &w, &s, None));
    let mut gu = p("mlp.gate_proj.weight");
    gu.extend_from_slice(&p("mlp.up_proj.weight"));
    let (w, s) = q4_of(layout::GU.rows, H, &gu);
    drop(gu);
    put(Which::Gu, items(&layout::GU, &w, &s, Some(D_FF)));
    let (w, s) = q4_of(layout::DOWN.rows, D_FF, &p("mlp.down_proj.weight"));
    put(Which::Down, items(&layout::DOWN, &w, &s, None));
}

fn write_stream(ck: &Checkpoint, path: &std::path::Path) {
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
    file.set_len(layout::STREAM as u64).unwrap();
    let t0 = std::time::Instant::now();
    // a layer's f32 matrices are ~2.6 GB at once: a few layers in flight
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(12);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let l = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if l >= NUM_HIDDEN_LAYERS {
                    break;
                }
                write_layer(ck, &file, l);
                eprintln!("stream   : layer {l} packed, {:.0}s", t0.elapsed().as_secs_f32());
            });
        }
    });
    let (w, s) = q4_of(layout::LM_HEAD.rows, H, &ck.f32s("language_model.lm_head.weight"));
    for (c, it) in items(&layout::LM_HEAD, &w, &s, None).into_iter().enumerate() {
        file.write_all_at(&it, layout::head_item(c).start as u64).unwrap();
    }
    file.sync_all().unwrap();
    eprintln!("stream   : {} layers and the lm_head in {:.0}s", NUM_HIDDEN_LAYERS, t0.elapsed().as_secs_f32());
}

/// Every dense tensor at its slot, bytes copied as stored; kind and size checked against the table.
fn write_dense(ck: &Checkpoint, path: &std::path::Path) {
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path).unwrap();
    file.set_len(layout::DENSE_BYTES as u64).unwrap();
    let put = |off: usize, t: &layout::Tensor, name: &str| {
        assert_eq!(ck.kind(name), if t.kind == layout::Kind::F32 { "F32" } else { "BF16" }, "{name}");
        let w = ck.raw(name);
        assert_eq!(w.len(), t.bytes(), "{name}: {} bytes, the table says {} ({:?})", w.len(), t.bytes(), ck.shape(name));
        file.write_all_at(w, off as u64).unwrap();
    };
    for (i, t) in layout::HEAD.iter().enumerate() {
        put(layout::DENSE_HEAD + layout::within(&layout::HEAD, i), t, t.name);
    }
    for l in 0..NUM_HIDDEN_LAYERS {
        for (i, t) in layout::LAYER_NORMS.iter().enumerate() {
            put(layout::layer_norms(l) + layout::within(&layout::LAYER_NORMS, i), t, &lname(l, t.name));
        }
    }
    for (i, t) in layout::ENC_HEAD.iter().enumerate() {
        put(layout::DENSE_ENC_HEAD + layout::within(&layout::ENC_HEAD, i), t, &format!("audio_tower.{}", t.name));
    }
    for l in 0..ENC_NUM_HIDDEN_LAYERS {
        for (i, t) in layout::ENC_LAYER.iter().enumerate() {
            put(layout::enc_layer(l) + layout::within(&layout::ENC_LAYER, i), t, &format!("audio_tower.layers.{l}.{}", t.name));
        }
    }
    for (i, t) in layout::PROJECTOR.iter().enumerate() {
        put(layout::DENSE_PROJECTOR + layout::within(&layout::PROJECTOR, i), t, &format!("multi_modal_projector.{}", t.name));
    }
    file.sync_all().unwrap();
}

pub fn build() {
    let t0 = std::time::Instant::now();
    let snapshot = arg("--snapshot");
    let edge = arg("--edge");
    let root = arg("--out");
    let out = root.join(NAME);
    fs::create_dir_all(&out).unwrap();
    if !root.join("CACHEDIR.TAG").exists() {
        fs::write(root.join("CACHEDIR.TAG"), "Signature: 8a477f597d28d172789f06886806bc55\n# rockem bundle: rebuilt from the checkpoint by `bundle`\n").unwrap();
    }
    let ck = Checkpoint::open(&snapshot);
    if !flag("--dense-only") {
        write_stream(&ck, &out.join(layout::STREAM_FILE));
    }
    if !flag("--stream-only") {
        write_dense(&ck, &out.join(layout::DENSE_FILE));
        write_vocab(&edge, &out.join(layout::VOCAB_FILE), VOCAB_SIZE);
        for f in [layout::TOKENIZER_FILE, layout::MEL_FILE] {
            fs::copy(edge.join(f), out.join(f)).unwrap();
        }
        assert_eq!(fs::metadata(out.join(layout::MEL_FILE)).unwrap().len() as usize, layout::MEL_BYTES, "mel_filters.f32");
    }
    let files = [layout::STREAM_FILE, layout::DENSE_FILE, layout::VOCAB_FILE, layout::TOKENIZER_FILE, layout::MEL_FILE];
    let t2 = std::time::Instant::now();
    let shas: Vec<String> = files.iter().map(|f| format!("\"{NAME}/{f}\" = \"sha256:{}\"", sha256_file(&out.join(f)))).collect();
    eprintln!("hashed {} files in {:.0}s", files.len(), t2.elapsed().as_secs_f32());
    let ck_toml = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/voxtral-small-24b/checkpoint.toml")).unwrap_or_default();
    // the entry: name, recipe, the wire's constants, the encoder's input the edge computes; appended to a manifest
    // that already lists other models (their [files] kept), written fresh otherwise
    let entry = format!(
        "[[model]]\nname = \"{NAME}\"\nrecipe = \"{}\"\nbos = {BOS_TOKEN}\neos = {EOS_TOKEN}\nprefix = [{BOS_TOKEN}]\neos_tokens = [{EOS_TOKEN}]\nvocab_size = {VOCAB_SIZE}\n\n[model.audio]\nplaceholder = {AUDIO_ID}\nwindow = [{NUM_MEL_BINS}, {MEL_FRAMES}]\nrows = {ROWS_PER_WINDOW}\nsample_rate = 16000\nhop = 160\nfft_window = {}\n",
        layout::RECIPE,
        layout::N_FFT
    );
    let manifest = root.join("manifest.toml");
    let text = if manifest.exists() {
        let old = fs::read_to_string(&manifest).unwrap();
        assert!(!old.contains(&format!("name = \"{NAME}\"")), "{}: already lists {NAME}", manifest.display());
        let (models, rest) = old.split_once("[files]").expect("[files]");
        let (files_old, comment) = rest.split_once("# checkpoint.toml").map_or((rest.to_string(), String::new()), |(f, c)| (f.to_string(), format!("# checkpoint.toml{c}")));
        format!("{}{entry}\n[files]{}{}\n\n# checkpoint.toml ({NAME})\n{}\n{comment}", models.trim_end().to_string() + "\n\n", files_old.trim_end_matches('\n'), format!("\n{}", shas.join("\n")), ck_toml.lines().map(|l| format!("# {l}")).collect::<Vec<_>>().join("\n"))
    } else {
        format!("{entry}\n[files]\n{}\n\n# checkpoint.toml\n{}\n", shas.join("\n"), ck_toml.lines().map(|l| format!("# {l}")).collect::<Vec<_>>().join("\n"))
    };
    let mut m = fs::File::create(&manifest).unwrap();
    m.write_all(text.as_bytes()).unwrap();
    eprintln!("bundle written in {:.0}s", t0.elapsed().as_secs_f32());
    println!("{}", manifest.display());
}
