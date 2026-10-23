//! The GLM-5.3-Flash bundle: the dense weights exactly as the checkpoint
//! stores them (bf16, f32, FP8 + block scales; `layout::dense`), the routed
//! experts INT4 round-to-nearest (`q4g128`; GPTQ does not beat RTN on this
//! model — analysis.md §2), the prefix state after `[gMASK]<sop>` computed by
//! the f32 reference, the console vocabulary, the checkpoint's tokenizer and
//! chat template, and the manifest.
//!
//!   bundle --model glm-5.3-flash --snapshot <dir> --out <dir> [--experts-only [--only-layer L] | --dense-only] [--kernels <gfx1100 .elf>]
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

/// One expert → its items in the stream (offset, bytes): gate rows then up rows fused per item, then down.
fn pack_expert(ck: &Checkpoint, l: usize, e: usize) -> Vec<(usize, Vec<u8>)> {
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
    glm_5_3_flash::expert::pack_items(l, e, &guw, &gus, &dw, &dsc)
}

/// The stream's earlier formats, for `--relayout`: `legacy` is the four packed matrices back to
/// back per expert (13,369,344 bytes, until 2026-10-16); `v1` is the first node-run layout, one
/// expert per 16 MiB record of four 4 MiB runs (the morning of 2026-10-16, never shipped).
mod legacy {
    use super::*;
    use common::q4::GROUP;
    pub const GATE_UP_W: usize = 2 * D_FF * H / 2;
    pub const GATE_UP_S: usize = 2 * D_FF * (H / GROUP);
    pub const DOWN_W: usize = H * D_FF / 2;
    pub const DOWN_S: usize = H * (D_FF / GROUP);
    pub const RECORD: usize = GATE_UP_W + 4 * GATE_UP_S + DOWN_W + 4 * DOWN_S;
    /// The four matrices of a legacy record.
    pub fn split(old: &[u8]) -> (&[u8], Vec<f32>, &[u8], Vec<f32>) {
        assert_eq!(old.len(), RECORD);
        let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        let (guw, rest) = old.split_at(GATE_UP_W);
        let (gus, rest) = rest.split_at(4 * GATE_UP_S);
        let (dw, dsc) = rest.split_at(DOWN_W);
        (guw, f32s(gus), dw, f32s(dsc))
    }
}
mod v1 {
    use super::*;
    use glm_5_3_flash::layout::{DOWN_ITEM, DOWN_ITEMS, GU_ITEM, GU_ITEMS, GU_PER_RUN};
    pub const RUN: usize = 4 << 20;
    pub const RECORD: usize = 4 * RUN;
    pub const DOWN_PER_RUN: usize = DOWN_ITEMS / 4;
    pub fn gu_item(c: usize) -> std::ops::Range<usize> {
        let s = (c / GU_PER_RUN) * RUN + (c % GU_PER_RUN) * GU_ITEM;
        s..s + GU_ITEM
    }
    pub fn down_item(c: usize) -> std::ops::Range<usize> {
        let s = (c / DOWN_PER_RUN) * RUN + GU_PER_RUN * GU_ITEM + (c % DOWN_PER_RUN) * DOWN_ITEM;
        s..s + DOWN_ITEM
    }
    /// A v1 record's items as the current layout's (offset, bytes) for expert (l, e): the item bytes are identical.
    pub fn items(rec: &[u8], l: usize, e: usize) -> Vec<(usize, Vec<u8>)> {
        assert_eq!(rec.len(), RECORD);
        let mut out = Vec::with_capacity(GU_ITEMS + DOWN_ITEMS);
        for c in 0..GU_ITEMS {
            out.push((layout::gu_item(l, e, c).start, rec[gu_item(c)].to_vec()));
        }
        for c in 0..DOWN_ITEMS {
            out.push((layout::down_item(l, e, c).start, rec[down_item(c)].to_vec()));
        }
        out
    }
}

/// One block held in memory, addressed as the stream addresses it: the sampled dequantized weights
/// of expert (l, e) against its legacy record's.
fn check_block(blk: &[u8], base: usize, l: usize, e: usize, old: &[u8]) {
    use common::q4::GROUP;
    use glm_5_3_flash::layout::{DOWN_CHUNK, DOWN_ITEMS, DOWN_W, GU_CHUNK, GU_ITEMS, GU_S, GU_W};
    let (guw, gus, dw, dsc) = legacy::split(old);
    let f32s = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let mut checked = 0usize;
    for c in 0..GU_ITEMS {
        let r = layout::gu_item(l, e, c);
        let it = &blk[r.start - base..r.end - base];
        let (gw, uw) = (&it[..GU_W], &it[GU_W..2 * GU_W]);
        let (gs, us) = (f32s(&it[2 * GU_W..2 * GU_W + 4 * GU_S]), f32s(&it[2 * GU_W + 4 * GU_S..]));
        for (r_base, iw, is) in [(c * GU_CHUNK, gw, &gs), (D_FF + c * GU_CHUNK, uw, &us)] {
            for rr in 0..GU_CHUNK {
                for kk in (0..H).step_by(37) {
                    let a = q4::unpack(H, guw, r_base + rr, kk) as f32 * gus[q4::scale_index(H, r_base + rr, kk / GROUP)];
                    let b = q4::unpack(H, iw, rr, kk) as f32 * is[q4::scale_index(H, rr, kk / GROUP)];
                    assert!(a == b, "gate/up item {c} row {rr} col {kk}: {a} vs {b}");
                    checked += 1;
                }
            }
        }
    }
    for c in 0..DOWN_ITEMS {
        let r = layout::down_item(l, e, c);
        let it = &blk[r.start - base..r.end - base];
        let (iw, is) = (&it[..DOWN_W], f32s(&it[DOWN_W..]));
        for rr in 0..DOWN_CHUNK {
            for kk in (0..D_FF).step_by(37) {
                let a = q4::unpack(D_FF, dw, c * DOWN_CHUNK + rr, kk) as f32 * dsc[q4::scale_index(D_FF, c * DOWN_CHUNK + rr, kk / GROUP)];
                let b = q4::unpack(D_FF, iw, rr, kk) as f32 * is[q4::scale_index(D_FF, rr, kk / GROUP)];
                assert!(a == b, "down item {c} row {rr} col {kk}: {a} vs {b}");
                checked += 1;
            }
        }
    }
    eprintln!("relayout : expert ({l}, {e}) checks out against its legacy record ({checked} weights compared)");
}

/// `bundle --model glm-5.3-flash --relayout <experts file> [--from legacy|v1]`: the experts stream
/// into the current block layout, in place, a block of `XBLOCK` experts at a time in stream order.
/// From `v1` (16 MiB records) every block's new place starts at or before its sources, over bytes
/// already consumed, and the file shrinks at the end. From `legacy` (13,369,344-byte records,
/// the same total size as the current layout) the sources of block `b` are exactly the bytes
/// block `b` will occupy, so each block is read whole before it is written. A side file holds
/// the block in flight and `.done` the blocks placed: an interruption loses nothing.
pub fn relayout(path: &std::path::Path, from: &str) {
    use glm_5_3_flash::layout::{XBLOCK, XBLOCK_BYTES};
    use std::io::{Read, Seek, SeekFrom, Write};
    let src_record = match from {
        "legacy" => legacy::RECORD,
        "v1" => v1::RECORD,
        o => panic!("--from {o}: legacy or v1"),
    };
    let experts = layout::MOE_LAYERS * E;
    let blocks = experts / XBLOCK;
    let mut f = fs::OpenOptions::new().read(true).write(true).open(path).unwrap();
    let len = f.metadata().unwrap().len() as usize;
    let side = path.with_extension("bin.relayout");
    let done_path = path.with_extension("bin.relayout.done");
    let mut done: usize = fs::read_to_string(&done_path).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    if done == 0 {
        assert_eq!(len, experts * src_record, "{}: not a {from} stream ({len} bytes)", path.display());
    } else if let Ok(blk) = fs::read(&side) {
        if blk.len() == XBLOCK_BYTES {
            f.seek(SeekFrom::Start((done * XBLOCK_BYTES) as u64)).unwrap();
            f.write_all(&blk).unwrap();
            done += 1;
            fs::write(&done_path, format!("{done}\n")).unwrap();
        }
    }
    let t0 = std::time::Instant::now();
    let mut src = vec![0u8; src_record];
    let mut blk = vec![0u8; XBLOCK_BYTES];
    let per_layer = E / XBLOCK;
    while done < blocks {
        let b = done;
        let (l, e0) = (b / per_layer + FIRST_K_DENSE_REPLACE, (b % per_layer) * XBLOCK);
        let base = b * XBLOCK_BYTES;
        assert_eq!(base, layout::share(l, e0, 0));
        let mut first_old: Option<Vec<u8>> = None;
        for j in 0..XBLOCK {
            let e = e0 + j;
            f.seek(SeekFrom::Start(((b * XBLOCK + j) * src_record) as u64)).unwrap();
            f.read_exact(&mut src).unwrap();
            let items = match from {
                "legacy" => {
                    if j == 0 && b == 0 {
                        first_old = Some(src.clone());
                    }
                    let (guw, gus, dw, dsc) = legacy::split(&src);
                    glm_5_3_flash::expert::pack_items(l, e, guw, &gus, dw, &dsc)
                }
                _ => v1::items(&src, l, e),
            };
            for (off, bytes) in items {
                blk[off - base..off - base + bytes.len()].copy_from_slice(&bytes);
            }
        }
        if let Some(old) = first_old {
            check_block(&blk, base, l, e0, &old);
        }
        fs::write(&side, &blk).unwrap();
        f.seek(SeekFrom::Start(base as u64)).unwrap();
        f.write_all(&blk).unwrap();
        done += 1;
        fs::write(&done_path, format!("{done}\n")).unwrap();
        if done % 20 == 0 || done == blocks {
            f.sync_data().unwrap();
            eprintln!("relayout : {done}/{blocks} blocks, {:.0}s", t0.elapsed().as_secs_f32());
        }
    }
    f.set_len(layout::STREAM as u64).unwrap();
    f.sync_all().unwrap();
    let _ = fs::remove_file(&side);
    let _ = fs::remove_file(&done_path);
    eprintln!("relayout : {blocks} blocks in place in {:.1}s, {} bytes; update the manifest's sha256 of the file", t0.elapsed().as_secs_f32(), layout::STREAM);
}

/// Every MoE layer's experts (the main stack's, then the draft layer's), or with `--only-layer L`
/// one layer's into the file as it stands (the file grown to the stream's length: how the draft
/// layer joins a bundle built before it).
fn write_experts(ck: &Checkpoint, path: &std::path::Path) {
    let only: Option<usize> = harness::arg("--only-layer").map(|v| v.to_string_lossy().parse().unwrap());
    let jobs: Vec<(usize, usize)> = (FIRST_K_DENSE_REPLACE..layout::LAYERS_IN_FILE).filter(|&l| only.is_none_or(|o| o == l)).flat_map(|l| (0..E).map(move |e| (l, e))).collect();
    let file = fs::OpenOptions::new().read(true).write(true).create(true).truncate(only.is_none()).open(path).unwrap();
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
                for (off, bytes) in pack_expert(ck, l, e) {
                    file.write_all_at(&bytes, off as u64).unwrap();
                }
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
    for l in 0..layout::LAYERS_IN_FILE {
        let mut off = layout::layer(l);
        let ss = layout::sections(l);
        let secs: Vec<&[layout::Tensor]> = if l == layout::MTP_LAYER { vec![ss[0], ss[1], ss[2], layout::ffn(l)] } else { ss.to_vec() };
        for s in secs {
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
    // the prefix positions' hidden states: the streams' mean before the final norm (the draft layer's input)
    for t in 0..n {
        for d in 0..H {
            let mut acc = 0f32;
            for j in 0..HC {
                acc += streams[(t * HC + j) * H + d];
            }
            rows.push(acc / HC as f32);
        }
    }
    assert_eq!(rows.len(), layout::PREFIX_LEN);
    fs::write(path, unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, 4 * rows.len()) }).unwrap();
}

pub fn build() {
    if let Some(p) = harness::arg("--relayout") {
        return relayout(&p, &harness::arg("--from").map_or("legacy".to_string(), |v| v.to_string_lossy().into_owned()));
    }
    let snapshot = arg("--snapshot");
    // the bundle's root holds the manifest and CACHEDIR.TAG; this model's files sit under a directory named for it
    let root = arg("--out");
    let out = root.join("glm-5.3-flash");
    fs::create_dir_all(&out).unwrap();
    fs::create_dir_all(&out).unwrap();
    // a bundle is a function of the checkpoint and the recipe: reproducible, so backups skip it
    fs::write(root.join("CACHEDIR.TAG"), "Signature: 8a477f597d28d172789f06886806bc55\n# rockem bundle: rebuilt from the checkpoint by `bundle --model glm-5.3-flash`\n").unwrap();
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
    let shas: Vec<String> = files.iter().map(|f| format!("\"glm-5.3-flash/{f}\" = \"sha256:{}\"", sha256_file(&out.join(f)))).collect();
    eprintln!("hashed {} files in {:.0}s", files.len(), t2.elapsed().as_secs_f32());
    let ck_toml = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/glm-5.3-flash/checkpoint.toml")).unwrap_or_default();
    let mut m = fs::File::create(root.join("manifest.toml")).unwrap();
    // [[model]]: one entry a model the bundle carries (v5, docs/queue/094) — its name, its recipe and geometry,
    // and the constants the wire assumes so the edge speaks it without reading the crate: `bos` is the prefix's
    // first id and `eos` its first end id, with the full sets beside them
    writeln!(
        m,
        "[[model]]\nname = \"glm-5.3-flash\"\nrecipe = \"{}\"\nrecord_bytes = {}\nexperts = {}\nlayers = [{}, {}]\nbos = {}\neos = {}\nprefix = [{}]\neos_tokens = [{}]\nvocab_size = {VOCAB_SIZE}\n\n[files]\n{}\n\n# checkpoint.toml\n{}",
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
    println!("{}", root.join("manifest.toml").display());
}
