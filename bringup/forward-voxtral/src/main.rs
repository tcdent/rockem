//! forward-voxtral — the shipped host decoder (`voxtral_small_24b::decoder`) over an oracle set, diffed
//! against the goldens (bringup/voxtral/oracle.py): the oracle's `embed` rows are the decoder's inputs at
//! every position (the audio rows included, so the encoder is out of the picture), the positions go through
//! in steps of `STEP`, and every layer's residual output and the logits are diffed. The decoder is INT4
//! with INT8 activations against an f32 reference on bf16 values, so the numbers to read are the layer
//! outputs' relative error growth and the logits' top-1 agreement over the transcript — the recipe's cost,
//! with a kernel error standing out as a jump.
//!
//!   forward-voxtral --goldens <bundles/goldens-voxtral/SET> --bundle <bundle dir> [--positions N] [--tol 1e-2]
//!                   [--serial]   the serial pool (one core), to separate the pool from the arithmetic
//!                   [--encoder]  the encoder and projector over the set's windows (`windows.f32`) instead: the encoder's
//!                                output and the rows diffed against `enc_out` and `rows` (f32 on bf16 both sides: tight)

use glm_5_3_flash::construct::{physical_cores, Threads};
use harness::{arg, arg_required, flag, Golden, Report};
use voxtral_small_24b::decoder::{Decoder, STEP};
use voxtral_small_24b::encoder::Encoder;
use voxtral_small_24b::layout;
use voxtral_small_24b::weights::{Dense, Stream};
use voxtral_small_24b::*;

fn main() {
    let gdir = arg_required("--goldens");
    let bundle = arg_required("--bundle").join("voxtral-small-24b");
    let tol: f32 = arg("--tol").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(1e-2);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let prompt_len = meta["prompt_len"].as_u64().unwrap() as usize;
    let n_all = arg("--positions").map(|p| p.to_str().unwrap().parse::<usize>().unwrap()).unwrap_or(ids.len()).min(ids.len());
    let gold = Golden::load(&gdir.join("edges.safetensors"));
    let t0 = std::time::Instant::now();
    let dense_b = std::fs::read(bundle.join(layout::DENSE_FILE)).unwrap();
    let stream_f = std::fs::File::open(bundle.join(layout::STREAM_FILE)).unwrap();
    let stream_m = unsafe { memmap2::Mmap::map(&stream_f).unwrap() };
    let (dense, stream) = (Dense::new(&dense_b), Stream::new(&stream_m));
    let serial = common::pool::Serial;
    let threads;
    let pool: &dyn common::pool::Pool = if flag("--serial") {
        &serial
    } else {
        threads = Threads::new(&physical_cores());
        &threads
    };
    if flag("--encoder") {
        return encoder_gate(dense, pool, &gdir, &gold, &meta, tol);
    }
    let mut dec = Decoder::new(dense, stream, pool, 1, n_all.max(64));
    dec.trace = Some(Vec::new());
    println!("bundle   : {} in {:.1}s; {} positions of {} (prompt {prompt_len}), transcript {:?}", bundle.display(), t0.elapsed().as_secs_f32(), n_all, ids.len(), meta["transcript"].as_str().unwrap_or(""));
    let mut report = Report::new(tol);
    let (mut agree, mut scored) = (0usize, 0usize);
    let mut pos = 0;
    while pos < n_all {
        let n = (n_all - pos).min(STEP);
        let t1 = std::time::Instant::now();
        let rows: Vec<&[f32]> = (0..n).map(|t| gold.row("embed", pos + t).unwrap()).collect();
        dec.step(&ids[pos..pos + n], pos, |t| Some(rows[t]), true);
        let tr = dec.trace.as_ref().unwrap();
        for t in 0..n {
            for l in 0..NUM_HIDDEN_LAYERS {
                let key = format!("l{l}.out");
                if let Some(g) = gold.row(&key, pos + t) {
                    report.record(&key, &tr[(l * n + t) * H..(l * n + t + 1) * H], g, &format!("pos {}", pos + t));
                }
            }
            let mine = &dec.logits[t * VOCAB_SIZE..(t + 1) * VOCAB_SIZE];
            if let Some(g) = gold.row("logits", pos + t) {
                report.record("logits", mine, g, &format!("pos {}", pos + t));
                if pos + t + 1 >= prompt_len && pos + t + 1 < ids.len() {
                    // a transcript position: the id the reference wrote next, against our argmax
                    scored += 1;
                    agree += (common::softmax::argmax(mine) as u32 == ids[pos + t + 1]) as usize;
                }
            }
        }
        println!("step at {pos} ({n} tokens): {:.2}s", t1.elapsed().as_secs_f32());
        pos += n;
    }
    report.print();
    println!("top-1 over the transcript: {agree}/{scored}; the reference's own {:.3}", meta["top1_agreement"].as_f64().unwrap_or(0.0));
}

fn encoder_gate(dense: Dense, pool: &dyn common::pool::Pool, gdir: &std::path::Path, gold: &Golden, meta: &serde_json::Value, tol: f32) {
    let windows = meta["windows"].as_u64().unwrap() as usize;
    let bytes = std::fs::read(gdir.join("windows.f32")).unwrap();
    assert_eq!(bytes.len(), windows * NUM_MEL_BINS * MEL_FRAMES * 4, "windows.f32");
    let mel: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let t0 = std::time::Instant::now();
    let mut enc = Encoder::new(dense, pool);
    enc.trace = Some(Vec::new());
    println!("encoder  : built in {:.1}s; {windows} windows", t0.elapsed().as_secs_f32());
    let mut report = Report::new(tol);
    let mut rows = vec![0f32; ROWS_PER_WINDOW * H];
    for w in 0..windows {
        let t1 = std::time::Instant::now();
        enc.encode(&mel[w * NUM_MEL_BINS * MEL_FRAMES..(w + 1) * NUM_MEL_BINS * MEL_FRAMES], &mut rows);
        let tr = enc.trace.as_ref().unwrap();
        for p in 0..ENC_POSITIONS {
            if let Some(g) = gold.row("enc_out", w * ENC_POSITIONS + p) {
                report.record("enc_out", &tr[p * ENC_H..(p + 1) * ENC_H], g, &format!("window {w} pos {p}"));
            }
        }
        for r in 0..ROWS_PER_WINDOW {
            if let Some(g) = gold.row("rows", w * ROWS_PER_WINDOW + r) {
                report.record("rows", &rows[r * H..(r + 1) * H], g, &format!("window {w} row {r}"));
            }
        }
        println!("window {w}: {:.2}s", t1.elapsed().as_secs_f32());
    }
    report.print();
}
