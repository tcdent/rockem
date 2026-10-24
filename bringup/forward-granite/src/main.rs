//! forward-granite — the shipped host encoder and decoder (`granite_speech_4_1_2b`) over an oracle set
//! (bringup/granite/oracle.py), diffed against the goldens. The decoder gate takes the oracle's `embed` rows
//! as the decoder's inputs at every position (the audio rows included, so the encoder is out of the
//! picture), steps of `STEP`, every layer's residual output and the logits diffed: the INT4 recipe's cost
//! against an f32 reference on bf16 values (run on the `-q4` set for the kernels' error alone). The
//! encoder gate runs the set's log-mel (`mel.f32`) through the encoder and projector and diffs every block's
//! output, the encoder's output and the rows (f32 on bf16 both sides: tight).
//!
//!   forward-granite --goldens <bundles/goldens-granite/SET> --bundle <bundle dir> [--positions N] [--tol 1e-2]
//!                   [--serial]      the serial pool (one core), to separate the pool from the arithmetic
//!                   [--encoder]     the encoder gate instead
//!                   [--transcribe]  end to end: the set's log-mel through the encoder, its prompt ids through the
//!                                   decoder with the rows at the placeholders, greedy decoding to EOS or --new tokens;
//!                                   the transcript through the bundle's vocab.bin, the phase times

use glm_5_3_flash::construct::{physical_cores, Threads};
use granite_speech_4_1_2b::decoder::{Decoder, STEP};
use granite_speech_4_1_2b::encoder::Encoder;
use granite_speech_4_1_2b::layout;
use granite_speech_4_1_2b::weights::{Dense, Stream};
use granite_speech_4_1_2b::*;
use harness::{arg, arg_required, flag, Golden, Report};

fn main() {
    let bundle = arg_required("--bundle").join("granite-speech-4.1-2b");
    let tol: f32 = arg("--tol").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(1e-2);
    let gdir = arg_required("--goldens");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(gdir.join("meta.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let prompt_len = meta["prompt_len"].as_u64().unwrap() as usize;
    let frames = 2 * meta["frames"].as_u64().unwrap() as usize;
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
    println!("bundle   : {} in {:.1}s; {} ids (prompt {prompt_len}), {frames} mel frames; transcript {:?}", bundle.display(), t0.elapsed().as_secs_f32(), ids.len(), meta["transcript"].as_str().unwrap_or(""));
    let mel = || -> Vec<f32> {
        let bytes = std::fs::read(gdir.join("mel.f32")).unwrap();
        assert_eq!(bytes.len(), NUM_MEL_BINS * frames * 4, "mel.f32");
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    };
    if flag("--encoder") {
        let gold = Golden::load(&gdir.join("edges.safetensors"));
        return encoder_gate(dense, pool, &mel(), frames, &gold, tol);
    }
    if flag("--transcribe") {
        return transcribe(&bundle, dense, stream, pool, &mel(), frames, &ids[..prompt_len], arg("--new").map(|p| p.to_str().unwrap().parse().unwrap()).unwrap_or(128));
    }
    let gold = Golden::load(&gdir.join("edges.safetensors"));
    let n_all = arg("--positions").map(|p| p.to_str().unwrap().parse::<usize>().unwrap()).unwrap_or(ids.len()).min(ids.len());
    let mut dec = Decoder::new(dense, stream, pool, 1, n_all.max(64));
    dec.trace = Some(Vec::new());
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
                    let (abs, rel) = report.record(&key, &tr[(l * n + t) * H..(l * n + t + 1) * H], g, &format!("pos {}", pos + t));
                    if l == 0 && pos + t < 4 {
                        println!("  l0.out pos {}: max|Δ| {abs:.3e} rel {rel:.3e}; mine[..4] {:?} gold[..4] {:?}", pos + t, &tr[t * H..t * H + 4], &g[..4]);
                    }
                }
            }
            let mine = &dec.logits[t * VOCAB_PAD..t * VOCAB_PAD + VOCAB_SIZE];
            if let Some(g) = gold.row("logits", pos + t) {
                report.record("logits", mine, g, &format!("pos {}", pos + t));
                if pos + t + 1 >= prompt_len && pos + t + 1 < ids.len() {
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

fn phases_line(ph: &[f64], names: &[&str]) -> String {
    names.iter().zip(ph).map(|(n, s)| format!("{n} {:.2}", s)).collect::<Vec<_>>().join(", ")
}

fn encoder_gate(dense: Dense, pool: &dyn common::pool::Pool, mel: &[f32], frames: usize, gold: &Golden, tol: f32) {
    let t0 = std::time::Instant::now();
    let mut enc = Encoder::new(dense, pool);
    enc.trace = Some(Vec::new());
    enc.trace_blocks = Some(Vec::new());
    println!("encoder  : built in {:.1}s", t0.elapsed().as_secs_f32());
    let t = frames / 2;
    let mut rows = vec![0f32; t.div_ceil(WINDOW_SIZE) * QUERIES * H];
    let t1 = std::time::Instant::now();
    let n_rows = enc.encode(mel, frames, &mut rows);
    println!("encode   : {:.2}s for {t} frames → {n_rows} rows ({})", t1.elapsed().as_secs_f32(), phases_line(&enc.phases, &granite_speech_4_1_2b::encoder::PHASES));
    let mut report = Report::new(tol);
    for (l, blk) in enc.trace_blocks.as_ref().unwrap().iter().enumerate() {
        // entry ENC_NUM_HIDDEN_LAYERS is block 8's output after the CTC term: diffed against the same golden as block 8
        let key = if l == ENC_NUM_HIDDEN_LAYERS { format!("enc.l{}+ctc", ENC_NUM_HIDDEN_LAYERS / 2 - 1) } else { format!("enc.l{l}") };
        let gkey = format!("enc.l{}", if l == ENC_NUM_HIDDEN_LAYERS { ENC_NUM_HIDDEN_LAYERS / 2 - 1 } else { l });
        for p in 0..t {
            if let Some(g) = gold.row(&gkey, p) {
                report.record(&key, &blk[p * ENC_H..(p + 1) * ENC_H], g, &format!("pos {p}"));
            }
        }
    }
    let tr = enc.trace.as_ref().unwrap();
    for p in 0..t {
        if let Some(g) = gold.row("enc.out", p) {
            report.record("enc.out", &tr[p * ENC_H..(p + 1) * ENC_H], g, &format!("pos {p}"));
        }
    }
    for r in 0..n_rows {
        if let Some(g) = gold.row("rows", r) {
            report.record("rows", &rows[r * H..(r + 1) * H], g, &format!("row {r}"));
        }
    }
    report.print();
}

/// The bundle's vocabulary: byte offsets then bytes, by id (`vocab.bin`).
struct Vocab {
    offs: Vec<u32>,
    bytes: Vec<u8>,
}
impl Vocab {
    fn load(path: &std::path::Path) -> Self {
        let b = std::fs::read(path).unwrap();
        let n = VOCAB_SIZE + 1;
        let offs = b[..4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Vocab { offs, bytes: b[4 * n..].to_vec() }
    }
    fn token(&self, id: usize) -> &[u8] {
        &self.bytes[self.offs[id] as usize..self.offs[id + 1] as usize]
    }
}

fn transcribe(bundle: &std::path::Path, dense: Dense, stream: Stream, pool: &dyn common::pool::Pool, mel: &[f32], frames: usize, ids: &[u32], max_new: usize) {
    let vocab = Vocab::load(&bundle.join(layout::VOCAB_FILE));
    let t0 = std::time::Instant::now();
    let mut enc = Encoder::new(dense, pool);
    let t = frames / 2;
    let mut rows = vec![0f32; t.div_ceil(WINDOW_SIZE) * QUERIES * H];
    let t_built = t0.elapsed();
    let t1 = std::time::Instant::now();
    let n_rows = enc.encode(mel, frames, &mut rows);
    let t_enc = t1.elapsed();
    let enc_phases = phases_line(&enc.phases, &granite_speech_4_1_2b::encoder::PHASES);
    drop(enc);
    let mut dec = Decoder::new(dense, stream, pool, 1, ids.len() + max_new + 8);
    let placeholders = ids.iter().filter(|&&id| id == AUDIO_ID).count();
    assert_eq!(placeholders, n_rows, "placeholders vs rows");
    let t2 = std::time::Instant::now();
    let (mut taken, mut pos) = (0usize, 0usize);
    let mut dec_phases = [0f64; 10];
    while pos < ids.len() {
        let n = (ids.len() - pos).min(STEP);
        let mut step_rows: Vec<Option<&[f32]>> = Vec::with_capacity(n);
        for t in 0..n {
            if ids[pos + t] == AUDIO_ID {
                step_rows.push(Some(&rows[taken * H..(taken + 1) * H]));
                taken += 1;
            } else {
                step_rows.push(None);
            }
        }
        let sr = &step_rows;
        dec.step(&ids[pos..pos + n], pos, |t| sr[t], false);
        for (a, p) in dec_phases.iter_mut().zip(dec.phases) {
            *a += p;
        }
        pos += n;
    }
    let t_prefill = t2.elapsed();
    let last_n = ids.len() - (ids.len() - 1) / STEP * STEP;
    let mut next = common::softmax::argmax(&dec.logits[(last_n - 1) * VOCAB_PAD..(last_n - 1) * VOCAB_PAD + VOCAB_SIZE]) as u32;
    let t3 = std::time::Instant::now();
    let mut out = Vec::new();
    while out.len() < max_new && !EOS_TOKENS.contains(&next) {
        out.push(next);
        dec.step(&[next], pos, |_| None, false);
        pos += 1;
        next = common::softmax::argmax(&dec.logits[..VOCAB_SIZE]) as u32;
    }
    let text: Vec<u8> = out.iter().flat_map(|&id| vocab.token(id as usize).to_vec()).collect();
    println!("transcript ({} ids, EOS {}): {}", out.len(), EOS_TOKENS.contains(&next), String::from_utf8_lossy(&text));
    println!("encoder built {:.1}s; encode {:.2}s for {t} frames ({enc_phases}); prefill {} ids {:.2}s (step phases summed: {}); {} tokens at {:.0} ms each", t_built.as_secs_f32(), t_enc.as_secs_f32(), ids.len(), t_prefill.as_secs_f32(), phases_line(&dec_phases, &granite_speech_4_1_2b::decoder::PHASES), out.len(), t3.elapsed().as_secs_f64() * 1e3 / out.len().max(1) as f64);
}
