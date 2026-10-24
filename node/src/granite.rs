//! Granite Speech 4.1 2B on the host (docs/queue/096): the engine behind the `Engine` trait — the decoder over
//! the q4 stream (`granite_speech_4_1_2b::decoder`), the Conformer encoder and Q-Former projector (`encoder`)
//! turning a request's AUDIO windows into rows the placeholders take, both on one pinned pool. The encoder runs
//! over all of a request's windows at once (its attention blocks and convolutions span them), so the windows are
//! queued as they arrive and encoded when the first placeholder needs a row. Transcription is greedy. CPU only:
//! the cards stay GLM's (tcdent, 2026-10-24: a model suited to the CPU, the GPUs free for the LLM).

use crate::engine::{Engine, Sampling};
use crate::loader;
use glm_5_3_flash::construct::{node_count, Threads};
use std::collections::VecDeque;
use granite_speech_4_1_2b::decoder::{Decoder, POSITION, STEP};
use granite_speech_4_1_2b::encoder::Encoder;
use granite_speech_4_1_2b::layout;
use granite_speech_4_1_2b::weights::{Dense, Stream};
use granite_speech_4_1_2b::*;

pub const SEED: u64 = 0x726f_636b_656d_0004;

pub struct Granite {
    dec: Decoder<'static>,
    enc: Encoder<'static>,
    pool: &'static Threads,
    /// The request's windows not yet encoded (`[NUM_MEL_BINS][MEL_FRAMES]` bin-major each), encoded together when a
    /// placeholder first needs a row.
    pending: Vec<Vec<f32>>,
    /// Rows encoded and not yet consumed, `H` f32 each.
    rows: VecDeque<Vec<f32>>,
    prefix_rows: Vec<f32>,
    prefix_logits: Vec<f32>,
    last_pos: usize,
    empty: Vec<f32>,
}

impl Granite {
    /// `pool`: the node's one pinned pool, shared with every other host engine.
    pub fn open(bundle: &std::path::Path, mem: loader::Mem, max_ctx: usize, pool: &'static Threads) -> Box<dyn Engine> {
        let t0 = std::time::Instant::now();
        let dense = loader::load(&bundle.join(layout::DENSE_FILE), mem);
        // the stream's regions bound to nodes: region r (share r of every matrix) on node r mod nodes
        let nodes = node_count().max(1);
        let ranges: Vec<(usize, usize, usize)> = (0..layout::RUNS).map(|r| (r * layout::REGION, layout::REGION, r)).collect();
        let stream = loader::load_bound(&bundle.join(layout::STREAM_FILE), mem, &ranges, nodes);
        println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense.len() + stream.len()) as f64 / 1e9);
        pool.place(stream.as_ptr() as usize, stream.len(), loader::placement(stream));
        let (dense, stream) = (Dense::new(dense), Stream::new(stream));
        let mut dec = Decoder::new(dense, stream, pool, nodes, max_ctx);
        let enc = Encoder::new(dense, pool);
        // the prefix (the template's `USER`) from nothing: its rows and logits
        dec.step(&PREFIX_IDS[..], 0, |_| None, true);
        let mut prefix_rows = vec![0f32; PREFIX_IDS.len() * POSITION];
        for (p, r) in prefix_rows.chunks_exact_mut(POSITION).enumerate() {
            dec.rows(p, r);
        }
        let prefix_logits = dec.logits[..VOCAB_SIZE].to_vec();
        println!("granite  : {} positions of context ({:.1} GB of K/V), the prefix computed in {:.1}s", dec.max_ctx(), (dec.max_ctx() * POSITION) as f64 * 4.0 / 1e9, t0.elapsed().as_secs_f32());
        Box::new(Granite { dec, enc, pool, pending: Vec::new(), rows: VecDeque::new(), prefix_rows, prefix_logits, last_pos: 0, empty: Vec::new() })
    }
}

impl Granite {
    /// The pending windows, concatenated bin-major, through the encoder and projector: `ROWS_PER_WINDOW` rows each,
    /// queued for the placeholders. A row a request on `track-step` (`granite_window`: the encoder's phases).
    fn encode_pending(&mut self) {
        let n = self.pending.len();
        assert!(n > 0, "a placeholder with no window");
        let frames = n * MEL_FRAMES;
        let mut mel = vec![0f32; NUM_MEL_BINS * frames];
        for (w, win) in self.pending.iter().enumerate() {
            for bin in 0..NUM_MEL_BINS {
                mel[bin * frames + w * MEL_FRAMES..bin * frames + (w + 1) * MEL_FRAMES].copy_from_slice(&win[bin * MEL_FRAMES..(bin + 1) * MEL_FRAMES]);
            }
        }
        self.pending.clear();
        let mut rows = vec![0f32; n * ROWS_PER_WINDOW * H];
        let t0 = std::time::Instant::now();
        let got = self.enc.encode(&mel, frames, &mut rows);
        assert_eq!(got, n * ROWS_PER_WINDOW, "rows a window");
        let p = self.enc.phases.map(|s| s * 1e3);
        prow::row!("track-step", "granite_window", {
            "windows" => n, "frames" => frames, "ms" => t0.elapsed().as_secs_f64() * 1e3, "input_ms" => p[0], "ff_ms" => p[1], "attn_ms" => p[2],
            "conv_ms" => p[3], "norm_ms" => p[4], "ctc_ms" => p[5], "proj_ms" => p[6],
        });
        for r in rows.chunks_exact(H) {
            self.rows.push_back(r.to_vec());
        }
    }
}

impl Engine for Granite {
    fn name(&self) -> &'static str {
        "granite-speech-4.1-2b"
    }
    fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }
    fn is_eos(&self, id: u32) -> bool {
        EOS_TOKENS.contains(&id)
    }
    fn tokens(&self) -> usize {
        STEP
    }
    fn prefix_ids(&self) -> &[u32] {
        PREFIX_IDS
    }
    fn position(&self) -> usize {
        POSITION
    }
    fn context_limit(&self) -> usize {
        self.dec.max_ctx()
    }
    fn snapshot_len(&self) -> usize {
        0
    }
    fn seed_value(&self) -> u64 {
        SEED
    }
    fn prefix(&self) -> (&[f32], &[f32]) {
        (&self.prefix_rows, &self.empty)
    }
    fn seed_prefix(&mut self) {
        self.dec.seed(&self.prefix_rows)
    }
    fn seed(&mut self, rows: &[f32], _snap: &[f32]) {
        self.dec.seed(rows)
    }
    fn snapshot(&self, _out: &mut [f32]) {}
    fn run(&mut self, toks: &[u32], pos: usize, _trace: bool, all_logits: bool) {
        assert!(!toks.is_empty() && toks.len() <= STEP);
        self.last_pos = pos;
        // the placeholders' rows, in order: the pending windows encoded together when the first is needed
        let mut rows: Vec<Option<Vec<f32>>> = Vec::with_capacity(toks.len());
        for &t in toks {
            if t == AUDIO_ID {
                if self.rows.is_empty() {
                    self.encode_pending();
                }
                rows.push(Some(self.rows.pop_front().expect("a placeholder with no row queued")));
            } else {
                rows.push(None);
            }
        }
        let rows_ref: &Vec<Option<Vec<f32>>> = &rows;
        // SAFETY: the rows outlive the step; the decoder only reads them within it
        let rows_static: &'static Vec<Option<Vec<f32>>> = unsafe { std::mem::transmute(rows_ref) };
        let t0 = std::time::Instant::now();
        self.dec.step(toks, pos, |t| rows_static[t].as_deref(), all_logits);
        // the step's time by phase (decoder::PHASES), a row a step
        let (ms, p) = (t0.elapsed().as_secs_f64() * 1e3, self.dec.phases.map(|s| s * 1e3));
        prow::row!("track-step", "granite_step", {
            "pos" => pos, "width" => toks.len(), "ms" => ms, "embed_ms" => p[0], "norm_ms" => p[1], "qkv_ms" => p[2], "rope_ms" => p[3],
            "attn_ms" => p[4], "o_ms" => p[5], "gu_ms" => p[6], "silu_ms" => p[7], "down_ms" => p[8], "head_ms" => p[9],
        });
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        self.dec.rows(self.last_pos + i, out)
    }
    fn sampling(&self) -> &'static Sampling {
        &Sampling::GREEDY
    }
    fn logits_row(&self, i: usize) -> &[f32] {
        &self.dec.logits[i * VOCAB_PAD..i * VOCAB_PAD + VOCAB_SIZE]
    }
    fn set_logits(&mut self, i: usize, logits: &[f32]) {
        self.dec.logits[i * VOCAB_PAD..i * VOCAB_PAD + VOCAB_SIZE].copy_from_slice(&logits[..VOCAB_SIZE]);
    }
    fn draw(&mut self, i: usize, u: f64) -> u32 {
        common::sample::draw(self.logits_row(i), &Sampling::GREEDY, u) as u32
    }
    fn score(&mut self, i: usize, id: u32) -> f32 {
        common::softmax::log_prob(self.logits_row(i), id as usize)
    }
    fn draw_prefix(&mut self, u: f64) -> u32 {
        common::sample::draw(&self.prefix_logits, &Sampling::GREEDY, u) as u32
    }
    fn score_prefix(&mut self, id: u32) -> f32 {
        common::softmax::log_prob(&self.prefix_logits, id as usize)
    }
    fn idle(&mut self) {
        self.pool.idle()
    }
    fn trace(&mut self, i: usize) -> (&[f32], &[u32], &[f32]) {
        (&self.empty, &[], &self.dec.logits[i * VOCAB_PAD..i * VOCAB_PAD + VOCAB_SIZE])
    }
    fn audio_window(&self) -> usize {
        NUM_MEL_BINS * MEL_FRAMES
    }
    fn audio_placeholder(&self) -> u32 {
        AUDIO_ID
    }
    fn audio(&mut self, window: &[f32]) {
        self.pending.push(window.to_vec());
    }
    fn audio_queued(&self) -> usize {
        self.pending.len() * ROWS_PER_WINDOW + self.rows.len()
    }
    fn audio_reset(&mut self) {
        self.pending.clear();
        self.rows.clear();
    }
}
