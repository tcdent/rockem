//! Voxtral-Small-24B on the host: the engine behind the `Engine` trait — the decoder over the q4 stream
//! (`voxtral_small_24b::decoder`), the audio encoder and projector (`encoder`) turning each AUDIO window
//! into rows the placeholders take, both on one pinned pool. Transcription is greedy. CPU only: the cards
//! stay GLM's (tcdent, 2026-10-23: both models resident, not parallel, the wire selecting).

use crate::engine::{Engine, Sampling};
use crate::loader;
use glm_5_3_flash::construct::{node_count, physical_cores, Threads};
use std::collections::VecDeque;
use voxtral_small_24b::decoder::{Decoder, POSITION, STEP};
use voxtral_small_24b::encoder::Encoder;
use voxtral_small_24b::layout;
use voxtral_small_24b::weights::{Dense, Stream};
use voxtral_small_24b::*;

pub const SEED: u64 = 0x726f_636b_656d_0003;

pub struct Voxtral {
    dec: Decoder<'static>,
    enc: Encoder<'static>,
    pool: &'static Threads,
    /// Rows the windows queued, a window's `ROWS_PER_WINDOW · H` f32 each; `taken` of the front one consumed.
    rows: VecDeque<Vec<f32>>,
    taken: usize,
    prefix_rows: Vec<f32>,
    prefix_logits: Vec<f32>,
    last_pos: usize,
    empty: Vec<f32>,
}

impl Voxtral {
    pub fn open(bundle: &std::path::Path, mem: loader::Mem, max_ctx: usize) -> Box<dyn Engine> {
        let t0 = std::time::Instant::now();
        let dense = loader::load(&bundle.join(layout::DENSE_FILE), mem);
        // the stream's runs bound to nodes: run r of every layer (and of the head) on node r mod nodes
        let nodes = node_count().max(1);
        let mut ranges = Vec::new();
        for l in 0..NUM_HIDDEN_LAYERS {
            for r in 0..layout::RUNS {
                ranges.push((l * layout::LAYER + r * layout::LAYER_RUN, layout::LAYER_RUN, r));
            }
        }
        for r in 0..layout::RUNS {
            ranges.push((NUM_HIDDEN_LAYERS * layout::LAYER + r * layout::HEAD_RUN, layout::HEAD_RUN, r));
        }
        let stream = loader::load_bound(&bundle.join(layout::STREAM_FILE), mem, &ranges, nodes);
        println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense.len() + stream.len()) as f64 / 1e9);
        let pool: &'static Threads = Box::leak(Box::new(Threads::new(&physical_cores())));
        let (dense, stream) = (Dense::new(dense), Stream::new(stream));
        let mut dec = Decoder::new(dense, stream, pool, nodes, max_ctx);
        let enc = Encoder::new(dense, pool);
        // the prefix (BOS) from nothing: its rows and logits
        dec.step(&PREFIX_IDS[..], 0, |_| None, true);
        let mut prefix_rows = vec![0f32; PREFIX_IDS.len() * POSITION];
        for (p, r) in prefix_rows.chunks_exact_mut(POSITION).enumerate() {
            dec.rows(p, r);
        }
        let prefix_logits = dec.logits[..VOCAB_SIZE].to_vec();
        println!("voxtral  : {} positions of context ({:.1} GB of K/V), the prefix computed in {:.1}s", dec.max_ctx(), 2.0 * (dec.max_ctx() * POSITION) as f64 * 4.0 / 2.0 / 1e9, t0.elapsed().as_secs_f32());
        Box::new(Voxtral { dec, enc, pool, rows: VecDeque::new(), taken: 0, prefix_rows, prefix_logits, last_pos: 0, empty: Vec::new() })
    }
}

impl Engine for Voxtral {
    fn name(&self) -> &'static str {
        "voxtral-small-24b"
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
        // the placeholders' rows, in order, from the windows' queue
        let mut rows: Vec<Option<Vec<f32>>> = Vec::with_capacity(toks.len());
        for &t in toks {
            if t == AUDIO_ID {
                let front = self.rows.front().expect("a placeholder with no row queued");
                rows.push(Some(front[self.taken * H..(self.taken + 1) * H].to_vec()));
                self.taken += 1;
                if self.taken == ROWS_PER_WINDOW {
                    self.rows.pop_front();
                    self.taken = 0;
                }
            } else {
                rows.push(None);
            }
        }
        let rows_ref: &Vec<Option<Vec<f32>>> = &rows;
        // SAFETY: the rows outlive the step; the decoder only reads them within it
        let rows_static: &'static Vec<Option<Vec<f32>>> = unsafe { std::mem::transmute(rows_ref) };
        self.dec.step(toks, pos, |t| rows_static[t].as_deref(), all_logits);
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        self.dec.rows(self.last_pos + i, out)
    }
    fn sampling(&self) -> &'static Sampling {
        &Sampling::GREEDY
    }
    fn logits_row(&self, i: usize) -> &[f32] {
        &self.dec.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE]
    }
    fn set_logits(&mut self, i: usize, logits: &[f32]) {
        self.dec.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE].copy_from_slice(&logits[..VOCAB_SIZE]);
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
        (&self.empty, &[], &self.dec.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE])
    }
    fn audio_window(&self) -> usize {
        NUM_MEL_BINS * MEL_FRAMES
    }
    fn audio_placeholder(&self) -> u32 {
        AUDIO_ID
    }
    fn audio(&mut self, window: &[f32]) {
        let mut rows = vec![0f32; ROWS_PER_WINDOW * H];
        self.enc.encode(window, &mut rows);
        self.rows.push_back(rows);
    }
    fn audio_queued(&self) -> usize {
        self.rows.len() * ROWS_PER_WINDOW - self.taken
    }
    fn audio_reset(&mut self) {
        self.rows.clear();
        self.taken = 0;
    }
}
