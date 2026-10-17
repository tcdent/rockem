//! GLM-5.3-Flash as the node runs it: the bundle loaded into hugetlbfs — the
//! dense weights as the checkpoint stores them, the INT4 expert stream, the
//! prefix state — and the CPU realization (`construct::CpuModel`) behind
//! `Engine`. The card's realization follows, kernel by kernel, diffed against
//! this one.
//!
//! The prefix's logits are the node's own: at start the prefix ids run through
//! the model from nothing, which also measures the realization against the
//! bundle's prefix state (the f32 reference's) — one line on the console.

use crate::engine::{Engine, Sampling};
use crate::loader;
use common::trace::{NoTrace, Trace, NONE};
use glm_5_3_flash::construct::{physical_cores, CpuExperts, CpuModel, Threads};
use glm_5_3_flash::layout::{self, POSITION, SNAPSHOT};
use glm_5_3_flash::weights::Weights;
use glm_5_3_flash::{EOS_TOKENS, FIRST_POS, H, NUM_HIDDEN_LAYERS, PREFIX_IDS, SAMPLING, SEED, STEP, TOP_K, VOCAB_SIZE};

/// Every layer's residual output and routing for each token of a step.
struct DebugTrace {
    tok: usize,
    layers: Vec<f32>, // [STEP][NUM_HIDDEN_LAYERS][H]
    topk: Vec<u32>,   // [STEP][NUM_HIDDEN_LAYERS][TOP_K], 0 for the dense layers
}
impl Trace for DebugTrace {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        if layer != NONE && name == "out" {
            let at = (self.tok * NUM_HIDDEN_LAYERS + layer) * H;
            self.layers[at..at + H].copy_from_slice(data);
        }
    }
    fn topk(&mut self, layer: usize, idx: &[usize], _weight: &[f32]) {
        for (k, &e) in idx.iter().enumerate() {
            self.topk[(self.tok * NUM_HIDDEN_LAYERS + layer) * TOP_K + k] = e as u32;
        }
    }
    fn routed(&mut self, _: usize, _: usize, _: &[f32]) {}
}

pub struct Glm {
    pool: &'static Threads,
    experts: &'static glm_5_3_flash::expert::Stream<'static>,
    /// Serving's expert runs since the last re-placement: each node's items and run ends, and the tokens.
    train: ([usize; 8], [f64; 8], usize),
    /// The profile's token count already folded into `train`.
    train_seen: usize,
    /// A region ran out of room: no more rounds.
    train_stopped: bool,
    /// `rockem.profile=N`: a line of the token's time by phase every N tokens (0: never).
    profile_every: usize,
    w: Weights<'static>,
    model: CpuModel<'static, CpuExperts<'static>>,
    prefix: &'static [f32],
    prefix_logits: Vec<f32>,
    /// The last step's logits, a row per token, and the position of its first token.
    logits: Vec<f32>,
    last_pos: usize,
    dtr: DebugTrace,
}

impl Glm {
    pub fn open(bundle: &std::path::Path, mem: loader::Mem, max_ctx: usize, gpu: bool, profile_every: usize, timeline: bool) -> Box<dyn Engine> {
        let t0 = std::time::Instant::now();
        let dense = loader::load(&bundle.join(layout::DENSE_FILE), mem);
        // the experts, placed equally first; the pool then measures where each node ends its runs
        // and the items move so every node ends at once (below, once the pool is up)
        let stream = loader::load_placed(&bundle.join(layout::EXPERTS_FILE), mem, &layout::placement_runs(glm_5_3_flash::construct::node_count()));
        println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense.len() + stream.bytes.len()) as f64 / 1e9);
        let w = Weights::from_dense(dense);
        let prefix_b = loader::load(&bundle.join(layout::PREFIX_FILE), loader::Mem::Anon);
        let (p, prefix, q) = unsafe { prefix_b.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty() && prefix.len() == layout::PREFIX_LEN, "prefix file");
        // the pool: a thread per physical core, every socket, told where every page of the weights lives
        let cores = physical_cores();
        let pool: &'static Threads = Box::leak(Box::new(Threads::new(&cores)));
        let mut by_node = [0usize; 8];
        for buf in [dense, stream.bytes] {
            let nodes = loader::placement(buf);
            for &d in &nodes {
                by_node[d as usize] += 1;
            }
            pool.place(buf.as_ptr() as usize, buf.len(), nodes);
        }
        let used: Vec<String> = by_node.iter().filter(|&&n| n > 0).map(|&n| format!("{:.1}", n as f64 * 2.0 / 1024.0)).collect();
        println!("threads  : {} pinned, one per physical core; the weights' 2 MiB pages by node: {} GB", cores.len(), used.join(" / "));
        let experts: &'static glm_5_3_flash::expert::Stream<'static> = Box::leak(Box::new(stream));
        // the placement trained on the pool itself: the rates its nodes show on real expert runs, at
        // start on random picks (below), then on serving's own runs between requests (`retrain`)
        if experts.nodes() > 1 {
            let rates = calibrate(experts, pool, 24);
            let r = place(experts, &rates, None);
            println!("placement: at start, {r}");
        }
        let mut model = CpuModel::new(CpuExperts::new(experts, pool), pool, max_ctx);
        if gpu {
            // the KDA layers on the cards, split by layer order between them; everything else here
            let elf = loader::load(&bundle.join(layout::KERNELS_FILE), loader::Mem::Anon);
            model.placed = Some(Box::new(crate::glm_kfd::Cards::new(elf, &w, max_ctx, timeline)));
        }
        // the prefix from nothing: its logits, and the realization against the bundle's state
        let t1 = std::time::Instant::now();
        for (pos, &id) in PREFIX_IDS.iter().enumerate() {
            model.step(&w, id, pos, None, &mut NoTrace, true);
        }
        let prefix_logits = model.logits.clone();
        let mut snap = vec![0f32; SNAPSHOT];
        model.snapshot(&mut snap);
        let mut rows = vec![0f32; FIRST_POS * POSITION];
        for pos in 0..FIRST_POS {
            model.rows(pos, &mut rows[pos * POSITION..(pos + 1) * POSITION]);
        }
        let (prows, psnap) = prefix.split_at(FIRST_POS * POSITION);
        let dev = |a: &[f32], b: &[f32]| a.iter().zip(b).fold((0f32, 0f32), |(d, m), (x, y)| (d.max((x - y).abs()), m.max(y.abs())));
        let (dr, mr) = dev(&rows, prows);
        let (ds, ms) = dev(&snap, psnap);
        println!("prefix   : {} tokens in {:.1}s; this realization vs the bundle's f32 state: rows {:.1e} of {:.1e}, memory {:.1e} of {:.1e}", FIRST_POS, t1.elapsed().as_secs_f32(), dr, mr, ds, ms);
        model.seed_prefix(&w, prefix);
        println!("model    : glm-5.3-flash, ctx {max_ctx}, temperature {}, top-k {}, top-p {}, dense native, experts q4 in DRAM, {}", SAMPLING.temperature, SAMPLING.top_k, SAMPLING.top_p, if gpu { "the KDA layers on the cards, the rest on this CPU's cores" } else { "every op on this CPU's cores" });
        model.profile = Default::default();
        Box::new(Glm {
            pool,
            experts,
            train: ([0; 8], [0.0; 8], 0),
            train_seen: 0,
            train_stopped: false,
            profile_every,
            w,
            model,
            prefix,
            prefix_logits,
            logits: vec![0.0; STEP * VOCAB_SIZE],
            last_pos: 0,
            dtr: DebugTrace { tok: 0, layers: vec![0.0; STEP * NUM_HIDDEN_LAYERS * H], topk: vec![0; STEP * NUM_HIDDEN_LAYERS * TOP_K] },
        })
    }
    /// Fold the profile's per-node run ends into the training accumulators.
    fn take_training(&mut self) {
        let p = &mut self.model.profile;
        for d in 0..8 {
            self.train.0[d] += p.node_items[d];
            self.train.1[d] += p.node_end[d];
            p.node_items[d] = 0;
            p.node_end[d] = 0.0;
        }
        self.train.2 += p.tokens - self.train_seen;
        self.train_seen = p.tokens;
    }
    /// Between requests: re-place the experts from serving's own run ends when a node ends its runs
    /// later than the others by more than the grain of an item — the placement trained on traffic.
    fn retrain(&mut self) {
        self.take_training();
        let (items, ends, tokens) = self.train;
        if tokens < 32 || self.experts.nodes() < 2 || self.train_stopped {
            return;
        }
        let nodes = self.experts.nodes();
        let rates: Vec<f64> = (0..nodes).map(|d| items[d] as f64 / (1e3 * ends[d]).max(1e-9f64)).collect();
        let (lo, hi) = ends[..nodes].iter().fold((f64::MAX, 0f64), |(lo, hi), &e| (lo.min(e), hi.max(e)));
        if hi > lo * 1.02 {
            let r = place(self.experts, &rates, Some(&mut self.model.routed));
            println!("placement: after {tokens} tokens, {r}");
            self.train_stopped = r.contains("no room");
        }
        self.train = ([0; 8], [0.0; 8], 0);
    }
}

impl Engine for Glm {
    fn name(&self) -> &'static str {
        "glm-5.3-flash"
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
        &PREFIX_IDS
    }
    fn position(&self) -> usize {
        POSITION
    }
    fn snapshot_len(&self) -> usize {
        SNAPSHOT
    }
    fn seed_value(&self) -> u64 {
        SEED
    }
    fn prefix(&self) -> (&[f32], &[f32]) {
        self.prefix.split_at(FIRST_POS * POSITION)
    }
    fn seed_prefix(&mut self) {
        self.model.seed_prefix(&self.w, self.prefix)
    }
    fn seed(&mut self, rows: &[f32], snap: &[f32]) {
        self.model.seed(&self.w, rows, snap)
    }
    fn snapshot(&self, out: &mut [f32]) {
        self.model.snapshot(out)
    }
    /// One token after another: `n` tokens are `n` steps.
    fn run(&mut self, toks: &[u32], pos: usize, trace: bool, all_logits: bool) {
        assert!(!toks.is_empty() && toks.len() <= STEP);
        self.last_pos = pos;
        if (trace || all_logits || self.model.placed.is_none()) || toks.len() == 1 {
            // a token at a time: every token's logits (scoring, tracing), or the CPU realization
            for (i, &tok) in toks.iter().enumerate() {
                let logits = all_logits || i == toks.len() - 1;
                if trace {
                    self.dtr.tok = i;
                    self.model.step(&self.w, tok, pos + i, None, &mut self.dtr, logits);
                } else {
                    self.model.step(&self.w, tok, pos + i, None, &mut NoTrace, logits);
                }
                if logits {
                    self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE].copy_from_slice(&self.model.logits);
                }
            }
        } else {
            // the step at once on the cards: the last token's logits
            self.model.step_batch(&self.w, toks, pos, None, &mut NoTrace, true);
            let i = toks.len() - 1;
            self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE].copy_from_slice(&self.model.logits);
        }
        if self.profile_every > 0 && self.model.profile.tokens >= self.profile_every {
            for l in self.model.profile.line().split('\n') {
                println!("profile  : {}", l.trim_start_matches("profile  : "));
            }
            let c = self.model.coverage(&[8, 16, 32, 64]);
            println!("experts  : since start, a layer's 8 / 16 / 32 / 64 most-picked experts took {:.0} / {:.0} / {:.0} / {:.0} % of its picks", 100.0 * c[0], 100.0 * c[1], 100.0 * c[2], 100.0 * c[3]);
            self.take_training();
            self.model.profile = Default::default();
            self.train_seen = 0;
        }
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        self.model.rows(self.last_pos + i, out)
    }
    fn draw(&mut self, i: usize, u: f64) -> u32 {
        common::sample::draw(&self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE], &SAMPLING, u) as u32
    }
    fn score(&mut self, i: usize, id: u32) -> f32 {
        common::softmax::log_prob(&self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE], id as usize)
    }
    fn draw_prefix(&mut self, u: f64) -> u32 {
        common::sample::draw(&self.prefix_logits, &SAMPLING, u) as u32
    }
    fn score_prefix(&mut self, id: u32) -> f32 {
        common::softmax::log_prob(&self.prefix_logits, id as usize)
    }
    fn idle(&mut self) {
        self.retrain();
        self.pool.idle()
    }
    fn trace(&mut self, i: usize) -> (&[f32], &[u32], &[f32]) {
        let a = i * NUM_HIDDEN_LAYERS;
        (&self.dtr.layers[a * H..(a + NUM_HIDDEN_LAYERS) * H], &self.dtr.topk[a * TOP_K..(a + NUM_HIDDEN_LAYERS) * TOP_K], &self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE])
    }
}

const _: () = assert!(Sampling::GREEDY.top_k == 1);

/// Re-place the stream's items by `rates` with the fewest moves; `routed` (serving's experts, if
/// any) relearns its slices' nodes. Returns the line to log.
fn place(experts: &'static glm_5_3_flash::expert::Stream<'static>, rates: &[f64], routed: Option<&mut CpuExperts<'static>>) -> String {
    let t0 = std::time::Instant::now();
    let total: f64 = rates.iter().sum();
    let fmt = |v: &[f64]| v.iter().map(|r| format!("{r:.0}")).collect::<Vec<_>>().join(" / ");
    let shares = rates.iter().map(|r| format!("{:.1}", 100.0 * r / total)).collect::<Vec<_>>().join(" / ");
    let new = layout::placement_from(&experts.node_of_items(), rates);
    match experts.migrate(&new) {
        Ok((moved, bytes)) => {
            if let Some(r) = routed {
                r.relearn();
            }
            format!("node rates {} items/ms → shares {shares} %: {moved} items ({:.2} GB) moved in {:.2} s", fmt(rates), bytes as f64 / 1e9, t0.elapsed().as_secs_f64())
        }
        Err(d) => format!("node rates {} items/ms → shares {shares} %: node {d}'s region has no room; placement kept", fmt(rates)),
    }
}

/// Each node's rate on real expert runs (items a ms): `tokens` tokens' worth of MoE layers with
/// random picks and a random input through the pool, the ends read from it.
fn calibrate(stream: &'static glm_5_3_flash::expert::Stream<'static>, pool: &'static Threads, tokens: usize) -> Vec<f64> {
    use common::pool::Pool;
    use glm_5_3_flash::ffn::Routed;
    use glm_5_3_flash::{E, FIRST_K_DENSE_REPLACE};
    let mut cpu = CpuExperts::new(stream, pool);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let x: Vec<f32> = (0..H).map(|_| (next() % 2001) as f32 / 1000.0 - 1.0).collect();
    let mut out = vec![0f32; TOP_K * H];
    let _ = pool.node_ends();
    for _ in 0..tokens {
        for l in FIRST_K_DENSE_REPLACE..NUM_HIDDEN_LAYERS {
            let mut picks = [0usize; TOP_K];
            let mut k = 0;
            while k < TOP_K {
                let e = (next() % E as u64) as usize;
                if !picks[..k].contains(&e) {
                    picks[k] = e;
                    k += 1;
                }
            }
            cpu.run(l, &picks, &x, &mut out);
        }
    }
    let (items, ends) = pool.node_ends();
    (0..stream.nodes()).map(|d| items[d] as f64 / (1e3 * ends[d]).max(1e-9f64)).collect()
}
