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
use glm_5_3_flash::{EOS_TOKENS, FIRST_POS, H, NUM_HIDDEN_LAYERS, PREFIX_IDS, SAMPLING, SEED, TOKENS, TOP_K, VOCAB_SIZE};

/// Every layer's residual output and routing for each token of a step.
struct DebugTrace {
    tok: usize,
    layers: Vec<f32>, // [TOKENS][NUM_HIDDEN_LAYERS][H]
    topk: Vec<u32>,   // [TOKENS][NUM_HIDDEN_LAYERS][TOP_K], 0 for the dense layers
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
        // the experts placed by the box's own measurement: each node's share of every expert in
        // proportion to its stream rate, so a layer's expert run ends on every node at once
        let rates = crate::probe::node_rates();
        let total: f64 = rates.iter().sum();
        println!("placement: node streams {} GB/s → item shares {} %", rates.iter().map(|r| format!("{r:.0}")).collect::<Vec<_>>().join(" / "), rates.iter().map(|r| format!("{:.1}", 100.0 * r / total)).collect::<Vec<_>>().join(" / "));
        let experts: &'static glm_5_3_flash::expert::Stream<'static> = Box::leak(Box::new(loader::load_placed(&bundle.join(layout::EXPERTS_FILE), mem, &layout::placement(&rates))));
        println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense.len() + experts.bytes.len()) as f64 / 1e9);
        let w = Weights::from_dense(dense);
        let prefix_b = loader::load(&bundle.join(layout::PREFIX_FILE), loader::Mem::Anon);
        let (p, prefix, q) = unsafe { prefix_b.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty() && prefix.len() == layout::PREFIX_LEN, "prefix file");
        // the pool: a thread per physical core, every socket, told where every page of the weights lives
        let cores = physical_cores();
        let pool: &'static Threads = Box::leak(Box::new(Threads::new(&cores)));
        let mut by_node = [0usize; 8];
        for buf in [dense, experts.bytes] {
            let nodes = loader::placement(buf);
            for &d in &nodes {
                by_node[d as usize] += 1;
            }
            pool.place(buf.as_ptr() as usize, buf.len(), nodes);
        }
        let used: Vec<String> = by_node.iter().filter(|&&n| n > 0).map(|&n| format!("{:.1}", n as f64 * 2.0 / 1024.0)).collect();
        println!("threads  : {} pinned, one per physical core; the weights' 2 MiB pages by node: {} GB", cores.len(), used.join(" / "));
        let mut model = CpuModel::new(CpuExperts::new(experts, pool), pool, max_ctx);
        if gpu {
            // the KDA layers on the cards, split by layer order between them; everything else here
            let elf = loader::load(&bundle.join(layout::KERNELS_FILE), loader::Mem::Anon);
            model.placed = Some(Box::new(crate::glm_kfd::Cards::new(elf, &w, max_ctx, timeline)));
        }
        // the prefix from nothing: its logits, and the realization against the bundle's state
        let t1 = std::time::Instant::now();
        for (pos, &id) in PREFIX_IDS.iter().enumerate() {
            model.step(&w, id, pos, None, &mut NoTrace);
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
            profile_every,
            w,
            model,
            prefix,
            prefix_logits,
            logits: vec![0.0; TOKENS * VOCAB_SIZE],
            last_pos: 0,
            dtr: DebugTrace { tok: 0, layers: vec![0.0; TOKENS * NUM_HIDDEN_LAYERS * H], topk: vec![0; TOKENS * NUM_HIDDEN_LAYERS * TOP_K] },
        })
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
        TOKENS
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
    fn run(&mut self, toks: &[u32], pos: usize, trace: bool) {
        assert!(!toks.is_empty() && toks.len() <= TOKENS);
        self.last_pos = pos;
        for (i, &tok) in toks.iter().enumerate() {
            if trace {
                self.dtr.tok = i;
                self.model.step(&self.w, tok, pos + i, None, &mut self.dtr);
            } else {
                self.model.step(&self.w, tok, pos + i, None, &mut NoTrace);
            }
            self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE].copy_from_slice(&self.model.logits);
        }
        if self.profile_every > 0 && self.model.profile.tokens >= self.profile_every {
            println!("profile  : {}", self.model.profile.line());
            let c = self.model.coverage(&[8, 16, 32, 64]);
            println!("experts  : since start, a layer's 8 / 16 / 32 / 64 most-picked experts took {:.0} / {:.0} / {:.0} / {:.0} % of its picks", 100.0 * c[0], 100.0 * c[1], 100.0 * c[2], 100.0 * c[3]);
            self.model.profile = Default::default();
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
        self.pool.idle()
    }
    fn trace(&mut self, i: usize) -> (&[f32], &[u32], &[f32]) {
        let a = i * NUM_HIDDEN_LAYERS;
        (&self.dtr.layers[a * H..(a + NUM_HIDDEN_LAYERS) * H], &self.dtr.topk[a * TOP_K..(a + NUM_HIDDEN_LAYERS) * TOP_K], &self.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE])
    }
}

const _: () = assert!(Sampling::GREEDY.top_k == 1);
