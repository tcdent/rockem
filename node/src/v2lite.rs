//! DeepSeek-V2-Lite as the node runs it: the bundle loaded into hugetlbfs, the
//! model on the card (`rockem.gpu=1`, the routed experts on the card or in
//! this host's DRAM) or on this CPU, behind `Engine`.

use crate::engine::{Engine, Sampling};
use crate::{kfd, loader};
use common::trace::{NoTrace, Trace, NONE};
use deepseek_v2_lite::layout;
use deepseek_v2_lite::{CpuAttention, CpuExperts, CpuFfn, CpuModel, Model, State, Weights, BOS_TOKEN, EOS_TOKEN, FIRST_POS, H, NUM_HIDDEN_LAYERS, SAMPLING, SEED, TOKENS, TOP_K, VOCAB_SIZE};

/// Every layer's residual output and every routing decision for one position.
struct DebugTrace {
    layers: Vec<f32>, // [NUM_HIDDEN_LAYERS][H]
    topk: Vec<u32>,   // [NUM_HIDDEN_LAYERS][TOP_K], 0 for the dense layer
}
impl Trace for DebugTrace {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        if layer != NONE && name == "out" {
            self.layers[layer * H..(layer + 1) * H].copy_from_slice(data);
        }
    }
    fn topk(&mut self, layer: usize, idx: &[usize], _weight: &[f32]) {
        for (k, &e) in idx.iter().enumerate() {
            self.topk[layer * TOP_K + k] = e as u32;
        }
    }
    fn routed(&mut self, _: usize, _: usize, _: &[f32]) {}
}

pub struct V2Lite {
    w: Weights<'static>,
    model: Box<dyn Model>,
    st: State,
    /// The sink: BOS's cache rows, the bundle's `sink.f32`.
    prefix_rows: &'static [f32],
    /// BOS's logits, computed once at start: what scores or draws the first token after it.
    prefix_logits: Vec<f32>,
    dtr: DebugTrace,
}

const PREFIX: [u32; FIRST_POS] = [BOS_TOKEN as u32];

impl V2Lite {
    /// `gpu`: the whole model on the card through KFD; `experts_host`: with it, the routed experts in this host's DRAM.
    pub fn open(bundle: &std::path::Path, mem: loader::Mem, max_ctx: usize, gpu: bool, experts_host: bool) -> Box<dyn Engine> {
        let t0 = std::time::Instant::now();
        let dense_b = loader::load(&bundle.join(layout::DENSE_FILE), mem);
        let experts = loader::load(&bundle.join(layout::EXPERTS_FILE), mem);
        println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense_b.len() + experts.len()) as f64 / 1e9);
        let (p, dense, q) = unsafe { dense_b.align_to::<u16>() };
        assert!(p.is_empty() && q.is_empty(), "dense file alignment");
        let w = Weights::from_dense(dense);
        let st = State::new(max_ctx);
        let model: Box<dyn Model> = if gpu {
            let k = kfd::Kfd::new(loader::load(&bundle.join(layout::KERNELS_FILE), loader::Mem::Anon), false);
            // the placement table: every routed expert on the card, or every one in this host's
            // DRAM — the manifest's table replaces this once there is one
            let place = vec![if experts_host { 1u32 } else { 0 }; deepseek_v2_lite::N_ROUTED_EXPERTS];
            Box::new(kfd::KfdModel::new(k, dense, &experts[..layout::STREAM], &st.rope, max_ctx, &place))
        } else {
            Box::new(CpuModel { attention: CpuAttention::new(max_ctx), ffn: CpuFfn::new(CpuExperts::new(&experts[..layout::STREAM])) })
        };
        // the sink token's state: every sequence starts from it at FIRST_POS
        let sink_b = loader::load(&bundle.join(layout::SINK_FILE), loader::Mem::Anon);
        let (p, sink, q) = unsafe { sink_b.align_to::<f32>() };
        assert!(p.is_empty() && q.is_empty() && sink.len() == layout::SINK_LEN, "sink file");
        let mut e = V2Lite { w, model, st, prefix_rows: sink, prefix_logits: Vec::new(), dtr: DebugTrace { layers: vec![0.0; NUM_HIDDEN_LAYERS * H], topk: vec![0; NUM_HIDDEN_LAYERS * TOP_K] } };
        // the sink's logits: BOS through the model once, at position 0, before anything is served
        e.st.decode(&e.w, BOS_TOKEN as u32, 0, &mut e.model, &Sampling::GREEDY, 0.0, &mut NoTrace);
        e.prefix_logits = e.trace(0).2.to_vec();
        println!("model    : deepseek-v2-lite, ctx {max_ctx}, temperature {}, top-k {}, top-p {}", SAMPLING.temperature, SAMPLING.top_k, SAMPLING.top_p);
        Box::new(e)
    }
}

impl Engine for V2Lite {
    fn name(&self) -> &'static str {
        "deepseek-v2-lite"
    }
    fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }
    fn is_eos(&self, id: u32) -> bool {
        id as usize == EOS_TOKEN
    }
    fn tokens(&self) -> usize {
        TOKENS
    }
    fn prefix_ids(&self) -> &[u32] {
        &PREFIX
    }
    fn position(&self) -> usize {
        layout::POSITION
    }
    fn snapshot_len(&self) -> usize {
        0
    }
    fn seed_value(&self) -> u64 {
        SEED
    }
    fn prefix(&self) -> (&[f32], &[f32]) {
        (self.prefix_rows, &[])
    }
    fn seed_prefix(&mut self) {
        self.model.seed(self.prefix_rows)
    }
    fn seed(&mut self, rows: &[f32], _snap: &[f32]) {
        self.model.seed(rows)
    }
    fn snapshot(&self, _out: &mut [f32]) {}
    fn run(&mut self, toks: &[u32], pos: usize, trace: bool) {
        if trace {
            self.st.run(&self.w, toks, pos, &mut self.model, &mut self.dtr)
        } else {
            self.st.run(&self.w, toks, pos, &mut self.model, &mut NoTrace)
        }
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        self.model.rows(i, out)
    }
    fn draw(&mut self, i: usize, u: f64) -> u32 {
        self.model.draw(i, &SAMPLING, u, &self.st.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE]) as u32
    }
    fn score(&mut self, i: usize, id: u32) -> f32 {
        self.model.score(i, id as usize, &self.st.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE])
    }
    /// On the host, from the logits kept at start: a tier that draws on the card
    /// draws from the last step's row there, which is the sink's only until the
    /// first sequence runs.
    fn draw_prefix(&mut self, u: f64) -> u32 {
        common::sample::draw(&self.prefix_logits, &SAMPLING, u) as u32
    }
    fn score_prefix(&mut self, id: u32) -> f32 {
        common::softmax::log_prob(&self.prefix_logits, id as usize)
    }
    fn trace(&mut self, i: usize) -> (&[f32], &[u32], &[f32]) {
        let lg = &mut self.st.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
        self.model.token_trace(i, &mut self.dtr.layers, &mut self.dtr.topk, lg);
        (&self.dtr.layers, &self.dtr.topk, lg)
    }
}
