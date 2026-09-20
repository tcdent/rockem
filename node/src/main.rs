//! rockem node. Construction then execution: report the platform, load the
//! bundle into memory the way the design says (hugetlbfs, first-touched by
//! the thread that will read it), build the model, then serve: on the wire
//! (`wire.rs`: u32 token ids in and out over TCP, nothing transformed) or,
//! without `rockem.listen`, on the console — its text twin: a line of ids in,
//! the continuation out as text, streamed.
//!
//! Configuration is the kernel command line (`rockem.*` tokens) or, on a dev
//! host, the same tokens as program arguments:
//!   rockem.bundle=/bundle      directory with dense.bf16.bin, experts.q4g128.bin, vocab.bin
//!   rockem.mem=anon            dev: anonymous memory instead of hugetlbfs
//!   rockem.gpu=1               the whole model on the card, the router and the draw included: one wait per token
//!   rockem.experts=host        with rockem.gpu=1: the routed experts in this host's DRAM, the card waiting at each layer
//!   rockem.ctx=2048            positions of attention cache on the card or in the CPU model (default 2048): the
//!                              node's memory budget, not the bundle's, so the wire's header reports it
//!   rockem.listen=0.0.0.0:7332 serve the wire on this address; otherwise the console
//!   rockem.max=64              console: tokens to generate per line (the wire runs to EOS or a full context)
//!   rockem.cache=<GB>          host memory for cached contexts (default: half of what is free)
//!   rockem.debug=<dir>         dev: write every position's logits as <dir>/pos<N>.f32 and print the top-2 margin
//!   rockem.shell=1 / rockem.exit=1   dev conveniences (see os/qemu.sh)
mod kfd;
mod loader;
mod platform;
mod wire;

use deepseek_v2_lite::layout;
use deepseek_v2_lite::{Attention, CpuAttention, CpuExperts, CpuFfn, CpuModel, Model, Sampling, State, Weights, BOS_TOKEN, EOS_TOKEN, FIRST_POS, SAMPLING, SEED, TOKENS, VOCAB_SIZE};
use std::io::{BufRead, Read, Write};

fn config(key: &str) -> Option<String> {
    let cmd = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    std::env::args().skip(1).chain(cmd.split_whitespace().map(String::from)).find_map(|t| t.strip_prefix(&format!("{key}=")).map(String::from))
}

/// The console vocabulary: byte offsets then bytes, by id.
struct Vocab {
    offs: Vec<u32>,
    bytes: &'static [u8],
}
impl Vocab {
    fn load(b: &'static [u8]) -> Self {
        let n = deepseek_v2_lite::VOCAB_SIZE + 1;
        let offs = b[..4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Vocab { offs, bytes: &b[4 * n..] }
    }
    fn token(&self, id: usize) -> &[u8] {
        &self.bytes[self.offs[id] as usize..self.offs[id + 1] as usize]
    }
}

/// Dev: every layer's residual output and every routing decision for one
/// position, so two placements can be compared layer by layer.
struct DebugTrace {
    layers: Vec<f32>, // [NUM_HIDDEN_LAYERS][H]
    topk: Vec<u32>,   // [layers][TOP_K], 0 for the dense layer
}
impl DebugTrace {
    fn new() -> Self {
        DebugTrace { layers: vec![0.0; deepseek_v2_lite::NUM_HIDDEN_LAYERS * deepseek_v2_lite::H], topk: vec![0; deepseek_v2_lite::NUM_HIDDEN_LAYERS * deepseek_v2_lite::TOP_K] }
    }
}
impl common::trace::Trace for DebugTrace {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        if layer != common::trace::NONE && name == "out" {
            self.layers[layer * deepseek_v2_lite::H..(layer + 1) * deepseek_v2_lite::H].copy_from_slice(data);
        }
    }
    fn topk(&mut self, layer: usize, idx: &[usize], _weight: &[f32]) {
        for (k, &e) in idx.iter().enumerate() {
            self.topk[layer * deepseek_v2_lite::TOP_K + k] = e as u32;
        }
    }
    fn routed(&mut self, _: usize, _: usize, _: &[f32]) {}
}

/// Cached contexts: the attention cache rows of every transcript this node
/// has computed, in host memory, content-addressed by a running hash of the
/// ids. A context is `[pos][layout::POSITION]` f32 plus its ids; the sink is
/// the one-position context every transcript begins with. Prefix reuse is
/// transparent: as a prompt's ids arrive they are matched against the index
/// and nothing is computed until the first id the store has not seen; the
/// card is then seeded from the longest match (one write through the BAR)
/// and computing resumes there. Eviction is LRU by context, never the one
/// being written, over a budget of positions. Nothing persists.
/// Where a closed context's rows live, `[pos][layout::POSITION]`: DRAM today;
/// NVMe striped over drives the node owns through vfio is the next
/// realization, chosen at construction. The context being written is always
/// the store's one DRAM region (the kernel mirrors into it, the card is
/// seeded from it); a context becomes a `Segment` when the next one starts.
trait Segment {
    /// Positions `0..n`, streamed into `out`.
    fn read(&self, n: usize, out: &mut [f32]);
}
struct Dram(Vec<f32>);
impl Segment for Dram {
    fn read(&self, n: usize, out: &mut [f32]) {
        let len = n * layout::POSITION;
        common::copy::stream(bytes_mut(&mut out[..len]), bytes(&self.0[..len]));
    }
}
fn bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }
}
fn bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, 4 * v.len()) }
}
struct Context {
    ids: Vec<u32>,
    keys: Vec<u64>, // keys[n] names ids[..=n]
    rows: Box<dyn Segment>,
    used: u64,
}
struct Store {
    contexts: Vec<Option<Context>>,
    index: std::collections::HashMap<u64, (usize, usize)>, // prefix key → (context, positions)
    budget: usize, // positions
    held: usize,
    clock: u64,
    max_ctx: usize,
    active: Vec<f32>, // the context being written: max_ctx positions, untouched pages cost nothing
}
impl Store {
    fn key(prev: u64, id: u32) -> u64 {
        id.to_le_bytes().iter().fold(prev, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }
    /// The context being written becomes a segment; its rows are no longer the active region.
    fn close(&mut self, ctx: usize) {
        if let Some(c) = self.contexts[ctx].as_mut() {
            c.rows = Box::new(Dram(std::mem::take(&mut self.active)));
        }
    }
    /// A new context, the active one, holding `n` positions read from `from` (the sink when `from` is none).
    fn open(&mut self, from: Option<(usize, usize)>, sink: &[f32]) -> usize {
        if self.active.is_empty() {
            self.active = vec![0f32; self.max_ctx * layout::POSITION];
        }
        let mut c = Context { ids: Vec::new(), keys: Vec::new(), rows: Box::new(Dram(Vec::new())), used: 0 };
        match from {
            Some((src, n)) => {
                let p = self.contexts[src].as_ref().unwrap();
                c.ids.extend_from_slice(&p.ids[..n]);
                c.keys.extend_from_slice(&p.keys[..n]);
                p.rows.read(n, &mut self.active);
            }
            None => {
                c.ids.push(BOS_TOKEN as u32);
                c.keys.push(Self::key(0xcbf2_9ce4_8422_2325, BOS_TOKEN as u32));
                self.active[..layout::POSITION].copy_from_slice(sink);
            }
        }
        self.held += c.ids.len();
        self.clock += 1;
        c.used = self.clock;
        let slot = self.contexts.iter().position(|c| c.is_none()).unwrap_or_else(|| {
            self.contexts.push(None);
            self.contexts.len() - 1
        });
        self.index.entry(c.keys[0]).or_insert((slot, 1));
        self.contexts[slot] = Some(c);
        slot
    }
    /// Record position `n` of `ctx` (its id, key and rows are in place) and stay under budget.
    fn grew(&mut self, ctx: usize, n: usize) {
        let key = self.contexts[ctx].as_ref().unwrap().keys[n];
        self.index.entry(key).or_insert((ctx, n + 1));
        self.held += 1;
        while self.held > self.budget {
            let Some(victim) = self.contexts.iter().enumerate().filter(|(i, c)| *i != ctx && c.is_some()).min_by_key(|(_, c)| c.as_ref().unwrap().used).map(|(i, _)| i) else { break };
            let c = self.contexts[victim].take().unwrap();
            for (n, k) in c.keys.iter().enumerate() {
                if self.index.get(k) == Some(&(victim, n + 1)) {
                    self.index.remove(k);
                }
            }
            self.held -= c.ids.len();
        }
    }
    fn touch(&mut self, ctx: usize) {
        self.clock += 1;
        self.contexts[ctx].as_mut().unwrap().used = self.clock;
    }
}

/// One sequence at a time on this node: the model, its cache, and the
/// sampler, driven by the same three moves from the wire or the console.
struct Session<'a> {
    w: Weights<'a>,
    model: Box<dyn Model>,
    st: State,
    sink: &'a [f32],
    /// The logits at the sink, computed once at start: what scores or draws the first token after BOS.
    sink_logits: Vec<f32>,
    /// SCORE was received on this sequence: each id fed is scored first.
    scoring: bool,
    /// The row of the last step's logits that belongs to its last token.
    last_row: usize,
    next: u32, // the draw made at the last computed position
    pos: usize,
    max_ctx: usize,
    started: bool,
    debug: Option<std::path::PathBuf>,
    dtr: DebugTrace,
    store: Store,
    active: usize,
    /// While the prompt is a prefix the store holds: (context, positions matched); nothing computed yet.
    pending: Option<(usize, usize)>,
    hits: usize,
}
impl Session<'_> {
    /// BOS: a new context. Whether the card is seeded from the sink or from a
    /// longer cached prefix is decided by the ids that follow.
    fn start(&mut self) {
        // the active region belongs to the active context whenever it is allocated; a new
        // connection resets `started` without closing, so the region, not the flag, decides
        if !self.store.active.is_empty() {
            self.store.close(self.active);
        }
        self.active = self.store.open(None, self.sink);
        // position 0 is the sink until a longer cached prefix replaces it; seeding here is what
        // makes a sequence that matches nothing identical to one that matches everything
        self.model.seed(&self.store.active[..layout::POSITION]);
        // the longest cached prefix so far is the sink itself, in some closed context — never this one
        self.pending = self.store.index.get(&self.store.contexts[self.active].as_ref().unwrap().keys[0]).copied().filter(|&(c, _)| c != self.active);
        self.pos = FIRST_POS;
        self.started = true;
        self.scoring = false;
        self.hits = 0;
    }
    /// Leave the store's matching state with the logits current for the next
    /// position: nothing to do if only BOS matched (the sink's logits stand);
    /// otherwise the matched prefix but its last position is installed and that
    /// position is recomputed for its logits.
    fn settle(&mut self) -> bool {
        match self.pending {
            Some((_, 1)) => {
                self.pending = None;
                true
            }
            Some((ctx, n)) => {
                let last = self.store.contexts[ctx].as_ref().unwrap().ids[n - 1];
                self.materialize(n - 1);
                self.feed(last)
            }
            None => true,
        }
    }
    /// ln p(id) at the next position, before `id` is fed there: from the last
    /// computed token's logits, the sink's if none was.
    fn score(&mut self, id: u32) -> f32 {
        if self.pos == FIRST_POS {
            common::softmax::log_prob(&self.sink_logits, id as usize)
        } else {
            let r = self.last_row;
            self.model.score(r, id as usize, &self.st.logits[r * VOCAB_SIZE..(r + 1) * VOCAB_SIZE])
        }
    }
    /// Leave the matching state: the active context becomes a copy of the
    /// matched prefix's first `n` positions, the card is seeded from it, and
    /// computing resumes at `n`.
    fn materialize(&mut self, n: usize) {
        let (src, matched) = self.pending.take().unwrap();
        let n = n.min(matched);
        self.store.touch(src);
        let old = self.active;
        self.active = self.store.open(Some((src, n)), self.sink);
        self.store.held -= self.store.contexts[old].take().unwrap().ids.len();
        self.model.seed(&self.store.active[..n * layout::POSITION]);
        self.pos = n;
        self.hits = n - 1;
    }
    /// `ids` through the model at the next positions, as few steps as they
    /// allow (TOKENS per step): ids still matching the store advance the
    /// match and cost nothing; from the first that does not, the rest are
    /// computed in steps. In scoring mode every id's ln p is given to `emit`
    /// before it is fed, in order. False when the context is full.
    fn feed_many(&mut self, ids: &[u32], emit: &mut dyn FnMut(f32)) -> bool {
        let mut at = 0;
        while at < ids.len() {
            let id = ids[at];
            if let Some((ctx, n)) = self.pending {
                let key = Store::key(self.store.contexts[ctx].as_ref().unwrap().keys[n - 1], id);
                if let Some(&(c2, n2)) = self.store.index.get(&key) {
                    if c2 != self.active && self.store.contexts[c2].as_ref().map(|c| c.ids.get(n2 - 1) == Some(&id)).unwrap_or(false) && n2 == n + 1 && n2 < self.max_ctx {
                        self.pending = Some((c2, n2));
                        at += 1;
                        continue;
                    }
                }
                self.materialize(n);
            }
            // the rest of the ids are computed: a step of up to TOKENS at a time
            let n = (ids.len() - at).min(TOKENS).min(self.max_ctx.saturating_sub(self.pos));
            if n == 0 {
                return false;
            }
            if self.scoring {
                let lp = self.score(ids[at]);
                emit(lp);
            }
            self.step(&ids[at..at + n]);
            if self.scoring {
                for j in 1..n {
                    let lp = self.model.score(j - 1, ids[at + j] as usize, &self.st.logits[(j - 1) * VOCAB_SIZE..j * VOCAB_SIZE]);
                    emit(lp);
                }
            }
            at += n;
        }
        true
    }
    /// One step: `toks` at positions `pos..pos + n`, their rows into the store,
    /// the draw at the last, the trace of each under debug.
    fn step(&mut self, toks: &[u32]) {
        let s = self;
        let n = toks.len();
        let pos = s.pos;
        if s.debug.is_some() {
            s.st.run(&s.w, toks, pos, &mut s.model, &mut s.dtr);
        } else {
            s.st.run(&s.w, toks, pos, &mut s.model, &mut common::trace::NoTrace);
        }
        // the draw at a position is a function of the seed and the position alone, so a
        // sequence draws the same whether its prefix was computed or served from the store
        let u = common::sample::uniform_at(SEED, (pos + n - 1) as u64);
        s.next = s.model.draw(n - 1, &SAMPLING, u, &s.st.logits[(n - 1) * VOCAB_SIZE..n * VOCAB_SIZE]) as u32;
        s.last_row = n - 1;
        for i in 0..n {
            let p = pos + i;
            s.model.rows(i, &mut s.store.active[p * layout::POSITION..(p + 1) * layout::POSITION]);
            let c = s.store.contexts[s.active].as_mut().unwrap();
            c.ids.push(toks[i]);
            let key = Store::key(c.keys[p - 1], toks[i]);
            c.keys.push(key);
            s.store.grew(s.active, p);
            if let Some(d) = &s.debug {
                s.model.token_trace(i, &mut s.dtr.layers, &mut s.dtr.topk, &mut s.st.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE]);
                let raw = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }.to_vec();
                std::fs::write(d.join(format!("pos{p}.layers.f32")), raw(&s.dtr.layers)).unwrap();
                std::fs::write(d.join(format!("pos{p}.topk.u32")), unsafe { std::slice::from_raw_parts(s.dtr.topk.as_ptr() as *const u8, 4 * s.dtr.topk.len()) }).unwrap();
                let lg = &s.st.logits[i * VOCAB_SIZE..(i + 1) * VOCAB_SIZE];
                let mut ix = (0..lg.len()).collect::<Vec<_>>();
                ix.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
                eprintln!("  pos {p}: argmax {} margin {:.4e} (top2 {})", ix[0], lg[ix[0]] - lg[ix[1]], ix[1]);
                std::fs::write(d.join(format!("pos{p}.f32")), raw(lg)).unwrap();
            }
        }
        s.pos = pos + n;
    }
    /// One token through the model at the next position. False when the context is full.
    fn feed(&mut self, id: u32) -> bool {
        self.feed_many(&[id], &mut |_| {})
    }
    /// GENERATE: sample, emit, feed, until EOS or `limit` tokens. Every emitted
    /// id goes through the model, so the sequence can continue from it. False
    /// when the context filled or `emit` refused.
    fn generate(&mut self, limit: usize, mut emit: impl FnMut(u32) -> bool) -> bool {
        if !self.settle() {
            return false;
        }
        if self.pos == FIRST_POS {
            // nothing fed after BOS: the first draw is from the sink's logits
            self.next = self.model.draw(0, &SAMPLING, common::sample::uniform_at(SEED, 0), &self.sink_logits) as u32;
        }
        for _ in 0..limit {
            let id = self.next;
            if !emit(id) || !self.feed(id) {
                return false;
            }
            if id as usize == EOS_TOKEN {
                break;
            }
        }
        true
    }
}

/// The wire, one connection at a time (one cache).
fn serve(listen: &str, session: &mut Session, identity: u64) {
    let listener = std::net::TcpListener::bind(listen).unwrap();
    println!("ready    : wire on {listen}, identity {identity:016x}");
    for conn in listener.incoming() {
        let Ok(mut s) = conn else { continue };
        s.set_nodelay(true).unwrap();
        let mut h = [0u8; wire::HELLO];
        if s.read_exact(&mut h).is_err() || !matches!(wire::accept(&h), Some(want) if want == 0 || want == identity) || s.write_all(&wire::header(identity, (session.max_ctx - FIRST_POS) as u32)).is_err() {
            continue;
        }
        session.started = false;
        let mut word = [0u8; 4];
        let mut ids: Vec<u32> = Vec::new();
        // ids are fed as they arrive, in whatever steps arrive: a prompt that is already in the
        // socket's buffer goes through the model TOKENS at a time; a stream of single ids one at a time
        let flush = |session: &mut Session, ids: &mut Vec<u32>, s: &mut std::net::TcpStream| -> bool {
            if ids.is_empty() {
                return true;
            }
            let mut ok = true;
            let done = session.started && session.feed_many(ids, &mut |lp| ok &= s.write_all(&lp.to_bits().to_le_bytes()).is_ok());
            ids.clear();
            done && ok
        };
        let more_now = |s: &std::net::TcpStream| -> bool {
            // four more bytes already here? (peeked, never consumed, so nothing is lost)
            let mut peek = [0u8; 4];
            s.set_nonblocking(true).unwrap();
            let r = matches!(s.peek(&mut peek), Ok(4));
            s.set_nonblocking(false).unwrap();
            r
        };
        while s.read_exact(&mut word).is_ok() {
            let ok = match u32::from_le_bytes(word) {
                id if id as usize == BOS_TOKEN => {
                    let ok = flush(session, &mut ids, &mut s);
                    session.start();
                    ok
                }
                wire::GENERATE => flush(session, &mut ids, &mut s) && session.started && session.generate(session.max_ctx, |id| s.write_all(&id.to_le_bytes()).is_ok()),
                wire::SCORE => {
                    let ok = flush(session, &mut ids, &mut s);
                    session.scoring = session.started;
                    ok && session.started && session.settle()
                }
                id if (id as usize) < VOCAB_SIZE => {
                    if !session.started {
                        false
                    } else {
                        ids.push(id);
                        if ids.len() >= TOKENS || !more_now(&s) {
                            flush(session, &mut ids, &mut s)
                        } else {
                            true
                        }
                    }
                }
                _ => false,
            };
            if !ok {
                break;
            }
        }
    }
}

fn main() {
    platform::report();
    let cmdline = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    if cmdline.split_whitespace().any(|t| t == "rockem.shell=1") {
        println!("-- dev: handing over to /bin/sh --");
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new("/bin/sh").exec();
        eprintln!("exec /bin/sh failed: {err}");
    }
    let Some(bundle) = config("rockem.bundle") else {
        if cmdline.split_whitespace().any(|t| t == "rockem.exit=1") {
            println!("-- rockem.exit=1: leaving --");
            std::process::exit(0);
        }
        println!("-- no rockem.bundle=<dir>; nothing to serve; halting --");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(3600));
        }
    };

    // --- construction ---
    let mem = if config("rockem.mem").as_deref() == Some("anon") { loader::Mem::Anon } else { loader::Mem::Hugetlb };
    let t0 = std::time::Instant::now();
    let bundle = std::path::Path::new(&bundle);
    let dense_b = loader::load(&bundle.join(layout::DENSE_FILE), mem);
    let experts = loader::load(&bundle.join(layout::EXPERTS_FILE), mem);
    let vocab_b = loader::load(&bundle.join(layout::VOCAB_FILE), loader::Mem::Anon);
    println!("bundle   : {} loaded in {:.1}s ({:.1} GB)", bundle.display(), t0.elapsed().as_secs_f32(), (dense_b.len() + experts.len()) as f64 / 1e9);
    let (p, dense, q) = unsafe { dense_b.align_to::<u16>() };
    assert!(p.is_empty() && q.is_empty(), "dense file alignment");
    let w = Weights::from_dense(dense);
    let vocab = Vocab::load(vocab_b);
    let max_ctx: usize = config("rockem.ctx").and_then(|v| v.parse().ok()).unwrap_or(2048);
    assert!(max_ctx > FIRST_POS, "rockem.ctx must leave room after BOS");
    let mut st = State::new(max_ctx);
    let mut model: Box<dyn Model> = if config("rockem.gpu").is_some() {
        let k = kfd::Kfd::new(loader::load(&bundle.join(layout::KERNELS_FILE), loader::Mem::Anon), false);
        // the placement table: every routed expert on the card, or every one in this host's
        // DRAM (rockem.experts=host) — the manifest's table replaces this once there is one
        let place = vec![if config("rockem.experts").as_deref() == Some("host") { 1u32 } else { 0 }; deepseek_v2_lite::N_ROUTED_EXPERTS];
        Box::new(kfd::KfdModel::new(k, dense, &experts[..layout::STREAM], &st.rope, max_ctx, &place))
    } else {
        Box::new(CpuModel { attention: CpuAttention::new(max_ctx), ffn: CpuFfn::new(CpuExperts::new(&experts[..layout::STREAM])) })
    };
    // the sink token's state: every sequence starts from it at FIRST_POS
    let sink_b = loader::load(&bundle.join(layout::SINK_FILE), loader::Mem::Anon);
    let (p, sink, q) = unsafe { sink_b.align_to::<f32>() };
    assert!(p.is_empty() && q.is_empty() && sink.len() == layout::SINK_LEN, "sink file");
    let debug = config("rockem.debug").map(std::path::PathBuf::from);
    if let Some(d) = &debug {
        std::fs::create_dir_all(d).unwrap();
    }
    // the context store: a budget in positions, by default half of what the box reports free —
    // an assumed mean between the store and everything else the node holds; `rockem.cache` sets it
    let avail_kb: usize = std::fs::read_to_string("/proc/meminfo").unwrap_or_default().lines().find_map(|l| l.strip_prefix("MemAvailable:")).and_then(|v| v.trim().split(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let budget_bytes = config("rockem.cache").and_then(|v| v.parse::<usize>().ok()).map(|gb| gb << 30).unwrap_or(avail_kb << 9);
    let store = Store { contexts: Vec::new(), index: std::collections::HashMap::new(), budget: (budget_bytes / (layout::POSITION * 4)).max(max_ctx), held: 0, clock: 0, max_ctx, active: Vec::new() };
    println!("cache    : {} cached positions ({:.1} GB of host memory), {} KB each", store.budget, budget_bytes as f64 / 1e9, layout::POSITION * 4 / 1024);
    // the sink's logits: BOS through the model once, at position 0, before anything is served
    let mut sink_logits = {
        st.decode(&w, BOS_TOKEN as u32, 0, &mut model, &Sampling::GREEDY, 0.0, &mut common::trace::NoTrace);
        st.logits[..VOCAB_SIZE].to_vec()
    };
    model.token_trace(0, &mut vec![0f32; deepseek_v2_lite::NUM_HIDDEN_LAYERS * deepseek_v2_lite::H], &mut vec![0u32; deepseek_v2_lite::NUM_HIDDEN_LAYERS * deepseek_v2_lite::TOP_K], &mut sink_logits);
    let mut session = Session { w, model, st, sink, sink_logits, scoring: false, last_row: 0, next: 0, pos: FIRST_POS, max_ctx, started: false, debug, dtr: DebugTrace::new(), store, active: 0, pending: None, hits: 0 };
    println!("model    : deepseek-v2-lite, ctx {max_ctx}, temperature {}, top-k {}, top-p {}", SAMPLING.temperature, SAMPLING.top_k, SAMPLING.top_p);

    // --- execution ---
    if let Some(listen) = config("rockem.listen") {
        let identity = wire::identity(&std::fs::read(bundle.join("manifest.toml")).unwrap_or_default());
        serve(&listen, &mut session, identity);
        return;
    }
    // the console: one line is a prompt, its newline is GENERATE
    let max_new: usize = config("rockem.max").and_then(|v| v.parse().ok()).unwrap_or(64);
    println!("ready    : type token ids, e.g.  100000 549 6077 280 7239 317  ({max_new} tokens per line)");
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let ids: Vec<u32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if ids.is_empty() {
            continue;
        }
        let t1 = std::time::Instant::now();
        // the line's ids after BOS go through the model in steps, as the wire's do
        if ids[0] as usize != BOS_TOKEN || ids.iter().skip(1).any(|&id| id as usize == BOS_TOKEN) {
            println!("[a prompt is one BOS ({BOS_TOKEN}) then ids: position 0 is the precomputed sink]");
            continue;
        }
        session.start();
        if !session.feed_many(&ids[1..], &mut |_| {}) {
            println!("[the context is full]");
            continue;
        }
        for &id in &ids {
            out.write_all(vocab.token(id as usize)).unwrap();
        }
        if !session.started {
            continue;
        }
        let t_prefill = t1.elapsed();
        let t2 = std::time::Instant::now();
        let mut n = 0;
        session.generate(max_new, |id| {
            n += 1;
            out.write_all(vocab.token(id as usize)).unwrap();
            out.flush().is_ok()
        });
        let t_gen = t2.elapsed();
        println!();
        println!("[{} prompt tokens in {:.2}s ({} cached), {n} generated at {:.2} tok/s]", ids.len(), t_prefill.as_secs_f32(), session.hits, n as f32 / t_gen.as_secs_f32());
    }
}
