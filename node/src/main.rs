//! rockem node. Construction then execution: report the platform, open the
//! bundle — its manifest lists its models (`[[model]]`, each one's files in a
//! directory named for it), and each model's engine loads it into memory the
//! way the design says (hugetlbfs, first-touched by the thread that will read
//! it) and builds itself, every model resident — then serve: on the wire
//! (`wire.rs`: u32 token ids in and out over TCP, nothing transformed; the
//! hello names the model a connection speaks, one connection at a time across
//! all of them) or, without `rockem.listen`, on the console — its text twin for
//! one model: a line of ids in, the continuation out as text, streamed. One
//! binary carries every model (`engine.rs`).
//!
//! Configuration is the kernel command line (`rockem.*` tokens) or, on a dev
//! host, the same tokens as program arguments:
//!   rockem.bundle=/bundle      a bundle directory (docs/bundle.md); `manifest.toml` lists its models
//!   rockem.model=<name>        console: the model to serve, when the bundle has more than one
//!   rockem.mem=anon            dev: anonymous memory instead of hugetlbfs
//!   rockem.gpu=1               V2-Lite: the whole model on the card, the router and the draw included: one wait per token;
//!                              GLM: the recurrent-attention layers on the cards, split by layer between them
//!   rockem.experts=host        with rockem.gpu=1: the routed experts in this host's DRAM, the card waiting at each layer
//!   rockem.ctx=2048            positions of attention cache on the card or in the CPU model (default 2048): the
//!                              node's memory budget, not the bundle's, so the wire's header reports it
//!   rockem.listen=0.0.0.0:7332 serve the wire on this address; otherwise the console
//!   rockem.max=64              console: tokens to generate per line (the wire runs to EOS or a full context)
//!   rockem.cache=<GB>          host memory for cached contexts (default: half of what is free)
//!   rockem.debug=<dir>         dev: write every position's logits as <dir>/pos<N>.f32 and print the top-2 margin
//!   rockem.resident=<GB>       GLM, with rockem.gpu: VRAM a card may spend on resident routed experts (default: what is free past a reserve, up to 12); 0 leaves the cards to a gate or bench beside the node
//!   rockem.prow=<host:port>    the prow server the run's rows go to (default PROW_SERVER below)
//!   rockem.probe=1             measure this box (DRAM, GEMV rates, each card's VRAM, link and BAR) with
//!                              the bundle's code object if there is one, report, and idle: the baseline
//!   rockem.shell=1 / rockem.exit=1   dev conveniences (see os/qemu.sh)
//!
//! Measurements are prow rows, not console lines: the categories compiled in (`node/Cargo.toml`)
//! decide what is recorded, and the console is the server's log — what it runs, and what went wrong.
mod engine;
mod glm;
mod glm_kfd;
mod hsmp;
mod kfd;
mod loader;
mod platform;
mod probe;
mod v2lite;
mod v2lite_kfd;
mod voxtral;
mod wire;

use engine::Engine;
use std::io::{BufRead, Read, Write};

/// The kernel command line, then the program's arguments; the last occurrence of a key
/// wins, as for the kernel's own parameters — so a netboot line that repeats ours overrides
/// the image's, and an argument given by hand overrides both (a node run from a shell on a
/// booted box listens where it is told, not where the image's node does).
/// Where the run's rows go: prow-server's ingest on elsenor (`rockem.prow` overrides).
const PROW_SERVER: &str = "192.168.88.3:7380";
/// The prow sender's core: the SMT sibling of physical core 72, a housekeeping core (`dev.cmdline`'s
/// isolation leaves 0/24/48/72 and their siblings to the kernel). The pool holds one thread a physical
/// core, the main thread core 0 and the expert loader core 24; the sender wakes every 100 ms.
const PROW_CORE: usize = 168;

fn config(key: &str) -> Option<String> {
    let cmd = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    cmd.split_whitespace().map(String::from).chain(std::env::args().skip(1)).filter_map(|t| t.strip_prefix(&format!("{key}=")).map(String::from)).last()
}

/// The `name = "…"` of every `[[model]]` entry in a manifest, in order.
fn model_names(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_model = false;
    for line in manifest.lines() {
        let l = line.trim();
        if l.starts_with('[') {
            in_model = l == "[[model]]";
        } else if in_model {
            if let Some(v) = l.strip_prefix("name") {
                if let Some(n) = v.trim().strip_prefix('=').map(|v| v.trim().trim_matches('"').to_string()) {
                    names.push(n);
                }
                in_model = false;
            }
        }
    }
    names
}

/// The console vocabulary: byte offsets then bytes, by id.
struct Vocab {
    offs: Vec<u32>,
    bytes: &'static [u8],
}
impl Vocab {
    fn load(b: &'static [u8], vocab_size: usize) -> Self {
        let n = vocab_size + 1;
        let offs = b[..4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Vocab { offs, bytes: &b[4 * n..] }
    }
    fn token(&self, id: usize) -> &[u8] {
        &self.bytes[self.offs[id] as usize..self.offs[id + 1] as usize]
    }
}

/// Cached contexts: the per-position rows of every transcript this node has
/// computed, in host memory, content-addressed by a running hash of the ids,
/// plus — for a model with sequence state that is not per position — the
/// snapshot of that state at the transcript's end. A context is
/// `[pos][position]` f32 plus its ids; the prefix is the `first_pos`-position
/// context every transcript begins with. Prefix reuse is transparent: as a
/// prompt's ids arrive they are matched against the index and nothing is
/// computed until the first id the store has not seen; the model is then
/// seeded from the longest match and computing resumes there. A model with
/// snapshots can resume only at a cached transcript's end, where its snapshot
/// was taken; a partial match of such a transcript resumes from the prefix and
/// recomputes the matched ids. Eviction is LRU by context, never the one being
/// written, over a budget of positions. Nothing persists.
/// Where a closed context's rows live, `[pos][position]`: DRAM today; NVMe
/// striped over drives the node owns through vfio is the next realization,
/// chosen at construction. The context being written is always the store's
/// one DRAM region (the kernel mirrors into it, the card is seeded from it); a
/// context becomes a `Segment` when the next one starts.
trait Segment {
    /// The first `len` f32s, streamed into `out`.
    fn read(&self, len: usize, out: &mut [f32]);
}
struct Dram(Vec<f32>);
impl Segment for Dram {
    fn read(&self, len: usize, out: &mut [f32]) {
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
    /// The sequence state at the context's end, `snapshot_len` f32s (empty while the context is active),
    /// and the logits there.
    snap: Vec<f32>,
    snap_logits: Vec<f32>,
    /// The sequence state at other positions — at the end of every prompt generation started from,
    /// and at a reply's end — `(positions, state, the logits there)`, so the next turn of a chat (the
    /// same prompt, the reply as the API re-renders it, the new message) resumes where its prompt
    /// matched, and the same prompt again draws at once (2026-10-16).
    marks: Vec<(usize, Vec<f32>, Vec<f32>)>,
    used: u64,
}
/// Marks a transcript keeps (the newest): a mark is a whole sequence state (~150 MB for GLM-5.3-Flash);
/// a long chat would keep one a turn, and only the recent ones are ever resumed from.
const MARKS_KEPT: usize = 8;
struct Store {
    contexts: Vec<Option<Context>>,
    index: std::collections::HashMap<u64, (usize, usize)>, // prefix key → (context, positions)
    budget: usize, // positions
    /// What one kept sequence state (a mark, or a closed context's snapshot, with its logits) costs in
    /// positions of rows: the budget counts both, so a thousand short transcripts cannot keep a thousand
    /// states (the box went out of memory on a GSM8K run, 2026-10-18: 913 contexts × 153 MB).
    state_cost: usize,
    held: usize,
    clock: u64,
    max_ctx: usize,
    position: usize,
    active: Vec<f32>, // the context being written: max_ctx positions, untouched pages cost nothing
}
impl Store {
    fn key(prev: u64, id: u32) -> u64 {
        Self::key_bytes(prev, &id.to_le_bytes())
    }
    fn key_bytes(prev: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(prev, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }
    /// The context being written becomes a segment with `snap` as its end state; its rows are no longer the active region.
    fn close(&mut self, ctx: usize, snap: Vec<f32>, logits: Vec<f32>) {
        if let Some(c) = self.contexts[ctx].as_mut() {
            c.rows = Box::new(Dram(std::mem::take(&mut self.active)));
            let kept = !snap.is_empty() && c.snap.is_empty();
            c.snap = snap;
            c.snap_logits = logits;
            if kept {
                self.held += self.state_cost;
                self.trim(ctx);
            }
        }
    }
    /// A new context, the active one, holding `n` positions read from `from`, or the prefix (its ids and rows) when `from` is none.
    fn open(&mut self, from: Option<(usize, usize)>, prefix_ids: &[u32], prefix_rows: &[f32]) -> usize {
        if self.active.is_empty() {
            self.active = vec![0f32; self.max_ctx * self.position];
        }
        let mut c = Context { ids: Vec::new(), keys: Vec::new(), rows: Box::new(Dram(Vec::new())), snap: Vec::new(), snap_logits: Vec::new(), marks: Vec::new(), used: 0 };
        match from {
            Some((src, n)) => {
                let p = self.contexts[src].as_ref().unwrap();
                c.ids.extend_from_slice(&p.ids[..n]);
                c.keys.extend_from_slice(&p.keys[..n]);
                p.rows.read(n * self.position, &mut self.active);
            }
            None => {
                for (i, &id) in prefix_ids.iter().enumerate() {
                    c.ids.push(id);
                    c.keys.push(Self::key(if i == 0 { 0xcbf2_9ce4_8422_2325 } else { c.keys[i - 1] }, id));
                }
                self.active[..prefix_rows.len()].copy_from_slice(prefix_rows);
            }
        }
        self.held += self.cost(&c);
        self.clock += 1;
        c.used = self.clock;
        let slot = self.contexts.iter().position(|c| c.is_none()).unwrap_or_else(|| {
            self.contexts.push(None);
            self.contexts.len() - 1
        });
        for (n, &k) in c.keys.iter().enumerate() {
            self.index.entry(k).or_insert((slot, n + 1));
        }
        self.contexts[slot] = Some(c);
        slot
    }
    /// Record position `n` of `ctx` (its id, key and rows are in place) and stay under budget.
    fn grew(&mut self, ctx: usize, n: usize) {
        let key = self.contexts[ctx].as_ref().unwrap().keys[n];
        self.index.entry(key).or_insert((ctx, n + 1));
        self.held += 1;
        self.trim(ctx);
    }
    /// A context's share of the budget: its positions, and its kept states as positions.
    fn cost(&self, c: &Context) -> usize {
        c.ids.len() + (c.marks.len() + usize::from(!c.snap.is_empty())) * self.state_cost
    }
    /// Evict least-recently-used contexts, never `keep`, until the budget holds.
    fn trim(&mut self, keep: usize) {
        while self.held > self.budget {
            let Some(victim) = self.contexts.iter().enumerate().filter(|(i, c)| *i != keep && c.is_some()).min_by_key(|(_, c)| c.as_ref().unwrap().used).map(|(i, _)| i) else { break };
            let c = self.contexts[victim].take().unwrap();
            for (n, k) in c.keys.iter().enumerate() {
                if self.index.get(k) == Some(&(victim, n + 1)) {
                    self.index.remove(k);
                }
            }
            self.held -= self.cost(&c);
        }
    }
    fn touch(&mut self, ctx: usize) {
        self.clock += 1;
        self.contexts[ctx].as_mut().unwrap().used = self.clock;
    }
}

/// One sequence at a time on this node: the model behind its engine, the
/// store, and the sampler, driven by the same three moves from the wire or
/// the console.
struct Session {
    e: Box<dyn Engine>,
    /// SCORE was received on this sequence: each id fed is scored first.
    scoring: bool,
    /// The row of the last step's logits that belongs to its last token.
    last_row: usize,
    next: u32, // the draw made at the last computed position
    pos: usize,
    max_ctx: usize,
    started: bool,
    /// Prefix ids the client still owes after the first: they are checked, never fed.
    prefix_left: usize,
    debug: Option<std::path::PathBuf>,
    store: Store,
    active: usize,
    /// While the prompt is a prefix the store holds: (context, positions matched); nothing computed yet.
    pending: Option<(usize, usize)>,
    hits: usize,
    /// Speculative decoding: the model carries a draft layer (the cards run it after every step).
    spec: bool,
    /// The drafts made after the last step, `[vocab]` each: the token after `next`, then the token after that one.
    drafts: Vec<Vec<f32>>,
    /// Of those, the ones the next step verifies (`redraft`'s gate), and each draft's probability for the token
    /// it will propose.
    depth: usize,
    draft_q: Vec<f64>,
    draft_made: usize,
    /// Verify steps run and drafts accepted, over the connection.
    verified: usize,
    accepted: usize,
    /// The encoder windows of this sequence, folded (FNV over their bytes) into every placeholder's store key: two
    /// transcripts match only if their ids and their windows agree (094). Reset at BOS.
    audio_salt: u64,
    /// Placeholders fed this sequence, against the rows its windows queued: each must have a row.
    audio_fed: usize,
    audio_rows: usize,
}
/// The uniforms the speculative path draws with, apart from the positions' own.
const DRAFT_SALT: u64 = 0x5eed_d4af_7000_0001;
const RESIDUAL_SALT: u64 = 0x5eed_d4af_7000_0002;
impl Session {
    /// The store key of `id` after `prev`: a placeholder's carries the sequence's windows too.
    fn key_of(&self, prev: u64, id: u32) -> u64 {
        let k = Store::key(prev, id);
        if id == self.e.audio_placeholder() { Store::key_bytes(k, &self.audio_salt.to_le_bytes()) } else { k }
    }
    /// The prompt's positions past the prefix: fed, or matched in the store and not yet resumed.
    fn prompt(&self) -> usize {
        self.pending.map_or(self.pos, |(_, n)| n) - self.first_pos()
    }
    fn first_pos(&self) -> usize {
        self.e.first_pos()
    }
    /// The first prefix id: a new context. Whether the model is seeded from the
    /// prefix or from a longer cached transcript is decided by the ids that follow.
    fn start(&mut self) {
        // the active region belongs to the active context whenever it is allocated; a new
        // connection resets `started` without closing, so the region, not the flag, decides
        if !self.store.active.is_empty() {
            let mut snap = vec![0f32; self.e.snapshot_len()];
            self.e.snapshot(&mut snap);
            let logits = if self.pos > self.first_pos() { self.e.logits_row(self.last_row).to_vec() } else { Vec::new() };
            self.store.close(self.active, snap, logits);
        }
        let (rows, _) = self.e.prefix();
        self.active = self.store.open(None, self.e.prefix_ids(), rows);
        // the prefix is the sequence's state until a longer cached transcript replaces it; seeding here
        // is what makes a sequence that matches nothing identical to one that matches everything
        self.e.seed_prefix();
        // the longest cached prefix so far is the prefix itself, in some closed context — never this one
        let first = self.first_pos();
        self.pending = self.store.index.get(&self.store.contexts[self.active].as_ref().unwrap().keys[first - 1]).copied().filter(|&(c, _)| c != self.active);
        self.pos = first;
        self.prefix_left = first - 1;
        self.started = true;
        self.scoring = false;
        self.hits = 0;
    }
    /// Leave the store's matching state with the logits current for the next
    /// position: nothing to do if only the prefix matched (its logits stand);
    /// otherwise the matched transcript but its last position is installed and
    /// that position is recomputed for its logits. False if the client never
    /// finished the prefix, or the context is full.
    fn settle(&mut self) -> bool {
        if self.prefix_left > 0 {
            return false;
        }
        match self.pending {
            Some((_, n)) if n == self.first_pos() => {
                self.pending = None;
                true
            }
            Some((_, n)) => {
                // the whole prompt is cached: resume at its end where the state and logits were kept,
                // else at the latest state before it and feed the rest (the last id's step gives the draw)
                self.materialize(n)
            }
            None => true,
        }
    }
    /// ln p(id) at the next position, before `id` is fed there: from the last
    /// computed token's logits, the prefix's if none was.
    fn score(&mut self, id: u32) -> f32 {
        if self.pos == self.first_pos() {
            self.e.score_prefix(id)
        } else {
            self.e.score(self.last_row, id)
        }
    }
    /// Leave the matching state: the active context becomes a copy of the
    /// matched transcript's first `n` positions, the model is seeded from it,
    /// and computing resumes at `n` — or, for a model with sequence state, at
    /// the latest snapshot at or before `n` (the transcript's end, a mark at a
    /// prompt's end, else the prefix), the matched ids after it recomputed.
    /// False when the context is full.
    fn materialize(&mut self, n: usize) -> bool {
        let (src, matched) = self.pending.take().unwrap();
        let n = n.min(matched);
        self.store.touch(src);
        let first = self.first_pos();
        let (at, replay): (usize, Vec<u32>) = {
            let c = self.store.contexts[src].as_ref().unwrap();
            let at = if self.e.snapshot_len() == 0 || n == first {
                n
            } else {
                let mut best = first;
                if n == c.ids.len() && !c.snap.is_empty() {
                    best = n;
                }
                for (p, _, _) in &c.marks {
                    if *p <= n && *p > best {
                        best = *p;
                    }
                }
                best
            };
            (at, c.ids[at..n].to_vec())
        };
        let old = self.active;
        let (rows, _) = self.e.prefix();
        self.active = self.store.open(Some((src, at)), self.e.prefix_ids(), rows);
        let gone = self.store.contexts[old].take().unwrap();
        self.store.held -= self.store.cost(&gone);
        let mut drawn = false;
        if at == first {
            self.e.seed_prefix();
        } else {
            let c = self.store.contexts[src].as_ref().unwrap();
            let (snap, logits): (&[f32], &[f32]) = if self.e.snapshot_len() == 0 {
                (&[], &[])
            } else if at == c.ids.len() && !c.snap.is_empty() {
                (&c.snap, &c.snap_logits)
            } else {
                let m = c.marks.iter().find(|(p, _, _)| *p == at).unwrap();
                (&m.1, &m.2)
            };
            self.e.seed(&self.store.active[..at * self.store.position], snap);
            // the logits kept with the state: the draw at this position needs no step
            if !logits.is_empty() {
                self.e.set_logits(0, logits);
                self.last_row = 0;
                self.next = self.e.draw(0, common::sample::uniform_at(self.e.seed_value(), (at - 1) as u64));
                drawn = true;
            }
        }
        self.pos = at;
        self.hits = at - first;
        prow::row!("track-store", "store_resume", { "matched" => n, "transcript" => self.store.contexts[src].as_ref().unwrap().ids.len(), "at" => at, "drawn" => drawn, "replayed" => replay.len() });
        replay.is_empty() || self.feed_many(&replay, &mut |_| {}, &|| false)
    }
    /// The sequence state now, kept with the active context at this position (the end of a
    /// prompt): what a later turn resumes from. Nothing for a model whose state is all rows.
    fn mark(&mut self) {
        let len = self.e.snapshot_len();
        if len == 0 || self.pos == self.first_pos() {
            return;
        }
        let c = self.store.contexts[self.active].as_mut().unwrap();
        if c.marks.last().is_some_and(|(p, _, _)| *p == self.pos) {
            return;
        }
        let t0 = std::time::Instant::now();
        let mut snap = vec![0f32; len];
        self.e.snapshot(&mut snap);
        let logits = self.e.logits_row(self.last_row).to_vec();
        prow::row!("track-store", "store_keep", { "pos" => self.pos, "mb" => 4.0 * len as f64 / 1e6, "ms" => t0.elapsed().as_secs_f64() * 1e3 });
        c.marks.push((self.pos, snap, logits));
        // the newest MARKS_KEPT stay; every kept state is charged to the budget, which evicts other contexts
        let dropped = c.marks.len().saturating_sub(MARKS_KEPT);
        c.marks.drain(..dropped);
        self.store.held += self.store.state_cost;
        self.store.held -= dropped * self.store.state_cost;
        self.store.trim(self.active);
    }
    /// `ids` through the model at the next positions, as few steps as they
    /// allow (`tokens()` per step): the prefix's remaining ids are checked and
    /// swallowed, ids still matching the store advance the match and cost
    /// nothing, and from the first that does not, the rest are computed in
    /// steps. In scoring mode every id's ln p is given to `emit` before it is
    /// fed, in order. False when the context is full or the prefix is not the model's.
    /// `gone` is asked before every computed step: a peer that closed during a long prompt's prefill is noticed
    /// there, not at the first write after it (an abandoned 18k prompt held the node — and every connection
    /// waiting behind it — for the whole prefill, 2026-10-23), so abort is closing the connection in prefill too.
    fn feed_many(&mut self, ids: &[u32], emit: &mut dyn FnMut(f32), gone: &dyn Fn() -> bool) -> bool {
        let mut at = 0;
        while at < ids.len() {
            let id = ids[at];
            if self.prefix_left > 0 {
                let first = self.first_pos();
                if id != self.e.prefix_ids()[first - self.prefix_left] {
                    return false;
                }
                self.prefix_left -= 1;
                at += 1;
                continue;
            }
            if let Some((ctx, n)) = self.pending {
                let key = self.key_of(self.store.contexts[ctx].as_ref().unwrap().keys[n - 1], id);
                if let Some(&(c2, n2)) = self.store.index.get(&key) {
                    if c2 != self.active && self.store.contexts[c2].as_ref().map(|c| c.ids.get(n2 - 1) == Some(&id)).unwrap_or(false) && n2 == n + 1 && n2 < self.max_ctx {
                        self.pending = Some((c2, n2));
                        at += 1;
                        continue;
                    }
                }
                if !self.materialize(n) {
                    return false;
                }
            }
            // the rest of the ids are computed: a step of up to `tokens()` at a time
            let n = (ids.len() - at).min(self.e.tokens()).min(self.max_ctx.saturating_sub(self.pos));
            if n == 0 || gone() {
                return false;
            }
            if self.scoring {
                let lp = self.score(ids[at]);
                emit(lp);
            }
            self.step(&ids[at..at + n]);
            if self.scoring {
                for j in 1..n {
                    let lp = self.e.score(j - 1, ids[at + j]);
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
        s.e.run(toks, pos, s.debug.is_some(), s.scoring || s.debug.is_some());
        // the draw at a position is a function of the seed and the position alone, so a
        // sequence draws the same whether its prefix was computed or served from the store
        let u = common::sample::uniform_at(s.e.seed_value(), (pos + n - 1) as u64);
        s.next = s.e.draw(n - 1, u);
        s.last_row = n - 1;
        s.commit(toks, 0, pos);
        if s.spec {
            s.redraft(toks, pos);
        }
        if s.debug.is_some() {
            for i in 0..n {
                let p = pos + i;
                let d = s.debug.as_ref().unwrap();
                let (layers, topk, lg) = s.e.trace(i);
                let raw = |v: &[f32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len()) }.to_vec();
                std::fs::write(d.join(format!("pos{p}.layers.f32")), raw(layers)).unwrap();
                std::fs::write(d.join(format!("pos{p}.topk.u32")), unsafe { std::slice::from_raw_parts(topk.as_ptr() as *const u8, 4 * topk.len()) }).unwrap();
                let mut ix = (0..lg.len()).collect::<Vec<_>>();
                ix.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap());
                eprintln!("  pos {p}: argmax {} margin {:.4e} (top2 {})", ix[0], lg[ix[0]] - lg[ix[1]], ix[1]);
                std::fs::write(d.join(format!("pos{p}.f32")), raw(lg)).unwrap();
            }
        }
    }
    /// After a step of `toks` at `pos..`: the drafts for the next step — the first over the step's positions
    /// from the main model's hidden states, each further one from the draft layer's own output and the
    /// token the draft before it proposed (sampled with the same uniform the verify will draw).
    fn redraft(&mut self, toks: &[u32], pos: usize) {
        let s = self;
        let seed = s.e.seed_value();
        let sam = s.e.sampling();
        let keep = s.e.draft_keep();
        s.e.draft(toks, s.next, pos, &mut s.drafts[0]);
        let end = pos + toks.len(); // the position of `next`
        // each draft's token (the verify draws the same) and the draft's own probability for it: the chain goes on
        // past a draft only when it is that sure (queue 081); the probabilities are the draft's alone, drawn with its
        // own uniforms, so how many drafts a step verifies leaves the emitted distribution the model's
        let q = |s: &mut Self, d: usize| {
            let x = common::sample::draw(&s.drafts[d], sam, common::sample::uniform_at(seed ^ DRAFT_SALT, (end + d + 1) as u64)) as u32;
            let p = common::sample::dist(&s.drafts[d], sam).iter().find(|(i, _)| *i == x).map_or(0.0, |e| e.1);
            s.draft_q[d] = p;
            (x, p)
        };
        let mut made = 1;
        let (mut x, mut p) = q(s, 0);
        // a further draft writes the draft layer's row at `end + made - 1`: never past the context (at 32k the third
        // draft's row was position `max_ctx`, out of the cache, 2026-10-20)
        while made < s.drafts.len() && p >= keep && end + made - 1 < s.max_ctx {
            s.e.draft_next(x, end + made - 1, &mut s.drafts[made]);
            (x, p) = q(s, made);
            made += 1;
        }
        // the first draft is always verified; a further one only when sure of its own token
        s.depth = if made > 1 && p < keep { made - 1 } else { made };
        s.draft_made = made;
    }
    /// Tokens `toks` computed at `pos..` (their rows at `row0..` of the last step) into the active
    /// context: rows, ids, keys; the position advances past them.
    fn commit(&mut self, toks: &[u32], row0: usize, pos: usize) {
        let (ph, salt) = (self.e.audio_placeholder(), self.audio_salt);
        let s = self;
        let position = s.store.position;
        for (i, &tok) in toks.iter().enumerate() {
            let p = pos + i;
            s.e.rows(row0 + i, &mut s.store.active[p * position..(p + 1) * position]);
            let c = s.store.contexts[s.active].as_mut().unwrap();
            c.ids.push(tok);
            let key = {
                let k = Store::key(c.keys[p - 1], tok);
                if tok == ph { Store::key_bytes(k, &salt.to_le_bytes()) } else { k }
            };
            c.keys.push(key);
            s.store.grew(s.active, p);
        }
        s.pos = pos + toks.len();
    }
    /// One token through the model at the next position. False when the context is full.
    fn feed(&mut self, id: u32) -> bool {
        self.feed_many(&[id], &mut |_| {}, &|| false)
    }
    /// GENERATE: sample, emit, feed, until EOS or `limit` tokens. Every emitted
    /// id goes through the model, so the sequence can continue from it. False
    /// when the context filled or `emit` refused. The state is kept at the prompt's
    /// end and at the reply's end — before the EOS, or after the `limit`th id — the
    /// two places a chat's next turn can match to.
    fn generate(&mut self, limit: usize, mut emit: impl FnMut(u32) -> bool) -> bool {
        if !self.settle() {
            return false;
        }
        self.mark();
        let first = self.first_pos();
        if self.pos == first {
            // nothing fed after the prefix: the first draw is from the prefix's logits
            self.next = self.e.draw_prefix(common::sample::uniform_at(self.e.seed_value(), (first - 1) as u64));
        }
        let mut emitted = 0;
        while emitted < limit {
            let id = self.next;
            let eos = self.e.is_eos(id);
            if eos {
                self.mark();
            }
            if !emit(id) {
                return false;
            }
            emitted += 1;
            if eos || !self.spec || emitted >= limit || self.pos + 2 > self.max_ctx {
                if !self.feed(id) {
                    return false;
                }
                if eos {
                    return true;
                }
                continue;
            }
            // speculative: the drafts made after the last step — the token after `id`, then the one after
            // that — are verified in one step with `id`: the model's own distribution at each position
            // decides, in order, and the draw after the accepted ones comes from the same step
            let s = self.e.sampling();
            let seed = self.e.seed_value();
            let pos = self.pos;
            let k = self.depth.min(limit - emitted).min(self.max_ctx - self.pos - 1);
            let mut toks = vec![id];
            for d in 0..k {
                toks.push(common::sample::draw(&self.drafts[d], s, common::sample::uniform_at(seed ^ DRAFT_SALT, (pos + 1 + d) as u64)) as u32);
            }
            self.e.run(&toks, pos, false, true);
            self.verified += 1;
            // `kept`: the computed tokens that stand (`id` and the accepted drafts); `tok`: the token after them
            let mut kept = 1;
            let mut tok = None;
            for d in 0..k {
                let p = pos + 1 + d;
                let (accepted, t) = common::sample::verify(self.e.logits_row(d), &self.drafts[d], s, toks[d + 1], common::sample::uniform_at(seed, p as u64), common::sample::uniform_at(seed ^ RESIDUAL_SALT, p as u64));
                if !accepted {
                    tok = Some(t);
                    break;
                }
                self.accepted += 1;
                kept += 1;
            }
            // the draft chain's gate a verify step (081): each draft's probability for its token (-1 where none was
            // made), the drafts made and verified, and the ones accepted
            let qd = |d: usize| if d < self.draft_made { self.draft_q[d] } else { -1.0 };
            prow::row!("track-step", "spec_step", { "pos" => pos, "q1" => qd(0), "q2" => qd(1), "made" => self.draft_made, "verified" => k, "accepted" => kept - 1 });
            self.next = match tok {
                Some(t) => t,
                None => self.e.draw(k, common::sample::uniform_at(seed, (pos + 1 + k) as u64)),
            };
            if kept < toks.len() {
                // the state after the last kept token is restored; the rejected tokens' rows are overwritten later
                self.e.rewind(kept - 1);
            }
            self.last_row = kept - 1;
            self.commit(&toks[..kept], 0, pos);
            self.redraft(&toks[..kept], pos);
            for &x in &toks[1..kept] {
                let eos = self.e.is_eos(x);
                if eos {
                    self.mark();
                }
                if !emit(x) {
                    return false;
                }
                emitted += 1;
                if eos {
                    return true;
                }
            }
        }
        self.mark();
        true
    }
}

/// The wire, one connection at a time (one cache).
fn serve(listen: &str, sessions: &mut [Session], identity: u64) {
    let listener = std::net::TcpListener::bind(listen).unwrap();
    println!("ready    : wire on {listen}, identity {identity:016x}, models {}", sessions.iter().map(|s| s.e.name()).collect::<Vec<_>>().join(" "));
    // between connections (the API opens one a request) the engines' threads rest: without this the last run's
    // busy state held 95 workers in MWAITX, 100 % user time to the kernel, on an idle node (2026-10-15). The
    // accept blocks inside `incoming()`, so the call goes before the loop and after each connection's end.
    sessions.iter_mut().for_each(|s| s.e.idle());
    for conn in listener.incoming() {
        let Ok(mut s) = conn else { continue };
        s.set_nodelay(true).unwrap();
        // the hello names the model; a name the bundle lacks is refused like a wrong identity (no default)
        let mut h = [0u8; wire::HELLO];
        if s.read_exact(&mut h).is_err() {
            continue;
        }
        let Some(session) = wire::accept(&h).filter(|&(want, _)| want == 0 || want == identity).and_then(|(_, name)| sessions.iter_mut().find(|s| s.e.name() == name)) else { continue };
        let (bos, vocab_size, tokens, first) = (session.e.bos(), session.e.vocab_size(), session.e.tokens(), session.first_pos());
        if s.write_all(&wire::header(identity, (session.max_ctx - first) as u32)).is_err() {
            continue;
        }
        session.started = false;
        let mut word = [0u8; 4];
        let mut ids: Vec<u32> = Vec::new();
        // ids are fed as they arrive, in whatever steps arrive: a prompt that is already in the
        // socket's buffer goes through the model `tokens` at a time; a stream of single ids one at a time
        let flush = |session: &mut Session, ids: &mut Vec<u32>, s: &mut std::net::TcpStream| -> bool {
            if ids.is_empty() {
                return true;
            }
            let mut ok = true;
            // the peer closed? `poll` for POLLRDHUP: the peer's shutdown shows even with its prompt still unread
            // in the socket (a peek sees those bytes, not the FIN queued behind them — the first form of this,
            // aff62, let an aborted 18k prompt run its whole prefill)
            let fd = { use std::os::fd::AsRawFd; s.as_raw_fd() };
            let gone = || peer_hung_up(fd);
            let done = session.started && session.feed_many(ids, &mut |lp| ok &= s.write_all(&lp.to_bits().to_le_bytes()).is_ok(), &gone);
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
        loop {
            // about to wait for the next word: the engine's spinning threads may rest
            if !more_now(&s) {
                session.e.idle();
            }
            if s.read_exact(&mut word).is_err() {
                break;
            }
            let ok = match u32::from_le_bytes(word) {
                id if id == bos => {
                    // rows still queued: the sequence ends with placeholders unfilled — a protocol error (094)
                    let ok = flush(session, &mut ids, &mut s) && session.e.audio_queued() == 0;
                    session.e.audio_reset();
                    (session.audio_salt, session.audio_fed, session.audio_rows) = (0, 0, 0);
                    session.start();
                    ok
                }
                wire::GENERATE => {
                    let mut lw = [0u8; 4];
                    if s.read_exact(&mut lw).is_err() {
                        break;
                    }
                    let limit = match u32::from_le_bytes(lw) as usize {
                        0 => session.max_ctx,
                        n => n,
                    };
                    let mut ok = flush(session, &mut ids, &mut s) && session.started && session.e.audio_queued() == 0;
                    if ok {
                        let (prompt, t0, mut n) = (session.prompt(), std::time::Instant::now(), 0usize);
                        ok = session.generate(limit, |id| {
                            n += 1;
                            s.write_all(&id.to_le_bytes()).is_ok()
                        });
                        request_row(session, "wire", prompt, n, t0.elapsed().as_secs_f64() * 1e3);
                    }
                    ok
                }
                wire::SCORE => {
                    let ok = flush(session, &mut ids, &mut s);
                    session.scoring = session.started;
                    ok && session.started && session.e.audio_queued() == 0 && session.settle()
                }
                // an encoder's window (094): the model's window of f32s follows; encoded now, its rows queued for the
                // placeholders fed after it, its bytes folded into their store keys. Refused by a model without an
                // encoder, before BOS, or short
                wire::AUDIO => {
                    let n = session.e.audio_window();
                    if n == 0 || !session.started || !flush(session, &mut ids, &mut s) {
                        false
                    } else {
                        let mut buf = vec![0u8; n * 4];
                        if s.read_exact(&mut buf).is_err() {
                            false
                        } else {
                            session.audio_salt = Store::key_bytes(session.audio_salt, &buf);
                            let w: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                            session.e.audio(&w);
                            session.audio_rows = session.audio_fed + session.e.audio_queued();
                            true
                        }
                    }
                }
                id if (id as usize) < vocab_size => {
                    if !session.started || (id == session.e.audio_placeholder() && session.audio_fed >= session.audio_rows) {
                        false
                    } else {
                        if id == session.e.audio_placeholder() {
                            session.audio_fed += 1;
                        }
                        ids.push(id);
                        if ids.len() >= tokens || !more_now(&s) {
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
        session.e.idle();
    }
}

/// Whether the peer has shut down its side of `s` (Linux `poll` with POLLRDHUP, no wait): true on hang-up or
/// error, false while it lives, whatever is still unread.
fn peer_hung_up(fd: i32) -> bool {
    #[repr(C)]
    struct PollFd {
        fd: i32,
        events: i16,
        revents: i16,
    }
    extern "C" {
        fn poll(fds: *mut PollFd, nfds: u64, timeout: i32) -> i32;
    }
    const POLLERR: i16 = 0x008;
    const POLLHUP: i16 = 0x010;
    const POLLRDHUP: i16 = 0x2000;
    let mut p = PollFd { fd, events: POLLRDHUP, revents: 0 };
    let n = unsafe { poll(&mut p, 1, 0) };
    n < 0 || (n > 0 && p.revents & (POLLRDHUP | POLLHUP | POLLERR) != 0)
}

/// A generated continuation: the prompt's positions past the prefix (`hits` of them from the store),
/// the ids generated and the time it took; the verify steps and drafts accepted since the last.
fn request_row(session: &mut Session, mode: &str, prompt: usize, generated: usize, ms: f64) {
    prow::row!("track-store", "request", { "mode" => mode, "prompt" => prompt, "hits" => session.hits, "generated" => generated, "ms" => ms, "verified" => session.verified, "accepted" => session.accepted });
    session.verified = 0;
    session.accepted = 0;
}

fn halt(msg: &str) -> ! {
    println!("-- {msg}; halting --");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("hsmp") {
        hsmp::cli(&argv[2..]);
        return;
    }
    if argv.get(1).map(String::as_str) == Some("probe") && argv.get(2).map(String::as_str) == Some("dram") {
        probe::dram_only();
        return;
    }
    platform::report();
    let cmdline = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
    if cmdline.split_whitespace().any(|t| t == "rockem.shell=1") {
        println!("-- dev: handing over to /bin/sh --");
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new("/bin/sh").exec();
        eprintln!("exec /bin/sh failed: {err}");
    }
    if config("rockem.probe").as_deref() == Some("1") {
        let elf = config("rockem.bundle").map(|b| std::path::Path::new(&b).join("kernels.gfx1100.elf")).filter(|p| p.exists()).map(|p| std::fs::read(p).unwrap());
        probe::run(elf.as_deref());
        if cmdline.split_whitespace().any(|t| t == "rockem.exit=1") || std::env::args().any(|t| t == "rockem.exit=1") {
            std::process::exit(0);
        }
        halt("probe done");
    }
    let Some(bundle) = config("rockem.bundle") else {
        if cmdline.split_whitespace().any(|t| t == "rockem.exit=1") {
            println!("-- rockem.exit=1: leaving --");
            std::process::exit(0);
        }
        halt("no rockem.bundle=<dir>; nothing to serve");
    };

    // --- construction: the manifest names the model, its engine builds it ---
    let max_ctx: usize = config("rockem.ctx").and_then(|v| v.parse().ok()).unwrap_or(2048);
    let gpu = config("rockem.gpu").is_some();
    let mem = if config("rockem.mem").as_deref() == Some("anon") { loader::Mem::Anon } else { loader::Mem::Hugetlb };
    let bundle = std::path::Path::new(&bundle);
    let manifest = std::fs::read(bundle.join("manifest.toml")).unwrap_or_else(|e| halt(&format!("{}: {e}", bundle.join("manifest.toml").display())));
    let names = model_names(&String::from_utf8_lossy(&manifest));
    if names.is_empty() {
        halt("the manifest lists no model ([[model]] name)");
    }
    prow::Run::new("rockem")
        .attrs(prow::build_attrs!())
        .server(config("rockem.prow").unwrap_or_else(|| PROW_SERVER.into()))
        .core(PROW_CORE)
        .attr("model", names.join(" "))
        .attr("bundle", bundle.display().to_string())
        .attr("ctx", max_ctx)
        .attr("gpu", gpu)
        .attr("cmdline", cmdline.trim())
        .attr("image", std::fs::read_to_string("/etc/rockem-profile").unwrap_or_default().trim())
        .start();
    let experts_host = config("rockem.experts").as_deref() == Some("host");
    let debug = config("rockem.debug").map(std::path::PathBuf::from);
    if let Some(d) = &debug {
        std::fs::create_dir_all(d).unwrap();
    }
    // the context stores: a budget in positions shared equally by the models, by default half of what the box
    // reports free — an assumed mean between the stores and everything else the node holds; `rockem.cache` sets it
    let avail_kb: usize = std::fs::read_to_string("/proc/meminfo").unwrap_or_default().lines().find_map(|l| l.strip_prefix("MemAvailable:")).and_then(|v| v.trim().split(' ').next()).and_then(|v| v.parse().ok()).unwrap_or(0);
    let budget_bytes = config("rockem.cache").and_then(|v| v.parse::<usize>().ok()).map(|gb| gb << 30).unwrap_or(avail_kb << 9) / names.len();
    // the host pool: one, a thread per physical core on every socket, shared by every host engine (two pools pinned
    // to the same cores hung the node, 2026-10-23); the run builder's core (083) is the last core of the main thread's
    // CCD, out of the pool (its CCD's cores past three do not touch anyway) — a box with one CCD keeps every core in
    let host_pool = || -> (&'static glm_5_3_flash::construct::Threads, Option<usize>) {
        let mut cores = glm_5_3_flash::construct::physical_cores();
        let main_ccd = glm_5_3_flash::construct::ccd_key(cores[0]);
        let builder = if cores.len() > 16 { cores.iter().rposition(|&c| glm_5_3_flash::construct::ccd_key(c) == main_ccd).filter(|&i| i > 0).map(|i| cores.remove(i)) } else { None };
        println!("threads  : {} pinned, one per physical core{}", cores.len(), builder.map_or(String::new(), |b| format!(", core {b} the run builder's")));
        (Box::leak(Box::new(glm_5_3_flash::construct::Threads::new(&cores))), builder)
    };
    let mut pool: Option<(&'static glm_5_3_flash::construct::Threads, Option<usize>)> = None;
    let host = |pool: &mut Option<(&'static glm_5_3_flash::construct::Threads, Option<usize>)>| *pool.get_or_insert_with(host_pool);
    // every model resident, each from its own directory of the bundle; a connection names the one it speaks
    let mut sessions: Vec<Session> = Vec::new();
    for name in &names {
        let dir = bundle.join(name);
        let e: Box<dyn Engine> = match name.as_str() {
            "deepseek-v2-lite" => v2lite::V2Lite::open(&dir, mem, max_ctx, gpu, experts_host),
            "glm-5.3-flash" => {
                let (p, b) = host(&mut pool);
                glm::Glm::open(&dir, mem, max_ctx, gpu, config("rockem.resident").and_then(|v| v.parse::<usize>().ok()).map_or(glm_kfd::RESIDENT_BYTES, |gb| gb << 30), p, b)
            }
            "voxtral-small-24b" => voxtral::Voxtral::open(&dir, mem, max_ctx, host(&mut pool).0),
            other => halt(&format!("the manifest lists model {other:?}, which this node does not carry")),
        };
        // the node's budget, capped by what the model itself can hold
        let max_ctx = max_ctx.min(e.context_limit());
        assert!(max_ctx > e.first_pos(), "rockem.ctx must leave room after the prefix");
        let position = e.position();
        let state_cost = ((e.snapshot_len() + e.vocab_size()) * 4).div_ceil(position * 4);
        let store = Store { contexts: Vec::new(), index: std::collections::HashMap::new(), budget: (budget_bytes / (position * 4)).max(max_ctx), state_cost, held: 0, clock: 0, max_ctx, position, active: Vec::new() };
        println!("cache    : {} cached positions ({:.1} GB of host memory), {} KB each{}, for {}", store.budget, budget_bytes as f64 / 1e9, position * 4 / 1024, if e.snapshot_len() > 0 { format!(", {} MB a kept state ({} positions' worth; the newest {MARKS_KEPT} marks a transcript)", e.snapshot_len() * 4 >> 20, state_cost) } else { String::new() }, e.name());
        let first = e.first_pos();
        let spec = e.drafts() && debug.is_none();
        let e_depth = if spec { e.draft_depth() } else { 0 };
        let vocab_n = e.vocab_size();
        let session = Session { e, scoring: false, last_row: 0, next: 0, pos: first, max_ctx, started: false, prefix_left: 0, debug: debug.clone(), store, active: 0, pending: None, hits: 0, spec, drafts: vec![vec![0.0; vocab_n]; e_depth], depth: e_depth.min(1), draft_q: vec![0.0; e_depth], draft_made: e_depth.min(1), verified: 0, accepted: 0, audio_salt: 0, audio_fed: 0, audio_rows: 0 };
        if spec {
            println!("spec     : the draft layer drafts up to {e_depth} tokens after every step, a further one while the last gives its token {} or more; verified in the next", session.e.draft_keep());
        }
        sessions.push(session);
    }

    // --- execution ---
    if let Some(listen) = config("rockem.listen") {
        let identity = wire::identity(&manifest);
        serve(&listen, &mut sessions, identity);
        return;
    }
    // the console speaks one model: the only one, or the one `rockem.model` names
    let want = config("rockem.model");
    let at = match (sessions.len(), &want) {
        (1, None) => 0,
        (_, Some(w)) => names.iter().position(|n| n == w).unwrap_or_else(|| halt(&format!("rockem.model={w:?} is not one of the bundle's models"))),
        _ => halt("the bundle has more than one model: the console needs rockem.model=<name>"),
    };
    let mut session = sessions.swap_remove(at);
    let vocab = Vocab::load(loader::load(&bundle.join(&names[at]).join("vocab.bin"), loader::Mem::Anon), session.e.vocab_size());
    let bos = session.e.bos();
    // the console: one line is a prompt, its newline is GENERATE
    let max_new: usize = config("rockem.max").and_then(|v| v.parse().ok()).unwrap_or(64);
    let prefix: Vec<String> = session.e.prefix_ids().iter().map(|id| id.to_string()).collect();
    println!("ready    : type token ids, e.g.  {} 549 6077 280 7239 317  ({max_new} tokens per line)", prefix.join(" "));
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        session.e.idle();
        let ids: Vec<u32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if ids.is_empty() {
            continue;
        }
        let t1 = std::time::Instant::now();
        // the line's ids after the prefix go through the model in steps, as the wire's do
        if ids[0] != bos || ids.iter().skip(1).any(|&id| id == bos) {
            println!("[a prompt is the prefix ({}) then ids: the prefix's positions are precomputed]", prefix.join(" "));
            continue;
        }
        session.start();
        if !session.feed_many(&ids[1..], &mut |_| {}, &|| false) {
            println!("[the context is full, or the prefix is not the model's]");
            continue;
        }
        for &id in &ids {
            out.write_all(vocab.token(id as usize)).unwrap();
        }
        if !session.started {
            continue;
        }
        let t_prefill = t1.elapsed();
        let prompt = session.prompt();
        let t2 = std::time::Instant::now();
        let mut n = 0;
        session.generate(max_new, |id| {
            n += 1;
            out.write_all(vocab.token(id as usize)).unwrap();
            out.flush().is_ok()
        });
        let t_gen = t2.elapsed();
        request_row(&mut session, "console", prompt, n, t_gen.as_secs_f64() * 1e3);
        println!();
        println!("[{} prompt tokens in {:.2}s ({} cached), {n} generated at {:.2} tok/s]", ids.len(), t_prefill.as_secs_f32(), session.hits, n as f32 / t_gen.as_secs_f32());
    }
}

/// The node's allocator (`common::alloc`): size classes and thread magazines over slabs never unmapped. musl's own cost
/// the main thread 77 µs a CPU layer of page faults, 3.5 ms a verify step (bench-alloc, 2026-10-22).
#[global_allocator]
static ALLOC: common::alloc::Classes = common::alloc::Classes;
