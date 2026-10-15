//! Construction: a sequence's state per layer, the scratch allocated once,
//! the routed experts from the bundle's stream, the thread pool, and the CPU
//! realization — every op in this process, the reference placement every card
//! kernel is diffed against — with the step that ties the fused kernels
//! together one token at a time. `std` only; nothing on the token path allocates.

use crate::ffn::Routed;
use crate::layout::{self, KDA_MEMORY, KDA_WINDOW, MLA_ROW, POSITION};
use crate::weights::{Attn, Ffn, Weights};
use crate::*;
use common::gemv::{self, widen};
use common::pool::Pool;
use common::rmsnorm::rmsnorm;
use common::trace::{Trace, NONE};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Worker threads pinned to `cores`, spinning for work. A job is one
/// allocation — the function, its item count, the item counter and the done
/// counter — published through one atomic pointer, so whatever job a worker
/// loads, it draws items from that job's own counter and counts completions
/// there: a worker that wakes late to a finished job finds its counter spent
/// and leaves. `run` publishes a new job only when no worker is registered
/// (`active`), and frees a job one generation after replacing it, so a worker
/// that registered during the swap never reads freed memory. Between jobs the
/// workers spin for a while, then yield, so an idle node does not burn its sockets.
pub struct Threads {
    shared: std::sync::Arc<Shared>,
    /// The job before the current one, freed at the next publish.
    retired: std::sync::atomic::AtomicPtr<Job>,
}
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}
struct Shared {
    job: std::sync::atomic::AtomicPtr<Job>,
    active: AtomicUsize,
    quit: std::sync::atomic::AtomicBool,
}
extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
}
fn pin(core: usize) {
    let mut mask = [0u64; 16];
    mask[core / 64] |= 1 << (core % 64);
    unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
}
/// Every item of `job` this thread can claim, in order.
unsafe fn work(job: *const Job) {
    let (f, n) = (&*(*job).f, (*job).n);
    loop {
        let i = (*job).next.fetch_add(1, Ordering::AcqRel);
        if i >= n {
            break;
        }
        f(i);
        (*job).done.fetch_add(1, Ordering::AcqRel);
    }
}
impl Threads {
    /// One worker per core after the first; the calling thread, pinned to `cores[0]`, is the other.
    pub fn new(cores: &[usize]) -> Self {
        assert!(!cores.is_empty());
        pin(cores[0]);
        let shared = std::sync::Arc::new(Shared { job: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()), active: AtomicUsize::new(0), quit: std::sync::atomic::AtomicBool::new(false) });
        for &core in &cores[1..] {
            let sh = shared.clone();
            std::thread::spawn(move || {
                pin(core);
                let mut seen: *mut Job = std::ptr::null_mut();
                let mut idle = 0u32;
                loop {
                    if sh.job.load(Ordering::Acquire) == seen {
                        if sh.quit.load(Ordering::Relaxed) {
                            return;
                        }
                        idle += 1;
                        if idle < 20_000 {
                            std::hint::spin_loop();
                        } else {
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        continue;
                    }
                    idle = 0;
                    // registered first, then the job as it is now: whatever it is, it is alive and its counters are its own
                    sh.active.fetch_add(1, Ordering::AcqRel);
                    let job = sh.job.load(Ordering::Acquire);
                    seen = job;
                    if !job.is_null() {
                        unsafe { work(job) };
                    }
                    sh.active.fetch_sub(1, Ordering::AcqRel);
                }
            });
        }
        Threads { shared, retired: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }
    }
}
impl Drop for Threads {
    fn drop(&mut self) {
        self.shared.quit.store(true, Ordering::Relaxed);
    }
}
impl Pool for Threads {
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let sh = &*self.shared;
        while sh.active.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        // no worker is inside any job: the one retired at the last publish can go
        let fp: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Box::into_raw(Box::new(Job { f: fp, n, next: AtomicUsize::new(0), done: AtomicUsize::new(0) }));
        let old = sh.job.swap(job, Ordering::AcqRel);
        let older = self.retired.swap(old, Ordering::AcqRel);
        if !older.is_null() {
            drop(unsafe { Box::from_raw(older) });
        }
        unsafe { work(job) };
        while unsafe { (*job).done.load(Ordering::Acquire) } != n {
            std::hint::spin_loop();
        }
    }
}

/// The cores a pool on this machine should take: one per physical core (the
/// first SMT sibling of each), every socket, as /sys lists them.
pub fn physical_cores() -> Vec<usize> {
    let mut cores = Vec::new();
    let mut i = 0;
    while let Ok(s) = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list")) {
        let first: usize = s.trim().split([',', '-']).next().unwrap().parse().unwrap();
        if first == i {
            cores.push(i);
        }
        i += 1;
    }
    if cores.is_empty() {
        cores = (0..std::thread::available_parallelism().map_or(1, |t| t.get())).collect();
    }
    cores
}

/// The routed experts from the bundle's expert stream in this process's memory,
/// each expert's rows split over the pool: first the gate/up rows in chunks of
/// `CHUNK` hidden channels — a chunk's clamped SwiGLU and its INT8 groups are
/// its own — then the down rows in chunks of `DOWN_CHUNK` over every expert's
/// quantized hidden state. Every row's arithmetic is `expert::forward`'s
/// exactly (the same tiles, groups and flush); only the cores differ.
pub struct CpuExperts<'a> {
    stream: &'a [u8],
    pool: &'a dyn Pool,
    xq: Vec<i8>,
    xs: Vec<f32>,
    xg: Vec<i32>,
    hq: Vec<i8>,  // [TOP_K][D_FF]
    hs: Vec<f32>, // [TOP_K][D_FF / GROUP]
    hg: Vec<i32>,
    elapsed: f64,
}
/// Hidden channels an item takes: two INT8 groups, eight items an expert.
const CHUNK: usize = 256;
/// Output rows an item takes for the down projection: eight items an expert.
const DOWN_CHUNK: usize = 512;
const _: () = assert!(D_FF % CHUNK == 0 && CHUNK % common::q4::GROUP == 0 && CHUNK % common::q4::TILE == 0 && H % DOWN_CHUNK == 0 && DOWN_CHUNK % common::q4::TILE == 0);

impl<'a> CpuExperts<'a> {
    pub fn new(stream: &'a [u8], pool: &'a dyn Pool) -> Self {
        use common::q4::GROUP;
        assert_eq!(stream.len(), layout::STREAM);
        CpuExperts { stream, pool, xq: vec![0; H], xs: vec![0.0; H / GROUP], xg: vec![0; H / GROUP], hq: vec![0; TOP_K * D_FF], hs: vec![0.0; TOP_K * D_FF / GROUP], hg: vec![0; TOP_K * D_FF / GROUP], elapsed: 0.0 }
    }
}
impl Routed for CpuExperts<'_> {
    fn elapsed(&mut self) -> f64 {
        std::mem::take(&mut self.elapsed)
    }
    fn run(&mut self, l: usize, idx: &[usize; TOP_K], x: &[f32], out: &mut [f32]) {
        use common::q4::{self, Act, QAct, GROUP, TILE};
        use common::swiglu::swiglu_clamped;
        let t0 = std::time::Instant::now();
        QAct { q: &mut self.xq, scale: &mut self.xs, gsum: &mut self.xg }.quantize(&x[..H]);
        let xa = Act { q: &self.xq, scale: &self.xs, gsum: &self.xg };
        let stream = self.stream;
        let (hq_p, hs_p, hg_p, out_p) = (self.hq.as_mut_ptr() as usize, self.hs.as_mut_ptr() as usize, self.hg.as_mut_ptr() as usize, out.as_mut_ptr() as usize);
        // the gate/up rows: item (k, c) owns expert k's hidden channels c·CHUNK.. and their groups
        self.pool.run(TOP_K * (D_FF / CHUNK), &move |i| {
            let (k, c) = (i / (D_FF / CHUNK), i % (D_FF / CHUNK));
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[k])]);
            let rows = |r0: usize| (&w.gate_up_w[r0 / TILE * 2 * H..(r0 + CHUNK) / TILE * 2 * H], &w.gate_up_s[r0 / TILE * (H / GROUP) * TILE..(r0 + CHUNK) / TILE * (H / GROUP) * TILE]);
            let mut gu = [0f32; 2 * CHUNK];
            let (gw, gs) = rows(c * CHUNK);
            q4::gemv::<H>(CHUNK, gw, gs, &xa, &mut gu[..CHUNK]);
            let (uw, us) = rows(D_FF + c * CHUNK);
            q4::gemv::<H>(CHUNK, uw, us, &xa, &mut gu[CHUNK..]);
            let mut h = [0f32; CHUNK];
            swiglu_clamped::<CHUNK>(&gu, SWIGLU_LIMIT, &mut h);
            let (q, sc, g) = unsafe {
                (
                    std::slice::from_raw_parts_mut((hq_p as *mut i8).add(k * D_FF + c * CHUNK), CHUNK),
                    std::slice::from_raw_parts_mut((hs_p as *mut f32).add(k * D_FF / GROUP + c * CHUNK / GROUP), CHUNK / GROUP),
                    std::slice::from_raw_parts_mut((hg_p as *mut i32).add(k * D_FF / GROUP + c * CHUNK / GROUP), CHUNK / GROUP),
                )
            };
            QAct { q, scale: sc, gsum: g }.quantize(&h);
        });
        // the down rows: item (k, c) owns rows c·DOWN_CHUNK.. of expert k's output over its whole hidden state
        let (hq, hs, hg) = (&self.hq, &self.hs, &self.hg);
        self.pool.run(TOP_K * (H / DOWN_CHUNK), &move |i| {
            let (k, c) = (i / (H / DOWN_CHUNK), i % (H / DOWN_CHUNK));
            let w = expert::Weights::from_record(&stream[layout::record(l, idx[k])]);
            let ha = Act { q: &hq[k * D_FF..(k + 1) * D_FF], scale: &hs[k * D_FF / GROUP..(k + 1) * D_FF / GROUP], gsum: &hg[k * D_FF / GROUP..(k + 1) * D_FF / GROUP] };
            let r0 = c * DOWN_CHUNK;
            let (dw, ds) = (&w.down_w[r0 / TILE * 2 * D_FF..(r0 + DOWN_CHUNK) / TILE * 2 * D_FF], &w.down_s[r0 / TILE * (D_FF / GROUP) * TILE..(r0 + DOWN_CHUNK) / TILE * (D_FF / GROUP) * TILE]);
            let o = unsafe { std::slice::from_raw_parts_mut((out_p as *mut f32).add(k * H + r0), DOWN_CHUNK) };
            q4::gemv::<D_FF>(DOWN_CHUNK, dw, ds, &ha, o);
        });
        self.elapsed += t0.elapsed().as_secs_f64();
    }
}

/// KDA layers placed away from this CPU — on a card — behind one interface.
/// `step` runs placed layer `l` over the block input `x` (`[H]`, before the
/// input norm: the placement norms it) into `out`; `seed` and `snapshot`
/// move a placed layer's state (its convolution window and memory). A layer
/// the placement does not hold runs here.
pub trait KdaPlaced {
    fn placed(&self, l: usize) -> bool;
    fn step(&mut self, l: usize, x: &[f32], out: &mut [f32]);
    fn seed(&mut self, l: usize, window: &[f32], mem: &[f32]);
    fn snapshot(&self, l: usize, window: &mut [f32], mem: &mut [f32]);
}

/// One layer's sequence state: a KDA layer's convolution window and memory
/// (the snapshot's), or an MLA layer's cache rows and pool keys (the rows').
pub enum LayerState {
    Kda { window: Vec<f32>, mem: Vec<f32> },
    Mla { cache: Vec<f32>, pools: Vec<f32> },
}

/// A token's time by phase, summed over steps until `take`n (bring-up and the node's
/// `rockem.profile` line). The experts are the routed realization's own count.
#[derive(Default, Clone, Copy)]
pub struct Profile {
    pub tokens: usize,
    pub hc: f64,
    pub kda_card: f64,
    pub kda_cpu: f64,
    pub mla: f64,
    pub dense_ffn: f64,
    pub router_shared: f64,
    pub experts: f64,
    pub head: f64,
}
impl Profile {
    /// One line of milliseconds a token, the phases in order and their sum.
    pub fn line(&self) -> String {
        let n = self.tokens.max(1) as f64 / 1e3;
        let sum = self.hc + self.kda_card + self.kda_cpu + self.mla + self.dense_ffn + self.router_shared + self.experts + self.head;
        format!("{} tokens: hc {:.1}  kda card {:.1}  kda cpu {:.1}  mla {:.1}  dense ffn {:.1}  router+shared {:.1}  experts {:.1}  head {:.1}  = {:.1} ms a token", self.tokens, self.hc / n, self.kda_card / n, self.kda_cpu / n, self.mla / n, self.dense_ffn / n, self.router_shared / n, self.experts / n, self.head / n, sum / n)
    }
}

pub struct CpuModel<'p, R: Routed> {
    pub routed: R,
    pub pool: &'p dyn Pool,
    pub profile: Profile,
    /// KDA layers that run elsewhere; `None` runs every layer here.
    pub placed: Option<Box<dyn KdaPlaced>>,
    pub max_ctx: usize,
    pub layers: Vec<LayerState>,
    /// The last step's logits, `[VOCAB_SIZE]`.
    pub logits: Vec<f32>,
    streams: Vec<f32>,
    x: Vec<f32>,
    xn: Vec<f32>,
    y: Vec<f32>,
    mean: Vec<f32>,
    hc_flat: Vec<f32>,
    hc_coef: Vec<f32>,
    k_mixed: Vec<f32>,
    k_conv: Vec<f32>,
    k_fa: Vec<f32>,
    k_f: Vec<f32>,
    k_beta: Vec<f32>,
    k_ga: Vec<f32>,
    k_gate: Vec<f32>,
    k_normed: Vec<f32>,
    k_heads: Vec<f32>,
    m_qr: Vec<f32>,
    m_qa: Vec<f32>,
    m_q: Vec<f32>,
    m_kva: Vec<f32>,
    m_ik: Vec<f32>,
    m_iq: Vec<f32>,
    m_iw: Vec<f32>,
    m_pscore: Vec<f32>,
    m_sel: Vec<usize>,
    m_heads: Vec<f32>,
    m_hs: Vec<f32>,
    f_gu: Vec<f32>,
    f_h: Vec<f32>,
    f_expert_out: Vec<f32>,
    f_shared: Vec<f32>,
    f_score: Vec<f32>,
}

impl<'p, R: Routed> CpuModel<'p, R> {
    pub fn new(routed: R, pool: &'p dyn Pool, max_ctx: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"), "the shipped realizations need AVX2+FMA");
        assert!(max_ctx >= FIRST_POS + 1);
        let layers = (0..NUM_HIDDEN_LAYERS)
            .map(|l| if LINEAR_ATTENTION[l] { LayerState::Kda { window: vec![0.0; KDA_WINDOW], mem: vec![0.0; KDA_MEMORY] } } else { LayerState::Mla { cache: vec![0.0; max_ctx * MLA_ROW], pools: vec![0.0; max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM] } })
            .collect();
        CpuModel {
            routed,
            pool,
            profile: Profile::default(),
            placed: None,
            max_ctx,
            layers,
            logits: vec![0.0; VOCAB_SIZE],
            streams: vec![0.0; HC * H],
            x: vec![0.0; H],
            xn: vec![0.0; H],
            y: vec![0.0; H],
            mean: vec![0.0; H],
            hc_flat: vec![0.0; HC * H],
            hc_coef: vec![0.0; HC_COEFS],
            k_mixed: vec![0.0; CONV_DIM],
            k_conv: vec![0.0; CONV_DIM],
            k_fa: vec![0.0; KDA_DIM],
            k_f: vec![0.0; KDA_QKV],
            k_beta: vec![0.0; KDA_HEADS],
            k_ga: vec![0.0; KDA_DIM],
            k_gate: vec![0.0; KDA_QKV],
            k_normed: vec![0.0; KDA_QKV],
            k_heads: vec![0.0; KDA_HEADS * kda::HEAD_SCRATCH],
            m_qr: vec![0.0; Q_LORA_RANK],
            m_qa: vec![0.0; Q_LORA_RANK],
            m_q: vec![0.0; MLA_HEADS * Q_HEAD],
            m_kva: vec![0.0; KV_LORA_RANK],
            m_ik: vec![0.0; INDEX_HEAD_DIM],
            m_iq: vec![0.0; INDEX_N_HEADS * INDEX_HEAD_DIM],
            m_iw: vec![0.0; INDEX_N_HEADS],
            m_pscore: vec![0.0; max_ctx.div_ceil(INDEX_KPOOL) + 1],
            m_sel: vec![0; DENSE_BELOW],
            m_heads: vec![0.0; MLA_HEADS * V_HEAD_DIM],
            m_hs: vec![0.0; MLA_HEADS * mla::HEAD_SCRATCH],
            f_gu: vec![0.0; 2 * INTERMEDIATE_SIZE],
            f_h: vec![0.0; INTERMEDIATE_SIZE],
            f_expert_out: vec![0.0; TOP_K * H],
            f_shared: vec![0.0; H],
            f_score: vec![0.0; E],
        }
    }

    /// Install `rows` (`[n][POSITION]`) as positions `0..n` and `snap` (`[SNAPSHOT]`)
    /// as the KDA layers' state at `n`; the MLA layers' pool keys are rebuilt from the rows.
    pub fn seed(&mut self, w: &Weights, rows: &[f32], snap: &[f32]) {
        assert_eq!(rows.len() % POSITION, 0);
        assert_eq!(snap.len(), layout::SNAPSHOT);
        let n = rows.len() / POSITION;
        assert!(n <= self.max_ctx);
        let (mut ki, mut mi) = (0, 0);
        for (l, st) in self.layers.iter_mut().enumerate() {
            match (st, &w.layers[l].attn) {
                (LayerState::Kda { window, mem }, _) => {
                    let base = ki * layout::KDA_STATE;
                    window.copy_from_slice(&snap[base..base + KDA_WINDOW]);
                    mem.copy_from_slice(&snap[base + KDA_WINDOW..base + layout::KDA_STATE]);
                    if let Some(p) = self.placed.as_mut().filter(|p| p.placed(l)) {
                        p.seed(l, window, mem);
                    }
                    ki += 1;
                }
                (LayerState::Mla { cache, pools }, Attn::Mla(mw)) => {
                    for p in 0..n {
                        cache[p * MLA_ROW..(p + 1) * MLA_ROW].copy_from_slice(&rows[p * POSITION + mi * MLA_ROW..p * POSITION + (mi + 1) * MLA_ROW]);
                    }
                    for j in 0..n / INDEX_KPOOL {
                        let (c, pk) = (&*cache, &mut pools[j * INDEX_HEAD_DIM..(j + 1) * INDEX_HEAD_DIM]);
                        mla::pool_key(mw, c, j, pk);
                    }
                    mi += 1;
                }
                _ => unreachable!("a layer's state and weights disagree"),
            }
        }
    }
    /// The bundle's prefix file: its rows, then its snapshot.
    pub fn seed_prefix(&mut self, w: &Weights, prefix: &[f32]) {
        assert_eq!(prefix.len(), layout::PREFIX_LEN);
        let (rows, snap) = prefix.split_at(FIRST_POS * POSITION);
        self.seed(w, rows, snap)
    }
    /// The KDA layers' state now, `[SNAPSHOT]`.
    pub fn snapshot(&self, out: &mut [f32]) {
        let mut ki = 0;
        for (l, st) in self.layers.iter().enumerate() {
            if let LayerState::Kda { window, mem } = st {
                let base = ki * layout::KDA_STATE;
                let (w, m) = out[base..base + layout::KDA_STATE].split_at_mut(KDA_WINDOW);
                match self.placed.as_ref().filter(|p| p.placed(l)) {
                    Some(p) => p.snapshot(l, w, m),
                    None => {
                        w.copy_from_slice(window);
                        m.copy_from_slice(mem);
                    }
                }
                ki += 1;
            }
        }
    }
    /// Position `pos`'s rows, `[POSITION]`.
    pub fn rows(&self, pos: usize, out: &mut [f32]) {
        let mut mi = 0;
        for st in &self.layers {
            if let LayerState::Mla { cache, .. } = st {
                out[mi * MLA_ROW..(mi + 1) * MLA_ROW].copy_from_slice(&cache[pos * MLA_ROW..(pos + 1) * MLA_ROW]);
                mi += 1;
            }
        }
    }

    /// Token `tok` at position `pos` — the state holds `0..pos` — leaving its
    /// logits in `self.logits`. `forced`: the MoE layers' picks to take instead
    /// of the router's, `[MoE layer][TOP_K]` (bring-up only).
    pub fn step(&mut self, w: &Weights, tok: u32, pos: usize, forced: Option<&[[usize; TOP_K]]>, tr: &mut impl Trace) {
        assert!(pos < self.max_ctx);
        let s = self;
        let row = &w.embed[tok as usize * H..(tok as usize + 1) * H];
        for j in 0..HC {
            for d in 0..H {
                s.streams[j * H + d] = widen(row[d]);
            }
        }
        tr.edge(NONE, "embed", &s.streams[..H]);
        let mut t = std::time::Instant::now();
        let mut lap = |acc: &mut f64| {
            let now = std::time::Instant::now();
            *acc += (now - t).as_secs_f64();
            t = now;
        };
        let p = &mut s.profile;
        p.tokens += 1;
        for l in 0..NUM_HIDDEN_LAYERS {
            let lw = &w.layers[l];
            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_attn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.input_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "attn_in", &s.xn);
            lap(&mut p.hc);
            match (&lw.attn, &mut s.layers[l]) {
                (Attn::Kda(_), _) if s.placed.as_ref().is_some_and(|p| p.placed(l)) => {
                    s.placed.as_mut().unwrap().step(l, &s.x, &mut s.y);
                    lap(&mut p.kda_card);
                }
                (Attn::Kda(a), LayerState::Kda { window, mem }) => {
                    let mut ks = kda::Scratch { mixed: &mut s.k_mixed, conv: &mut s.k_conv, fa: &mut s.k_fa, f: &mut s.k_f, beta: &mut s.k_beta, ga: &mut s.k_ga, gate: &mut s.k_gate, normed: &mut s.k_normed, heads: &mut s.k_heads };
                    kda::step(l, a, &s.xn, window, mem, &mut ks, s.pool, tr, &mut s.y);
                    lap(&mut p.kda_cpu);
                }
                (Attn::Mla(a), LayerState::Mla { cache, pools }) => {
                    let mut ms = mla::Scratch { qr: &mut s.m_qr, qa: &mut s.m_qa, q: &mut s.m_q, kva: &mut s.m_kva, ik: &mut s.m_ik, iq: &mut s.m_iq, iw: &mut s.m_iw, pscore: &mut s.m_pscore, sel: &mut s.m_sel, heads: &mut s.m_heads, hs: &mut s.m_hs };
                    mla::step(l, a, &s.xn, cache, pools, pos, &mut ms, s.pool, tr, &mut s.y);
                    lap(&mut p.mla);
                }
                _ => unreachable!("a layer's state and weights disagree"),
            }
            tr.edge(l, "attn_out", &s.y);
            hc::post(&mix, &s.y, &mut s.streams);

            let mut hcs = hc::Scratch { flat: &mut s.hc_flat, coef: &mut s.hc_coef };
            let mix = hc::pre(&lw.hc_ffn, &s.streams, &mut hcs, s.pool, &mut s.x);
            rmsnorm::<H>(&s.x, &lw.post_ln, RMS_NORM_EPS, &mut s.xn);
            tr.edge(l, "ffn_in", &s.xn);
            lap(&mut p.hc);
            let mut fs = ffn::Scratch { gu: &mut s.f_gu, h: &mut s.f_h, expert_out: &mut s.f_expert_out, shared: &mut s.f_shared, score: &mut s.f_score };
            match &lw.ffn {
                Ffn::Dense(d) => {
                    ffn::dense(l, d, &s.xn, &mut fs, s.pool, tr, &mut s.y);
                    lap(&mut p.dense_ffn);
                }
                Ffn::Moe(m) => {
                    ffn::moe(l, m, &s.xn, &mut fs, s.pool, &mut s.routed, forced.map(|f| &f[l - FIRST_K_DENSE_REPLACE]), tr, &mut s.y);
                    let e = s.routed.elapsed();
                    p.experts += e;
                    lap(&mut p.router_shared);
                    p.router_shared -= e;
                }
            }
            tr.edge(l, "ffn_out", &s.y);
            hc::post(&mix, &s.y, &mut s.streams);
            for d in 0..H {
                let mut acc = 0f32;
                for j in 0..HC {
                    acc += s.streams[j * H + d];
                }
                s.mean[d] = acc / HC as f32;
            }
            tr.edge(l, "out", &s.mean);
            lap(&mut p.hc);
        }
        rmsnorm::<H>(&s.mean, &w.norm, RMS_NORM_EPS, &mut s.xn);
        gemv::bf16_par::<VOCAB_SIZE, H>(w.lm_head, &s.xn, &mut s.logits, s.pool);
        tr.edge(NONE, "logits", &s.logits);
        lap(&mut p.head);
    }
}
