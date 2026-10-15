//! DeepSeek-V2-Lite on the card through the launcher (`kfd.rs`): the whole
//! model, one sync per token.

use crate::kfd::{raw, raw_u32, read, Args, Kernel, Kfd, Mem, WG};
use common::q4::GROUP;
use common::trace::Trace;
use deepseek_v2_lite::layout::{self, dense as dl};
use deepseek_v2_lite::{attn, ffn, Attention, FeedForward, Head, Model, Rope, Sampling, D_FF, H, HEADS, INTERMEDIATE_SIZE, KVA_DIM, KV_ROW, NUM_HIDDEN_LAYERS, N_ROUTED_EXPERTS, O_IN_DIM, QK_ROPE_HEAD_DIM, Q_DIM, SHARED_FF, TOKENS, TOP_K, VOCAB_SIZE};
use std::rc::Rc;
use std::sync::atomic::{fence, AtomicU32, Ordering};

impl Kfd {
    /// The bundle's dense file, resident: every realization indexes it by `layout::dense` offsets.
    pub fn upload_dense(&self, dense: &[u16]) -> Mem {
        let t0 = std::time::Instant::now();
        let d = self.vram_visible(dense.len() * 2);
        let t1 = std::time::Instant::now();
        self.upload(&d, 0, raw(dense));
        println!("gpu      : {:.1} GB of dense weights resident in VRAM (mapped in {:.2}s, written at {:.1} GB/s)", dense.len() as f64 * 2.0 / 1e9, (t1 - t0).as_secs_f32(), dense.len() as f64 * 2.0 / t1.elapsed().as_secs_f64() / 1e9);
        d
    }
}

/// The model on the card, one sync per token. The residual stays in VRAM
/// from the embedding to the head, the router runs on the card (`route`),
/// every layer's launches are posted as they come, and the host waits once,
/// in `logits`. What the host needs afterwards — the position's cache rows
/// for the store, and under `rockem.debug` each layer's residual and routing
/// — the kernels mirror into host memory as they go, read after that wait.
pub struct KfdModel<'a> {
    k: Rc<Kfd>,
    proj: Kernel,
    absorb: Kernel,
    mla: Kernel,
    merge: Kernel,
    out: Kernel,
    rows: Kernel,
    dense_down: Kernel,
    shared_down: Kernel,
    route: Kernel,
    up: Kernel,
    down: Kernel,
    sample: Kernel,
    score: Kernel,
    d_dense: Mem,
    d_stream: Mem,
    d_cache: Mem, // [layers][max_ctx][KV_ROW]
    d_cos: Mem,
    d_sin: Mem,
    // per-step buffers, TOKENS deep
    d_h: Mem,  // [T][H], the residual; host-visible for the embedding write
    d_q: Mem,
    d_kva: Mem,
    d_heads: Mem,
    d_qabs: Mem, // [T][HEADS][KV_LORA_RANK], the absorbed queries
    d_part: Mem,
    d_wt: Mem,
    d_xq: Mem,
    d_xs: Mem,
    d_xg: Mem,
    d_expert: Mem, // [T][TOP_K] u32, for `group`
    d_count: Mem,  // COUNT_WORDS u32: counts, the active experts, the arrival words
    d_logit: Mem,  // [T][E], the router rows' dots
    d_list: Mem,   // [E][T] u32
    d_egu: Mem,    // [T][TOP_K][2·D_FF]
    d_eout: Mem,   // [T][TOP_K][H]
    d_gu: Mem,     // [T][2·INTERMEDIATE_SIZE] f32, the widest
    d_mirror: Mem, // host: each token's cache rows, [T][layer][KV_ROW]
    d_layers: Mem, // host: every layer's residual out, [layer][T][H]
    d_topk: Mem,   // host, coherent: every layer's routing, [layer][T][TOP_K] u32
    d_logits: Mem, // VRAM, host-visible: [T][VOCAB_SIZE]
    d_next: Mem,   // host: the drawn token, or a score
    max_ctx: usize,
    /// The placement table, a word per expert: 0 on the card, 1 in this host's DRAM
    /// (`d_place` is its copy for the router). The host's share of the experts runs
    /// from `stream` in this process's memory, on the card's integers mirrored to
    /// `d_xq_m`/`d_xs_m`/`d_xg_m` (`[layer][T]…`), into its slots of `d_eout_h`
    /// (`host_experts`). The card waits for data, not for the host: the router's
    /// last workgroup writes `epoch` to the layer's word of `d_done` when its
    /// routing and integers are out, and the combine kernel reads a host slot
    /// once its word of `d_valid` reads `epoch` (`d_ready` is the combine's own
    /// broadcast of that among its workgroups). No packet, no signal.
    place: Vec<u32>,
    d_place: Mem,
    /// Layers whose combine is posted and whose host share is not yet done: (layer, the step's token count, epoch).
    pending: Vec<(usize, usize, u32)>,
    epoch: u32,
    d_done: Mem,   // host, coherent: [layer] u32
    d_valid: Mem,  // host, coherent: [T][TOP_K] u32
    d_eout_h: Mem, // host, coherent: [T][TOP_K][H], the host's slots
    d_ready: Mem,  // VRAM: the combine's broadcast word
    /// The host's expert time this token: computing, and waiting for marks (seconds; `bench-token` reads and resets).
    pub host_busy: f64,
    pub host_wait: f64,
    stream: &'a [u8],
    scratch: deepseek_v2_lite::expert::ScratchBuf,
    d_xq_m: Mem, // host, coherent: [layer][T][H] i8
    d_xs_m: Mem, // host, coherent: [layer][T][H / GROUP] f32
    d_xg_m: Mem, // host, coherent: [layer][T][H / GROUP] i32
}

impl<'a> KfdModel<'a> {
    pub fn new(k: Rc<Kfd>, dense: &[u16], stream: &'a [u8], rope: &Rope, max_ctx: usize, place: &[u32]) -> Self {
        assert_eq!(stream.len(), layout::STREAM);
        assert_eq!(place.len(), N_ROUTED_EXPERTS);
        let d_place = k.vram_visible(N_ROUTED_EXPERTS * 4);
        k.upload(&d_place, 0, raw_u32(place));
        let d_ready = k.vram_visible(4);
        k.upload(&d_ready, 0, &[0u8; 4]);
        println!("experts  : {} of {} in this host's DRAM, the rest on the card", place.iter().filter(|&&p| p == 1).count(), N_ROUTED_EXPERTS);
        let d_dense = k.upload_dense(dense);
        let t0 = std::time::Instant::now();
        let d_stream = k.vram_visible(stream.len());
        let t1 = std::time::Instant::now();
        k.upload(&d_stream, 0, stream);
        println!("gpu      : {:.1} GB of experts resident in VRAM (mapped in {:.2}s, written at {:.1} GB/s)", stream.len() as f64 / 1e9, (t1 - t0).as_secs_f32(), stream.len() as f64 / t1.elapsed().as_secs_f64() / 1e9);
        let d_cache = k.vram_visible(NUM_HIDDEN_LAYERS * max_ctx * KV_ROW * 4);
        let d_cos = k.vram_visible(rope.cos.len() * 4);
        k.upload(&d_cos, 0, raw(&rope.cos));
        let d_sin = k.vram_visible(rope.sin.len() * 4);
        k.upload(&d_sin, 0, raw(&rope.sin));
        println!("gpu      : attention cache for {max_ctx} positions in VRAM, steps of up to {TOKENS} tokens");
        const T: usize = TOKENS;
        let d_count = k.vram_visible(ffn::gfx11_consts::COUNT_WORDS * 4);
        k.upload(&d_count, 0, &vec![0u8; ffn::gfx11_consts::COUNT_WORDS * 4]); // the arrival word starts at zero
        KfdModel {
            proj: k.kernel("attn_proj"),
            absorb: k.kernel("attn_absorb"),
            mla: k.kernel("attn_mla"),
            merge: k.kernel("attn_merge"),
            out: k.kernel("attn_out"),
            rows: k.kernel("bf16_rows"),
            dense_down: k.kernel("dense_down"),
            shared_down: k.kernel("shared_down"),
            route: k.kernel("route"),
            up: k.kernel("experts_up"),
            down: k.kernel("experts_down"),
            sample: k.kernel("sample"),
            score: k.kernel("score"),
            d_dense,
            d_stream,
            d_cache,
            d_cos,
            d_sin,
            d_h: k.vram_visible(T * H * 4),
            d_q: k.vram(T * Q_DIM * 4),
            d_kva: k.vram(T * KVA_DIM * 4),
            d_heads: k.vram(T * O_IN_DIM * 4),
            d_qabs: k.vram(T * attn::gfx11_consts::QABS * 4),
            d_part: k.vram(T * attn::gfx11_consts::partials(max_ctx) * 4),
            d_wt: k.vram(T * TOP_K * 4),
            d_xq: k.vram(T * H),
            d_xs: k.vram(T * 4 * H / GROUP),
            d_xg: k.vram(T * 4 * H / GROUP),
            d_expert: k.vram(T * TOP_K * 4),
            d_count,
            d_logit: k.vram(T * N_ROUTED_EXPERTS * 4),
            d_list: k.vram(N_ROUTED_EXPERTS * T * 4),
            d_egu: k.vram(T * TOP_K * 2 * D_FF * 4),
            d_eout: k.vram(T * TOP_K * H * 4),
            d_gu: k.vram(T * 2 * INTERMEDIATE_SIZE * 4),
            d_mirror: k.host(T * layout::POSITION * 4),
            d_layers: k.host(NUM_HIDDEN_LAYERS * T * H * 4),
            // coherent: the host reads the picks mid-token, behind the router's done word, with no
            // packet release to write the card's L2 back (cached host memory read stale picks, 2026-09-20)
            d_topk: k.host_coherent(NUM_HIDDEN_LAYERS * T * TOP_K * 4),
            d_logits: k.vram_visible(T * VOCAB_SIZE * 4), // the card draws from it; the host reads it only to trace
            d_next: k.host(4),
            max_ctx,
            place: place.to_vec(),
            d_place,
            pending: Vec::new(),
            epoch: 0,
            d_done: k.host_coherent(NUM_HIDDEN_LAYERS * 4),
            d_valid: k.host_coherent(T * TOP_K * 4),
            d_eout_h: k.host_coherent(T * TOP_K * H * 4),
            d_ready,
            host_busy: 0.0,
            host_wait: 0.0,
            stream,
            scratch: deepseek_v2_lite::expert::ScratchBuf::new(),
            d_xq_m: k.host_coherent(NUM_HIDDEN_LAYERS * T * H),
            d_xs_m: k.host_coherent(NUM_HIDDEN_LAYERS * T * (H / GROUP) * 4),
            d_xg_m: k.host_coherent(NUM_HIDDEN_LAYERS * T * (H / GROUP) * 4),
            k,
        }
    }
    fn dense_at(&self, elems: usize) -> usize {
        self.d_dense.at(elems * 2)
    }
    /// Wait for everything posted (benches; the node waits in `draw`).
    #[allow(dead_code)]
    pub fn sync(&mut self) {
        self.host_experts(); // the gates still posted, before waiting behind them
        self.k.sync()
    }
    /// The last token's launches with their card timestamps (see `Kfd::timeline`).
    #[allow(dead_code)]
    pub fn timeline(&self) -> Vec<(&'static str, u64, u64)> {
        self.k.timeline()
    }
    /// The last token's kernel statistics (see `Kfd::stats`).
    #[allow(dead_code)]
    pub fn stats(&self) -> Vec<(&'static str, [Vec<f32>; 4])> {
        self.k.stats()
    }
    /// `xn = rmsnorm(h) ⊙ ln` for the step's `n` tokens: what every rows kernel reads.
    /// `out[n][m] = W[m][H] · (rmsnorm(h[n]) ⊙ ln)`, posted; `free` lets it launch alongside the dispatch before it.
    fn rows(&self, ln: usize, w: usize, out: &Mem, m: usize, n: usize, free: bool) {
        let mut a = Args::new();
        a.ptr(self.d_h.va).ptr(ln).ptr(w).ptr(out.va).u32(m as u32).u32(n as u32);
        let lds = ffn::gfx11_consts::normed_rows_lds(H, n) as u32;
        if free {
            self.k.launch_free(&self.rows, ffn::gfx11_consts::rows_grid(m), WG, lds, &a);
        } else {
            self.k.launch(&self.rows, ffn::gfx11_consts::rows_grid(m), WG, lds, &a);
        }
    }
}

impl Attention for KfdModel<'_> {
    fn seed(&mut self, rows: &[f32]) {
        assert_eq!(rows.len() % layout::POSITION, 0);
        // the store is position-major and the card layer-major: gather each layer's rows in
        // DRAM, then write the layer as one run — a BAR write is at the link only in long runs
        let n = rows.len() / layout::POSITION;
        let mut run = vec![0f32; n * KV_ROW];
        for l in 0..NUM_HIDDEN_LAYERS {
            for p in 0..n {
                run[p * KV_ROW..(p + 1) * KV_ROW].copy_from_slice(&rows[p * layout::POSITION + l * KV_ROW..p * layout::POSITION + (l + 1) * KV_ROW]);
            }
            self.k.upload(&self.d_cache, attn::gfx11_consts::cache_layer(l, self.max_ctx), raw(&run));
        }
    }
    fn rows(&self, i: usize, out: &mut [f32]) {
        read(&self.d_mirror, i * layout::POSITION * 4, out);
    }
    /// Layer 0 carries the embeddings in; after that `h` on the host is not read again until the trace.
    fn step(&mut self, l: usize, _w: &attn::Weights, h: &mut [f32], pos: usize, n: usize, _cos: &[f32], _sin: &[f32], scale: f32, _tr: &mut dyn Trace) {
        assert!(n >= 1 && n <= TOKENS && pos + n <= self.max_ctx);
        if l == 0 {
            self.k.upload(&self.d_h, 0, raw(&h[..n * H]));
        }
        let b = dl::layer(l);
        let (ln, wq, wkva, kvaln, kvb, wo) = (self.dense_at(b + dl::INPUT_LN), self.dense_at(b + dl::Q_PROJ), self.dense_at(b + dl::KV_A), self.dense_at(b + dl::KV_A_LN), self.dense_at(b + dl::KV_B), self.dense_at(b + dl::O_PROJ));
        let cache = self.d_cache.at(attn::gfx11_consts::cache_layer(l, self.max_ctx));
        let cos = self.d_cos.at(pos * QK_ROPE_HEAD_DIM * 4);
        let sin = self.d_sin.at(pos * QK_ROPE_HEAD_DIM * 4);
        let max_splits = attn::gfx11_consts::max_splits(self.max_ctx) as u32;
        let mut a1 = Args::new();
        a1.ptr(self.d_h.va).ptr(ln).ptr(wq).ptr(wkva).ptr(self.d_q.va).ptr(self.d_kva.va).u32(n as u32);
        self.k.launch(&self.proj, attn::gfx11_consts::PROJ_GRID, WG, attn::gfx11_consts::normed_rows_lds(H, n) as u32, &a1);
        // the absorbed queries and the cache rows, one launch after the projections
        let mut ab = Args::new();
        ab.ptr(self.d_q.va).ptr(kvb).ptr(self.d_qabs.va).u32(n as u32).ptr(self.d_kva.va).ptr(kvaln).ptr(cache).u32(pos as u32).ptr(cos).ptr(sin).ptr(self.d_mirror.at(l * KV_ROW * 4));
        self.k.launch(&self.absorb, attn::gfx11_consts::absorb_grid(n), WG, attn::gfx11_consts::ABSORB_LDS as u32, &ab);
        let mut a2 = Args::new();
        a2.ptr(self.d_qabs.va).ptr(cache).u32(pos as u32).ptr(self.d_q.va).ptr(cos).ptr(sin).f32(scale).ptr(self.d_part.va).u32(max_splits);
        self.k.launch(&self.mla, attn::gfx11_consts::mla_grid(pos, n), WG, attn::gfx11_consts::MLA_LDS as u32, &a2);
        let mut am = Args::new();
        am.ptr(self.d_part.va).u32(pos as u32).u32(max_splits).ptr(kvb).ptr(self.d_heads.va);
        self.k.launch(&self.merge, (n * HEADS) as u32, WG, attn::gfx11_consts::MERGE_LDS as u32, &am);
        let mut a3 = Args::new();
        a3.ptr(self.d_heads.va).ptr(wo).ptr(self.d_h.va).u32(n as u32);
        self.k.launch(&self.out, attn::gfx11_consts::OUT_GRID, WG, attn::gfx11_consts::rows_lds(O_IN_DIM, n) as u32, &a3);
    }
}

impl FeedForward for KfdModel<'_> {
    fn dense(&mut self, l: usize, _w: &ffn::Dense, _h: &mut [f32], n: usize, _tr: &mut dyn Trace) {
        let b = dl::layer(l);
        let (ln, gate, down) = (self.dense_at(b + dl::POST_LN), self.dense_at(b + dl::DENSE_GATE), self.dense_at(b + dl::DENSE_DOWN));
        self.rows(ln, gate, &self.d_gu, 2 * INTERMEDIATE_SIZE, n, false); // gate rows then up rows, contiguous in the file
        let mut a2 = Args::new();
        a2.ptr(self.d_gu.va).ptr(down).ptr(self.d_h.va).ptr(self.d_layers.at(l * TOKENS * H * 4)).u32(n as u32);
        self.k.launch(&self.dense_down, ffn::gfx11_consts::rows_grid(H), WG, ffn::gfx11_consts::rows_lds(INTERMEDIATE_SIZE, n) as u32, &a2);
    }

    fn moe(&mut self, l: usize, _w: &ffn::Moe, _h: &mut [f32], n: usize, _tr: &mut dyn Trace) {
        use deepseek_v2_lite::expert::gfx11_consts as ex;
        let b = dl::layer(l);
        let (ln, router, sg, sd) = (self.dense_at(b + dl::POST_LN), self.dense_at(b + dl::ROUTER), self.dense_at(b + dl::SHARED_GATE), self.dense_at(b + dl::SHARED_DOWN));
        let base = layout::record(l, 0).start as u64;
        let mut ar = Args::new();
        ar.ptr(self.d_h.va).ptr(ln).ptr(router).ptr(self.d_logit.va).ptr(self.d_wt.va).ptr(self.d_xq.va).ptr(self.d_xs.va).ptr(self.d_xg.va).ptr(self.d_expert.va).ptr(self.d_topk.at(l * TOKENS * TOP_K * 4)).u32(n as u32).ptr(self.d_count.va).ptr(self.d_list.va);
        self.epoch = self.epoch.wrapping_add(1).max(1);
        ar.ptr(self.d_place.va).ptr(self.d_xq_m.at(l * TOKENS * H)).ptr(self.d_xs_m.at(l * TOKENS * (H / GROUP) * 4)).ptr(self.d_xg_m.at(l * TOKENS * (H / GROUP) * 4)).ptr(self.d_done.at(l * 4)).u32(self.epoch);
        self.k.launch(&self.route, (n * ffn::gfx11_consts::ROUTE_WGS) as u32, WG, ffn::gfx11_consts::ROUTE_LDS as u32, &ar);
        let mut au = Args::new();
        au.ptr(self.d_stream.va).u64(base).ptr(self.d_count.va).ptr(self.d_list.va).ptr(self.d_xq.va).ptr(self.d_xs.va).ptr(self.d_xg.va).ptr(self.d_egu.va);
        self.k.launch(&self.up, ex::up_grid(n), WG, ex::experts_lds(H, n) as u32, &au);
        // the shared experts' rows need only the residual, complete since the router
        // launched: free to overlap the routed experts' up
        self.rows(ln, sg, &self.d_gu, 2 * SHARED_FF, n, true);
        let mut ad = Args::new();
        ad.ptr(self.d_stream.va).u64(base).ptr(self.d_count.va).ptr(self.d_list.va).ptr(self.d_egu.va).ptr(self.d_eout.va);
        self.k.launch(&self.down, ex::down_grid(n), WG, ex::experts_lds(D_FF, n) as u32, &ad);
        let mut a2 = Args::new();
        a2.ptr(self.d_gu.va).ptr(sd).ptr(self.d_eout.va).ptr(self.d_eout_h.va).ptr(self.d_wt.va).ptr(self.d_h.va).ptr(self.d_layers.at(l * TOKENS * H * 4)).u32(n as u32);
        a2.ptr(self.d_place.va).ptr(self.d_expert.va).ptr(self.d_valid.va).ptr(self.d_ready.va).u32(self.epoch);
        self.pending.push((l, n, self.epoch));
        self.k.launch(&self.shared_down, ffn::gfx11_consts::rows_grid(H), WG, ffn::gfx11_consts::rows_lds(SHARED_FF, n) as u32, &a2);
    }
}

impl KfdModel<'_> {
    /// This host's share of the step's experts, layer by layer behind the card:
    /// for every layer posted, wait for its done word, run every (token, slot)
    /// whose expert is placed here from the mirrored integers into its slot of
    /// `d_eout_h`, and write the slot's valid word. An empty share is an empty
    /// loop body. Runs once the step is posted (`logits`), so the card is ahead
    /// the whole time, and before any wait (`sync`), so a driver that posts
    /// layers alone cannot deadlock on the combine's wait.
    pub fn host_experts(&mut self) {
        use deepseek_v2_lite::expert;
        let mut out = vec![0f32; H];
        for (l, n, epoch) in std::mem::take(&mut self.pending) {
            let t = std::time::Instant::now();
            let done = unsafe { &*(self.d_done.at(l * 4) as *const AtomicU32) };
            while done.load(Ordering::Acquire) != epoch {
                std::hint::spin_loop();
                assert!(t.elapsed().as_secs() < 10, "gpu hang: layer {l}'s routing never came");
            }
            self.host_wait += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            for i in 0..n {
                for kk in 0..TOP_K {
                    let mut e = [0u32; 1];
                    read(&self.d_topk, ((l * TOKENS + i) * TOP_K + kk) * 4, &mut e);
                    let e = e[0] as usize;
                    if self.place[e] == 1 {
                        let ew = expert::Weights::from_record(&self.stream[layout::record(l, e)]);
                        let (xq, xs, xg) = unsafe {
                            (
                                std::slice::from_raw_parts(self.d_xq_m.at((l * TOKENS + i) * H) as *const i8, H),
                                std::slice::from_raw_parts(self.d_xs_m.at((l * TOKENS + i) * (H / GROUP) * 4) as *const f32, H / GROUP),
                                std::slice::from_raw_parts(self.d_xg_m.at((l * TOKENS + i) * (H / GROUP) * 4) as *const i32, H / GROUP),
                            )
                        };
                        expert::forward_q(&ew, xq, xs, xg, &mut self.scratch.scratch(), &mut out);
                        // the rows, then the slot's word: the card polls the word and reads the rows after it
                        unsafe {
                            std::ptr::copy_nonoverlapping(out.as_ptr(), self.d_eout_h.at((i * TOP_K + kk) * H * 4) as *mut f32, H);
                            fence(Ordering::SeqCst);
                            (*(self.d_valid.at((i * TOP_K + kk) * 4) as *const AtomicU32)).store(epoch, Ordering::Release);
                        }
                    }
                }
            }
            self.host_busy += t.elapsed().as_secs_f64();
        }
    }
}

impl Head for KfdModel<'_> {
    /// Posts the head for every token of the step; `logits` is filled on the card, row per token.
    fn logits(&mut self, _norm: &[f32], _lm_head: &[u16], _h: &[f32], n: usize, _logits: &mut [f32]) {
        let (ln, w) = (self.dense_at(dl::NORM), self.dense_at(dl::LM_HEAD));
        self.rows(ln, w, &self.d_logits, VOCAB_SIZE, n, false);
        // the step is posted: the host's experts follow the card through the layers
        self.host_experts();
    }
}

impl Model for KfdModel<'_> {
    fn token_trace(&self, i: usize, layers: &mut [f32], topk: &mut [u32], logits: &mut [f32]) {
        for l in 0..NUM_HIDDEN_LAYERS {
            read(&self.d_layers, (l * TOKENS + i) * H * 4, &mut layers[l * H..(l + 1) * H]);
            read(&self.d_topk, (l * TOKENS + i) * TOP_K * 4, &mut topk[l * TOP_K..(l + 1) * TOP_K]);
        }
        read(&self.d_logits, i * VOCAB_SIZE * 4, logits); // through the BAR: uncached reads, the trace's price
    }
    /// ln p(id) on the card, one wait.
    fn score(&mut self, i: usize, id: usize, _logits: &[f32]) -> f32 {
        let mut a = Args::new();
        a.ptr(self.d_logits.at(i * VOCAB_SIZE * 4)).ptr(self.d_next.va).u32(id as u32);
        self.k.launch(&self.score, 1, WG, ffn::gfx11_consts::SCORE_LDS as u32, &a);
        self.k.sync();
        let mut v = [0f32; 1];
        read(&self.d_next, 0, &mut v);
        v[0]
    }
    /// The one wait of the step: the draw on the card from token `i`'s row, then
    /// the token read from host memory. The logits stay on the card (`token_trace` reads them).
    fn draw(&mut self, i: usize, recipe: &Sampling, u: f64, _logits: &[f32]) -> usize {
        let mut a = Args::new();
        a.ptr(self.d_logits.at(i * VOCAB_SIZE * 4)).ptr(self.d_next.va).f64(u).f32(recipe.temperature).u32(recipe.top_k as u32).f32(recipe.top_p);
        self.k.launch(&self.sample, 1, WG, ffn::gfx11_consts::SAMPLE_LDS as u32, &a);
        self.k.sync();
        let mut next = [0u32; 1];
        read(&self.d_next, 0, &mut next);
        next[0] as usize
    }
}
