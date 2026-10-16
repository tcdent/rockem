//! GLM-5.3-Flash's blocks on the cards, behind `construct::Placed` — every
//! block but the routed experts, and the residual streams themselves. Each
//! card keeps its own copy of the four streams and runs the hyper-connection
//! on them (`hc_coef`, `hc_mix`: the block input into VRAM, where the block
//! reads it; `hc_post`: the block output back into the streams), so a token
//! is a sequence of launches per card the host posts a layer at a time and
//! waits for only where it has work of its own: the routed experts.
//!
//! Each KDA layer is split by heads into `PARTS` parts, one card each where
//! there are two: a part holds its heads' projections, the layer's small
//! tensors and the state, and a step runs on each card the normed projections
//! (`rows_h`: the gates' first halves, beta and the part's q, k and v rows as
//! one launch), the gates' second halves (`rows_d2`: f_b over f_a's output and
//! g_b over g_a's, one launch), the delta rule
//! with the heads' memory resident in registers (`kda_step` over the part's
//! heads), and the part's half of the output projection (`rows_part`) into
//! host memory every card sees. Each layer's FFN half that is not the routed
//! experts — the dense layers' MLP, the MoE layers' shared expert — is on the
//! card of the KDA layer at or before it as FP8 rows (`ffn_gu`, `ffn_down_*`);
//! the MLA layers' projections and attention likewise. The head is the final
//! norm and lm_head as one `rows_h` launch over the streams' mean.
//!
//! What crosses between the cards, or between a card and the host, goes
//! through uncached host memory mapped on every card (`Kfd::host_shared`): a
//! block output's parts, the FFN half's input for the host's experts and
//! their output back, the early router's logits. A word per handoff says the
//! data is there: the producer's queue writes it after the data (`hc_signal`),
//! the consumer's queue holds until it reads the value (`hc_wait`) — the card
//! waits for data, never the host for a card except to read. The sequence
//! number a half of a layer carries is the value. Block outputs the host
//! would read through the BAR come this way too (the host reads its BAR at
//! 10–30 MB/s; probe 2026-10-15); block inputs the host writes (the embedding
//! into the streams, a latent, a selection) stream in through the BAR at 16 GB/s.
//! The layers split in order between the cards the topology lists, by their
//! KDA layers; the head sits with the last layer.

use crate::kfd::{read, Args, Kfd, Mem};
use glm_5_3_flash::construct::Placed;
use glm_5_3_flash::ffn::gfx11_consts::{DENSE_GU_ROWS, SHARED_GU_ROWS};
use glm_5_3_flash::hc::gfx11_consts::{CHANNEL_GRID, COEF_ROWS, MIX as HC_MIX, PRE_GRID, PRE_LDS, SLICES as HC_SLICES};
use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::mla::gfx11_consts::{ATTEND_GRID, ATTEND_LDS, IX_ROWS, O_IN};
use glm_5_3_flash::DENSE_BELOW;
use glm_5_3_flash::{INDEX_HEAD_DIM, INDEX_N_HEADS, KV_LORA_RANK, Q_LORA_RANK};
use glm_5_3_flash::weights::{Attn, Ffn, Fp8, Weights};
use glm_5_3_flash::{CONV_DIM, CONV_TAPS, D_FF, H, HC, INTERMEDIATE_SIZE, KDA_DIM, KDA_HEADS, KDA_QKV, LINEAR_ATTENTION, MLA_HEADS, NUM_HIDDEN_LAYERS, N_SHARED_EXPERTS, Q_HEAD, VOCAB_SIZE};
use std::sync::atomic::AtomicU32;
use std::rc::Rc;

struct Card {
    k: Rc<Kfd>,
    rows_h: crate::kfd::Kernel,
    rows_d2: crate::kfd::Kernel,
    rows_part: crate::kfd::Kernel,
    step: crate::kfd::Kernel,
    copy: crate::kfd::Kernel,
    ffn_gu: crate::kfd::Kernel,
    ffn_down_shared: crate::kfd::Kernel,
    ffn_down_dense: crate::kfd::Kernel,
    mla_qb: crate::kfd::Kernel,
    mla_iq: crate::kfd::Kernel,
    mla_attend: crate::kfd::Kernel,
    mla_o: crate::kfd::Kernel,
    hc_pre: crate::kfd::Kernel,
    hc_post: crate::kfd::Kernel,
    hc_mean: crate::kfd::Kernel,
    hc_handoff: crate::kfd::Kernel,
    // the residual streams and the hyper-connection's scratch: this card's copy
    streams: Mem, // [HC][H] f32, host-visible: the embedding goes in through the BAR
    part: Mem,    // [HC_COEFS][SLICES] f32: the coefficient GEMV's partials
    count: Mem,   // one u32: `hc_pre`'s arrival count, 0 between launches
    mix: Mem,     // [MIX] f32: pre | post | comb
    mean: Mem,    // [H] f32, host memory: the streams' mean when bring-up asks for a layer's output
    // the step's buffers, one token deep; a set per part slot, for the parts on this card
    x: Mem,      // [H] f32: the block input, the hyper-connection's collapse
    // block outputs land here (VRAM), and the handoff copies them out to the shared host buffers
    yv: Mem,     // [PARTS][H] f32: the KDA parts' shares of the block output
    fyv: Mem,    // [H] f32: the FFN rows
    myv: Mem,    // [H] f32: the MLA output projection
    scv: Mem,    // [E] f32: the early router's logits
    smixs: Vec<Mem>,   // [PARTS][SMALL_ROWS + 3·PART_QKV] f32: fa | ga | beta | q | k | v, one launch's rows
    fgs: Vec<Mem>,     // [PARTS][2][PART_QKV]: a part's f | gate
    normeds: Vec<Mem>, // [PARTS][PART_QKV]
    snap: Mem,   // [KDA_WINDOW + KDA_MEMORY] f32, host memory: a layer's state on its way out
    // the FFN's and the head's
    fx: Mem,     // [H] f32: the streams' mean, the head's input
    gu: Mem,     // [DENSE_GU_ROWS] f32: gate | up
    lg: Mem,     // [VOCAB_SIZE] f32, host memory: the logits
    // the MLA's
    mqa: Mem,    // [Q_LORA_RANK] f32: q_a's output, the input of q_b and the indexer's query
    mq: Mem,     // [MLA_HEADS·Q_HEAD] f32: the query, the attention's input
    mproj: Mem,  // host memory: kva [KV_LORA_RANK] | ix [IX_ROWS] | iq [INDEX_N_HEADS·INDEX_HEAD_DIM]
    msel: Mem,   // [DENSE_BELOW] u32, host-visible: the positions attended
    mheads: Mem, // [O_IN] f32: the heads' outputs, the output projection's input
}
/// Host memory every card and the host see (`Kfd::host_shared`): what one produces and
/// another consumes within a token. A layer's buffers alternate by its parity, so a
/// producer a layer ahead never overwrites what a consumer has yet to read (the handoff
/// chain keeps them within one layer of each other).
struct Shared {
    ys: [Mem; 2],  // [2][PARTS][H] f32: the KDA parts' shares of the block output, by layer parity
    fy: [Mem; 2],  // [H] f32: the FFN card's rows (the dense MLP or the shared expert)
    my: [Mem; 2],  // [H] f32: the MLA card's output projection
    yexp: [Mem; 2], // [H] f32: the host's routed experts
    xh: Mem,       // [H] f32: a MoE layer's FFN input for the host
    scores: Mem,   // [E] f32: the early router's logits for the host
    scores_ffn: Mem, // [E] f32: the FFN half's router over its input, for the host's picks
    words: Mem,    // the handoff words, a cache line each: `word()`
}
/// A handoff word's byte offset in `Shared::words`: the kind, and the card for a block output.
const fn word(kind: usize, card: usize) -> usize {
    (kind * MAX_CARDS + card) * 64
}
const MAX_CARDS: usize = 4;
/// The attention block's output is in (a card's), the early router's logits are in (the FFN card's),
/// the FFN half's input is in (the FFN card's), the FFN rows are in (the FFN card's), the host's experts are in.
const W_Y: usize = 0;
const W_ROUTE: usize = 1;
const W_X: usize = 2;
const W_FFN: usize = 3;
const W_EXPERTS: usize = 4;
const W_KINDS: usize = 5;
const MKVA: usize = 0;
const MIX: usize = MKVA + KV_LORA_RANK;
const MIQ: usize = MIX + IX_ROWS;
const MPROJ: usize = MIQ + INDEX_N_HEADS * INDEX_HEAD_DIM;
/// Rows of the fused small projection: f_a's, g_a's, b's.
const SMALL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS;

/// One part of a KDA layer — `PART_HEADS` of its heads — on a card: the heads' rows
/// of q, k and v, their f_b and g_b rows, their half of every output projection row
/// (`[H][PART_QKV]`), and the layer's small tensors (every part has its own copy: the
/// input norm, the taps, the gates' first halves and beta rows, dt_bias, a_log,
/// o_norm) with full-width state buffers of which it touches its heads' entries.
struct Part {
    card: usize,
    head0: usize,
    input_ln: Mem,
    sq: Mem,       // [SMALL_ROWS + 3·PART_QKV][H] bf16: f_a's rows, g_a's, b's, then the part's q rows, k rows, v rows — one launch
    o: Mem,        // [H][PART_QKV] bf16
    taps: Mem,     // [3][KDA_QKV][CONV_TAPS] bf16
    fg: Mem,       // [2·PART_QKV][KDA_DIM] bf16: the part's f_b rows, g_b rows
    dt_bias: Mem,
    a_log: Mem,
    o_norm: Mem,
    window: Mem, // [CONV_TAPS-1][CONV_DIM] f32
    mem: Mem,    // [KDA_HEADS][KDA_DIM][KDA_DIM] f32
}
struct Layer {
    parts: Vec<Part>,
}

/// A layer's FFN half on a card: the gate|up rows back to back (the gate's
/// first) with their scales likewise, the down rows, the post norm.
struct FfnOn {
    card: usize,
    dense: bool,
    ln: Mem,
    /// The layer's hyper-connections, on every card: (attention half, FFN half) × (fn rows bf16, base f32, scale f32).
    hc: Vec<[(Mem, Mem, Mem); 2]>,
    gu_w: Mem,
    gu_s: Mem,
    down_w: Mem,
    down_s: Mem,
    /// A MoE layer's router rows (`[E][H]` bf16), for the early routing.
    router: Option<Mem>,
}

/// An MLA layer's projections on a card: q_a and kv_a (FP8, over the input-normed
/// block input), the indexer's key | gate | head weights as one bf16 matrix, q_b (FP8)
/// and the indexer's query (bf16) over q_a's output normed with q_a's norm, the output projection (FP8).
struct MlaOn {
    card: usize,
    input_ln: Mem,
    qa_ln: Mem,
    q_a: (Mem, Mem),
    kv_a: (Mem, Mem),
    ix: Mem,
    q_b: (Mem, Mem),
    ix_wq_b: Mem,
    kv_b: Mem, // [MLA_HEADS][KVB_HEAD][KV_LORA_RANK] bf16: the absorbed attention's W_uk and W_uv per head
    lat: Mem,  // [max_ctx][KV_LORA_RANK] f32: the latents, the host's cache rows mirrored
    o: (Mem, Mem),
}

pub struct Cards {
    cards: Vec<Card>,
    shared: Shared,
    layers: Vec<Option<Layer>>,
    ffns: Vec<Option<FfnOn>>,
    mlas: Vec<Option<MlaOn>>,
    /// The head's card, the final norm (bf16) and lm_head.
    head: (usize, Mem, Mem),
    /// The handoff sequence: a value a half of a layer's words carry, and each layer's (attention, FFN) values this token.
    seq: u32,
    half_seq: Vec<[u32; 2]>,
    /// With the launches stamped: each card's first packet of the token, and the tokens since the last report.
    token_from: Vec<u64>,
    tokens: u32,
}

fn raw<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A bf16-stored norm, widened at load, back to its bf16 bits (exact: the low half is zero).
fn bf16_of(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&v| (v.to_bits() >> 16) as u16).collect()
}

impl Cards {
    /// Open every gfx1100 card, load the code object on each, and place the
    /// blocks on them in order: the KDA layers an equal share each, every
    /// layer's FFN half with the KDA layer at or before it, the head last.
    pub fn new(elf: &[u8], w: &Weights, max_ctx: usize, instrument: bool) -> Self {
        let n = Kfd::cards().len();
        assert!(n > 0, "rockem.gpu: no gfx1100 card");
        let t0 = std::time::Instant::now();
        let cards: Vec<Card> = (0..n)
            .map(|i| {
                let k = Kfd::open(elf, instrument, i);
                let alloc = |floats: usize| k.vram_visible(floats * 4);
                Card {
                    rows_h: k.kernel("rows_h"),
                    rows_d2: k.kernel("rows_d2"),
                    rows_part: k.kernel("rows_part"),
                    step: k.kernel("kda_step"),
                    copy: k.kernel("copy_f32"),
                    ffn_gu: k.kernel("ffn_gu"),
                    ffn_down_shared: k.kernel("ffn_down_shared"),
                    ffn_down_dense: k.kernel("ffn_down_dense"),
                    mla_qb: k.kernel("mla_qb"),
                    mla_iq: k.kernel("mla_iq"),
                    mla_attend: k.kernel("mla_attend"),
                    mla_o: k.kernel("mla_o"),
                    hc_pre: k.kernel("hc_pre"),
                    hc_post: k.kernel("hc_post"),
                    hc_mean: k.kernel("hc_mean"),
                    hc_handoff: k.kernel("hc_handoff"),
                    streams: alloc(HC * H),
                    part: k.vram(COEF_ROWS * HC_SLICES * 4),
                    count: {
                        let m = k.vram_visible(64);
                        k.upload(&m, 0, &[0u8; 64]);
                        m
                    },
                    mix: k.vram(HC_MIX * 4),
                    mean: k.host(H * 4),
                    x: k.vram(H * 4),
                    yv: k.vram(g::PARTS * H * 4),
                    fyv: k.vram(H * 4),
                    myv: k.vram(H * 4),
                    scv: k.vram(glm_5_3_flash::E * 4),
                    smixs: (0..g::PARTS).map(|_| alloc(SMALL_ROWS + 3 * g::PART_QKV)).collect(),
                    fgs: (0..g::PARTS).map(|_| alloc(2 * g::PART_QKV)).collect(),
                    normeds: (0..g::PARTS).map(|_| alloc(g::PART_QKV)).collect(),
                    snap: k.host((KDA_WINDOW + KDA_MEMORY) * 4),
                    fx: k.vram(H * 4),
                    gu: alloc(DENSE_GU_ROWS),
                    lg: k.host(VOCAB_SIZE * 4),
                    mqa: alloc(Q_LORA_RANK),
                    mq: alloc(MLA_HEADS * Q_HEAD),
                    mproj: k.host(MPROJ * 4),
                    msel: alloc(DENSE_BELOW),
                    mheads: alloc(O_IN),
                    k,
                }
            })
            .collect();
        assert!(n <= MAX_CARDS, "{n} cards: the handoff words are laid out for {MAX_CARDS}");
        Kfd::clocks_high();
        // the handoffs: host memory card 0 allocates and every other card maps too
        let shared = {
            let k0 = &cards[0].k;
            let others: Vec<&Kfd> = cards[1..].iter().map(|c| &*c.k).collect();
            let buf = || k0.host_shared(H * 4, &others);
            Shared {
                ys: [k0.host_shared(g::PARTS * H * 4, &others), k0.host_shared(g::PARTS * H * 4, &others)],
                fy: [buf(), buf()],
                my: [buf(), buf()],
                yexp: [buf(), buf()],
                xh: buf(),
                scores: k0.host_shared(glm_5_3_flash::E * 4, &others),
                scores_ffn: k0.host_shared(glm_5_3_flash::E * 4, &others),
                words: {
                    let m = k0.host_shared(word(W_KINDS, 0), &others);
                    unsafe { std::ptr::write_bytes(m.ptr() as *mut u8, 0, m.size) };
                    m
                },
            }
        };
        let kda_layers: Vec<usize> = (0..NUM_HIDDEN_LAYERS).filter(|&l| LINEAR_ATTENTION[l]).collect();
        let per = kda_layers.len().div_ceil(n);
        // a layer's card: that of the KDA layer at or before it (the first card before any)
        let card_of = |l: usize| (0..=l).filter(|&j| LINEAR_ATTENTION[j]).count().saturating_sub(1) / per;
        let mut layers: Vec<Option<Layer>> = (0..NUM_HIDDEN_LAYERS).map(|_| None).collect();
        let mut bytes = 0usize;
        for &l in &kda_layers {
            let Attn::Kda(kw) = &w.layers[l].attn else { unreachable!() };
            let mut taps: Vec<u16> = Vec::with_capacity(g::TAPS);
            for part in kw.conv {
                taps.extend_from_slice(part);
            }
            let input_ln = bf16_of(w.layers[l].input_ln);
            let zero_w = vec![0f32; KDA_WINDOW];
            let zero_m = vec![0f32; KDA_MEMORY];
            let mut parts = Vec::with_capacity(g::PARTS);
            for i in 0..g::PARTS {
                // part i on card i where there is one; the parts of a one-card box share it
                let card = (card_of(l) + i) % n;
                let k = &cards[card].k;
                let up = |b: &[u8]| -> Mem {
                    let m = k.vram_visible(b.len());
                    k.upload(&m, 0, b);
                    m
                };
                let head0 = i * g::PART_HEADS;
                // the part's heads' rows of a per-head-row matrix of width `width`
                fn rows<'a>(m: &'a [u16], width: usize, head0: usize) -> &'a [u16] {
                    &m[head0 * KDA_DIM * width..(head0 * KDA_DIM + g::PART_QKV) * width]
                }
                // the output projection's rows, this part's half of each
                let mut o_half: Vec<u16> = Vec::with_capacity(H * g::PART_QKV);
                for r in 0..H {
                    o_half.extend_from_slice(&kw.o[r * KDA_QKV + i * g::PART_QKV..r * KDA_QKV + (i + 1) * g::PART_QKV]);
                }
                let sq = {
                    let (q, kk, v) = (rows(kw.q, H, head0), rows(kw.k, H, head0), rows(kw.v, H, head0));
                    let m = k.vram_visible((SMALL_ROWS + 3 * g::PART_QKV) * H * 2);
                    let mut off = 0;
                    for part in [kw.f_a, kw.g_a, kw.b, q, kk, v] {
                        k.upload(&m, off, raw(part));
                        off += part.len() * 2;
                    }
                    assert_eq!(off, m.size);
                    m
                };
                let pair = |a: &[u16], b: &[u16]| -> Mem {
                    let m = k.vram_visible((a.len() + b.len()) * 2);
                    k.upload(&m, 0, raw(a));
                    k.upload(&m, a.len() * 2, raw(b));
                    m
                };
                let p = Part {
                    card,
                    head0,
                    input_ln: up(raw(&input_ln)),
                    sq,
                    o: up(raw(&o_half)),
                    taps: up(raw(&taps)),
                    fg: pair(rows(kw.f_b, KDA_DIM, head0), rows(kw.g_b, KDA_DIM, head0)),
                    dt_bias: up(raw(kw.dt_bias)),
                    a_log: up(raw(kw.a_log)),
                    o_norm: up(raw(kw.o_norm)),
                    window: up(raw(&zero_w)),
                    mem: up(raw(&zero_m)),
                };
                bytes += 2 * ((SMALL_ROWS + 3 * g::PART_QKV) * H + H * g::PART_QKV + 2 * g::PART_QKV * KDA_DIM + taps.len()) + 4 * (KDA_WINDOW + KDA_MEMORY);
                parts.push(p);
            }
            layers[l] = Some(Layer { parts });
        }
        // every layer's FFN half but the routed experts
        let mut ffns: Vec<Option<FfnOn>> = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        let (mut dense, mut shared_n) = (0, 0);
        for l in 0..NUM_HIDDEN_LAYERS {
            let card = card_of(l);
            let k = &cards[card].k;
            let pair = |a: Fp8, b: Fp8| -> (Mem, Mem) {
                let wm = k.vram_visible(a.w.len() + b.w.len());
                k.upload(&wm, 0, a.w);
                k.upload(&wm, a.w.len(), b.w);
                let sm = k.vram_visible((a.s.len() + b.s.len()) * 4);
                k.upload(&sm, 0, raw(a.s));
                k.upload(&sm, a.s.len() * 4, raw(b.s));
                (wm, sm)
            };
            let one = |a: Fp8| -> (Mem, Mem) {
                let wm = k.vram_visible(a.w.len());
                k.upload(&wm, 0, a.w);
                let sm = k.vram_visible(a.s.len() * 4);
                k.upload(&sm, 0, raw(a.s));
                (wm, sm)
            };
            let ln = {
                let m = k.vram_visible(H * 2);
                k.upload(&m, 0, raw(&bf16_of(w.layers[l].post_ln)));
                m
            };
            let (is_dense, (gu_w, gu_s), (down_w, down_s), router) = match &w.layers[l].ffn {
                Ffn::Dense(d) => (true, pair(d.gate, d.up), one(d.down), None),
                Ffn::Moe(m) => {
                    let r = k.vram_visible(m.router.len() * 2);
                    k.upload(&r, 0, raw(m.router));
                    (false, pair(m.shared_gate, m.shared_up), one(m.shared_down), Some(r))
                }
            };
            if is_dense {
                dense += 1;
            } else {
                shared_n += 1;
            }
            bytes += gu_w.size + gu_s.size + down_w.size + down_s.size + ln.size + router.as_ref().map_or(0, |r| r.size);
            // the layer's hyper-connections on every card: each runs them over its own streams
            let hc: Vec<[(Mem, Mem, Mem); 2]> = cards
                .iter()
                .map(|c| {
                    let one = |h: &glm_5_3_flash::weights::Hc| {
                        let f = c.k.vram_visible(h.fn_.len() * 2);
                        c.k.upload(&f, 0, raw(h.fn_));
                        let b = c.k.vram_visible(h.base.len() * 4);
                        c.k.upload(&b, 0, raw(h.base));
                        let sc = c.k.vram_visible(h.scale.len() * 4);
                        c.k.upload(&sc, 0, raw(h.scale));
                        (f, b, sc)
                    };
                    [one(&w.layers[l].hc_attn), one(&w.layers[l].hc_ffn)]
                })
                .collect();
            bytes += hc.iter().flatten().map(|(f, b, sc)| f.size + b.size + sc.size).sum::<usize>();
            ffns.push(Some(FfnOn { card, dense: is_dense, ln, hc, gu_w, gu_s, down_w, down_s, router }));
        }
        // the MLA layers' projections
        let mut mlas: Vec<Option<MlaOn>> = (0..NUM_HIDDEN_LAYERS).map(|_| None).collect();
        let mut mla_n = 0;
        for l in 0..NUM_HIDDEN_LAYERS {
            let Attn::Mla(mw) = &w.layers[l].attn else { continue };
            let card = card_of(l);
            let k = &cards[card].k;
            let one = |a: Fp8| -> (Mem, Mem) {
                let wm = k.vram_visible(a.w.len());
                k.upload(&wm, 0, a.w);
                let sm = k.vram_visible(a.s.len() * 4);
                k.upload(&sm, 0, raw(a.s));
                (wm, sm)
            };
            let up = |b: &[u8]| -> Mem {
                let m = k.vram_visible(b.len());
                k.upload(&m, 0, b);
                m
            };
            let ix = {
                let m = k.vram_visible(IX_ROWS * H * 2);
                k.upload(&m, 0, raw(mw.ix_wk));
                k.upload(&m, mw.ix_wk.len() * 2, raw(mw.ix_gate));
                k.upload(&m, (mw.ix_wk.len() + mw.ix_gate.len()) * 2, raw(mw.ix_weights));
                m
            };
            let on = MlaOn { card, input_ln: up(raw(&bf16_of(w.layers[l].input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lat: k.vram_visible(max_ctx * KV_LORA_RANK * 4), o: one(mw.o) };
            bytes += on.q_a.0.size + on.q_a.1.size + on.kv_a.0.size + on.kv_a.1.size + on.ix.size + on.q_b.0.size + on.q_b.1.size + on.ix_wq_b.size + on.kv_b.size + on.lat.size + on.o.0.size + on.o.1.size;
            mlas[l] = Some(on);
            mla_n += 1;
        }
        // the head with the last layer
        let head = {
            let card = card_of(NUM_HIDDEN_LAYERS - 1);
            let k = &cards[card].k;
            let norm = k.vram_visible(H * 2);
            k.upload(&norm, 0, raw(&bf16_of(w.norm)));
            let lm = k.vram_visible(w.lm_head.len() * 2);
            k.upload(&lm, 0, raw(w.lm_head));
            bytes += norm.size + lm.size;
            (card, norm, lm)
        };
        println!("gpu      : {} KDA layers, {mla_n} MLA layers (their latents to {max_ctx} positions), {dense} dense FFNs, {shared_n} shared experts, the hyper-connections, the streams and the head on {} card{} ({:.1} GB with their state) in {:.1}s", kda_layers.len(), n, if n == 1 { "" } else { "s" }, bytes as f64 / 1e9, t0.elapsed().as_secs_f32());
        Cards { cards, shared, layers, ffns, mlas, head, seq: 0, half_seq: vec![[0; 2]; NUM_HIDDEN_LAYERS], token_from: vec![0; n], tokens: 0 }
    }

    /// With the launches stamped, every 16 tokens: a line per card of the last token's time by
    /// kernel (busy µs and count, most first) and the gaps between launches — where a token's
    /// card time goes.
    fn timeline_report(&mut self) {
        if !self.cards[0].k.instrumented() {
            return;
        }
        self.tokens += 1;
        if self.tokens % 16 != 0 {
            return;
        }
        for (i, c) in self.cards.iter().enumerate() {
            let tl = c.k.timeline_from(self.token_from[i]);
            let mut by: std::collections::BTreeMap<&str, (f64, usize)> = std::collections::BTreeMap::new();
            let (mut busy, mut gaps, mut prev_end) = (0f64, 0f64, 0u64);
            for &(name, s, e) in &tl {
                if e <= s {
                    continue;
                }
                let us = (e - s) as f64 / 100.0;
                if prev_end != 0 && s > prev_end {
                    gaps += (s - prev_end) as f64 / 100.0;
                }
                prev_end = e;
                busy += us;
                let ent = by.entry(if name.is_empty() { "barrier" } else { name }).or_insert((0.0, 0));
                ent.0 += us;
                ent.1 += 1;
            }
            let span = tl.last().map_or(0, |l| l.2).saturating_sub(tl.first().map_or(0, |f| f.1)) as f64 / 100.0;
            let mut v: Vec<_> = by.into_iter().collect();
            v.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            let parts: Vec<String> = v.iter().map(|(n, (us, cnt))| format!("{n} {us:.0}/{cnt}")).collect();
            println!("timeline : card {i}: {} launches over {span:.0} µs, {busy:.0} busy, {gaps:.0} between; by kernel (µs/count): {}", tl.len(), parts.join(", "));
        }
    }

    /// The next handoff value: never 0, which the words start at.
    fn next_seq(&mut self) -> u32 {
        self.seq = self.seq.wrapping_add(1).max(1);
        self.seq
    }
    /// The host-side atomic of a handoff word.
    fn word_ptr(&self, kind: usize, card: usize) -> *const AtomicU32 {
        self.shared.words.at(word(kind, card)) as *const AtomicU32
    }
    /// Card `c`'s handoff launch: `value` into word `mine` (when given) once everything before
    /// it is done, then the queue holds until the words `w0` and `w1` (those given) read `value`.
    fn handoff(&self, c: &Card, copy: Option<(usize, usize, usize)>, mine: Option<(usize, usize)>, w0: Option<(usize, usize)>, w1: Option<(usize, usize)>, value: u32) {
        let at = |w: Option<(usize, usize)>| w.map_or(0, |(k, d)| self.shared.words.at(word(k, d)));
        let (src, dst, n) = copy.unwrap_or((0, 0, 0));
        let mut a = Args::new();
        a.ptr(src).ptr(dst).u32(n as u32).ptr(at(mine)).ptr(at(w0)).ptr(at(w1)).u32(value);
        c.k.launch(&c.hc_handoff, 1, g::WG, 0, &a);
    }
    /// The hyper-connection's input half on card `c`, one launch: the coefficients over its streams,
    /// the mix, the collapse into the block input `x` — and into `x_host` for the routed experts when
    /// given (the host's word follows the router's logits, in the handoff after them).
    fn hc_pre(&self, c: &Card, hc: &(Mem, Mem, Mem), x_host: Option<usize>) {
        let mut a = Args::new();
        a.ptr(c.streams.va).ptr(hc.0.va).ptr(c.part.va).ptr(hc.1.va).ptr(hc.2.va).ptr(c.mix.va).ptr(c.x.va).ptr(x_host.unwrap_or(0)).ptr(c.count.va);
        c.k.launch(&c.hc_pre, PRE_GRID, g::WG, PRE_LDS, &a);
    }
    /// The hyper-connection's output half on card `c`, the handoff first: this card's `copy`
    /// (src, dst, floats) out to shared memory and `value` into its word `mine`, then the queue
    /// holds for `w0` and `w1`, then the block output (`ya` + `yb`) into its streams. One launch.
    #[allow(clippy::too_many_arguments)]
    fn hc_post(&self, c: &Card, copy: Option<(usize, usize, usize)>, mine: Option<(usize, usize)>, w0: Option<(usize, usize)>, w1: Option<(usize, usize)>, value: u32, ya: usize, yb: Option<usize>) {
        let at = |w: Option<(usize, usize)>| w.map_or(0, |(k, d)| self.shared.words.at(word(k, d)));
        let (src, dst, n) = copy.unwrap_or((0, 0, 0));
        let mut a = Args::new();
        a.ptr(c.mix.va).ptr(ya).ptr(yb.unwrap_or(0)).ptr(c.streams.va).ptr(src).ptr(dst).u32(n as u32).ptr(at(mine)).ptr(at(w0)).ptr(at(w1)).u32(value);
        c.k.launch(&c.hc_post, CHANNEL_GRID, g::WG, 0, &a);
    }
    /// The early router on the FFN card: layer `l`'s FFN router over the attention block's input normed with the
    /// FFN half's norm, its logits to the host with the half's word.
    fn route_early(&self, l: usize, value: u32) {
        let f = self.ffns[l].as_ref().unwrap();
        if let Some(router) = &f.router {
            let c = &self.cards[f.card];
            self.rows(c, &c.rows_h, c.x.va, Some(f.ln.va), router.va, c.scv.va, glm_5_3_flash::E, H);
            self.handoff(c, Some((c.scv.va, self.shared.scores.va, glm_5_3_flash::E)), Some((W_ROUTE, f.card)), None, None, value);
        }
    }
    /// A KDA block's output into every card's streams: each card copies the shares it made out to
    /// the shared buffer and says so (its word), waits for the producers' words, and folds the sum in.
    /// `parts[c]`: the part slots card `c` holds, contiguous.
    fn attention_out(&self, value: u32, parity: usize, parts: &[Vec<usize>]) {
        let producers: Vec<usize> = (0..self.cards.len()).filter(|&c| !parts[c].is_empty()).collect();
        let ys = &self.shared.ys[parity];
        for c in &self.cards {
            let mine = &parts[c.k.index];
            let copy = mine.first().map(|&p0| (c.yv.at(p0 * H * 4), ys.at(p0 * H * 4), mine.len() * H));
            self.hc_post(c, copy, (!mine.is_empty()).then_some((W_Y, c.k.index)), Some((W_Y, producers[0])), producers.get(1).map(|&p| (W_Y, p)), value, ys.va, Some(ys.at(H * 4)));
        }
    }

    /// One FP8 rows launch on card `c`: `out[row] = W[row] · x` (normed with `ln` when given) for `m` rows of width `k_width`, scales `s`.
    fn rows8(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize) {
        self.rows8_at(c, kernel, x, ln, w, s, out, m, k_width, true)
    }
    fn rows8_free(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize) {
        self.rows8_at(c, kernel, x, ln, w, s, out, m, k_width, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn rows8_at(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize, barrier: bool) {
        let mut a = Args::new();
        a.ptr(x);
        if let Some(ln) = ln {
            a.ptr(ln);
        }
        a.ptr(w).ptr(s).ptr(out).u32(m as u32).u32(1);
        let lds = g::fp8_rows_lds(k_width, 1) as u32;
        if barrier {
            c.k.launch(kernel, g::rows_grid(m), g::WG, lds, &a);
        } else {
            c.k.launch_free(kernel, g::rows_grid(m), g::WG, lds, &a);
        }
    }

    /// One rows launch on card `c`: `out[row] = W[row] · x` (normed with `ln` when given) for `m` rows of width `k_width`. Addresses are device VAs.
    fn rows(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize) {
        self.rows_at(c, kernel, x, ln, w, out, m, k_width, true)
    }
    /// `rows` launched free: as soon as the launch before it has launched, so the two run together —
    /// for a launch whose inputs were complete when the one before it launched.
    fn rows_free(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize) {
        self.rows_at(c, kernel, x, ln, w, out, m, k_width, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn rows_at(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, barrier: bool) {
        let mut a = Args::new();
        a.ptr(x);
        if let Some(ln) = ln {
            a.ptr(ln);
        }
        a.ptr(w).ptr(out).u32(m as u32).u32(1);
        let lds = if ln.is_some() { g::normed_rows_lds(k_width, 1) } else { g::rows_lds(k_width, 1) } as u32;
        if barrier {
            c.k.launch(kernel, g::rows_grid(m), g::WG, lds, &a);
        } else {
            c.k.launch_free(kernel, g::rows_grid(m), g::WG, lds, &a);
        }
    }
}

impl Placed for Cards {
    fn begin(&mut self, row: &[f32]) {
        // every card: the token's handoffs restart, the embedding into each stream
        let mut streams = Vec::with_capacity(HC * H);
        for _ in 0..HC {
            streams.extend_from_slice(&row[..H]);
        }
        for (i, c) in self.cards.iter().enumerate() {
            c.k.upload(&c.streams, 0, raw(&streams));
            self.token_from[i] = c.k.posted();
        }
    }
    fn attention(&mut self, l: usize) {
        let value = self.next_seq();
        self.half_seq[l][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], None);
        }
        self.route_early(l, value);
        let lay = self.layers[l].as_ref().unwrap();
        let parity = l % 2;
        for (i, p) in lay.parts.iter().enumerate() {
            let c = &self.cards[p.card];
            let (smix, fg, normed) = (&c.smixs[i], &c.fgs[i], &c.normeds[i]);
            // the gates' first halves and beta, then q | k | v: one launch over the normed block input
            self.rows(c, &c.rows_h, c.x.va, Some(p.input_ln.va), p.sq.va, smix.va, SMALL_ROWS + 3 * g::PART_QKV, H);
            // the gates' second halves as one launch: f_b over f_a's output, g_b over g_a's
            let mut a = Args::new();
            a.ptr(smix.va).ptr(p.fg.va).ptr(fg.va).u32(2 * g::PART_QKV as u32).u32(1).u32(g::PART_QKV as u32);
            c.k.launch(&c.rows_d2, g::rows_grid(2 * g::PART_QKV), g::WG, g::rows_lds(KDA_DIM, 1) as u32, &a);
            let mixed = smix.at(SMALL_ROWS * 4);
            let mut a = Args::new();
            a.ptr(mixed).ptr(p.taps.va).ptr(p.window.va).ptr(fg.va).ptr(p.dt_bias.va).ptr(p.a_log.va).ptr(smix.at(2 * KDA_DIM * 4)).ptr(fg.at(g::PART_QKV * 4)).ptr(p.mem.va).ptr(p.o_norm.va).ptr(normed.va).u32(p.head0 as u32).u32(1);
            c.k.launch(&c.step, g::STEP_GRID, g::WG, g::STEP_LDS as u32, &a);
            self.rows(c, &c.rows_part, normed.va, None, p.o.va, c.yv.at(i * H * 4), H, g::PART_QKV);
        }
        // each card's shares out and its word; every card folds the sum into its streams
        let mut parts: Vec<Vec<usize>> = vec![Vec::new(); self.cards.len()];
        for (i, p) in lay.parts.iter().enumerate() {
            parts[p.card].push(i);
        }
        self.attention_out(value, parity, &parts);
    }
    fn early_logits(&mut self, l: usize, logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_ROUTE, f.card), self.half_seq[l][0]);
        read(&self.shared.scores, 0, &mut logits[..glm_5_3_flash::E]);
    }
    fn mla_post(&mut self, l: usize) {
        let value = self.next_seq();
        self.half_seq[l][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], None);
        }
        self.route_early(l, value);
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        let ln = Some(m.input_ln.va);
        // over the normed block input: q_a into VRAM for the next two, kv_a and the indexer's three out to the host
        self.rows8(c, &c.ffn_gu, c.x.va, ln, m.q_a.0.va, m.q_a.1.va, c.mqa.va, Q_LORA_RANK, H);
        self.rows8_free(c, &c.ffn_gu, c.x.va, ln, m.kv_a.0.va, m.kv_a.1.va, c.mproj.at(MKVA * 4), KV_LORA_RANK, H);
        self.rows_free(c, &c.rows_h, c.x.va, ln, m.ix.va, c.mproj.at(MIX * 4), IX_ROWS, H);
        // over q_a's output normed with q_a's norm: the query and, alongside it, the indexer's query
        let qln = Some(m.qa_ln.va);
        self.rows8(c, &c.mla_qb, c.mqa.va, qln, m.q_b.0.va, m.q_b.1.va, c.mq.va, MLA_HEADS * Q_HEAD, Q_LORA_RANK);
        self.rows_free(c, &c.mla_iq, c.mqa.va, qln, m.ix_wq_b.va, c.mproj.at(MIQ * 4), INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK);
        c.k.fence();
        // the other cards fold the output in once it is there (posted now: they wait for the word)
        for (i, o) in self.cards.iter().enumerate() {
            if i != m.card {
                self.hc_post(o, None, None, Some((W_Y, m.card)), None, value, self.shared.my[l % 2].va, None);
            }
        }
    }
    fn mla_wait(&mut self, l: usize, kva: &mut [f32], ik: &mut [f32], igate: &mut [f32], iq: &mut [f32], iw: &mut [f32]) {
        let c = &self.cards[self.mlas[l].as_ref().unwrap().card];
        c.k.wait();
        read(&c.mproj, MKVA * 4, &mut kva[..KV_LORA_RANK]);
        read(&c.mproj, MIX * 4, &mut ik[..INDEX_HEAD_DIM]);
        read(&c.mproj, (MIX + INDEX_HEAD_DIM) * 4, &mut igate[..INDEX_HEAD_DIM]);
        read(&c.mproj, (MIX + 2 * INDEX_HEAD_DIM) * 4, &mut iw[..INDEX_N_HEADS]);
        read(&c.mproj, MIQ * 4, &mut iq[..INDEX_N_HEADS * INDEX_HEAD_DIM]);
    }
    fn mla_attend(&mut self, l: usize, pos: usize, latent: &[f32], sel: &[usize]) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the position's latent into the mirror, the selection, the heads, the output projection, the streams
        c.k.upload(&m.lat, pos * KV_LORA_RANK * 4, raw(&latent[..KV_LORA_RANK]));
        let sel32: Vec<u32> = sel.iter().map(|&p| p as u32).collect();
        c.k.upload(&c.msel, 0, raw(&sel32));
        let mut a = Args::new();
        a.ptr(c.mq.va).ptr(m.kv_b.va).ptr(m.lat.va).ptr(c.msel.va).u32(sel.len() as u32).ptr(c.mheads.va);
        c.k.launch(&c.mla_attend, ATTEND_GRID, g::WG, ATTEND_LDS as u32, &a);
        let my = self.shared.my[l % 2].va;
        self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.myv.va, H, O_IN);
        self.hc_post(c, Some((c.myv.va, my, H)), Some((W_Y, m.card)), None, None, self.half_seq[l][0], my, None);
    }
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[f32]) {
        let m = self.mlas[l].as_ref().unwrap();
        self.cards[m.card].k.upload(&m.lat, pos * KV_LORA_RANK * 4, raw(&latent[..KV_LORA_RANK]));
    }
    fn ffn(&mut self, l: usize) {
        let value = self.next_seq();
        self.half_seq[l][1] = value;
        let f = self.ffns[l].as_ref().unwrap();
        let parity = l % 2;
        for c in &self.cards {
            // the FFN card's collapse goes to the host too when the routed experts need it
            let to_host = (!f.dense && c.k.index == f.card).then_some(self.shared.xh.va);
            self.hc_pre(c, &f.hc[c.k.index][1], to_host);
        }
        let c = &self.cards[f.card];
        if let Some(router) = &f.router {
            // the router over the normed FFN input, its logits to the host with the input's word
            self.rows(c, &c.rows_h, c.x.va, Some(f.ln.va), router.va, c.scv.va, glm_5_3_flash::E, H);
            self.handoff(c, Some((c.scv.va, self.shared.scores_ffn.va, glm_5_3_flash::E)), Some((W_X, f.card)), None, None, value);
        }
        let (rows, down, k_width) = if f.dense { (DENSE_GU_ROWS, &c.ffn_down_dense, INTERMEDIATE_SIZE) } else { (SHARED_GU_ROWS, &c.ffn_down_shared, N_SHARED_EXPERTS * D_FF) };
        let fy = self.shared.fy[parity].va;
        self.rows8(c, &c.ffn_gu, c.x.va, Some(f.ln.va), f.gu_w.va, f.gu_s.va, c.gu.va, rows, H);
        self.rows8(c, down, c.gu.va, None, f.down_w.va, f.down_s.va, c.fyv.va, H, k_width);
        for o in &self.cards {
            let producer = o.k.index == f.card;
            let copy = producer.then_some((o.fyv.va, fy, H));
            let mine = producer.then_some((W_FFN, f.card));
            if f.dense {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), None, value, fy, None);
            } else {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), Some((W_EXPERTS, 0)), value, fy, Some(self.shared.yexp[parity].va));
            }
        }
    }
    fn ffn_input(&mut self, l: usize, x: &mut [f32], logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_X, f.card), self.half_seq[l][1]);
        read(&self.shared.xh, 0, &mut x[..H]);
        read(&self.shared.scores_ffn, 0, &mut logits[..glm_5_3_flash::E]);
    }
    fn ffn_routed(&mut self, l: usize, y: &[f32]) {
        let buf = &self.shared.yexp[l % 2];
        unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), buf.ptr() as *mut f32, H) };
        unsafe { &*self.word_ptr(W_EXPERTS, 0) }.store(self.half_seq[l][1], std::sync::atomic::Ordering::Release);
    }
    fn layer_out(&mut self, _l: usize, mean: &mut [f32]) {
        let c = &self.cards[0];
        let mut a = Args::new();
        a.ptr(c.streams.va).ptr(c.mean.va);
        c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
        c.k.sync();
        read(&c.mean, 0, &mut mean[..H]);
    }
    fn logits(&mut self, logits: &mut [f32]) {
        let (card, norm, lm) = &self.head;
        let c = &self.cards[*card];
        let mut a = Args::new();
        a.ptr(c.streams.va).ptr(c.fx.va);
        c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
        self.rows(c, &c.rows_h, c.fx.va, Some(norm.va), lm.va, c.lg.va, VOCAB_SIZE, H);
        // the token ends when every card is done with its streams: the next begins by overwriting them
        for c in &self.cards {
            c.k.sync();
        }
        read(&c.lg, 0, &mut logits[..VOCAB_SIZE]);
        self.timeline_report();
    }
    fn seed(&mut self, l: usize, window: &[f32], mem: &[f32]) {
        // every part holds the full state buffers; each touches only its heads' entries
        for p in &self.layers[l].as_ref().unwrap().parts {
            let k = &self.cards[p.card].k;
            k.upload(&p.window, 0, raw(&window[..KDA_WINDOW]));
            k.upload(&p.mem, 0, raw(&mem[..KDA_MEMORY]));
        }
    }
    fn snapshot(&self, l: usize, window: &mut [f32], mem: &mut [f32]) {
        // each part's state out through host memory (a copy launch), its heads' entries into the layer's
        for p in &self.layers[l].as_ref().unwrap().parts {
            let c = &self.cards[p.card];
            let copy = |src: &Mem, dst_off: usize, n: usize| {
                let mut a = Args::new();
                a.ptr(src.va).ptr(c.snap.at(dst_off * 4)).u32(n as u32);
                c.k.launch(&c.copy, n.div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &a);
            };
            copy(&p.window, 0, KDA_WINDOW);
            copy(&p.mem, KDA_WINDOW, KDA_MEMORY);
            c.k.sync();
            let (h0, h1) = (p.head0 * KDA_DIM, (p.head0 + g::PART_HEADS) * KDA_DIM);
            for tap in 0..CONV_TAPS - 1 {
                for part in 0..3 {
                    let at = tap * CONV_DIM + part * KDA_QKV;
                    read(&c.snap, (at + h0) * 4, &mut window[at + h0..at + h1]);
                }
            }
            let (m0, m1) = (p.head0 * KDA_DIM * KDA_DIM, (p.head0 + g::PART_HEADS) * KDA_DIM * KDA_DIM);
            read(&c.snap, (KDA_WINDOW + m0) * 4, &mut mem[m0..m1]);
        }
    }
}

const _: () = assert!(CONV_DIM == 3 * KDA_QKV && CONV_TAPS == 4);
