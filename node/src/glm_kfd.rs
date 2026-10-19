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
use glm_5_3_flash::hc::gfx11_consts::{CHANNEL_GRID, MIX as HC_MIX, PRE_COUNTS, PRE_GRID, PRE_LDS, PRE_STRIDE, ROUTE_ROWS};
use glm_5_3_flash::kda::gfx11_consts as g;
use common::gemv::ROWS;
use glm_5_3_flash::ffn::gfx11_consts::ROWS_GU;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::mla::gfx11_consts::{aw_chunk, aw_chunks, aw_grid, ixw_grid, select_slices, select_state_init, AW_DECODE_CHUNKS, AW_HEAD_TILES, AW_LDS, AW_MERGE_LDS, AW_PART, AW_WG, IXW_LDS, IX_ROWS, O_IN, ROWS_QB, SELECT_LDS, SELECT_STATE};
use glm_5_3_flash::expert::gfx11_consts::{DOWN_WGS as XPT_DOWN_WGS, SLOT_BYTES, UP_WGS as XPT_UP_WGS, XG_DOWN_WGS, XG_GU_WGS, XG_MAX_JOBS, XPT_LDS, XQ_GROUPS, XQ_WGS};
use glm_5_3_flash::DENSE_BELOW;
use glm_5_3_flash::TOP_K;
use glm_5_3_flash::{INDEX_HEAD_DIM, INDEX_N_HEADS, KV_LORA_RANK, Q_LORA_RANK, V_HEAD_DIM};
use glm_5_3_flash::weights::{Attn, Ffn, Fp8, Weights};
use glm_5_3_flash::{CONV_DIM, CONV_TAPS, D_FF, H, HC, INDEX_KPOOL, INTERMEDIATE_SIZE, SPEC_TOKENS, KDA_DIM, KDA_HEADS, KDA_QKV, LINEAR_ATTENTION, MLA_HEADS, NUM_HIDDEN_LAYERS, N_SHARED_EXPERTS, Q_HEAD, STEP, TOKENS, VOCAB_SIZE};
use std::sync::atomic::AtomicU32;
use std::rc::Rc;

/// The routed experts resident on the cards: a slot a card holds one expert's bytes (`SLOT_BYTES`);
/// a layer may hold `per_layer` slots a card, filled on first use and then replaced least-frequently-used
/// with a margin, so the set tracks the conversation (a static hotlist does not travel between texts:
/// 0.19 of picks, 2026-10-17). An expert's bits are the same on either tier (`expert::gfx11`), so the
/// placement never changes a result. Sized for the VRAM free at the configured context.
struct Resident {
    /// `[MOE_LAYERS][E]`: the card and slot an expert sits in (`card << 24 | slot`), `PENDING` while its slot
    /// is being filled, or `NONE`. Shared with the loader, which writes the slot once the bytes are there.
    table: std::sync::Arc<Vec<AtomicU32>>,
    loader: Option<std::sync::mpsc::Sender<(usize, u32, usize, usize)>>,
    /// Loads queued and not yet landed; at most `LOADS_IN_FLIGHT` (the rest stay on the host this step).
    inflight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// Loads the loader may make, against `issued` (the loads it has made): raised by one a decode step —
    /// a loader streaming flat out doubled every card lap (its BAR writes share the link with the handoffs)
    /// and the expert phase (its core and its DRAM reads come out of the pool's), 2026-10-17 — and without
    /// bound inside a prefill step, where the cards idle through the CPU's expert phase and decode begins
    /// with the slots the prompt's own routing chose (063, 2026-10-18).
    allowed: std::sync::Arc<std::sync::atomic::AtomicU64>,
    issued: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Steps begun (the first MoE layer's): the counts' aging and the layer turn run on it.
    steps: u64,
    /// Whether this step is a prefill step (wider than a verify step): its claims come after each layer's
    /// join, when no job of the layer is reading a slot, so a resident in use this step can still be replaced.
    prefill: bool,
    /// Picks seen, `[MOE_LAYERS][E]`, halved every `RESIDENT_HALF_LIFE` steps.
    count: Vec<u32>,
    /// The layer whose misses claim first this step (`steps % MOE_LAYERS`): the one load a step goes to the
    /// earliest layer with a miss, so without a turn the first layers took every load and the rest stayed
    /// empty — 3–4 % of picks resident after 2,600 steps (2026-10-18 10:05).
    turn: usize,
    /// Per card: each slot's (layer, expert), or `NONE`; the free slots; the slots a layer holds.
    owner: Vec<Vec<(u32, u32)>>,
    free: Vec<Vec<u32>>,
    held: Vec<Vec<u32>>,
    per_layer: Vec<usize>,
    /// Per card: the slots' bytes.
    slots: Vec<Mem>,
    /// The jobs in flight a card for the layer being joined: (slot, token, k).
    jobs: Vec<Vec<(u32, u32, u8)>>,
    /// Uploads made (bring-up: the warm-up's cost).
    uploads: usize,
}
const NONE: u32 = u32::MAX;
const PENDING: u32 = u32::MAX - 1;
const LOADS_IN_FLIGHT: usize = 1;
/// A prefill step's claims a MoE layer: the layer's most-picked experts not resident, free slots first, then
/// up to `PREFILL_REPLACES` of the layer's least-picked residents replaced past the margin (063).
const PREFILL_CLAIMS: usize = 16;
const PREFILL_REPLACES: usize = 4;
/// The loader's core: a core on the cards' socket (the KFD topology puts the cards on NUMA nodes 2 and 3;
/// `bench-p2p` 2026-10-18: a 13 MB BAR write runs at 21.6 GB/s from nodes 2 / 3 and at 10–17 from nodes
/// 0 / 1), other than the main thread's (core 0): a slow 13 MB write over a degraded link held the main
/// thread's core for a step at a time (2026-10-17). The pool takes every physical core, so the loader
/// shares this one with a worker whichever it is — and the socket matters more than the BAR rate: on core
/// 48 (node 2, the cards' socket, the slower memory) the expert lap read 9.9–10.5 ms a step token and the
/// route lap 2.0 against 9.2–9.8 and 1.3 on core 24 (2026-10-18 09:50, three boots): the loader's reads
/// slow the pool's slowest node, which bounds the phase. Core 24 it stays.
const LOADER_CORE: usize = 24;
/// Replace a resident expert of the layer only when the newcomer has been picked this many more times.
/// The counts age: halved every `RESIDENT_HALF_LIFE` steps (16 steps are the 32 positions measured below at draft depth 1; a prefill of 512-token steps ages over 8k positions). The routing replay (2026-10-18,
/// `bringup/gates/routing_replay.py` on the code and chat traces, 21 / 18 slots a layer): lifetime counts
/// with a margin of 4 hit 0.406 / 0.285 (code / chat) and 0.374 / 0.260; a half-life of 32 and a margin
/// of 2 hit 0.438 / 0.364 and 0.405 / 0.335 — the chat text is the one where recency counts (LRU 0.348,
/// ARC 0.366 there, but 0.294 / 0.345 on code). Belady's ceiling: 0.552 / 0.577.
const RESIDENT_MARGIN: u32 = 2;
const RESIDENT_HALF_LIFE: u64 = 16;
/// VRAM a card may spend on resident experts at most (the node); what is free past a reserve decides.
pub const RESIDENT_BYTES: usize = 12 << 30;
/// VRAM left unallocated on a card: the driver's and the queue's own, and room for a bring-up gate's buffers.
const RESIDENT_RESERVE: usize = 1 << 30;

struct Card {
    k: Rc<Kfd>,
    xpt_gu: crate::kfd::Kernel,
    xpt_down: crate::kfd::Kernel,
    /// The FFN input's host-side work done here (001/064): norm, quantize, router pick; `xcount` its arrival word.
    ffn_act: crate::kfd::Kernel,
    xcount: Mem,
    xpt_gemm_gu: crate::kfd::Kernel,
    xpt_quant: crate::kfd::Kernel,
    xpt_gemm_down: crate::kfd::Kernel,
    /// The resident experts' work (038): the step's quantized activations and the group/job tables
    /// (host-visible), the jobs' gate/up state and quantized hidden states (VRAM); the outputs go to
    /// `Shared::xout`.
    xq: Mem,
    xs: Mem,
    xg: Mem,
    xjobs: Mem,
    xgu: Mem,
    xhq: Mem,
    xhs: Mem,
    xhg: Mem,
    rows_h: crate::kfd::Kernel,
    rows_h_w: crate::kfd::Kernel,
    rows_d2: crate::kfd::Kernel,
    rows_part: crate::kfd::Kernel,
    rows_part_w: crate::kfd::Kernel,
    mtp_eh_w: crate::kfd::Kernel,
    ffn_gu_w: crate::kfd::Kernel,
    ffn_down_shared_w: crate::kfd::Kernel,
    ffn_down_dense_w: crate::kfd::Kernel,
    mla_qb_w: crate::kfd::Kernel,
    mla_iq_w: crate::kfd::Kernel,
    mla_o_w: crate::kfd::Kernel,
    step: crate::kfd::Kernel,
    copy: crate::kfd::Kernel,
    ffn_gu: crate::kfd::Kernel,
    ffn_down_shared: crate::kfd::Kernel,
    ffn_down_dense: crate::kfd::Kernel,
    mla_qb: crate::kfd::Kernel,
    mla_iq: crate::kfd::Kernel,
    mla_qabs: crate::kfd::Kernel,
    mla_qabs_w: crate::kfd::Kernel,
    mla_attend_w: crate::kfd::Kernel,
    mla_attend_merge: crate::kfd::Kernel,
    mla_uv: crate::kfd::Kernel,
    mla_uv_w: crate::kfd::Kernel,
    ix_score_w: crate::kfd::Kernel,
    ix_select: crate::kfd::Kernel,
    add2: crate::kfd::Kernel,
    mtp_eh: crate::kfd::Kernel,
    mla_o: crate::kfd::Kernel,
    hc_pre: crate::kfd::Kernel,
    hc_post: crate::kfd::Kernel,
    hc_mean: crate::kfd::Kernel,
    // the residual streams and the hyper-connection's scratch: this card's copy
    streams: Mem, // [HC][H] f32, host-visible: the embedding goes in through the BAR
    part: Mem,    // [PRE_PARTS] f32: the coefficient GEMV's partials and the norm's
    count: Mem,   // [PRE_COUNTS] u32: `hc_pre`'s arrival counts, 0 between launches
    go: Mem,      // one u32: `hc_post`'s release word for its workgroups
    mix: Mem,     // [MIX] f32: pre | post | comb
    mean: Mem,    // [H] f32, host memory: the streams' mean when bring-up asks for a layer's output
    // the step's buffers, `STEP` tokens deep (token i's row at i · width); a set per part slot, for the parts on this card
    x: Mem,      // [STEP][H] f32: the block input, the hyper-connection's collapse
    // block outputs land here (VRAM), and the handoff copies them out to the shared host buffers
    yv: Mem,     // [PARTS][STEP][H] f32: the KDA parts' shares of the block output
    fyv: Mem,    // [STEP][H] f32: the FFN rows
    myv: Mem,    // [STEP][H] f32: the MLA output projection
    smixs: Vec<Mem>,   // [PARTS][STEP][SMALL_ROWS + 3·PART_QKV] f32: fa | ga | beta | q | k | v, one launch's rows
    fgs: Vec<Mem>,     // [PARTS][STEP][2][PART_QKV]: a part's f | gate
    normeds: Vec<Mem>, // [PARTS][STEP][PART_QKV]
    snap: Mem,   // [KDA_WINDOW + KDA_MEMORY] f32, host memory: a layer's state on its way out
    // the FFN's and the head's
    fx: Mem,     // [TOKENS][H] f32: the streams' mean, the head's input, a pass of tokens
    gu: Mem,     // [STEP][DENSE_GU_ROWS] f32: gate | up
    lg: Mem,     // [TOKENS][VOCAB_SIZE] f32, host memory: a pass of tokens' logits
    // the MLA's
    mqa: Mem,    // [STEP][Q_LORA_RANK] f32: q_a's output, the input of q_b and the indexer's query
    mq: Mem,     // [STEP][MLA_HEADS·Q_HEAD] f32: the query, the attention's input
    mproj: Mem,  // host memory: kva [STEP][KV_LORA_RANK] | ix [STEP][IX_ROWS]
    msel: Mem,   // [STEP][DENSE_BELOW] u32, host-visible: the positions each token of the step attends
    mseln: Mem,  // [STEP] u32: how many each token attends
    miq: Mem,    // [STEP][INDEX_N_HEADS·INDEX_HEAD_DIM] f32: the indexer's query, the card's scorer's input
    mscore: Mem, // [STEP][pools] f32: the indexer's scores, the card's selection's input
    mheads: Mem, // [STEP][O_IN] f32: the heads' outputs, the output projection's input
    mqabs: Mem,  // [STEP][MLA_HEADS][KV_LORA_RANK] bf16: q_abs, the attend's A operand (`mla_qabs`)
    molat: Mem,  // [STEP][MLA_HEADS][KV_LORA_RANK] f32: the normalized mixes, W_uv's input
    mpart: Mem,  // [SPEC_TOKENS][AW_HEAD_TILES][AW_DECODE_CHUNKS][AW_PART] f32: a verify step's attend partials
    mselst: Mem, // [STEP][SELECT_STATE] u32: the selection's cross-workgroup state, left as a launch expects it
}
/// The draft layer's own pieces on the head's card (its MLA attention is `mlas[MTP_LAYER]`, its
/// routed experts the host's as layer `MTP_LAYER`): `eh_proj`, the norms, the shared expert, and
/// the `[2·H]` input staging.
struct MtpOn {
    card: usize,
    eh: Mem,        // [H][2·H] bf16
    post_ln: Mem,   // bf16
    head_norm: Vec<Mem>, // bf16, on every card: the shared head's rows are split across them (040)
    gu_w: Mem,
    gu_s: Mem,
    down_w: Mem,
    down_s: Mem,
    dx: Mem, // [STEP][2·H] f32: enorm(embed(next)) | hnorm(hidden), a step's positions
}
/// Host memory every card and the host see (`Kfd::host_shared`): what one produces and
/// another consumes within a token. A layer's buffers alternate by its parity, so a
/// producer a layer ahead never overwrites what a consumer has yet to read (the handoff
/// chain keeps them within one layer of each other).
struct Shared {
    yexp: [Mem; 2], // [STEP][H] f32: the host's routed experts
    xh: Mem,       // [STEP][H] f32: a MoE layer's FFN input for the host
    xout: Mem,     // [STEP·TOP_K][H] f32, the host's own pages mapped for the cards: the routed experts' outputs by (token, pick) — the cards' jobs and the host's experts (038)
    xact: Mem,     // the FFN input's quantized activations [STEP][H] i8 | [STEP][HG] f32 | [STEP][HG] i32, then the picks [STEP][TOP_K] u32 and weights f32: `ffn_act`'s, for the host (XA_*)
    scores_ffn: Mem, // [STEP][ROUTE_ROWS] f32: the FFN half's router over its input (this layer's rows, then the next layer's: the prefetch's predictor), for the host's picks
    words: Mem,    // the handoff words, a cache line each: `word()`
}
/// A handoff word's byte offset in `Shared::words`: the kind, and the card for a block output.
/// `Shared::xact`'s regions, in bytes.
const HG: usize = H / common::q4::GROUP;
const XA_Q: usize = 0;
const XA_S: usize = XA_Q + STEP * H;
const XA_G: usize = XA_S + STEP * HG * 4;
const XA_P: usize = XA_G + STEP * HG * 4;
const XA_W: usize = XA_P + STEP * TOP_K * 4;
const XA_END: usize = XA_W + STEP * TOP_K * 4;
const fn word(kind: usize, card: usize) -> usize {
    (kind * MAX_CARDS + card) * 64
}
const MAX_CARDS: usize = 4;
/// The attention block's output is in (a card's), the early router's logits are in (the FFN card's),
/// the FFN half's input is in (the FFN card's), the FFN rows are in (the FFN card's), the host's experts are in.
const W_Y: usize = 0;
const W_X: usize = 2;
const W_FFN: usize = 3;
const W_EXPERTS: usize = 4;
/// The FFN input's quantized activations, picks and weights are in (the FFN card's `ffn_act`, for the host).
const W_ACT: usize = 5;
const W_KINDS: usize = 6;
const MKVA: usize = 0;
const MIX: usize = MKVA + STEP * KV_LORA_RANK;
const MPROJ: usize = MIX + STEP * IX_ROWS;
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
    // a speculative verify step's states a token (this part's heads' entries), for `kda_rewind`
    ring: Mem,   // [SPEC_TOKENS][KDA_HEADS][KDA_DIM][KDA_DIM] f32
    ring_w: Mem, // [SPEC_TOKENS][CONV_TAPS-1][CONV_DIM] f32
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
    /// A MoE layer's post-norm weights (f32) and the router's selection bias (f32): `ffn_act`'s, the host's arithmetic.
    act: Option<(Mem, Mem)>,
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
    pools: Mem, // [max_ctx / INDEX_KPOOL][INDEX_HEAD_DIM] f32: the pool keys, the host's mirrored
    o: (Mem, Mem),
}

/// A card's receiving side of the card-to-card handoffs (queue 001): the peer's share of a KDA block's
/// output, the FFN card's rows and the MLA card's output projection, by layer parity, in this card's
/// VRAM (fine-grained, so its own polls and reads bypass its L2), mapped for the peers to write; and
/// its copy of the handoff words — the producers write their word here too, so a card waits on its own
/// memory and the host memory sees only what the host itself reads or writes.
struct Peer {
    ys: [Mem; 2], // [STEP][H] f32: the peer's part's share
    fy: [Mem; 2], // [STEP][H] f32
    my: [Mem; 2], // [STEP][H] f32
    words: Mem,   // `word()` offsets, a cache line each
}

pub struct Cards {
    cards: Vec<Card>,
    peers: Vec<Peer>,
    shared: Shared,
    layers: Vec<Option<Layer>>,
    ffns: Vec<Option<FfnOn>>,
    mlas: Vec<Option<MlaOn>>,
    /// The head on every card: the final norm (bf16) and this card's rows of lm_head — the vocabulary split
    /// across the cards (`head_rows`), each card's part over its own copy of the streams (040).
    head: Vec<(Mem, Mem)>,
    /// The last layer's card: the traces read the streams' means there.
    head_card: usize,
    mtp: Option<MtpOn>,
    resident: Resident,
    /// Built with the launches stamped (`track-launches`): the bring-up checks run too.
    instrument: bool,
    /// The handoff sequence: a value a half of a layer's words carry, and each layer's (attention, FFN) values this token.
    seq: u32,
    half_seq: Vec<[u32; 2]>,
    /// With the launches stamped: each card's first packet of the step.
    token_from: Vec<u64>,
    /// Pools the context can hold: the scores' row stride.
    npools: usize,
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
    pub fn new(elf: &[u8], w: &Weights, max_ctx: usize, instrument: bool, resident_bytes: usize) -> Self {
        let n = Kfd::cards().len();
        assert!(n > 0, "rockem.gpu: no gfx1100 card");
        let t0 = std::time::Instant::now();
        let cards: Vec<Card> = (0..n)
            .map(|i| {
                let k = Kfd::open(elf, instrument, i);
                let alloc = |floats: usize| k.vram_visible(floats * 4);
                Card {
                    rows_h: k.kernel("rows_h"),
                    rows_h_w: k.kernel("rows_h_w"),
                    rows_part_w: k.kernel("rows_part_w"),
                    mtp_eh_w: k.kernel("mtp_eh_w"),
                    ffn_gu_w: k.kernel("ffn_gu_w"),
                    ffn_down_shared_w: k.kernel("ffn_down_shared_w"),
                    ffn_down_dense_w: k.kernel("ffn_down_dense_w"),
                    mla_qb_w: k.kernel("mla_qb_w"),
                    mla_iq_w: k.kernel("mla_iq_w"),
                    mla_o_w: k.kernel("mla_o_w"),
                    rows_d2: k.kernel("rows_d2"),
                    rows_part: k.kernel("rows_part"),
                    step: k.kernel("kda_step"),
                    copy: k.kernel("copy_f32"),
                    xpt_gu: k.kernel("xpt_gu"),
                    xpt_down: k.kernel("xpt_down"),
                    ffn_act: k.kernel("ffn_act"),
                    xcount: {
                        let m = k.vram_visible(4096);
                        k.upload(&m, 0, &[0u8; 4096]);
                        m
                    },
                    xpt_gemm_gu: k.kernel("xpt_gemm_gu"),
                    xpt_quant: k.kernel("xpt_quant"),
                    xpt_gemm_down: k.kernel("xpt_gemm_down"),
                    xq: k.vram_visible(STEP * H),
                    xs: k.vram_visible(STEP * (H / common::q4::GROUP) * 4),
                    xg: k.vram_visible(STEP * (H / common::q4::GROUP) * 4),
                    xjobs: k.vram_visible((glm_5_3_flash::E * 4 + XG_MAX_JOBS * 3) * 4),
                    xgu: k.vram(XG_MAX_JOBS * 2 * D_FF * 4),
                    xhq: k.vram(XG_MAX_JOBS * D_FF),
                    xhs: k.vram(XG_MAX_JOBS * XQ_GROUPS * 4),
                    xhg: k.vram(XG_MAX_JOBS * XQ_GROUPS * 4),
                    ffn_gu: k.kernel("ffn_gu"),
                    ffn_down_shared: k.kernel("ffn_down_shared"),
                    ffn_down_dense: k.kernel("ffn_down_dense"),
                    mla_qb: k.kernel("mla_qb"),
                    mla_iq: k.kernel("mla_iq"),
                    mla_qabs: k.kernel("mla_qabs"),
                    mla_qabs_w: k.kernel("mla_qabs_w"),
                    mla_attend_w: k.kernel("mla_attend_w"),
                    mla_attend_merge: k.kernel("mla_attend_merge"),
                    mla_uv: k.kernel("mla_uv"),
                    mla_uv_w: k.kernel("mla_uv_w"),
                    ix_score_w: k.kernel("ix_score_w"),
                    ix_select: k.kernel("ix_select"),
                    add2: k.kernel("add2"),
                    mtp_eh: k.kernel("mtp_eh"),
                    mla_o: k.kernel("mla_o"),
                    hc_pre: k.kernel("hc_pre"),
                    hc_post: k.kernel("hc_post"),
                    hc_mean: k.kernel("hc_mean"),
                    streams: alloc(STEP * HC * H),
                    part: k.vram(STEP * PRE_STRIDE * 4),
                    count: {
                        let m = k.vram_visible(PRE_COUNTS * 4);
                        k.upload(&m, 0, &vec![0u8; PRE_COUNTS * 4]);
                        m
                    },
                    go: {
                        let m = k.vram_visible(64);
                        k.upload(&m, 0, &[0u8; 64]);
                        m
                    },
                    mix: k.vram(STEP * HC_MIX * 4),
                    mean: k.host(H * 4),
                    x: k.vram(STEP * H * 4),
                    yv: k.vram(g::PARTS * STEP * H * 4),
                    fyv: k.vram(STEP * H * 4),
                    myv: k.vram(STEP * H * 4),
                    smixs: (0..g::PARTS).map(|_| alloc(STEP * (SMALL_ROWS + 3 * g::PART_QKV))).collect(),
                    fgs: (0..g::PARTS).map(|_| alloc(STEP * 2 * g::PART_QKV)).collect(),
                    normeds: (0..g::PARTS).map(|_| alloc(STEP * g::PART_QKV)).collect(),
                    snap: k.host((KDA_WINDOW + KDA_MEMORY) * 4),
                    fx: k.vram(TOKENS * H * 4),
                    gu: alloc(STEP * DENSE_GU_ROWS),
                    lg: k.host(TOKENS * VOCAB_SIZE * 4),
                    mqa: alloc(STEP * Q_LORA_RANK),
                    mq: alloc(STEP * MLA_HEADS * Q_HEAD),
                    mproj: k.host(MPROJ * 4),
                    msel: alloc(STEP * DENSE_BELOW),
                    mseln: alloc(STEP),
                    miq: alloc(STEP * INDEX_N_HEADS * INDEX_HEAD_DIM),
                    mscore: alloc(STEP * max_ctx.div_ceil(INDEX_KPOOL)),
                    mheads: alloc(STEP * O_IN),
                    mqabs: k.vram(STEP * MLA_HEADS * KV_LORA_RANK * 2),
                    molat: k.vram(STEP * MLA_HEADS * KV_LORA_RANK * 4),
                    mpart: k.vram(SPEC_TOKENS * AW_HEAD_TILES * AW_DECODE_CHUNKS * AW_PART * 4),
                    mselst: {
                        let m = k.vram_visible(STEP * SELECT_STATE * 4);
                        let mut v = vec![0u32; STEP * SELECT_STATE];
                        for i in 0..STEP {
                            select_state_init(&mut v[i * SELECT_STATE..(i + 1) * SELECT_STATE]);
                        }
                        k.upload(&m, 0, raw(&v));
                        m
                    },
                    k,
                }
            })
            .collect();
        assert!(n <= MAX_CARDS, "{n} cards: the handoff words are laid out for {MAX_CARDS}");
        // hc_pre's workgroups wait for the last of them: the grid must fit the card's LDS at once (64 KB a CU)
        for c in &cards {
            let per_cu = (PRE_GRID as usize).div_ceil(c.k.cu);
            assert!(per_cu * PRE_LDS as usize <= 64 * 1024, "hc_pre: {PRE_GRID} workgroups at {PRE_LDS} B of LDS are {per_cu} a CU on {} CUs and cannot all be resident", c.k.cu);
        }
        Kfd::clocks_high();
        // the card-to-card handoffs: each card's receiving buffers and words in its own VRAM, mapped for the peers (001)
        assert!(cards.len() <= 2, "the handoff words reach one peer (`mine2`): {} cards", cards.len());
        let peers: Vec<Peer> = (0..cards.len())
            .map(|i| {
                let k = &cards[i].k;
                let others: Vec<&Kfd> = cards.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, c)| &*c.k).collect();
                let buf = || k.vram_shared(STEP * H * 4, &others);
                let words = k.vram_shared(word(W_KINDS, 0), &others);
                unsafe { std::ptr::write_bytes(words.ptr() as *mut u8, 0, words.size) };
                Peer { ys: [buf(), buf()], fy: [buf(), buf()], my: [buf(), buf()], words }
            })
            .collect();
        // the host's handoffs: host memory card 0 allocates and every other card maps too
        let shared = {
            let k0 = &cards[0].k;
            let others: Vec<&Kfd> = cards[1..].iter().map(|c| &*c.k).collect();
            let buf = || k0.host_shared(STEP * H * 4, &others);
            Shared {
                yexp: [buf(), buf()],
                xh: buf(),
                xout: k0.host_owned(STEP * TOP_K * H * 4, &others),
                xact: k0.host_shared(XA_END, &others),
                scores_ffn: k0.host_shared(STEP * ROUTE_ROWS * 4, &others),
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
                    ring: k.vram(SPEC_TOKENS * KDA_MEMORY * 4),
                    ring_w: k.vram(SPEC_TOKENS * KDA_WINDOW * 4),
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
            let act = match &w.layers[l].ffn {
                Ffn::Moe(m) => {
                    let ln32 = k.vram_visible(H * 4);
                    k.upload(&ln32, 0, raw(w.layers[l].post_ln));
                    let bias = k.vram_visible(glm_5_3_flash::E * 4);
                    k.upload(&bias, 0, raw(m.bias));
                    Some((ln32, bias))
                }
                Ffn::Dense(_) => None,
            };
            let (is_dense, (gu_w, gu_s), (down_w, down_s), router) = match &w.layers[l].ffn {
                Ffn::Dense(d) => (true, pair(d.gate, d.up), one(d.down), None),
                Ffn::Moe(m) => {
                    // this layer's router rows, then the next MoE layer's (zeros past the last): the FFN half's
                    // one router launch predicts the next layer's picks over this layer's FFN input
                    let r = k.vram_visible(ROUTE_ROWS * H * 2);
                    k.upload(&r, 0, raw(m.router));
                    match w.layers.get(l + 1).map(|lw| &lw.ffn) {
                        Some(Ffn::Moe(m2)) => k.upload(&r, m.router.len() * 2, raw(m2.router)),
                        _ => k.upload(&r, m.router.len() * 2, &vec![0u8; m.router.len() * 2]),
                    }
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
            ffns.push(Some(FfnOn { card, dense: is_dense, ln, hc, gu_w, gu_s, down_w, down_s, router, act }));
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
            let on = MlaOn { card, input_ln: up(raw(&bf16_of(w.layers[l].input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lat: k.vram_visible(max_ctx * KV_LORA_RANK * 2), pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 2), o: one(mw.o) };
            bytes += on.q_a.0.size + on.q_a.1.size + on.kv_a.0.size + on.kv_a.1.size + on.ix.size + on.q_b.0.size + on.q_b.1.size + on.ix_wq_b.size + on.kv_b.size + on.lat.size + on.o.0.size + on.o.1.size;
            mlas[l] = Some(on);
            mla_n += 1;
        }
        // the head with the last layer
        let head_card = card_of(NUM_HIDDEN_LAYERS - 1);
        let head: Vec<(Mem, Mem)> = (0..cards.len())
            .map(|c| {
                let k = &cards[c].k;
                let norm = k.vram_visible(H * 2);
                k.upload(&norm, 0, raw(&bf16_of(w.norm)));
                let (r0, rows) = head_rows(c, cards.len());
                let lm = k.vram_visible(rows * H * 2);
                k.upload(&lm, 0, raw(&w.lm_head[r0 * H..(r0 + rows) * H]));
                bytes += norm.size + lm.size;
                (norm, lm)
            })
            .collect();
        prow::row!("track-boot", "boot", { "what" => "cards", "s" => t0.elapsed().as_secs_f64(), "gb" => bytes as f64 / 1e9 });
        println!("gpu      : {} KDA layers, {mla_n} MLA layers (their latents to {max_ctx} positions), {dense} dense FFNs, {shared_n} shared experts, the hyper-connections, the streams and the head on {} card{} ({:.1} GB with their state) in {:.1}s", kda_layers.len(), n, if n == 1 { "" } else { "s" }, bytes as f64 / 1e9, t0.elapsed().as_secs_f32());
        // the draft layer: its MLA attention as layer MTP_LAYER on the head's card (the shared lm_head is
        // there), its own pieces beside it
        let mtp = w.mtp.as_ref().map(|mt| {
            let card = head_card;
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
            let mw = &mt.attn;
            let ix = {
                let m = k.vram_visible(IX_ROWS * H * 2);
                k.upload(&m, 0, raw(mw.ix_wk));
                k.upload(&m, mw.ix_wk.len() * 2, raw(mw.ix_gate));
                k.upload(&m, (mw.ix_wk.len() + mw.ix_gate.len()) * 2, raw(mw.ix_weights));
                m
            };
            mlas.push(Some(MlaOn { card, input_ln: up(raw(&bf16_of(mt.input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lat: k.vram_visible(max_ctx * KV_LORA_RANK * 2), pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 2), o: one(mw.o) }));
            let (gu_w, gu_s) = {
                let (g, u) = (mt.ffn.shared_gate, mt.ffn.shared_up);
                let wm = k.vram_visible(g.w.len() + u.w.len());
                let sm = k.vram_visible((g.s.len() + u.s.len()) * 4);
                // gate rows then up rows, their scales likewise, as the main layers' shared expert is laid out
                k.upload(&wm, 0, g.w);
                k.upload(&wm, g.w.len(), u.w);
                k.upload(&sm, 0, raw(g.s));
                k.upload(&sm, g.s.len() * 4, raw(u.s));
                (wm, sm)
            };
            let (down_w, down_s) = one(mt.ffn.shared_down);
            println!("gpu      : the draft layer on card {card}: its attention, shared expert and eh_proj ({:.2} GB)", (mt.eh_proj.len() * 2 + gu_w.size + down_w.size) as f64 / 1e9);
            MtpOn { card, eh: up(raw(mt.eh_proj)), post_ln: up(raw(&bf16_of(mt.post_ln))), head_norm: cards.iter().map(|cd| { let m = cd.k.vram_visible(H * 2); cd.k.upload(&m, 0, raw(&bf16_of(mt.head_norm))); m }).collect(), gu_w, gu_s, down_w, down_s, dx: k.vram_visible(STEP * 2 * H * 4) }
        });
        // the resident experts' slots last: what each card has free past the reserve, up to the budget
        let resident = {
            let moe = glm_5_3_flash::layout::MOE_LAYERS;
            let mut slots = Vec::with_capacity(n);
            let mut per_layer = Vec::with_capacity(n);
            let mut counts = Vec::with_capacity(n);
            for c in &cards {
                let bytes = resident_bytes.min(c.k.vram_free().saturating_sub(RESIDENT_RESERVE)) / SLOT_BYTES * SLOT_BYTES;
                let count = bytes / SLOT_BYTES;
                println!("gpu      : card {}: {:.1} GB allocated, {count} routed experts resident ({:.1} GB), {} a layer, filled on use and replaced least-picked past a margin of {RESIDENT_MARGIN} (counts halved every {RESIDENT_HALF_LIFE} positions)", c.k.index, c.k.allocated.get() as f64 / 1e9, bytes as f64 / 1e9, count / moe);
                slots.push(c.k.vram_visible(bytes.max(SLOT_BYTES)));
                per_layer.push(count / moe);
                counts.push(count);
            }
            Resident { table: std::sync::Arc::new((0..moe * glm_5_3_flash::E).map(|_| AtomicU32::new(NONE)).collect()), loader: None, inflight: Default::default(), allowed: Default::default(), issued: Default::default(), steps: 0, prefill: false, count: vec![0; moe * glm_5_3_flash::E], turn: 0, owner: counts.iter().map(|&k| vec![(NONE, NONE); k]).collect(), free: counts.iter().map(|&k| (0..k as u32).rev().collect()).collect(), held: (0..n).map(|_| vec![0; moe]).collect(), per_layer, slots, jobs: (0..n).map(|_| Vec::new()).collect(), uploads: 0 }
        };
        Cards { cards, peers, shared, layers, ffns, mlas, head, head_card, mtp, resident, instrument, seq: 0, half_seq: vec![[0; 2]; NUM_HIDDEN_LAYERS], token_from: vec![0; n], npools: max_ctx.div_ceil(INDEX_KPOOL) }
    }

    /// With the launches stamped (`track-launches`): the step's launches on every card as zones on
    /// lanes `card0`, `card1`, …, their completion stamps in UTC through the card's clock. A step of
    /// more launches than the ring holds keeps its last `RING_PACKETS`.
    fn timeline_report(&self) {
        if !self.cards[0].k.instrumented() {
            return;
        }
        for (i, c) in self.cards.iter().enumerate() {
            let lane = format!("card{}", c.k.index);
            for &(name, s, e) in &c.k.timeline_from(self.token_from[i]) {
                if e > s {
                    prow::zone_at!("track-launches", if name.is_empty() { "barrier" } else { name }, lane.as_str(), c.k.utc(s), c.k.utc(e));
                }
            }
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
    /// The hyper-connection's input half on card `c`, one launch: the coefficients over its streams,
    /// the mix, the collapse into the block input `x` — and into `x_host` for the routed experts when
    /// given (the host's word follows the router's logits, in the handoff after them).
    /// `router`: (the FFN half's norm, the router rows, the host's scores, the host's word) — the
    /// router over the collapse, run by the same launch, its logits and word to the host.
    fn hc_pre(&self, c: &Card, hc: &(Mem, Mem, Mem), x_host: Option<usize>, value: u32, router: Option<(usize, usize, usize, usize)>, n: usize) {
        let (ln, rw, sc, wd) = router.unwrap_or((0, 0, 0, 0));
        // the next layer's rows only for a step of a few tokens (the prefetch's predictor); a long step names most experts
        let rows = if router.is_none() { 0 } else if n <= SPEC_TOKENS { ROUTE_ROWS } else { glm_5_3_flash::E };
        let mut a = Args::new();
        a.ptr(c.streams.va).ptr(hc.0.va).ptr(c.part.va).ptr(hc.1.va).ptr(hc.2.va).ptr(c.mix.va).ptr(c.x.va).ptr(x_host.unwrap_or(0)).ptr(c.count.va).u32(value).ptr(ln).ptr(rw).ptr(sc).ptr(wd).u32(rows as u32).u32(n as u32);
        c.k.launch(&c.hc_pre, PRE_GRID, g::WG, PRE_LDS, &a);
    }
    /// The hyper-connection's output half on card `c`, the handoff first: this card's `copy`
    /// (src, dst, floats) out to shared memory and `value` into its word `mine`, then the queue
    /// holds for `w0` and `w1`, then the block output (`ya` + `yb`) into its streams. One launch.
    #[allow(clippy::too_many_arguments)]
    /// `copy`: (src in this card's VRAM, the peer's receiving buffer, floats) — a step of many tokens copies
    /// with the copy kernel's grid first (one workgroup moving 16 MB over the link would hold the launch),
    /// a few tokens' inside `hc_post`. `mine`: this card's word, written to its own copy and the peer's.
    /// `w0`, `w1`: words read from this card's own copy. `ya`, `yb`: the shares, in this card's VRAM.
    #[allow(clippy::too_many_arguments)]
    fn hc_post(&self, c: &Card, copy: Option<(usize, usize, usize)>, mine: Option<(usize, usize)>, w0: Option<(usize, usize)>, w1: Option<(usize, usize)>, value: u32, ya: usize, yb: Option<usize>, tokens: usize) {
        let me = c.k.index;
        let own = |w: Option<(usize, usize)>| w.map_or(0, |(k, d)| self.peers[me].words.at(word(k, d)));
        let peer = (0..self.cards.len()).find(|&j| j != me);
        let mine2 = match (mine, peer) {
            (Some((k, d)), Some(j)) => self.peers[j].words.at(word(k, d)),
            _ => 0,
        };
        let (mut src, mut dst2, mut n) = copy.unwrap_or((0, 0, 0));
        if copy.is_some() && tokens > SPEC_TOKENS {
            let mut a = Args::new();
            a.ptr(src).ptr(dst2).u32(n as u32);
            c.k.launch(&c.copy, n.div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &a);
            (src, dst2, n) = (0, 0, 0);
        }
        let mut a = Args::new();
        a.ptr(c.mix.va).ptr(ya).ptr(yb.unwrap_or(0)).ptr(c.streams.va).ptr(src).ptr(0usize).ptr(dst2).u32(n as u32).ptr(own(mine)).ptr(mine2).ptr(own(w0)).ptr(own(w1)).u32(value).ptr(c.go.va).u32(tokens as u32);
        c.k.launch(&c.hc_post, CHANNEL_GRID, g::WG, 0, &a);
    }
    /// The FFN router's place: on the FFN card of a MoE layer, `hc_pre` runs layer `l`'s FFN router (and the
    /// next layer's, the prefetch's predictor) over the half's collapsed input (normed with the FFN half's
    /// norm) and hands the logits to the host with the half's word.
    fn ffn_router(&self, l: usize, c: &Card) -> Option<(usize, usize, usize, usize)> {
        let f = self.ffns[l].as_ref().unwrap();
        f.router.as_ref().filter(|_| c.k.index == f.card).map(|r| (f.ln.va, r.va, self.shared.scores_ffn.va, self.shared.words.at(word(W_X, f.card))))
    }
    /// A KDA block's output into every card's streams: each card copies the shares it made out to
    /// the shared buffer and says so (its word), waits for the producers' words, and folds the sum in.
    /// `parts[c]`: the part slots card `c` holds, contiguous.
    fn attention_out(&self, value: u32, parity: usize, parts: &[Vec<usize>], n: usize) {
        let producers: Vec<usize> = (0..self.cards.len()).filter(|&c| !parts[c].is_empty()).collect();
        for c in &self.cards {
            let me = c.k.index;
            let mine = &parts[me];
            let peer = (0..self.cards.len()).find(|&j| j != me);
            // this card's share stays where `rows_part` wrote it (`yv`, its parts' slots); a peer's share arrives in
            // this card's receiving buffer. With a peer each card holds one part; alone it holds both in `yv`.
            let (ya, yb) = match (mine.first(), peer) {
                (Some(&p0), Some(_)) => {
                    assert!(mine.len() == 1, "one part a card with a peer");
                    (c.yv.at(p0 * STEP * H * 4), Some(self.peers[me].ys[parity].va))
                }
                (Some(&p0), None) => (c.yv.at(p0 * STEP * H * 4), (mine.len() > 1).then(|| c.yv.at((p0 + 1) * STEP * H * 4))),
                (None, _) => (self.peers[me].ys[parity].va, None),
            };
            let copy = match (mine.first(), peer) {
                (Some(&p0), Some(j)) => Some((c.yv.at(p0 * STEP * H * 4), self.peers[j].ys[parity].va, n * H)),
                _ => None,
            };
            self.hc_post(c, copy, (!mine.is_empty()).then_some((W_Y, me)), Some((W_Y, producers[0])), producers.get(1).map(|&p| (W_Y, p)), value, ya, yb, n);
        }
    }

    /// FP8 rows on card `c` over the step's `n` tokens: `out[i][row] = W[row] · x[i]` (normed with `ln`
    /// when given) for `m` rows of width `k_width`, scales `s`; `x` is `[n][k_width]`, `out` `[n][m]`.
    /// A launch takes `TOKENS` tokens at most (the kernels' slab), so a step is passes of that many.
    #[allow(clippy::too_many_arguments)]
    fn rows8(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows8_at(c, kernel, x, k_width, ln, w, s, out, m, k_width, n, r, true)
    }
    #[allow(clippy::too_many_arguments)]
    fn rows8_free(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows8_at(c, kernel, x, k_width, ln, w, s, out, m, k_width, n, r, false)
    }
    /// `rows8` with the input's token stride `xs` given: the down kernels read token `i`'s gate|up row
    /// at `x[i · 2 · k_width]` (`swiglu_at`), so a pass past the first starts at the stride's multiple,
    /// not the width's (the second pass of a step's dense and shared-expert down read from the
    /// middle of token 16's row until 2026-10-16; a step of 32 or fewer tokens is one pass and never saw it).
    #[allow(clippy::too_many_arguments)]
    fn rows8_at(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, xs: usize, ln: Option<usize>, w: usize, s: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize, barrier: bool) {
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            let mut a = Args::new();
            a.ptr(x + p0 * xs * 4);
            if let Some(ln) = ln {
                a.ptr(ln);
            }
            a.ptr(w).ptr(s).ptr(out + p0 * m * 4).u32(m as u32).u32(np as u32);
            let lds = g::fp8_rows_lds(k_width, np) as u32;
            if barrier || p0 > 0 {
                c.k.launch(kernel, g::rows_grid(m, np, r), g::WG, lds, &a);
            } else {
                c.k.launch_free(kernel, g::rows_grid(m, np, r), g::WG, lds, &a);
            }
            p0 += np;
        }
    }

    /// An MLA layer's projections over the step's block input in `c.x`: q_a into VRAM for the next two,
    /// kv_a and the indexer's three out to the host, the query and the indexer's query; then a fence.
    fn mla_rows(&self, m: &MlaOn, n: usize) {
        let c = &self.cards[m.card];
        let ln = Some(m.input_ln.va);
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.ffn_gu_w, c.x.va, ln, m.q_a.0.va, Some(m.q_a.1.va), c.mqa.va, Q_LORA_RANK, n, false);
            self.rows_w(c, &c.ffn_gu_w, c.x.va, ln, m.kv_a.0.va, Some(m.kv_a.1.va), c.mproj.at(MKVA * 4), KV_LORA_RANK, n, true);
        } else {
            self.rows8(c, &c.ffn_gu, c.x.va, ln, m.q_a.0.va, m.q_a.1.va, c.mqa.va, Q_LORA_RANK, H, n, ROWS_GU);
            self.rows8_free(c, &c.ffn_gu, c.x.va, ln, m.kv_a.0.va, m.kv_a.1.va, c.mproj.at(MKVA * 4), KV_LORA_RANK, H, n, ROWS_GU);
        }
        if n > SPEC_TOKENS {
            let mut a = Args::new();
            a.ptr(c.x.va).ptr(m.input_ln.va).ptr(m.ix.va).ptr(c.mproj.at(MIX * 4)).u32(IX_ROWS as u32).u32(n as u32);
            c.k.launch_free(&c.rows_h_w, g::wmma_grid(IX_ROWS, n), g::WG, g::WMMA_LDS, &a);
        } else {
            self.rows_free(c, &c.rows_h, c.x.va, ln, m.ix.va, c.mproj.at(MIX * 4), IX_ROWS, H, n, ROWS);
        }
        // over q_a's output normed with q_a's norm: the query and, alongside it, the indexer's query (VRAM: the card scores the pools)
        let qln = Some(m.qa_ln.va);
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_qb_w, c.mqa.va, qln, m.q_b.0.va, Some(m.q_b.1.va), c.mq.va, MLA_HEADS * Q_HEAD, n, false);
            self.rows_w(c, &c.mla_iq_w, c.mqa.va, qln, m.ix_wq_b.va, None, c.miq.va, INDEX_N_HEADS * INDEX_HEAD_DIM, n, true);
        } else {
            self.rows8(c, &c.mla_qb, c.mqa.va, qln, m.q_b.0.va, m.q_b.1.va, c.mq.va, MLA_HEADS * Q_HEAD, Q_LORA_RANK, n, ROWS_QB);
            self.rows_free(c, &c.mla_iq, c.mqa.va, qln, m.ix_wq_b.va, c.miq.va, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK, n, ROWS);
        }
        c.k.fence();
    }
    /// `x[..n·H] += a + b` on card `c` (`b` 0 for none): the draft layer's residual.
    fn add2(&self, c: &Card, x: usize, a: usize, b: usize, n: usize) {
        let mut args = Args::new();
        args.ptr(x).ptr(a).ptr(b).u32((n * H) as u32);
        c.k.launch(&c.add2, (n * H / 256) as u32, g::WG, 0, &args);
    }
    /// `n·H` floats from VRAM to host memory on card `c` (a copy launch).
    fn copy_out(&self, c: &Card, src: usize, dst: usize, n: usize) {
        let mut args = Args::new();
        args.ptr(src).ptr(dst).u32((n * H) as u32);
        c.k.launch(&c.copy, (n * H).div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &args);
    }
    /// bf16 rows on card `c` over the step's `n` tokens (as `rows8`). Addresses are device VAs.
    #[allow(clippy::too_many_arguments)]
    fn rows(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows_at(c, kernel, x, ln, w, out, m, k_width, n, r, true)
    }
    /// A matrix-path rows launch over a step of many tokens (`rows_wmma`): the kernel's args in the scalar
    /// twin's order (x, [ln], w, [s], out, m, n); the grid covers the rows and the tokens at once.
    #[allow(clippy::too_many_arguments)]
    fn rows_w(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, s: Option<usize>, out: usize, m: usize, n: usize, free: bool) {
        let mut a = Args::new();
        a.ptr(x);
        if let Some(ln) = ln {
            a.ptr(ln);
        }
        a.ptr(w);
        if let Some(s) = s {
            a.ptr(s);
        }
        a.ptr(out).u32(m as u32).u32(n as u32);
        if free {
            c.k.launch_free(kernel, g::wmma_grid(m, n), g::WG, g::WMMA_LDS, &a);
        } else {
            c.k.launch(kernel, g::wmma_grid(m, n), g::WG, g::WMMA_LDS, &a);
        }
    }
    /// `rows` launched free: as soon as the launch before it has launched, so the two run together —
    /// for a launch whose inputs were complete when the one before it launched.
    #[allow(clippy::too_many_arguments)]
    fn rows_free(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows_at(c, kernel, x, ln, w, out, m, k_width, n, r, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn rows_at(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize, barrier: bool) {
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            let mut a = Args::new();
            a.ptr(x + p0 * k_width * 4);
            if let Some(ln) = ln {
                a.ptr(ln);
            }
            a.ptr(w).ptr(out + p0 * m * 4).u32(m as u32).u32(np as u32);
            let lds = if ln.is_some() { g::normed_rows_lds(k_width, np) } else { g::rows_lds(k_width, np) } as u32;
            if barrier || p0 > 0 {
                c.k.launch(kernel, g::rows_grid(m, np, r), g::WG, lds, &a);
            } else {
                c.k.launch_free(kernel, g::rows_grid(m, np, r), g::WG, lds, &a);
            }
            p0 += np;
        }
    }
}

/// Card `c`'s rows of lm_head among `cards`: the vocabulary in parts of whole WMMA row groups (which the
/// scalar rows kernel's groups divide), the last card taking the remainder. The head and the draft's head
/// stream 1.27 GB a step each on one card otherwise (040).
fn head_rows(c: usize, cards: usize) -> (usize, usize) {
    let part = VOCAB_SIZE.div_ceil(cards).next_multiple_of(g::WMMA_ROWS);
    let r0 = (c * part).min(VOCAB_SIZE);
    (r0, (VOCAB_SIZE - r0).min(part))
}

impl Cards {
    /// A step begins at its first MoE layer: the layer turn advances, the counts age at the half-life, and
    /// the loader's budget is set — one more load for a decode step, no bound for a prefill step (063).
    fn begin_step(&mut self, prefill: bool) {
        use std::sync::atomic::Ordering;
        let r = &mut self.resident;
        r.steps += 1;
        r.prefill = prefill;
        r.turn = (r.steps % r.held[0].len() as u64) as usize;
        if r.steps % RESIDENT_HALF_LIFE == 0 {
            for c in r.count.iter_mut() {
                *c >>= 1;
            }
        }
        let issued = r.issued.load(Ordering::Acquire);
        r.allowed.store(if prefill { u64::MAX } else { issued + 1 }, Ordering::Release);
    }
    /// Expert `e` of MoE layer `ml` claims a slot and its load is queued: a free slot on the card holding fewer
    /// of the layer, else — with `replace` — the layer's least-picked resident not in use when `e` leads it
    /// by the margin. `Some(true)` when a resident was replaced, `Some(false)` for a free slot, `None` for no slot.
    fn claim(&mut self, ml: usize, e: usize, in_use: &[bool], replace: bool) -> Option<bool> {
        use std::sync::atomic::Ordering;
        let e_n = glm_5_3_flash::E;
        let r = &mut self.resident;
        let mut pick_card = None;
        for c in (0..self.cards.len()).rev() {
            if (r.held[c][ml] as usize) < r.per_layer[c] && !r.free[c].is_empty() && pick_card.is_none_or(|p: usize| r.held[c][ml] < r.held[p][ml]) {
                pick_card = Some(c);
            }
        }
        let (c, sl, by_replacement) = match pick_card {
            Some(c) => (c, r.free[c].pop().unwrap(), false),
            None if replace => {
                let mut victim: Option<(usize, u32, u32)> = None; // (card, slot, count)
                for c in 0..self.cards.len() {
                    for (sl, &(ol, oe)) in r.owner[c].iter().enumerate() {
                        if ol as usize != ml || in_use[oe as usize] || r.table[ml * e_n + oe as usize].load(Ordering::Relaxed) == PENDING {
                            continue;
                        }
                        let cnt = r.count[ml * e_n + oe as usize];
                        if victim.is_none_or(|v| cnt < v.2) {
                            victim = Some((c, sl as u32, cnt));
                        }
                    }
                }
                match victim {
                    Some((c, sl, cnt)) if r.count[ml * e_n + e] >= cnt + RESIDENT_MARGIN => {
                        let (_, oe) = r.owner[c][sl as usize];
                        r.table[ml * e_n + oe as usize].store(NONE, Ordering::Release);
                        r.held[c][ml] -= 1;
                        (c, sl, true)
                    }
                    _ => return None,
                }
            }
            None => return None,
        };
        r.owner[c][sl as usize] = (ml as u32, e as u32);
        r.held[c][ml] += 1;
        r.uploads += 1;
        r.table[ml * e_n + e].store(PENDING, Ordering::Release);
        r.inflight.fetch_add(1, Ordering::AcqRel);
        r.loader.as_ref().expect("the residency's loader is set before the first step").send((c, sl, ml, e)).unwrap();
        Some(by_replacement)
    }
    /// A prefill step's claims for MoE layer `ml`: the layer's most-picked experts not resident claim slots —
    /// free ones first, then up to `PREFILL_REPLACES` replacements — loaded while the step runs, so decode
    /// begins with the residency the prompt's routing chose (063). `in_use`: the experts with a job this step.
    fn claim_prefill(&mut self, ml: usize, in_use: &[bool]) {
        use std::sync::atomic::Ordering;
        let e_n = glm_5_3_flash::E;
        let r = &self.resident;
        let mut order: Vec<usize> = (0..e_n).filter(|&e| r.count[ml * e_n + e] > 0 && r.table[ml * e_n + e].load(Ordering::Relaxed) == NONE).collect();
        order.sort_by_key(|&e| std::cmp::Reverse(r.count[ml * e_n + e]));
        let mut replaced = 0;
        for &e in order.iter().take(PREFILL_CLAIMS) {
            match self.claim(ml, e, in_use, replaced < PREFILL_REPLACES) {
                Some(true) => replaced += 1,
                Some(false) => {}
                None => break, // the rest are picked less: no slot for them either
            }
        }
    }

}

impl Placed for Cards {
    fn begin(&mut self, rows: &[f32], n: usize) {
        // every card: the step's handoffs restart, each token's embedding into each of its streams
        let mut streams = Vec::with_capacity(n * HC * H);
        for i in 0..n {
            for _ in 0..HC {
                streams.extend_from_slice(&rows[i * H..(i + 1) * H]);
            }
        }
        for (i, c) in self.cards.iter().enumerate() {
            c.k.upload(&c.streams, 0, raw(&streams));
            self.token_from[i] = c.k.posted();
        }
    }
    fn attention(&mut self, l: usize, n: usize) {
        let value = self.next_seq();
        // a step of a verify's size keeps a state a token (`kda_rewind`); one token and a prompt's step do not
        let ring = (2..=SPEC_TOKENS).contains(&n);
        self.half_seq[l][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], None, value, None, n);
        }
        let lay = self.layers[l].as_ref().unwrap();
        let parity = l % 2;
        const M_SMALL: usize = SMALL_ROWS + 3 * g::PART_QKV;
        for (i, p) in lay.parts.iter().enumerate() {
            let c = &self.cards[p.card];
            let (smix, fg, normed) = (&c.smixs[i], &c.fgs[i], &c.normeds[i]);
            // the gates' first halves and beta, then q | k | v: one launch over the normed block input, [n][M_SMALL];
            // a step of many tokens takes the matrix path (bf16 activations, numerically its own)
            if n > SPEC_TOKENS {
                let mut a = Args::new();
                a.ptr(c.x.va).ptr(p.input_ln.va).ptr(p.sq.va).ptr(smix.va).u32(M_SMALL as u32).u32(n as u32);
                c.k.launch(&c.rows_h_w, g::wmma_grid(M_SMALL, n), g::WG, g::WMMA_LDS, &a);
            } else {
                self.rows(c, &c.rows_h, c.x.va, Some(p.input_ln.va), p.sq.va, smix.va, M_SMALL, H, n, ROWS);
            }
            // the gates' second halves as one launch: f_b over f_a's output, g_b over g_a's — token i's input at smix[i · M_SMALL]
            let mut p0 = 0;
            while p0 < n {
                let np = (n - p0).min(TOKENS);
                let mut a = Args::new();
                a.ptr(smix.at(p0 * M_SMALL * 4)).ptr(p.fg.va).ptr(fg.at(p0 * 2 * g::PART_QKV * 4)).u32(2 * g::PART_QKV as u32).u32(np as u32).u32(g::PART_QKV as u32).u32(M_SMALL as u32);
                c.k.launch(&c.rows_d2, g::rows_grid(2 * g::PART_QKV, np, ROWS), g::WG, g::rows_lds(KDA_DIM, np) as u32, &a);
                p0 += np;
            }
            // the delta rule over the step's tokens in order, the heads' memory in registers across them
            let mixed = smix.at(SMALL_ROWS * 4);
            let mut a = Args::new();
            a.ptr(mixed).ptr(p.taps.va).ptr(p.window.va).ptr(fg.va).ptr(p.dt_bias.va).ptr(p.a_log.va).ptr(smix.at(2 * KDA_DIM * 4)).ptr(fg.at(g::PART_QKV * 4)).ptr(p.mem.va).ptr(p.o_norm.va).ptr(normed.va).u32(p.head0 as u32).u32(n as u32).u32(M_SMALL as u32).u32(2 * g::PART_QKV as u32).u32(M_SMALL as u32).ptr(if ring { p.ring.va } else { 0 }).ptr(if ring { p.ring_w.va } else { 0 });
            c.k.launch(&c.step, g::STEP_GRID, g::WG, g::STEP_LDS as u32, &a);
            if n > SPEC_TOKENS {
                let mut a = Args::new();
                a.ptr(normed.va).ptr(p.o.va).ptr(c.yv.at(i * STEP * H * 4)).u32(H as u32).u32(n as u32);
                c.k.launch(&c.rows_part_w, g::wmma_grid(H, n), g::WG, g::WMMA_LDS, &a);
            } else {
                self.rows(c, &c.rows_part, normed.va, None, p.o.va, c.yv.at(i * STEP * H * 4), H, g::PART_QKV, n, ROWS);
            }
        }
        // each card's shares out and its word; every card folds the sum into its streams
        let mut parts: Vec<Vec<usize>> = vec![Vec::new(); self.cards.len()];
        for (i, p) in lay.parts.iter().enumerate() {
            parts[p.card].push(i);
        }
        self.attention_out(value, parity, &parts, n);
    }
    fn mla_post(&mut self, l: usize, n: usize) {
        let value = self.next_seq();
        self.half_seq[l][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], None, value, None, n);
        }
        let m = self.mlas[l].as_ref().unwrap();
        self.mla_rows(m, n);
        // the other cards fold the output in once it is there (posted now: they wait for the word)
        for (i, o) in self.cards.iter().enumerate() {
            if i != m.card {
                self.hc_post(o, None, None, Some((W_Y, m.card)), None, value, self.peers[i].my[l % 2].va, None, n);
            }
        }
    }
    fn hidden_all(&mut self, n: usize, out: &mut [f32]) {
        // the streams' means a pass of TOKENS at a time, through the head card's logits buffer (host memory)
        let c = &self.cards[self.head_card];
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            let mut a = Args::new();
            a.ptr(c.streams.at(p0 * HC * H * 4)).ptr(c.fx.va).u32(np as u32);
            c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
            let mut b = Args::new();
            b.ptr(c.fx.va).ptr(c.lg.va).u32((np * H) as u32);
            c.k.launch(&c.copy, (np * H).div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &b);
            c.k.sync();
            read(&c.lg, 0, &mut out[p0 * H..(p0 + np) * H]);
            p0 += np;
        }
    }
    fn checks(&self) -> bool {
        self.instrument
    }
    fn resident_stats(&self) -> (usize, usize, usize) {
        let r = &self.resident;
        let total: usize = r.owner.iter().map(|o| o.len()).sum();
        let free: usize = r.free.iter().map(|f| f.len()).sum();
        (total - free, total, r.uploads)
    }
    fn resident(&self, l: usize, e: usize) -> bool {
        let at = self.resident.table[(l - glm_5_3_flash::FIRST_K_DENSE_REPLACE) * glm_5_3_flash::E + e].load(std::sync::atomic::Ordering::Relaxed);
        at != NONE && at != PENDING
    }
    fn xpt_loader(&mut self, fetch: Box<dyn Fn(usize, usize, &mut [u8]) + Send + 'static>) {
        // the loader: gathers an expert from the stream and streams it into its slot (write-combining
        // stores over the BAR), then publishes the slot in the table; the main thread keeps the bookkeeping
        let (tx, rx) = std::sync::mpsc::channel::<(usize, u32, usize, usize)>();
        let table = self.resident.table.clone();
        let inflight = self.resident.inflight.clone();
        let allowed = self.resident.allowed.clone();
        let issued = self.resident.issued.clone();
        let bases: Vec<usize> = self.resident.slots.iter().map(|m| m.ptr() as usize).collect();
        std::thread::Builder::new().name("xpt-loader".into()).spawn(move || {
            extern "C" {
                fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
            }
            let mut mask = [0u64; 16];
            mask[LOADER_CORE / 64] |= 1 << (LOADER_CORE % 64);
            unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
            let mut scratch = vec![0u8; SLOT_BYTES];
            for (card, slot, ml, e) in rx {
                // paced by the main thread: one load a decode step, free inside a prefill step
                while issued.load(std::sync::atomic::Ordering::Acquire) >= allowed.load(std::sync::atomic::Ordering::Acquire) {
                    std::thread::park_timeout(std::time::Duration::from_micros(200));
                }
                fetch(ml + glm_5_3_flash::FIRST_K_DENSE_REPLACE, e, &mut scratch);
                let dst = unsafe { std::slice::from_raw_parts_mut((bases[card] + slot as usize * SLOT_BYTES) as *mut u8, SLOT_BYTES) };
                common::copy::stream(dst, &scratch);
                table[ml * glm_5_3_flash::E + e].store((card as u32) << 24 | slot, std::sync::atomic::Ordering::Release);
                issued.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                inflight.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }
        }).unwrap();
        self.resident.loader = Some(tx);
    }
    fn xpt_dispatch(&mut self, l: usize, n: usize, picks: &mut [[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32]) -> usize {
        use std::sync::atomic::Ordering;
        prow::zone!("track-zones", "dispatch");
        let ml = l - glm_5_3_flash::FIRST_K_DENSE_REPLACE;
        let e_n = glm_5_3_flash::E;
        let prefill = n > SPEC_TOKENS;
        if ml == 0 {
            self.begin_step(prefill);
        }
        let r = &mut self.resident;
        for j in r.jobs.iter_mut() {
            j.clear();
        }
        // the resident picks become jobs, up to a card's XG_MAX_JOBS (the rest stay on the host); the experts
        // with a job this step are never a victim
        let mut in_use = [false; glm_5_3_flash::E];
        let mut misses: Vec<usize> = Vec::new();
        for i in 0..n {
            for k in 0..TOP_K {
                let e = picks[i][k];
                if e >= e_n {
                    continue; // a slot no expert runs in
                }
                r.count[ml * e_n + e] += 1;
                let at = r.table[ml * e_n + e].load(Ordering::Acquire);
                if at != NONE && at != PENDING {
                    let card = (at >> 24) as usize;
                    if r.jobs[card].len() < XG_MAX_JOBS {
                        r.jobs[card].push((at & 0xff_ffff, i as u32, k as u8));
                        picks[i][k] = e_n;
                        in_use[e] = true;
                    }
                } else if at == NONE && !prefill && !misses.contains(&e) {
                    misses.push(e);
                }
            }
        }
        if !prefill {
            // the misses claim slots for the steps to come (one load a step, the layer turn first; one replacement a layer)
            let mut replaced = false;
            for &e in &misses {
                let r = &self.resident;
                if ml < r.turn || r.inflight.load(Ordering::Acquire) >= LOADS_IN_FLIGHT {
                    break;
                }
                if self.claim(ml, e, &in_use, !replaced) == Some(true) {
                    replaced = true;
                }
            }
        }
        // the launches: a step of a few tokens runs a token a job (`xpt_gu`, `xpt_down` over `[jobs][3]`: slot,
        // token, output row); a prefill step's jobs grouped by slot — a group table (slot, first job, jobs, 0)
        // and the jobs (token, output row) — through gate/up over every group, the hidden states quantized,
        // down into the shared output rows. The same split as the rows kernels', at SPEC_TOKENS.
        let r = &mut self.resident;
        let mut total = 0;
        for c in 0..self.cards.len() {
            let jobs = &mut r.jobs[c];
            if jobs.is_empty() {
                continue;
            }
            total += jobs.len();
            let card = &self.cards[c];
            // the FFN card quantized the step's activations into its own buffers (`ffn_act`); the other card takes
            // the host's copy of them
            if c != self.ffns[l].as_ref().unwrap().card {
                card.k.upload(&card.xq, 0, raw(xq));
                card.k.upload(&card.xs, 0, raw(xs));
                card.k.upload(&card.xg, 0, raw(xg));
            }
            if !prefill {
                let list: Vec<u32> = jobs.iter().flat_map(|&(sl, tok, k)| [sl, tok, tok * TOP_K as u32 + k as u32]).collect();
                card.k.upload(&card.xjobs, 0, raw(&list));
                let mut a = Args::new();
                a.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(card.xq.va).ptr(card.xs.va).ptr(card.xg.va).ptr(card.xgu.va);
                card.k.launch(&card.xpt_gu, (jobs.len() * XPT_UP_WGS) as u32, g::WG, XPT_LDS as u32, &a);
                let mut b = Args::new();
                b.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(card.xgu.va).ptr(self.shared.xout.va);
                card.k.launch(&card.xpt_down, (jobs.len() * XPT_DOWN_WGS) as u32, g::WG, XPT_LDS as u32, &b);
                card.k.fence();
                continue;
            }
            jobs.sort_unstable_by_key(|&(slot, tok, k)| (slot, tok, k));
            let mut table: Vec<u32> = Vec::with_capacity(e_n * 4 + jobs.len() * 2);
            let mut groups = 0u32;
            let mut j0 = 0;
            while j0 < jobs.len() {
                let slot = jobs[j0].0;
                let mut j1 = j0;
                while j1 < jobs.len() && jobs[j1].0 == slot {
                    j1 += 1;
                }
                table.extend_from_slice(&[slot, j0 as u32, (j1 - j0) as u32, 0]);
                groups += 1;
                j0 = j1;
            }
            let jobs_at = e_n * 4;
            table.resize(jobs_at, 0);
            for &(_, tok, k) in jobs.iter() {
                table.extend_from_slice(&[tok, tok * TOP_K as u32 + k as u32]);
            }
            card.k.upload(&card.xjobs, 0, raw(&table));
            let jobs_va = card.xjobs.at(jobs_at * 4);
            let mut a = Args::new();
            a.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(jobs_va).ptr(card.xq.va).ptr(card.xs.va).ptr(card.xg.va).ptr(card.xgu.va);
            card.k.launch(&card.xpt_gemm_gu, groups * XG_GU_WGS as u32, g::WG, 0, &a);
            let mut q = Args::new();
            q.ptr(card.xgu.va).ptr(card.xhq.va).ptr(card.xhs.va).ptr(card.xhg.va);
            card.k.launch(&card.xpt_quant, (jobs.len() * XQ_WGS) as u32, g::WG, 0, &q);
            let mut b = Args::new();
            b.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(jobs_va).ptr(card.xhq.va).ptr(card.xhs.va).ptr(card.xhg.va).ptr(self.shared.xout.va);
            card.k.launch(&card.xpt_gemm_down, groups * XG_DOWN_WGS as u32, g::WG, 0, &b);
            card.k.fence();
        }
        total
    }
    fn xpt_join(&mut self, l: usize, _out: &mut [f32]) {
        // the jobs' rows are in the shared output already; the wait is all
        for c in 0..self.cards.len() {
            if !self.resident.jobs[c].is_empty() {
                self.cards[c].k.wait();
            }
        }
        // a prefill step's claims, now that no job of the layer reads a slot (063)
        if self.resident.prefill {
            let none = [false; glm_5_3_flash::E];
            self.claim_prefill(l - glm_5_3_flash::FIRST_K_DENSE_REPLACE, &none);
        }
    }
    fn expert_out(&self) -> *mut f32 {
        self.shared.xout.ptr() as *mut f32
    }
    fn mtp_begin(&mut self, cat: &[f32], n: usize) {
        let t = self.mtp.as_ref().expect("no draft layer on the cards");
        let c = &self.cards[t.card];
        // eh_proj over the staged inputs into the block input slots, then the attention's projections
        c.k.upload(&t.dx, 0, raw(&cat[..n * 2 * H]));
        if n > SPEC_TOKENS {
            let mut a = Args::new();
            a.ptr(t.dx.va).ptr(t.eh.va).ptr(c.x.va).u32(H as u32).u32(n as u32);
            c.k.launch(&c.mtp_eh_w, g::wmma_grid(H, n), g::WG, g::WMMA_LDS, &a);
        } else {
            self.rows(c, &c.mtp_eh, t.dx.va, None, t.eh.va, c.x.va, H, 2 * H, n, ROWS);
        }
        let m = self.mlas[glm_5_3_flash::layout::MTP_LAYER].as_ref().unwrap();
        self.mla_rows(m, n);
    }
    fn mtp_attn_out(&mut self, hidden: &mut [f32], n: usize) {
        let t = self.mtp.as_ref().unwrap();
        let c = &self.cards[t.card];
        let m = self.mlas[glm_5_3_flash::layout::MTP_LAYER].as_ref().unwrap();
        // the output projection, the residual, the block's states out to the host for its routing
        self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.myv.va, H, O_IN, n, ROWS);
        self.add2(c, c.x.va, c.myv.va, 0, n);
        self.copy_out(c, c.x.va, self.shared.xh.va, n);
        c.k.sync();
        read(&self.shared.xh, 0, &mut hidden[..n * H]);
    }
    fn mtp_ffn(&mut self, routed: &[f32], hidden: &mut [f32], n: usize) {
        let t = self.mtp.as_ref().unwrap();
        let c = &self.cards[t.card];
        // the shared expert over the post-normed states, the host's routed experts, the residual
        let yexp = &self.shared.yexp[0];
        unsafe { std::ptr::copy_nonoverlapping(routed.as_ptr(), yexp.ptr() as *mut f32, n * H) };
        self.rows8(c, &c.ffn_gu, c.x.va, Some(t.post_ln.va), t.gu_w.va, t.gu_s.va, c.gu.va, SHARED_GU_ROWS, H, n, ROWS_GU);
        self.rows8_at(c, &c.ffn_down_shared, c.gu.va, 2 * N_SHARED_EXPERTS * D_FF, None, t.down_w.va, t.down_s.va, c.fyv.va, H, N_SHARED_EXPERTS * D_FF, n, ROWS, true);
        self.add2(c, c.x.va, c.fyv.va, yexp.va, n);
        self.copy_out(c, c.x.va, self.shared.xh.va, n);
        c.k.sync();
        read(&self.shared.xh, 0, &mut hidden[..n * H]);
    }
    fn mtp_head(&mut self, i: usize, logits: &mut [f32]) {
        let t = self.mtp.as_ref().unwrap();
        // every card's rows of the shared head over the draft's hidden state: in the draft card's `x`, and
        // for the other cards in host memory since `mtp_ffn` (the same bytes, so the parts are one head)
        for (ci, c) in self.cards.iter().enumerate() {
            let (_, lm) = &self.head[ci];
            let (_, rows) = head_rows(ci, self.cards.len());
            let x = if ci == t.card {
                c.x.at(i * H * 4)
            } else {
                // into this card's head input first: the rows kernel's workgroups each read the whole
                // activation, which over the BAR cost the draft 1.8 ms (2026-10-19 00:15)
                self.copy_out(c, self.shared.xh.at(i * H * 4), c.fx.va, 1);
                c.fx.va
            };
            self.rows(c, &c.rows_h, x, Some(t.head_norm[ci].va), lm.va, c.lg.va, rows, H, 1, ROWS);
        }
        for (ci, c) in self.cards.iter().enumerate() {
            let (r0, rows) = head_rows(ci, self.cards.len());
            c.k.sync();
            read(&c.lg, 0, &mut logits[r0..r0 + rows]);
        }
    }
    fn mla_wait(&mut self, l: usize, i: usize, kva: &mut [f32], ik: &mut [f32], igate: &mut [f32]) {
        let c = &self.cards[self.mlas[l].as_ref().unwrap().card];
        if i == 0 {
            c.k.wait();
        }
        read(&c.mproj, (MKVA + i * KV_LORA_RANK) * 4, &mut kva[..KV_LORA_RANK]);
        let ix = MIX + i * IX_ROWS;
        read(&c.mproj, ix * 4, &mut ik[..INDEX_HEAD_DIM]);
        read(&c.mproj, (ix + INDEX_HEAD_DIM) * 4, &mut igate[..INDEX_HEAD_DIM]);
    }
    fn ix_pools(&mut self, l: usize, pools: &[u16], from: usize, to: usize) {
        let m = self.mlas[l].as_ref().unwrap();
        if to > from {
            self.cards[m.card].k.upload(&m.pools, from * INDEX_HEAD_DIM * 2, raw(&pools[from * INDEX_HEAD_DIM..to * INDEX_HEAD_DIM]));
        }
    }
    fn ix_select(&mut self, l: usize, pos: usize, n: usize, count: Option<&mut usize>) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the scores of every selecting token's pools, then each token's selection into the attend's lists;
        // nothing comes back to the host but a count asked for (bring-up)
        let stride = self.npools;
        if glm_5_3_flash::mla::selects(pos + n - 1) {
            // the scores on the matrix path (docs/queue/019): a tile of 16 tokens against 128 pools a
            // workgroup, over the pools the step's last token sees
            let visible = (pos + n) / INDEX_KPOOL;
            let (scale, wscale) = glm_5_3_flash::mla::score_scales();
            let mut a = Args::new();
            a.ptr(c.miq.va).ptr(c.mproj.at(MIX * 4)).ptr(m.pools.va).ptr(c.mscore.va).u32(n as u32).u32(visible as u32).u32(stride as u32).f32(scale).f32(wscale);
            c.k.launch(&c.ix_score_w, ixw_grid(n, visible), 256, IXW_LDS as u32, &a);
        }
        // the selection across slices of the pools at a verify step past 32k positions (docs/queue/019)
        let slices = select_slices(n, (pos + n) / INDEX_KPOOL);
        let mut a = Args::new();
        a.ptr(c.mscore.va).u32(stride as u32).u32(pos as u32).u32(slices as u32).ptr(c.mselst.va).ptr(c.msel.va).ptr(c.mseln.va);
        c.k.launch(&c.ix_select, (n * slices) as u32, g::WG, SELECT_LDS as u32, &a);
        if let Some(count) = count {
            c.k.sync();
            let mut w = [0u32; 1];
            read(&c.mseln, 0, &mut w);
            *count = w[0] as usize;
        }
    }
    fn mla_attend(&mut self, l: usize, n: usize, pos: usize, latents: &[u16]) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the step's latents into the mirror; the selections are the card's (`ix_select`). The attend on the
        // matrix path (docs/queue/018, 017): q_abs for every head, the attend over (token, 16 heads) — a
        // verify step's token in chunks of positions merged after — and W_uv over the step; the few-token
        // twins of q_abs and W_uv below SPEC_TOKENS, like the rows kernels'
        c.k.upload(&m.lat, pos * KV_LORA_RANK * 2, raw(&latents[..n * KV_LORA_RANK]));
        let mut a = Args::new();
        a.ptr(c.mq.va).ptr(m.kv_b.va).ptr(c.mqabs.va);
        if n > SPEC_TOKENS {
            a.u32(n as u32);
            c.k.launch(&c.mla_qabs_w, MLA_HEADS as u32 * g::wmma_grid(KV_LORA_RANK, n), g::WG, g::WMMA_LDS, &a);
        } else {
            c.k.launch(&c.mla_qabs, (n * MLA_HEADS) as u32, 256, 0, &a);
        }
        let (chunk, chunks) = (aw_chunk(n), aw_chunks(n));
        let mut a = Args::new();
        a.ptr(c.mqabs.va).ptr(m.lat.va).ptr(c.msel.va).ptr(c.mseln.va).ptr(c.molat.va).ptr(c.mpart.va).u32(chunks as u32).u32(chunk as u32);
        c.k.launch(&c.mla_attend_w, aw_grid(n), AW_WG, AW_LDS as u32, &a);
        if chunks > 1 {
            let mut a = Args::new();
            a.ptr(c.mpart.va).ptr(c.molat.va);
            c.k.launch(&c.mla_attend_merge, (n * AW_HEAD_TILES) as u32, AW_WG, AW_MERGE_LDS as u32, &a);
        }
        let mut a = Args::new();
        a.ptr(c.molat.va).ptr(m.kv_b.va).ptr(c.mheads.va);
        if n > SPEC_TOKENS {
            a.u32(n as u32);
            c.k.launch(&c.mla_uv_w, MLA_HEADS as u32 * g::wmma_grid(V_HEAD_DIM, n), g::WG, g::WMMA_LDS, &a);
        } else {
            c.k.launch(&c.mla_uv, (n * MLA_HEADS) as u32, 256, (KV_LORA_RANK * 4) as u32, &a);
        }
    }
    fn mla_out(&mut self, l: usize, n: usize) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the output projection over every token's heads, then the streams (the projection out to the peer's receiving buffer)
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_o_w, c.mheads.va, None, m.o.0.va, Some(m.o.1.va), c.myv.va, H, n, false);
        } else {
            self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.myv.va, H, O_IN, n, ROWS);
        }
        let copy = (0..self.cards.len()).find(|&j| j != m.card).map(|j| (c.myv.va, self.peers[j].my[l % 2].va, n * H));
        self.hc_post(c, copy, Some((W_Y, m.card)), None, None, self.half_seq[l][0], c.myv.va, None, n);
    }
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[u16]) {
        let m = self.mlas[l].as_ref().unwrap();
        self.cards[m.card].k.upload(&m.lat, pos * KV_LORA_RANK * 2, raw(&latent[..KV_LORA_RANK]));
    }
    fn ffn(&mut self, l: usize, n: usize) {
        let value = self.next_seq();
        self.half_seq[l][1] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            // the FFN card's collapse goes to the host too when the routed experts need it, with the router's logits and the word
            let to_host = (!f.dense && c.k.index == f.card).then_some(self.shared.xh.va);
            let router = self.ffn_router(l, c);
            self.hc_pre(c, &f.hc[c.k.index][1], to_host, value, router, n);
        }
        let c = &self.cards[f.card];
        if let Some((ln32, bias)) = &f.act {
            // the FFN input's norm, quantize and router pick here, into this card's activation buffers and the host's
            let xa = &self.shared.xact;
            let mut a = Args::new();
            a.ptr(c.x.va).ptr(ln32.va).ptr(bias.va).ptr(self.shared.scores_ffn.va).ptr(c.xq.va).ptr(c.xs.va).ptr(c.xg.va).ptr(xa.at(XA_Q)).ptr(xa.at(XA_S)).ptr(xa.at(XA_G)).ptr(xa.at(XA_P)).ptr(xa.at(XA_W)).ptr(c.xcount.va).ptr(self.shared.words.at(word(W_ACT, f.card))).u32(value).u32(n as u32);
            c.k.launch(&c.ffn_act, n as u32, g::WG, glm_5_3_flash::ffn::gfx11_consts::ACT_LDS, &a);
        }
        let (rows, down, k_width) = if f.dense { (DENSE_GU_ROWS, &c.ffn_down_dense, INTERMEDIATE_SIZE) } else { (SHARED_GU_ROWS, &c.ffn_down_shared, N_SHARED_EXPERTS * D_FF) };
        if n > SPEC_TOKENS {
            let down_w = if f.dense { &c.ffn_down_dense_w } else { &c.ffn_down_shared_w };
            self.rows_w(c, &c.ffn_gu_w, c.x.va, Some(f.ln.va), f.gu_w.va, Some(f.gu_s.va), c.gu.va, rows, n, false);
            self.rows_w(c, down_w, c.gu.va, None, f.down_w.va, Some(f.down_s.va), c.fyv.va, H, n, false);
        } else {
            self.rows8(c, &c.ffn_gu, c.x.va, Some(f.ln.va), f.gu_w.va, f.gu_s.va, c.gu.va, rows, H, n, ROWS_GU);
            self.rows8_at(c, down, c.gu.va, 2 * k_width, None, f.down_w.va, f.down_s.va, c.fyv.va, H, k_width, n, ROWS, true);
        }
        // a dense layer's output half now; a MoE layer's waits for the host's experts and is posted after the
        // resident experts' launches (`ffn_post`): a queue is in order, and the output half spins on the word
        if f.dense {
            self.ffn_post(l, n);
        }
    }
    fn ffn_post(&mut self, l: usize, n: usize) {
        let value = self.half_seq[l][1];
        let f = self.ffns[l].as_ref().unwrap();
        let parity = l % 2;
        for o in &self.cards {
            let me = o.k.index;
            let producer = me == f.card;
            // the producer's rows stay in its `fyv`; the peer reads them from its receiving buffer
            let copy = if producer { (0..self.cards.len()).find(|&j| j != me).map(|j| (o.fyv.va, self.peers[j].fy[parity].va, n * H)) } else { None };
            let fy = if producer { o.fyv.va } else { self.peers[me].fy[parity].va };
            let mine = producer.then_some((W_FFN, f.card));
            if f.dense {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), None, value, fy, None, n);
            } else {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), Some((W_EXPERTS, 0)), value, fy, Some(self.shared.yexp[parity].va), n);
            }
        }
    }
    fn ffn_logits(&mut self, l: usize, n: usize, x: &mut [f32], logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_X, f.card), self.half_seq[l][1]);
        read(&self.shared.xh, 0, &mut x[..n * H]);
        read(&self.shared.scores_ffn, 0, &mut logits[..n * ROUTE_ROWS]);
    }
    fn ffn_input(&mut self, l: usize, n: usize, xq: &mut [i8], xs: &mut [f32], xg: &mut [i32], picks: &mut [[usize; TOP_K]], wts: &mut [[f32; TOP_K]], logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_ACT, f.card), self.half_seq[l][1]);
        // the router's logits too: their second half is the next layer's rows, the prefetch's predictor (005)
        read(&self.shared.scores_ffn, 0, &mut logits[..n * ROUTE_ROWS]);
        let xa = &self.shared.xact;
        read(xa, XA_Q, &mut xq[..n * H]);
        read(xa, XA_S, &mut xs[..n * HG]);
        read(xa, XA_G, &mut xg[..n * HG]);
        let mut p = vec![0u32; n * TOP_K];
        read(xa, XA_P, &mut p);
        for (i, pk) in picks[..n].iter_mut().enumerate() {
            for k in 0..TOP_K {
                pk[k] = p[i * TOP_K + k] as usize;
            }
        }
        read(xa, XA_W, unsafe { std::slice::from_raw_parts_mut(wts.as_mut_ptr() as *mut f32, n * TOP_K) });
    }
    fn ffn_routed(&mut self, l: usize, n: usize, y: &[f32]) {
        // streamed: the handoff buffer is write-combining for the CPU, and a plain copy of a 1024-token
        // step's 16 MB took 4.8 ms a layer (0.2 ms a prompt token, 2026-10-19)
        let buf = &self.shared.yexp[l % 2];
        common::copy::stream(unsafe { std::slice::from_raw_parts_mut(buf.ptr() as *mut u8, n * H * 4) }, raw(&y[..n * H]));
        // the word into every card's own copy (the cards wait on their own VRAM, 001)
        for pe in &self.peers {
            unsafe { &*(pe.words.at(word(W_EXPERTS, 0)) as *const AtomicU32) }.store(self.half_seq[l][1], std::sync::atomic::Ordering::Release);
        }
    }
    fn layer_out(&mut self, _l: usize, i: usize, mean: &mut [f32]) {
        let c = &self.cards[self.head_card];
        let mut a = Args::new();
        a.ptr(c.streams.at(i * HC * H * 4)).ptr(c.mean.va).u32(1);
        c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
        c.k.sync();
        read(&c.mean, 0, &mut mean[..H]);
    }
    fn logits(&mut self, from: usize, n: usize, logits: &mut [f32]) {
        // a pass of TOKENS at a time, on every card at once: the streams' means (every card holds the streams),
        // the norm and this card's rows of lm_head over them; the parts read back a token into the one vector
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            for (ci, c) in self.cards.iter().enumerate() {
                let (norm, lm) = &self.head[ci];
                let (_, rows) = head_rows(ci, self.cards.len());
                let mut a = Args::new();
                a.ptr(c.streams.at((from + p0) * HC * H * 4)).ptr(c.fx.va).u32(np as u32);
                c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
                if np > SPEC_TOKENS {
                    let mut a = Args::new();
                    a.ptr(c.fx.va).ptr(norm.va).ptr(lm.va).ptr(c.lg.va).u32(rows as u32).u32(np as u32);
                    c.k.launch(&c.rows_h_w, g::wmma_grid(rows, np), g::WG, g::WMMA_LDS, &a);
                } else {
                    self.rows(c, &c.rows_h, c.fx.va, Some(norm.va), lm.va, c.lg.va, rows, H, np, ROWS);
                }
            }
            for (ci, c) in self.cards.iter().enumerate() {
                let (r0, rows) = head_rows(ci, self.cards.len());
                c.k.sync();
                for i in 0..np {
                    read(&c.lg, i * rows * 4, &mut logits[(p0 + i) * VOCAB_SIZE + r0..(p0 + i) * VOCAB_SIZE + r0 + rows]);
                }
            }
            p0 += np;
        }
        // the step ends when every card is done with its streams: the next begins by overwriting them
        for c in &self.cards {
            c.k.sync();
        }
        self.timeline_report();
    }
    fn finish(&mut self) {
        for c in &self.cards {
            c.k.sync();
        }
        self.timeline_report();
    }
    fn kda_rewind(&mut self, l: usize, slot: usize) {
        // every part: its state buffers from the ring's slot (a copy launch each; the ring holds every
        // part's heads' entries where its own buffers do)
        for p in &self.layers[l].as_ref().unwrap().parts {
            let c = &self.cards[p.card];
            let copy = |src: usize, dst: usize, n: usize| {
                let mut a = Args::new();
                a.ptr(src).ptr(dst).u32(n as u32);
                c.k.launch(&c.copy, n.div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &a);
            };
            copy(p.ring.at(slot * KDA_MEMORY * 4), p.mem.va, KDA_MEMORY);
            copy(p.ring_w.at(slot * KDA_WINDOW * 4), p.window.va, KDA_WINDOW);
        }
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
