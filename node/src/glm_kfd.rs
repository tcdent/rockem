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
use glm_5_3_flash::hc::gfx11_consts::{ACT_SLOTS, CHANNEL_GRID, MIX as HC_MIX, PRE_COUNTS, PRE_GRID, PRE_LDS, PRE_STRIDE, ROUTE_ROWS};
use glm_5_3_flash::kda::gfx11_consts as g;
use common::gemv::ROWS;
use glm_5_3_flash::ffn::gfx11_consts::ROWS_GU;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::mla::gfx11_consts::{lat_local, lat_owner, lat_rows, LAT_GRAIN, aw_chunk, aw_chunks, aw_grid, ixw_grid, select_slices, select_state_init, AW_DECODE_CHUNKS, AW_HEAD_TILES, AW_LDS, AW_MERGE_LDS, AW_PART, AW_WG, IXW_LDS, IX_ROWS, O_IN, ROWS_QB, SELECT_LDS, SELECT_STATE};
use glm_5_3_flash::expert::gfx11_consts::{DOWN_WGS as XPT_DOWN_WGS, SLOT_BYTES, UP_WGS as XPT_UP_WGS, XG_DOWN_WGS, XG_GU_WGS, XG_MAX_JOBS, XPT_LDS, XQ_GROUPS, XQ_WGS};
use glm_5_3_flash::{CHUNKS, DENSE_BELOW};
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
    /// `[MOE_LAYERS]`: a bit a expert picked this window (since the last half-life); folded into
    /// `window_seen` at the half-life, so the `experts` row can read a layer's distinct experts
    /// against its `held` (063's instrument).
    window: Vec<[u64; (glm_5_3_flash::E + 63) / 64]>,
    /// `[MOE_LAYERS]`: distinct experts picked over the last window, published at the fold.
    window_seen: Vec<u32>,
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
/// The counts' shift when a new prompt arrives (the first prefill step after decode steps): the
/// residency is the current text's hotlist, not the box's lifetime average — served, the routing
/// touches nearly every expert of a layer within a half-life's window (the `experts` row's `seen`,
/// 2026-10-24: median 224 of 288, p90 288, at 8 slots a layer), so counts summed across requests
/// hold a global hotlist that matches none of them (the node's own note: a static hotlist does not
/// travel between texts, 0.19 of picks). A shift of 4 puts the last prompt's picks 16× over the
/// one before it, and the routing replay's per-text numbers (0.438 / 0.364, 2026-10-18) are the
/// target this aging buys back.
const RESIDENT_PROMPT_AGE: u32 = 4;
/// VRAM a card may spend on resident experts at most (the node); what is free past a reserve decides.
pub const RESIDENT_BYTES: usize = 12 << 30;
/// VRAM left unallocated on a card: the driver's and the queue's own, and room for a bring-up gate's buffers.
const RESIDENT_RESERVE: usize = 1 << 30;

struct Card {
    k: Rc<Kfd>,
    xpt_gu: crate::kfd::Kernel,
    xpt_down: crate::kfd::Kernel,
    /// The FFN router's scores in VRAM beside the host's copy (`[STEP][ROUTE_ROWS]` f32): `hc_pre`'s act phase reads them.
    scores_v: Mem,
    /// The FFN input's routing weights in VRAM (`[STEP][TOP_K]` f32): the act phase's copy, the leader's routed sum reads it (002).
    wv: Mem,
    xpt_gemm_gu: crate::kfd::Kernel,
    xpt_quant: crate::kfd::Kernel,
    xpt_gemm_down: crate::kfd::Kernel,
    /// The resident experts' work (038): the step's quantized activations and the group/job tables
    /// (host-visible), the jobs' gate/up state and quantized hidden states (VRAM); the outputs go to
    /// the leader's rows (`Peer::rows`).
    xq: Mem,
    xs: Mem,
    xg: Mem,
    xjobs: Mem,
    xgu: Mem,
    xhq: Mem,
    xhs: Mem,
    xhg: Mem,
    rows_h: crate::kfd::Kernel,
    rows_h1: crate::kfd::Kernel,
    rows_h_w: crate::kfd::Kernel,
    rows_d2: crate::kfd::Kernel,
    // the KDA matrices are held in exp4 (029): the decode path's kernels read it, a prefill step unpacks it to bf16
    rows_h_e4: crate::kfd::Kernel,
    rows_d2_e4: crate::kfd::Kernel,
    rows_part_e4: crate::kfd::Kernel,
    exp4_unpack: crate::kfd::Kernel,
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
    hc_route_ahead: crate::kfd::Kernel,
    /// The prefetch predictor's prediction and update, `plan::gfx11`; its second margin's input (the FFN input plus the
    /// shared expert's output, `[STEP][H]` f32), the router's logits over it (`[STEP][ROUTE_ROWS]` f32, layer l + 1's
    /// from `E`, as `scores_v`) and the few-token router's word (unread).
    pred_step: crate::kfd::Kernel,
    pred_rows: crate::kfd::Kernel,
    pred_learn: crate::kfd::Kernel,
    pred_input: crate::kfd::Kernel,
    /// The rows pass's input from `hc_post`: `[STEP][H]` f32 (a long step's rows kernel), the same normed as bf16 with
    /// the waves' partial sums of squares (a decode step's `pred_rows`) (093).
    pred_x: Mem,
    pred_xn: Mem,
    pred_ss: Mem,
    pred_s2: Mem,
    pred_word: Mem,
    /// `pred_step`'s per-token key terms and its workgroups' arrival count (093).
    pred_keys: Mem,
    pred_count: Mem,
    /// The early pass's co-routing sums `[SPEC_TOKENS][E]` and thresholds `[SPEC_TOKENS]` for the rows pass (093).
    pred_aux: Mem,
    mla_sel_split: crate::kfd::Kernel,
    mla_two_merge: crate::kfd::Kernel,
    word_wait: crate::kfd::Kernel,
    word_post: crate::kfd::Kernel,
    // the MLA attend's half on this card (058): its rows of the selection, its (max, sum) and value-space heads, the split's counter
    mown: Mem,   // [STEP][DENSE_BELOW] u32: this card's local rows of each token's selection (the MLA card's split)
    mown_n: Mem, // [STEP] u32
    mml: Mem,    // [STEP][MLA_HEADS][2] f32
    mhd: Mem,    // [STEP][O_IN] f32: W_uv over this card's unnormalized mix
    mdone: Mem,  // the split's arrival counter
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
    // a prefill step's KDA matrix as bf16 (`exp4_unpack`): `sq` the largest, `o` after it in the same buffer (the queue's
    // order and barriers keep them apart), the gates' `fg` beside them
    kda_bf: Mem,    // [SMALL_ROWS + 3·PART_QKV][H] bf16
    kda_bf_fg: Mem, // [2·PART_QKV][KDA_DIM] bf16
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
    yexp: [Mem; 2], // [STEP][H] f32: the draft layer's routed experts, staged by the host (`mtp_ffn`)
    xh: Mem,       // [STEP][H] f32: a MoE layer's FFN input for the host
    xact: Mem,     // the FFN input's quantized activations [STEP][H] i8 | [STEP][HG] f32 | [STEP][HG] i32, then the picks [STEP][TOP_K] u32 and weights f32: `hc_pre`'s act phase, for the host (XA_*)
    scores_ffn: Mem, // [STEP][ROUTE_ROWS] f32: the FFN half's router over its input (this layer's rows, then the next layer's: the prefetch's predictor), for the host's picks
    labels: Mem,   // [NUM_HIDDEN_LAYERS][STEP][TOP_K] u32: each MoE layer's picks, for the prefetch predictor's pairs on the cards
    plan: Mem,     // [plan::PLAN] u32: a decode step's plan for the next layer, best first (`pred_step`)
    plan2: Mem,    // [plan::PLAN] u32: the same from the second pass, behind `hc_post`, over the whole FFN output (093)
    words: Mem,    // the handoff words, a cache line each: `word()`
}
/// A handoff word's byte offset in `Shared::words`: the kind, and the card for a block output.
/// The kernels a stats code object stamps (`common::stats`), reported by `timeline_report`.
/// `hc_pre`'s slots are its phases in order; `pred_rows` and `mla_o` (008's FP8 rows question,
/// 2026-10-24) run through `rows_pass`, whose slots are its phases: 0 the activation staging
/// rounds, 1 the weight rounds (the FP8 decode and dot), 2 the tail (the norm's reduction, the
/// ready, the emits) — shader cycles a wave, wave 0 of each workgroup the clean read.
const STAT_KERNELS: &[&str] = &["hc_pre", "pred_rows", "mla_o"];
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
/// A chunk's set of handoff words (037): two chunks of a step in flight carry their own values.
const WORD_SET: usize = word(W_KINDS, 0);
const MAX_CARDS: usize = 4;
/// The attention block's output is in (a card's), the early router's logits are in (the FFN card's),
/// the FFN half's input is in (the FFN card's), the FFN rows are in (the FFN card's), the host's experts are in.
const W_Y: usize = 0;
const W_X: usize = 2;
const W_FFN: usize = 3;
const W_EXPERTS: usize = 4;
/// The FFN input's quantized activations, picks and weights are in (the FFN card's `hc_pre` act phase, for the host).
const W_ACT: usize = 5;
/// The MLA attend split across the cards (058): the other card's selected rows are in, the MLA card's `q_a` output is
/// in (the other card's copy), the other card's heads (value space) and (max, sum) are in (the MLA card's copy).
const W_SEL: usize = 6;
const W_MQA: usize = 7;
const W_HD: usize = 8;
/// The FFN card's quantized FFN input is in the other card's receiving buffers, for its resident experts.
const W_XQ: usize = 9;
/// A decode step's prefetch plan is in (the FFN card's `pred_step`, for the host).
const W_PLAN: usize = 10;
/// The rows-based plan (093): the predictor's second pass behind `hc_post`, over the whole FFN output.
const W_PLAN2: usize = 11;
const W_KINDS: usize = 12;
/// The resident experts' completion line (`Kfd::fence_line`): a prefill chunk's residents are waited for after the
/// other chunk's card half has posted and waited for its own fences (037).
const XPT_LINE: usize = 1;
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
    sq: Mem,       // [SMALL_ROWS + 3·PART_QKV][H] exp4: f_a's rows, g_a's, b's, then the part's q rows, k rows, v rows — one launch
    o: Mem,        // [H][PART_QKV] exp4
    taps: Mem,     // [3][KDA_QKV][CONV_TAPS] bf16
    fg: Mem,       // [2·PART_QKV][KDA_DIM] exp4: the part's f_b rows, g_b rows
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
    /// A MoE layer's post-norm weights (f32), the router's selection bias (f32) and `hc_pre`'s act table
    /// (`ACT_SLOTS` device addresses) on its FFN card.
    act: Option<(Mem, Mem, Mem)>,
    /// The prefetch predictor's pair (this layer → the next MoE layer) on this card: its state (`plan::STATE` f32) and
    /// its tokens' two margins `[STEP][E]` f32 each, kept from its prediction for its update.
    pred: Option<(Mem, Mem, Mem)>,
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
    /// The latents, the host's cache rows mirrored, split by position across the cards (058): `lats[c]` holds the
    /// positions `lat_owner` gives card `c`, at `lat_local` rows (`[lat_rows(max_ctx)][KV_LORA_RANK]` bf16).
    lats: Vec<Mem>,
    /// The other card's copies of what its half of the attend needs: `q_a`'s norm, `q_b`, `kv_b`.
    other: usize,
    qa_ln2: Mem,
    q_b2: (Mem, Mem),
    kv_b2: Mem,
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
    /// The routed experts' output rows when this card leads a MoE layer (002): `[STEP · TOP_K][H]` f32 by (token,
    /// pick), written by the host's experts through the BAR and by the cards' resident jobs (the peer's by peer write).
    rows: Mem,
    /// The leader's routed sums, forwarded here when the peer leads (`[STEP][H]` f32, by layer parity).
    yr: [Mem; 2],
    /// The MLA attend split (058), when the other card leads the layer: its selection's rows for this card and their
    /// counts, its `q_a` output; when this card leads: the other card's value-space heads and (max, sum).
    psel: Mem,   // [STEP][DENSE_BELOW] u32
    psel_n: Mem, // [STEP] u32
    pmqa: Mem,   // [STEP][Q_LORA_RANK] f32
    phd: Mem,    // [STEP][O_IN] f32
    pml: Mem,    // [STEP][MLA_HEADS][2] f32
    /// The FFN card's quantized FFN input (`hc_pre`'s act phase) for this card's resident experts when the other card
    /// is the layer's FFN card: `[STEP][H]` i8, `[STEP][HG]` f32 scales, `[STEP][HG]` i32 sums — copied card to card (it
    /// was the host's upload through the BAR, ~1 ms a prefill chunk on its thread, 2026-10-20).
    axq: Mem,
    axs: Mem,
    axg: Mem,
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
    /// The prefetch predictor's pairs for `pred_learn`: each card's table (`plan::LEARN_TAB` addresses a pair) and its
    /// pair count.
    pred_tab: Vec<(Mem, usize)>,
    /// The handoff sequence: a value a half of a layer's words carry, and each layer's (attention, FFN) values this
    /// step, a pair a chunk.
    seq: u32,
    half_seq: Vec<[[u32; 2]; CHUNKS]>,
    /// The chunk the calls address (037): its index (its words) and its first token (its rows in every step buffer).
    ch: usize,
    t0: usize,
    /// The step's chunks (`glm_5_3_flash::chunk_count`).
    chunks: usize,
    /// The value an MLA layer's split-attend words carry this call (058): drawn when its projections are posted.
    mla_value: u32,
    /// With the launches stamped: each card's first packet of the step.
    token_from: Vec<u64>,
    /// Pools the context can hold: the scores' row stride.
    npools: usize,
}

/// The other card's copies for its half of an MLA layer's attend (058): (the other card, `q_a`'s norm, `q_b`, `kv_b`).
fn mla_other(cards: &[Card], card: usize, mw: &glm_5_3_flash::weights::Mla) -> (usize, Mem, (Mem, Mem), Mem) {
    assert!(cards.len() == 2, "the MLA attend is split across two cards (058): {} cards", cards.len());
    let other = 1 - card;
    let k = &cards[other].k;
    let up = |b: &[u8]| -> Mem {
        let m = k.vram_visible(b.len());
        k.upload(&m, 0, b);
        m
    };
    let wm = k.vram_visible(mw.q_b.w.len());
    k.upload(&wm, 0, mw.q_b.w);
    let sm = k.vram_visible(mw.q_b.s.len() * 4);
    k.upload(&sm, 0, raw(mw.q_b.s));
    (other, up(raw(&bf16_of(mw.q_a_ln))), (wm, sm), up(raw(mw.kv_b)))
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
                    rows_h1: k.kernel("rows_h1"),
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
                    rows_h_e4: k.kernel("rows_h_e4"),
                    rows_d2_e4: k.kernel("rows_d2_e4"),
                    rows_part_e4: k.kernel("rows_part_e4"),
                    exp4_unpack: k.kernel("exp4_unpack"),
                    step: k.kernel("kda_step"),
                    copy: k.kernel("copy_f32"),
                    xpt_gu: k.kernel("xpt_gu"),
                    xpt_down: k.kernel("xpt_down"),
                    scores_v: k.vram(STEP * ROUTE_ROWS * 4),
                    wv: k.vram(STEP * TOP_K * 4),
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
                    hc_route_ahead: k.kernel("hc_route_ahead"),
                    pred_step: k.kernel("pred_step"),
                    pred_rows: k.kernel("pred_rows"),
                    pred_learn: k.kernel("pred_learn"),
                    pred_input: k.kernel("pred_input"),
                    pred_x: k.vram(STEP * H * 4),
                    pred_xn: k.vram(STEP * H * 2),
                    pred_ss: k.vram(STEP * glm_5_3_flash::plan::PRED_SS * 4),
                    pred_s2: k.vram(STEP * ROUTE_ROWS * 4),
                    pred_word: k.vram(64),
                    pred_keys: k.vram(SPEC_TOKENS * glm_5_3_flash::E * 4),
                    pred_aux: k.vram((SPEC_TOKENS * glm_5_3_flash::E + SPEC_TOKENS) * 4),
                    // host-visible: zeroed by an upload (a device-only `vram` buffer cannot be uploaded to — the node
                    // segfaulted in its card setup on the first try, 2026-10-22)
                    pred_count: {
                        let m = k.vram_visible(64);
                        k.upload(&m, 0, &[0u8; 64]);
                        m
                    },
                    mla_sel_split: k.kernel("mla_sel_split"),
                    mla_two_merge: k.kernel("mla_two_merge"),
                    word_wait: k.kernel("word_wait"),
                    word_post: k.kernel("word_post"),
                    mown: k.vram(STEP * DENSE_BELOW * 4),
                    mown_n: k.vram(STEP * 4),
                    mml: k.vram(STEP * MLA_HEADS * 2 * 4),
                    mhd: k.vram(STEP * O_IN * 4),
                    mdone: {
                        let m = k.vram_visible(64);
                        k.upload(&m, 0, &[0u8; 64]);
                        m
                    },
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
                    kda_bf: k.vram((SMALL_ROWS + 3 * g::PART_QKV) * H * 2),
                    kda_bf_fg: k.vram(2 * g::PART_QKV * KDA_DIM * 2),
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
                let words = k.vram_shared(CHUNKS * WORD_SET, &others);
                unsafe { std::ptr::write_bytes(words.ptr() as *mut u8, 0, words.size) };
                Peer { ys: [buf(), buf()], fy: [buf(), buf()], my: [buf(), buf()], rows: k.vram_shared(STEP * TOP_K * H * 4, &others), yr: [buf(), buf()], psel: k.vram_shared(STEP * DENSE_BELOW * 4, &others), psel_n: k.vram_shared(STEP * 4, &others), pmqa: k.vram_shared(STEP * Q_LORA_RANK * 4, &others), phd: k.vram_shared(STEP * O_IN * 4, &others), pml: k.vram_shared(STEP * MLA_HEADS * 2 * 4, &others), axq: k.vram_shared(STEP * H, &others), axs: k.vram_shared(STEP * HG * 4, &others), axg: k.vram_shared(STEP * HG * 4, &others), words }
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
                xact: k0.host_shared(XA_END, &others),
                scores_ffn: k0.host_shared(STEP * ROUTE_ROWS * 4, &others),
                labels: k0.host_shared(NUM_HIDDEN_LAYERS * STEP * TOP_K * 4, &others),
                plan: k0.host_shared(glm_5_3_flash::plan::PLAN * 4, &others),
                plan2: k0.host_shared(glm_5_3_flash::plan::PLAN * 4, &others),
                words: {
                    let m = k0.host_shared(CHUNKS * WORD_SET, &others);
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
                // the part's three matrices in exp4 (029: lossless, 77 % of the bytes), encoded here from the bundle's bf16
                let exp4 = |w: &[u16], m: usize, kk: usize| -> (Mem, usize) {
                    let e = common::exp4::encode(w, m, kk);
                    let b = e.planes();
                    let mm = k.vram_visible(b.len());
                    k.upload(&mm, 0, &b);
                    (mm, b.len())
                };
                let (sq, sq_bytes) = {
                    let (q, kk, v) = (rows(kw.q, H, head0), rows(kw.k, H, head0), rows(kw.v, H, head0));
                    let mut w: Vec<u16> = Vec::with_capacity((SMALL_ROWS + 3 * g::PART_QKV) * H);
                    for part in [kw.f_a, kw.g_a, kw.b, q, kk, v] {
                        w.extend_from_slice(part);
                    }
                    exp4(&w, SMALL_ROWS + 3 * g::PART_QKV, H)
                };
                let (o, o_bytes) = exp4(&o_half, H, g::PART_QKV);
                let (fg, fg_bytes) = {
                    let mut w = rows(kw.f_b, KDA_DIM, head0).to_vec();
                    w.extend_from_slice(rows(kw.g_b, KDA_DIM, head0));
                    exp4(&w, 2 * g::PART_QKV, KDA_DIM)
                };
                let p = Part {
                    card,
                    head0,
                    input_ln: up(raw(&input_ln)),
                    sq,
                    o,
                    taps: up(raw(&taps)),
                    fg,
                    dt_bias: up(raw(kw.dt_bias)),
                    a_log: up(raw(kw.a_log)),
                    o_norm: up(raw(kw.o_norm)),
                    window: up(raw(&zero_w)),
                    mem: up(raw(&zero_m)),
                    ring: k.vram(SPEC_TOKENS * KDA_MEMORY * 4),
                    ring_w: k.vram(SPEC_TOKENS * KDA_WINDOW * 4),
                };
                bytes += sq_bytes + o_bytes + fg_bytes + 2 * taps.len() + 4 * (KDA_WINDOW + KDA_MEMORY);
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
                    let cd = &cards[card];
                    let xa = &shared.xact;
                    let tab: [u64; ACT_SLOTS] = [ln32.va, bias.va, cd.xq.va, cd.xs.va, cd.xg.va, xa.at(XA_Q), xa.at(XA_S), xa.at(XA_G), xa.at(XA_P), xa.at(XA_W), cd.wv.va].map(|a| a as u64);
                    let table = k.vram_visible(ACT_SLOTS * 8);
                    k.upload(&table, 0, raw(&tab));
                    Some((ln32, bias, table))
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
            // the prefetch predictor's pair where the next layer is MoE too: its state from the next layer's router bias
            let pred = match (&w.layers[l].ffn, w.layers.get(l + 1).map(|lw| &lw.ffn)) {
                (Ffn::Moe(_), Some(Ffn::Moe(m2))) => {
                    let k = &cards[card].k;
                    let st = k.vram_visible(glm_5_3_flash::plan::STATE * 4);
                    k.upload(&st, 0, raw(&glm_5_3_flash::plan::init(m2.bias)));
                    let (d, d2) = (k.vram(STEP * glm_5_3_flash::E * 4), k.vram(STEP * glm_5_3_flash::E * 4));
                    bytes += st.size + d.size + d2.size;
                    Some((st, d, d2))
                }
                _ => None,
            };
            ffns.push(Some(FfnOn { card, dense: is_dense, ln, hc, gu_w, gu_s, down_w, down_s, router, act, pred }));
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
            let (other, qa_ln2, q_b2, kv_b2) = mla_other(&cards, card, mw);
            let on = MlaOn { card, input_ln: up(raw(&bf16_of(w.layers[l].input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lats: (0..cards.len()).map(|c| cards[c].k.vram_visible(lat_rows(max_ctx) * KV_LORA_RANK * 2)).collect(), other, qa_ln2, q_b2, kv_b2, pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 2), o: one(mw.o) };
            bytes += on.q_a.0.size + on.q_a.1.size + on.kv_a.0.size + on.kv_a.1.size + on.ix.size + on.q_b.0.size + on.q_b.1.size + on.ix_wq_b.size + on.kv_b.size + on.lats[card].size + on.o.0.size + on.o.1.size;
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
            let (other, qa_ln2, q_b2, kv_b2) = mla_other(&cards, card, mw);
            mlas.push(Some(MlaOn { card, input_ln: up(raw(&bf16_of(mt.input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lats: (0..cards.len()).map(|c| cards[c].k.vram_visible(lat_rows(max_ctx) * KV_LORA_RANK * 2)).collect(), other, qa_ln2, q_b2, kv_b2, pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 2), o: one(mw.o) }));
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
            Resident { table: std::sync::Arc::new((0..moe * glm_5_3_flash::E).map(|_| AtomicU32::new(NONE)).collect()), loader: None, inflight: Default::default(), allowed: Default::default(), issued: Default::default(), steps: 0, prefill: false, count: vec![0; moe * glm_5_3_flash::E], turn: 0, owner: counts.iter().map(|&k| vec![(NONE, NONE); k]).collect(), free: counts.iter().map(|&k| (0..k as u32).rev().collect()).collect(), held: (0..n).map(|_| vec![0; moe]).collect(), per_layer, slots: slots, jobs: (0..n).map(|_| Vec::new()).collect(), uploads: 0, window: (0..moe).map(|_| [0; (glm_5_3_flash::E + 63) / 64]).collect(), window_seen: vec![0; moe] }
        };
        // the prefetch predictor's pairs, a table a card
        let pred_tab: Vec<(Mem, usize)> = cards
            .iter()
            .map(|c| {
                let mut t: Vec<u64> = Vec::new();
                for (l, f) in ffns.iter().enumerate() {
                    if let Some((st, d, d2)) = f.as_ref().filter(|f| f.card == c.k.index).and_then(|f| f.pred.as_ref()) {
                        let slot = |l: usize| shared.labels.at(l * STEP * TOP_K * 4) as u64;
                        t.extend_from_slice(&[st.va as u64, d.va as u64, slot(l), slot(l + 1), d2.va as u64]);
                    }
                }
                let m = c.k.vram_visible((t.len() * 8).max(8));
                c.k.upload(&m, 0, raw(&t));
                (m, t.len() / glm_5_3_flash::plan::LEARN_TAB)
            })
            .collect();
        Cards { cards, peers, shared, layers, ffns, mlas, head, head_card, mtp, resident, pred_tab, seq: 0, half_seq: vec![[[0; 2]; CHUNKS]; NUM_HIDDEN_LAYERS], ch: 0, t0: 0, chunks: 1, mla_value: 0, token_from: vec![0; n], npools: max_ctx.div_ceil(INDEX_KPOOL) }
    }

    /// With the launches stamped (`track-launches`): the step's launches on every card as zones on
    /// lanes `card0`, `card1`, …, their completion stamps in UTC through the card's clock. A step of
    /// more launches than the ring holds keeps its last `RING_PACKETS`. Reported launches are not reported again:
    /// the window moves past them (the draft after a step reports its own, 081).
    fn timeline_report(&mut self) {
        if !self.cards[0].k.instrumented() {
            return;
        }
        self.timeline_lanes();
        for (i, c) in self.cards.iter().enumerate() {
            self.token_from[i] = c.k.posted();
        }
    }
    fn timeline_lanes(&self) {
        for (i, c) in self.cards.iter().enumerate() {
            let lane = format!("card{}", c.k.index);
            let tl = c.k.timeline_from(self.token_from[i]);
            for &(name, s, e) in &tl {
                if e > s {
                    prow::zone_at!("track-launches", if name.is_empty() { "barrier" } else { name }, lane.as_str(), c.k.utc(s), c.k.utc(e));
                }
            }
            // a stats code object: the stamped kernels' phases in shader cycles, a row a launch — wave 0's value, the
            // mean and the max over the waves a slot (what each slot times is the kernel's own comment)
            let first = self.token_from[i].max(c.k.posted().saturating_sub(tl.len() as u64));
            for (idx, name, slots) in c.k.stats_from(first, STAT_KERNELS) {
                let at = (idx - first) as usize;
                let dur_us = tl.get(at).map_or(0.0, |&(_, s, e)| (e.saturating_sub(s)) as f64 / 100.0);
                let w0 = |k: usize| slots[k].first().copied().unwrap_or(0.0) as f64;
                let mean = |k: usize| if slots[k].is_empty() { 0.0 } else { slots[k].iter().map(|&v| v as f64).sum::<f64>() / slots[k].len() as f64 };
                let max = |k: usize| slots[k].iter().copied().fold(0f32, f32::max) as f64;
                // over the workgroups' wave 0 only (every `STAT_WAVES_PER_WG`th entry): the slots a kernel stamps from
                // wave 0 of every workgroup read clean here, where the mean and max over all waves hold stale entries
                let wg0 = |k: usize| slots[k].iter().step_by(common::stats::STAT_WAVES_PER_WG as usize).copied();
                let wgmin = |k: usize| wg0(k).fold(f32::INFINITY, f32::min) as f64;
                let wgmax = |k: usize| wg0(k).fold(0f32, f32::max) as f64;
                let wgdistinct = |k: usize| { let mut v: Vec<u32> = wg0(k).map(f32::to_bits).collect(); v.sort_unstable(); v.dedup(); v.len() };
                prow::row!("track-launches", "kernel_stats", { "lane" => lane.as_str(), "kernel" => name, "dur_us" => dur_us, "waves" => slots[0].len(),
                    "s0_w0" => w0(0), "s0_mean" => mean(0), "s0_max" => max(0), "s1_w0" => w0(1), "s1_mean" => mean(1), "s1_max" => max(1),
                    "s2_w0" => w0(2), "s2_mean" => mean(2), "s2_max" => max(2), "s3_w0" => w0(3), "s3_mean" => mean(3), "s3_max" => max(3),
                    "s0_wgmin" => wgmin(0), "s0_wgmax" => wgmax(0), "s1_wgmin" => wgmin(1), "s1_wgmax" => wgmax(1), "s2_wgmin" => wgmin(2), "s2_wgmax" => wgmax(2), "s2_wgdistinct" => wgdistinct(2) });
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
        self.shared.words.at(self.wd(kind, card)) as *const AtomicU32
    }
    /// A handoff word's offset in the chunk's set.
    fn wd(&self, kind: usize, card: usize) -> usize {
        self.ch * WORD_SET + word(kind, card)
    }
    /// Device address of the chunk's rows in a step buffer of `width` f32 a token (`tokb` for bytes a token).
    fn tok(&self, m: &Mem, width: usize) -> usize {
        m.at(self.t0 * width * 4)
    }
    fn tokb(&self, m: &Mem, bytes: usize) -> usize {
        m.at(self.t0 * bytes)
    }
    /// The hyper-connection's input half on card `c`, one launch: the coefficients over its streams,
    /// the mix, the collapse into the block input `x`.
    /// `router`: (the FFN half's norm, the router rows, the host's scores, the host's word) — the
    /// router over the collapse, run by the same launch, its logits and word to the host.
    /// `act`: a MoE layer's act table on its FFN card — the FFN input's norm, quantize and router picks done in
    /// the launch's tail once the scores are in (queue 001), the host's word at its end.
    #[allow(clippy::too_many_arguments)]
    fn hc_pre(&self, c: &Card, hc: &(Mem, Mem, Mem), value: u32, router: Option<(usize, usize, usize, usize)>, act: Option<usize>, n: usize) {
        // the chunk's rows (037); the act phase's table rows from its first token, and its word in the chunk's set
        let act_word = if act.is_some() { self.shared.words.at(self.wd(W_ACT, c.k.index)) } else { 0 };
        let (ln, rw, sc, wd) = router.unwrap_or((0, 0, 0, 0));
        // this layer's router rows here; a step of a few tokens runs the next layer's (the prefetch's predictor) in
        // `hc_route_ahead` behind this launch, which then writes the router's word — a host-memory store leaves only at a
        // launch's end, so rows behind the act phase's word in this launch held the word (2026-10-20). A long step names
        // most experts and prefetches nothing.
        // a long step's router takes the next layer's rows too (the matrix path, the same launch): the prefetch
        // predictor's labels from every prompt position (phase 4)
        let rows = if router.is_none() { 0 } else if n > SPEC_TOKENS { ROUTE_ROWS } else { glm_5_3_flash::E };
        let ahead = router.is_some() && n <= SPEC_TOKENS;
        let mut a = Args::new();
        a.ptr(self.tok(&c.streams, HC * H)).ptr(hc.0.va).ptr(self.tok(&c.part, PRE_STRIDE)).ptr(hc.1.va).ptr(hc.2.va).ptr(self.tok(&c.mix, HC_MIX)).ptr(self.tok(&c.x, H)).ptr(c.count.va).u32(value).ptr(ln).ptr(rw).ptr(sc).ptr(self.tok(&c.scores_v, ROUTE_ROWS)).ptr(if ahead { 0 } else { wd }).ptr(act.unwrap_or(0)).ptr(act_word).u32(self.t0 as u32).u32(rows as u32).u32(n as u32);
        c.k.launch(&c.hc_pre, PRE_GRID, g::WG, PRE_LDS, &a);
        if ahead {
            let mut b = Args::new();
            b.ptr(self.tok(&c.x, H)).ptr(ln).ptr(rw).ptr(sc).ptr(self.tok(&c.scores_v, ROUTE_ROWS)).ptr(wd).ptr(c.count.va).u32(value).u32(n as u32);
            c.k.launch(&c.hc_route_ahead, PRE_GRID, g::WG, 64, &b);
        }
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
    fn hc_post(&self, c: &Card, copy: Option<(usize, usize, usize)>, mine: Option<(usize, usize)>, w0: Option<(usize, usize)>, w1: Option<(usize, usize)>, value: u32, ya: usize, yb: Option<usize>, tokens: usize, routed: Option<(usize, usize, usize, usize)>, pred: Option<(usize, usize, usize, usize, usize)>) {
        let me = c.k.index;
        let own = |w: Option<(usize, usize)>| w.map_or(0, |(k, d)| self.peers[me].words.at(self.wd(k, d)));
        let peer = (0..self.cards.len()).find(|&j| j != me);
        let mine2 = match (mine, peer) {
            (Some((k, d)), Some(j)) => self.peers[j].words.at(self.wd(k, d)),
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
        a.ptr(self.tok(&c.mix, HC_MIX)).ptr(ya).ptr(yb.unwrap_or(0)).ptr(self.tok(&c.streams, HC * H)).ptr(src).ptr(0usize).ptr(dst2).u32(n as u32).ptr(own(mine)).ptr(mine2).ptr(own(w0)).ptr(own(w1)).u32(value).ptr(c.go.va).u32(tokens as u32);
        let (rows, wts, fwd, fwd_word) = routed.unwrap_or((0, 0, 0, 0));
        a.ptr(rows).ptr(wts).ptr(fwd).ptr(fwd_word);
        // the predictor's rows pass behind this launch: its input written here (the block input in; the f32 input, the
        // next layer's norm weights, the normed bf16 input and the partial sums of squares out)
        let (x, px, pln, pxn, pss) = pred.unwrap_or((0, 0, 0, 0, 0));
        a.ptr(x).ptr(px).ptr(pln).ptr(pxn).ptr(pss);
        c.k.launch(&c.hc_post, CHANNEL_GRID, g::WG, 0, &a);
    }
    /// The FFN router's place: on the FFN card of a MoE layer, `hc_pre` runs layer `l`'s FFN router (and the
    /// next layer's, the prefetch's predictor) over the half's collapsed input (normed with the FFN half's
    /// norm) and hands the logits to the host with the half's word.
    fn ffn_router(&self, l: usize, c: &Card) -> Option<(usize, usize, usize, usize)> {
        let f = self.ffns[l].as_ref().unwrap();
        f.router.as_ref().filter(|_| c.k.index == f.card).map(|r| (f.ln.va, r.va, self.tok(&self.shared.scores_ffn, ROUTE_ROWS), self.shared.words.at(self.wd(W_X, f.card))))
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
                    (c.yv.at((p0 * STEP + self.t0) * H * 4), Some(self.tok(&self.peers[me].ys[parity], H)))
                }
                (Some(&p0), None) => (c.yv.at((p0 * STEP + self.t0) * H * 4), (mine.len() > 1).then(|| c.yv.at(((p0 + 1) * STEP + self.t0) * H * 4))),
                (None, _) => (self.tok(&self.peers[me].ys[parity], H), None),
            };
            let copy = match (mine.first(), peer) {
                (Some(&p0), Some(j)) => Some((c.yv.at((p0 * STEP + self.t0) * H * 4), self.tok(&self.peers[j].ys[parity], H), n * H)),
                _ => None,
            };
            self.hc_post(c, copy, (!mine.is_empty()).then_some((W_Y, me)), Some((W_Y, producers[0])), producers.get(1).map(|&p| (W_Y, p)), value, ya, yb, n, None, None);
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
    fn mla_rows(&self, m: &MlaOn, n: usize, value: u32) {
        let c = &self.cards[m.card];
        let ln = Some(m.input_ln.va);
        let (x, mqa, mq, miq) = (self.tok(&c.x, H), self.tok(&c.mqa, Q_LORA_RANK), self.tok(&c.mq, MLA_HEADS * Q_HEAD), self.tok(&c.miq, INDEX_N_HEADS * INDEX_HEAD_DIM));
        let (kva, ix) = (c.mproj.at((MKVA + self.t0 * KV_LORA_RANK) * 4), c.mproj.at((MIX + self.t0 * IX_ROWS) * 4));
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.ffn_gu_w, x, ln, m.q_a.0.va, Some(m.q_a.1.va), mqa, Q_LORA_RANK, n, false);
        } else {
            self.rows8(c, &c.ffn_gu, x, ln, m.q_a.0.va, m.q_a.1.va, mqa, Q_LORA_RANK, H, n, ROWS_GU);
        }
        // q_a's output to the other card, which runs q_b and its half of the attend (058), then its word there
        self.copy_words(c, mqa, self.tok(&self.peers[m.other].pmqa, Q_LORA_RANK), n * Q_LORA_RANK);
        self.post_word(c, self.peers[m.other].words.at(self.wd(W_MQA, m.card)), value);
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.ffn_gu_w, x, ln, m.kv_a.0.va, Some(m.kv_a.1.va), kva, KV_LORA_RANK, n, true);
        } else {
            self.rows8_free(c, &c.ffn_gu, x, ln, m.kv_a.0.va, m.kv_a.1.va, kva, KV_LORA_RANK, H, n, ROWS_GU);
        }
        if n > SPEC_TOKENS {
            let mut a = Args::new();
            a.ptr(x).ptr(m.input_ln.va).ptr(m.ix.va).ptr(ix).u32(IX_ROWS as u32).u32(n as u32);
            c.k.launch_free(&c.rows_h_w, g::wmma_grid(IX_ROWS, n), g::WG, g::WMMA_LDS, &a);
        } else {
            self.rows_free(c, &c.rows_h, None, x, ln, m.ix.va, ix, IX_ROWS, H, n, ROWS);
        }
        // over q_a's output normed with q_a's norm: the query and, alongside it, the indexer's query (VRAM: the card scores the pools)
        let qln = Some(m.qa_ln.va);
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_qb_w, mqa, qln, m.q_b.0.va, Some(m.q_b.1.va), mq, MLA_HEADS * Q_HEAD, n, false);
            self.rows_w(c, &c.mla_iq_w, mqa, qln, m.ix_wq_b.va, None, miq, INDEX_N_HEADS * INDEX_HEAD_DIM, n, true);
        } else {
            self.rows8(c, &c.mla_qb, mqa, qln, m.q_b.0.va, m.q_b.1.va, mq, MLA_HEADS * Q_HEAD, Q_LORA_RANK, n, ROWS_QB);
            self.rows_free(c, &c.mla_iq, None, mqa, qln, m.ix_wq_b.va, miq, INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK, n, ROWS);
        }
        c.k.fence();
        // behind the fence the host waits on, while it finishes the rows: q_abs here, and the other card's q_a in, its
        // q_b and q_abs (058) — none of them wait for the host's rows, only the attends do
        self.qabs(c, mq, m.kv_b.va, n);
        let o = &self.cards[m.other];
        self.hold_word(o, W_MQA, m.card, value);
        let (omq, omqa) = (self.tok(&o.mq, MLA_HEADS * Q_HEAD), self.tok(&self.peers[m.other].pmqa, Q_LORA_RANK));
        if n > SPEC_TOKENS {
            self.rows_w(o, &o.mla_qb_w, omqa, Some(m.qa_ln2.va), m.q_b2.0.va, Some(m.q_b2.1.va), omq, MLA_HEADS * Q_HEAD, n, false);
        } else {
            self.rows8(o, &o.mla_qb, omqa, Some(m.qa_ln2.va), m.q_b2.0.va, m.q_b2.1.va, omq, MLA_HEADS * Q_HEAD, Q_LORA_RANK, n, ROWS_QB);
        }
        self.qabs(o, omq, m.kv_b2.va, n);
    }
    /// q_abs over the query `mq` with `kvb`'s W_uk on card `c`, the step's `n` tokens, into its `mqabs`.
    fn qabs(&self, c: &Card, mq: usize, kvb: usize, n: usize) {
        let mut a = Args::new();
        a.ptr(mq).ptr(kvb).ptr(self.tokb(&c.mqabs, MLA_HEADS * KV_LORA_RANK * 2));
        if n > SPEC_TOKENS {
            a.u32(n as u32);
            c.k.launch(&c.mla_qabs_w, MLA_HEADS as u32 * g::wmma_grid(KV_LORA_RANK, n), g::WG, g::WMMA_LDS, &a);
        } else {
            c.k.launch(&c.mla_qabs, (n * MLA_HEADS) as u32, 256, 0, &a);
        }
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
    /// One card's half of a split MLA attend (058) on card `c` over the step's `n` tokens: the attend over its rows
    /// (`sel`, `seln`) of its half mirror `lat` with its `q_abs` in raw mode (a verify step's chunks merged raw), W_uv
    /// over the unnormalized mix into `hd` (`[n][O_IN]`) with the (max, sum) into `ml` (the MLA card's VRAM, or this card's).
    #[allow(clippy::too_many_arguments)]
    fn attend_half(&self, c: &Card, kvb: usize, lat: &Mem, sel: usize, seln: usize, hd: usize, ml: usize, n: usize) {
        let (mqabs, molat) = (self.tokb(&c.mqabs, MLA_HEADS * KV_LORA_RANK * 2), self.tok(&c.molat, MLA_HEADS * KV_LORA_RANK));
        let (chunk, chunks) = (aw_chunk(n), aw_chunks(n));
        let mut a = Args::new();
        a.ptr(mqabs).ptr(lat.va).ptr(sel).ptr(seln).ptr(molat).ptr(c.mpart.va).ptr(ml).u32(chunks as u32).u32(chunk as u32);
        c.k.launch(&c.mla_attend_w, aw_grid(n), AW_WG, AW_LDS as u32, &a);
        if chunks > 1 {
            let mut a = Args::new();
            a.ptr(c.mpart.va).ptr(molat).ptr(ml);
            c.k.launch(&c.mla_attend_merge, (n * AW_HEAD_TILES) as u32, AW_WG, AW_MERGE_LDS as u32, &a);
        }
        let mut a = Args::new();
        a.ptr(molat).ptr(kvb).ptr(hd);
        if n > SPEC_TOKENS {
            a.u32(n as u32);
            c.k.launch(&c.mla_uv_w, MLA_HEADS as u32 * g::wmma_grid(V_HEAD_DIM, n), g::WG, g::WMMA_LDS, &a);
        } else {
            c.k.launch(&c.mla_uv, (n * MLA_HEADS) as u32, 256, (KV_LORA_RANK * 4) as u32, &a);
        }
    }
    /// `n` floats from `src` to `dst` on card `c` (the copy kernel; `dst` may be a peer's VRAM).
    fn copy_words(&self, c: &Card, src: usize, dst: usize, n: usize) {
        let mut a = Args::new();
        a.ptr(src).ptr(dst).u32(n as u32);
        c.k.launch(&c.copy, n.div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &a);
    }
    /// `value` into the word at `w` once card `c`'s launches before it are done (058's handoffs between the cards).
    fn post_word(&self, c: &Card, w: usize, value: u32) {
        let mut a = Args::new();
        a.ptr(w).u32(value);
        c.k.launch(&c.word_post, 1, 32, 0, &a);
    }
    /// Card `c`'s queue held until its own copy of word `kind` (from card `from`) reads `value`.
    fn hold_word(&self, c: &Card, kind: usize, from: usize, value: u32) {
        let mut a = Args::new();
        a.ptr(self.peers[c.k.index].words.at(self.wd(kind, from))).u32(value);
        c.k.launch(&c.word_wait, 1, 32, 0, &a);
    }
    /// A step's (or a seed's) latents `rows` (`[n][KV_LORA_RANK]` bf16, positions `pos..`) into their owners' half
    /// mirrors (058): runs of `LAT_GRAIN` positions alternate between the cards.
    fn lat_upload(&self, m: &MlaOn, pos: usize, rows: &[u16], n: usize) {
        let mut i = 0;
        while i < n {
            let p = pos + i;
            let run = (LAT_GRAIN - p % LAT_GRAIN).min(n - i);
            let o = lat_owner(p);
            self.cards[o].k.upload(&m.lats[o], lat_local(p) * KV_LORA_RANK * 2, raw(&rows[i * KV_LORA_RANK..(i + run) * KV_LORA_RANK]));
            i += run;
        }
    }
    /// bf16 rows on card `c` over the step's `n` tokens (as `rows8`). Addresses are device VAs.
    #[allow(clippy::too_many_arguments)]
    fn rows(&self, c: &Card, kernel: &crate::kfd::Kernel, narrow: Option<&crate::kfd::Kernel>, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows_at(c, kernel, narrow, x, ln, w, out, m, k_width, n, r, true)
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
    /// The exp4 `[m][k]` matrix at `w` unpacked to bf16 at `out` (029), after the launches before it.
    fn exp4_unpack(&self, c: &Card, w: usize, out: usize, m: usize, k: usize) {
        let mut a = Args::new();
        a.ptr(w).ptr(out).u32(m as u32).u32(k as u32);
        c.k.launch(&c.exp4_unpack, (m * k / 8).div_ceil(256) as u32, 256, 0, &a);
    }
    /// `rows` launched free: as soon as the launch before it has launched, so the two run together —
    /// for a launch whose inputs were complete when the one before it launched.
    #[allow(clippy::too_many_arguments)]
    fn rows_free(&self, c: &Card, kernel: &crate::kfd::Kernel, narrow: Option<&crate::kfd::Kernel>, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize) {
        self.rows_at(c, kernel, narrow, x, ln, w, out, m, k_width, n, r, false)
    }
    #[allow(clippy::too_many_arguments)]
    fn rows_at(&self, c: &Card, kernel: &crate::kfd::Kernel, narrow: Option<&crate::kfd::Kernel>, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize, n: usize, r: usize, barrier: bool) {
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            // 060: past one token, a launch whose wide grid is under ~2 workgroups a CU keeps one row a
            // wave: `ROWS` 4 makes m H a 128-workgroup launch on 84 CUs, and the waves, not the rows,
            // stream the weights (bench-rows 2026-10-24: m 256/1024 at n 2–4 run 24–27 µs against 9–11
            // at one token, even cache-resident). The rows' arithmetic is unchanged, only their
            // grouping — the outputs are the same bits.
            let (kernel, r) = match narrow {
                // narrow below one workgroup a CU: bench-rows 2026-10-24 (µs at n 4, wide / narrow) m 1024 27.0 / 19.7,
                // 1536 34.9 / 22.4, 2048 35.2 / 33.1 (64 workgroups), 3072 34.1 / 37.6 (96), 4096 36.8 / 50.1 (128):
                // past ~a workgroup a CU the one-row waves' LDS reads cost more than the occupancy buys
                Some(k1) if np > 1 && m.div_ceil(common::gemv::rows_per_wg(np, r)) < c.k.cu => (k1, 1),
                _ => (kernel, r),
            };
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
        // a new prompt after decode steps (a request's prefill, or a scoring text): the counts age hard,
        // so the residency retargets to the current text within the prompt's own steps (063's aging)
        if prefill && !r.prefill && r.steps > 1 {
            for c in r.count.iter_mut() {
                *c >>= RESIDENT_PROMPT_AGE;
            }
            for (ml, w) in r.window.iter_mut().enumerate() {
                r.window_seen[ml] = w.iter().map(|x| x.count_ones()).sum();
                *w = [0; (glm_5_3_flash::E + 63) / 64];
            }
        }
        r.prefill = prefill;
        r.turn = (r.steps % r.held[0].len() as u64) as usize;
        if r.steps % RESIDENT_HALF_LIFE == 0 {
            for (ml, w) in r.window.iter_mut().enumerate() {
                r.window_seen[ml] = w.iter().map(|x| x.count_ones()).sum();
                *w = [0; (glm_5_3_flash::E + 63) / 64];
            }
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
        (self.ch, self.t0, self.chunks) = (0, 0, glm_5_3_flash::chunk_count(n));
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
    fn chunk(&mut self, c: usize, t0: usize) {
        (self.ch, self.t0) = (c, t0);
    }
    fn attention(&mut self, l: usize, n: usize) {
        let value = self.next_seq();
        // a step of a verify's size keeps a state a token (`kda_rewind`); one token and a prompt's step do not
        let ring = (2..=SPEC_TOKENS).contains(&n);
        self.half_seq[l][self.ch][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], value, None, None, n);
        }
        let lay = self.layers[l].as_ref().unwrap();
        let parity = l % 2;
        const M_SMALL: usize = SMALL_ROWS + 3 * g::PART_QKV;
        for (i, p) in lay.parts.iter().enumerate() {
            let c = &self.cards[p.card];
            let (smix, fg, normed) = (&c.smixs[i], &c.fgs[i], &c.normeds[i]);
            // the gates' first halves and beta, then q | k | v: one launch over the normed block input, [n][M_SMALL];
            // a step of many tokens takes the matrix path (bf16 activations, numerically its own)
            // the matrices are exp4 (029): a decode or verify step's kernels read it; a prefill step unpacks each matrix
            // to the card's bf16 scratch once and runs the bf16 kernels on it — the same bits either way
            if n > SPEC_TOKENS {
                self.exp4_unpack(c, p.sq.va, c.kda_bf.va, M_SMALL, H);
                let mut a = Args::new();
                a.ptr(self.tok(&c.x, H)).ptr(p.input_ln.va).ptr(c.kda_bf.va).ptr(self.tok(smix, M_SMALL)).u32(M_SMALL as u32).u32(n as u32);
                c.k.launch(&c.rows_h_w, g::wmma_grid(M_SMALL, n), g::WG, g::WMMA_LDS, &a);
                self.exp4_unpack(c, p.fg.va, c.kda_bf_fg.va, 2 * g::PART_QKV, KDA_DIM);
            } else {
                self.rows(c, &c.rows_h_e4, None, self.tok(&c.x, H), Some(p.input_ln.va), p.sq.va, self.tok(smix, M_SMALL), M_SMALL, H, n, g::E4_ROWS);
            }
            // the gates' second halves as one launch: f_b over f_a's output, g_b over g_a's — token i's input at smix[i · M_SMALL]
            let (fg_kernel, fg_w, fg_r) = if n > SPEC_TOKENS { (&c.rows_d2, c.kda_bf_fg.va, ROWS) } else { (&c.rows_d2_e4, p.fg.va, g::E4_ROWS) };
            let mut p0 = 0;
            while p0 < n {
                let np = (n - p0).min(TOKENS);
                let mut a = Args::new();
                a.ptr(smix.at((self.t0 + p0) * M_SMALL * 4)).ptr(fg_w).ptr(fg.at((self.t0 + p0) * 2 * g::PART_QKV * 4)).u32(2 * g::PART_QKV as u32).u32(np as u32).u32(g::PART_QKV as u32).u32(M_SMALL as u32);
                c.k.launch(fg_kernel, g::rows_grid(2 * g::PART_QKV, np, fg_r), g::WG, g::rows_lds(KDA_DIM, np) as u32, &a);
                p0 += np;
            }
            // the delta rule over the step's tokens in order, the heads' memory in registers across them
            let (smix0, fg0) = (self.tok(smix, M_SMALL), self.tok(fg, 2 * g::PART_QKV));
            let mixed = smix0 + SMALL_ROWS * 4;
            let mut a = Args::new();
            a.ptr(mixed).ptr(p.taps.va).ptr(p.window.va).ptr(fg0).ptr(p.dt_bias.va).ptr(p.a_log.va).ptr(smix0 + 2 * KDA_DIM * 4).ptr(fg0 + g::PART_QKV * 4).ptr(p.mem.va).ptr(p.o_norm.va).ptr(self.tok(normed, g::PART_QKV)).u32(p.head0 as u32).u32(n as u32).u32(M_SMALL as u32).u32(2 * g::PART_QKV as u32).u32(M_SMALL as u32).ptr(if ring { p.ring.va } else { 0 }).ptr(if ring { p.ring_w.va } else { 0 });
            c.k.launch(&c.step, g::STEP_GRID, g::WG, g::STEP_LDS as u32, &a);
            if n > SPEC_TOKENS {
                // `sq`'s unpack is consumed by now (the queue is in order and this launch waits on the ones before it)
                self.exp4_unpack(c, p.o.va, c.kda_bf.va, H, g::PART_QKV);
                let mut a = Args::new();
                a.ptr(self.tok(normed, g::PART_QKV)).ptr(c.kda_bf.va).ptr(c.yv.at((i * STEP + self.t0) * H * 4)).u32(H as u32).u32(n as u32);
                c.k.launch(&c.rows_part_w, g::wmma_grid(H, n), g::WG, g::WMMA_LDS, &a);
            } else {
                self.rows(c, &c.rows_part_e4, None, self.tok(normed, g::PART_QKV), None, p.o.va, c.yv.at((i * STEP + self.t0) * H * 4), H, g::PART_QKV, n, g::E4_ROWS);
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
        self.half_seq[l][self.ch][0] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            self.hc_pre(c, &f.hc[c.k.index][0], value, None, None, n);
        }
        self.mla_value = value;
        let m = self.mlas[l].as_ref().unwrap();
        self.mla_rows(m, n, value);
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
        cfg!(feature = "bringup-checks")
    }
    fn resident_stats(&self) -> glm_5_3_flash::construct::ResidentStats {
        use glm_5_3_flash::construct::ResidentStats;
        let r = &self.resident;
        let total: usize = r.owner.iter().map(|o| o.len()).sum();
        let free: usize = r.free.iter().map(|f| f.len()).sum();
        let moe = glm_5_3_flash::layout::MOE_LAYERS;
        let mut held: Vec<u32> = (0..moe).map(|ml| r.held.iter().map(|h| h[ml]).sum()).collect();
        held.sort_unstable();
        let mut seen = r.window_seen.clone();
        seen.sort_unstable();
        let mid = held.len() / 2;
        ResidentStats {
            filled: total - free,
            slots: total,
            loads: r.uploads,
            per_layer: r.per_layer.iter().sum(),
            held_min: held.first().copied().unwrap_or(0),
            held_med: held.get(mid).copied().unwrap_or(0),
            held_max: held.last().copied().unwrap_or(0),
            seen_min: seen.first().copied().unwrap_or(0),
            seen_med: seen.get(mid).copied().unwrap_or(0),
            seen_max: seen.last().copied().unwrap_or(0),
        }
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
    fn xpt_dispatch(&mut self, l: usize, n: usize, picks: &mut [[usize; TOP_K]], marked: &mut dyn FnMut(&[[usize; TOP_K]])) -> usize {
        use std::sync::atomic::Ordering;
        prow::zone!("track-zones", "dispatch");
        let ml = l - glm_5_3_flash::FIRST_K_DENSE_REPLACE;
        let e_n = glm_5_3_flash::E;
        let prefill = n > SPEC_TOKENS;
        if ml == 0 && self.ch == 0 {
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
                r.window[ml][e / 64] |= 1 << (e % 64);
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
        marked(picks);
        // the launches: a step of a few tokens runs a token a job (`xpt_gu`, `xpt_down` over `[jobs][3]`: slot,
        // token, output row); a prefill step's jobs grouped by slot — a group table (slot, first job, jobs, 0)
        // and the jobs (token, output row) — through gate/up over every group, the hidden states quantized,
        // down into the shared output rows. The same split as the rows kernels', at SPEC_TOKENS.
        // the chunk's rows of the activations and of the outputs: a job's token is the chunk's (037)
        let fc = self.ffns[l].as_ref().unwrap().card;
        let (t0, xout) = (self.t0, self.peers[fc].rows.at(self.t0 * TOP_K * H * 4));
        // the activations where each card's jobs read them: the FFN card's own (its act phase wrote them), the other card's
        // receiving buffers, filled card to card with a word (the host never touches them)
        let value = self.half_seq[l][self.ch][1];
        let acts: Vec<(usize, usize, usize, usize)> = (0..self.cards.len())
            .map(|c| {
                if c == fc {
                    let k = &self.cards[c];
                    (k.xq.at(t0 * H), k.xs.at(t0 * HG * 4), k.xg.at(t0 * HG * 4), 0)
                } else {
                    let p = &self.peers[c];
                    (p.axq.at(t0 * H), p.axs.at(t0 * HG * 4), p.axg.at(t0 * HG * 4), p.words.at(self.wd(W_XQ, fc)))
                }
            })
            .collect();
        let r = &mut self.resident;
        let mut total = 0;
        for c in 0..self.cards.len() {
            let jobs = &mut r.jobs[c];
            if jobs.is_empty() {
                continue;
            }
            total += jobs.len();
            let card = &self.cards[c];
            // the FFN card quantized the step's activations into its own buffers (`hc_pre`'s act phase); the other card's come
            // over from it — three copies and the word on the FFN card's queue, the wait on this one's
            let (aq, as_, ag, aw) = acts[c];
            if c != fc {
                let f = &self.cards[fc];
                for (src, dst, words) in [(f.xq.at(t0 * H), aq, n * H / 4), (f.xs.at(t0 * HG * 4), as_, n * HG), (f.xg.at(t0 * HG * 4), ag, n * HG)] {
                    let mut a = Args::new();
                    a.ptr(src).ptr(dst).u32(words as u32);
                    f.k.launch(&f.copy, words.div_ceil(g::COPY_WG_WORDS) as u32, g::WG, 0, &a);
                }
                let mut a = Args::new();
                a.ptr(aw).u32(value);
                f.k.launch(&f.word_post, 1, 32, 0, &a);
                let mut a = Args::new();
                a.ptr(aw).u32(value);
                card.k.launch(&card.word_wait, 1, 32, 0, &a);
            }
            if !prefill {
                let list: Vec<u32> = jobs.iter().flat_map(|&(sl, tok, k)| [sl, tok, tok * TOP_K as u32 + k as u32]).collect();
                card.k.upload(&card.xjobs, 0, raw(&list));
                let mut a = Args::new();
                a.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(aq).ptr(as_).ptr(ag).ptr(card.xgu.va);
                card.k.launch(&card.xpt_gu, (jobs.len() * XPT_UP_WGS) as u32, g::WG, XPT_LDS as u32, &a);
                let mut b = Args::new();
                b.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(card.xgu.va).ptr(xout);
                card.k.launch(&card.xpt_down, (jobs.len() * XPT_DOWN_WGS) as u32, g::WG, XPT_LDS as u32, &b);
                card.k.fence_line(XPT_LINE);
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
            a.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(jobs_va).ptr(aq).ptr(as_).ptr(ag).ptr(card.xgu.va);
            card.k.launch(&card.xpt_gemm_gu, groups * XG_GU_WGS as u32, g::WG, 0, &a);
            let mut q = Args::new();
            q.ptr(card.xgu.va).ptr(card.xhq.va).ptr(card.xhs.va).ptr(card.xhg.va);
            card.k.launch(&card.xpt_quant, (jobs.len() * XQ_WGS) as u32, g::WG, 0, &q);
            let mut b = Args::new();
            b.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(jobs_va).ptr(card.xhq.va).ptr(card.xhs.va).ptr(card.xhg.va).ptr(xout);
            card.k.launch(&card.xpt_gemm_down, groups * XG_DOWN_WGS as u32, g::WG, 0, &b);
            card.k.fence_line(XPT_LINE);
        }
        total
    }
    fn xpt_join(&mut self, l: usize, _out: &mut [f32]) {
        // the jobs' rows are in the shared output already; the wait is all
        for c in 0..self.cards.len() {
            if !self.resident.jobs[c].is_empty() {
                self.cards[c].k.wait_line(XPT_LINE);
            }
        }
        // a prefill step's claims, now that no job of the layer reads a slot (063)
        if self.resident.prefill && self.ch + 1 == self.chunks {
            let none = [false; glm_5_3_flash::E];
            self.claim_prefill(l - glm_5_3_flash::FIRST_K_DENSE_REPLACE, &none);
        }
    }
    fn expert_out(&self, l: usize) -> *mut f32 {
        self.peers[self.ffns[l].as_ref().unwrap().card].rows.ptr() as *mut f32
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
            self.rows(c, &c.mtp_eh, None, t.dx.va, None, t.eh.va, c.x.va, H, 2 * H, n, ROWS);
        }
        let value = self.next_seq();
        self.mla_value = value;
        let m = self.mlas[glm_5_3_flash::layout::MTP_LAYER].as_ref().unwrap();
        self.mla_rows(m, n, value);
    }
    fn mtp_attn_out(&mut self, hidden: &mut [f32], n: usize) {
        let t = self.mtp.as_ref().unwrap();
        let c = &self.cards[t.card];
        let m = self.mlas[glm_5_3_flash::layout::MTP_LAYER].as_ref().unwrap();
        // the output projection, the residual, the block's states out to the host for its routing; a prefill step's
        // positions on the matrix path as the main layers' (the decode kernel in 32-token passes was ~15 ms of a
        // 576-position draft, 2026-10-20)
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_o_w, c.mheads.va, None, m.o.0.va, Some(m.o.1.va), c.myv.va, H, n, false);
        } else {
            self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.myv.va, H, O_IN, n, ROWS);
        }
        self.add2(c, c.x.va, c.myv.va, 0, n);
        self.copy_out(c, c.x.va, self.shared.xh.va, n);
        c.k.sync();
        read(&self.shared.xh, 0, &mut hidden[..n * H]);
    }
    fn mtp_routed(&self) -> *mut f32 {
        self.shared.yexp[0].ptr() as *mut f32
    }
    fn mtp_ffn(&mut self, hidden: &mut [f32], n: usize) {
        let t = self.mtp.as_ref().unwrap();
        let c = &self.cards[t.card];
        // the shared expert over the post-normed states, the host's routed experts, the residual
        let yexp = &self.shared.yexp[0];
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.ffn_gu_w, c.x.va, Some(t.post_ln.va), t.gu_w.va, Some(t.gu_s.va), c.gu.va, SHARED_GU_ROWS, n, false);
            self.rows_w(c, &c.ffn_down_shared_w, c.gu.va, None, t.down_w.va, Some(t.down_s.va), c.fyv.va, H, n, false);
        } else {
            self.rows8(c, &c.ffn_gu, c.x.va, Some(t.post_ln.va), t.gu_w.va, t.gu_s.va, c.gu.va, SHARED_GU_ROWS, H, n, ROWS_GU);
            self.rows8_at(c, &c.ffn_down_shared, c.gu.va, 2 * N_SHARED_EXPERTS * D_FF, None, t.down_w.va, t.down_s.va, c.fyv.va, H, N_SHARED_EXPERTS * D_FF, n, ROWS, true);
        }
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
            self.rows(c, &c.rows_h, None, x, Some(t.head_norm[ci].va), lm.va, c.lg.va, rows, H, 1, ROWS);
        }
        for (ci, c) in self.cards.iter().enumerate() {
            let (r0, rows) = head_rows(ci, self.cards.len());
            c.k.sync();
            read(&c.lg, 0, &mut logits[r0..r0 + rows]);
        }
        // the draft's launches on the card lanes (the step's report closed before the draft ran)
        self.timeline_report();
    }
    fn mla_proj(&mut self, l: usize) -> glm_5_3_flash::construct::MlaProj {
        let c = &self.cards[self.mlas[l].as_ref().unwrap().card];
        c.k.wait();
        let base = c.mproj.ptr() as *const f32;
        unsafe { glm_5_3_flash::construct::MlaProj { kva: base.add(MKVA + self.t0 * KV_LORA_RANK), ix: base.add(MIX + self.t0 * IX_ROWS), ix_rows: IX_ROWS } }
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
            a.ptr(self.tok(&c.miq, INDEX_N_HEADS * INDEX_HEAD_DIM)).ptr(c.mproj.at((MIX + self.t0 * IX_ROWS) * 4)).ptr(m.pools.va).ptr(self.tok(&c.mscore, stride)).u32(n as u32).u32(visible as u32).u32(stride as u32).f32(scale).f32(wscale);
            c.k.launch(&c.ix_score_w, ixw_grid(n, visible), 256, IXW_LDS as u32, &a);
        }
        // the selection across slices of the pools at a verify step past 32k positions (docs/queue/019); a
        // few-token step's own fixed-order merge measured slower than these slices (060, 2026-10-24)
        let slices = select_slices(n, (pos + n) / INDEX_KPOOL);
        let mut a = Args::new();
        a.ptr(self.tok(&c.mscore, stride)).u32(stride as u32).u32(pos as u32).u32(slices as u32).ptr(self.tok(&c.mselst, SELECT_STATE)).ptr(self.tok(&c.msel, DENSE_BELOW)).ptr(self.tok(&c.mseln, 1));
        c.k.launch(&c.ix_select, (n * slices) as u32, g::WG, SELECT_LDS as u32, &a);
        if let Some(count) = count {
            c.k.sync();
            let mut w = [0u32; 1];
            read(&c.mseln, self.t0 * 4, &mut w);
            *count = w[0] as usize;
        }
    }
    fn mla_attend(&mut self, l: usize, n: usize, pos: usize, latents: &[u16]) {
        let m = self.mlas[l].as_ref().unwrap();
        let value = self.mla_value;
        // the step's latents into their owners' half mirrors; the selections are the MLA card's (`ix_select`), split by
        // owner into each card's rows (058). Each card: q_abs for every head, the attend over its rows in raw mode (the
        // unnormalized mix and the heads' (max, sum)) — a verify step's token in chunks of positions merged after — and
        // W_uv over the step; the other card's go straight into the MLA card's VRAM with their word, and the MLA card
        // merges the two. The few-token twins of q_abs and W_uv below SPEC_TOKENS, like the rows kernels'
        self.lat_upload(m, pos, latents, n);
        let (mc, oc) = (&self.cards[m.card], &self.cards[m.other]);
        {
            let c = mc;
            let mut a = Args::new();
            a.ptr(self.tok(&c.msel, DENSE_BELOW)).ptr(self.tok(&c.mseln, 1)).ptr(self.tok(&c.mown, DENSE_BELOW)).ptr(self.tok(&c.mown_n, 1));
            a.ptr(self.tok(&self.peers[m.other].psel, DENSE_BELOW)).ptr(self.tok(&self.peers[m.other].psel_n, 1)).u32(m.card as u32);
            a.ptr(self.peers[m.other].words.at(self.wd(W_SEL, m.card))).ptr(c.mdone.va).u32(value).u32(n as u32);
            c.k.launch(&c.mla_sel_split, n as u32, g::WG, 64, &a);
        }
        // the other card (its q_abs is posted with the projections): its rows in, its half — W_uv into its own VRAM, then
        // one copy across to the MLA card (W_uv's scattered stores straight over the link ran 79 µs against 36), its word
        self.hold_word(oc, W_SEL, m.card, value);
        self.attend_half(oc, m.kv_b2.va, &m.lats[m.other], self.tok(&self.peers[m.other].psel, DENSE_BELOW), self.tok(&self.peers[m.other].psel_n, 1), self.tok(&oc.mhd, O_IN), self.tok(&self.peers[m.card].pml, MLA_HEADS * 2), n);
        self.copy_words(oc, self.tok(&oc.mhd, O_IN), self.tok(&self.peers[m.card].phd, O_IN), n * O_IN);
        self.post_word(oc, self.peers[m.card].words.at(self.wd(W_HD, m.other)), value);
        // the MLA card's half, then the other card's in, the merge into the heads
        self.attend_half(mc, m.kv_b.va, &m.lats[m.card], self.tok(&mc.mown, DENSE_BELOW), self.tok(&mc.mown_n, 1), self.tok(&mc.mhd, O_IN), self.tok(&mc.mml, MLA_HEADS * 2), n);
        self.hold_word(mc, W_HD, m.other, value);
        let mut a = Args::new();
        a.ptr(self.tok(&mc.mhd, O_IN)).ptr(self.tok(&mc.mml, MLA_HEADS * 2)).ptr(self.tok(&self.peers[m.card].phd, O_IN)).ptr(self.tok(&self.peers[m.card].pml, MLA_HEADS * 2)).ptr(self.tok(&mc.mheads, O_IN));
        mc.k.launch(&mc.mla_two_merge, (n * MLA_HEADS) as u32, V_HEAD_DIM as u32, 0, &a);
    }
    fn mla_out(&mut self, l: usize, n: usize) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        let (mheads, myv) = (self.tok(&c.mheads, O_IN), self.tok(&c.myv, H));
        // the output projection over every token's heads, then the streams (the projection out to the peer's receiving buffer)
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_o_w, mheads, None, m.o.0.va, Some(m.o.1.va), myv, H, n, false);
        } else {
            self.rows8(c, &c.mla_o, mheads, None, m.o.0.va, m.o.1.va, myv, H, O_IN, n, ROWS);
        }
        let copy = (0..self.cards.len()).find(|&j| j != m.card).map(|j| (myv, self.tok(&self.peers[j].my[l % 2], H), n * H));
        let value = self.half_seq[l][self.ch][0];
        self.hc_post(c, copy, Some((W_Y, m.card)), None, None, value, myv, None, n, None, None);
        // the other card folds the output in once it is there (posted after its half of the attend, which its queue runs first)
        let o = &self.cards[m.other];
        self.hc_post(o, None, None, Some((W_Y, m.card)), None, value, self.tok(&self.peers[m.other].my[l % 2], H), None, n, None, None);
    }
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[u16]) {
        let m = self.mlas[l].as_ref().unwrap();
        self.lat_upload(m, pos, latent, 1);
    }
    fn ffn(&mut self, l: usize, n: usize) {
        let value = self.next_seq();
        self.half_seq[l][self.ch][1] = value;
        let f = self.ffns[l].as_ref().unwrap();
        for c in &self.cards {
            // the FFN card's collapse goes to the host too when the routed experts need it, with the router's logits and the word
            let router = self.ffn_router(l, c);
            let act = f.act.as_ref().filter(|_| c.k.index == f.card).map(|(_, _, tab)| tab.va);
            self.hc_pre(c, &f.hc[c.k.index][1], value, router, act, n);
        }
        let c = &self.cards[f.card];
        let (rows, down, k_width) = if f.dense { (DENSE_GU_ROWS, &c.ffn_down_dense, INTERMEDIATE_SIZE) } else { (SHARED_GU_ROWS, &c.ffn_down_shared, N_SHARED_EXPERTS * D_FF) };
        let (x, gu, fyv) = (self.tok(&c.x, H), self.tok(&c.gu, rows), self.tok(&c.fyv, H));
        if n > SPEC_TOKENS {
            let down_w = if f.dense { &c.ffn_down_dense_w } else { &c.ffn_down_shared_w };
            self.rows_w(c, &c.ffn_gu_w, x, Some(f.ln.va), f.gu_w.va, Some(f.gu_s.va), gu, rows, n, false);
            self.rows_w(c, down_w, gu, None, f.down_w.va, Some(f.down_s.va), fyv, H, n, false);
        } else {
            self.rows8(c, &c.ffn_gu, x, Some(f.ln.va), f.gu_w.va, f.gu_s.va, gu, rows, H, n, ROWS_GU);
            self.rows8_at(c, down, gu, 2 * k_width, None, f.down_w.va, f.down_s.va, fyv, H, k_width, n, ROWS, true);
        }
        // a dense layer's output half now; a MoE layer's waits for the host's experts and is posted after the
        // resident experts' launches (`ffn_post`): a queue is in order, and the output half spins on the word
        if f.dense {
            self.ffn_post(l, n);
        }
    }
    fn ffn_post(&mut self, l: usize, n: usize) {
        let value = self.half_seq[l][self.ch][1];
        let f = self.ffns[l].as_ref().unwrap();
        let parity = l % 2;
        let peer = (0..self.cards.len()).find(|&j| j != f.card);
        for o in &self.cards {
            let me = o.k.index;
            let producer = me == f.card;
            // the producer's rows stay in its `fyv`; the peer reads them from its receiving buffer
            let copy = if producer { peer.map(|j| (self.tok(&o.fyv, H), self.tok(&self.peers[j].fy[parity], H), n * H)) } else { None };
            let fy = if producer { self.tok(&o.fyv, H) } else { self.tok(&self.peers[me].fy[parity], H) };
            let mine = producer.then_some((W_FFN, f.card));
            if f.dense {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), None, value, fy, None, n, None, None);
            } else if producer {
                // the leader (002): once the host's word says every expert row is in its VRAM, the routed sum of
                // the rows in pick order, folded in and forwarded to the peer with the peer's copy of the word
                let rows = self.peers[me].rows.at(self.t0 * TOP_K * H * 4);
                let fwd = peer.map(|j| (self.tok(&self.peers[j].yr[parity], H), self.peers[j].words.at(self.wd(W_EXPERTS, 0))));
                let (fwd, fwd_word) = fwd.unwrap_or((0, 0));
                // the leader writes the predictor's rows-pass input too (x + the whole FFN output) where it has every term
                let pred = f.pred.as_ref().map(|_| (self.tok(&o.x, H), self.tok(&o.pred_x, H), f.ln.va, self.tokb(&o.pred_xn, H * 2), self.tok(&o.pred_ss, glm_5_3_flash::plan::PRED_SS)));
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), Some((W_EXPERTS, 0)), value, fy, None, n, Some((rows, self.tok(&o.wv, TOP_K), fwd, fwd_word)), pred);
            } else {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), Some((W_EXPERTS, 0)), value, fy, Some(self.tok(&self.peers[me].yr[parity], H)), n, None, None);
            }
        }
    }
    fn plan_step(&mut self, l: usize, n: usize, picks: &[[usize; TOP_K]]) {
        // this layer's picks into its label slot: pair l − 1's labels, pair l's co-routing input
        let slot = |l: usize| self.shared.labels.at((l * STEP + self.t0) * TOP_K * 4);
        let lab = unsafe { std::slice::from_raw_parts_mut(slot(l) as *mut u32, n * TOP_K) };
        for (i, p) in picks[..n].iter().enumerate() {
            for k in 0..TOP_K {
                lab[i * TOP_K + k] = p[k] as u32;
            }
        }
        // the early plan: a decode step's, ahead of the output half, from the FFN input plus the shared output
        if n <= SPEC_TOKENS {
            self.pred_pass(l, n, false);
        }
    }
    fn plan_step_rows(&mut self, l: usize, n: usize) {
        // behind `hc_post`: the routed rows are in; the margins stored for the learning are this pass's
        self.pred_pass(l, n, true);
    }
    fn plan_learn(&mut self, n: usize) {
        use glm_5_3_flash::plan::{LEARN_LDS, LEARN_WG};
        for (card, (tab, pairs)) in self.pred_tab.iter().enumerate() {
            if *pairs > 0 {
                let c = &self.cards[card];
                let mut a = Args::new();
                a.ptr(tab.va).u32(0).u32(0).u32(n as u32);
                c.k.launch(&c.pred_learn, *pairs as u32, LEARN_WG, LEARN_LDS, &a);
            }
        }
    }
    fn plan(&mut self, l: usize, out: &mut [usize]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_PLAN, f.card), self.half_seq[l][self.ch][1]);
        let mut p = [0u32; glm_5_3_flash::plan::PLAN];
        read(&self.shared.plan, 0, &mut p);
        for (o, &e) in out.iter_mut().zip(&p) {
            *o = e as usize;
        }
    }
    fn plan2(&mut self, l: usize, out: &mut [usize]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_PLAN2, f.card), self.half_seq[l][self.ch][1]);
        let mut p = [0u32; glm_5_3_flash::plan::PLAN];
        read(&self.shared.plan2, 0, &mut p);
        for (o, &e) in out.iter_mut().zip(&p) {
            *o = e as usize;
        }
    }
    fn ffn_logits(&mut self, l: usize, n: usize, x: &mut [f32], logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_X, f.card), self.half_seq[l][self.ch][1]);
        // the FFN input as the card collapsed it, copied out on demand (bring-up only; serving never reads it)
        let c = &self.cards[f.card];
        self.copy_out(c, self.tok(&c.x, H), self.tok(&self.shared.xh, H), n);
        c.k.sync();
        read(&self.shared.xh, self.t0 * H * 4, &mut x[..n * H]);
        read(&self.shared.scores_ffn, self.t0 * ROUTE_ROWS * 4, &mut logits[..n * ROUTE_ROWS]);
    }
    fn ffn_parts(&mut self, l: usize, n: usize, shared: &mut [f32], rows: &mut [f32]) {
        // lab only: through the host's FFN-input buffer, a copy and a sync each (the output half is done by now)
        let f = self.ffns[l].as_ref().unwrap();
        let c = &self.cards[f.card];
        self.copy_out(c, self.tok(&c.fyv, H), self.shared.xh.va, n);
        c.k.sync();
        read(&self.shared.xh, 0, &mut shared[..n * H]);
        self.copy_out(c, self.peers[f.card].rows.at(self.t0 * TOP_K * H * 4), self.shared.xh.va, n * TOP_K);
        c.k.sync();
        read(&self.shared.xh, 0, &mut rows[..n * TOP_K * H]);
    }
    fn ffn_input(&mut self, l: usize, n: usize, xq: &mut [i8], xs: &mut [f32], xg: &mut [i32], picks: &mut [[usize; TOP_K]], wts: &mut [[f32; TOP_K]], pool: &dyn common::pool::Pool) {
        let f = self.ffns[l].as_ref().unwrap();
        {
            prow::zone!("track-zones", "act.wait");
            crate::kfd::wait_word(self.word_ptr(W_ACT, f.card), self.half_seq[l][self.ch][1]);
        }
        prow::zone!("track-zones", "act.read");
        let t0 = self.t0;
        let xa = &self.shared.xact;
        // the activations (~2 MB a prefill chunk) in 64 KB pieces over the pool, idle until they are in: one core streamed
        // them at ~4.5 GB/s, 0.49 ms a chunk on the critical path (2026-10-20); a verify step's are one piece, inline
        {
            const PIECE: usize = 64 << 10;
            let (src, dst, len) = (xa.at(XA_Q + t0 * H), xq.as_mut_ptr() as usize, n * H);
            pool.run(len.div_ceil(PIECE), &move |i| {
                let (o, e) = (i * PIECE, ((i + 1) * PIECE).min(len));
                let s = unsafe { std::slice::from_raw_parts((src + o) as *const u8, e - o) };
                let d = unsafe { std::slice::from_raw_parts_mut((dst + o) as *mut u8, e - o) };
                common::copy::stream_read(d, s);
            });
        }
        read(xa, XA_S + t0 * HG * 4, &mut xs[..n * HG]);
        read(xa, XA_G + t0 * HG * 4, &mut xg[..n * HG]);
        let mut p = vec![0u32; n * TOP_K];
        read(xa, XA_P + t0 * TOP_K * 4, &mut p);
        for (i, pk) in picks[..n].iter_mut().enumerate() {
            for k in 0..TOP_K {
                pk[k] = p[i * TOP_K + k] as usize;
            }
        }
        read(xa, XA_W + t0 * TOP_K * 4, unsafe { std::slice::from_raw_parts_mut(wts.as_mut_ptr() as *mut f32, n * TOP_K) });
    }
    fn ffn_done(&mut self, l: usize) {
        // every expert row is in the leader's VRAM (the host's workers fenced their write-combining stores; the
        // residents' fences were waited for): the leader's copy of the word, the leader sums and forwards (002)
        unsafe { core::arch::x86_64::_mm_sfence() };
        let f = self.ffns[l].as_ref().unwrap();
        unsafe { &*(self.peers[f.card].words.at(self.wd(W_EXPERTS, 0)) as *const AtomicU32) }.store(self.half_seq[l][self.ch][1], std::sync::atomic::Ordering::Release);
    }
    fn streams(&mut self, l: usize, i: usize, out: &mut [f32]) {
        // lab only: the layer's FFN card's streams of token `i`, through the host's FFN-input buffer
        let f = self.ffns[l].as_ref().unwrap();
        let c = &self.cards[f.card];
        self.copy_out(c, c.streams.at(i * HC * H * 4), self.shared.xh.va, HC);
        c.k.sync();
        read(&self.shared.xh, 0, &mut out[..HC * H]);
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
                    self.rows(c, &c.rows_h, None, c.fx.va, Some(norm.va), lm.va, c.lg.va, rows, H, np, ROWS);
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

impl Cards {
    /// One pass of pair l's predictor on the FFN card: the second margin's input (the FFN input plus the shared output,
    /// plus the weighted routed rows when `rows`), the next layer's router over it, `pred_step` — a decode step's plan
    /// into `plan` (early) or `plan2` (rows) with its word; a long step's into the margins only.
    fn pred_pass(&mut self, l: usize, n: usize, rows: bool) {
        use glm_5_3_flash::plan::{PLAN_LDS, PLAN_PRED_WGS, PLAN_WG};
        const E: usize = glm_5_3_flash::E;
        let slot = |l: usize| self.shared.labels.at((l * STEP + self.t0) * TOP_K * 4);
        let decode = n <= SPEC_TOKENS;
        let Some(f) = self.ffns[l].as_ref().filter(|f| f.pred.is_some()) else { return };
        let (st, d, d2) = f.pred.as_ref().unwrap();
        let c = &self.cards[f.card];
        let (px, pxn, pss, ps2) = (self.tok(&c.pred_x, H), self.tokb(&c.pred_xn, H * 2), self.tok(&c.pred_ss, glm_5_3_flash::plan::PRED_SS), self.tok(&c.pred_s2, ROUTE_ROWS));
        // the routed rows: the host's experts' rows as the leader holds them once its word is in (this pass runs behind
        // `hc_post`, which waited for it), weighted in pick order (093); null for the early pass
        let router = f.router.as_ref().unwrap().va;
        if rows && decode {
            // `hc_post` wrote the input (x + the whole FFN output); the next layer's router, the margins and the plan in
            // one launch (`pred_rows`: three launches were ~60 µs on the card's chain behind `hc_post`, 2026-10-22)
            let mut a = Args::new();
            a.ptr(pxn).ptr(pss).ptr(router).ptr(ps2).ptr(st.va).ptr(self.tok(&c.scores_v, ROUTE_ROWS)).ptr(d.at(self.t0 * E * 4)).ptr(d2.at(self.t0 * E * 4)).ptr(c.pred_aux.va);
            a.ptr(self.shared.plan2.va).ptr(self.shared.words.at(self.wd(W_PLAN2, f.card))).u32(self.half_seq[l][self.ch][1]).u32(n as u32).ptr(c.pred_keys.va).ptr(c.pred_count.va);
            c.k.launch(&c.pred_rows, glm_5_3_flash::plan::PRED_ROWS_BLOCKS, PLAN_WG, PLAN_LDS, &a);
            return;
        }
        if rows {
            // a long step's rows pass: the stacked router through the matrix path, the margins for the learning
            if decode {
                self.rows(c, &c.rows_h, Some(&c.rows_h1), px, Some(f.ln.va), router, ps2, ROUTE_ROWS, H, n, ROWS);
            } else {
                let mut a = Args::new();
                a.ptr(px).ptr(f.ln.va).ptr(router).ptr(ps2).u32(ROUTE_ROWS as u32).u32(n as u32);
                c.k.launch(&c.rows_h_w, g::wmma_grid(ROUTE_ROWS, n), g::WG, g::WMMA_LDS, &a);
            }
        } else {
            let mut a = Args::new();
            a.ptr(px).ptr(self.tok(&c.x, H)).ptr(self.tok(&c.fyv, H)).ptr(0usize).ptr(0usize).u32((n * H) as u32);
            c.k.launch(&c.pred_input, (n * H).div_ceil(256) as u32, g::WG, 0, &a);
            let mut a = Args::new();
            if decode {
                a.ptr(px).ptr(f.ln.va).ptr(router).ptr(ps2).ptr(ps2).ptr(c.pred_word.va).ptr(c.count.va).u32(0).u32(n as u32);
                c.k.launch(&c.hc_route_ahead, PRE_GRID, g::WG, 64, &a);
            } else {
                a.ptr(px).ptr(f.ln.va).ptr(router).ptr(ps2).u32(ROUTE_ROWS as u32).u32(n as u32);
                c.k.launch(&c.rows_h_w, g::wmma_grid(ROUTE_ROWS, n), g::WG, g::WMMA_LDS, &a);
            }
        }
        let mut a = Args::new();
        a.ptr(st.va).ptr(self.tok(&c.scores_v, ROUTE_ROWS)).ptr(ps2).ptr(d.at(self.t0 * E * 4)).ptr(d2.at(self.t0 * E * 4)).ptr(slot(l));
        if decode {
            let (plan, word) = if rows { (self.shared.plan2.va, self.shared.words.at(self.wd(W_PLAN2, f.card))) } else { (self.shared.plan.va, self.shared.words.at(self.wd(W_PLAN, f.card))) };
            a.ptr(plan).ptr(word);
        } else {
            a.ptr(0).ptr(0);
        }
        // the early decode pass leaves its thresholds and co-routing sums for the rows pass; a long step's does not
        a.u32(self.half_seq[l][self.ch][1]).u32(n as u32).ptr(c.pred_keys.va).ptr(c.pred_count.va).ptr(if decode && !rows { c.pred_aux.va } else { 0 });
        c.k.launch(&c.pred_step, if decode { n as u32 } else { PLAN_PRED_WGS as u32 }, PLAN_WG, PLAN_LDS, &a);
    }
}
