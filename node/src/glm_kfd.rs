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
use glm_5_3_flash::mla::gfx11_consts::{ATTEND_GRID, ATTEND_LDS, IX_ROWS, O_IN, ROWS_QB, SELECT_LDS};
use glm_5_3_flash::expert::gfx11_consts::{DOWN_WGS as XPT_DOWN_WGS, SLOT_BYTES, UP_WGS as XPT_UP_WGS, XPT_LDS};
use glm_5_3_flash::DENSE_BELOW;
use glm_5_3_flash::TOP_K;
use glm_5_3_flash::{INDEX_HEAD_DIM, INDEX_N_HEADS, KV_LORA_RANK, Q_LORA_RANK};
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
    /// Steps begun (the first MoE layer's dispatch): the loader paces itself to one load a step — a loader
    /// streaming flat out doubled every card lap (its BAR writes share the link with the handoffs) and the
    /// expert phase (its core and its DRAM reads come out of the pool's), 2026-10-17.
    ticks: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Picks seen, `[MOE_LAYERS][E]`.
    count: Vec<u32>,
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
/// The loader's core: a housekeeping core on the cards' socket other than the main thread's (core 0): a slow
/// 13 MB write over a degraded link held the main thread's core for a step at a time (2026-10-17).
const LOADER_CORE: usize = 24;
/// Replace a resident expert of the layer only when the newcomer has been picked this many more times.
const RESIDENT_MARGIN: u32 = 4;
/// VRAM a card may spend on resident experts at most (the node); what is free past a reserve decides.
pub const RESIDENT_BYTES: usize = 12 << 30;
/// VRAM left unallocated on a card: the driver's and the queue's own, and room for a bring-up gate's buffers.
const RESIDENT_RESERVE: usize = 1 << 30;

struct Card {
    k: Rc<Kfd>,
    xpt_gu: crate::kfd::Kernel,
    xpt_down: crate::kfd::Kernel,
    /// The resident experts' quantized activations and jobs (host-visible), the gate/up state, the outputs to the host.
    xq: Mem,
    xs: Mem,
    xg: Mem,
    xjobs: Mem,
    xgu: Mem,
    xout: Mem,
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
    mla_attend: crate::kfd::Kernel,
    ix_score: crate::kfd::Kernel,
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
}
/// The draft layer's own pieces on the head's card (its MLA attention is `mlas[MTP_LAYER]`, its
/// routed experts the host's as layer `MTP_LAYER`): `eh_proj`, the norms, the shared expert, and
/// the `[2·H]` input staging.
struct MtpOn {
    card: usize,
    eh: Mem,        // [H][2·H] bf16
    post_ln: Mem,   // bf16
    head_norm: Mem, // bf16
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
    ys: [Mem; 2],  // [2][PARTS][STEP][H] f32: the KDA parts' shares of the block output, by layer parity
    fy: [Mem; 2],  // [STEP][H] f32: the FFN card's rows (the dense MLP or the shared expert)
    my: [Mem; 2],  // [STEP][H] f32: the MLA card's output projection
    yexp: [Mem; 2], // [STEP][H] f32: the host's routed experts
    xh: Mem,       // [STEP][H] f32: a MoE layer's FFN input for the host
    scores_ffn: Mem, // [STEP][ROUTE_ROWS] f32: the FFN half's router over its input (this layer's rows, then the next layer's: the prefetch's predictor), for the host's picks
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
const W_X: usize = 2;
const W_FFN: usize = 3;
const W_EXPERTS: usize = 4;
const W_KINDS: usize = 5;
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

pub struct Cards {
    cards: Vec<Card>,
    shared: Shared,
    layers: Vec<Option<Layer>>,
    ffns: Vec<Option<FfnOn>>,
    mlas: Vec<Option<MlaOn>>,
    /// The head's card, the final norm (bf16) and lm_head.
    head: (usize, Mem, Mem),
    mtp: Option<MtpOn>,
    resident: Resident,
    /// Built with the launches stamped (`rockem.timeline`): the bring-up checks run too.
    instrument: bool,
    /// The handoff sequence: a value a half of a layer's words carry, and each layer's (attention, FFN) values this token.
    seq: u32,
    half_seq: Vec<[u32; 2]>,
    /// With the launches stamped: each card's first packet of the token, and the tokens since the last report.
    token_from: Vec<u64>,
    /// The step's token count (`begin`), for the report at `finish`.
    last_n: usize,
    /// Pools the context can hold: the scores' row stride.
    npools: usize,
    tokens: u32,
}

fn raw_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
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
                    xq: k.vram_visible(SPEC_TOKENS * H),
                    xs: k.vram_visible(SPEC_TOKENS * (H / common::q4::GROUP) * 4),
                    xg: k.vram_visible(SPEC_TOKENS * (H / common::q4::GROUP) * 4),
                    xjobs: k.vram_visible(SPEC_TOKENS * TOP_K * 8),
                    xgu: k.vram(SPEC_TOKENS * TOP_K * 2 * D_FF * 4),
                    xout: k.host(SPEC_TOKENS * TOP_K * H * 4),
                    ffn_gu: k.kernel("ffn_gu"),
                    ffn_down_shared: k.kernel("ffn_down_shared"),
                    ffn_down_dense: k.kernel("ffn_down_dense"),
                    mla_qb: k.kernel("mla_qb"),
                    mla_iq: k.kernel("mla_iq"),
                    mla_attend: k.kernel("mla_attend"),
                    ix_score: k.kernel("ix_score"),
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
        // the handoffs: host memory card 0 allocates and every other card maps too
        let shared = {
            let k0 = &cards[0].k;
            let others: Vec<&Kfd> = cards[1..].iter().map(|c| &*c.k).collect();
            let buf = || k0.host_shared(STEP * H * 4, &others);
            Shared {
                ys: [k0.host_shared(g::PARTS * STEP * H * 4, &others), k0.host_shared(g::PARTS * STEP * H * 4, &others)],
                fy: [buf(), buf()],
                my: [buf(), buf()],
                yexp: [buf(), buf()],
                xh: buf(),
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
            let on = MlaOn { card, input_ln: up(raw(&bf16_of(w.layers[l].input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lat: k.vram_visible(max_ctx * KV_LORA_RANK * 4), pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 4), o: one(mw.o) };
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
        // the draft layer: its MLA attention as layer MTP_LAYER on the head's card (the shared lm_head is
        // there), its own pieces beside it
        let mtp = w.mtp.as_ref().map(|mt| {
            let card = head.0;
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
            mlas.push(Some(MlaOn { card, input_ln: up(raw(&bf16_of(mt.input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), kv_b: up(raw(mw.kv_b)), lat: k.vram_visible(max_ctx * KV_LORA_RANK * 4), pools: k.vram_visible(max_ctx.div_ceil(INDEX_KPOOL) * INDEX_HEAD_DIM * 4), o: one(mw.o) }));
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
            MtpOn { card, eh: up(raw(mt.eh_proj)), post_ln: up(raw(&bf16_of(mt.post_ln))), head_norm: up(raw(&bf16_of(mt.head_norm))), gu_w, gu_s, down_w, down_s, dx: k.vram_visible(STEP * 2 * H * 4) }
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
                println!("gpu      : card {}: {:.1} GB allocated, {count} routed experts resident ({:.1} GB), {} a layer, filled on use and replaced least-used past a margin of {RESIDENT_MARGIN}", c.k.index, c.k.allocated.get() as f64 / 1e9, bytes as f64 / 1e9, count / moe);
                slots.push(c.k.vram_visible(bytes.max(SLOT_BYTES)));
                per_layer.push(count / moe);
                counts.push(count);
            }
            Resident { table: std::sync::Arc::new((0..moe * glm_5_3_flash::E).map(|_| AtomicU32::new(NONE)).collect()), loader: None, inflight: Default::default(), ticks: Default::default(), count: vec![0; moe * glm_5_3_flash::E], owner: counts.iter().map(|&k| vec![(NONE, NONE); k]).collect(), free: counts.iter().map(|&k| (0..k as u32).rev().collect()).collect(), held: (0..n).map(|_| vec![0; moe]).collect(), per_layer, slots, jobs: (0..n).map(|_| Vec::new()).collect(), uploads: 0 }
        };
        Cards { cards, shared, layers, ffns, mlas, head, mtp, resident, instrument, seq: 0, half_seq: vec![[0; 2]; NUM_HIDDEN_LAYERS], token_from: vec![0; n], last_n: 1, npools: max_ctx.div_ceil(INDEX_KPOOL), tokens: 0 }
    }

    /// With the launches stamped, every 16 tokens: a line per card of the last token's time by
    /// kernel (busy µs and count, most first) and the gaps between launches — where a token's
    /// card time goes.
    /// With `rockem.timeline`: every step of more than one token, and every sixteenth single token.
    fn timeline_report(&mut self, n: usize) {
        if !self.cards[0].k.instrumented() {
            return;
        }
        self.tokens += 1;
        if n == 1 && self.tokens % 16 != 0 {
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
            println!("timeline : card {i}, {n} tokens: {} launches{} over {span:.0} µs, {busy:.0} busy, {gaps:.0} between; by kernel (µs/count): {}", tl.len(), if c.k.posted() - self.token_from[i] > tl.len() as u64 { " (the ring's last)" } else { "" }, parts.join(", "));
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
    fn hc_post(&self, c: &Card, copy: Option<(usize, usize, usize)>, mine: Option<(usize, usize)>, w0: Option<(usize, usize)>, w1: Option<(usize, usize)>, value: u32, ya: usize, yb: Option<usize>, tokens: usize) {
        let at = |w: Option<(usize, usize)>| w.map_or(0, |(k, d)| self.shared.words.at(word(k, d)));
        let (src, dst, n) = copy.unwrap_or((0, 0, 0));
        let mut a = Args::new();
        a.ptr(c.mix.va).ptr(ya).ptr(yb.unwrap_or(0)).ptr(c.streams.va).ptr(src).ptr(dst).u32(n as u32).ptr(at(mine)).ptr(at(w0)).ptr(at(w1)).u32(value).ptr(c.go.va).u32(tokens as u32);
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
        let ys = &self.shared.ys[parity];
        for c in &self.cards {
            let mine = &parts[c.k.index];
            // a part's share is [STEP][H] at its slot: this card's parts' n tokens each, one span from the first slot
            let copy = mine.first().map(|&p0| (c.yv.at(p0 * STEP * H * 4), ys.at(p0 * STEP * H * 4), ((mine.len() - 1) * STEP + n) * H));
            self.hc_post(c, copy, (!mine.is_empty()).then_some((W_Y, c.k.index)), Some((W_Y, producers[0])), producers.get(1).map(|&p| (W_Y, p)), value, ys.va, Some(ys.at(STEP * H * 4)), n);
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

impl Placed for Cards {
    fn begin(&mut self, rows: &[f32], n: usize) {
        self.last_n = n;
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
                self.hc_post(o, None, None, Some((W_Y, m.card)), None, value, self.shared.my[l % 2].va, None, n);
            }
        }
    }
    fn hidden_all(&mut self, n: usize, out: &mut [f32]) {
        // the streams' means a pass of TOKENS at a time, through the head card's logits buffer (host memory)
        let c = &self.cards[self.head.0];
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
        let ticks = self.resident.ticks.clone();
        let bases: Vec<usize> = self.resident.slots.iter().map(|m| m.ptr() as usize).collect();
        std::thread::Builder::new().name("xpt-loader".into()).spawn(move || {
            extern "C" {
                fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
            }
            let mut mask = [0u64; 16];
            mask[LOADER_CORE / 64] |= 1 << (LOADER_CORE % 64);
            unsafe { sched_setaffinity(0, 128, mask.as_ptr()) };
            let mut scratch = vec![0u8; SLOT_BYTES];
            let mut served = 0u64;
            for (card, slot, ml, e) in rx {
                // one load a step: wait for a step the last load did not see
                loop {
                    let t = ticks.load(std::sync::atomic::Ordering::Acquire);
                    if t > served {
                        served = t;
                        break;
                    }
                    std::thread::park_timeout(std::time::Duration::from_micros(200));
                }
                fetch(ml + glm_5_3_flash::FIRST_K_DENSE_REPLACE, e, &mut scratch);
                let dst = unsafe { std::slice::from_raw_parts_mut((bases[card] + slot as usize * SLOT_BYTES) as *mut u8, SLOT_BYTES) };
                common::copy::stream(dst, &scratch);
                table[ml * glm_5_3_flash::E + e].store((card as u32) << 24 | slot, std::sync::atomic::Ordering::Release);
                inflight.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }
        }).unwrap();
        self.resident.loader = Some(tx);
    }
    fn xpt_dispatch(&mut self, l: usize, n: usize, picks: &mut [[usize; TOP_K]], xq: &[i8], xs: &[f32], xg: &[i32]) -> usize {
        use std::sync::atomic::Ordering;
        let ml = l - glm_5_3_flash::FIRST_K_DENSE_REPLACE;
        let e_n = glm_5_3_flash::E;
        let r = &mut self.resident;
        let loader = r.loader.as_ref().expect("the residency's loader is set before the first step");
        for j in r.jobs.iter_mut() {
            j.clear();
        }
        if ml == 0 {
            r.ticks.fetch_add(1, Ordering::AcqRel);
        }
        // the resident picks become jobs; the experts with a job this step are never a victim
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
                    r.jobs[(at >> 24) as usize].push((at & 0xff_ffff, i as u32, k as u8));
                    picks[i][k] = e_n;
                    in_use[e] = true;
                } else if at == NONE && !misses.contains(&e) {
                    misses.push(e);
                }
            }
        }
        // the misses claim slots for the steps to come: a free one on the card holding fewer of this layer,
        // else the layer's least-picked resident when the miss leads it by the margin (one a layer a step)
        let mut replaced = false;
        for &e in &misses {
            if r.inflight.load(Ordering::Acquire) >= LOADS_IN_FLIGHT {
                break;
            }
            let mut pick_card = None;
            for c in (0..self.cards.len()).rev() {
                if (r.held[c][ml] as usize) < r.per_layer[c] && !r.free[c].is_empty() && pick_card.is_none_or(|p: usize| r.held[c][ml] < r.held[p][ml]) {
                    pick_card = Some(c);
                }
            }
            let slot = match pick_card {
                Some(c) => Some((c, r.free[c].pop().unwrap())),
                None if !replaced => {
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
                            replaced = true;
                            Some((c, sl))
                        }
                        _ => None,
                    }
                }
                None => None,
            };
            if let Some((c, sl)) = slot {
                r.owner[c][sl as usize] = (ml as u32, e as u32);
                r.held[c][ml] += 1;
                r.uploads += 1;
                r.table[ml * e_n + e].store(PENDING, Ordering::Release);
                r.inflight.fetch_add(1, Ordering::AcqRel);
                loader.send((c, sl, ml, e)).unwrap();
            }
        }
        let mut total = 0;
        for c in 0..self.cards.len() {
            let jobs = &r.jobs[c];
            if jobs.is_empty() {
                continue;
            }
            total += jobs.len();
            let card = &self.cards[c];
            card.k.upload(&card.xq, 0, raw_i8(xq));
            card.k.upload(&card.xs, 0, raw(xs));
            card.k.upload(&card.xg, 0, raw(xg));
            let list: Vec<u32> = jobs.iter().flat_map(|&(sl, tok, _)| [sl, tok]).collect();
            card.k.upload(&card.xjobs, 0, raw(&list));
            let mut a = Args::new();
            a.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(card.xq.va).ptr(card.xs.va).ptr(card.xg.va).ptr(card.xgu.va);
            card.k.launch(&card.xpt_gu, (jobs.len() * XPT_UP_WGS) as u32, g::WG, XPT_LDS as u32, &a);
            let mut b = Args::new();
            b.ptr(r.slots[c].va).ptr(card.xjobs.va).ptr(card.xgu.va).ptr(card.xout.va);
            card.k.launch(&card.xpt_down, (jobs.len() * XPT_DOWN_WGS) as u32, g::WG, XPT_LDS as u32, &b);
            card.k.fence();
        }
        total
    }
    fn xpt_join(&mut self, _l: usize, out: &mut [f32]) {
        for c in 0..self.cards.len() {
            let jobs = &self.resident.jobs[c];
            if jobs.is_empty() {
                continue;
            }
            let card = &self.cards[c];
            card.k.wait();
            let y = unsafe { std::slice::from_raw_parts(card.xout.ptr() as *const f32, jobs.len() * H) };
            for (j, &(_, tok, k)) in jobs.iter().enumerate() {
                let at = (tok as usize * TOP_K + k as usize) * H;
                out[at..at + H].copy_from_slice(&y[j * H..(j + 1) * H]);
            }
        }
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
        let c = &self.cards[t.card];
        let (_, _, lm) = &self.head;
        self.rows(c, &c.rows_h, c.x.at(i * H * 4), Some(t.head_norm.va), lm.va, c.lg.va, VOCAB_SIZE, H, 1, ROWS);
        c.k.sync();
        read(&c.lg, 0, &mut logits[..VOCAB_SIZE]);
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
    fn ix_pools(&mut self, l: usize, pools: &[f32], from: usize, to: usize) {
        let m = self.mlas[l].as_ref().unwrap();
        if to > from {
            self.cards[m.card].k.upload(&m.pools, from * INDEX_HEAD_DIM * 4, raw(&pools[from * INDEX_HEAD_DIM..to * INDEX_HEAD_DIM]));
        }
    }
    fn ix_select(&mut self, l: usize, pos: usize, n: usize, count: Option<&mut usize>) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the scores of every selecting token's pools, then each token's selection into the attend's lists;
        // nothing comes back to the host but a count asked for (bring-up)
        let stride = self.npools;
        if glm_5_3_flash::mla::selects(pos + n - 1) {
            let nb = ((pos + n) / INDEX_KPOOL).div_ceil(256);
            let (scale, wscale) = glm_5_3_flash::mla::score_scales();
            let mut a = Args::new();
            a.ptr(c.miq.va).ptr(c.mproj.at(MIX * 4)).ptr(m.pools.va).ptr(c.mscore.va).u32(pos as u32).u32(nb as u32).u32(stride as u32).f32(scale).f32(wscale);
            c.k.launch(&c.ix_score, (n * nb) as u32, g::WG, 0, &a);
        }
        let mut a = Args::new();
        a.ptr(c.mscore.va).u32(stride as u32).u32(pos as u32).ptr(c.msel.va).ptr(c.mseln.va);
        c.k.launch(&c.ix_select, n as u32, g::WG, SELECT_LDS as u32, &a);
        if let Some(count) = count {
            c.k.sync();
            let mut w = [0u32; 1];
            read(&c.mseln, 0, &mut w);
            *count = w[0] as usize;
        }
    }
    fn mla_attend(&mut self, l: usize, n: usize, pos: usize, latents: &[f32]) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        // the step's latents into the mirror; the selections are the card's (`ix_select`); one launch over (token, heads)
        c.k.upload(&m.lat, pos * KV_LORA_RANK * 4, raw(&latents[..n * KV_LORA_RANK]));
        let mut a = Args::new();
        a.ptr(c.mq.va).ptr(m.kv_b.va).ptr(m.lat.va).ptr(c.msel.va).ptr(c.mseln.va).ptr(c.mheads.va);
        c.k.launch(&c.mla_attend, ATTEND_GRID * n as u32, g::WG, ATTEND_LDS as u32, &a);
    }
    fn mla_out(&mut self, l: usize, n: usize) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        let my = self.shared.my[l % 2].va;
        // the output projection over every token's heads, then the streams
        if n > SPEC_TOKENS {
            self.rows_w(c, &c.mla_o_w, c.mheads.va, None, m.o.0.va, Some(m.o.1.va), c.myv.va, H, n, false);
        } else {
            self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.myv.va, H, O_IN, n, ROWS);
        }
        self.hc_post(c, Some((c.myv.va, my, n * H)), Some((W_Y, m.card)), None, None, self.half_seq[l][0], my, None, n);
    }
    fn mla_seed(&mut self, l: usize, pos: usize, latent: &[f32]) {
        let m = self.mlas[l].as_ref().unwrap();
        self.cards[m.card].k.upload(&m.lat, pos * KV_LORA_RANK * 4, raw(&latent[..KV_LORA_RANK]));
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
        let fy = self.shared.fy[parity].va;
        for o in &self.cards {
            let producer = o.k.index == f.card;
            let copy = producer.then_some((o.fyv.va, fy, n * H));
            let mine = producer.then_some((W_FFN, f.card));
            if f.dense {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), None, value, fy, None, n);
            } else {
                self.hc_post(o, copy, mine, Some((W_FFN, f.card)), Some((W_EXPERTS, 0)), value, fy, Some(self.shared.yexp[parity].va), n);
            }
        }
    }
    fn ffn_input(&mut self, l: usize, n: usize, x: &mut [f32], logits: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        crate::kfd::wait_word(self.word_ptr(W_X, f.card), self.half_seq[l][1]);
        read(&self.shared.xh, 0, &mut x[..n * H]);
        read(&self.shared.scores_ffn, 0, &mut logits[..n * ROUTE_ROWS]);
    }
    fn ffn_routed(&mut self, l: usize, n: usize, y: &[f32]) {
        let buf = &self.shared.yexp[l % 2];
        unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), buf.ptr() as *mut f32, n * H) };
        unsafe { &*self.word_ptr(W_EXPERTS, 0) }.store(self.half_seq[l][1], std::sync::atomic::Ordering::Release);
    }
    fn layer_out(&mut self, _l: usize, i: usize, mean: &mut [f32]) {
        let c = &self.cards[self.head.0];
        let mut a = Args::new();
        a.ptr(c.streams.at(i * HC * H * 4)).ptr(c.mean.va).u32(1);
        c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
        c.k.sync();
        read(&c.mean, 0, &mut mean[..H]);
    }
    fn logits(&mut self, from: usize, n: usize, logits: &mut [f32]) {
        let (card, norm, lm) = &self.head;
        let c = &self.cards[*card];
        // a pass of TOKENS at a time: the streams' means, the norm and lm_head over them, the pass read back
        let mut p0 = 0;
        while p0 < n {
            let np = (n - p0).min(TOKENS);
            let mut a = Args::new();
            a.ptr(c.streams.at((from + p0) * HC * H * 4)).ptr(c.fx.va).u32(np as u32);
            c.k.launch(&c.hc_mean, CHANNEL_GRID, g::WG, 0, &a);
            if np > SPEC_TOKENS {
                let mut a = Args::new();
                a.ptr(c.fx.va).ptr(norm.va).ptr(lm.va).ptr(c.lg.va).u32(VOCAB_SIZE as u32).u32(np as u32);
                c.k.launch(&c.rows_h_w, g::wmma_grid(VOCAB_SIZE, np), g::WG, g::WMMA_LDS, &a);
            } else {
                self.rows(c, &c.rows_h, c.fx.va, Some(norm.va), lm.va, c.lg.va, VOCAB_SIZE, H, np, ROWS);
            }
            c.k.sync();
            read(&c.lg, 0, &mut logits[p0 * VOCAB_SIZE..(p0 + np) * VOCAB_SIZE]);
            p0 += np;
        }
        // the step ends when every card is done with its streams: the next begins by overwriting them
        for c in &self.cards {
            c.k.sync();
        }
        self.timeline_report(self.last_n);
    }
    fn finish(&mut self) {
        for c in &self.cards {
            c.k.sync();
        }
        self.timeline_report(self.last_n);
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
