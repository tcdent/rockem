//! GLM-5.3-Flash's blocks on the cards, behind `construct::Placed`. Each
//! KDA layer is split by heads into `PARTS` parts, one card each where there
//! are two: a part holds its heads' projections, the layer's small tensors
//! and the state, and a step runs on each card the normed projections
//! (`rows_h`: the gates' first halves and beta once a card, then the part's
//! q, k, v rows), the gates' second halves (`rows_d`), the delta rule with
//! the heads' memory resident in registers (`kda_step` over the part's
//! heads), and the part's half of the output projection (`rows_part`); the
//! cards are waited for and the host sums the parts. Each layer's FFN half that is not the routed experts — the
//! dense layers' MLP, the MoE layers' shared expert — is on the same card as
//! FP8 rows: a gate|up launch over the normed block input and a down launch
//! over its SwiGLU (`ffn_gu`, `ffn_down_*`), started before the routed
//! experts run on the host and finished after them. The head is the final
//! norm and lm_head as one `rows_h` launch over the vocabulary.
//!
//! Block inputs come in through host-visible VRAM (a BAR write streams at 16
//! GB/s); every output goes out through host memory the last launch writes
//! across the link, because the host reads its BAR at 10–30 MB/s — 470–690 µs
//! for 20 KB, against 0.2 µs from host memory (probe, 2026-10-15). The
//! residual streams stay on the host, where the hyper-connections run. The
//! layers split in order between the cards the topology lists, by their KDA
//! layers; a layer's FFN sits with the KDA layer at or before it, the head
//! with the last layer.

use crate::kfd::{read, Args, Kfd, Mem};
use glm_5_3_flash::construct::Placed;
use glm_5_3_flash::ffn::gfx11_consts::{DENSE_GU_ROWS, SHARED_GU_ROWS};
use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::mla::gfx11_consts::{IX_ROWS, O_IN};
use glm_5_3_flash::{INDEX_HEAD_DIM, INDEX_N_HEADS, KV_LORA_RANK, Q_LORA_RANK};
use glm_5_3_flash::weights::{Attn, Ffn, Fp8, Weights};
use glm_5_3_flash::{CONV_DIM, CONV_TAPS, D_FF, H, INTERMEDIATE_SIZE, KDA_DIM, KDA_HEADS, KDA_QKV, LINEAR_ATTENTION, MLA_HEADS, NUM_HIDDEN_LAYERS, N_SHARED_EXPERTS, Q_HEAD, VOCAB_SIZE};
use std::rc::Rc;

struct Card {
    k: Rc<Kfd>,
    rows_h: crate::kfd::Kernel,
    rows_d: crate::kfd::Kernel,
    rows_part: crate::kfd::Kernel,
    step: crate::kfd::Kernel,
    copy: crate::kfd::Kernel,
    ffn_gu: crate::kfd::Kernel,
    ffn_down_shared: crate::kfd::Kernel,
    ffn_down_dense: crate::kfd::Kernel,
    mla_qb: crate::kfd::Kernel,
    mla_iq: crate::kfd::Kernel,
    mla_o: crate::kfd::Kernel,
    // the step's buffers, one token deep
    x: Mem,      // [H] f32, host-visible: the block input
    mixed: Mem,  // [3][KDA_QKV] f32: q | k | v
    small: Mem,  // [KDA_DIM] fa | [KDA_DIM] ga | [KDA_HEADS] beta
    f: Mem,      // [KDA_QKV]
    gate: Mem,   // [KDA_QKV]
    normed: Mem, // [KDA_QKV]
    ys: Vec<Mem>, // [PARTS][H] f32, host memory: each part's share of the block output
    snap: Mem,   // [KDA_WINDOW + KDA_MEMORY] f32, host memory: a layer's state on its way out
    // the FFN's and the head's
    fx: Mem,     // [H] f32, host-visible: the FFN's block input, or the streams' mean
    gu: Mem,     // [DENSE_GU_ROWS] f32: gate | up
    fy: Mem,     // [H] f32, host memory: the FFN's output
    lg: Mem,     // [VOCAB_SIZE] f32, host memory: the logits
    // the MLA's
    mqa: Mem,    // [Q_LORA_RANK] f32: q_a's output, the input of q_b and the indexer's query
    mproj: Mem,  // host memory: q [MLA_HEADS·Q_HEAD] | kva [KV_LORA_RANK] | ix [IX_ROWS] | iq [INDEX_N_HEADS·INDEX_HEAD_DIM]
    mheads: Mem, // [O_IN] f32, host-visible: the heads' outputs
    my: Mem,     // [H] f32, host memory: the output projection
}
const MQ: usize = 0;
const MKVA: usize = MQ + MLA_HEADS * Q_HEAD;
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
    qkv: [Mem; 3], // each [PART_QKV][H] bf16
    o: Mem,        // [H][PART_QKV] bf16
    taps: Mem,     // [3][KDA_QKV][CONV_TAPS] bf16
    small: Mem,    // [SMALL_ROWS][H] bf16: f_a's rows, g_a's, b's
    f_b: Mem,      // [PART_QKV][KDA_DIM]
    g_b: Mem,
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
    gu_w: Mem,
    gu_s: Mem,
    down_w: Mem,
    down_s: Mem,
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
    o: (Mem, Mem),
}

pub struct Cards {
    cards: Vec<Card>,
    layers: Vec<Option<Layer>>,
    ffns: Vec<Option<FfnOn>>,
    mlas: Vec<Option<MlaOn>>,
    /// The head's card, the final norm (bf16) and lm_head.
    head: Option<(usize, Mem, Mem)>,
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
    pub fn new(elf: &[u8], w: &Weights) -> Self {
        let n = Kfd::cards().len();
        assert!(n > 0, "rockem.gpu: no gfx1100 card");
        let t0 = std::time::Instant::now();
        let cards: Vec<Card> = (0..n)
            .map(|i| {
                let k = Kfd::open(elf, false, i);
                let alloc = |floats: usize| k.vram_visible(floats * 4);
                Card {
                    rows_h: k.kernel("rows_h"),
                    rows_d: k.kernel("rows_d"),
                    rows_part: k.kernel("rows_part"),
                    step: k.kernel("kda_step"),
                    copy: k.kernel("copy_f32"),
                    ffn_gu: k.kernel("ffn_gu"),
                    ffn_down_shared: k.kernel("ffn_down_shared"),
                    ffn_down_dense: k.kernel("ffn_down_dense"),
                    mla_qb: k.kernel("mla_qb"),
                    mla_iq: k.kernel("mla_iq"),
                    mla_o: k.kernel("mla_o"),
                    x: alloc(H),
                    mixed: alloc(3 * KDA_QKV),
                    small: alloc(SMALL_ROWS),
                    f: alloc(KDA_QKV),
                    gate: alloc(KDA_QKV),
                    normed: alloc(KDA_QKV),
                    ys: (0..g::PARTS).map(|_| k.host(H * 4)).collect(),
                    snap: k.host((KDA_WINDOW + KDA_MEMORY) * 4),
                    fx: alloc(H),
                    gu: alloc(DENSE_GU_ROWS),
                    fy: k.host(H * 4),
                    lg: k.host(VOCAB_SIZE * 4),
                    mqa: alloc(Q_LORA_RANK),
                    mproj: k.host(MPROJ * 4),
                    mheads: alloc(O_IN),
                    my: k.host(H * 4),
                    k,
                }
            })
            .collect();
        Kfd::clocks_high();
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
                let small = {
                    let m = k.vram_visible(SMALL_ROWS * H * 2);
                    k.upload(&m, 0, raw(kw.f_a));
                    k.upload(&m, kw.f_a.len() * 2, raw(kw.g_a));
                    k.upload(&m, 2 * kw.f_a.len() * 2, raw(kw.b));
                    m
                };
                let p = Part {
                    card,
                    head0,
                    input_ln: up(raw(&input_ln)),
                    qkv: [up(raw(rows(kw.q, H, head0))), up(raw(rows(kw.k, H, head0))), up(raw(rows(kw.v, H, head0)))],
                    o: up(raw(&o_half)),
                    taps: up(raw(&taps)),
                    small,
                    f_b: up(raw(rows(kw.f_b, KDA_DIM, head0))),
                    g_b: up(raw(rows(kw.g_b, KDA_DIM, head0))),
                    dt_bias: up(raw(kw.dt_bias)),
                    a_log: up(raw(kw.a_log)),
                    o_norm: up(raw(kw.o_norm)),
                    window: up(raw(&zero_w)),
                    mem: up(raw(&zero_m)),
                };
                bytes += 2 * (3 * g::PART_QKV * H + H * g::PART_QKV + SMALL_ROWS * H + 2 * g::PART_QKV * KDA_DIM + taps.len()) + 4 * (KDA_WINDOW + KDA_MEMORY);
                parts.push(p);
            }
            layers[l] = Some(Layer { parts });
        }
        // every layer's FFN half but the routed experts
        let mut ffns: Vec<Option<FfnOn>> = Vec::with_capacity(NUM_HIDDEN_LAYERS);
        let (mut dense, mut shared) = (0, 0);
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
            let (is_dense, (gu_w, gu_s), (down_w, down_s)) = match &w.layers[l].ffn {
                Ffn::Dense(d) => (true, pair(d.gate, d.up), one(d.down)),
                Ffn::Moe(m) => (false, pair(m.shared_gate, m.shared_up), one(m.shared_down)),
            };
            if is_dense {
                dense += 1;
            } else {
                shared += 1;
            }
            bytes += gu_w.size + gu_s.size + down_w.size + down_s.size + ln.size;
            ffns.push(Some(FfnOn { card, dense: is_dense, ln, gu_w, gu_s, down_w, down_s }));
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
            let on = MlaOn { card, input_ln: up(raw(&bf16_of(w.layers[l].input_ln))), qa_ln: up(raw(&bf16_of(mw.q_a_ln))), q_a: one(mw.q_a), kv_a: one(mw.kv_a), ix, q_b: one(mw.q_b), ix_wq_b: up(raw(mw.ix_wq_b)), o: one(mw.o) };
            bytes += on.q_a.0.size + on.q_a.1.size + on.kv_a.0.size + on.kv_a.1.size + on.ix.size + on.q_b.0.size + on.q_b.1.size + on.ix_wq_b.size + on.o.0.size + on.o.1.size;
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
            Some((card, norm, lm))
        };
        println!("gpu      : {} KDA layers, {mla_n} MLA layers' projections, {dense} dense FFNs, {shared} shared experts and the head on {} card{} ({:.1} GB with their state) in {:.1}s", kda_layers.len(), n, if n == 1 { "" } else { "s" }, bytes as f64 / 1e9, t0.elapsed().as_secs_f32());
        Cards { cards, layers, ffns, mlas, head }
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
    fn kda(&self, l: usize) -> bool {
        self.layers[l].is_some()
    }
    fn ffn(&self, l: usize) -> bool {
        self.ffns[l].is_some()
    }
    fn head(&self) -> bool {
        self.head.is_some()
    }
    fn ffn_start(&mut self, l: usize, x: &[f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        let c = &self.cards[f.card];
        c.k.upload(&c.fx, 0, raw(&x[..H]));
        let (rows, down, k_width) = if f.dense { (DENSE_GU_ROWS, &c.ffn_down_dense, INTERMEDIATE_SIZE) } else { (SHARED_GU_ROWS, &c.ffn_down_shared, N_SHARED_EXPERTS * D_FF) };
        self.rows8(c, &c.ffn_gu, c.fx.va, Some(f.ln.va), f.gu_w.va, f.gu_s.va, c.gu.va, rows, H);
        self.rows8(c, down, c.gu.va, None, f.down_w.va, f.down_s.va, c.fy.va, H, k_width);
    }
    fn ffn_finish(&mut self, l: usize, out: &mut [f32]) {
        let f = self.ffns[l].as_ref().unwrap();
        let c = &self.cards[f.card];
        c.k.sync();
        read(&c.fy, 0, &mut out[..H]);
    }
    fn logits(&mut self, mean: &[f32], logits: &mut [f32]) {
        let (card, norm, lm) = self.head.as_ref().unwrap();
        let c = &self.cards[*card];
        c.k.upload(&c.fx, 0, raw(&mean[..H]));
        self.rows(c, &c.rows_h, c.fx.va, Some(norm.va), lm.va, c.lg.va, VOCAB_SIZE, H);
        c.k.sync();
        read(&c.lg, 0, &mut logits[..VOCAB_SIZE]);
    }
    fn mla(&self, l: usize) -> bool {
        self.mlas[l].is_some()
    }
    fn mla_project(&mut self, l: usize, x: &[f32], q: &mut [f32], kva: &mut [f32], ik: &mut [f32], igate: &mut [f32], iq: &mut [f32], iw: &mut [f32]) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        c.k.upload(&c.fx, 0, raw(&x[..H]));
        let ln = Some(m.input_ln.va);
        // over the normed block input: q_a into VRAM for the next two, kv_a and the indexer's three out to the host
        self.rows8(c, &c.ffn_gu, c.fx.va, ln, m.q_a.0.va, m.q_a.1.va, c.mqa.va, Q_LORA_RANK, H);
        self.rows8_free(c, &c.ffn_gu, c.fx.va, ln, m.kv_a.0.va, m.kv_a.1.va, c.mproj.at(MKVA * 4), KV_LORA_RANK, H);
        self.rows_free(c, &c.rows_h, c.fx.va, ln, m.ix.va, c.mproj.at(MIX * 4), IX_ROWS, H);
        // over q_a's output normed with q_a's norm: the query and, alongside it, the indexer's query
        let qln = Some(m.qa_ln.va);
        self.rows8(c, &c.mla_qb, c.mqa.va, qln, m.q_b.0.va, m.q_b.1.va, c.mproj.at(MQ * 4), MLA_HEADS * Q_HEAD, Q_LORA_RANK);
        self.rows_free(c, &c.mla_iq, c.mqa.va, qln, m.ix_wq_b.va, c.mproj.at(MIQ * 4), INDEX_N_HEADS * INDEX_HEAD_DIM, Q_LORA_RANK);
        c.k.sync();
        read(&c.mproj, MQ * 4, &mut q[..MLA_HEADS * Q_HEAD]);
        read(&c.mproj, MKVA * 4, &mut kva[..KV_LORA_RANK]);
        read(&c.mproj, MIX * 4, &mut ik[..INDEX_HEAD_DIM]);
        read(&c.mproj, (MIX + INDEX_HEAD_DIM) * 4, &mut igate[..INDEX_HEAD_DIM]);
        read(&c.mproj, (MIX + 2 * INDEX_HEAD_DIM) * 4, &mut iw[..INDEX_N_HEADS]);
        read(&c.mproj, MIQ * 4, &mut iq[..INDEX_N_HEADS * INDEX_HEAD_DIM]);
    }
    fn mla_output(&mut self, l: usize, heads: &[f32], out: &mut [f32]) {
        let m = self.mlas[l].as_ref().unwrap();
        let c = &self.cards[m.card];
        c.k.upload(&c.mheads, 0, raw(&heads[..O_IN]));
        self.rows8(c, &c.mla_o, c.mheads.va, None, m.o.0.va, m.o.1.va, c.my.va, H, O_IN);
        c.k.sync();
        read(&c.my, 0, &mut out[..H]);
    }
    fn kda_step(&mut self, l: usize, x: &[f32], out: &mut [f32]) {
        let lay = self.layers[l].as_ref().unwrap();
        // on each card a part uses: the block input, and the gates' first halves and beta (every head) over the normed input
        let mut seen = [usize::MAX; g::PARTS];
        for (i, p) in lay.parts.iter().enumerate() {
            if !seen[..i].contains(&p.card) {
                let c = &self.cards[p.card];
                c.k.upload(&c.x, 0, raw(&x[..H]));
                self.rows(c, &c.rows_h, c.x.va, Some(p.input_ln.va), p.small.va, c.small.va, SMALL_ROWS, H);
            }
            seen[i] = p.card;
        }
        for (i, p) in lay.parts.iter().enumerate() {
            let c = &self.cards[p.card];
            let ln = Some(p.input_ln.va);
            let at = p.head0 * KDA_DIM * 4; // the part's heads' offset into a [KDA_QKV] buffer
            // q, k, v, and the gates' second halves alongside them (the small projections completed before q launched)
            self.rows(c, &c.rows_h, c.x.va, ln, p.qkv[0].va, c.mixed.at(at), g::PART_QKV, H);
            for part in 1..3 {
                self.rows_free(c, &c.rows_h, c.x.va, ln, p.qkv[part].va, c.mixed.at(part * KDA_QKV * 4 + at), g::PART_QKV, H);
            }
            self.rows_free(c, &c.rows_d, c.small.va, None, p.f_b.va, c.f.at(at), g::PART_QKV, KDA_DIM);
            self.rows_free(c, &c.rows_d, c.small.at(KDA_DIM * 4), None, p.g_b.va, c.gate.at(at), g::PART_QKV, KDA_DIM);
            let mut a = Args::new();
            a.ptr(c.mixed.va).ptr(p.taps.va).ptr(p.window.va).ptr(c.f.va).ptr(p.dt_bias.va).ptr(p.a_log.va).ptr(c.small.at(2 * KDA_DIM * 4)).ptr(c.gate.va).ptr(p.mem.va).ptr(p.o_norm.va).ptr(c.normed.va).u32(p.head0 as u32).u32(1);
            c.k.launch(&c.step, g::STEP_GRID, g::WG, g::STEP_LDS as u32, &a);
            self.rows(c, &c.rows_part, c.normed.at(at), None, p.o.va, c.ys[i].va, H, g::PART_QKV);
        }
        for (i, p) in lay.parts.iter().enumerate() {
            if !seen[..i].contains(&p.card) {
                self.cards[p.card].k.sync();
            }
        }
        // the block output is the parts' sum
        read(&self.cards[lay.parts[0].card].ys[0], 0, &mut out[..H]);
        let mut part = [0f32; H];
        for (i, p) in lay.parts.iter().enumerate().skip(1) {
            read(&self.cards[p.card].ys[i], 0, &mut part[..]);
            for d in 0..H {
                out[d] += part[d];
            }
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
