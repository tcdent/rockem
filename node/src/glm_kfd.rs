//! GLM-5.3-Flash's recurrent-attention layers on the cards. Each KDA layer
//! placed on a card holds its weights and its state there — the projections,
//! the convolution taps, the gates' parameters, the window and the memory —
//! and a step runs its ten launches on that card's queue with one wait: the
//! normed projections (`rows_h`: q, k and v as one launch over their rows
//! back to back, then the gates' first halves and beta as another), the gates'
//! second halves (`rows_d`), the delta rule with the memory resident in
//! registers (`kda_step`), and the output projection (`rows_qkv`) — six
//! launches a layer. The block input comes in and the block
//! output goes out through host-visible VRAM; the residual streams stay on
//! the host, where the hyper-connections and the FFN half still run. The 34
//! layers split in order between the cards the topology lists.

use crate::kfd::{read, Args, Kfd, Mem};
use glm_5_3_flash::construct::KdaPlaced;
use glm_5_3_flash::kda::gfx11_consts as g;
use glm_5_3_flash::layout::{KDA_MEMORY, KDA_WINDOW};
use glm_5_3_flash::weights::{Attn, Weights};
use glm_5_3_flash::{CONV_DIM, CONV_TAPS, H, KDA_DIM, KDA_HEADS, KDA_QKV, LINEAR_ATTENTION, NUM_HIDDEN_LAYERS};
use std::rc::Rc;

struct Card {
    k: Rc<Kfd>,
    rows_h: crate::kfd::Kernel,
    rows_d: crate::kfd::Kernel,
    rows_qkv: crate::kfd::Kernel,
    step: crate::kfd::Kernel,
    // the step's buffers, one token deep
    x: Mem,      // [H] f32, host-visible: the block input
    mixed: Mem,  // [3][KDA_QKV] f32: q | k | v
    small: Mem,  // [KDA_DIM] fa | [KDA_DIM] ga | [KDA_HEADS] beta
    f: Mem,      // [KDA_QKV]
    gate: Mem,   // [KDA_QKV]
    normed: Mem, // [KDA_QKV]
    y: Mem,      // [H] f32, host-visible: the block output
}
/// Rows of the fused small projection: f_a's, g_a's, b's.
const SMALL_ROWS: usize = 2 * KDA_DIM + KDA_HEADS;

struct Layer {
    card: usize,
    input_ln: Mem,
    qkv: Mem,   // [3·KDA_QKV][H] bf16: q's rows, k's, v's
    o: Mem,
    taps: Mem,  // [3][KDA_QKV][CONV_TAPS] bf16
    small: Mem, // [SMALL_ROWS][H] bf16: f_a's rows, g_a's, b's
    f_b: Mem,
    g_b: Mem,
    dt_bias: Mem,
    a_log: Mem,
    o_norm: Mem,
    window: Mem, // [CONV_TAPS-1][CONV_DIM] f32
    mem: Mem,    // [KDA_HEADS][KDA_DIM][KDA_DIM] f32
}

pub struct KdaCards {
    cards: Vec<Card>,
    layers: Vec<Option<Layer>>,
}

fn raw<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl KdaCards {
    /// Open every gfx1100 card, load the code object on each, and place the KDA
    /// layers on them in order, an equal share each.
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
                    rows_qkv: k.kernel("rows_qkv"),
                    step: k.kernel("kda_step"),
                    x: alloc(H),
                    mixed: alloc(3 * KDA_QKV),
                    small: alloc(SMALL_ROWS),
                    f: alloc(KDA_QKV),
                    gate: alloc(KDA_QKV),
                    normed: alloc(KDA_QKV),
                    y: alloc(H),
                    k,
                }
            })
            .collect();
        let kda_layers: Vec<usize> = (0..NUM_HIDDEN_LAYERS).filter(|&l| LINEAR_ATTENTION[l]).collect();
        let per = kda_layers.len().div_ceil(n);
        let mut layers: Vec<Option<Layer>> = (0..NUM_HIDDEN_LAYERS).map(|_| None).collect();
        let mut bytes = 0usize;
        for (i, &l) in kda_layers.iter().enumerate() {
            let card = i / per;
            let k = &cards[card].k;
            let Attn::Kda(kw) = &w.layers[l].attn else { unreachable!() };
            let up = |b: &[u8]| -> Mem {
                let m = k.vram_visible(b.len());
                k.upload(&m, 0, b);
                m
            };
            let mut taps: Vec<u16> = Vec::with_capacity(g::TAPS);
            for part in kw.conv {
                taps.extend_from_slice(part);
            }
            let input_ln: Vec<u16> = w.layers[l].input_ln.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
            let zero_w = vec![0f32; KDA_WINDOW];
            let zero_m = vec![0f32; KDA_MEMORY];
            // q, k, v back to back, and the three small projections: one launch each
            let qkv = {
                let m = k.vram_visible(3 * kw.q.len() * 2);
                k.upload(&m, 0, raw(kw.q));
                k.upload(&m, kw.q.len() * 2, raw(kw.k));
                k.upload(&m, 2 * kw.q.len() * 2, raw(kw.v));
                m
            };
            let small = {
                let m = k.vram_visible(SMALL_ROWS * H * 2);
                k.upload(&m, 0, raw(kw.f_a));
                k.upload(&m, kw.f_a.len() * 2, raw(kw.g_a));
                k.upload(&m, 2 * kw.f_a.len() * 2, raw(kw.b));
                m
            };
            let lay = Layer {
                card,
                input_ln: up(raw(&input_ln)),
                qkv,
                o: up(raw(kw.o)),
                taps: up(raw(&taps)),
                small,
                f_b: up(raw(kw.f_b)),
                g_b: up(raw(kw.g_b)),
                dt_bias: up(raw(kw.dt_bias)),
                a_log: up(raw(kw.a_log)),
                o_norm: up(raw(kw.o_norm)),
                window: up(raw(&zero_w)),
                mem: up(raw(&zero_m)),
            };
            bytes += 2 * (3 * kw.q.len() + kw.o.len() + SMALL_ROWS * H + kw.f_b.len() + kw.g_b.len() + taps.len()) + 4 * (KDA_WINDOW + KDA_MEMORY);
            layers[l] = Some(lay);
        }
        println!("gpu      : {} KDA layers on {} card{} ({:.1} GB with their state) in {:.1}s", kda_layers.len(), n, if n == 1 { "" } else { "s" }, bytes as f64 / 1e9, t0.elapsed().as_secs_f32());
        KdaCards { cards, layers }
    }

    /// One rows launch on card `c`: `out[row] = W[row] · x` (normed with `ln` when given) for `m` rows of width `k_width`. Addresses are device VAs.
    fn rows(&self, c: &Card, kernel: &crate::kfd::Kernel, x: usize, ln: Option<usize>, w: usize, out: usize, m: usize, k_width: usize) {
        let mut a = Args::new();
        a.ptr(x);
        if let Some(ln) = ln {
            a.ptr(ln);
        }
        a.ptr(w).ptr(out).u32(m as u32).u32(1);
        let lds = if ln.is_some() { g::normed_rows_lds(k_width, 1) } else { g::rows_lds(k_width, 1) } as u32;
        c.k.launch(kernel, g::rows_grid(m), g::WG, lds, &a);
    }
}

impl KdaPlaced for KdaCards {
    fn placed(&self, l: usize) -> bool {
        self.layers[l].is_some()
    }
    fn step(&mut self, l: usize, x: &[f32], out: &mut [f32]) {
        let lay = self.layers[l].as_ref().unwrap();
        let c = &self.cards[lay.card];
        c.k.upload(&c.x, 0, raw(&x[..H]));
        // q | k | v as one launch, the gates' first halves and beta as another, all over the normed input
        let ln = Some(lay.input_ln.va);
        self.rows(c, &c.rows_h, c.x.va, ln, lay.qkv.va, c.mixed.va, 3 * KDA_QKV, H);
        self.rows(c, &c.rows_h, c.x.va, ln, lay.small.va, c.small.va, SMALL_ROWS, H);
        self.rows(c, &c.rows_d, c.small.va, None, lay.f_b.va, c.f.va, KDA_QKV, KDA_DIM);
        self.rows(c, &c.rows_d, c.small.at(KDA_DIM * 4), None, lay.g_b.va, c.gate.va, KDA_QKV, KDA_DIM);
        let mut a = Args::new();
        a.ptr(c.mixed.va).ptr(lay.taps.va).ptr(lay.window.va).ptr(c.f.va).ptr(lay.dt_bias.va).ptr(lay.a_log.va).ptr(c.small.at(2 * KDA_DIM * 4)).ptr(c.gate.va).ptr(lay.mem.va).ptr(lay.o_norm.va).ptr(c.normed.va).u32(1);
        c.k.launch(&c.step, g::STEP_GRID, g::WG, g::STEP_LDS as u32, &a);
        self.rows(c, &c.rows_qkv, c.normed.va, None, lay.o.va, c.y.va, H, KDA_QKV);
        c.k.sync();
        read(&c.y, 0, &mut out[..H]);
    }
    fn seed(&mut self, l: usize, window: &[f32], mem: &[f32]) {
        let lay = self.layers[l].as_ref().unwrap();
        let k = &self.cards[lay.card].k;
        k.upload(&lay.window, 0, raw(&window[..KDA_WINDOW]));
        k.upload(&lay.mem, 0, raw(&mem[..KDA_MEMORY]));
    }
    fn snapshot(&self, l: usize, window: &mut [f32], mem: &mut [f32]) {
        let lay = self.layers[l].as_ref().unwrap();
        read(&lay.window, 0, &mut window[..KDA_WINDOW]);
        read(&lay.mem, 0, &mut mem[..KDA_MEMORY]);
    }
}

const _: () = assert!(CONV_DIM == 3 * KDA_QKV && CONV_TAPS == 4);
