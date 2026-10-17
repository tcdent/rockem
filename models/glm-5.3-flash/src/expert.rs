//! One routed expert on its `q4g128` record: `out = down(silu(min(gate·x, L)) ⊙
//! clamp(up·x, ±L))`, L the model's `swiglu_limit`, with the activation
//! quantizer and the clamped SwiGLU fused in — f32 in, f32 out,
//! one signature. This is the unit the dispatcher issues per expert per
//! thread; the GPU realization lands here as a grouped kernel over the same
//! bytes.

use crate::layout::{self, DOWN_ITEMS, GU_CHUNK, GU_ITEMS, GU_S, GU_W};
use crate::{D_FF, H};
use common::q4::{self, QAct};
use common::swiglu::swiglu_clamped;

/// The expert stream as it lies in memory: its bytes, and where every item is. In the file
/// an item's place is `layout::gu_item` / `layout::down_item`; loaded, the items are packed
/// into one region per NUMA node in the proportion of the nodes' measured stream rates
/// (`layout::placement`), so the table is the one definition of where an item is and the
/// file layout never has to change with the box (tcdent, 2026-10-16: the observed ratios
/// place the experts so every node's share ends at once).
pub struct Stream<'a> {
    pub bytes: &'a [u8],
    /// Byte offset in `bytes` of every item, `[MOE_LAYERS][E][ITEMS]` (gate/up items, then down).
    /// Atomics (relaxed; free to read) so `migrate` can retrain the placement on a shared stream
    /// between runs — the caller sees to it that no run is in flight.
    pub addr: std::vec::Vec<core::sync::atomic::AtomicU64>,
    /// The node of every item, as placed.
    pub node: std::vec::Vec<core::sync::atomic::AtomicU8>,
    /// A placed stream's regions, one a node: where it starts in `bytes`, its capacity, and the
    /// bytes used — items moved to a node by `migrate` go at the end of its region.
    pub regions: std::sync::Mutex<std::vec::Vec<Region>>,
}
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub base: usize,
    pub cap: usize,
    pub used: usize,
}
pub const ITEMS: usize = GU_ITEMS + DOWN_ITEMS;
#[inline(always)]
pub const fn item_index(l: usize, e: usize, j: usize) -> usize {
    ((l - crate::FIRST_K_DENSE_REPLACE) * crate::E + e) * ITEMS + j
}
impl<'a> Stream<'a> {
    /// The file's own layout: an item where `layout` puts it.
    pub fn file(bytes: &'a [u8]) -> Self {
        assert_eq!(bytes.len(), layout::STREAM);
        let mut addr = std::vec::Vec::with_capacity(layout::MOE_LAYERS * crate::E * ITEMS);
        for l in crate::FIRST_K_DENSE_REPLACE..crate::NUM_HIDDEN_LAYERS {
            for e in 0..crate::E {
                for c in 0..GU_ITEMS {
                    addr.push(layout::gu_item(l, e, c).start as u64);
                }
                for c in 0..DOWN_ITEMS {
                    addr.push(layout::down_item(l, e, c).start as u64);
                }
            }
        }
        Self::placed(bytes, addr, std::vec![0; layout::MOE_LAYERS * crate::E * ITEMS], std::vec::Vec::new())
    }
    /// A placed copy: the loader's regions, with the table it filled.
    pub fn placed(bytes: &'a [u8], addr: std::vec::Vec<u64>, node: std::vec::Vec<u8>, regions: std::vec::Vec<Region>) -> Self {
        use core::sync::atomic::{AtomicU64, AtomicU8};
        assert_eq!(addr.len(), layout::MOE_LAYERS * crate::E * ITEMS);
        assert_eq!(node.len(), addr.len());
        Stream { bytes, addr: addr.into_iter().map(AtomicU64::new).collect(), node: node.into_iter().map(AtomicU8::new).collect(), regions: std::sync::Mutex::new(regions) }
    }
    /// How many regions (nodes) the stream is placed over; 0 for the file's own layout.
    pub fn nodes(&self) -> usize {
        self.regions.lock().unwrap().len()
    }
    /// The node of every item, a copy.
    pub fn node_of_items(&self) -> std::vec::Vec<u8> {
        self.node.iter().map(|n| n.load(core::sync::atomic::Ordering::Relaxed)).collect()
    }
    #[inline(always)]
    pub const fn item_len(j: usize) -> usize {
        if j < GU_ITEMS { layout::GU_ITEM } else { layout::DOWN_ITEM }
    }
    /// Re-place the items: every item whose node differs in `node` is copied to the end of its new
    /// node's region and the table points there (the old copy stays, dead). Returns (items moved,
    /// bytes), or the node whose region has no room — then nothing is changed. No run may be in
    /// flight: the table changes under the readers' feet.
    pub fn migrate(&self, node: &[u8]) -> Result<(usize, usize), usize> {
        use core::sync::atomic::Ordering::Relaxed;
        assert_eq!(node.len(), self.node.len());
        let base = self.bytes.as_ptr() as usize;
        let mut regions = self.regions.lock().unwrap();
        let nodes = regions.len();
        // the moves by destination node, then one thread a destination copies them in order
        let mut moves: std::vec::Vec<std::vec::Vec<usize>> = std::vec![std::vec::Vec::new(); nodes];
        for i in 0..node.len() {
            if node[i] != self.node[i].load(Relaxed) {
                moves[node[i] as usize].push(i);
            }
        }
        for (d, list) in moves.iter().enumerate() {
            let need: usize = list.iter().map(|&i| Self::item_len(i % ITEMS)).sum();
            if regions[d].used + need > regions[d].cap {
                return Err(d);
            }
        }
        let (mut n_moved, mut b_moved) = (0, 0);
        let mut dst_of = std::vec![0u64; node.len()];
        for (d, list) in moves.iter().enumerate() {
            let r = &mut regions[d];
            for &i in list {
                let len = Self::item_len(i % ITEMS);
                dst_of[i] = (r.base + r.used) as u64;
                r.used += len;
                n_moved += 1;
                b_moved += len;
            }
        }
        std::thread::scope(|sc| {
            for list in &moves {
                let (addr, dst_of) = (&self.addr, &dst_of);
                sc.spawn(move || {
                    for &i in list {
                        let len = Self::item_len(i % ITEMS);
                        unsafe { core::ptr::copy_nonoverlapping((base + addr[i].load(Relaxed) as usize) as *const u8, (base + dst_of[i] as usize) as *mut u8, len) };
                    }
                });
            }
        });
        for list in &moves {
            for &i in list {
                self.addr[i].store(dst_of[i], Relaxed);
            }
        }
        for (i, &n) in node.iter().enumerate() {
            self.node[i].store(n, Relaxed);
        }
        Ok((n_moved, b_moved))
    }
}

/// The compute's unit is a part of a file item: `SUB` parts a gate/up item (`GU_ROWS` hidden rows
/// each) and a down item (`DOWN_ROWS` output rows). A file item is placed and prefetched whole;
/// its parts are what a core streams at a time — halving the grain of a run's end (a core's
/// last part) and of an expert's split across the nodes (2026-10-16). A part's rows and scales
/// are contiguous inside the item (whole `q4` tiles), so this is addressing, not a copy.
pub const SUB: usize = 2;
pub const GU_ROWS: usize = GU_CHUNK / SUB;
pub const GU_PARTS: usize = GU_ITEMS * SUB;
pub const DOWN_ROWS: usize = layout::DOWN_CHUNK / SUB;
pub const DOWN_PARTS: usize = DOWN_ITEMS * SUB;
const _: () = assert!(GU_ROWS % common::q4::TILE == 0 && DOWN_ROWS % common::q4::TILE == 0);

/// One expert of the stream, read by item (`layout`): the unit a core streams.
pub struct Weights<'a> {
    stream: &'a Stream<'a>,
    l: usize,
    e: usize,
}

fn f32s(b: &[u8]) -> &[f32] {
    let (p, m, q) = unsafe { b.align_to::<f32>() };
    assert!(p.is_empty() && q.is_empty());
    m
}

impl<'a> Weights<'a> {
    /// Expert `e` of MoE layer `l` in `stream`.
    #[inline(always)]
    pub fn at(stream: &'a Stream<'a>, l: usize, e: usize) -> Self {
        debug_assert_eq!(stream.bytes.as_ptr() as usize % 4, 0);
        Weights { stream, l, e }
    }
    /// Gate/up part `c` (of `GU_PARTS`): (gate rows, up rows, gate scales, up scales) — each `q4` slices over `GU_ROWS` rows of width `H`.
    #[inline(always)]
    pub fn gu(&self, c: usize) -> (&'a [u8], &'a [u8], &'a [f32], &'a [f32]) {
        let it = self.gu_bytes(c / SUB);
        let (h, w, sc) = (c % SUB, GU_W / SUB, 4 * GU_S / SUB);
        (&it[h * w..(h + 1) * w], &it[GU_W + h * w..GU_W + (h + 1) * w], f32s(&it[2 * GU_W + h * sc..2 * GU_W + (h + 1) * sc]), f32s(&it[2 * GU_W + 4 * GU_S + h * sc..2 * GU_W + 4 * GU_S + (h + 1) * sc]))
    }
    /// Down part `c` (of `DOWN_PARTS`): (rows, scales) — `q4` slices over `DOWN_ROWS` rows of width `D_FF`.
    #[inline(always)]
    pub fn down(&self, c: usize) -> (&'a [u8], &'a [f32]) {
        let it = self.down_bytes(c / SUB);
        let (h, w, sc) = (c % SUB, layout::DOWN_W / SUB, 4 * layout::DOWN_S / SUB);
        (&it[h * w..(h + 1) * w], f32s(&it[layout::DOWN_W + h * sc..layout::DOWN_W + (h + 1) * sc]))
    }
    /// The item's bytes, for a core to pull through its caches.
    #[inline(always)]
    pub fn gu_bytes(&self, c: usize) -> &'a [u8] {
        let a = self.stream.addr[item_index(self.l, self.e, c)].load(core::sync::atomic::Ordering::Relaxed) as usize;
        &self.stream.bytes[a..a + layout::GU_ITEM]
    }
    #[inline(always)]
    pub fn down_bytes(&self, c: usize) -> &'a [u8] {
        let a = self.stream.addr[item_index(self.l, self.e, GU_ITEMS + c)].load(core::sync::atomic::Ordering::Relaxed) as usize;
        &self.stream.bytes[a..a + layout::DOWN_ITEM]
    }
}

/// An expert's items from whole `q4::pack`ed matrices — gate|up (`[2·D_FF][H]`, gate rows first)
/// and down (`[H][D_FF]`) — as (offset in the stream, bytes), the bundle builder's and the
/// relayout's one definition of the bytes.
#[cfg(any(feature = "std", test))]
pub fn pack_items(l: usize, e: usize, guw: &[u8], gus: &[f32], dw: &[u8], dsc: &[f32]) -> std::vec::Vec<(usize, std::vec::Vec<u8>)> {
    use common::q4::{GROUP, TILE};
    use layout::{DOWN_CHUNK, DOWN_W};
    assert_eq!(guw.len(), 2 * D_FF * H / 2);
    assert_eq!(gus.len(), 2 * D_FF * (H / GROUP));
    assert_eq!(dw.len(), H * D_FF / 2);
    assert_eq!(dsc.len(), H * (D_FF / GROUP));
    let le = |v: &[f32]| -> std::vec::Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut items = std::vec::Vec::with_capacity(GU_ITEMS + DOWN_ITEMS);
    for c in 0..GU_ITEMS {
        let (g0, u0) = (c * GU_CHUNK, D_FF + c * GU_CHUNK);
        let rows = |r0: usize| &guw[r0 / TILE * 2 * H..(r0 + GU_CHUNK) / TILE * 2 * H];
        let scales = |r0: usize| &gus[r0 / TILE * (H / GROUP) * TILE..(r0 + GU_CHUNK) / TILE * (H / GROUP) * TILE];
        let mut it = std::vec::Vec::with_capacity(layout::GU_ITEM);
        it.extend_from_slice(rows(g0));
        it.extend_from_slice(rows(u0));
        it.extend_from_slice(&le(scales(g0)));
        it.extend_from_slice(&le(scales(u0)));
        assert_eq!(it.len(), layout::GU_ITEM);
        items.push((layout::gu_item(l, e, c).start, it));
    }
    for c in 0..DOWN_ITEMS {
        let r0 = c * DOWN_CHUNK;
        let mut it = std::vec::Vec::with_capacity(layout::DOWN_ITEM);
        it.extend_from_slice(&dw[r0 / TILE * 2 * D_FF..(r0 + DOWN_CHUNK) / TILE * 2 * D_FF]);
        it.extend_from_slice(&le(&dsc[r0 / TILE * (D_FF / GROUP) * TILE..(r0 + DOWN_CHUNK) / TILE * (D_FF / GROUP) * TILE]));
        assert_eq!(it.len(), DOWN_W + 4 * layout::DOWN_S);
        items.push((layout::down_item(l, e, c).start, it));
    }
    items
}

/// Per-thread scratch, sized once at construction.
pub struct Scratch<'a> {
    pub xa: QAct<'a>,      // K = H
    pub gu: &'a mut [f32], // 2·D_FF
    pub h: &'a mut [f32],  // D_FF
    pub ha: QAct<'a>,      // K = D_FF
}

/// The owning twin of `Scratch`: allocated once, in one place per binary,
/// and borrowed as a `Scratch` per call. The arena replaces this by
/// substitution.
#[cfg(any(feature = "std", test))]
pub struct ScratchBuf {
    xq: Vec<i8>,
    xs: Vec<f32>,
    xg: Vec<i32>,
    gu: Vec<f32>,
    h: Vec<f32>,
    hq: Vec<i8>,
    hs: Vec<f32>,
    hg: Vec<i32>,
}

#[cfg(any(feature = "std", test))]
impl ScratchBuf {
    pub fn new() -> Self {
        use common::q4::GROUP;
        ScratchBuf {
            xq: vec![0; H],
            xs: vec![0.0; H / GROUP],
            xg: vec![0; H / GROUP],
            gu: vec![0.0; 2 * D_FF],
            h: vec![0.0; D_FF],
            hq: vec![0; D_FF],
            hs: vec![0.0; D_FF / GROUP],
            hg: vec![0; D_FF / GROUP],
        }
    }
    pub fn scratch(&mut self) -> Scratch<'_> {
        Scratch {
            xa: QAct { q: &mut self.xq, scale: &mut self.xs, gsum: &mut self.xg },
            gu: &mut self.gu,
            h: &mut self.h,
            ha: QAct { q: &mut self.hq, scale: &mut self.hs, gsum: &mut self.hg },
        }
    }
}
#[cfg(any(feature = "std", test))]
impl Default for ScratchBuf {
    fn default() -> Self {
        Self::new()
    }
}

/// The realization this target ships.
#[inline(always)]
pub fn forward(w: &Weights, x: &[f32], s: &mut Scratch, out: &mut [f32]) {
    s.xa.quantize(&x[..H]);
    let xa = s.xa.act();
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4::gemv::<H>(GU_ROWS, gw, gs, &xa, &mut s.gu[c * GU_ROWS..(c + 1) * GU_ROWS]);
        q4::gemv::<H>(GU_ROWS, uw, us, &xa, &mut s.gu[D_FF + c * GU_ROWS..D_FF + (c + 1) * GU_ROWS]);
    }
    swiglu_clamped::<D_FF>(s.gu, crate::SWIGLU_LIMIT, s.h);
    s.ha.quantize(s.h);
    let ha = s.ha.act();
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4::gemv::<D_FF>(DOWN_ROWS, dw, ds, &ha, &mut out[c * DOWN_ROWS..(c + 1) * DOWN_ROWS]);
    }
}

/// The same from an activation already quantized — the card's integers,
/// mirrored to the host, so both tiers' experts see the same `xq`: `xq` is
/// `[H]`, `xs` and `xg` `[H / GROUP]`.
#[inline(always)]
pub fn forward_q(w: &Weights, xq: &[i8], xs: &[f32], xg: &[i32], s: &mut Scratch, out: &mut [f32]) {
    s.xa.q.copy_from_slice(&xq[..H]);
    s.xa.scale.copy_from_slice(&xs[..H / q4::GROUP]);
    s.xa.gsum.copy_from_slice(&xg[..H / q4::GROUP]);
    let xa = s.xa.act();
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4::gemv::<H>(GU_ROWS, gw, gs, &xa, &mut s.gu[c * GU_ROWS..(c + 1) * GU_ROWS]);
        q4::gemv::<H>(GU_ROWS, uw, us, &xa, &mut s.gu[D_FF + c * GU_ROWS..D_FF + (c + 1) * GU_ROWS]);
    }
    swiglu_clamped::<D_FF>(s.gu, crate::SWIGLU_LIMIT, s.h);
    s.ha.quantize(s.h);
    let ha = s.ha.act();
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4::gemv::<D_FF>(DOWN_ROWS, dw, ds, &ha, &mut out[c * DOWN_ROWS..(c + 1) * DOWN_ROWS]);
    }
}

/// Same composition over the scalar inner loop, to diff against.
pub fn forward_scalar(w: &Weights, x: &[f32], s: &mut Scratch, out: &mut [f32]) {
    s.xa.quantize(&x[..H]);
    let xa = s.xa.act();
    for c in 0..GU_PARTS {
        let (gw, uw, gs, us) = w.gu(c);
        q4::gemv_scalar(H, GU_ROWS, gw, gs, &xa, &mut s.gu[c * GU_ROWS..(c + 1) * GU_ROWS]);
        q4::gemv_scalar(H, GU_ROWS, uw, us, &xa, &mut s.gu[D_FF + c * GU_ROWS..D_FF + (c + 1) * GU_ROWS]);
    }
    swiglu_clamped::<D_FF>(s.gu, crate::SWIGLU_LIMIT, s.h);
    s.ha.quantize(s.h);
    let ha = s.ha.act();
    for c in 0..DOWN_PARTS {
        let (dw, ds) = w.down(c);
        q4::gemv_scalar(D_FF, DOWN_ROWS, dw, ds, &ha, &mut out[c * DOWN_ROWS..(c + 1) * DOWN_ROWS]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::q4::GROUP;

    /// A random record: every realization on the same bytes and input must
    /// agree with the scalar composition to summation-order noise.
    #[test]
    fn realizations_agree_on_same_bytes() {
        let mut rnd = common::testing::Rng::new(0x9E3779B97F4A7C15);
        let mut rnd = || rnd.next();
        let mk = |m: usize, k: usize, rnd: &mut dyn FnMut() -> u64| {
            let w: Vec<i8> = (0..m * k).map(|_| (rnd() % 16) as i8 - 8).collect();
            let s: Vec<f32> = (0..m * (k / GROUP)).map(|_| 0.01 + (rnd() % 100) as f32 * 1e-4).collect();
            q4::pack(m, k, &w, &s)
        };
        let (guw, gus) = mk(2 * D_FF, H, &mut rnd);
        let (dw, ds) = mk(H, D_FF, &mut rnd);
        // one block's worth of stream, the expert at (FIRST_K_DENSE_REPLACE, 0): its items land inside it
        let mut block = vec![0u32; layout::XBLOCK_BYTES / 4]; // 4-aligned
        let stream: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, layout::XBLOCK_BYTES) };
        let (l, e) = (crate::FIRST_K_DENSE_REPLACE, 0);
        for (off, it) in pack_items(l, e, &guw, &gus, &dw, &ds) {
            stream[off..off + it.len()].copy_from_slice(&it);
        }
        let w = Weights { stream, l, e };
        let x: Vec<f32> = (0..H).map(|_| ((rnd() % 2000) as f32 - 1000.0) * 1e-3).collect();

        let run = |f: &dyn Fn(&Weights, &[f32], &mut Scratch, &mut [f32])| {
            let mut buf = ScratchBuf::new();
            let mut out = vec![0f32; H];
            f(&w, &x, &mut buf.scratch(), &mut out);
            out
        };
        let a = run(&forward_scalar);
        let b = run(&forward);
        assert_eq!(a, b, "realizations must be bit-identical on the same bytes");
        assert!(a.iter().any(|v| *v != 0.0));
    }
}

/// Launch geometry of the gfx11 realization, visible to hosts that launch it.
pub mod gfx11_consts {
    use crate::{D_FF, H, N_ROUTED_EXPERTS, TOKENS};
    use common::q4::TILE;
    pub use crate::N_ROUTED_EXPERTS as EXPERTS;
    pub const WG: u32 = 256;
    pub const UP_TILES: usize = 2 * D_FF / TILE;
    pub const DOWN_TILES: usize = H / TILE;
    /// Workgroups per expert (eight 4-row tiles each), and the grid over every expert.
    pub const UP_WGS: usize = UP_TILES / 8;
    pub const DOWN_WGS: usize = DOWN_TILES / 8;
    /// Bytes of one staged step (two groups) of one token's quantized activation (`q4::gfx11::STEP_BYTES`).
    pub const STEP_BYTES: usize = 2 * common::q4::GROUP + 16;
    /// LDS an expert kernel over `K`-wide activations needs for a step of `n`
    /// tokens (as `ffn::gfx11_consts::rows_lds`, in quantized steps).
    pub const fn experts_lds(k: usize, n: usize) -> usize {
        let steps = (k / common::q4::GROUP).div_ceil(2);
        let flight = common::gemv::FLIGHT;
        let slots = if n <= 1 { if steps < flight { steps } else { flight } } else { TOKENS };
        slots * STEP_BYTES
    }
    /// Workgroups for a step of `n` tokens: the experts that can be active, at most all.
    pub const fn up_grid(n: usize) -> u32 {
        let e = if n * crate::TOP_K < N_ROUTED_EXPERTS { n * crate::TOP_K } else { N_ROUTED_EXPERTS };
        (e * UP_WGS) as u32
    }
    pub const fn down_grid(n: usize) -> u32 {
        let e = if n * crate::TOP_K < N_ROUTED_EXPERTS { n * crate::TOP_K } else { N_ROUTED_EXPERTS };
        (e * DOWN_WGS) as u32
    }
    // a workgroup's eight tiles must belong to one expert for the shared staging
    const _: () = assert!(UP_TILES % 8 == 0 && DOWN_TILES % 8 == 0);
}

// The GPU realization of the experts is not this copy of V2-Lite's: GLM-5.3-Flash's
// 288 experts, eight a token, get their own kernel when experts move to the card.
