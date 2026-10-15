//! One routed expert on its `q4g128` record: `out = down(silu(gate·x) ⊙ (up·x))`
//! with the activation quantizer and the SwiGLU fused in — f32 in, f32 out,
//! one signature. This is the unit the dispatcher issues per expert per
//! thread; the GPU realization lands here as a grouped kernel over the same
//! bytes.

use crate::layout::{DOWN_W, GATE_UP_S, GATE_UP_W, RECORD};
use crate::{D_FF, H};
use common::q4::{self, QAct};
use common::swiglu::swiglu;

pub struct Weights<'a> {
    pub gate_up_w: &'a [u8],
    pub gate_up_s: &'a [f32],
    pub down_w: &'a [u8],
    pub down_s: &'a [f32],
}

impl<'a> Weights<'a> {
    /// Split one record (4-byte aligned) into its four streams.
    pub fn from_record(rec: &'a [u8]) -> Self {
        assert_eq!(rec.len(), RECORD);
        assert_eq!(rec.as_ptr() as usize % 4, 0);
        let f32s = |b: &'a [u8]| -> &'a [f32] {
            let (p, m, q) = unsafe { b.align_to::<f32>() };
            assert!(p.is_empty() && q.is_empty());
            m
        };
        let (a, rest) = rec.split_at(GATE_UP_W);
        let (b, rest) = rest.split_at(4 * GATE_UP_S);
        let (c, d) = rest.split_at(DOWN_W);
        Weights { gate_up_w: a, gate_up_s: f32s(b), down_w: c, down_s: f32s(d) }
    }
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
    q4::gemv::<H>(2 * D_FF, w.gate_up_w, w.gate_up_s, &s.xa.act(), s.gu);
    swiglu::<D_FF>(s.gu, s.h);
    s.ha.quantize(s.h);
    q4::gemv::<D_FF>(H, w.down_w, w.down_s, &s.ha.act(), out);
}

/// The same from an activation already quantized — the card's integers,
/// mirrored to the host, so both tiers' experts see the same `xq`: `xq` is
/// `[H]`, `xs` and `xg` `[H / GROUP]`.
#[inline(always)]
pub fn forward_q(w: &Weights, xq: &[i8], xs: &[f32], xg: &[i32], s: &mut Scratch, out: &mut [f32]) {
    s.xa.q.copy_from_slice(&xq[..H]);
    s.xa.scale.copy_from_slice(&xs[..H / q4::GROUP]);
    s.xa.gsum.copy_from_slice(&xg[..H / q4::GROUP]);
    q4::gemv::<H>(2 * D_FF, w.gate_up_w, w.gate_up_s, &s.xa.act(), s.gu);
    swiglu::<D_FF>(s.gu, s.h);
    s.ha.quantize(s.h);
    q4::gemv::<D_FF>(H, w.down_w, w.down_s, &s.ha.act(), out);
}

/// Same composition over the scalar inner loop, to diff against.
pub fn forward_scalar(w: &Weights, x: &[f32], s: &mut Scratch, out: &mut [f32]) {
    s.xa.quantize(&x[..H]);
    q4::gemv_scalar(H, 2 * D_FF, w.gate_up_w, w.gate_up_s, &s.xa.act(), s.gu);
    swiglu::<D_FF>(s.gu, s.h);
    s.ha.quantize(s.h);
    q4::gemv_scalar(D_FF, H, w.down_w, w.down_s, &s.ha.act(), out);
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
        let mut rec = Vec::with_capacity(RECORD);
        rec.extend_from_slice(&guw);
        rec.extend(gus.iter().flat_map(|v| v.to_le_bytes()));
        rec.extend_from_slice(&dw);
        rec.extend(ds.iter().flat_map(|v| v.to_le_bytes()));
        let rec: Vec<u32> = rec.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(); // 4-aligned
        let rec: &[u8] = unsafe { core::slice::from_raw_parts(rec.as_ptr() as *const u8, RECORD) };
        let w = Weights::from_record(rec);
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

#[cfg(target_arch = "amdgpu")]
pub mod gfx11 {
    //! The GPU realization, grouped by expert over the tokens of a step, two
    //! launches split only at the grid-wide dependency: `experts_up` computes
    //! the fused gate/up projection for every (token, slot) an expert serves
    //! (`gu[token][slot][2·D_FF]`); `experts_down` applies SwiGLU and
    //! quantizes in its staging (per step of two groups, into LDS) and runs
    //! the down projection (`out[token][slot][H]`). Every expert's record is
    //! read once a step, however many tokens it serves; the tokens' quantized
    //! activations are staged a step at a time and each (row, token) is the
    //! same integer sums and the same flush in the same group order as the
    //! CPU's, so the output bytes match the CPU realization's whether one
    //! token stepped or thirty-two. A workgroup is eight 4-row tiles of one
    //! expert (8 waves); experts that serve no token this step return at once.
    use super::gfx11_consts::*;
    use super::*;
    use crate::layout::RECORD;
    use crate::{N_ROUTED_EXPERTS, TOKENS, TOP_K};
    use common::gemv::gfx11::{round, PASS};
    use common::q4::gfx11::gemv_tile_step;
    use common::q4::{GROUP, TILE};
    use common::swiglu::silu;
    use core::arch::amdgpu::{wave_reduce_add, wave_reduce_umax, workgroup_id_x, workitem_id_x};
    const _: () = assert!(TOKENS == common::q4::gfx11::TOKENS && STEP_BYTES == common::q4::gfx11::STEP_BYTES && GROUP == 128);

    /// `gu[tok][slot][2·D_FF] = [gate; up] · xq[tok]` for every (tok, slot) of
    /// each expert (`count`, `list` from `group`); `xq`/`xs`/`xg` are
    /// `[n][H]`, `[n][H/GROUP]` ×2, from `route`. Grid: up_grid(n) × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn experts_up(stream: *const u8, base: u64, count: *const u32, list: *const u32, xq: *const i8, xs: *const f32, xg: *const i32, gu: *mut f32) {
        unsafe {
            let wg = workgroup_id_x() as usize;
            let (j, blk) = (wg / UP_WGS, wg % UP_WGS);
            if j >= *count.add(N_ROUTED_EXPERTS) as usize {
                return;
            }
            let e = *count.add(N_ROUTED_EXPERTS + 1 + j) as usize;
            let n = *count.add(e) as usize;
            let tid = workitem_id_x();
            let (wave, lane) = ((tid / 32) as usize, tid % 32);
            let tile = blk * 8 + wave;
            let rec = stream.add(base as usize + e * RECORD);
            let ws = rec.add(GATE_UP_W) as *const f32;
            let slab: *mut u8 = common::gfx11::lds();
            let list = list.add(e * TOKENS);
            let (w_tile, ws_tile) = (rec.add(tile * H * 2), ws.add(tile * (H / GROUP) * TILE));
            let mut p0 = 0;
            while p0 < n {
                let np = if n - p0 < PASS { n - p0 } else { PASS };
                // a round's token-steps are 32 chunks of 8 bytes each (plus two scales and Σq):
                // the 256 threads take the chunks round-robin, so one token's eight steps are one copy a thread
                // laid out `[r][stride][STEP_BYTES]`, stride the pass width, tokens past `np` unstaged
                let stage = |s0: usize, r: usize, stride: usize, dst: *mut u8| {
                    let t = tid as usize;
                    let chunks = np * r * 32;
                    let mut c = t;
                    while c < chunks {
                        let (ts, part) = (c / 32, c % 32);
                        let (k, i) = (ts / np, ts % np);
                        let tok = (*list.add(p0 + i) >> 3) as usize;
                        stage_one::<H>(part, xq.add(tok * H), xs.add(tok * (H / GROUP)), xg.add(tok * (H / GROUP)), s0 + k, dst.add((k * stride + i) * STEP_BYTES));
                        c += 256;
                    }
                };
                // widths 1, 2, 4, 8: an expert's list is a few tokens at thirty-two a step, and
                // a width computes its padding (the rows kernels' passes are full or single)
                let mut y = [0f32; PASS];
                match np {
                    1 => tile_pass::<H, 1>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    2 => tile_pass::<H, 2>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    3 | 4 => tile_pass::<H, 4>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    _ => tile_pass::<H, PASS>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                };
                if lane < 16 && lane % 4 == 0 {
                    let rr = (lane as usize % 16) / 4;
                    let mut i = 0;
                    while i < PASS {
                        if i < np {
                            let ent = *list.add(p0 + i) as usize;
                            *gu.add(((ent >> 3) * TOP_K + (ent & 7)) * 2 * D_FF + tile * TILE + rr) = y[i];
                        }
                        i += 1;
                    }
                }
                p0 += PASS;
            }
        }
    }

    /// One pass at width `NP` over the wave's tile: rounds of steps, each
    /// staged whole by `stage(s0, r, NP, slab)` — the round's steps
    /// `s0..s0 + r` laid out `[r][NP][STEP_BYTES]` — then the tile's sums per
    /// token. Full rounds carry no guards; one guarded round takes the tail.
    #[inline(always)]
    unsafe fn tile_pass<const K: usize, const NP: usize>(lane: u32, w_tile: *const u8, ws_tile: *const f32, slab: *mut u8, stage: &impl Fn(usize, usize, usize, *mut u8), y: &mut [f32; PASS]) {
        let steps = (K / GROUP).div_ceil(2);
        let r = round(NP, steps);
        let mut s0 = 0;
        while s0 + r <= steps {
            stage(s0, r, NP, slab);
            common::gfx11::workgroup_sync();
            let mut k = 0;
            while k < r {
                gemv_tile_step::<K, NP>(lane, w_tile, ws_tile, s0 + k, slab.add(k * NP * STEP_BYTES), y);
                k += 1;
            }
            common::gfx11::workgroup_sync();
            s0 += r;
        }
        let tail = steps - s0;
        if tail > 0 {
            stage(s0, tail, NP, slab);
            common::gfx11::workgroup_sync();
            let mut k = 0;
            while k < tail {
                gemv_tile_step::<K, NP>(lane, w_tile, ws_tile, s0 + k, slab.add(k * NP * STEP_BYTES), y);
                k += 1;
            }
            common::gfx11::workgroup_sync();
        }
    }

    /// One token's step `st` (two groups) of a quantized activation into `dst` in
    /// `q4::gfx11` step layout, by the 32 threads `part`.
    #[inline(always)]
    unsafe fn stage_one<const K: usize>(part: usize, q: *const i8, ascale: *const f32, gsum: *const i32, st: usize, dst: *mut u8) {
        let ng = K / GROUP;
        let g = 2 * st;
        let b = g * GROUP + part * 8;
        if b < K {
            core::ptr::write_unaligned(dst.add(part * 8) as *mut u64, core::ptr::read_unaligned(q.add(b) as *const u64));
        }
        if part < 2 && g + part < ng {
            *(dst.add(2 * GROUP) as *mut f32).add(part) = *ascale.add(g + part);
            *(dst.add(2 * GROUP + 8) as *mut i32).add(part) = *gsum.add(g + part);
        }
    }

    /// `out[tok][slot][H] = down · quant(silu(gate) ⊙ up)` for every (tok, slot)
    /// of each expert. Each group of a round is quantized as `QAct::quantize`
    /// does — absmax/127, round-half-away, Σq — by a wave, the round's
    /// (token, step, group) triples taken round-robin by the eight waves.
    /// Grid: down_grid(n) × 256.
    #[no_mangle]
    pub extern "gpu-kernel" fn experts_down(stream: *const u8, base: u64, count: *const u32, list: *const u32, gu: *const f32, out: *mut f32) {
        unsafe {
            let wg = workgroup_id_x() as usize;
            let (j, blk) = (wg / DOWN_WGS, wg % DOWN_WGS);
            if j >= *count.add(N_ROUTED_EXPERTS) as usize {
                return;
            }
            let e = *count.add(N_ROUTED_EXPERTS + 1 + j) as usize;
            let n = *count.add(e) as usize;
            let tid = workitem_id_x();
            let (wave, lane) = ((tid / 32) as usize, tid % 32);
            let tile = blk * 8 + wave;
            let w = stream.add(base as usize + e * RECORD).add(GATE_UP_W + 4 * GATE_UP_S);
            let ws = w.add(DOWN_W) as *const f32;
            let slab: *mut u8 = common::gfx11::lds();
            let list = list.add(e * TOKENS);
            let ng = D_FF / GROUP;
            let (w_tile, ws_tile) = (w.add(tile * D_FF * 2), ws.add(tile * (D_FF / GROUP) * TILE));
            let mut p0 = 0;
            while p0 < n {
                let np = if n - p0 < PASS { n - p0 } else { PASS };
                let stage = |s0: usize, r: usize, stride: usize, dst: *mut u8| {
                    // the round's groups: r steps of two, the last step of the matrix one
                    let g0 = 2 * s0;
                    let g1 = if 2 * (s0 + r) < ng { 2 * (s0 + r) } else { ng };
                    let groups = g1 - g0;
                    let mut trip = wave;
                    while trip < groups * np {
                        let (i, gi) = (trip / groups, trip % groups);
                        let (k, gp) = (gi / 2, gi % 2);
                        let ent = *list.add(p0 + i) as usize;
                        let g = gu.add(((ent >> 3) * TOP_K + (ent & 7)) * 2 * D_FF);
                        let k0 = (g0 + gi) * GROUP + lane as usize * 4;
                        let mut v = [0f32; 4];
                        let mut amax = 0f32;
                        let mut jj = 0;
                        while jj < 4 {
                            v[jj] = silu(*g.add(k0 + jj)) * *g.add(D_FF + k0 + jj);
                            let a = if v[jj] < 0.0 { -v[jj] } else { v[jj] };
                            if a > amax {
                                amax = a;
                            }
                            jj += 1;
                        }
                        let amax = f32::from_bits(wave_reduce_umax::<0>(amax.to_bits()));
                        let s = amax / 127.0;
                        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                        let mut packed = 0u32;
                        let mut sum = 0i32;
                        let mut jj = 0;
                        while jj < 4 {
                            let rd = common::math::round(v[jj] * inv);
                            let rd = if rd > 127.0 { 127.0 } else if rd < -127.0 { -127.0 } else { rd };
                            let q = rd as i8;
                            packed |= ((q as u8) as u32) << (8 * jj);
                            sum += q as i32;
                            jj += 1;
                        }
                        let sum = wave_reduce_add::<0>(sum as u32) as i32;
                        let d = dst.add((k * stride + i) * STEP_BYTES);
                        *(d.add(gp * GROUP + lane as usize * 4) as *mut u32) = packed;
                        if lane == 0 {
                            *(d.add(2 * GROUP) as *mut f32).add(gp) = s;
                            *(d.add(2 * GROUP + 8) as *mut i32).add(gp) = sum;
                        }
                        trip += 8;
                    }
                };
                let mut y = [0f32; PASS];
                match np {
                    1 => tile_pass::<D_FF, 1>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    2 => tile_pass::<D_FF, 2>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    3 | 4 => tile_pass::<D_FF, 4>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                    _ => tile_pass::<D_FF, PASS>(lane, w_tile, ws_tile, slab, &stage, &mut y),
                };
                if lane < 16 && lane % 4 == 0 {
                    let rr = (lane as usize % 16) / 4;
                    let mut i = 0;
                    while i < PASS {
                        if i < np {
                            let ent = *list.add(p0 + i) as usize;
                            *out.add(((ent >> 3) * TOP_K + (ent & 7)) * H + tile * TILE + rr) = y[i];
                        }
                        i += 1;
                    }
                }
                p0 += PASS;
            }
        }
    }
    const _: () = assert!(H % GROUP == 0 && D_FF % GROUP == 0 && TOP_K < 8);
}
