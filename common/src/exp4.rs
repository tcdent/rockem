//! A lossless 12-bit form of a bf16 matrix (029): the weights' exponents carry ~2.6 bits of information (the KDA
//! projections, 2026-10-25), so each weight keeps its sign and mantissa as a byte and its exponent as a 4-bit offset
//! below its group's largest. A rows kernel rebuilds the exact bf16 in a register (`super::gemv::gfx11::Exp4`), so
//! VRAM streams 12.25 bits a weight instead of 16 and the products are the bf16 kernel's, bit for bit.
//!
//! A `[m][K]` matrix in groups of `GROUP` along k, three planes:
//! - `chunk` `[m][K / 8][12]` u8: each run of eight weights (a rows kernel lane's share of a step) as its eight
//!   sign-and-mantissa bytes — the weight's sign in bit 7, its seven mantissa bits below (the bf16's bits 15 and 0–6) —
//!   then its eight 4-bit codes in four bytes, weight j's in nibble j: its exponent is the group's base plus the code.
//!   Twelve bytes in one place are one 96-bit load a lane (three narrow loads a step ran the kernel load-bound,
//!   bench-exp4 2026-10-25);
//! - `head` `[m][K / GROUP]` u32: the group's base exponent in bits 0–7 — its largest exponent less 15, or 0 — and in
//!   bits 8–31 one plus its slot in `raw` when the group is stored whole (0 otherwise);
//! - `raw` `[slots][GROUP]` u16: the bf16 of every group holding an exponent below its base — 0.68 % of the KDA
//!   projections' groups (a weight more than 15 binades below its group's largest, or an exponent-0 weight in a group
//!   whose base is past 0: 14,716 of 4.7 billion weights), read as bf16 by the lanes of that group.
//!
//! No code is set aside for zeros: they are that rare, and without one the decode is an add with no condition, two
//! weights at a time in one register (`super::gemv::gfx11::exp4_row_step`).
#[cfg(any(test, feature = "std"))]
use std::vec;
#[cfg(any(test, feature = "std"))]
use std::vec::Vec;

pub const GROUP: usize = 128;
/// Binades a group's codes span above its base.
pub const SPAN: u8 = 15;

/// Bytes a run of eight weights takes in `chunk`.
pub const CHUNK: usize = 12;
/// Bytes of the three planes of a `[m][k]` matrix with `slots` raw groups.
pub const fn bytes(m: usize, k: usize, slots: usize) -> usize {
    m * (k / 8) * CHUNK + 4 * m * (k / GROUP) + 2 * GROUP * slots
}

/// A matrix in the three planes, and its raw groups.
#[cfg(any(test, feature = "std"))]
pub struct Exp4 {
    pub chunk: Vec<u8>,
    pub head: Vec<u32>,
    pub raw: Vec<u16>,
}
#[cfg(any(test, feature = "std"))]
impl Exp4 {
    pub fn slots(&self) -> usize {
        self.raw.len() / GROUP
    }
}

#[inline(always)]
fn exp(v: u16) -> u8 {
    ((v >> 7) & 0xFF) as u8
}

/// The planes of a bf16 `[m][k]` matrix (`k` a multiple of `GROUP`).
#[cfg(any(test, feature = "std"))]
pub fn encode(w: &[u16], m: usize, k: usize) -> Exp4 {
    assert!(k % GROUP == 0 && w.len() == m * k);
    let groups = k / GROUP;
    let mut e = Exp4 { chunk: vec![0; m * (k / 8) * CHUNK], head: vec![0; m * groups], raw: Vec::new() };
    for r in 0..m {
        for g in 0..groups {
            let ws = &w[r * k + g * GROUP..r * k + (g + 1) * GROUP];
            let base = ws.iter().map(|&v| exp(v)).max().unwrap().saturating_sub(SPAN);
            let fits = ws.iter().all(|&v| exp(v) >= base);
            let mut head = base as u32;
            if !fits {
                head |= ((e.slots() + 1) as u32) << 8;
                e.raw.extend_from_slice(ws);
            }
            e.head[r * groups + g] = head;
            for (j, &v) in ws.iter().enumerate() {
                let at = r * k + g * GROUP + j;
                let ch = (at / 8) * CHUNK;
                e.chunk[ch + at % 8] = (((v >> 8) & 0x80) | (v & 0x7F)) as u8;
                // a raw group's codes are never read; written all the same so the planes are a function of `w` alone
                let c = exp(v).saturating_sub(base).min(SPAN);
                e.chunk[ch + 8 + (at % 8) / 2] |= c << (4 * (at % 2));
            }
        }
    }
    assert!(e.slots() < 1 << 24, "raw slots past the head's 24 bits");
    e
}

/// Weight `(r, j)` of the planes as bf16: what the kernel rebuilds.
#[cfg(any(test, feature = "std"))]
#[inline(always)]
pub fn weight(e: &Exp4, k: usize, r: usize, j: usize) -> u16 {
    let head = e.head[r * (k / GROUP) + j / GROUP];
    let slot = (head >> 8) as usize;
    if slot != 0 {
        return e.raw[(slot - 1) * GROUP + j % GROUP];
    }
    let at = r * k + j;
    let ch = (at / 8) * CHUNK;
    let c = (e.chunk[ch + 8 + (at % 8) / 2] >> (4 * (at % 2))) & 15;
    let ex = (head & 0xFF) as u16 + c as u16;
    let mb = e.chunk[ch + at % 8] as u16;
    ((mb & 0x80) << 8) | (ex << 7) | (mb & 0x7F)
}

/// The whole matrix back as bf16.
#[cfg(any(test, feature = "std"))]
pub fn decode(e: &Exp4, m: usize, k: usize) -> Vec<u16> {
    let mut w = vec![0u16; m * k];
    for r in 0..m {
        for j in 0..k {
            w[r * k + j] = weight(e, k, r, j);
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    #[test]
    fn round_trip_every_kind_of_group() {
        let (m, k) = (6, 4 * GROUP);
        let mut rng = Rng::new(11);
        let mut w: Vec<u16> = (0..m * k).map(|_| (rng.unit().to_bits() >> 16) as u16).collect();
        // zeros, negative zero and subnormals (exponent 0) in a group whose base is past 0: it goes raw
        w[3] = 0;
        w[5] = 0x8000;
        w[7] = 0x0013;
        w[9] = 0x8001;
        // a weight 20 binades below its group's largest: that group goes raw
        w[k + 130] = (w[k + 130] & 0x807F) | (((exp(w[k + 131]) as u16).saturating_sub(20).max(1)) << 7);
        // a group of tiny weights whose largest exponent is under 15: base 0, its zeros coded, not raw
        for j in 2 * k..2 * k + GROUP {
            w[j] = (w[j] & 0x807F) | (((j % 9) as u16) << 7);
        }
        w[2 * k + 4] = 0;
        let e = encode(&w, m, k);
        assert_eq!(e.slots(), 2, "the zeros' group and the deep weight's group went raw, the tiny group did not");
        assert_eq!(decode(&e, m, k), w);
        assert_eq!(e.chunk.len() + 4 * e.head.len() + 2 * e.raw.len(), bytes(m, k, e.slots()));
    }
}
