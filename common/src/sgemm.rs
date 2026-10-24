//! f32 GEMM for the host's dense half — the audio encoder's projections, its convolutions (GEMMs over an
//! unrolled input) and its attention: `out[t][j] = Σ_k x[t][k] · w[j][k]`, both operands k-contiguous as the
//! checkpoint stores them. The weights are packed once into panels of `NR` rows laid k-major (`pack`), so
//! the inner loop is two 32-byte weight loads and `MR` broadcasts of the activations feeding `2·MR` FMAs
//! into `2·MR` accumulators (twelve of the sixteen ymm at `MR` = 6): the FMA pipes are the bound, not the
//! loads and not a horizontal reduction a dot (`linalg::dot`, the form before this, reduces eight lanes for
//! every output and re-reads the activation row for every weight row). A tile walks `KC` of k at a time so
//! the panel it reads (16 KB) stays in L1 across the positions that reuse it; the partial sums live in a
//! small C scratch and reach `out` once, through the epilogue (bias, GELU, accumulate into the residual).
//! An item of the pool is `MB` positions × `NB` columns (`items`, `item`), the items ordered by column block so
//! a contiguous run of them shares a weight block (`complexes` in the encoder: one core complex's L3 holds the
//! block while its cores stream the positions through it); `block` is the product onto a destination without an
//! epilogue, for the attention's scores and its values.
//!
//! Not bit-identical to the dot form (a different summation order); the gate is a tolerance one.

use core::arch::x86_64::*;
use core::ops::Range;
use std::vec::Vec;

/// Weight rows a panel (two ymm of outputs).
pub const NR: usize = 16;
/// Positions a tile (the broadcast operand).
pub const MR: usize = 6;
/// k a tile walks before its partial sums reach the C scratch: a panel's `NR · KC · 4` bytes = 16 KB in L1.
pub const KC: usize = 256;
/// Positions an item takes.
pub const MB: usize = 48;
/// Columns an item takes (`NB / NR` panels): the weight block an item streams through, `NB · k · 4` bytes.
pub const NB: usize = 128;
const _: () = assert!(MB % MR == 0 && NB % NR == 0);

/// A packed matrix: `m` output rows of `k`, as panels `[m / NR][k][NR]`; `bias` a row of `m`, added before the GELU.
pub struct Packed<'a> {
    pub w: &'a [f32],
    pub m: usize,
    pub k: usize,
    pub bias: Option<&'a [f32]>,
}

/// How an item's tile reaches `out`: `v = c + bias`, then `gelu(v)` if `gelu`, then `out += v` if `add`, else `out = v`.
#[derive(Clone, Copy)]
pub struct Epilogue {
    pub gelu: bool,
    pub add: bool,
}
pub const STORE: Epilogue = Epilogue { gelu: false, add: false };
pub const GELU: Epilogue = Epilogue { gelu: true, add: false };
pub const ADD: Epilogue = Epilogue { gelu: false, add: true };
pub const GELU_ADD: Epilogue = Epilogue { gelu: true, add: true };

/// `w[j][kk] = at(j, kk)` for `j < rows`, zero rows up to `m` (`m % NR == 0`) → panels `[m / NR][k][NR]`.
pub fn pack_with(m: usize, rows: usize, k: usize, at: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    assert!(m % NR == 0 && rows <= m);
    let mut out = std::vec![0f32; m * k];
    pack_into(&mut out, rows, k, at);
    out
}

/// `pack_with` into a buffer already sized `[m / NR][k][NR]` (rows past `rows` are left as they are).
#[inline]
pub fn pack_into(out: &mut [f32], rows: usize, k: usize, at: impl Fn(usize, usize) -> f32) {
    for j in 0..rows {
        let base = (j / NR) * k * NR + j % NR;
        for kk in 0..k {
            out[base + kk * NR] = at(j, kk);
        }
    }
}

/// `w[m][k]` f32 → panels.
pub fn pack(w: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    pack_with(m, m, k, |j, kk| w[j * k + kk])
}

/// `w[m][k]` bf16 → panels, widened.
pub fn pack_bf16(w: &[u16], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), m * k);
    pack_with(m, m, k, |j, kk| f32::from_bits((w[j * k + kk] as u32) << 16))
}

/// Items an `n` positions × `m` columns product splits into.
pub fn items(m: usize, n: usize) -> usize {
    (m / NR).div_ceil(NB / NR) * n.div_ceil(MB)
}

/// Item `i` of `items(w.m, n)` — column block `i / tbs`, position block `i % tbs` — its positions × its columns
/// of `out[t][j]` (`t < n`, row stride `ldo`) from `x[t][kk]` (row stride `ldx`), through the epilogue.
pub unsafe fn item(w: &Packed, i: usize, x: *const f32, ldx: usize, n: usize, out: *mut f32, ldo: usize, ep: Epilogue) {
    let panels = w.m / NR;
    let tbs = n.div_ceil(MB);
    let (cb, tb) = (i / tbs, i % tbs);
    let (p0, p1) = (cb * (NB / NR), ((cb + 1) * (NB / NR)).min(panels));
    let (t0, t1) = (tb * MB, (tb * MB + MB).min(n));
    let (rows, cols) = (t1 - t0, (p1 - p0) * NR);
    let mut c = core::mem::MaybeUninit::<[f32; MB * NB]>::uninit();
    let c = c.as_mut_ptr() as *mut f32;
    block(w.w.as_ptr(), w.k, p0..p1, x.add(t0 * ldx), ldx, rows, c, NB);
    epilogue(c, rows, cols, w.bias.map(|b| b.as_ptr().add(p0 * NR)), out.add(t0 * ldo + p0 * NR), ldo, ep);
}

/// `c[t][(p - panels.start) · NR + j] = Σ_k x[t][k] · w[p · NR + j][k]` for `t < n`: the product of `n` rows of `x`
/// (row stride `ldx`) against the packed panels `panels` of `w` (`k` wide), onto `c` (row stride `ldc`), the
/// whole k in blocks of `KC`.
pub unsafe fn block(w: *const f32, k: usize, panels: Range<usize>, x: *const f32, ldx: usize, n: usize, c: *mut f32, ldc: usize) {
    let mut k0 = 0;
    while k0 < k {
        let kc = KC.min(k - k0);
        for p in panels.clone() {
            tiles(w.add((p * k + k0) * NR), kc, x.add(k0), ldx, n, c.add((p - panels.start) * NR), ldc, k0 == 0);
        }
        k0 += kc;
    }
}

/// One panel against `n` rows: tiles of `MR`, then the remainder.
#[target_feature(enable = "avx2,fma")]
unsafe fn tiles(w: *const f32, kc: usize, x: *const f32, ldx: usize, n: usize, c: *mut f32, ldc: usize, first: bool) {
    let mut t = 0;
    while t + MR <= n {
        tile::<MR>(w, kc, x.add(t * ldx), ldx, c.add(t * ldc), ldc, first);
        t += MR;
    }
    let (x, c) = (x.add(t * ldx), c.add(t * ldc));
    match n - t {
        1 => tile::<1>(w, kc, x, ldx, c, ldc, first),
        2 => tile::<2>(w, kc, x, ldx, c, ldc, first),
        3 => tile::<3>(w, kc, x, ldx, c, ldc, first),
        4 => tile::<4>(w, kc, x, ldx, c, ldc, first),
        5 => tile::<5>(w, kc, x, ldx, c, ldc, first),
        _ => {}
    }
}

/// `M` rows × one panel over `kc`: `2·M` accumulators, two weight loads and `M` broadcasts a k. The sums are
/// stored (`first`) or added into `c`.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile<const M: usize>(w: *const f32, kc: usize, x: *const f32, ldx: usize, c: *mut f32, ldc: usize, first: bool) {
    let mut a0 = [_mm256_setzero_ps(); M];
    let mut a1 = [_mm256_setzero_ps(); M];
    for kk in 0..kc {
        let w0 = _mm256_loadu_ps(w.add(kk * NR));
        let w1 = _mm256_loadu_ps(w.add(kk * NR + 8));
        for t in 0..M {
            let xv = _mm256_broadcast_ss(&*x.add(t * ldx + kk));
            a0[t] = _mm256_fmadd_ps(xv, w0, a0[t]);
            a1[t] = _mm256_fmadd_ps(xv, w1, a1[t]);
        }
    }
    for t in 0..M {
        let o = c.add(t * ldc);
        if first {
            _mm256_storeu_ps(o, a0[t]);
            _mm256_storeu_ps(o.add(8), a1[t]);
        } else {
            _mm256_storeu_ps(o, _mm256_add_ps(_mm256_loadu_ps(o), a0[t]));
            _mm256_storeu_ps(o.add(8), _mm256_add_ps(_mm256_loadu_ps(o.add(8)), a1[t]));
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn epilogue(c: *const f32, rows: usize, cols: usize, bias: Option<*const f32>, out: *mut f32, ldo: usize, ep: Epilogue) {
    for t in 0..rows {
        let mut j = 0;
        while j < cols {
            let mut v = _mm256_loadu_ps(c.add(t * NB + j));
            if let Some(b) = bias {
                v = _mm256_add_ps(v, _mm256_loadu_ps(b.add(j)));
            }
            if ep.gelu {
                v = gelu8(v);
            }
            let o = out.add(t * ldo + j);
            if ep.add {
                v = _mm256_add_ps(v, _mm256_loadu_ps(o));
            }
            _mm256_storeu_ps(o, v);
            j += 8;
        }
    }
}

/// e^x a lane: `math::exp`'s expression (Cephes `expf`), eight at a time.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn exp8(x: __m256) -> __m256 {
    let x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0)), _mm256_set1_ps(-87.0));
    let n = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.442_695_04), _mm256_set1_ps(0.5)));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_375), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let p = _mm256_set1_ps(1.987_569_2e-4);
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.398_199_9e-3));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.333_452e-3));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.166_579_6e-2));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.666_666_5e-1));
    let p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.000_000_1e-1));
    let p = _mm256_add_ps(_mm256_fmadd_ps(_mm256_mul_ps(p, r), r, r), _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
    _mm256_mul_ps(p, _mm256_castsi256_ps(e))
}

/// Exact GELU `x · Φ(x)` a lane, erf by Abramowitz–Stegun 7.1.26 (1.5e-7), the encoder's `gelu` eight at a time.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gelu8(x: __m256) -> __m256 {
    let sign = _mm256_set1_ps(-0.0);
    let z = _mm256_mul_ps(x, _mm256_set1_ps(core::f32::consts::FRAC_1_SQRT_2));
    let a = _mm256_andnot_ps(sign, z);
    let t = _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_fmadd_ps(_mm256_set1_ps(0.3275911), a, _mm256_set1_ps(1.0)));
    let p = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429), t, _mm256_set1_ps(-1.453152027));
    let p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421413741));
    let p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284496736));
    let p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254829592));
    let y = _mm256_fnmadd_ps(_mm256_mul_ps(p, t), exp8(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(a, a))), _mm256_set1_ps(1.0));
    // erf(z) = y with z's sign; gelu = ½ x (1 + erf z)
    let erf = _mm256_or_ps(y, _mm256_and_ps(z, sign));
    _mm256_mul_ps(_mm256_mul_ps(x, _mm256_set1_ps(0.5)), _mm256_add_ps(_mm256_set1_ps(1.0), erf))
}

/// `gelu` over a slice in place.
pub fn gelu(x: &mut [f32]) {
    unsafe { gelu_slice(x) }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_slice(x: &mut [f32]) {
    let p = x.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= x.len() {
        _mm256_storeu_ps(p.add(i), gelu8(_mm256_loadu_ps(p.add(i))));
        i += 8;
    }
    while i < x.len() {
        let v = _mm256_set1_ps(*p.add(i));
        *p.add(i) = _mm256_cvtss_f32(gelu8(v));
        i += 1;
    }
}

/// Softmax of `s[..valid]` in place, `s[valid..]` set to zero (the row's length a multiple of 8).
pub fn softmax_row(s: &mut [f32], valid: usize) {
    debug_assert!(s.len() % 8 == 0 && valid <= s.len());
    unsafe { softmax_row_avx2(s, valid) }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_avx2(s: &mut [f32], valid: usize) {
    let p = s.as_mut_ptr();
    for i in valid..s.len() {
        *p.add(i) = f32::NEG_INFINITY;
    }
    let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i < s.len() {
        m = _mm256_max_ps(m, _mm256_loadu_ps(p.add(i)));
        i += 8;
    }
    let m = hmax(m);
    let mv = _mm256_set1_ps(m);
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i < s.len() {
        let e = exp8(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv));
        _mm256_storeu_ps(p.add(i), e);
        sum = _mm256_add_ps(sum, e);
        i += 8;
    }
    // the padding went to -inf then exp(-inf - m) = exp(-87 clamp) ≠ 0: zero it, and take it out of the sum
    let mut pad = 0f32;
    for i in valid..s.len() {
        pad += *p.add(i);
        *p.add(i) = 0.0;
    }
    let inv = _mm256_set1_ps(1.0 / (crate::gemv::avx2::hsum(sum) - pad));
    let mut i = 0;
    while i < s.len() {
        _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), inv));
        i += 8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hmax(v: __m256) -> f32 {
    let x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_max_ss(x, _mm_shuffle_ps::<1>(x, x));
    _mm_cvtss_f32(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    fn naive(w: &[f32], b: Option<&[f32]>, m: usize, k: usize, x: &[f32], n: usize) -> Vec<f32> {
        let mut out = std::vec![0f32; n * m];
        for t in 0..n {
            for j in 0..m {
                let mut s = b.map_or(0.0, |b| b[j]);
                for kk in 0..k {
                    s += x[t * k + kk] * w[j * k + kk];
                }
                out[t * m + j] = s;
            }
        }
        out
    }

    #[test]
    fn items_match_naive_with_remainders() {
        let mut rng = Rng::new(5);
        // two column blocks (the second of one panel), two position blocks (the second of 3 rows), two k-blocks
        let (m, k, n) = (144, 300, 51);
        let w: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
        let b: Vec<f32> = (0..m).map(|_| rng.unit()).collect();
        let x: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        let want = naive(&w, Some(&b), m, k, &x, n);
        let pw = pack(&w, m, k);
        let p = Packed { w: &pw, m, k, bias: Some(&b) };
        for (ep, f) in [(STORE, (|v: f32, _o: f32| v) as fn(f32, f32) -> f32), (GELU, |v, _| crate_gelu(v)), (ADD, |v, o| v + o), (GELU_ADD, |v, o| crate_gelu(v) + o)] {
            let mut out: Vec<f32> = (0..n * m).map(|i| i as f32 * 1e-3).collect();
            let before = out.clone();
            for i in 0..items(m, n) {
                unsafe { item(&p, i, x.as_ptr(), k, n, out.as_mut_ptr(), m, ep) };
            }
            for i in 0..n * m {
                let want = f(want[i], before[i]);
                assert!((out[i] - want).abs() < 1e-3 * (1.0 + want.abs()), "{i}: {} vs {want}", out[i]);
            }
        }
    }

    fn crate_gelu(x: f32) -> f32 {
        let z = x * core::f32::consts::FRAC_1_SQRT_2;
        let s = if z < 0.0 { -1.0 } else { 1.0 };
        let a = z.abs();
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * crate::math::exp(-a * a);
        0.5 * x * (1.0 + s * y)
    }

    #[test]
    fn block_strided_operands() {
        // the attention's shape: x rows strided wider than k, several panels onto a wide destination
        let mut rng = Rng::new(9);
        let (m, k, n, ldx, ldc) = (32, 64, 13, 200, 40);
        let w: Vec<f32> = (0..m * k).map(|_| rng.unit()).collect();
        let x: Vec<f32> = (0..n * ldx).map(|_| rng.unit()).collect();
        let pw = pack(&w, m, k);
        let mut c = std::vec![0f32; n * ldc];
        unsafe { block(pw.as_ptr(), k, 0..m / NR, x.as_ptr(), ldx, n, c.as_mut_ptr(), ldc) };
        for t in 0..n {
            for j in 0..m {
                let want: f32 = (0..k).map(|kk| x[t * ldx + kk] * w[j * k + kk]).sum();
                assert!((c[t * ldc + j] - want).abs() < 1e-4, "{t} {j}");
            }
        }
    }

    #[test]
    fn exp_gelu_softmax_match_scalar() {
        let xs: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.7).collect();
        let mut g = xs.clone();
        gelu(&mut g);
        for (i, &x) in xs.iter().enumerate() {
            let e = unsafe { _mm256_cvtss_f32(exp8(_mm256_set1_ps(x))) };
            let want = crate::math::exp(x);
            assert!((e - want).abs() <= 2e-6 * want, "exp {x}: {e} vs {want}");
            assert!((g[i] - crate_gelu(x)).abs() < 1e-5 * (1.0 + x.abs()), "gelu {x}");
        }
        let mut s: Vec<f32> = (0..24).map(|i| (i as f32) * 0.3 - 2.0).collect();
        softmax_row(&mut s, 20);
        let mx = (0..20).map(|i| i as f32 * 0.3 - 2.0).fold(f32::MIN, f32::max);
        let z: f32 = (0..20).map(|i| crate::math::exp(i as f32 * 0.3 - 2.0 - mx)).sum();
        for i in 0..24 {
            let want = if i < 20 { crate::math::exp(i as f32 * 0.3 - 2.0 - mx) / z } else { 0.0 };
            assert!((s[i] - want).abs() < 1e-6, "softmax {i}: {} vs {want}", s[i]);
        }
    }
}
