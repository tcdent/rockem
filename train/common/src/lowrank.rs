//! The adapter's steps. An adapter on a frozen `[m][k]` matrix is `ΔW = B · A`,
//! `A: [r][k]`, `B: [m][r]`, both f32, `r` a runtime rank. Forward:
//! `y += B · (A · x)`; backward, given `dy` and the saved `z = A · x`:
//! `t = Bᵀ · dy`, `dB += dy ⊗ z`, `dA += t ⊗ x`, `dx += Aᵀ · t`. Plain loops:
//! `r` is tens against a `k` of thousands, so these are a few percent of the
//! base matrix's bytes and the compiler's vectors are enough.

/// `z[0..r) = A[r × k] · x[0..k)`.
#[inline(always)]
pub fn down(r: usize, k: usize, a: &[f32], x: &[f32], z: &mut [f32]) {
    for i in 0..r {
        let row = &a[i * k..(i + 1) * k];
        let mut s = 0f32;
        for j in 0..k {
            s += row[j] * x[j];
        }
        z[i] = s;
    }
}

/// `y[0..m) += B[m × r] · z[0..r)`.
#[inline(always)]
pub fn up(m: usize, r: usize, b: &[f32], z: &[f32], y: &mut [f32]) {
    for i in 0..m {
        let row = &b[i * r..(i + 1) * r];
        let mut s = 0f32;
        for j in 0..r {
            s += row[j] * z[j];
        }
        y[i] += s;
    }
}

/// `t[0..r) = Bᵀ · dy`, `dB += dy ⊗ z`.
#[inline(always)]
pub fn up_backward(m: usize, r: usize, b: &[f32], z: &[f32], dy: &[f32], t: &mut [f32], db: &mut [f32]) {
    for v in t[..r].iter_mut() {
        *v = 0.0;
    }
    for i in 0..m {
        let row = &b[i * r..(i + 1) * r];
        let g = &mut db[i * r..(i + 1) * r];
        let d = dy[i];
        for j in 0..r {
            t[j] += row[j] * d;
            g[j] += d * z[j];
        }
    }
}

/// `dA += t ⊗ x`, `dx += Aᵀ · t`.
#[inline(always)]
pub fn down_backward(r: usize, k: usize, a: &[f32], x: &[f32], t: &[f32], da: &mut [f32], dx: &mut [f32]) {
    for i in 0..r {
        let row = &a[i * k..(i + 1) * k];
        let g = &mut da[i * k..(i + 1) * k];
        let ti = t[i];
        for j in 0..k {
            g[j] += ti * x[j];
            dx[j] += row[j] * ti;
        }
    }
}
