"""GPTQ for the oracle, the algorithm of bringup/quant.py (the reference the bundle's Rust
quantizer is checked against), over torch tensors: symmetric INT4, group 128 along the
input, scale = absmax/7 per (row, group) from the error-fed weights at the group's first
column, integers in [-8, 7] chosen with Hessian error feedback. `rtn` is the same
rounding without the feedback, the control."""
import torch

GROUP = 128


def round_away(x):
    """Round half away from zero, as `common::math::round` does (torch rounds half to even)."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def hinv(Hh, damp=0.01):
    """Upper Cholesky factor U of H⁻¹ for a batch of Hessians [E][k][k]; dead columns (zero
    diagonal) get a unit diagonal so the inverse exists (their weights are zeroed by the
    caller). Returns (U, dead)."""
    diag = torch.diagonal(Hh, dim1=-2, dim2=-1)
    dead = diag == 0
    diag[dead] = 1.0
    Hh = Hh + (damp * diag.mean(-1))[:, None, None] * torch.eye(Hh.shape[-1])
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hh))
    return torch.linalg.cholesky(Hinv, upper=True), dead


def gptq(Wm, U, dead):
    """GPTQ over a batch of matrices. Wm [E][m][k] f32, U [E or 1][k][k], dead [E or 1][k].
    Returns the dequantized weights [E][m][k] f32 (q · s), what the oracle multiplies by."""
    E, m, k = Wm.shape
    Wt = Wm.transpose(1, 2).contiguous()  # [E][k][m]
    for e in range(E):
        Wt[e, dead[min(e, dead.shape[0] - 1)], :] = 0.0
    out = torch.empty(E, k, m)
    B = GROUP
    for i1 in range(0, k, B):
        i2 = min(i1 + B, k)
        Wb = Wt[:, i1:i2, :].clone()
        Err = torch.zeros(E, i2 - i1, m)
        Hb = U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:
                s = Wb[:, i:i + GROUP, :].abs().amax(1) / 7.0
                s[s == 0] = 1.0
            w = Wb[:, i, :]
            qi = torch.clamp(round_away(w * (1.0 / s)), -8, 7)
            out[:, col, :] = qi * s
            err = (w - qi * s) / Hb[:, i, i, None]
            Wb[:, i:, :] -= Hb[:, i, i:, None] * err[:, None, :]
            Err[:, i, :] = err
        Wt[:, i2:, :] -= U[:, i1:i2, i2:].transpose(1, 2) @ Err
    return out.transpose(1, 2).contiguous()


def rtn(Wm, group=GROUP):
    """Round-to-nearest with the same scale and range, the shipped RTN: [.., m, k] f32 → f32."""
    k = Wm.shape[-1]
    g = Wm.reshape(*Wm.shape[:-1], k // group, group)
    s = g.abs().amax(-1, keepdim=True) / 7.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.clamp(round_away(g / s), -8, 7) * s).reshape(Wm.shape)
