# train — the training side

The code a continual-adaptation round needs and the node never serves with. Its own Cargo
workspace, kept apart from the inference tree on purpose (tcdent, 2026-10-25: training "can be
somewhat isolated"); it depends on `common` and `models/<name>` and changes nothing in them.
The design it implements is `~/Work/localai/continual-adaptation.md` and `sdpo.md`; the
decisions taken for this platform are in the crate docs.

```
train/
  common/          training primitives with no model in them: the transposed products over
                   the inference byte formats (the gradient at a frozen matrix's input), the
                   adapter's low-rank steps
  glm-5.3-flash/   GLM-5.3-Flash's layer backwards over glm_5_3_flash's bytes and kernels, its
                   adapters, and the f32 reference each backward is diffed against
```

Conventions, inherited from the inference tree: a definition (scalar) and one realization a
target, bit-identical where the arithmetic allows it; an f32 reference the kernel is diffed
against at the real shape, and the reference itself checked against finite differences at a
small shape; one op at a time, measured.

## What exists

| op | where | gate |
|---|---|---|
| `dx += Wᵀ·dy` over `q4g128` | `common/src/q4.rs` | AVX2 = scalar bit for bit; scalar = dense `Wᵀ·dy` |
| adapter steps `y += B·(A·x)` and their backward | `common/src/lowrank.rs` | through the expert's diffs |
| routed expert with adapter: forward (keeps what the backward needs) and backward | `glm-5.3-flash/src/expert.rs` | zero adapter = `glm_5_3_flash::expert::forward` value for value; forward and backward within ~2e-6 of the dense f32 reference (gate 1e-4) |
| `dx += Wᵀ·dy` over FP8 128×128-block rows | `common/src/fp8.rs` | AVX2 = scalar bit for bit (both `fma`); scalar = dense `Wᵀ·dy` |
| `dx += Σ dy_r·W[row_r]` over bf16 rows (the router's picks) | `common/src/bf16.rs` | AVX2 = scalar bit for bit; = dense |
| the clamped SwiGLU's and the RMS norm's backward | `common/src/swiglu.rs`, `common/src/rmsnorm.rs` | finite differences |
| the MoE half on its normed input: router (frozen; picked rows), experts with adapters and their weights, shared expert — to `d xn` | `glm-5.3-flash/src/ffn.rs` | zero adapters = `ffn::moe_routed` + `moe_shared` value for value; vs the dense reference at the real shape 1e-4, except the two adapter gradients downstream of the quantized ĥ, gated at a quantizer step (the sides' h differ by summation order; a boundary element flips one INT8 step) |
| the hyper-connection around a block half: `post`'s backward to `d y` and the residual's share; `pre`'s through the sigmoids, the softmax and the Sinkhorn chain (unrolled in reverse over recomputed intermediates), `fn_ᵀ·dc` and the unweighted norm, to `d S` | `glm-5.3-flash/src/hc.rs` | over the inference forward, vs the dense reference 1e-4 at the real shape |
| the dense f32 reference of the above | `glm-5.3-flash/src/reference.rs` | finite differences on every parameter class and on `dx`, for the expert and for the whole MoE (router path included) |

## Measured

`cargo run --release --example expert_rate` — one expert, one core, cache-hot (one expert is
~12.6 MB), so the instruction cost of each op rather than the DRAM stream:

| 2026-11-01, EPYC 7551P (Zen 1, the dev box; not metatron) | µs | × inference forward |
|---|---|---|
| inference forward | 1,869 | 1.00 |
| forward with a rank-16 adapter | 2,220 | 1.19 |
| backward, no adapter | 5,882 | 3.15 |
| backward, rank 16 | 6,050 | 3.24 |

The backward reads the same bytes as the forward once more, but each nibble becomes an f32 and
multiplies an f32 coefficient (no integer dot), so it runs ~0.5 instructions a weight against
the forward's ~0.14: instruction-bound where the forward is memory-bound. Whether that shows at
all under the pool on metatron (48 cores sharing the DRAM stream) is the next measurement; the
lever if it does is an integer inner loop over a quantized coefficient. The adapter at rank 16
is +19 % on the forward here — ~0.6 MB of f32 against 12.6 MB of expert, read with plain loops.
