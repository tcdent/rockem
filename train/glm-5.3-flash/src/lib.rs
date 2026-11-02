//! GLM-5.3-Flash, the training side. The inference crate (`glm_5_3_flash`)
//! is the model: its constants, its weights as they lie in the bundle, its
//! forward kernels. This crate adds what a training round needs and nothing
//! the node serves with: the backward of each layer given the gradient at
//! its output, the adapters that are the only parameters that move, and the
//! f32 reference each backward is diffed against.
//!
//! What moves and what does not (`~/Work/localai/continual-adaptation.md` §3,
//! §7): the routed experts carry adapters; router, attention, embeddings,
//! norms, the shared expert and the base experts are frozen. A frozen
//! matrix's backward is the transposed product over its inference bytes
//! (`train_common`): the gradient at its input, never at its weights.
//!
//! Per layer, in the order a backward visits them:
//!
//! ```text
//! expert    one routed expert with its adapter: forward saving what the backward
//!           needs, and the backward — dx and the adapter's gradients
//! ffn       the MoE half on its normed input: the router's picked rows, the experts
//!           and their weights, the shared expert — to d xn
//! hc        the hyper-connection around a block half: post's backward to d y and the
//!           residual's share, pre's through the Sinkhorn chain and the norm to d S
//! kda       a run of positions through a KDA layer: the inference step forward, the
//!           recurrence in reverse per head (recomputed from the entering memory), the
//!           convolution over the run's timeline, the projections — to d xn and the
//!           entering state's gradient
//! mla       a run of positions through an MLA layer: the inference step forward; the
//!           absorbed attention's backward straight into the latents' gradient, the query
//!           path, then the run's own latents through their norm and kv_a — to d xn
//! head      the streams' mean, the final norm, lm_head — to d streams
//! loss      a position's loss against a top-K teacher, and its gradient at the logits
//! model     the layers over a run: the embedding to the logits forward, the logits'
//!           gradient down to the first MoE layer's input backward; runs chain through
//!           the states and a carry
//! optim     Adam on an expert's adapter under the per-expert movement cap; the distance
//!           from the base as the drift instrument
//! round     a training round: sessions fed as ids with the teacher's targets, forward in
//!           runs with the states snapshotted at each boundary, the backward recomputing
//!           each run from its snapshot in reverse; a step per session; the report
//! ```

pub mod expert;
pub mod ffn;
pub mod hc;
pub mod head;
pub mod kda;
pub mod loss;
pub mod mla;
pub mod model;
pub mod optim;
pub mod reference;
pub mod round;
#[cfg(test)]
pub mod testing;
