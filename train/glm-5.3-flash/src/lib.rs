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
//! ```
//! The rest of the layer (MLA, the head) follows, one op at a time.

pub mod expert;
pub mod ffn;
pub mod hc;
pub mod kda;
pub mod reference;
