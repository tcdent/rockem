//! This model's bundle formats. The routed experts are a stream of fixed-size
//! records in (layer, expert) order for layers `FIRST_K_DENSE_REPLACE..`:
//! one `q4g128` record per expert — gate rows then up rows fused, then down,
//! each weight stream followed by its scales, every stream a multiple of 64
//! bytes so f32 scales stay aligned and records tile.

use crate::{D_FF, H};
use common::q4::GROUP;

pub const GATE_UP_W: usize = 2 * D_FF * H / 2;
pub const GATE_UP_S: usize = 2 * D_FF * (H / GROUP); // f32 count
pub const DOWN_W: usize = H * D_FF / 2;
pub const DOWN_S: usize = H * (D_FF / GROUP); // f32 count
pub const RECORD: usize = GATE_UP_W + 4 * GATE_UP_S + DOWN_W + 4 * DOWN_S;
pub const LAYER: usize = crate::N_ROUTED_EXPERTS * RECORD;
pub const STREAM: usize = (crate::NUM_HIDDEN_LAYERS - crate::FIRST_K_DENSE_REPLACE) * LAYER;
/// The expert recipe string the manifest carries: the record format and how
/// its integers were chosen (`bundle`: RTN, or GPTQ against `calib.ids`).
pub const RECIPE_RTN: &str = "experts:q4g128-a8-rtn";
pub const RECIPE_GPTQ: &str = "experts:q4g128-a8-gptq";

const _: () = assert!(RECORD % 64 == 0 && H % GROUP == 0 && D_FF % GROUP == 0);

/// Byte range of expert `e` in MoE layer `l` within the stream.
#[inline(always)]
pub const fn record(l: usize, e: usize) -> core::ops::Range<usize> {
    let start = (l - crate::FIRST_K_DENSE_REPLACE) * LAYER + e * RECORD;
    start..start + RECORD
}

/// The dense (non-expert) weights, bf16, one file, in the order below.
/// Offsets are in elements. Both the bundle builder and the loader walk
/// these constants, so the file has one definition. ~2.6 GB for V2-Lite.
pub mod dense {
    use crate::*;

    pub const EMBED: usize = 0; // [VOCAB_SIZE][H]
    pub const NORM: usize = EMBED + VOCAB_SIZE * H; // [H]
    pub const LM_HEAD: usize = NORM + H; // [VOCAB_SIZE][H]
    pub const LAYERS: usize = LM_HEAD + VOCAB_SIZE * H;

    // within a layer, in order
    pub const INPUT_LN: usize = 0; // [H]
    pub const POST_LN: usize = INPUT_LN + H; // [H]
    pub const Q_PROJ: usize = POST_LN + H; // [Q_DIM][H]
    pub const KV_A: usize = Q_PROJ + Q_DIM * H; // [KVA_DIM][H]
    pub const KV_A_LN: usize = KV_A + KVA_DIM * H; // [KV_LORA_RANK]
    pub const KV_B: usize = KV_A_LN + KV_LORA_RANK; // [KVB_DIM][KV_LORA_RANK]
    pub const O_PROJ: usize = KV_B + KVB_DIM * KV_LORA_RANK; // [H][O_IN_DIM]
    pub const FFN: usize = O_PROJ + H * O_IN_DIM;
    // dense layer: gate [I][H], up [I][H], down [H][I]
    pub const DENSE_GATE: usize = FFN;
    pub const DENSE_UP: usize = DENSE_GATE + INTERMEDIATE_SIZE * H;
    pub const DENSE_DOWN: usize = DENSE_UP + INTERMEDIATE_SIZE * H;
    pub const DENSE_FFN_LEN: usize = 3 * INTERMEDIATE_SIZE * H;
    // moe layer: router [N_ROUTED_EXPERTS][H], shared gate [S][H], up [S][H], down [H][S]
    pub const ROUTER: usize = FFN;
    pub const SHARED_GATE: usize = ROUTER + N_ROUTED_EXPERTS * H;
    pub const SHARED_UP: usize = SHARED_GATE + SHARED_FF * H;
    pub const SHARED_DOWN: usize = SHARED_UP + SHARED_FF * H;
    pub const MOE_FFN_LEN: usize = N_ROUTED_EXPERTS * H + 3 * SHARED_FF * H;

    pub const fn ffn_len(l: usize) -> usize {
        if l < FIRST_K_DENSE_REPLACE { DENSE_FFN_LEN } else { MOE_FFN_LEN }
    }
    /// Element offset of layer `l`.
    pub const fn layer(l: usize) -> usize {
        let mut off = LAYERS;
        let mut i = 0;
        while i < l {
            off += FFN + ffn_len(i);
            i += 1;
        }
        off
    }
    pub const TOTAL: usize = layer(NUM_HIDDEN_LAYERS);
    pub const BYTES: usize = TOTAL * 2;
}

/// The vocabulary for the console: `[u32; VOCAB_SIZE + 1]` byte offsets,
/// then the concatenated UTF-8 bytes of every token in id order.
pub const VOCAB_FILE: &str = "vocab.bin";
pub const DENSE_FILE: &str = "dense.bf16.bin";
pub const EXPERTS_FILE: &str = "experts.q4g128.bin";
/// The gfx1100 code object (`models/deepseek-v2-lite/gpu`), carried as data.
pub const KERNELS_FILE: &str = "kernels.gfx1100.elf";
/// The sink token's state: BOS has no context, so its forward pass is a
/// constant of the model. Its cache row for every layer, `[NUM_HIDDEN_LAYERS]
/// [KV_ROW]` f32, computed once at bundle build at full precision. Not a
/// recipe choice: every sequence is seeded from it and computed from
/// `FIRST_POS`; position 0 is never computed at runtime. See analysis.md §5.
pub const SINK_FILE: &str = "sink.f32";
/// The edge's half: the checkpoint's tokenizer and its config (the chat
/// template, the added tokens), copied verbatim so the API service that
/// encodes text for the wire reads the same directory the node does. The node
/// never opens them.
pub const TOKENIZER_FILES: [&str; 2] = ["tokenizer.json", "tokenizer_config.json"];
/// One position of the attention cache, every layer's row: `[layer][KV_ROW]`
/// f32. A cached context is `[pos][POSITION]`; the sink is one position.
pub const POSITION: usize = crate::NUM_HIDDEN_LAYERS * crate::KV_ROW;
pub const SINK_LEN: usize = POSITION;
