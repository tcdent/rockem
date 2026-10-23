//! The wire: a TCP byte stream of little-endian u32 words, with no framing
//! beyond TCP's own. Words below the vocabulary are token ids; the few above
//! it are verbs. The stream in is the transcript the model sees and the
//! stream out is the transcript it produces, and nothing is transformed on
//! the way, so a captured stream is a replay.
//!
//!   client → node   header, then words: BOS starts a sequence (the sink is
//!                   re-seeded, the cache is the sink's), ids are fed to the
//!                   model as they arrive, GENERATE means "sample from here" and is
//!                   followed by one word, the ids wanted at most (0: the node's
//!                   budget) — the node stops exactly where the client does, so
//!                   the state it keeps at the reply's end is the reply the client
//!                   has (v4, 2026-10-16); SCORE means "from here, answer every id
//!                   I feed with its log-probability before feeding it"
//!   node → client   header, then every sampled id the moment it exists,
//!                   EOS included, each fed back so the cache holds exactly
//!                   the stream; after EOS the node reads again, so more ids
//!                   and another GENERATE continue the same sequence. After
//!                   SCORE, one f32 (its bits in the word) per id fed: ln p
//!                   of that id given everything before it, under the model's
//!                   own distribution, no recipe; a GENERATE still streams ids
//!
//! The header at connect, framed by position: the client writes `rock`,
//! version u32, identity u64 — FNV-1a of the bundle's manifest.toml, which
//! names the checkpoint, the recipe and every file's hash; 0 accepts any — and
//! the model, 32 bytes: the `name` of one of the manifest's models, UTF-8,
//! zero-padded. A bundle carries one model or several, all resident; the
//! connection speaks the one it named — its ids, its verbs, its context store —
//! until it closes, and there is no default: a name the bundle lacks, an empty
//! one included, is refused like a wrong identity (v5, 2026-10-23, docs/queue/094).
//! The node answers with magic, version, identity and one more u32: the
//! positions a sequence may hold after BOS, prompt and generation together,
//! which is the node's context budget for that model and not a property of the
//! bundle. A mismatch, an unknown word, a full context or a closed peer ends the
//! connection; abort is closing it. Nagle is off, so a token leaves as it is
//! sampled. The console is the text twin: decimal ids in, a newline for
//! GENERATE, token bytes out.
//!
//! AUDIO (v5): one encoder window follows, of the size the model's manifest
//! entry states, raw little-endian f32; the node encodes it and each
//! placeholder id fed after takes the next row in place of its token embedding
//! (docs/queue/094). A model without an encoder refuses it.

pub const VERSION: u32 = 5;
pub const GENERATE: u32 = u32::MAX;
pub const SCORE: u32 = u32::MAX - 1;
pub const AUDIO: u32 = u32::MAX - 2;
/// client → node: magic, version, identity, model name
pub const HELLO: usize = 48;
/// node → client: magic, version, identity, then positions
pub const HEADER: usize = 20;
/// The model's name in the hello: UTF-8, zero-padded.
pub const MODEL_NAME: usize = 32;

pub fn identity(manifest: &[u8]) -> u64 {
    manifest.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

pub fn header(identity: u64, positions: u32) -> [u8; HEADER] {
    let mut h = [0u8; HEADER];
    h[..4].copy_from_slice(b"rock");
    h[4..8].copy_from_slice(&VERSION.to_le_bytes());
    h[8..16].copy_from_slice(&identity.to_le_bytes());
    h[16..].copy_from_slice(&positions.to_le_bytes());
    h
}

/// The client's hello, if it is one of ours at this version: its expected identity and the model it names
/// (a name that is empty or not UTF-8 is not a hello).
pub fn accept(h: &[u8; HELLO]) -> Option<(u64, &str)> {
    if &h[..4] != b"rock" || u32::from_le_bytes(h[4..8].try_into().unwrap()) != VERSION {
        return None;
    }
    let name = &h[16..16 + MODEL_NAME];
    let end = name.iter().position(|&b| b == 0).unwrap_or(MODEL_NAME);
    let name = std::str::from_utf8(&name[..end]).ok().filter(|n| !n.is_empty())?;
    Some((u64::from_le_bytes(h[8..16].try_into().unwrap()), name))
}
