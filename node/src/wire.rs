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
//! names the checkpoint, the recipe and every file's hash; 0 accepts any. The
//! node answers with the same three fields and one more u32: the positions a
//! sequence may hold after BOS, prompt and generation together, which is the
//! node's context budget and not a property of the bundle. A mismatch, an
//! unknown word, a full context or a closed peer ends the connection; abort is
//! closing it. Nagle is off, so a token leaves as it is sampled. The console is
//! the text twin: decimal ids in, a newline for GENERATE, token bytes out.

pub const VERSION: u32 = 4;
pub const GENERATE: u32 = u32::MAX;
pub const SCORE: u32 = u32::MAX - 1;
/// client → node: magic, version, identity
pub const HELLO: usize = 16;
/// node → client: the same, then positions
pub const HEADER: usize = 20;

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

/// The client's hello, if it is one of ours at this version: its expected identity.
pub fn accept(h: &[u8; HELLO]) -> Option<u64> {
    (&h[..4] == b"rock" && u32::from_le_bytes(h[4..8].try_into().unwrap()) == VERSION).then(|| u64::from_le_bytes(h[8..].try_into().unwrap()))
}
