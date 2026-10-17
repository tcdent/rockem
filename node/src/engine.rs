//! The node's view of a model: the constants the wire and the context store
//! need, the prefix every sequence starts from, and the step. One binary
//! carries every model and the bundle's manifest names which one (`[model]
//! name`); switching models is a restart (tcdent, 2026-10-15: "putting the
//! model on the system and the system is absolutely purpose built for that";
//! prune nothing from the binary until its size is a problem).
//!
//! Two kinds of sequence state meet here. **Rows** are per position — the
//! attention cache's row for every layer that has one — and a cached context is
//! `[pos][position()]` of them, seeded back in any prefix length. A **snapshot**
//! is state that is not per position: a recurrent layer's memory, which exists
//! only at the sequence's end. A model with snapshots (`snapshot_len() > 0`) can
//! resume a cached context only where one was taken — its end — and otherwise
//! recomputes from the prefix; a model without them resumes anywhere.

pub use common::sample::Sampling;

pub trait Engine {
    fn name(&self) -> &'static str;
    fn vocab_size(&self) -> usize;
    /// The id that starts a sequence on the wire: the first of `prefix_ids`.
    fn bos(&self) -> u32 {
        self.prefix_ids()[0]
    }
    fn is_eos(&self, id: u32) -> bool;
    /// The tokens a step takes at most.
    fn tokens(&self) -> usize;
    /// The ids every sequence begins with — one BOS, or a model's own prefix —
    /// whose state the bundle ships (the prefix rows and snapshot); computing
    /// starts at `prefix_ids().len()`.
    fn prefix_ids(&self) -> &[u32];
    fn first_pos(&self) -> usize {
        self.prefix_ids().len()
    }
    /// f32s of per-position state (one position's rows).
    fn position(&self) -> usize;
    /// f32s of sequence state that is not per position; 0 when every state is a row.
    fn snapshot_len(&self) -> usize;
    fn seed_value(&self) -> u64;
    /// The prefix's rows (`[first_pos][position]`) and snapshot, from the bundle.
    fn prefix(&self) -> (&[f32], &[f32]);
    /// Install the prefix's state: computing resumes at `first_pos()`.
    fn seed_prefix(&mut self);
    /// Install `rows` (`[n][position]`) as positions `0..n`, and `snap` as the
    /// state at `n` (empty for a model without snapshots); computing resumes at `n`.
    fn seed(&mut self, rows: &[f32], snap: &[f32]);
    /// The sequence state now, `snapshot_len()` f32s.
    fn snapshot(&self, out: &mut [f32]);
    /// `toks` at positions `pos..pos + n` through the model (`n <= tokens()`);
    /// afterwards each token's draw, score, rows and trace are by its index. With
    /// `all_logits` every token's logits are computed (scoring, tracing); otherwise only the
    /// last token's — a prompt token's head is 1.27 GB of lm_head it never uses (2026-10-16).
    fn run(&mut self, toks: &[u32], pos: usize, trace: bool, all_logits: bool);
    /// Token `i`'s rows of the last step, `position()` f32s.
    fn rows(&self, i: usize, out: &mut [f32]);
    /// Whether the model carries a draft layer (multi-token prediction).
    fn drafts(&self) -> bool {
        false
    }
    /// After a step of `toks` at `pos`: the draft layer over those positions (each with the token
    /// that follows it — the step's next token, `next` after the last), its cache kept; `logits`
    /// receives the draft of the token after `next`. The sampler's constants (`sampling`) and
    /// `common::sample::verify` decide acceptance.
    fn draft(&mut self, _toks: &[u32], _next: u32, _pos: usize, _logits: &mut [f32]) {
        unreachable!("no draft layer")
    }
    /// Drafts a verify step carries (0 without a draft layer); `draft` makes the first, `draft_next` each further one.
    fn draft_depth(&self) -> usize {
        0
    }
    /// The next depth: the draft layer at `pos` from its own last output and `next`, the token the last
    /// draft proposed; `logits` receives the draft of the token after it.
    fn draft_next(&mut self, _next: u32, _pos: usize, _logits: &mut [f32]) {
        unreachable!("no draft layer")
    }
    /// The sequence state back to after token `slot` of the last verify-sized step (its later tokens rejected).
    fn rewind(&mut self, _slot: usize) {
        unreachable!("no verify")
    }
    fn sampling(&self) -> &'static common::sample::Sampling;
    /// The token drawn under the model's sampling from token `i`'s logits with the uniform `u`.
    fn draw(&mut self, i: usize, u: f64) -> u32;
    /// ln p(id) at the position after token `i`.
    fn score(&mut self, i: usize, id: u32) -> f32;
    /// Token `i`'s logits of the last step, and their replacement: a context's kept state
    /// carries the logits at its end, so a sequence resumed there draws without a step.
    fn logits_row(&self, i: usize) -> &[f32];
    fn set_logits(&mut self, i: usize, logits: &[f32]);
    /// The same two from the prefix's logits: what the first computed position is drawn or scored from.
    fn draw_prefix(&mut self, u: f64) -> u32;
    fn score_prefix(&mut self, id: u32) -> f32;
    /// The node is about to wait for input: whatever the engine keeps spinning may rest.
    fn idle(&mut self) {}
    /// Dev: token `i`'s trace of the last step — every layer's residual output,
    /// every layer's routing, its logits — in the model's own shapes.
    fn trace(&mut self, i: usize) -> (&[f32], &[u32], &[f32]);
}
