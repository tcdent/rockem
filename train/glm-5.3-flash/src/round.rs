//! A training round, as the node will run it behind the wire's TRAIN: sessions arrive as ids
//! with the teacher's top-K attached to the positions that carry a loss (the assistant's);
//! each session is read forward in runs of `run_len` positions — states advancing, the KDA
//! states snapshotted at every run boundary, nothing else kept — and, when it ends, read
//! backward run by run in reverse: the run's states restored from its snapshot, its forward
//! recomputed with everything the backward needs, the loss taken where the targets are, the
//! backward run, the gradient carried into the run before. Then a step over every expert the
//! session touched.
//!
//! Memory is therefore one run's intermediates (~50 MB a position: the streams at each half of
//! every layer, KDA's and MLA's per-position state, the experts' activations) plus a KDA
//! snapshot a run boundary (~140 MB), never a session's. The MLA cache is the layer's own and
//! is rewritten identically by a recompute. The price is one more forward a session.
//!
//! A session starts from zero states at position 0. The node serves from the bundle's sink —
//! the shipped state after the `[gMASK]<sop>` prefix — which this computes instead of loading
//! (the same positions, in f32, so a difference is at most rounding; to be gated against the
//! served sink when the trainer meets a real bundle).

use crate::expert::Adapter;
use crate::loss::{forward_kl_topk, Teacher};
use crate::model::{self, Adapters, Carry, KdaSnapshot, Scratch, States};
use crate::optim::{self, ExpertState, Movement, Settings};
use glm_5_3_flash::expert::{Stream, Weights};
use glm_5_3_flash::weights::{Ffn, Weights as W};
use glm_5_3_flash::E;

/// A position's teacher: the top-K ids and their log-probabilities.
#[derive(Clone)]
pub struct Target {
    pub ids: Vec<u32>,
    pub logp: Vec<f32>,
}

/// The round's resident state: the adapters being trained, their gradients, the optimizer's
/// per-expert state, the sequence's layer states, and the session in flight.
pub struct Round<'a> {
    pub w: &'a W<'a>,
    pub stream: &'a Stream<'a>,
    pub ad: Adapters,
    pub grads: Adapters,
    pub opt: Vec<Option<Vec<Option<ExpertState>>>>, // [layer][expert], made when first touched
    pub settings: Settings,
    pub vocab: usize,
    pub run_len: usize,
    pub max_ctx: usize,
    states: States,
    scratch: Scratch,
    // the session in flight
    ids: Vec<u32>,
    targets: Vec<Option<Target>>,
    fed: usize,                  // positions already run forward
    snapshots: Vec<KdaSnapshot>, // the states before run i (run i covers i·run_len..)
    touched: Vec<Vec<bool>>,     // [layer][expert] since the last step
    /// The loss summed over the positions with targets since the last step, and their count.
    pub loss: f64,
    pub positions: usize,
}

/// What a step reports: per touched expert its movement, and the mean loss over the positions.
pub struct Report {
    pub loss: f32,
    pub positions: usize,
    pub moved: Vec<(usize, usize, Movement)>, // (layer, expert, movement)
}

impl<'a> Round<'a> {
    pub fn new(w: &'a W<'a>, stream: &'a Stream<'a>, ad: Adapters, settings: Settings, vocab: usize, run_len: usize, max_ctx: usize) -> Self {
        let r = ad.rank();
        let grads = Adapters::zeros(w, r);
        let opt = w.layers.iter().map(|l| matches!(l.ffn, Ffn::Moe(_)).then(|| (0..E).map(|_| None).collect())).collect();
        let touched = w.layers.iter().map(|l| if matches!(l.ffn, Ffn::Moe(_)) { vec![false; E] } else { Vec::new() }).collect();
        Round { w, stream, ad, grads, opt, settings, vocab, run_len, max_ctx, states: States::new(w, max_ctx), scratch: Scratch::new(r, max_ctx), ids: Vec::new(), targets: Vec::new(), fed: 0, snapshots: Vec::new(), touched, loss: 0.0, positions: 0 }
    }

    /// A new session: states reset, the one in flight (if any) must have been ended.
    pub fn begin(&mut self) {
        assert!(self.ids.is_empty(), "end the session in flight first");
        self.states.reset();
        self.fed = 0;
        self.snapshots.clear();
    }

    /// One position: its id, and the teacher's target if a loss is taken here. Runs forward as
    /// soon as a run's worth has arrived.
    pub fn feed(&mut self, id: u32, target: Option<Target>) {
        assert!(self.ids.len() < self.max_ctx, "the session exceeds the context budget");
        self.ids.push(id);
        self.targets.push(target);
        if self.ids.len() - self.fed == self.run_len {
            self.advance();
        }
    }

    /// Run the positions not yet fed forward (states only), snapshotting first.
    fn advance(&mut self) {
        let (a, b) = (self.fed, self.ids.len());
        if a == b {
            return;
        }
        self.snapshots.push(self.states.snapshot());
        let mut none = [0f32; 0];
        model::forward(self.w, self.stream, &self.ad, &self.ids[a..b], a, self.vocab, &[], &mut self.states, &mut self.scratch, &mut none);
        self.fed = b;
    }

    /// The session ends: the backward over its runs in reverse, each recomputed from its snapshot.
    pub fn end(&mut self) {
        self.advance();
        let runs = self.snapshots.len();
        let mut carry = Carry::zeros(self.w, self.max_ctx);
        let mut dlogits = Vec::new();
        for i in (0..runs).rev() {
            let (a, b) = (i * self.run_len, ((i + 1) * self.run_len).min(self.ids.len()));
            self.states.restore(&self.snapshots[i]);
            let at: Vec<usize> = (a..b).filter(|&p| self.targets[p].is_some()).map(|p| p - a).collect();
            let mut logits = vec![0f32; at.len() * self.vocab];
            let sv = model::forward(self.w, self.stream, &self.ad, &self.ids[a..b], a, self.vocab, &at, &mut self.states, &mut self.scratch, &mut logits);
            dlogits.clear();
            dlogits.resize(at.len() * self.vocab, 0.0);
            for (j, &t) in at.iter().enumerate() {
                let tg = self.targets[a + t].as_ref().unwrap();
                self.loss += forward_kl_topk(&logits[j * self.vocab..(j + 1) * self.vocab], &Teacher { ids: &tg.ids, logp: &tg.logp }, &mut dlogits[j * self.vocab..(j + 1) * self.vocab]) as f64;
                self.positions += 1;
            }
            // the experts this run routed to, for the step
            for (l, ls) in sv.layers.iter().enumerate() {
                if let Some(f) = &ls.ffn {
                    for s in f {
                        for &e in &s.idx {
                            self.touched[l][e] = true;
                        }
                    }
                }
            }
            model::backward(self.w, self.stream, &self.ad, &sv, &self.states, self.vocab, &dlogits, &mut self.scratch, &mut self.grads, &mut carry);
        }
        self.ids.clear();
        self.targets.clear();
        self.fed = 0;
        self.snapshots.clear();
    }

    /// Adam under the cap on every expert touched since the last step; the gradients are consumed.
    pub fn step(&mut self) -> Report {
        let mut moved = Vec::new();
        for l in 0..self.w.layers.len() {
            let Some(layer_opt) = self.opt[l].as_mut() else { continue };
            for e in 0..E {
                if !self.touched[l][e] {
                    continue;
                }
                self.touched[l][e] = false;
                let st = layer_opt[e].get_or_insert_with(|| ExpertState::new(self.ad.rank(), optim::base_norms(&Weights::at(self.stream, l, e))));
                let mv = optim::step(&self.settings, st, &mut self.ad.layers[l].as_mut().unwrap()[e], &mut self.grads.layers[l].as_mut().unwrap()[e]);
                moved.push((l, e, mv));
            }
        }
        let report = Report { loss: if self.positions > 0 { (self.loss / self.positions as f64) as f32 } else { 0.0 }, positions: self.positions, moved };
        self.loss = 0.0;
        self.positions = 0;
        report
    }

    /// The adapters as they stand (the entry in progress).
    pub fn adapters(&self) -> &Adapters {
        &self.ad
    }
    pub fn adapter(&self, l: usize, e: usize) -> Option<&Adapter> {
        self.ad.layers[l].as_ref().map(|a| &a[e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::tests::random_adapter;
    use crate::fixtures::{Bufs, Experts};
    use crate::testing::unit;
    use common::testing::Rng;
    use glm_5_3_flash::{FIRST_K_DENSE_REPLACE, TOP_K};
    use train_common::adam::Adam;

    fn grads_of(r: &Round) -> Vec<f32> {
        r.grads.layers.iter().flatten().flat_map(|l| l.iter().flat_map(|a| a.ga.iter().chain(&a.gb).chain(&a.ua).chain(&a.ub).chain(&a.da).chain(&a.db).copied())).collect()
    }

    /// A session fed in runs of 3 and the same session as one run give the same gradient: the
    /// snapshots, the recompute and the carry across runs are exact. Then a step reports.
    #[test]
    fn runs_recompose_the_session() {
        let mut rnd = Rng::new(0x5E55_0001);
        let vocab = 4096;
        let bufs = Bufs::new(vocab, &mut rnd);
        let w = bufs.weights();
        let experts = Experts::new(&mut rnd);
        let stream = experts.stream();
        let r = 4;
        let mut ad = Adapters::zeros(&w, r);
        for l in [FIRST_K_DENSE_REPLACE, FIRST_K_DENSE_REPLACE + 1] {
            for e in 0..TOP_K {
                ad.layers[l].as_mut().unwrap()[e] = random_adapter(r, 0.05, &mut rnd);
            }
        }
        let n = 7;
        let ids: Vec<u32> = (0..n).map(|_| (rnd.next() % vocab as u64) as u32).collect();
        let targets: Vec<Option<Target>> = (0..n).map(|t| (t % 2 == 1).then(|| Target { ids: (0..8).map(|_| (rnd.next() % vocab as u64) as u32).collect(), logp: unit(8, 2.0, &mut rnd) })).collect();
        let settings = Settings { adam: Adam::default(), cap: 1e-3 };

        let session = |run_len: usize, ad: &Adapters| -> (Vec<f32>, f64, usize) {
            let mut round = Round::new(&w, &stream, Adapters { layers: ad.layers.iter().map(|l| l.clone()).collect() }, settings, vocab, run_len, 16);
            round.begin();
            for t in 0..n {
                round.feed(ids[t], targets[t].clone());
            }
            round.end();
            (grads_of(&round), round.loss, round.positions)
        };
        let (g1, l1, p1) = session(n, &ad);
        let (g3, l3, p3) = session(3, &ad);
        assert_eq!((p1, p3), (3, 3), "positions 1, 3, 5 carry targets");
        assert!((l1 - l3).abs() <= 1e-6 * l1.abs(), "loss {l1} vs {l3}");
        let scale = g1.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = g1.iter().zip(&g3).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(worst <= 1e-5 * scale, "gradients differ by {worst} against a scale of {scale}");

        // a step on the run-fed round: every touched expert moves, within the cap
        let mut round = Round::new(&w, &stream, ad, settings, vocab, 3, 16);
        round.begin();
        for t in 0..n {
            round.feed(ids[t], targets[t].clone());
        }
        round.end();
        let rep = round.step();
        assert_eq!(rep.positions, 3);
        assert_eq!(rep.moved.len(), 2 * TOP_K, "both MoE layers' eight experts");
        for (l, e, mv) in &rep.moved {
            assert!(mv.scale > 0.0 && mv.scale <= 1.0 && mv.distance > 0.0, "layer {l} expert {e}: {mv:?}");
            let realized = mv.proposed * mv.scale;
            assert!(realized <= settings.cap * 1.01, "layer {l} expert {e}: moved {realized} past the cap");
        }
        assert!(grads_of(&round).iter().all(|v| *v == 0.0), "the step consumed the gradients");
    }
}
