//! Edge observer threaded through every fused kernel. The node installs
//! `NoTrace`, which compiles to nothing; bring-up installs a collector and
//! diffs by name against oracle goldens. Names are HF module paths relative
//! to the layer (or to the model when `layer == NONE`).

pub const NONE: usize = usize::MAX;

pub trait Trace {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]);
    fn topk(&mut self, layer: usize, idx: &[usize], weight: &[f32]);
    fn routed(&mut self, layer: usize, k: usize, data: &[f32]);
}

impl<T: Trace + ?Sized> Trace for &mut T {
    #[inline(always)]
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        (**self).edge(layer, name, data)
    }
    #[inline(always)]
    fn topk(&mut self, layer: usize, idx: &[usize], weight: &[f32]) {
        (**self).topk(layer, idx, weight)
    }
    #[inline(always)]
    fn routed(&mut self, layer: usize, k: usize, data: &[f32]) {
        (**self).routed(layer, k, data)
    }
}

pub struct NoTrace;
impl Trace for NoTrace {
    #[inline(always)]
    fn edge(&mut self, _: usize, _: &str, _: &[f32]) {}
    #[inline(always)]
    fn topk(&mut self, _: usize, _: &[usize], _: &[f32]) {}
    #[inline(always)]
    fn routed(&mut self, _: usize, _: usize, _: &[f32]) {}
}
