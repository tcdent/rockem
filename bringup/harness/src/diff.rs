//! Edge diffing. `Collect` is a `common::trace::Trace` that keeps every edge of one
//! token; `Report` accumulates the worst error per edge *class* (layer index
//! collapsed to `N`) and prints one table. Relative error is against the
//! max |reference| of the row, which is the number that survives a layer
//! count and a quantization change alike.

use common::trace::{Trace, NONE};
use std::collections::{BTreeMap, HashMap};

#[derive(Default)]
pub struct Collect {
    pub edges: Vec<(String, Vec<f32>)>,
    pub topk: HashMap<usize, (Vec<usize>, Vec<f32>)>,
    pub routed: HashMap<usize, Vec<Vec<f32>>>,
}

impl Trace for Collect {
    fn edge(&mut self, layer: usize, name: &str, data: &[f32]) {
        let key = if layer == NONE { name.to_string() } else { format!("model.layers.{layer}.{name}") };
        self.edges.push((key, data.to_vec()));
    }
    fn topk(&mut self, layer: usize, idx: &[usize], weight: &[f32]) {
        self.topk.insert(layer, (idx.to_vec(), weight.to_vec()));
    }
    fn routed(&mut self, layer: usize, k: usize, data: &[f32]) {
        let v = self.routed.entry(layer).or_default();
        assert_eq!(v.len(), k);
        v.push(data.to_vec());
    }
}

#[derive(Default, Clone, Debug)]
pub struct Worst {
    pub abs: f32,
    pub rel: f32,
    pub at: String,
    pub rows: usize,
}

#[derive(Default)]
pub struct Report {
    pub worst: BTreeMap<String, Worst>,
    /// Rows whose relative error exceeds this are flagged `!!` in the table.
    pub tol: f32,
}

pub fn class_of(key: &str) -> String {
    let mut parts: Vec<&str> = key.split('.').collect();
    if parts.len() > 2 && parts[1] == "layers" {
        parts[2] = "N";
    }
    parts.join(".")
}

/// (max |a−b|, max |b|) over a row.
pub fn err(mine: &[f32], gold: &[f32]) -> (f32, f32) {
    assert_eq!(mine.len(), gold.len());
    let mut abs = 0f32;
    let mut scale = 0f32;
    for (a, b) in mine.iter().zip(gold) {
        abs = abs.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    (abs, scale)
}

impl Report {
    pub fn new(tol: f32) -> Self {
        Report { worst: BTreeMap::new(), tol }
    }

    /// Record one row under `key`; returns (abs, rel) for callers that want to print inline.
    pub fn record(&mut self, key: &str, mine: &[f32], gold: &[f32], at: &str) -> (f32, f32) {
        let (abs, scale) = err(mine, gold);
        let rel = if scale > 0.0 { abs / scale } else { abs };
        let w = self.worst.entry(class_of(key)).or_default();
        w.rows += 1;
        if rel > w.rel {
            let rows = w.rows;
            *w = Worst { abs, rel, at: format!("{key} {at}"), rows };
        }
        (abs, rel)
    }

    /// A hard failure that has no numeric row (a routing set mismatch).
    pub fn fail(&mut self, key: &str, what: String) {
        let w = self.worst.entry(class_of(key)).or_default();
        w.rel = f32::INFINITY;
        w.at = what;
    }

    pub fn failures(&self) -> usize {
        self.worst.values().filter(|w| w.rel > self.tol).count()
    }

    pub fn print(&self) {
        println!("\n{:<48} {:>10} {:>10} {:>6}  worst at", "edge class", "max|Δ|", "rel", "rows");
        for (k, w) in &self.worst {
            let flag = if w.rel > self.tol { "!!" } else if w.rel > self.tol / 10.0 { " !" } else { "  " };
            println!("{flag}{:<46} {:>10.3e} {:>10.3e} {:>6}  {}", k, w.abs, w.rel, w.rows, w.at);
        }
    }
}
