//! One step's goldens as captured by `bringup/goldens.py`: f32 tensors and
//! int64 (or int32, widened) index tensors, `[T, ...]`, looked up by HF key.

use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct Golden {
    pub f32s: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub i64s: HashMap<String, (Vec<usize>, Vec<i64>)>,
}

impl Golden {
    pub fn load(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut g = Golden { f32s: HashMap::new(), i64s: HashMap::new() };
        for (name, t) in st.tensors() {
            let shape = t.shape().to_vec();
            match t.dtype() {
                Dtype::F32 => {
                    let v = t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    g.f32s.insert(name, (shape, v));
                }
                Dtype::I32 => {
                    let v = t.data().chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect();
                    g.i64s.insert(name, (shape, v));
                }
                Dtype::I64 => {
                    let v = t.data().chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
                    g.i64s.insert(name, (shape, v));
                }
                d => panic!("{name}: {d:?}"),
            }
        }
        g
    }
    /// Row `t` of a `[T, ...]` f32 tensor, flattened.
    pub fn row(&self, key: &str, t: usize) -> Option<&[f32]> {
        let (shape, v) = self.f32s.get(key)?;
        let n = v.len() / shape[0];
        Some(&v[t * n..(t + 1) * n])
    }
    /// Row `t` of a `[T, k]` int64 tensor.
    pub fn irow(&self, key: &str, t: usize) -> Option<&[i64]> {
        let (shape, v) = self.i64s.get(key)?;
        let n = v.len() / shape[0];
        Some(&v[t * n..(t + 1) * n])
    }
}
