//! Test-only helpers shared by every crate's tests: a seeded xorshift so
//! reproducible inputs mean the same thing in every file.

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    #[inline]
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1) with three decimals, as an f32.
    pub fn unit(&mut self) -> f32 {
        ((self.next() % 2000) as f32 - 1000.0) * 1e-3
    }
    /// INT4 in [-8, 7].
    pub fn i4(&mut self) -> i8 {
        (self.next() % 16) as i8 - 8
    }
}
