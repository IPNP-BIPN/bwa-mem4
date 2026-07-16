//! `Packed40`: a densely packed array of non-negative integers at **5 bytes each** (40 bits).
//!
//! Both the learned index's suffix array (positions `< 2^34` for hg38) and its co-located keys
//! (20 bases x 2 bits = 40 bits) fit in 40 bits, so storing them 5-byte instead of `u64` cuts each
//! from 49.6 GB to 31 GB at genome scale — the two biggest RAM items of the learned index. Values are
//! little-endian; only the low 5 bytes are stored.

use rayon::prelude::*;

/// A 5-byte-per-element array of values in `[0, 2^40)`.
#[derive(Clone)]
pub struct Packed40 {
    data: Vec<u8>,
    len: usize,
}

impl Packed40 {
    const W: usize = 5;
    const MAX: u64 = 1 << (8 * Self::W);

    #[inline]
    fn put(dst: &mut [u8], v: u64) {
        debug_assert!(v < Self::MAX, "value {v} exceeds 40 bits");
        dst.copy_from_slice(&v.to_le_bytes()[..Self::W]);
    }

    /// Pack an existing slice of values.
    pub fn from_slice(vals: &[i64]) -> Self {
        let len = vals.len();
        let mut data = vec![0u8; len * Self::W];
        for (i, &v) in vals.iter().enumerate() {
            Self::put(&mut data[i * Self::W..i * Self::W + Self::W], v as u64);
        }
        Packed40 { data, len }
    }

    /// Build `len` elements directly (in parallel) from `f(i)`, without ever materializing a `u64`
    /// array — used to compute the co-located keys straight into packed form (no memory spike).
    pub fn from_fn<F>(len: usize, f: F) -> Self
    where
        F: Fn(usize) -> u64 + Sync,
    {
        let mut data = vec![0u8; len * Self::W];
        data.par_chunks_mut(Self::W)
            .enumerate()
            .for_each(|(i, slot)| Self::put(slot, f(i)));
        Packed40 { data, len }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The value at `i`.
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        let o = i * Self::W;
        let mut b = [0u8; 8];
        b[..Self::W].copy_from_slice(&self.data[o..o + Self::W]);
        u64::from_le_bytes(b)
    }

    /// `partition_point` over `[lo, hi)`: count of leading elements where `pred(get(i))` holds (the
    /// predicate must be monotone true→false), matching `slice::partition_point`.
    #[inline]
    pub fn partition_point_in<P: Fn(u64) -> bool>(&self, lo: usize, hi: usize, pred: P) -> usize {
        let (mut a, mut b) = (lo, hi);
        while a < b {
            let mid = a + (b - a) / 2;
            if pred(self.get(mid)) {
                a = mid + 1;
            } else {
                b = mid;
            }
        }
        a - lo
    }

    /// Materialize `[lo, hi)` as `i64` (for suffix-array positions).
    pub fn range_i64(&self, lo: usize, hi: usize) -> Vec<i64> {
        (lo..hi).map(|i| self.get(i) as i64).collect()
    }

    /// Whether the whole array is sorted ascending (debug checks only; O(n)).
    pub fn is_sorted(&self) -> bool {
        (1..self.len).all(|i| self.get(i - 1) <= self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_partition_and_from_fn() {
        let vals: Vec<i64> = vec![0, 5, 5, 9, 12, 12, 100, 1_000_000, (1i64 << 34) - 1];
        let p = Packed40::from_slice(&vals);
        assert_eq!(p.len(), vals.len());
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(p.get(i), v as u64, "elem {i}");
        }
        assert_eq!(p.range_i64(1, 4), vec![5, 5, 9]);
        assert!(p.is_sorted());
        // partition_point matches slice::partition_point on thresholds and sub-ranges.
        for thr in [0u64, 5, 6, 12, 13, 1_000_000, u64::MAX] {
            for &(lo, hi) in &[(0usize, vals.len()), (1, 6), (3, 8), (0, 1)] {
                let want = vals[lo..hi].partition_point(|&x| (x as u64) < thr);
                let got = p.partition_point_in(lo, hi, |x| x < thr);
                assert_eq!(got, want, "thr={thr} range=({lo},{hi})");
            }
        }
        // from_fn builds the same array.
        let q = Packed40::from_fn(vals.len(), |i| vals[i] as u64);
        for i in 0..vals.len() {
            assert_eq!(q.get(i), p.get(i));
        }
    }
}
