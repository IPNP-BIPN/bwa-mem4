//! `Packed40`: a densely packed array of non-negative integers at **5 bytes each** (40 bits).
//!
//! Both the learned index's suffix array (positions `< 2^34` for hg38) and its co-located keys
//! (20 bases x 2 bits = 40 bits) fit in 40 bits, so storing them 5-byte instead of `u64` cuts each
//! from 49.6 GB to 31 GB at genome scale — the two biggest RAM items of the learned index. Values are
//! little-endian; only the low 5 bytes are stored.

use rayon::prelude::*;

/// A 5-byte-per-element array of values in `[0, 2^40)`.
///
/// # Layout (in memory only; this never reaches disk, so there is no format to keep compatible)
///
/// Element `i` occupies bytes `5i .. 5i+5`, little-endian, low 5 bytes of the `u64` only. No
/// padding, no alignment: `data.len() == 5 * len` exactly, and elements straddle machine-word
/// boundaries freely.
///
/// Worked micro-example, `v = 0x01_2345_6789`:
///   `v.to_le_bytes()` = `[89, 67, 45, 23, 01, 00, 00, 00]`, and the first 5 are stored as
///   `89 67 45 23 01`. Reading back zero-fills the top 3 bytes, so the round-trip is exact for any
///   value that fits.
///
/// INVARIANT: every stored value must be `< 2^40`. `put` only `debug_assert!`s this, so in release
/// a larger value is SILENTLY TRUNCATED to its low 40 bits, producing a wrong but well-formed
/// array. Callers are responsible: suffix-array positions are bounded by `2L + 1` and packed keys
/// by `2 * K <= 40` bits, both checked at their construction sites.
///
/// INVARIANT for [`Packed40::partition_point_in`]: `lo <= hi <= len`, and the predicate must be
/// monotone (all true, then all false) over `[lo, hi)`. A non-monotone predicate returns an
/// arbitrary index rather than failing.
#[derive(Clone)]
pub struct Packed40 {
    /// The raw byte backing store, exactly `5 * len` bytes, no padding and no alignment
    /// requirement. Element `i`'s five little-endian bytes are `data[5i .. 5i+5]`. Written only by
    /// the constructors ([`Packed40::from_slice`], [`Packed40::from_fn`]); after construction the
    /// array is immutable, which is why `get` needs no bounds bookkeeping beyond the slice index.
    data: Vec<u8>,
    /// Number of ELEMENTS (not bytes) stored. Always `data.len() / 5`. Valid element indices are
    /// `0..len`. Read by `len`, `is_empty`, `is_sorted` and by callers computing binary-search
    /// bounds; the two are kept in step only by the constructors, so they must never disagree.
    len: usize,
}

impl Packed40 {
    /// Bytes per element. 5 is the smallest width covering a 40-bit value; 4 would cap at 4.3e9,
    /// below the ~6.2e9 positions of a doubled human reference.
    ///
    /// Changing this changes the whole layout: `MAX` is derived from it, `from_slice`/`from_fn`
    /// allocate `len * W`, and `get`'s `buf[..W]` fill widens or narrows with it. Raising it to 6
    /// would still work (6 <= 8, so the `to_le_bytes()[..W]` and `buf[..W]` slicing stays in
    /// range) at the cost of 20% more RAM; lowering it below the bits actually needed silently
    /// truncates every stored value. Nothing on disk depends on it: this type is memory-only.
    const W: usize = 5;
    /// Exclusive upper bound on a storable value, `2^40 = 1_099_511_627_776`. Derived from `W`, so
    /// it tracks any width change automatically. Enforced only by a `debug_assert!` in `put`.
    ///
    /// Worked micro-example of the layout this bound goes with: element 3 starts at byte
    /// `3 * 5 = 15` and owns bytes 15..20. A value `V` is stored as
    /// `data[15] = V & 0xff`, `data[16] = (V >> 8) & 0xff`, ... `data[19] = (V >> 32) & 0xff`
    /// (little-endian, least significant byte first), and read back as
    /// `V = data[15] | data[16]<<8 | data[17]<<16 | data[18]<<24 | data[19]<<32`.
    const MAX: u64 = 1 << (8 * Self::W);

    /// Store `v` into a 5-byte slot. `dst` must be exactly `W` bytes: `copy_from_slice` panics on
    /// any other length, which is the real (non-debug) guard against a miscomputed offset.
    ///
    /// # Parameters
    /// * `dst`: the destination slot, which MUST be exactly `W` (5) bytes long, normally
    ///   `data[i*W .. i*W+W]` for the element index `i` being written. Supplied by the
    ///   constructors; any other length panics inside `copy_from_slice`.
    /// * `v`: the value to store, valid range `[0, MAX)` i.e. `[0, 2^40)`. In release builds a
    ///   larger value is silently truncated to its low 40 bits, so the caller owns this bound.
    #[inline]
    fn put(dst: &mut [u8], v: u64) {
        debug_assert!(v < Self::MAX, "value {v} exceeds 40 bits");
        dst.copy_from_slice(&v.to_le_bytes()[..Self::W]);
    }

    /// Pack an existing slice of values.
    ///
    /// # Parameters
    /// * `vals`: the values in element order, each of which must be non-negative and `< 2^40` once
    ///   cast to `u64`. The `as u64` cast means a negative `i64` becomes a huge value and is then
    ///   truncated, so callers must not pass negatives. In practice these are suffix-array text
    ///   positions (bounded by `2L + 1`) or packed 20-base keys (40 bits).
    ///
    /// # Returns
    /// A `Packed40` with `len == vals.len()`, where `get(i) == vals[i] as u64` for every `i`.
    pub fn from_slice(vals: &[i64]) -> Self {
        let len = vals.len();
        // 5 bytes per element, zero-initialised; every byte is overwritten by the loop below, the
        // zero fill is only to get a correctly sized allocation.
        let mut data = vec![0u8; len * Self::W];
        for (i, &v) in vals.iter().enumerate() {
            Self::put(&mut data[i * Self::W..i * Self::W + Self::W], v as u64);
        }
        Packed40 { data, len }
    }

    /// Build `len` elements directly (in parallel) from `f(i)`, without ever materializing a `u64`
    /// array — used to compute the co-located keys straight into packed form (no memory spike).
    ///
    /// # Parameters
    /// * `len`: number of ELEMENTS to build; the allocation is `len * 5` bytes.
    /// * `f`: called once per element index `i` in `0..len`, returning that element's value, which
    ///   must be `< 2^40`. Must be `Sync` and side-effect-free with respect to ordering: rayon
    ///   calls it from many threads in an unspecified order, though each `i` is visited exactly
    ///   once and writes only its own slot.
    ///
    /// # Returns
    /// A `Packed40` of `len` elements with `get(i) == f(i)`.
    pub fn from_fn<F>(len: usize, f: F) -> Self
    where
        F: Fn(usize) -> u64 + Sync,
    {
        let mut data = vec![0u8; len * Self::W];
        // Each chunk is one element's 5-byte slot, so the chunk index from `enumerate` IS the
        // element index; slots are disjoint, hence the parallel write is race-free.
        data.par_chunks_mut(Self::W)
            .enumerate()
            .for_each(|(i, slot)| Self::put(slot, f(i)));
        Packed40 { data, len }
    }

    /// Number of elements stored (not bytes; the byte size is `5 *` this).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The value at `i`. Panics (via the slice index) if `i >= len`.
    ///
    /// Reassembles through a zeroed 8-byte buffer rather than an unaligned 8-byte load, because the
    /// last element has only 5 readable bytes and an 8-byte load there would run past the
    /// allocation. The zero fill is also what guarantees the top 24 bits come back clear.
    ///
    /// # Parameters
    /// * `i`: ELEMENT index, valid range `0..len`. Out of range panics on the slice index.
    ///
    /// # Returns
    /// The stored value, always in `[0, 2^40)` since the top 3 bytes are zero-filled.
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        // First byte of element `i`'s slot; the slot is `byte_offset .. byte_offset + 5`.
        let byte_offset = i * Self::W;
        // Zeroed 8-byte staging buffer: bytes 5..8 stay zero and become the value's high 24 bits.
        let mut buf = [0u8; 8];
        buf[..Self::W].copy_from_slice(&self.data[byte_offset..byte_offset + Self::W]);
        u64::from_le_bytes(buf)
    }

    /// `partition_point` over `[lo, hi)`: count of leading elements where `pred(get(i))` holds (the
    /// predicate must be monotone true→false), matching `slice::partition_point`.
    ///
    /// # Parameters
    /// * `lo`: inclusive start ELEMENT index of the searched sub-range; must satisfy `lo <= hi`.
    /// * `hi`: exclusive end ELEMENT index; must satisfy `hi <= len`.
    /// * `pred`: tested on element VALUES (not indices). Must be monotone over `[lo, hi)`: true
    ///   for a prefix, false for the rest. A non-monotone predicate yields an arbitrary result
    ///   rather than a panic.
    ///
    /// # Returns
    /// The COUNT of leading elements in `[lo, hi)` satisfying `pred`, i.e. relative to `lo`, not an
    /// absolute index. Add `lo` to get the absolute boundary index.
    #[inline]
    pub fn partition_point_in<P: Fn(u64) -> bool>(&self, lo: usize, hi: usize, pred: P) -> usize {
        // Standard binary search for the true/false boundary. `a + (b - a) / 2` rather than
        // `(a + b) / 2` so the midpoint cannot overflow on a genome-scale array.
        //
        // Invariant at the top of each iteration: every element in `[lo, first_unknown)` satisfies
        // `pred`, every element in `[past_last, hi)` does not, and the boundary lies somewhere in
        // `[first_unknown, past_last]`. The range shrinks each pass, so the loop terminates with
        // `first_unknown == past_last == boundary`.
        let (mut first_unknown, mut past_last) = (lo, hi);
        while first_unknown < past_last {
            // Probe element, strictly inside `[first_unknown, past_last)`, so the range always
            // shrinks whichever branch is taken.
            let mid = first_unknown + (past_last - first_unknown) / 2;
            if pred(self.get(mid)) {
                first_unknown = mid + 1;
            } else {
                past_last = mid;
            }
        }
        first_unknown - lo
    }

    /// Materialize `[lo, hi)` as `i64` (for suffix-array positions).
    ///
    /// # Parameters
    /// * `lo`, `hi`: inclusive/exclusive ELEMENT indices, `lo <= hi <= len`. An `hi > len` panics
    ///   inside `get`; `lo > hi` simply yields an empty vector.
    ///
    /// # Returns
    /// `hi - lo` values in element order. The `as i64` cast is lossless because stored values are
    /// under `2^40`, far below `i64::MAX`.
    pub fn range_i64(&self, lo: usize, hi: usize) -> Vec<i64> {
        (lo..hi).map(|i| self.get(i) as i64).collect()
    }

    /// Whether the whole array is sorted ascending (debug checks only; O(n)).
    ///
    /// # Returns
    /// True if `get(i-1) <= get(i)` for all `i`, non-strict so equal neighbours are allowed. An
    /// empty or single-element array is trivially sorted.
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
