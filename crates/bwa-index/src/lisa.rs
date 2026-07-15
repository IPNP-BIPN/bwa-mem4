//! `LearnedSa`: BWA-MEME-style learned suffix-array exact search (Jung & Han 2022).
//!
//! BWA-MEME replaces bwa-mem2's FM-index backward search with a **plain suffix array** over the
//! bidirectional reference `[forward][revcomp]` plus a **learned index** (an [`Rmi`]) trained on the
//! first-`K` bases of every suffix. An exact-match lookup is: pack the query's first `K` bases into a
//! 64-bit key, ask the RMI for an approximate suffix-array position, then do a short bounded search
//! that compares the query against the reference suffixes (`compare_read_and_ref`) to pin the exact
//! SA interval `[lo, hi)`. That interval — the set of reference positions where the query occurs — is
//! the same object bwa-mem2's FM interval `[k, s]` denotes, so seeds built on it stay byte-identical.
//! The win: the number of memory accesses is independent of the match length (one key + a bounded
//! search), versus one `cp_occ` walk per base in the FM-index.
//!
//! This module is the exact-match core (validated against brute-force occurrence search). The
//! bidirectional interval and the SMEM "zigzag" driver that reproduce bwa-mem2's seeds byte-for-byte
//! build on top of it. See [[perf-phase9-levers]] / the LISA branch plan.

use crate::rmi::Rmi;
use crate::sais::suffix_array_with_sentinel;
use std::cmp::Ordering;

/// Bases packed into one learned key (BWA-MEME uses 32: a 64-bit 2-bit-packed key).
pub const K: usize = 32;

/// A suffix array over a binary reference (`0..=3` codes) plus a learned index over the first-`K`
/// bases of each suffix.
#[derive(Clone)]
pub struct LearnedSa {
    /// Binary reference, one base per byte in `0..=3` (the `.0123` string; `[forward][revcomp]` in
    /// the real pipeline, but `LearnedSa` itself is agnostic to what the bytes mean).
    ref_seq: Vec<u8>,
    /// Suffix array: length `ref_seq.len() + 1`, `sa[0] = ref_seq.len()` (the sentinel/empty suffix),
    /// `sa[1..]` the suffixes in lexicographic order.
    sa: Vec<i64>,
    /// `keys[i]` = first `K` bases of the suffix at `sa[i]`, 2-bit packed, most-significant base
    /// first (so key order == suffix order), zero-padded past the reference end.
    keys: Vec<u64>,
    rmi: Rmi,
}

/// First `K` bases at `pos` in `ref_seq`, 2-bit packed MSB-first, zero-padded past the end.
#[inline]
fn kmer_key(ref_seq: &[u8], pos: i64) -> u64 {
    let len = ref_seq.len();
    let p = pos as usize;
    let mut key = 0u64;
    for r in 0..K {
        let idx = p + r;
        let c = if idx < len { (ref_seq[idx] & 3) as u64 } else { 0 };
        key = (key << 2) | c;
    }
    key
}

/// First `K` bases of a query pattern (codes `0..=3`), packed the same way, zero-padded.
#[inline]
fn pattern_key(pattern: &[u8]) -> u64 {
    let mut key = 0u64;
    for r in 0..K {
        let c = if r < pattern.len() {
            (pattern[r] & 3) as u64
        } else {
            0
        };
        key = (key << 2) | c;
    }
    key
}

/// Find the first index in `[0, n)` where the monotone predicate `pred` (true…true, then
/// false…false) is `false`, seeded near `hint` with an exponential bracket then a binary search.
/// Correct for any `hint` (the bracket always contains the boundary); `hint` only affects speed.
fn seeded_partition_point<P: Fn(usize) -> bool>(n: usize, hint: usize, pred: P) -> usize {
    if n == 0 {
        return 0;
    }
    let h = hint.min(n - 1);
    let (mut a, mut b);
    if pred(h) {
        // Boundary is in (h, n]. Grow right until a false (or the end).
        a = h;
        let mut step = 1usize;
        loop {
            let probe = h.saturating_add(step);
            if probe >= n {
                b = n;
                break;
            }
            if !pred(probe) {
                b = probe;
                break;
            }
            a = probe;
            step *= 2;
        }
    } else {
        // Boundary is in [0, h]. Grow left until a true (or the start).
        b = h;
        let mut step = 1usize;
        loop {
            if h < step {
                a = 0;
                break;
            }
            let probe = h - step;
            if pred(probe) {
                a = probe;
                break;
            }
            b = probe;
            step *= 2;
        }
    }
    // Binary search for the first-false in [a, b]; boundary is guaranteed inside.
    while a < b {
        let mid = a + (b - a) / 2;
        if pred(mid) {
            a = mid + 1;
        } else {
            b = mid;
        }
    }
    a
}

impl LearnedSa {
    /// Build the suffix array, first-`K` keys, and learned index over `ref_seq` (codes `0..=3`).
    /// `n_leaves` sizes the RMI (a few thousand per million suffixes is reasonable).
    pub fn build(ref_seq: Vec<u8>, n_leaves: usize) -> Self {
        let sa = suffix_array_with_sentinel(&ref_seq);
        let keys: Vec<u64> = sa.iter().map(|&p| kmer_key(&ref_seq, p)).collect();
        let rmi = Rmi::build(&keys, n_leaves);
        LearnedSa {
            ref_seq,
            sa,
            keys,
            rmi,
        }
    }

    /// Number of suffix-array rows (including the sentinel row).
    pub fn len(&self) -> usize {
        self.sa.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ref_seq.is_empty()
    }

    /// Compare `pattern` against the reference suffix at SA row `i`, as a prefix comparison: equal
    /// means `pattern` is a prefix of that suffix. A suffix that ends before `pattern` does (running
    /// off the reference end) compares `Less` (shorter string sorts first), matching the sentinel.
    #[inline]
    fn cmp_pattern(&self, i: usize, pattern: &[u8]) -> Ordering {
        let start = self.sa[i] as usize;
        let len = self.ref_seq.len();
        for (j, &pc) in pattern.iter().enumerate() {
            let idx = start + j;
            if idx >= len {
                return Ordering::Less;
            }
            match self.ref_seq[idx].cmp(&pc) {
                Ordering::Equal => {}
                other => return other,
            }
        }
        Ordering::Equal
    }

    /// The suffix-array interval `[lo, hi)` of rows whose suffix has `pattern` as a prefix — i.e.
    /// every occurrence of `pattern` in the reference. Uses the learned index to seed the search.
    /// Empty pattern returns the whole array. Result-identical to a plain two-sided binary search
    /// over the suffix array (the learned prediction only narrows the window).
    pub fn exact_interval(&self, pattern: &[u8]) -> (usize, usize) {
        let n = self.sa.len();
        if pattern.is_empty() {
            return (0, n);
        }
        // RMI predicts where this key sorts among the stored first-K keys; a good seed for both ends.
        let hint = self.rmi.lower_bound(&self.keys, pattern_key(pattern));
        // lower bound: first row whose suffix is NOT < pattern.
        let lo = seeded_partition_point(n, hint, |i| self.cmp_pattern(i, pattern) == Ordering::Less);
        // upper bound: first row whose suffix is > pattern (prefix comparison).
        let hi =
            seeded_partition_point(n, hint, |i| self.cmp_pattern(i, pattern) != Ordering::Greater);
        (lo, hi)
    }

    /// Reference positions where `pattern` occurs exactly (the `sa` values of [`Self::exact_interval`]).
    pub fn occurrences(&self, pattern: &[u8]) -> Vec<i64> {
        let (lo, hi) = self.exact_interval(pattern);
        let mut v: Vec<i64> = self.sa[lo..hi].to_vec();
        v.sort_unstable();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force: every start position where `pattern` matches `ref_seq`.
    fn brute(ref_seq: &[u8], pattern: &[u8]) -> Vec<i64> {
        let n = ref_seq.len();
        let m = pattern.len();
        let mut v = Vec::new();
        if m == 0 || m > n {
            return v;
        }
        for s in 0..=(n - m) {
            if ref_seq[s..s + m] == *pattern {
                v.push(s as i64);
            }
        }
        v
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn exact_search_matches_bruteforce() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        // A few random references; include repetitive structure so intervals are non-trivial.
        for trial in 0..12 {
            let len = 200 + (lcg(&mut seed) as usize % 2000);
            let alpha = if trial % 3 == 0 { 2 } else { 4 }; // sometimes only {A,C} -> many repeats
            let ref_seq: Vec<u8> = (0..len)
                .map(|_| (lcg(&mut seed) % alpha) as u8)
                .collect();
            let lsa = LearnedSa::build(ref_seq.clone(), 256);

            // Probe patterns: random substrings of the reference (guaranteed to occur) of varied
            // length, plus random patterns (often absent), plus edge lengths around K=32.
            for _ in 0..300 {
                let mlen = 1 + (lcg(&mut seed) as usize % 40);
                let pattern: Vec<u8> = if lcg(&mut seed) % 2 == 0 && len > mlen {
                    let s = lcg(&mut seed) as usize % (len - mlen);
                    ref_seq[s..s + mlen].to_vec()
                } else {
                    (0..mlen).map(|_| (lcg(&mut seed) % alpha) as u8).collect()
                };
                let mut want = brute(&ref_seq, &pattern);
                want.sort_unstable();
                let got = lsa.occurrences(&pattern);
                assert_eq!(got, want, "ref_len={len} pattern={pattern:?}");
            }
        }
    }

    #[test]
    fn edge_cases() {
        let ref_seq = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // ACGTACGT
        let lsa = LearnedSa::build(ref_seq.clone(), 8);
        // Whole-string prefix.
        assert_eq!(lsa.occurrences(&[0, 1, 2, 3]), vec![0, 4]);
        // Single base.
        assert_eq!(lsa.occurrences(&[0]), vec![0, 4]);
        // Absent.
        assert!(lsa.occurrences(&[3, 3, 3]).is_empty());
        // Pattern longer than any suffix but matching the tail prefix-wise: "GT" + extra.
        assert!(lsa.occurrences(&[2, 3, 0, 1, 2, 3, 0, 1, 2]).is_empty());
        // Pattern longer than K (32) still works.
        let long_ref: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        let lsa2 = LearnedSa::build(long_ref.clone(), 16);
        let pat: Vec<u8> = (0..40).map(|i| ((i) % 4) as u8).collect(); // matches at position 0,4,...
        assert_eq!(lsa2.occurrences(&pat), brute(&long_ref, &pat));
    }
}
