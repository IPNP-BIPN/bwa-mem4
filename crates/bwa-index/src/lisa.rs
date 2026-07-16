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

use crate::packed::Packed40;
use crate::rmi::Rmi;
use crate::sais::suffix_array_with_sentinel;
use std::cmp::Ordering;

/// Bases packed into one learned key. 20 (40 bits, one `Packed40` element) is selective enough at
/// genome scale — 4^20 ≫ 6.2G positions, so almost no two suffixes share a key, keeping the co-located
/// comparison a single memory access — while halving the key RAM vs a 32-base `u64` (49.6 GB → 31 GB).
/// K only affects *speed* (key selectivity + RMI hint precision); the search result is always corrected
/// against the reference, so any K stays byte-identical.
pub const K: usize = 20;

/// A suffix array over a binary reference (`0..=3` codes) plus a learned index over the first-`K`
/// bases of each suffix.
#[derive(Clone)]
pub struct LearnedSa {
    /// Binary reference, one base per byte in `0..=3` (the `.0123` string; `[forward][revcomp]` in
    /// the real pipeline, but `LearnedSa` itself is agnostic to what the bytes mean).
    ref_seq: Vec<u8>,
    /// Suffix array (5-byte packed): length `ref_seq.len() + 1`, `sa[0] = ref_seq.len()` (the
    /// sentinel/empty suffix), `sa[1..]` the suffixes in lexicographic order.
    sa: Packed40,
    /// `keys[i]` = first `K` bases of the suffix at `sa[i]` (5-byte packed, 40 bits), most-significant
    /// base first (so key order == suffix order), zero-padded past the reference end.
    keys: Packed40,
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
        Self::from_sa(ref_seq, sa, n_leaves)
    }

    /// Build from a **prebuilt** suffix array (skipping SAIS): `sa` must be
    /// `suffix_array_with_sentinel(&ref_seq)` — e.g. extracted from an existing FM-index via
    /// `FmIndex::get_sa` for every row, which is byte-identical to running SAIS but far cheaper on a
    /// genome-scale reference and guarantees `sa[i] == fm.get_sa(i)`. Computes the first-`K` keys and
    /// trains the RMI over them.
    pub fn from_sa(ref_seq: Vec<u8>, sa: Vec<i64>, n_leaves: usize) -> Self {
        debug_assert_eq!(sa.len(), ref_seq.len() + 1, "sa must include the sentinel row");
        // Pack the SA to 5 bytes and drop the i64 array; then compute the keys **directly** into a
        // 5-byte packed array (no intermediate `Vec<u64>`), so the peak is `packed_sa + packed_keys`
        // (~62 GB at genome scale) rather than three full copies.
        let packed = Packed40::from_slice(&sa);
        drop(sa);
        let keys = Packed40::from_fn(packed.len(), |i| kmer_key(&ref_seq, packed.get(i) as i64));
        let rmi = Rmi::build(&keys, n_leaves);
        LearnedSa {
            ref_seq,
            sa: packed,
            keys,
            rmi,
        }
    }

    /// Number of suffix-array rows (including the sentinel row).
    pub fn len(&self) -> usize {
        self.sa.len()
    }

    /// The suffix-array value at `row` (a reference position in the `[fwd][rc]` space). Equals
    /// `FmIndex::get_sa(row)` over the same reference, so seeds materialized as `sa_at(k + i)` are
    /// byte-identical to the FM path.
    #[inline]
    pub fn sa_at(&self, row: usize) -> i64 {
        self.sa.get(row) as i64
    }

    pub fn is_empty(&self) -> bool {
        self.ref_seq.is_empty()
    }

    /// Compare `pattern` against the reference suffix at SA row `i`, as a prefix comparison: equal
    /// means `pattern` is a prefix of that suffix. A suffix that ends before `pattern` does (running
    /// off the reference end) compares `Less` (shorter string sorts first), matching the sentinel.
    #[inline]
    fn cmp_pattern(&self, i: usize, pattern: &[u8]) -> Ordering {
        let start = self.sa.get(i) as usize;
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

    /// Compare `pattern` against the suffix at SA row `i` using the **co-located 2-bit key**
    /// (`keys[i]`, one memory access) for the first `min(len, K)` bases, and only touch the reference
    /// on a tie. This is BWA-MEME's co-located-suffix comparison: the last-mile binary search resolves
    /// almost every step from a single cache line (`keys[i]`) instead of two random accesses
    /// (`sa[i]` → `ref_seq[start..]`). `pkey` is `pattern_key(pattern)`, precomputed once per search.
    ///
    /// Correctness: keys are 2-bit packed MSB-first, zero-padded past a suffix's end. Padding with `A`
    /// (0, the smallest base) can only make a short suffix compare *smaller*, never falsely `Greater`,
    /// so a decisive `Less`/`Greater` from the key is always correct; a key tie (`Equal`) is re-verified
    /// against the reference (handles short/padded suffixes and matches longer than `K`).
    #[inline]
    fn cmp_key(&self, i: usize, pkey: u64, pattern: &[u8]) -> Ordering {
        let l = pattern.len().min(K);
        let shift = 2 * (K - l) as u32; // keep only the top `l` bases of each key
        match (self.keys.get(i) >> shift).cmp(&(pkey >> shift)) {
            Ordering::Equal => self.cmp_pattern(i, pattern),
            other => other,
        }
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
        let pkey = pattern_key(pattern);
        let hint = self.rmi.lower_bound(&self.keys, pkey);
        // lower bound: first row whose suffix is NOT < pattern.
        let lo = seeded_partition_point(n, hint, |i| self.cmp_key(i, pkey, pattern) == Ordering::Less);
        // upper bound: first row whose suffix is > pattern (prefix comparison).
        let hi =
            seeded_partition_point(n, hint, |i| self.cmp_key(i, pkey, pattern) != Ordering::Greater);
        (lo, hi)
    }

    /// The bwa-mem2 bidirectional FMD interval `(k, l, s)` for `pattern`, reproduced from the plain
    /// `[fwd][rc]` suffix array (validated byte-identical to `FmIndex::backward_ext` walked over the
    /// same pattern, `bidirectional_interval_matches_fmindex`):
    ///   * `k` = SA lower-bound of `pattern` (forward interval start),
    ///   * `s` = interval size (occurrence count),
    ///   * `l` = SA lower-bound of `revcomp(pattern)` (reverse-complement interval start).
    /// `revcomp` here is reverse order with base complement `3 - c`, matching bwa's `nst_nt4`.
    pub fn bi_interval(&self, pattern: &[u8]) -> (i64, i64, i64) {
        let (lo, hi) = self.exact_interval(pattern);
        let rc: Vec<u8> = pattern.iter().rev().map(|&c| 3 - c).collect();
        let (rlo, _) = self.exact_interval(&rc);
        (lo as i64, rlo as i64, (hi - lo) as i64)
    }

    /// Narrow the SA interval `[lo, hi)` (all rows sharing a common `depth`-length prefix) to the
    /// sub-block whose character at column `depth` equals `c`. Returns the nested `[lo', hi')`. This is
    /// the cheap "append one base" step: appending to the pattern always nests within the current
    /// interval, so the whole forward extension of a match is a sequence of these narrowings, touching
    /// only the (shrinking) interval's rows and the reference — no FM `Occ`. `[lo, hi)` must be a valid
    /// interval whose rows agree on their first `depth` characters.
    pub fn narrow(&self, lo: usize, hi: usize, depth: usize, c: u8) -> (usize, usize) {
        // Rows in [lo, hi) are sorted, and among them the column-`depth` character is nondecreasing
        // (shorter suffixes — None — sort first, then 0,1,2,3). Two partition points bracket `c`.
        let lt = lo
            + self.sa.partition_point_in(lo, hi, |p| {
                let idx = p as usize + depth;
                match self.ref_seq.get(idx) {
                    None => true,          // shorter suffix: sorts before any character
                    Some(&ch) => ch < c,
                }
            });
        let le = lo
            + self.sa.partition_point_in(lo, hi, |p| {
                let idx = p as usize + depth;
                match self.ref_seq.get(idx) {
                    None => true,
                    Some(&ch) => ch <= c,
                }
            });
        (lt, le)
    }

    /// Longest exact match (a prefix of `pattern`) present in the reference, with its SA interval:
    /// returns `(match_len, lo, hi)` where `ref_seq[..]` contains `pattern[..match_len]` at exactly
    /// the rows `[lo, hi)`. This is BWA-MEME's fast LEM: one RMI jump to the pattern's sorted position,
    /// an LCP against the two neighbouring suffixes (the longest common prefix is always at the
    /// insertion point in a sorted suffix array), then one deep-interval search — fast precisely
    /// because a *long* match has a *narrow* interval, unlike the base-by-base narrowing of a byte-
    /// identical SMEM walk which must also touch the wide shallow intervals.
    pub fn lem(&self, pattern: &[u8]) -> (usize, usize, usize) {
        let n = self.sa.len();
        if pattern.is_empty() {
            return (0, 0, n);
        }
        // Suffix-order lower bound of the full pattern (first suffix not < pattern), RMI-seeded. The
        // suffix with the longest common prefix with `pattern` is always at this insertion point or
        // its predecessor (a sorted-suffix-array property), so the LEM is the larger of the two LCPs.
        let pkey = pattern_key(pattern);
        let hint = self.rmi.lower_bound(&self.keys, pkey);
        let lo_full =
            seeded_partition_point(n, hint, |i| self.cmp_key(i, pkey, pattern) == Ordering::Less);
        let ref_len = self.ref_seq.len();
        let lcp = |row: usize| -> usize {
            let start = self.sa.get(row) as usize;
            let mut l = 0usize;
            while l < pattern.len() && start + l < ref_len && self.ref_seq[start + l] == pattern[l] {
                l += 1;
            }
            l
        };
        let mut best = 0usize;
        if lo_full < n {
            best = best.max(lcp(lo_full));
        }
        if lo_full > 0 {
            best = best.max(lcp(lo_full - 1));
        }
        if best == 0 {
            return (0, 0, 0);
        }
        // Interval of the LEM prefix. `lo_full` (the full pattern's lower bound) sits inside this
        // prefix interval, so it is a tight seed for both partition points — no second RMI lookup.
        // When the whole pattern matched, `lo_full` already *is* the lower bound; only the upper
        // bound needs a search.
        let best_pat = &pattern[..best];
        let bkey = pattern_key(best_pat);
        let lo = if best == pattern.len() {
            lo_full
        } else {
            seeded_partition_point(n, lo_full, |i| self.cmp_key(i, bkey, best_pat) == Ordering::Less)
        };
        let hi = seeded_partition_point(n, lo_full, |i| {
            self.cmp_key(i, bkey, best_pat) != Ordering::Greater
        });
        (best, lo, hi)
    }

    /// Longest exact prefix of `pattern` that occurs **at least `min_intv` times**, with its SA
    /// interval. For `min_intv <= 1` this is just [`Self::lem`]. Otherwise, since occurrence count
    /// grows monotonically as the match shortens, we shorten one base at a time from the LEM until the
    /// interval is large enough (a few steps for the small `min_intv` used in reseeding). Used by
    /// round-2 reseeding, which searches for shorter, more-frequent seeds inside a long SMEM.
    pub fn lem_min_intv(&self, pattern: &[u8], min_intv: i64) -> (usize, usize, usize) {
        let (len, lo, hi) = self.lem(pattern);
        if len == 0 || (hi - lo) as i64 >= min_intv {
            return (len, lo, hi);
        }
        // Occurrence count is non-increasing in the prefix length, so the target length (longest L
        // with occ(L) >= min_intv) is a threshold crossing: binary-search it in [1, len].
        let occ = |q: usize| -> i64 {
            let (a, b) = self.exact_interval(&pattern[..q]);
            (b - a) as i64
        };
        let (mut a, mut b) = (1usize, len);
        while a < b {
            let mid = a + (b - a) / 2;
            if occ(mid) < min_intv {
                b = mid;
            } else {
                a = mid + 1;
            }
        }
        let l = a - 1; // longest length still occurring >= min_intv times
        if l == 0 {
            return (0, 0, 0);
        }
        let (lo2, hi2) = self.exact_interval(&pattern[..l]);
        (l, lo2, hi2)
    }

    /// Reference positions where `pattern` occurs exactly (the `sa` values of [`Self::exact_interval`]).
    pub fn occurrences(&self, pattern: &[u8]) -> Vec<i64> {
        let (lo, hi) = self.exact_interval(pattern);
        let mut v: Vec<i64> = self.sa.range_i64(lo, hi);
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

    /// Reverse-complement of a binary pattern (codes 0..=3): reverse order, complement `3 - c`
    /// (A=0<->T=3, C=1<->G=2), matching bwa's `nst_nt4` complement.
    fn revcomp(p: &[u8]) -> Vec<u8> {
        p.iter().rev().map(|&c| 3 - c).collect()
    }

    /// Cross-check the SA-based bidirectional interval hypothesis against the real `FmIndex`:
    /// walking `backward_ext` over pattern `P` yields `(k, l, s)`; the claim is
    ///   k == exact_interval(P).lo,  s == hi - lo,  l == exact_interval(revcomp(P)).lo
    /// over the same `[fwd][rc]` reference. If this holds, `LearnedSa` can reproduce bwa-mem2's
    /// FMD bi-interval byte-for-byte without the FM Occ function.
    #[test]
    fn bidirectional_interval_matches_fmindex() {
        use crate::fmindex::FmIndex;
        use std::path::Path;
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let two_l = reference.len();

        let mut seed = 0xdead_beef_cafe_babeu64;
        let mut checked = 0usize;
        for _ in 0..4000 {
            let mlen = 1 + (lcg(&mut seed) as usize % 60);
            if mlen >= two_l {
                continue;
            }
            let start = lcg(&mut seed) as usize % (two_l - mlen);
            let pat = &reference[start..start + mlen];

            // Ground truth from the FM-index bi-interval walk.
            let mut sm = fm.full_interval();
            for &c in pat.iter().rev() {
                sm = fm.backward_ext(sm, c as usize);
            }

            let (lo, hi) = lsa.exact_interval(pat);
            assert_eq!(sm.k, lo as i64, "k mismatch, pattern@{start} len {mlen}");
            assert_eq!(sm.s, (hi - lo) as i64, "s mismatch, pattern@{start} len {mlen}");

            let rc = revcomp(pat);
            let (rlo, rhi) = lsa.exact_interval(&rc);
            assert_eq!(
                sm.l, rlo as i64,
                "l mismatch, pattern@{start} len {mlen}: fm.l={} exact_lo(revcomp)={rlo}",
                sm.l
            );
            // Sanity: the revcomp interval has the same size as P's (strand symmetry).
            assert_eq!(rhi - rlo, hi - lo, "revcomp size mismatch, pattern@{start}");
            checked += 1;
        }
        assert!(checked > 3000, "too few patterns checked ({checked})");
    }

    /// Iterating `narrow` one base at a time from the full interval must reproduce `exact_interval`
    /// (and its size must track occurrence counts), for every prefix length of the pattern.
    #[test]
    fn narrow_reproduces_exact_interval() {
        let mut seed = 0x0f1e_2d3c_4b5a_6978u64;
        for trial in 0..8 {
            let len = 300 + (lcg(&mut seed) as usize % 1500);
            let alpha = if trial % 2 == 0 { 4 } else { 2 };
            let ref_seq: Vec<u8> = (0..len).map(|_| (lcg(&mut seed) % alpha) as u8).collect();
            let lsa = LearnedSa::build(ref_seq.clone(), 256);
            let n = lsa.len();
            for _ in 0..200 {
                let mlen = 1 + (lcg(&mut seed) as usize % 45);
                let s = if len > mlen {
                    lcg(&mut seed) as usize % (len - mlen)
                } else {
                    0
                };
                let pat = &ref_seq[s..(s + mlen).min(len)];
                // Walk narrow from the full interval, one base at a time.
                let (mut lo, mut hi) = (0usize, n);
                for (depth, &c) in pat.iter().enumerate() {
                    let (nlo, nhi) = lsa.narrow(lo, hi, depth, c);
                    lo = nlo;
                    hi = nhi;
                    // At every prefix length, the narrowed interval must equal exact_interval.
                    let (elo, ehi) = lsa.exact_interval(&pat[..=depth]);
                    assert_eq!((lo, hi), (elo, ehi), "prefix depth {depth} of pat@{s}");
                }
            }
        }
    }

    /// `lem` must return the longest prefix of the pattern present in the reference, with the correct
    /// interval, matching brute force.
    #[test]
    fn lem_matches_bruteforce() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for trial in 0..8 {
            let len = 300 + (lcg(&mut seed) as usize % 1500);
            let alpha = if trial % 2 == 0 { 4 } else { 2 };
            let ref_seq: Vec<u8> = (0..len).map(|_| (lcg(&mut seed) % alpha) as u8).collect();
            let lsa = LearnedSa::build(ref_seq.clone(), 256);
            for _ in 0..200 {
                let mlen = 1 + (lcg(&mut seed) as usize % 50);
                let pat: Vec<u8> = if lcg(&mut seed) % 2 == 0 && len > mlen {
                    let s = lcg(&mut seed) as usize % (len - mlen);
                    ref_seq[s..s + mlen].to_vec()
                } else {
                    (0..mlen).map(|_| (lcg(&mut seed) % alpha) as u8).collect()
                };
                // Brute: longest L with pat[..L] present.
                let brute_len = (0..=pat.len())
                    .rev()
                    .find(|&l| l == 0 || !brute(&ref_seq, &pat[..l]).is_empty())
                    .unwrap();
                let (ml, lo, hi) = lsa.lem(&pat);
                assert_eq!(ml, brute_len, "lem len, pat={pat:?}");
                if ml > 0 {
                    assert_eq!((lo, hi), lsa.exact_interval(&pat[..ml]), "lem interval");
                }
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
