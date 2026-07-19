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
//!
//! # Provenance
//!
//! There is **no bwa-mem2 C original for this file**. bwa-mem2 has no learned index; the C in
//! `reference/bwa-mem2/src/FMI_search.cpp` implements the FM-index (`backwardExt`, `GET_OCC`,
//! `get_sa_entry_compressed`) that this module is an alternative to. The only place C provenance is
//! claimed below is where a value or convention has to agree with the FM path bit-for-bit: the
//! `[forward][revcomp]` reference layout, the `3 - c` complement (bwa's `nst_nt4` alphabet is
//! A=0,C=1,G=2,T=3, so complementing is `3 - c`), and the meaning of the interval triple `(k, l, s)`
//! (`struct SMEM` in `FMI_search.h`). Everything else here is our own construction.
//!
//! # Vocabulary a first-time reader needs
//!
//! * **Reference space.** The `.0123` reference is the forward genome `F` (length `L`, the value
//!   bwa calls `l_pac`) concatenated with its reverse complement, so `ref_seq` has `2L` bytes, one
//!   base per byte in `0..=3`. Position `p < L` is forward strand; `p >= L` is the reverse-complement
//!   copy, and maps back to forward coordinate `2L - 1 - p`. This doubling is what makes a *single*
//!   forward-only search structure answer both strands.
//! * **Ambiguous bases.** `N` and friends never appear in `ref_seq`: `bntseq` replaced each with a
//!   pseudo-random base in `0..=3` and recorded the run in the `.amb` holes list. So every byte here
//!   really is in `0..=3` and no alphabet escape is needed. (Alignments that straddle a hole are
//!   filtered downstream, not here.)
//! * **Contig boundaries.** `ref_seq` has no separators between contigs; the `.ann` table maps a flat
//!   position to (contig, offset). This module is entirely flat-coordinate and never sees contigs.
//! * **Sentinel.** The suffix array has `2L + 1` rows, not `2L`: row 0 is the empty suffix (the
//!   implicit `$` that sorts before every real base), i.e. `sa[0] == ref_seq.len()`. bwa-mem2's
//!   `ref_seq_len` is likewise `2L + 1`. Keeping the sentinel row means `sa_at(row)` here equals
//!   `FmIndex::get_sa(row)` for the same `row`, which is the whole point: intervals are
//!   interchangeable between the two paths.
//! * **SA interval.** A half-open row range `[lo, hi)` of the suffix array. Because suffixes are
//!   sorted, all suffixes sharing a given prefix are contiguous, so "every occurrence of `pattern`"
//!   is exactly one such range, and `hi - lo` is the occurrence count. bwa-mem2 stores the same
//!   object as `(k, s)` = (start, size), plus `l` for the opposite strand.
//!
//! # Memory layout
//!
//! Nothing in this module is serialized to disk; `LearnedSa` is built in RAM (from an existing
//! FM-index, see [`LearnedSa::from_sa`]) and dropped with the process. The only layout that matters
//! is in-memory: `sa` and `keys` are [`Packed40`], 5 bytes per element, little-endian low-5-bytes of
//! a `u64`, no padding and no alignment requirement (element `i` occupies bytes `5i .. 5i+5`). At
//! human-genome scale (`2L + 1` ~ 6.2e9 rows) that is ~31 GB per array instead of ~49.6 GB as `u64`.

use crate::packed::Packed40;
use crate::rmi::Rmi;
use crate::sais::suffix_array_with_sentinel;
use std::cmp::Ordering;

/// Bases packed into one learned key. 20 (40 bits, one `Packed40` element) is selective enough at
/// genome scale — 4^20 ≫ 6.2G positions, so almost no two suffixes share a key, keeping the co-located
/// comparison a single memory access — while halving the key RAM vs a 32-base `u64` (49.6 GB → 31 GB).
/// K only affects *speed* (key selectivity + RMI hint precision); the search result is always corrected
/// against the reference, so any K stays byte-identical.
///
/// Hard upper bound: `2 * K` must be `<= 40` bits, because a key is stored in one [`Packed40`] slot.
/// K = 20 is exactly at that ceiling; raising it requires widening `Packed40` first. Valid range in
/// practice is `1 ..= 20`. Not a tuning knob exposed to the user: it is a compile-time constant here.
///
/// Bit layout of a key, spelled out: base `r` (0-based, `r` in `0..K`) occupies bit pair
/// `2*(K-1-r) .. 2*(K-1-r)+2`. At K = 20 the FIRST base sits in bits 38-39 (the top pair of the
/// 40 used) and the LAST in bits 0-1; bits 40-63 of the `u64` are always zero, which is what lets
/// the key live in one 5-byte `Packed40` slot. Changing K changes the shift widths in
/// [`kmer_key`], [`pattern_key`] and [`LearnedSa::cmp_key`] together, and all three must agree;
/// exceeding 20 silently overflows the packed slot and corrupts every key.
pub const K: usize = 20;

/// A suffix array over a binary reference (`0..=3` codes) plus a learned index over the first-`K`
/// bases of each suffix.
///
/// Invariants that every method below assumes and that `from_sa` is responsible for establishing:
///   * `sa.len() == keys.len() == ref_seq.len() + 1` (the `+1` is the sentinel row),
///   * `sa` is the true suffix array of `ref_seq` with the sentinel first, so `sa[0] == ref_seq.len()`,
///   * `keys` is sorted ascending (it is `kmer_key` of the suffixes in suffix order, and truncating
///     sorted strings to a fixed length preserves order), which is what the [`Rmi`] requires,
///   * every `ref_seq` byte is `< 4` (`kmer_key` masks with `& 3`, so a stray high bit would silently
///     alias rather than panic; violating this makes keys disagree with `ref_seq` and the last-mile
///     comparison would then be doing the wrong search).
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
    /// Learned index over `keys`: maps a 40-bit key to an approximate row, used only as a search
    /// *hint*. Wrong hints cost time, never correctness (see [`seeded_partition_point`]).
    rmi: Rmi,
}

/// First `K` bases at `pos` in `ref_seq`, 2-bit packed MSB-first, zero-padded past the end.
///
/// MSB-first is the load-bearing choice: it makes numeric order on the `u64` key agree with
/// lexicographic order on the base string, which is what lets `keys` be a sorted array the RMI can
/// index and what lets `cmp_key` decide a comparison without touching `ref_seq`.
///
/// Worked micro-example with K = 4 (the real K is 20): bases `A C G T` = codes `0 1 2 3` pack as
/// `((((0<<2)|1)<<2|2)<<2)|3` = `0b00_01_10_11` = 0x1B. Base at offset 0 lands in the top bit pair,
/// base at offset `K-1` in the bottom pair. A suffix only 2 bases long, `G T`, packs as
/// `0b10_11_00_00` = 0xB0: the two missing bases pad as `A` (code 0), the smallest base, which is why
/// padding can only make a short suffix sort too *early*, never too late (see [`LearnedSa::cmp_key`]).
///
/// `pos` is a row's suffix-array value, i.e. a position in `[0, ref_seq.len()]`; `pos == len` (the
/// sentinel row) yields the all-zero key.
///
/// # Parameters
///
/// * `ref_seq`: the binary reference, one base per byte, every byte in `0..=3`. A byte with high
///   bits set is masked (`& 3`) rather than rejected, so it aliases silently.
/// * `pos`: a REFERENCE POSITION (not a suffix-array row) in the flat 2L space, valid range
///   `[0, ref_seq.len()]`. Supplied by [`LearnedSa::from_sa`] as `sa[row]`. Must be non-negative;
///   a negative value wraps through the `as usize` cast into a huge index that reads as all-`A`.
///
/// # Returns
///
/// The `2*K`-bit key, right-aligned in a `u64` (bits `2*K .. 64` are zero).
#[inline]
fn kmer_key(ref_seq: &[u8], pos: i64) -> u64 {
    let len = ref_seq.len();
    let p = pos as usize;
    // Accumulator: after `r` iterations it holds the first `r` bases, each shifted up by two bits
    // per subsequent base, so the first base always ends up in the most significant occupied pair.
    let mut key = 0u64;
    for r in 0..K {
        // Past the reference end we shift in `A` (0) rather than stopping, so every key is exactly
        // `2*K` bits wide and keys are directly comparable as integers.
        // `idx` is an absolute reference position: the `r`-th base of the suffix starting at `p`.
        let idx = p + r;
        let c = if idx < len {
            (ref_seq[idx] & 3) as u64
        } else {
            0
        };
        key = (key << 2) | c;
    }
    key
}

/// First `K` bases of a query pattern (codes `0..=3`), packed the same way, zero-padded.
///
/// Must stay bit-for-bit consistent with [`kmer_key`]: the two are compared against each other in
/// [`LearnedSa::cmp_key`], and any divergence in padding or bit order would silently corrupt the
/// search window. A pattern shorter than `K` is zero-padded, exactly as a short suffix is.
///
/// # Parameters
///
/// * `pattern`: the query bases, one per byte in `0..=3` (bwa's `nst_nt4` codes), of any length.
///   Only the first `K` bases are consulted; a longer pattern is resolved later against the
///   reference by [`LearnedSa::cmp_pattern`]. Supplied by the seeding code from the read.
///
/// # Returns
///
/// The `2*K`-bit key, right-aligned in a `u64`, directly comparable with a [`kmer_key`] result.
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
///
/// This is the "galloping" / exponential-search idiom. It is the safety net that makes the learned
/// index result-preserving: a perfect hint costs `O(1)` probes, a garbage hint degrades gracefully to
/// `O(log n)` (the same cost as the plain binary search it replaces), and in neither case can the
/// returned boundary differ. That is why a mispredicting RMI can never change alignment output.
///
/// # Parameters
///
/// * `n`: the number of SUFFIX-ARRAY ROWS searched over, i.e. `sa.len()`; the search domain is
///   `0..n` and rows are not reference positions.
/// * `hint`: a guessed row index, normally the [`Rmi`] prediction. Any `usize` is accepted and is
///   clamped into `[0, n-1]`; correctness does not depend on it, only the probe count does.
/// * `pred`: must be MONOTONE over `0..n`: true for a (possibly empty) prefix of rows and false
///   for the rest. A non-monotone predicate returns an arbitrary row, not a panic.
///
/// # Returns
///
/// The row index of the first `false`, in `[0, n]`; `n` means `pred` was true for every row.
fn seeded_partition_point<P: Fn(usize) -> bool>(n: usize, hint: usize, pred: P) -> usize {
    if n == 0 {
        return 0;
    }
    // `h` is the hint clamped to a real row. `a`/`b` become the bracket `[a, b]` that provably
    // contains the boundary: `pred` is true at `a` (or `a == 0`) and false at `b` (or `b == n`).
    let h = hint.min(n - 1);
    let (mut a, mut b);
    if pred(h) {
        // Boundary is in (h, n]. Grow right until a false (or the end).
        a = h;
        // Stride in rows, doubled each iteration. Invariant at the top of the loop: `pred` is true
        // at every row up to and including `a`, and the boundary lies in `(a, n]`.
        let mut step = 1usize;
        loop {
            // Doubling stride from the hint: probe h+1, h+2, h+4, ... `saturating_add` so a huge
            // `step` cannot wrap around into a bogus low probe index.
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
        // Mirror of the right-hand case. Invariant at the top of the loop: `pred` is false at `b`
        // and at every row above it within the visited range, so the boundary lies in `[0, b]`.
        let mut step = 1usize;
        loop {
            // Mirror image going left. `h < step` is the underflow guard: it means the doubling
            // stride has already passed index 0, so 0 is the bracket's left end.
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
    // Loop invariant: pred is true everywhere below `a` and false at `b` (or `b == n`), so `a` is the
    // answer on exit. `a + (b - a) / 2` rather than `(a + b) / 2` to avoid overflow on huge indices.
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
    ///
    /// Runs SA-IS from scratch, so this is the *test/small-input* entry point. At genome scale use
    /// [`Self::from_sa`] with the SA read out of an already-built FM-index: identical result, but it
    /// avoids re-running suffix sorting over 6.2e9 symbols.
    ///
    /// # Parameters
    ///
    /// * `ref_seq`: the binary reference, taken by value because `LearnedSa` stores it. Every byte
    ///   must be in `0..=3`; in the real pipeline this is `[forward][revcomp]`, length `2L`.
    /// * `n_leaves`: RMI leaf count, forwarded verbatim to [`Rmi::build`]. Affects speed only.
    pub fn build(ref_seq: Vec<u8>, n_leaves: usize) -> Self {
        // `sa[row]` is a reference POSITION; the vector is indexed by ROW, with the sentinel at 0.
        let sa = suffix_array_with_sentinel(&ref_seq);
        Self::from_sa(ref_seq, sa, n_leaves)
    }

    /// Build from a **prebuilt** suffix array (skipping SAIS): `sa` must be
    /// `suffix_array_with_sentinel(&ref_seq)` — e.g. extracted from an existing FM-index via
    /// `FmIndex::get_sa` for every row, which is byte-identical to running SAIS but far cheaper on a
    /// genome-scale reference and guarantees `sa[i] == fm.get_sa(i)`. Computes the first-`K` keys and
    /// trains the RMI over them.
    ///
    /// # Parameters
    ///
    /// * `ref_seq`: as in [`Self::build`], taken by value and stored.
    /// * `sa`: the suffix array INCLUDING the sentinel row, so `sa.len() == ref_seq.len() + 1` and
    ///   `sa[0] == ref_seq.len()`. Indexed by row; each element is a reference position in
    ///   `[0, ref_seq.len()]`. Consumed and dropped early to cap peak memory. Only
    ///   `debug_assert`ed: a wrong SA silently produces wrong intervals.
    /// * `n_leaves`: RMI leaf count, forwarded to [`Rmi::build`].
    pub fn from_sa(ref_seq: Vec<u8>, sa: Vec<i64>, n_leaves: usize) -> Self {
        debug_assert_eq!(
            sa.len(),
            ref_seq.len() + 1,
            "sa must include the sentinel row"
        );
        // Pack the SA to 5 bytes and drop the i64 array; then compute the keys **directly** into a
        // 5-byte packed array (no intermediate `Vec<u64>`), so the peak is `packed_sa + packed_keys`
        // (~62 GB at genome scale) rather than three full copies.
        // Same content as `sa`, 5 bytes per row instead of 8. Still row-indexed, still positions.
        let packed = Packed40::from_slice(&sa);
        // `sa` is 8 bytes/row and `packed` is 5; dropping it here (rather than at end of scope) is
        // the difference between a ~50 GB and a ~31 GB peak on a human genome.
        drop(sa);
        // Row `i`'s key is read through the SA: `keys[i] = first K bases at sa[i]`. Because rows are
        // in suffix order, the resulting array is sorted ascending, which `Rmi::build` requires.
        // Note the access pattern is a random gather into `ref_seq`, hence `from_fn`'s parallelism.
        let keys = Packed40::from_fn(packed.len(), |i| kmer_key(&ref_seq, packed.get(i) as i64));
        // Trained on `keys`, so its predictions are ROW indices, never reference positions.
        let rmi = Rmi::build(&keys, n_leaves);
        LearnedSa {
            ref_seq,
            sa: packed,
            keys,
            rmi,
        }
    }

    /// Number of suffix-array rows (including the sentinel row), i.e. `2L + 1`, the same quantity
    /// bwa-mem2 calls `reference_seq_len`.
    pub fn len(&self) -> usize {
        self.sa.len()
    }

    /// The suffix-array value at `row` (a reference position in the `[fwd][rc]` space). Equals
    /// `FmIndex::get_sa(row)` over the same reference, so seeds materialized as `sa_at(k + i)` are
    /// byte-identical to the FM path.
    ///
    /// # Parameters
    ///
    /// * `row`: a suffix-array ROW index, valid range `[0, len())`. Panics (via `Packed40::get`)
    ///   out of range. Typically `k + i` for the `i`-th occurrence of an interval starting at `k`.
    ///
    /// # Returns
    ///
    /// The reference POSITION in flat 2L space, `[0, 2L]`, where that suffix begins. `2L` is only
    /// returned for the sentinel row 0.
    #[inline]
    pub fn sa_at(&self, row: usize) -> i64 {
        self.sa.get(row) as i64
    }

    /// True when the indexed reference has no bases at all. Note this is about `ref_seq`, not about
    /// the row count: even an empty reference has one row (the sentinel), so `len()` is 1 here.
    pub fn is_empty(&self) -> bool {
        self.ref_seq.is_empty()
    }

    /// Compare `pattern` against the reference suffix at SA row `i`, as a prefix comparison: equal
    /// means `pattern` is a prefix of that suffix. A suffix that ends before `pattern` does (running
    /// off the reference end) compares `Less` (shorter string sorts first), matching the sentinel.
    ///
    /// This is the ground truth the whole search is corrected against: it reads `ref_seq` directly,
    /// so it is immune to any key padding artefact and has no length ceiling at `K`. Cost is two
    /// dependent random accesses (`sa[i]`, then `ref_seq[start..]`), which is exactly what
    /// [`Self::cmp_key`] exists to avoid on the common path.
    ///
    /// Note the asymmetry: it is a *prefix* comparison, so `Equal` does not mean the strings are
    /// equal, it means the pattern is a prefix of the suffix. That is what makes a single monotone
    /// predicate able to bracket the whole occurrence interval.
    ///
    /// # Parameters
    ///
    /// * `i`: a suffix-array ROW index, `[0, len())`.
    /// * `pattern`: the query bases, codes `0..=3`, any length (no `K` ceiling here).
    ///
    /// # Returns
    ///
    /// `Less` if the suffix at row `i` sorts before `pattern`, `Greater` if after, `Equal` if
    /// `pattern` is a PREFIX of it (not necessarily equal to it).
    #[inline]
    fn cmp_pattern(&self, i: usize, pattern: &[u8]) -> Ordering {
        // Reference POSITION of row `i`'s suffix; the second of the two dependent random accesses.
        let start = self.sa.get(i) as usize;
        let len = self.ref_seq.len();
        for (j, &pc) in pattern.iter().enumerate() {
            // Absolute reference position of the pattern's `j`-th base under this alignment.
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
    ///
    /// # Parameters
    ///
    /// * `i`: a suffix-array ROW index, `[0, len())`.
    /// * `pkey`: `pattern_key(pattern)`, hoisted out of the search loop by the caller. It MUST
    ///   correspond to `pattern`; a stale `pkey` makes the fast path decide the wrong way and no
    ///   check catches it.
    /// * `pattern`: the same query bases the `pkey` was built from, codes `0..=3`.
    ///
    /// # Returns
    ///
    /// The same `Ordering` [`Self::cmp_pattern`] would return for `(i, pattern)`, just usually
    /// without touching the reference.
    #[inline]
    fn cmp_key(&self, i: usize, pkey: u64, pattern: &[u8]) -> Ordering {
        // Compare only the `l = min(len, K)` bases the pattern actually constrains. Shifting both
        // keys right by `2*(K - l)` bits drops the low `K - l` base slots from both sides at once,
        // which is cheaper and branch-free versus masking. Worked example at K = 4, pattern "GT"
        // (l = 2, shift = 4): key 0b10_11_00_00 >> 4 = 0b1011, pkey 0b10_11_00_00 >> 4 = 0b1011,
        // so the two padded tails cancel and the comparison is over "GT" alone.
        // `l` is how many of the key's `K` base slots the pattern actually constrains, `1..=K`.
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
    ///
    /// # Parameters
    ///
    /// * `pattern`: the query bases, codes `0..=3`, any length including 0. Supplied by the seeding
    ///   code as a slice of the read (already `nst_nt4`-encoded). A byte `>= 4` (an `N` that was
    ///   not re-encoded) is masked to `& 3` inside the key path but compared raw in
    ///   [`Self::cmp_pattern`], so the two would disagree: callers must not pass one.
    ///
    /// # Returns
    ///
    /// `(lo, hi)`, a half-open range of suffix-array ROWS (not positions). `hi - lo` is the
    /// occurrence count and is 0 when the pattern is absent.
    pub fn exact_interval(&self, pattern: &[u8]) -> (usize, usize) {
        // Total row count, `2L + 1`.
        let n = self.sa.len();
        if pattern.is_empty() {
            return (0, n);
        }
        // RMI predicts where this key sorts among the stored first-K keys; a good seed for both ends.
        // One key computation and one RMI lookup are shared by both partition points: when the
        // pattern is at least K long the interval is usually tiny, so `hint` sits within a handful of
        // rows of both `lo` and `hi` and each galloping search terminates in a couple of probes.
        // The query's first-K bases packed exactly as the stored keys are.
        let pkey = pattern_key(pattern);
        // A ROW index: where `pkey` would sort among the stored keys. Approximate in the sense that
        // it is the boundary for the K-truncated key, not for the full pattern; used only as a seed.
        let hint = self.rmi.lower_bound(&self.keys, pkey);
        // The two predicates are the two sides of the same prefix comparison, and both are monotone
        // over the sorted rows (Less..., then Equal..., then Greater...), which is what
        // `seeded_partition_point` requires:
        // lower bound: first row whose suffix is NOT < pattern.
        let lo = seeded_partition_point(n, hint, |i| {
            self.cmp_key(i, pkey, pattern) == Ordering::Less
        });
        // upper bound: first row whose suffix is > pattern (prefix comparison). Note `!= Greater`
        // keeps both Less and Equal on the true side, so `hi` lands just past the Equal block.
        let hi = seeded_partition_point(n, hint, |i| {
            self.cmp_key(i, pkey, pattern) != Ordering::Greater
        });
        (lo, hi)
    }

    /// The bwa-mem2 bidirectional FMD interval `(k, l, s)` for `pattern`, reproduced from the plain
    /// `[fwd][rc]` suffix array (validated byte-identical to `FmIndex::backward_ext` walked over the
    /// same pattern, `bidirectional_interval_matches_fmindex`):
    ///   * `k` = SA lower-bound of `pattern` (forward interval start),
    ///   * `s` = interval size (occurrence count),
    ///   * `l` = SA lower-bound of `revcomp(pattern)` (reverse-complement interval start).
    /// `revcomp` here is reverse order with base complement `3 - c`, matching bwa's `nst_nt4`.
    ///
    /// Why `l` can be recovered this way at all: the reference is `[forward][revcomp]`, so every
    /// occurrence of `P` on one strand has a mirror occurrence of `revcomp(P)` in the other half, and
    /// the two intervals therefore have equal size (asserted in the tests). bwa-mem2 maintains `l`
    /// incrementally inside `backwardExt` because its FM-index cannot re-search cheaply; here a second
    /// SA lookup is affordable, which is why this is a plain second `exact_interval` call rather than
    /// a port of the C's interval bookkeeping.
    ///
    /// Cost note: this allocates and searches a second pattern, so it is roughly twice the work of
    /// [`Self::exact_interval`]. The SMEM driver in `bwa-seed` avoids it on the hot path by extending
    /// with [`Self::narrow`] and only materializing `(k, l, s)` where the FM semantics demand it.
    ///
    /// # Parameters
    ///
    /// * `pattern`: the query bases, codes `0..=3`. Because the complement is computed as `3 - c`,
    ///   a byte `> 3` would underflow the `u8` subtraction and panic in debug.
    ///
    /// # Returns
    ///
    /// `(k, l, s)` as `i64` to match bwa-mem2's `SMEM` field types: `k` and `l` are suffix-array
    /// ROW indices (forward and reverse-complement interval starts), `s` is an occurrence COUNT.
    pub fn bi_interval(&self, pattern: &[u8]) -> (i64, i64, i64) {
        // Forward interval, in rows.
        let (lo, hi) = self.exact_interval(pattern);
        // The reverse complement of the query: reversed order, each base mapped A<->T, C<->G.
        let rc: Vec<u8> = pattern.iter().rev().map(|&c| 3 - c).collect();
        // Only the START row of the revcomp interval is kept; its size equals the forward one.
        let (rlo, _) = self.exact_interval(&rc);
        (lo as i64, rlo as i64, (hi - lo) as i64)
    }

    /// Narrow the SA interval `[lo, hi)` (all rows sharing a common `depth`-length prefix) to the
    /// sub-block whose character at column `depth` equals `c`. Returns the nested `[lo', hi')`. This is
    /// the cheap "append one base" step: appending to the pattern always nests within the current
    /// interval, so the whole forward extension of a match is a sequence of these narrowings, touching
    /// only the (shrinking) interval's rows and the reference — no FM `Occ`. `[lo, hi)` must be a valid
    /// interval whose rows agree on their first `depth` characters.
    ///
    /// # Parameters
    ///
    /// * `lo`, `hi`: the current interval as suffix-array ROW indices, half-open, `0 <= lo <= hi
    ///   <= len()`. Supplied by the caller as either the full interval `(0, len())` or the result
    ///   of the previous `narrow`.
    /// * `depth`: how many pattern characters are already matched, unitless, so column `depth` is
    ///   the next one. It is also the offset added to each row's reference POSITION to find the
    ///   character being compared, which is why it must equal the rows' common prefix length.
    /// * `c`: the base code being appended, `0..=3`.
    ///
    /// # Returns
    ///
    /// The nested interval `(lo', hi')` in rows, with `lo <= lo' <= hi' <= hi`; empty when no row
    /// continues with `c`.
    ///
    /// Violating the "rows agree on their first `depth` characters" precondition breaks the
    /// monotonicity the two partition points rely on and silently returns a wrong sub-range, so this
    /// is only safe when driven from the full interval one base at a time (which the test
    /// `narrow_reproduces_exact_interval` checks).
    ///
    /// Reads only `sa` (via `partition_point_in`) and `ref_seq`, never `keys` or the RMI: at depth
    /// beyond `K` a key would carry no information anyway.
    pub fn narrow(&self, lo: usize, hi: usize, depth: usize, c: u8) -> (usize, usize) {
        // Rows in [lo, hi) are sorted, and among them the column-`depth` character is nondecreasing
        // (shorter suffixes — None — sort first, then 0,1,2,3). Two partition points bracket `c`.
        // `partition_point_in` counts matching rows *within* the sub-range, so both results are
        // offsets relative to `lo` and must have `lo` added back to become absolute row indices.
        // `lt` = rows with column-`depth` char strictly less than `c` (the new interval's start).
        let lt = lo
            + self.sa.partition_point_in(lo, hi, |p| {
                // `p` is the row's reference POSITION, so `idx` is the absolute position of the
                // character in column `depth` of that suffix.
                let idx = p as usize + depth;
                match self.ref_seq.get(idx) {
                    None => true, // shorter suffix: sorts before any character
                    Some(&ch) => ch < c,
                }
            });
        // `le` = rows with column-`depth` char <= `c` (the new interval's end). The difference of the
        // two counts is the number of rows whose next character is exactly `c`.
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
    ///
    /// Returns `(0, 0, 0)` when not even the first base occurs (note this differs from the empty-
    /// pattern case, which returns the whole array `(0, 0, n)`); callers must not treat the two the
    /// same. LEM is not itself an SMEM: it is the forward-extension primitive the SMEM driver in
    /// `bwa-seed` calls, and the byte-identity argument lives there, not here.
    ///
    /// # Parameters
    ///
    /// * `pattern`: the query bases, codes `0..=3`, any length. Supplied by the SMEM driver as the
    ///   read suffix starting at the current position.
    ///
    /// # Returns
    ///
    /// `(match_len, lo, hi)`: `match_len` is a length in BASES, `0..=pattern.len()`; `lo`/`hi` are
    /// half-open suffix-array ROW indices for `pattern[..match_len]`.
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
        // ROW where the full pattern would be inserted. Not necessarily an occurrence: when the
        // pattern is absent this is simply where it would go, which is exactly what the LCP trick
        // below needs.
        let lo_full = seeded_partition_point(n, hint, |i| {
            self.cmp_key(i, pkey, pattern) == Ordering::Less
        });
        let ref_len = self.ref_seq.len();
        // Longest common prefix of `pattern` and the suffix at `row`, bounded by both the pattern
        // length and the reference end. Plain byte loop: it runs at most twice per `lem` call, so
        // there is nothing to gain from vectorizing it.
        // Takes a ROW, returns a length in bases.
        let lcp = |row: usize| -> usize {
            // Reference POSITION of that row's suffix.
            let start = self.sa.get(row) as usize;
            // Bases matched so far; on exit it is the LCP length.
            let mut l = 0usize;
            while l < pattern.len() && start + l < ref_len && self.ref_seq[start + l] == pattern[l]
            {
                l += 1;
            }
            l
        };
        // Only two candidates need checking: the insertion point and its predecessor. Any other row
        // is separated from `pattern` in sorted order by one of these two, and LCP with a sorted
        // neighbour can only shrink as you move away, so no third row can do better.
        // Best match length in BASES over the two candidate rows.
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
        // The matching prefix itself, and its packed key (differs from `pkey` whenever the whole
        // pattern did not match, because the truncation changes the low base slots).
        let best_pat = &pattern[..best];
        let bkey = pattern_key(best_pat);
        let lo = if best == pattern.len() {
            lo_full
        } else {
            seeded_partition_point(n, lo_full, |i| {
                self.cmp_key(i, bkey, best_pat) == Ordering::Less
            })
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
    ///
    /// `min_intv` is an occurrence count (unitless), a floor rather than a target: the result is the
    /// *longest* prefix still meeting it, and `min_intv <= 1` disables the constraint. Round-1
    /// seeding passes 1; round-2 reseeding passes the parent SMEM's own occurrence count plus one
    /// (`p.s + 1` in `bwa-seed`'s `smem_round_2_lsa`), which is what forces the re-seed to find a
    /// strictly *more frequent*, hence strictly shorter, match than the SMEM it came from. This
    /// mirrors `smem_round_2` in `reference/bwa-mem2/src/bwamem.cpp`. (`opt->split_width`, bwa's
    /// default 10, is a different knob: it gates *which* SMEMs get reseeded, not this floor.)
    ///
    /// # Parameters
    ///
    /// * `pattern`: the query bases, codes `0..=3`, as for [`Self::lem`].
    /// * `min_intv`: minimum occurrence count, unitless, `>= 0`. Supplied by the seeding round:
    ///   1 (no constraint) in round 1, `parent_smem.s + 1` in round-2 reseeding.
    ///
    /// # Returns
    ///
    /// `(match_len, lo, hi)` with the same meaning as [`Self::lem`], but `match_len` is the longest
    /// prefix whose interval size is at least `min_intv`. `(0, 0, 0)` when no non-empty prefix
    /// qualifies.
    pub fn lem_min_intv(&self, pattern: &[u8], min_intv: i64) -> (usize, usize, usize) {
        // The unconstrained longest exact match: the upper end of the search range below.
        let (len, lo, hi) = self.lem(pattern);
        if len == 0 || (hi - lo) as i64 >= min_intv {
            return (len, lo, hi);
        }
        // Occurrence count is non-increasing in the prefix length, so the target length (longest L
        // with occ(L) >= min_intv) is a threshold crossing: binary-search it in [1, len].
        // (The doc comment above describes shortening one base at a time; the code does the
        // equivalent binary search, same answer in O(log len) interval lookups instead of O(len).)
        // Occurrence count of the length-`q` prefix, i.e. its interval width in rows.
        let occ = |q: usize| -> i64 {
            let (a, b) = self.exact_interval(&pattern[..q]);
            (b - a) as i64
        };
        // Searching for the first length whose occurrence count drops below `min_intv`; `a` ends as
        // that first-failing length, so `a - 1` is the longest still-passing one. `a` starts at 1
        // because length 0 trivially passes and would make `a - 1` underflow-adjacent.
        // Bracket over match LENGTHS in bases, not over rows. Invariant at the top of the loop:
        // every length below `a` occurs at least `min_intv` times, and length `b` either fails or
        // is the (already known to fail) upper end `len`.
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
        // Re-derive the ROW interval for the chosen length; the binary search only found `l`.
        let (lo2, hi2) = self.exact_interval(&pattern[..l]);
        (l, lo2, hi2)
    }

    /// Reference positions where `pattern` occurs exactly (the `sa` values of [`Self::exact_interval`]).
    ///
    /// Positions are in the flat `[fwd][rc]` space, so a value `>= l_pac` denotes the reverse strand
    /// and must be mapped back by the caller. The sort is for test determinism (SA order is by
    /// suffix, not by position); the seeding path deliberately does *not* call this, it consumes the
    /// interval directly, because materializing every occurrence is what blows up on repeats.
    ///
    /// # Parameters
    ///
    /// * `pattern`: the query bases, codes `0..=3`. An empty pattern yields every position in the
    ///   reference (the whole array), which is almost never what a caller wants.
    ///
    /// # Returns
    ///
    /// Reference POSITIONS (not rows) in flat 2L space, ascending. Length is the occurrence count,
    /// which is unbounded on a repeat: this allocates one `i64` per occurrence.
    pub fn occurrences(&self, pattern: &[u8]) -> Vec<i64> {
        // Rows first, positions second: `range_i64` is the row-to-position translation.
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
    ///
    /// # Parameters
    ///
    /// * `ref_seq`: the same binary reference the `LearnedSa` was built over.
    /// * `pattern`: the query bases. An empty pattern returns no positions here, deliberately
    ///   unlike [`LearnedSa::exact_interval`], because the tests never probe it.
    ///
    /// # Returns
    ///
    /// Reference POSITIONS in ascending order, directly comparable with
    /// [`LearnedSa::occurrences`].
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

    /// Deterministic test PRNG: a 64-bit linear congruential generator (the Knuth/MMIX constants),
    /// used so every test input is reproducible from its hard-coded start seed.
    ///
    /// # Parameters
    ///
    /// * `seed`: the generator state, advanced in place. Any `u64` is a valid state.
    ///
    /// # Returns
    ///
    /// The top 31 bits of the new state (`>> 33`), because an LCG's low bits have short periods.
    /// Callers reduce it modulo the range they want.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn exact_search_matches_bruteforce() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        // A few random references; include repetitive structure so intervals are non-trivial.
        for trial in 0..12 {
            let len = 200 + (lcg(&mut seed) as usize % 2000);
            let alpha = if trial % 3 == 0 { 2 } else { 4 }; // sometimes only {A,C} -> many repeats
            let ref_seq: Vec<u8> = (0..len).map(|_| (lcg(&mut seed) % alpha) as u8).collect();
            let lsa = LearnedSa::build(ref_seq.clone(), 256);

            // Probe patterns: random substrings of the reference (guaranteed to occur) of varied
            // length, plus random patterns (often absent), plus edge lengths straddling K (=20), so
            // both the key-only and the fall-through-to-reference comparison paths get exercised.
            for _ in 0..300 {
                let mlen = 1 + (lcg(&mut seed) as usize % 40);
                let pattern: Vec<u8> = if lcg(&mut seed).is_multiple_of(2) && len > mlen {
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
    ///
    /// # Parameters
    ///
    /// * `p`: bases as codes `0..=3`. A code `> 3` underflows the `3 - c` subtraction and panics in
    ///   debug builds, which is the desired behaviour for a test helper.
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
            assert_eq!(
                sm.s,
                (hi - lo) as i64,
                "s mismatch, pattern@{start} len {mlen}"
            );

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
                let pat: Vec<u8> = if lcg(&mut seed).is_multiple_of(2) && len > mlen {
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
        // Pattern longer than K (20) still works: past K the key is exhausted and every comparison
        // falls through to `cmp_pattern` against the reference.
        let long_ref: Vec<u8> = (0..100).map(|i| (i % 4) as u8).collect();
        let lsa2 = LearnedSa::build(long_ref.clone(), 16);
        let pat: Vec<u8> = (0..40).map(|i| ((i) % 4) as u8).collect(); // matches at position 0,4,...
        assert_eq!(lsa2.occurrences(&pat), brute(&long_ref, &pat));
    }
}
