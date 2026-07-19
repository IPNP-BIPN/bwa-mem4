//! Suffix array construction by induced sorting (SA-IS), pure Rust.
//!
//! [`suffix_array_with_sentinel`] returns the suffix array of a byte string with an implicit `$`
//! sentinel that sorts before every real symbol. This matches bwa-mem2's convention exactly: the
//! returned vector has length `n+1`, element `0` is `n` (the sentinel suffix), and `[1..]` is the
//! suffix array of the real string (equivalent to `saisxx(s, sa+1, n)` followed by `sa[0]=n`).
//!
//! # Why this file exists at all
//!
//! bwa-mem2 calls Yuta Mori's `saisxx` template (`sais.h`) from `FMI_search::build_index`
//! (`FMI_search.cpp:364-365`), then prepends the sentinel by hand:
//! `status = saisxx(reference_seq.c_str(), suffix_array + 1, pac_len); suffix_array[0] = pac_len;`
//! A suffix array is UNIQUE, so byte-identity does not require copying Mori's algorithm, only
//! producing the same array and the same sentinel-first convention. That is why this is an
//! independent Rust implementation rather than a transliteration.
//!
//! # SA-IS in one paragraph, for the comments below to make sense
//!
//! Classify each position as S-type (its suffix is smaller than the suffix to its right) or L-type
//! (larger). An LMS position ("leftmost S") is an S-type immediately after an L-type; these are the
//! natural "seams" of the string. The algorithm's insight is that once the LMS suffixes are in the
//! right relative order, EVERY other suffix can be placed by two linear scans ("induction"), one
//! left-to-right filling L-types and one right-to-left filling S-types. Sorting the LMS suffixes is
//! itself a smaller suffix-array problem, solved by recursion on a "reduced string" whose letters
//! are names given to LMS substrings. The recursion shrinks by at least half each level, so the
//! whole thing is O(n).
//!
//! # Two implementations live here
//!
//! * [`suffix_array_with_sentinel`]: the clear one, `usize` arrays everywhere. Used by tests and as
//!   the reference for the fast one.
//! * [`suffix_array_inplace`]: the production one, packing the recursion's working data inside the
//!   output array itself. Same output, roughly 8-9 bytes per base instead of ~50, which is the
//!   difference between indexing hg38 on a workstation and not.
//!
//! # Vocabulary
//!
//! * **S-type / L-type**: a position is S-type if the suffix starting there is smaller than the
//!   suffix starting one to its right, L-type otherwise. Classified in one backward pass.
//! * **LMS position** ("leftmost S"): an S-type position whose left neighbour is L-type. These are
//!   the "seams" the algorithm sorts first and induces everything else from. No two are adjacent,
//!   which is the property the in-place variant's `p >> 1` slot trick depends on.
//! * **Bucket**: the contiguous run of suffix-array slots belonging to one starting symbol. The
//!   induction scans fill buckets from the front (L-types) and from the back (S-types).
//! * `alphabet_size` (called `k` in the literature and in Mori's code): one past the largest symbol
//!   value, so buckets are indexed `0..alphabet_size`.
//! * **Reduced string**: the LMS substrings replaced by integer names, forming the smaller suffix
//!   array problem the algorithm recurses on.
//!
//! # Reading order for this file
//!
//! 1. [`suffix_array_with_sentinel`] then [`sa_is`]: the clear implementation. Read [`induce`]
//!    first inside it, since it is the algorithm's core.
//! 2. Only then [`suffix_array_inplace`] / [`sais_rec`]: the same five steps, with the working
//!    arrays folded into the output array to save memory. If the two ever disagree, the clear one
//!    is the specification.
//!
//! INVARIANT shared by both: the input must be terminated by a symbol that is strictly smaller than
//! every other symbol and occurs exactly once. Both arrange this by mapping input bytes to `b + 1`
//! and appending a `0`. Without it the induction has no anchor and the result is silently wrong.

/// "No suffix placed here yet" marker for the `usize` implementation. Distinguishable from a real
/// position because a position is always `< n`.
///
/// The value itself is arbitrary as long as it is unreachable as a TEXT POSITION; `usize::MAX` is
/// the natural choice. Changing it to any value `< n` would make an unfilled SA slot
/// indistinguishable from a real suffix and silently corrupt the induction.
const EMPTY: usize = usize::MAX;

/// Suffix array of `s` including the sentinel row first.
///
/// Result length is `s.len()+1`; `result[0] == s.len()` and `result[1..]` is the suffix array of
/// `s` (shorter suffixes sort first on ties, i.e. the standard suffix array).
///
/// # Parameters
/// * `s`: the text, any byte string including empty. Bytes are arbitrary (all 256 values allowed);
///   no sentinel may be present in `s`, this function appends its own. Supplied by the index
///   builder as the 2-bit-decoded reference, or by tests.
///
/// # Returns
/// A `Vec<i64>` of length `s.len() + 1`. Each element is a TEXT POSITION into `s` (except
/// `result[0] == s.len()`, the position of the appended sentinel). Row `r` of the result names the
/// starting position of the `r`-th smallest suffix.
pub fn suffix_array_with_sentinel(s: &[u8]) -> Vec<i64> {
    // Length of the real text, in bytes. The internal problem size is `n + 1` (with sentinel).
    let n = s.len();
    // Map bytes to 1..=256 and append the 0 sentinel, so 0 is uniquely smallest. Alphabet size is
    // therefore 257, not 256: symbol 0 is the sentinel and no input byte can collide with it.
    // Costs one `usize` per input byte, which is exactly the memory `suffix_array_inplace` avoids.
    let mut t: Vec<usize> = Vec::with_capacity(n + 1);
    for &b in s {
        t.push(b as usize + 1);
    }
    t.push(0);
    sa_is(&t, 257).into_iter().map(|x| x as i64).collect()
}

/// LMS ("leftmost S") = an S-type position whose left neighbour is L-type. Position 0 can never be
/// LMS since it has no left neighbour, hence the `i > 0`. LMS positions are at least 2 apart (an
/// S-type immediately after another S-type is not leftmost), a spacing the in-place variant's
/// `p >> 1` naming slots depend on.
///
/// # Parameters
/// * `is_s`: the S/L classification, indexed by TEXT POSITION; `true` means S-type.
/// * `i`: a TEXT POSITION in `0..is_s.len()`.
///
/// # Returns
/// True if `i` is an LMS position. Never true for `i == 0`.
#[inline]
fn is_lms(is_s: &[bool], i: usize) -> bool {
    i > 0 && is_s[i] && !is_s[i - 1]
}

/// Bucket start offsets (`out[c]` = first index of symbol `c`'s bucket).
///
/// # Parameters
/// * `s`: the integer text; every symbol must be `< alphabet_size` or the counting loop panics.
/// * `alphabet_size`: one past the largest symbol value, so `0..alphabet_size` indexes buckets.
///
/// # Returns
/// A fresh vector of `alphabet_size` SUFFIX-ARRAY ROW indices: `out[c]` is the first SA row
/// belonging to symbol `c`. Callers mutate their own copy as a moving write cursor, which is why a
/// fresh vector is returned on every call rather than one being cached.
fn bucket_starts(s: &[usize], alphabet_size: usize) -> Vec<usize> {
    // Phase 1: `c[x]` is the OCCURRENCE COUNT of symbol `x` in the text.
    let mut c = vec![0usize; alphabet_size];
    for &x in s {
        c[x] += 1;
    }
    // Phase 2: exclusive prefix sum, turning counts into row offsets in place. Invariant at the top
    // of each iteration: `sum` is the total number of symbols strictly smaller than the current
    // one, which is exactly the first SA row of its bucket.
    let mut sum = 0;
    for slot in c.iter_mut() {
        // Saved before overwriting, since the slot is about to become an offset, not a count.
        let cnt = *slot;
        *slot = sum;
        sum += cnt;
    }
    c
}

/// Bucket end offsets (`out[c]` = one past the last index of symbol `c`'s bucket).
///
/// # Parameters
/// * `s`: the integer text; every symbol must be `< alphabet_size`.
/// * `alphabet_size`: one past the largest symbol value.
///
/// # Returns
/// A fresh vector of `alphabet_size` SUFFIX-ARRAY ROW indices: `out[c]` is one past the last SA row
/// of symbol `c`'s bucket. Callers pre-decrement it as a backwards write cursor, so, as with
/// `bucket_starts`, each call must get its own copy.
fn bucket_ends(s: &[usize], alphabet_size: usize) -> Vec<usize> {
    // Phase 1: occurrence counts per symbol.
    let mut c = vec![0usize; alphabet_size];
    for &x in s {
        c[x] += 1;
    }
    // Phase 2: inclusive prefix sum. Invariant: after processing symbol `x`, `sum` is the count of
    // symbols `<= x`, i.e. one past `x`'s last SA row.
    let mut sum = 0;
    for slot in c.iter_mut() {
        sum += *slot;
        *slot = sum;
    }
    c
}

/// Induce L-type then S-type suffixes from the LMS suffixes already placed in `sa`.
///
/// The heart of SA-IS, and it is two scans:
/// 1. LEFT TO RIGHT. For each already-placed suffix `j`, if `j-1` is L-type, append it to the FRONT
///    of its bucket. Correct because an L-type suffix is larger than its right neighbour, so it
///    must sit before it in that bucket, and by the time the scan reaches `j` the neighbour is
///    final. `starts[c]` is a moving write cursor.
/// 2. RIGHT TO LEFT, symmetrically, filling S-types from the BACK of their buckets. This scan
///    overwrites the provisional LMS placements from the caller with their final positions, which
///    is why the caller can seed them approximately in scan 1 and exactly here.
///
/// Both scans read from `sa` while writing to it. That is intentional: entries are always written
/// ahead of (scan 1) or behind (scan 2) the read cursor, so a value is never read before it is
/// final. Reordering or fusing the two scans breaks the algorithm.
///
/// # Parameters
/// * `sa`: the suffix array under construction, length `n`, indexed by SUFFIX-ARRAY ROW and holding
///   TEXT POSITIONS or `EMPTY`. On entry the caller must have placed the LMS suffixes (approximately
///   in step 1, exactly in step 4) at their bucket ends, everything else `EMPTY`. On exit every row
///   holds its final text position for that call's input placement.
/// * `s`: the integer text, symbols in `0..alphabet_size`, last symbol the unique smallest.
/// * `is_s`: S/L classification indexed by TEXT POSITION, as computed by the caller.
/// * `alphabet_size`: one past the largest symbol, sizing the bucket arrays.
fn induce(sa: &mut [usize], s: &[usize], is_s: &[bool], alphabet_size: usize) {
    // Problem size: text length and SA length are both `n`.
    let n = s.len();
    // Scan 1: left to right, placing L-type suffixes at the FRONT of their buckets.
    // `starts[c]` is a moving WRITE CURSOR: the next free row at the front of symbol `c`'s bucket.
    let mut starts = bucket_starts(s, alphabet_size);
    // Invariant at the top of each iteration: every L-type suffix that sorts before row `i` has
    // already been written to its final row, and rows `< i` are read-only from here on. Rows still
    // holding EMPTY at `i` are S-types that scan 2 will fill.
    for i in 0..n {
        // TEXT POSITION of the suffix currently sitting in row `i`, or EMPTY if unfilled.
        let j = sa[i];
        // Skip empties, and skip `j == 0` because position 0 has no predecessor `j-1` to induce.
        if j != EMPTY && j != 0 && !is_s[j - 1] {
            // CHARACTER at the predecessor position, hence the BUCKET the predecessor belongs in.
            let c = s[j - 1];
            sa[starts[c]] = j - 1;
            starts[c] += 1;
        }
    }
    // Scan 2: right to left, placing S-type suffixes at the BACK of their buckets.
    // `ends[c]` is a backwards WRITE CURSOR: one past the last free row of symbol `c`'s bucket.
    let mut ends = bucket_ends(s, alphabet_size);
    // Invariant at the top of each iteration: rows `> i` are final, and all L-type entries from
    // scan 1 are already correct, so every read here sees a settled text position. The provisional
    // LMS entries the caller seeded get overwritten by their exact placements during this scan.
    for i in (0..n).rev() {
        // TEXT POSITION in row `i` (never EMPTY by the end of this scan, but can be mid-scan).
        let j = sa[i];
        if j != EMPTY && j != 0 && is_s[j - 1] {
            // Bucket of the S-type predecessor.
            let c = s[j - 1];
            ends[c] -= 1;
            sa[ends[c]] = j - 1;
        }
    }
}

/// Whether the two LMS SUBSTRINGS (not suffixes) starting at `a` and `b` are identical.
///
/// An LMS substring runs from one LMS position up to and INCLUDING the next one. Two are equal only
/// if their symbols AND their S/L type sequences match and they end at the same relative offset,
/// which is what the `is_lms` check at `i > 0` enforces: hitting an LMS boundary in one but not the
/// other means different lengths, so unequal. Comparing types as well as symbols is not redundant;
/// equal symbol runs can still be different substrings when the following context differs.
///
/// # Parameters
/// * `s`: the integer text.
/// * `is_s`: S/L classification indexed by TEXT POSITION.
/// * `a`, `b`: TEXT POSITIONS, each an LMS position in `0..s.len()`, being compared.
///
/// # Returns
/// True if the two LMS substrings are character-for-character and type-for-type identical, so the
/// caller may give them the same name.
fn lms_substr_equal(s: &[usize], is_s: &[bool], a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    let n = s.len();
    // Offset from each substring's start; both are walked in lockstep.
    let mut i = 0usize;
    loop {
        // The two TEXT POSITIONS currently being compared, one in each substring.
        let pa = a + i;
        let pb = b + i;
        // Running off the end: equal only if both ran off together (the sentinel occurs once, so in
        // practice at most one substring can reach the end).
        if pa >= n || pb >= n {
            return pa >= n && pb >= n;
        }
        if s[pa] != s[pb] || is_s[pa] != is_s[pb] {
            return false;
        }
        // Past the first character, an LMS boundary terminates the substring. Whether each side has
        // reached its terminating LMS position.
        if i > 0 {
            let al = is_lms(is_s, pa);
            let bl = is_lms(is_s, pb);
            // If only one side ended here the substrings have different lengths, hence unequal; if
            // both ended, everything before matched, hence equal.
            if al || bl {
                return al && bl;
            }
        }
        i += 1;
    }
}

/// SA-IS proper: suffix array of integer string `s` over alphabet `0..alphabet_size`, whose last symbol must be
/// the unique smallest. Allocates fresh arrays per recursion level (the readable, memory-hungry
/// variant); see [`sais_rec`] for the packed one.
///
/// # Parameters
/// * `s`: the integer text. Symbols must lie in `0..alphabet_size`, and `s[n-1]` must be the unique
///   smallest symbol (the sentinel). Violating either gives a silently wrong answer, not a panic.
///   At the top level this is the byte text mapped to `b+1` with a `0` appended; at deeper levels it
///   is the reduced string of LMS names.
/// * `alphabet_size`: one past the largest symbol value, so buckets are `0..alphabet_size`. 257 at
///   the top level; `num_names` in recursive calls.
///
/// # Returns
/// The suffix array of `s`: a vector of length `n` whose row `r` holds the TEXT POSITION of the
/// `r`-th smallest suffix. Row 0 is always `n-1`, the sentinel suffix.
fn sa_is(s: &[usize], alphabet_size: usize) -> Vec<usize> {
    // Text length; also the number of suffixes, hence the SA length.
    let n = s.len();
    // The output suffix array, indexed by SUFFIX-ARRAY ROW, holding TEXT POSITIONS. Starts entirely
    // unfilled; the induction relies on `EMPTY` to tell placed rows from unplaced ones.
    let mut sa = vec![EMPTY; n];
    if n == 1 {
        sa[0] = 0;
        return sa;
    }

    // Classify S-type (true) / L-type positions, right to left. `s[i]` is S-type if it is smaller
    // than its right neighbour, or equal to it AND the neighbour is S-type (equal symbols inherit
    // the classification of the first position where the runs differ). The last symbol is the
    // sentinel, defined S-type, which anchors the whole recurrence. One backward pass, O(n).
    // `is_s[i] == true` means TEXT POSITION `i` is S-type.
    let mut is_s = vec![false; n];
    is_s[n - 1] = true;
    // Backwards is essential: `is_s[i]` depends on `is_s[i+1]`, already computed.
    for i in (0..n - 1).rev() {
        is_s[i] = s[i] < s[i + 1] || (s[i] == s[i + 1] && is_s[i + 1]);
    }

    // Step 1: place LMS suffixes at bucket ends, then induce.
    // `ends[c]` is a backwards WRITE CURSOR into symbol `c`'s bucket, consumed by this loop.
    let mut ends = bucket_ends(s, alphabet_size);
    // Reverse order so that, within one bucket, LMS positions land in increasing text order. Their
    // relative order is arbitrary at this stage; `induce` only needs them in the right buckets.
    for i in (0..n).rev() {
        if is_lms(&is_s, i) {
            // CHARACTER at the LMS position, i.e. which bucket the suffix belongs to.
            let c = s[i];
            ends[c] -= 1;
            sa[ends[c]] = i;
        }
    }
    // After this call the LMS SUBSTRINGS are correctly ordered in `sa` (the LMS SUFFIXES are not yet,
    // which is what the recursion in step 3 fixes).
    induce(&mut sa, s, &is_s, alphabet_size);

    // Step 2: name LMS substrings in sorted order.
    // The LMS TEXT POSITIONS read out of `sa` in sorted-substring order.
    let mut lms_sorted = Vec::new();
    for &p in sa.iter() {
        if p != EMPTY && is_lms(&is_s, p) {
            lms_sorted.push(p);
        }
    }
    // `names[p]` is the integer NAME of the LMS substring starting at TEXT POSITION `p`; `EMPTY` at
    // every non-LMS position (those entries are never read).
    let mut names = vec![EMPTY; n];
    // The name being assigned; increments only when the substring differs from its predecessor, so
    // equal substrings share a name and names come out in sorted order (0, 0, 1, 2, 2, ...).
    let mut cur = 0usize;
    names[lms_sorted[0]] = 0;
    // TEXT POSITION of the previous LMS substring in sorted order, for the equality test.
    let mut prev = lms_sorted[0];
    // Invariant at the top of each iteration: every LMS substring earlier in sorted order has been
    // named, and `cur` is the name given to `prev`.
    for &p in lms_sorted.iter().skip(1) {
        if !lms_substr_equal(s, &is_s, prev, p) {
            cur += 1;
        }
        names[p] = cur;
        prev = p;
    }
    // Number of DISTINCT names, i.e. the alphabet size of the reduced string. Names run 0..=cur.
    let num_names = cur + 1;

    // Reduced string in position order.
    // `lms_positions[k]` is the TEXT POSITION of the k-th LMS position in TEXT order; it is the map
    // used in step 4 to turn a reduced-string row back into a text position.
    let mut lms_positions = Vec::new();
    // `reduced[k]` is the NAME of that same LMS substring, so `reduced` is the smaller string whose
    // suffix array is exactly the sorted order of the LMS suffixes. Length <= n/2.
    let mut reduced = Vec::new();
    for (i, &name) in names.iter().enumerate() {
        if is_lms(&is_s, i) {
            lms_positions.push(i);
            reduced.push(name);
        }
    }

    // Step 3: suffix array of the reduced string. If every LMS substring got a distinct name the
    // reduced string is a permutation, so its suffix array is just the inverse permutation and the
    // recursion stops. Otherwise recurse. Since LMS positions are >= 2 apart, `reduced.len() <= n/2`
    // and the recursion depth is O(log n).
    // `reduced_sa[r]` is a ROW of the reduced problem: it holds an INDEX into `lms_positions` (and
    // into `reduced`), not a text position of `s`. Step 4 translates it.
    let reduced_sa = if num_names == reduced.len() {
        // Every name distinct, so `reduced` is a permutation of `0..len`, and its suffix array is
        // just its inverse: the suffix starting where name `k` sits is the k-th smallest.
        let mut rsa = vec![0usize; reduced.len()];
        for (i, &name) in reduced.iter().enumerate() {
            rsa[name] = i;
        }
        rsa
    } else {
        sa_is(&reduced, num_names)
    };

    // Step 4: re-place LMS suffixes in sorted order, then induce the final SA.
    // Wipe the substring-sorted contents; only the exact LMS suffix order below may seed the final
    // induction.
    for x in sa.iter_mut() {
        *x = EMPTY;
    }
    // Fresh backwards write cursors: the previous set was consumed in step 1.
    let mut ends = bucket_ends(s, alphabet_size);
    // Reverse order so that, filling each bucket from the back, the LMS suffixes end up in
    // increasing sorted order within the bucket.
    for &idx in reduced_sa.iter().rev() {
        // TEXT POSITION of this LMS suffix, recovered from its reduced-string index.
        let p = lms_positions[idx];
        // Its bucket.
        let c = s[p];
        ends[c] -= 1;
        sa[ends[c]] = p;
    }
    // With the LMS suffixes now in exact sorted order, this induction produces the final, complete
    // suffix array: every row holds its true text position.
    induce(&mut sa, s, &is_s, alphabet_size);

    sa
}

/// Memory-efficient in-place SA-IS: same result as [`suffix_array_with_sentinel`] but built with
/// ~8-9 bytes/base peak (the SA array plus a type bitvector and per-level bucket arrays), instead
/// of the ~50 bytes/base of the array-heavy version. This is what lets the full genome index fit in
/// RAM. The SA is unique, so the output is byte-identical to any correct construction.
///
/// # Parameters
/// * `s`: the text, any byte string including empty; all 256 byte values allowed, no sentinel
///   required or permitted (one is synthesised by `ByteStr`). Supplied by the index builder.
///
/// # Returns
/// Exactly what [`suffix_array_with_sentinel`] returns for the same input: length `s.len() + 1`,
/// `result[0] == s.len()`, and `result[1..]` the suffix array of `s`.
pub fn suffix_array_inplace(s: &[u8]) -> Vec<i64> {
    // The input is read as `b+1` with a `0` sentinel appended (uniquely smallest) via `ByteStr`,
    // with no i64 copy of the input. SA-IS runs on the full length-(n+1) string; the sentinel
    // suffix sorts first, so sa[0] == n automatically.
    // The single big allocation: the output SA, and also the scratch space every recursion level
    // packs its reduced problem into. One row per suffix of the sentinel-terminated text.
    let mut sa = vec![0i64; s.len() + 1];
    // 257 = 256 byte values shifted to 1..=256, plus symbol 0 reserved for the sentinel.
    sais_rec(&ByteStr(s), &mut sa, 257);
    sa
}

/// EMPTY marker for the in-place SA array (positions are always >= 0). Using `-1` rather than a
/// large sentinel lets the induction test `j > 0`, which rejects both "empty" and "position 0"
/// (position 0 has no predecessor to induce) in a single comparison.
///
/// Any negative value would work for the marker alone, but `-1` specifically is what makes the
/// combined `j > 0` test valid. Changing it to a positive value would break that test and require
/// re-splitting it into two comparisons in `induce_i`.
const IEMPTY: i64 = -1;

/// S-type classification as a bitvector (one bit per position): 1 bit per base rather than the
/// `Vec<bool>`'s 8, worth about 5.4 GB on a 2L human reference. LSB-first within each word, an
/// arbitrary choice with no on-disk consequences (this never leaves memory).
///
/// Field 0 is the bit storage: bit `i` (word `i >> 6`, bit `i & 63`) is 1 iff TEXT POSITION `i` is
/// S-type, 0 for L-type. Sized to cover `0..n`; bits past `n` in the final word stay zero and are
/// never read.
struct TypeBits(Vec<u64>);
impl TypeBits {
    /// Allocate a bitvector covering positions `0..n`, all initially L-type (zero).
    ///
    /// # Parameters
    /// * `n`: number of TEXT POSITIONS to cover; rounds up to `ceil(n/64)` 64-bit words.
    fn new(n: usize) -> Self {
        TypeBits(vec![0u64; n.div_ceil(64)])
    }
    /// Mark TEXT POSITION `i` as S-type. There is no clear operation: the classification pass only
    /// ever sets bits, starting from an all-L vector.
    ///
    /// # Parameters
    /// * `i`: TEXT POSITION in `0..n`; out of range panics on the word index.
    #[inline]
    fn set_s(&mut self, i: usize) {
        self.0[i >> 6] |= 1u64 << (i & 63);
    }
    /// Whether TEXT POSITION `i` is S-type.
    ///
    /// # Parameters
    /// * `i`: TEXT POSITION in `0..n`.
    #[inline]
    fn is_s(&self, i: usize) -> bool {
        (self.0[i >> 6] >> (i & 63)) & 1 != 0
    }
    /// LMS = S-type preceded by an L-type.
    ///
    /// # Parameters
    /// * `i`: TEXT POSITION in `0..n`. Position 0 is never LMS (no left neighbour), which also
    ///   keeps the `i - 1` below from underflowing.
    #[inline]
    fn is_lms(&self, i: usize) -> bool {
        i > 0 && self.is_s(i) && !self.is_s(i - 1)
    }
}

/// Read-only integer string for SA-IS. Implemented for the byte reference at the top level (no i64
/// copy of the input) and for an i64 slice (the reduced sub-problem packed inside the SA array).
trait IntStr {
    /// The CHARACTER at TEXT POSITION `i` (`i` must be `< len()`). Always non-negative and less
    /// than the `alphabet_size` passed alongside the string.
    fn get(&self, i: usize) -> i64;
    /// Number of symbols, counting the sentinel; equals the number of suffixes and the SA length.
    fn len(&self) -> usize;
}

/// Top-level: the byte reference mapped to `b+1` with a `0` sentinel appended (uniquely smallest).
///
/// Field 0 is the raw text, borrowed and never copied or modified: the `+1` shift and the trailing
/// `0` are synthesised on the fly by the `IntStr` impl, which is what avoids an i64 copy of the
/// whole genome.
struct ByteStr<'a>(&'a [u8]);
impl IntStr for ByteStr<'_> {
    #[inline]
    fn get(&self, i: usize) -> i64 {
        if i < self.0.len() {
            i64::from(self.0[i]) + 1
        } else {
            0 // appended sentinel
        }
    }
    #[inline]
    fn len(&self) -> usize {
        self.0.len() + 1
    }
}

/// Recursion levels: the reduced string, an i64 slice living in the SA array's tail.
///
/// Field 0 is the reduced string: one LMS NAME per element, values in `0..num_names`, already
/// sentinel terminated (the last LMS position in text order is the sentinel at `n-1`, which sorts
/// first among LMS substrings and so receives the unique smallest name, 0). Borrowed from the tail
/// of the parent level's `sa`, never owned.
struct IntSlice<'a>(&'a [i64]);
impl IntStr for IntSlice<'_> {
    #[inline]
    fn get(&self, i: usize) -> i64 {
        self.0[i]
    }
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// Bucket end offsets (`out[c]` = one past symbol `c`'s bucket).
///
/// The i64 twin of [`bucket_ends`]; see it for the two-phase count-then-prefix-sum explanation.
///
/// # Parameters
/// * `s`: the integer string, any [`IntStr`]; every symbol must be `< alphabet_size`.
/// * `alphabet_size`: one past the largest symbol value; sizes the returned vector.
///
/// # Returns
/// `alphabet_size` SUFFIX-ARRAY ROW indices, one past the last row of each symbol's bucket. A fresh
/// copy per call, since callers pre-decrement it as a write cursor.
fn bucket_ends_i<S: IntStr>(s: &S, alphabet_size: usize) -> Vec<i64> {
    // Occurrence count of each symbol, then rewritten in place into offsets.
    let mut c = vec![0i64; alphabet_size];
    for i in 0..s.len() {
        c[s.get(i) as usize] += 1;
    }
    // Running total of symbols seen so far; after each step it is one past that symbol's last row.
    let mut sum = 0;
    for slot in c.iter_mut() {
        sum += *slot;
        *slot = sum;
    }
    c
}

/// Bucket start offsets.
///
/// The i64 twin of [`bucket_starts`].
///
/// # Parameters
/// * `s`: the integer string; every symbol must be `< alphabet_size`.
/// * `alphabet_size`: one past the largest symbol value.
///
/// # Returns
/// `alphabet_size` SUFFIX-ARRAY ROW indices, the first row of each symbol's bucket. Fresh per call;
/// callers increment it as a write cursor.
fn bucket_starts_i<S: IntStr>(s: &S, alphabet_size: usize) -> Vec<i64> {
    let mut c = vec![0i64; alphabet_size];
    for i in 0..s.len() {
        c[s.get(i) as usize] += 1;
    }
    // Count of symbols strictly smaller than the current one, i.e. its bucket's first row.
    let mut sum = 0;
    for slot in c.iter_mut() {
        // Saved before the slot is overwritten with the offset.
        let cnt = *slot;
        *slot = sum;
        sum += cnt;
    }
    c
}

/// Induce L-type then S-type suffixes from the LMS suffixes already placed in `sa`.
///
/// The i64 twin of [`induce`]; that function's doc comment explains why the two scans work and why
/// reading and writing the same array is safe. The only difference here is the `IEMPTY == -1`
/// encoding, which lets `j > 0` reject "empty" and "position 0" in one comparison.
///
/// # Parameters
/// * `sa`: SA under construction, length `n`, indexed by SUFFIX-ARRAY ROW, holding TEXT POSITIONS or
///   `IEMPTY`. Caller must have seeded the LMS suffixes at their bucket ends and set everything else
///   to `IEMPTY`.
/// * `s`: the integer string, last symbol the unique smallest.
/// * `types`: S/L classification bitvector indexed by TEXT POSITION.
/// * `alphabet_size`: one past the largest symbol, sizing the bucket arrays.
fn induce_i<S: IntStr>(sa: &mut [i64], s: &S, types: &TypeBits, alphabet_size: usize) {
    let n = s.len();
    // Scan 1: left to right, L-types to the FRONT of their buckets.
    // `starts[c]` is the moving write cursor for symbol `c`'s bucket.
    let mut starts = bucket_starts_i(s, alphabet_size);
    // Invariant at the top of each iteration: all L-type suffixes sorting before row `i` are already
    // in their final rows; rows still `IEMPTY` belong to S-types and are filled by scan 2.
    for i in 0..n {
        // TEXT POSITION in row `i`, or `IEMPTY` (-1) if unfilled.
        let j = sa[i];
        // `j > 0` rejects both -1 (empty) and 0 (no predecessor position to induce from).
        if j > 0 && !types.is_s((j - 1) as usize) {
            // CHARACTER at the predecessor position, hence its BUCKET.
            let c = s.get((j - 1) as usize) as usize;
            sa[starts[c] as usize] = j - 1;
            starts[c] += 1;
        }
    }
    // Scan 2: right to left, S-types to the BACK of their buckets.
    // `ends[c]` is the backwards write cursor for symbol `c`'s bucket.
    let mut ends = bucket_ends_i(s, alphabet_size);
    // Invariant at the top of each iteration: rows `> i` are final. The caller's provisional LMS
    // seeds are overwritten with their exact placements as this scan proceeds.
    for i in (0..n).rev() {
        // TEXT POSITION in row `i`.
        let j = sa[i];
        if j > 0 && types.is_s((j - 1) as usize) {
            // Bucket of the S-type predecessor.
            let c = s.get((j - 1) as usize) as usize;
            ends[c] -= 1;
            sa[ends[c] as usize] = j - 1;
        }
    }
}

/// Whether the LMS substrings starting at `a` and `b` are equal (same symbols and S/L types up to
/// and including the next LMS boundary).
///
/// The i64 twin of [`lms_substr_equal`]; see that for why types are compared as well as symbols.
///
/// # Parameters
/// * `s`: the integer string.
/// * `types`: S/L classification bitvector indexed by TEXT POSITION.
/// * `a`, `b`: TEXT POSITIONS, both LMS positions in `0..s.len()`.
///
/// # Returns
/// True if the two LMS substrings are identical, so the caller may reuse the same name.
fn lms_eq<S: IntStr>(s: &S, types: &TypeBits, a: usize, b: usize) -> bool {
    if a == b {
        return true;
    }
    let n = s.len();
    // Common offset from both substring starts, advanced in lockstep.
    let mut i = 0usize;
    loop {
        // The pair of TEXT POSITIONS being compared this step.
        let (pa, pb) = (a + i, b + i);
        // Equal only if both ran off the end at the same offset.
        if pa >= n || pb >= n {
            return pa >= n && pb >= n;
        }
        if s.get(pa) != s.get(pb) || types.is_s(pa) != types.is_s(pb) {
            return false;
        }
        if i > 0 {
            // Whether each side has reached its terminating LMS boundary at this offset.
            let (al, bl) = (types.is_lms(pa), types.is_lms(pb));
            // Only one ended: different lengths, unequal. Both ended: everything matched, equal.
            if al || bl {
                return al && bl;
            }
        }
        i += 1;
    }
}

/// In-place SA-IS on integer string `s` (values in `0..alphabet_size`, `s[n-1]` the unique
/// smallest sentinel),
/// writing the suffix array of `s` into `sa` (length `n`). The reduced sub-problem is packed into
/// `sa` itself, so no per-level O(n) array is allocated (only a type bitvector and one stage-3 temp).
///
/// # Parameters
/// * `s`: the integer string for this level. Symbols must be in `0..alphabet_size` and `s[n-1]` must
///   be the unique smallest. At the top level a [`ByteStr`] over the reference; deeper down an
///   [`IntSlice`] borrowing the tail of the parent's `sa`.
/// * `sa`: output AND scratch, length exactly `s.len()`. Contents on entry are ignored (step 2
///   fills it). On exit, row `r` holds the TEXT POSITION of the `r`-th smallest suffix of `s`.
///   Between steps it also carries the compacted LMS list, the name slots and the reduced string;
///   the region comments below say which part means what at each point.
/// * `alphabet_size`: one past the largest symbol; 257 at the top level, `num_names` when
///   recursing. Sizes every bucket array, so an over-large value costs memory and a too-small one
///   panics on the bucket index.
fn sais_rec<S: IntStr>(s: &S, sa: &mut [i64], alphabet_size: usize) {
    // Problem size at this level: symbol count, suffix count and `sa.len()` are all `n`.
    let n = s.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        sa[0] = 0;
        return;
    }

    // ---- Step 1: classify every position S-type or L-type ------------------------------------
    // `types` is the S/L classification for this level, one bit per TEXT POSITION. The sentinel at
    // `n-1` is S-type by definition and anchors the backward recurrence.
    let mut types = TypeBits::new(n);
    types.set_s(n - 1);
    // Backwards, because position `i`'s type depends on `i+1`'s, which must already be decided.
    for i in (0..n - 1).rev() {
        // The CHARACTER at `i` and its right neighbour. Equal characters inherit the neighbour's
        // type, which propagates the decision from the first position where the two runs differ.
        let (si, si1) = (s.get(i), s.get(i + 1));
        if si < si1 || (si == si1 && types.is_s(i + 1)) {
            types.set_s(i);
        }
    }

    // ---- Step 2: bucket LMS suffixes at their bucket ends, then induce ------------------------
    // This sorts the LMS SUBSTRINGS (not yet the LMS suffixes), which is all step 3 needs.
    sa.fill(IEMPTY);
    // Backwards write cursors, one per symbol, consumed by the seeding loop below.
    let mut ends = bucket_ends_i(s, alphabet_size);
    // From 1 because position 0 can never be LMS. Reverse order so that, filling buckets from the
    // back, LMS positions land in increasing text order within each bucket. Their order here is not
    // yet meaningful; `induce_i` only needs them in the correct buckets.
    for i in (1..n).rev() {
        if types.is_lms(i) {
            // CHARACTER at the LMS position, i.e. its bucket.
            let c = s.get(i) as usize;
            ends[c] -= 1;
            sa[ends[c] as usize] = i as i64;
        }
    }
    induce_i(sa, s, &types, alphabet_size);

    // Compact the sorted LMS positions into sa[0..n_lms].
    // Number of LMS positions found so far, and simultaneously the write cursor for the compaction.
    // It ends as the total LMS count, which is `<= n/2` because LMS positions are >= 2 apart. That
    // bound is what every later region-fits argument in this function rests on.
    let mut n_lms = 0usize;
    // Invariant at the top of each iteration: `sa[0..n_lms]` holds the LMS TEXT POSITIONS found in
    // rows `< i`, in LMS-substring-sorted order. The write index never exceeds the read index `i`,
    // so this in-place compaction cannot clobber an unread entry.
    for i in 0..n {
        // TEXT POSITION in row `i` of the induced SA.
        let p = sa[i];
        if p != IEMPTY && types.is_lms(p as usize) {
            sa[n_lms] = p;
            n_lms += 1;
        }
    }

    // ---- Step 3: name the LMS substrings, then build the reduced string ------------------------
    // Store the name of the LMS at position p into sa[n_lms + p/2] (LMS positions are >= 2 apart,
    // so p/2 are distinct and fit in [n_lms, n_lms + n/2) <= n).
    //
    // This is the central space trick and deserves spelling out. We need a name per LMS position,
    // keyed by position, but have no array to spare. Because no two LMS positions are adjacent,
    // halving a position is INJECTIVE over LMS positions, so `p >> 1` gives every LMS its own slot.
    // Those slots occupy `[n_lms, n_lms + n/2)`, and `n_lms <= n/2`, so they fit in the region of
    // `sa` past the compacted LMS list without ever colliding with it. The region is cleared to
    // IEMPTY first so the gap-skipping compaction below can tell a written name from an unused slot.
    for x in sa[n_lms..n].iter_mut() {
        *x = IEMPTY;
    }
    // The NAME currently being handed out. Increments only on a difference from the previous
    // substring, so names are assigned in sorted order and equal substrings share one.
    let mut name: i64 = 0;
    // TEXT POSITION of the previous LMS substring in sorted order, or -1 on the first iteration
    // (there is nothing to compare against yet, hence the `prev >= 0` guard).
    let mut prev: i64 = -1;
    // Invariant at the top of each iteration: every LMS substring in `sa[0..idx]` has had its name
    // written to its `n_lms + (p >> 1)` slot, and `name` is the name given to `prev`.
    for idx in 0..n_lms {
        // TEXT POSITION of the idx-th smallest LMS substring, read from the compacted list.
        let p = sa[idx] as usize;
        if prev >= 0 && !lms_eq(s, &types, prev as usize, p) {
            name += 1;
        }
        sa[n_lms + (p >> 1)] = name;
        prev = p as i64;
    }
    // Alphabet size of the reduced string: the count of DISTINCT names. Names run 0..=name, so the
    // count is `name + 1`, except with no LMS positions at all where the reduced string is empty.
    let num_names = if n_lms == 0 { 0 } else { (name + 1) as usize };

    // Build the reduced string RA (length n_lms, values 0..num_names) into the tail
    // sa[n-n_lms..n], reading names down from sa[n_lms..n_lms+n/2] (safe: write index >= read
    // index throughout).
    {
        // Name slots span sa[n_lms ..= n_lms + (n-1)/2] (the max LMS position is the sentinel at
        // n-1).
        //
        // Walk both cursors DOWNWARDS, compacting the sparse name slots into the last `n_lms`
        // entries. Backwards is required for in-place safety: the write index `(n-n_lms)+j` is
        // always >= the read index `i`, so a name is never clobbered before it has been copied.
        // Going forwards would overwrite unread slots whenever the names are densely packed.
        // WRITE cursor, counting down: `j` is the next free slot from the end of the reduced
        // string's region `sa[n-n_lms..n]`, expressed relative to that region's start.
        let mut j = n_lms as i64 - 1;
        // READ cursor, counting down over the sparse name slots `sa[n_lms ..= n_lms + (n-1)/2]`.
        // Both are `i64`, not `usize`, so the loop can safely step one past the bottom of the range.
        let mut i = (n_lms + (n - 1) / 2) as i64;
        // Invariant at the top of each iteration: every name in slots `> i` has been copied into
        // `sa[(n-n_lms)+j+1 ..n]`, in the same relative (text) order, and `j+1` names remain to be
        // written. `(n-n_lms)+j >= i` holds throughout, which is what makes this safe in place.
        while i >= n_lms as i64 {
            if sa[i as usize] != IEMPTY {
                sa[(n - n_lms) + j as usize] = sa[i as usize];
                j -= 1;
            }
            i -= 1;
        }
    }

    // ---- Step 4: solve the reduced problem, recursing only if names repeat --------------------
    // SA1 goes into sa[0..n_lms]. The reduced string RA now occupies the tail `sa[n-n_lms..n]` and
    // the answer goes in the head `sa[0..n_lms]`; `n - n_lms >= n_lms` (LMS spacing again), so
    // `split_at_mut` cleanly separates read region from write region and the recursion needs no new
    // allocation of its own beyond a type bitvector and buckets.
    if num_names < n_lms {
        // Names repeat, so the reduced string is not a permutation and its order must actually be
        // computed. `tail` IS the reduced string (read-only to the callee), `head[..n_lms]` is the
        // disjoint region its answer is written into.
        let (head, tail) = sa.split_at_mut(n - n_lms);
        sais_rec(&IntSlice(tail), &mut head[..n_lms], num_names);
    } else {
        // All names distinct: SA1 is the inverse permutation of RA.
        // `r` is the NAME at reduced-string index `i`; since names are distinct and sorted, the
        // suffix starting at `i` is the r-th smallest, so row `r` gets `i`.
        for i in 0..n_lms {
            let r = sa[n - n_lms + i] as usize;
            sa[r] = i as i64;
        }
    }

    // ---- Step 5: map ranks back to text positions and induce the final SA ---------------------
    // `sorted_lms` is the one unavoidable O(n_lms) temp per level. Since `n_lms <= n/2` and each
    // level halves, the sum over all levels is under one extra `n`-sized array, which is what keeps
    // the "8-9 bytes per base" claim in [`suffix_array_inplace`] true.
    // `sorted_lms[r]` = the TEXT POSITION of the r-th smallest LMS SUFFIX. This is the exact seeding
    // order the final induction needs. It cannot live inside `sa` because the next step wipes `sa`.
    let mut sorted_lms = vec![0i64; n_lms];
    {
        // SA1 holds RANKS (indices into the LMS-in-text-order list), not text positions, so it must
        // be mapped back before induction. Rebuild that list into the tail, then index it by rank.
        // LMS positions in text order into sa[n-n_lms..n].
        // Write cursor into the tail region: `j` counts LMS positions emitted so far. Overwriting
        // the reduced string here is safe because step 4 has already consumed it.
        let mut j = 0usize;
        for i in 1..n {
            if types.is_lms(i) {
                sa[n - n_lms + j] = i as i64;
                j += 1;
            }
        }
        // Translate ranks to text positions: `sa[i]` (from SA1, still in the head) is a RANK, an
        // index into the just-rebuilt text-order LMS list, and indexing that list yields the TEXT
        // POSITION. `i` here is the SA row within the LMS ordering.
        for (i, out) in sorted_lms.iter_mut().enumerate() {
            *out = sa[n - n_lms + sa[i] as usize];
        }
    }
    // Discard all the scratch; only `sorted_lms` (now a separate allocation) survives.
    sa.fill(IEMPTY);
    // Fresh backwards write cursors for the final seeding.
    let mut ends = bucket_ends_i(s, alphabet_size);
    // Reverse order so that filling each bucket from the back leaves the LMS suffixes in increasing
    // sorted order within it.
    for &p in sorted_lms.iter().rev() {
        // Bucket of this LMS suffix.
        let c = s.get(p as usize) as usize;
        ends[c] -= 1;
        sa[ends[c] as usize] = p;
    }
    // With the LMS suffixes exactly sorted, this induction completes the suffix array: on return
    // every row of `sa` holds its final text position.
    induce_i(sa, s, &types, alphabet_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive suffix array (with sentinel-first), for validation.
    ///
    /// O(n^2 log n), so only usable on the short strings below, but it is obviously correct, which
    /// is the point: it is the independent oracle the two SA-IS implementations are checked against.
    ///
    /// # Parameters
    /// * `s`: the text; any bytes, any length short enough for a quadratic sort.
    ///
    /// # Returns
    /// The same convention as [`suffix_array_with_sentinel`]: length `s.len()+1`, sentinel row
    /// first.
    fn naive(s: &[u8]) -> Vec<i64> {
        let n = s.len();
        // All TEXT POSITIONS, about to be sorted by the suffix each one starts.
        let mut idx: Vec<usize> = (0..n).collect();
        // Plain lexicographic comparison of whole suffixes. A prefix sorts before its extensions,
        // which is exactly what an implicit smallest sentinel would give.
        idx.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
        // Row 0 is the sentinel suffix, at position `n`.
        let mut out = vec![n as i64];
        out.extend(idx.into_iter().map(|x| x as i64));
        out
    }

    /// Assert that the reference SA-IS matches the naive oracle on `s`.
    ///
    /// # Parameters
    /// * `s`: the text to test. Panics with the input echoed on any mismatch.
    fn check(s: &[u8]) {
        assert_eq!(
            suffix_array_with_sentinel(s),
            naive(s),
            "mismatch for {s:?}"
        );
    }

    #[test]
    fn known_strings() {
        check(b"");
        check(b"a");
        check(b"aa");
        check(b"banana");
        check(b"mississippi");
        check(b"abracadabra");
        check(&[0, 0, 0, 1, 0, 1, 1]);
        check(&[3, 2, 1, 0, 3, 2, 1, 0]);
    }

    #[test]
    fn random_small_alphabet() {
        // Deterministic pseudo-random strings over {0,1,2,3} (the DNA alphabet), various lengths.
        // Fixed xorshift64 seed: any non-zero constant works, and fixing it makes a failure
        // reproducible. Each test uses a different seed to cover different strings.
        let mut state: u64 = 0x1234_5678_9abc_def0;
        // xorshift64 step; returns the next pseudo-random word. Callers take it modulo a small
        // number, so only the low bits matter.
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            // Random string length, 0..199; includes the empty-string edge case.
            let len = (next() % 200) as usize;
            let s: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            check(&s);
        }
    }

    #[test]
    fn random_byte_alphabet() {
        // Different seed from the DNA test, so this covers a different set of strings.
        let mut state: u64 = 0xdead_beef_cafe_babe;
        // Same xorshift64 generator as above.
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..100 {
            let len = (next() % 150) as usize;
            let s: Vec<u8> = (0..len).map(|_| (next() % 256) as u8).collect();
            check(&s);
        }
    }

    /// The in-place SA-IS must produce exactly the same SA as the reference implementation.
    ///
    /// Checked against [`suffix_array_with_sentinel`] rather than `naive` so the comparison stays
    /// affordable on the multi-kilobyte inputs the in-place tests use; the reference is itself
    /// pinned to `naive` by the tests above.
    ///
    /// # Parameters
    /// * `s`: the text to test, up to tens of thousands of bytes.
    fn check_inplace(s: &[u8]) {
        assert_eq!(
            suffix_array_inplace(s),
            suffix_array_with_sentinel(s),
            "in-place mismatch for {s:?}"
        );
    }

    #[test]
    fn inplace_known_strings() {
        check_inplace(b"");
        check_inplace(b"a");
        check_inplace(b"aa");
        check_inplace(b"banana");
        check_inplace(b"mississippi");
        check_inplace(b"abracadabra");
        check_inplace(&[0, 0, 0, 1, 0, 1, 1]);
        check_inplace(&[3, 2, 1, 0, 3, 2, 1, 0]);
    }

    #[test]
    fn inplace_random_dna_and_bytes() {
        // Third distinct seed, for the in-place implementation's random coverage.
        let mut state: u64 = 0x0f1e_2d3c_4b5a_6978;
        // Same xorshift64 generator as above.
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..400 {
            let len = (next() % 300) as usize;
            // Alphabet size for this string: 4 (DNA, many repeated LMS substrings, so deep
            // recursion) or 256 (mostly distinct names, so the recursion usually stops at once).
            // Alternating the two exercises both branches of step 4.
            let alpha = if next() & 1 == 0 { 4 } else { 256 };
            let s: Vec<u8> = (0..len).map(|_| (next() % alpha) as u8).collect();
            check_inplace(&s);
        }
    }

    #[test]
    fn inplace_large_and_repetitive() {
        // Fourth distinct seed, for the large/repetitive stress cases.
        let mut state: u64 = 0xa5a5_1234_9999_0f0f;
        // Same xorshift64 generator as above.
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Larger DNA strings (deep recursion) and highly repetitive inputs (small name counts).
        for &len in &[1000usize, 5000, 20000] {
            let s: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            check_inplace(&s);
        }
        check_inplace(&vec![2u8; 4096]); // all identical -> maximal recursion depth
                                         // Period-3 string "012012...": every LMS substring is one of a handful, so names collide
                                         // heavily and the recursion goes deep with a tiny alphabet at each level.
        let periodic: Vec<u8> = (0..8192).map(|i| (i % 3) as u8).collect();
        check_inplace(&periodic);
    }
}
