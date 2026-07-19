//! LISA/BWA-MEME learned-index seeding: the same SMEM collection as [`crate`]'s FM-index path, but
//! every interval is obtained from a [`LearnedSa`] (plain `[fwd][rc]` suffix array + P-RMI) instead
//! of the FM-index `backward_ext`/`get_occ`.
//!
//! **Why this is byte-identical.** bwa-mem2's SMEM driver (`smems_from_pos` etc.) branches *only* on
//! the interval size `s` (against `min_intv`/`curr_s`) and on span lengths; the reverse-complement
//! start `l` is pure internal bookkeeping and the seed output reads only `k` and `s`. And
//! [`LearnedSa::bi_interval`] is proven (`bidirectional_interval_matches_fmindex`) to return the same
//! `(k, s)` as walking `backward_ext` over the same pattern. So we mirror the driver structure exactly
//! and replace each `backward_ext` with a span lookup `interval_of(codes[m..=n])`: the control flow —
//! and therefore the emitted SMEM set and the seeds derived from it — is identical to the FM path.
//! `l` is never needed and is left at 0.
//!
//! This is the correctness-first form (each interval is a from-scratch learned-index search). The
//! forward phase can later narrow incrementally ([`LearnedSa::narrow`]); the backward phase re-searches
//! (prepending does not nest in a suffix array). Byte-identity is validated against the FM path below.
//!
//! # How a learned index replaces the FM index
//!
//! Where the FM index answers "which BWT rows match P?" by walking `backward_ext` once per base
//! (each step a data-dependent DRAM load), a learned suffix array answers it *positionally*: the
//! suffix array is a sorted list, so the rows matching P form a contiguous range, and a learned
//! model (P-RMI, a small two-stage piecewise regression) predicts approximately where that range
//! starts from the first ~32 bases of P packed into an integer key. A short local search corrects
//! the prediction. Cost is O(1) model evaluations plus a bounded scan, independent of |P|, which is
//! why extending a match by many bases at once ("jump to the LEM") is cheap here and expensive
//! there.
//!
//! **LEM** below means "longest exact match": the longest prefix of the query, starting at a given
//! pivot, that still occurs at least `min_intv` times. The FM path discovers that boundary one base
//! at a time; the learned path binary-searches or jumps to it.
//!
//! # Why the structures differ from the FM path even though the output does not
//!
//! - Only `(k, s)` is meaningful. `l` is set to 0 everywhere: it exists in the FM path solely to
//!   express forward extension as backward extension on the reverse complement, and a suffix array
//!   can extend right directly.
//! - Backward extension cannot be incremental. Appending on the right *nests* inside the current SA
//!   range (a longer prefix is a sub-range), so [`LearnedSa::narrow`] works; prepending on the left
//!   does not nest, so the backward phase re-searches the whole span via [`interval_of`]. That is
//!   the asymmetry driving the "zigzag" formulation: left extension only locates a boundary, right
//!   extension does the emitting.
//!
//! # Status
//!
//! UNVERIFIED which of two recorded performance claims about this path is current: a project note
//! records the learned index measuring far slower than the FM index at genome scale, while
//! [`crate::mem_collect_smem_hybrid`]'s doc claims round 1 is roughly 2x the FM round 1. Both cannot
//! be describing the same configuration. Do not cite either number without re-measuring; the
//! correctness argument above is independent of the performance question and does hold.

use crate::MemSeed;
use bwa_core::MemOpt;
use bwa_index::lisa::LearnedSa;
use bwa_index::Smem;

/// The `(k, s)` of the exact match `codes[m..=n]` over the learned suffix array: `k` = forward SA
/// interval start, `s` = interval size. Same values as `FmIndex::backward_ext` walked over that
/// pattern (proven). `l` is not computed (the driver never reads it).
///
/// `m` and `n` are **inclusive** read offsets, matching the `Smem` convention, hence `codes[m..=n]`.
/// Invariant: `m <= n < codes.len()`, and every base in the span must be `< 4` (no `N`); callers
/// guarantee this by stopping their walks at the first ambiguous base.
///
/// # Parameters
/// - `lsa`: the built learned suffix array over the `[fwd][rc]` concatenated reference. Supplied by
///   the caller's index; read-only, must be the same index the resulting `k` will later be resolved
///   against via [`LearnedSa::sa_at`].
/// - `codes`: the whole read, 2-bit base codes (`A=0 C=1 G=2 T=3`, `4` = `N`/ambiguous). Indexed by
///   READ offset, not reference position.
/// - `m`, `n`: inclusive READ offsets delimiting the pattern; `0 <= m <= n < codes.len()`. Supplied
///   by the SMEM driver, which guarantees no code in `codes[m..=n]` is `>= 4`.
///
/// # Returns
/// `(k, s)` where `k` is the first suffix-array ROW of the occurrence interval (an index into the
/// SA, in `0..lsa.len()`, NOT a reference position) and `s` is the number of rows in it, i.e. the
/// occurrence count. `s == 0` means the pattern does not occur. Both are `i64` to match the FM
/// path's `Smem` field types.
#[inline]
fn interval_of(lsa: &LearnedSa, codes: &[u8], m: usize, n: usize) -> (i64, i64) {
    let (lo, hi) = lsa.exact_interval(&codes[m..=n]);
    (lo as i64, (hi - lo) as i64)
}

/// One-position SMEM search starting at `x` (LISA analog of [`crate::smems_from_pos`], line-for-line
/// with `backward_ext` replaced by [`interval_of`]). Appends SMEMs to `out`, returns `next_x`.
///
/// # Parameters
/// - `lsa`: the learned suffix array (see [`interval_of`]).
/// - `codes`: the whole read as 2-bit base codes, `4` meaning `N`. Indexed by READ offset.
/// - `x`: READ offset of the pivot to search from; `0 <= x < codes.len()`. Supplied by the caller's
///   scan loop, which passes back this function's return value.
/// - `min_seed_len`: emission floor in BASES. An SMEM shorter than this is never pushed to `out`.
///   Comes from `MemOpt::min_seed_len` (bwa default 19).
/// - `min_intv`: occurrence-count floor (unitless, `>= 1`). An interval with fewer than this many
///   SA rows is treated as "no longer a match". Round 1 passes 1; round-2 reseeding passes the
///   parent SMEM's occurrence count plus one, forcing a strictly more frequent (shorter) match.
/// - `prev`: caller-owned scratch buffer holding the current frontier of candidate SMEMs (see the
///   `num_prev` invariant below). Must be at least `codes.len() + 2` long. Contents on entry are
///   irrelevant; contents on exit are garbage. Exists only to avoid a per-position allocation.
/// - `out`: SMEM sink, appended to and never cleared, so the caller accumulates across pivots.
///
/// # Returns
/// `next_x`: the READ offset the caller's scan loop should resume from. Always `> x` except in the
/// interior-break case where the forward walk stops at `j` with a too-small interval, mirroring
/// bwa-mem2's `next_x` exactly.
fn smems_from_pos_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    x: usize,
    min_seed_len: i32,
    min_intv: i64,
    prev: &mut [Smem],
    out: &mut Vec<Smem>,
) -> usize {
    // Read length in bases; the exclusive upper bound on every READ offset below.
    let readlength = codes.len();
    // Total number of suffix-array ROWS (2*l_pac + 1 including the sentinel). Used as the initial
    // "everything matches" row bracket [0, n_sa).
    let n_sa = lsa.len();
    // Default resume point for the caller: one base past the pivot. Overwritten as the forward walk
    // advances, so on every exit path it holds the offset bwa-mem2 would resume from.
    let mut next_x = x + 1;
    // The pivot base itself (READ offset x). An ambiguous base cannot start a match, so bail.
    let a = codes[x];
    if a >= 4 {
        return next_x;
    }

    // Initial single-base interval, span [x, x]. Appending a base always nests within the current
    // interval, so the whole forward extension is a sequence of `narrow` calls (each two partition
    // points over the shrinking interval) instead of a from-scratch search per step — identical
    // result (`narrow` is proven to reproduce `exact_interval` for every prefix length).
    // `[lo, hi)` is the half-open SA ROW bracket of every suffix beginning with the span currently
    // matched. Right now that span is the single base `codes[x]`, so the bracket contains every
    // occurrence of that base in the `[fwd][rc]` reference. Rows, not reference positions.
    let (mut lo, mut hi) = lsa.narrow(0, n_sa, 0, a);
    // The best SMEM candidate for the current forward span: `m`/`n` are inclusive READ offsets,
    // `k` is its SA row start, `s` its occurrence count. `l` stays 0 (the FM path's RC start, unused
    // here). Reassigned at the end of each accepted forward step.
    let mut smem = Smem {
        rid: 0,
        m: x as u32,
        n: x as u32,
        k: lo as i64,
        l: 0,
        s: (hi - lo) as i64,
    };
    // Number of live entries in `prev`. During the forward phase `prev[..num_prev]` holds one
    // candidate per DISTINCT interval size seen so far, in increasing span length; those are exactly
    // the spans at which the occurrence count dropped, i.e. the only spans that can be maximal.
    let mut num_prev = 0usize;

    // Forward extension: span [x, j], j increasing (append codes[j] at column j-x).
    // Loop invariant at the top of each iteration: `[lo, hi)` is the SA row bracket of the span
    // `codes[x..j]` (READ offsets, exclusive j), `smem` describes that same span, and its occurrence
    // count `hi - lo` is `>= min_intv`.
    let mut j = x + 1;
    while j < readlength {
        // The base being appended, at READ offset j, i.e. column `j - x` of the pattern.
        let aj = codes[j];
        next_x = j + 1;
        if aj >= 4 {
            break;
        }
        // The row bracket after appending `aj`: a sub-range of `[lo, hi)` (appending nests), so
        // `nhi - nlo <= hi - lo`. Still SA rows.
        let (nlo, nhi) = lsa.narrow(lo, hi, j - x, aj);
        // Candidate for the one-base-longer span `codes[x..=j]`.
        let new_smem = Smem {
            rid: 0,
            m: x as u32,
            n: j as u32,
            k: nlo as i64,
            l: 0,
            s: (nhi - nlo) as i64,
        };

        // Always park the shorter span at the frontier slot, but only KEEP it (advance num_prev) if
        // appending changed the occurrence count: an unchanged count means the shorter span is not
        // maximal (it is subsumed by the longer one), so the slot gets overwritten next iteration.
        prev[num_prev] = smem;
        if new_smem.s != smem.s {
            num_prev += 1;
        }
        if new_smem.s < min_intv {
            next_x = j;
            break;
        }
        smem = new_smem;
        lo = nlo;
        hi = nhi;
        j += 1;
    }
    // The longest span reached is itself a candidate if it still meets the occurrence floor.
    if smem.s >= min_intv {
        prev[num_prev] = smem;
        num_prev += 1;
    }

    // Flip to descending span length: the backward phase must try the LONGEST right-extents first,
    // which is what makes the first surviving entry the maximal one.
    prev[..num_prev].reverse();

    // Backward extension: span [jj, sm.n].
    // `jj` is the candidate LEFT boundary as a READ offset, walking leftward from x-1. Signed so the
    // "fell off the read start" test is expressible. Invariant at the top of each iteration:
    // `prev[..num_prev]` holds the surviving candidates for left boundary `jj + 1`, ordered by
    // decreasing right extent `n`, each with a distinct occurrence count `>= min_intv`.
    let mut jj = x as i64 - 1;
    while jj >= 0 {
        // The base being prepended, at READ offset jj. An `N` terminates the walk.
        let a = codes[jj as usize];
        if a > 3 {
            break;
        }
        // Candidates surviving the prepend of `codes[jj]`; written into `prev[..num_curr]`,
        // overwriting entries already consumed (safe: num_curr never outruns p).
        let mut num_curr = 0usize;
        // Occurrence count of the last candidate kept this round. Starts at -1, a value no real
        // count can take, so the first kept candidate always passes the "count changed" test. Used
        // to keep only one candidate per distinct count.
        let mut curr_s = -1i64;

        // First sub-loop: scan candidates (longest right extent first) until either an SMEM is
        // emitted or the first survivor is found. Exits via `break` in both cases.
        let mut p = 0usize;
        while p < num_prev {
            // The candidate being extended: right extent `sm.n` (inclusive READ offset), left
            // boundary currently `jj + 1`.
            let sm = prev[p];
            // Re-search from scratch: prepending does not nest in a suffix array, so unlike the
            // forward phase there is no incremental `narrow`. `k` is an SA ROW, `s` a count.
            let (k, s) = interval_of(lsa, codes, jj as usize, sm.n as usize);
            let new_smem = Smem {
                rid: 0,
                m: jj as u32,
                n: sm.n,
                k,
                l: 0,
                s,
            };
            // Extending left killed the match (count fell below the floor), so `sm` itself was
            // maximal: emit it if it is long enough. Span length in bases is n - m + 1.
            if new_smem.s < min_intv
                && (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len)
            {
                out.push(sm);
                break;
            }
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
                break;
            }
            p += 1;
        }
        p += 1;
        // Second sub-loop: continue over the remaining (shorter) right extents, keeping one
        // candidate per distinct occurrence count. No emission happens here: a shorter right extent
        // that survives is not maximal at this left boundary yet.
        while p < num_prev {
            // Same meanings as the first sub-loop: `sm` a candidate, `k` an SA row, `s` a count.
            let sm = prev[p];
            let (k, s) = interval_of(lsa, codes, jj as usize, sm.n as usize);
            let new_smem = Smem {
                rid: 0,
                m: jj as u32,
                n: sm.n,
                k,
                l: 0,
                s,
            };
            if new_smem.s >= min_intv && new_smem.s != curr_s {
                curr_s = new_smem.s;
                prev[num_curr] = new_smem;
                num_curr += 1;
            }
            p += 1;
        }
        // The survivors become the frontier for the next (further-left) boundary.
        num_prev = num_curr;
        if num_curr == 0 {
            break;
        }
        jj -= 1;
    }
    // Ran off the read start (or hit an N) with survivors still live: the longest of them, at
    // `prev[0]` after the reverse, is maximal by construction and is the last SMEM to emit.
    if num_prev != 0 {
        let sm = prev[0];
        if (i64::from(sm.n) - i64::from(sm.m) + 1) >= i64::from(min_seed_len) {
            out.push(sm);
        }
    }
    next_x
}

/// Backward LEM: longest exact match ending at `pivot` and extending left, occurring at least
/// `min_intv` times. Realised as a forward LEM of the reverse-complement pattern
/// `p[t] = 3 - codes[pivot - t]` over the `[fwd][rc]` SA (this is how the concatenated reference gives
/// left-extension "for free"). Stops at a read boundary or an `N`.
///
/// The transform: reading the read leftward from `pivot` and complementing each base
/// (`3 - c`, since `A=0 C=1 G=2 T=3`) yields the reverse complement of the left-extension. A
/// left-occurrence in the forward half of the text is a right-occurrence of that pattern in the RC
/// half, and the SA indexes both halves, so a single forward search counts both.
///
/// Returns a **length in bases**, not a position: the caller derives the left boundary as
/// `pivot + 1 - len`.
///
/// # Parameters
/// - `lsa`: the learned suffix array over the `[fwd][rc]` reference; both halves must be present or
///   the reverse-complement trick below counts only half the occurrences.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `pivot`: inclusive READ offset the match must END at; `< codes.len()`.
/// - `min_intv`: occurrence-count floor (unitless, `>= 1`), the same value the caller will use for
///   the paired forward LEM (see [`zigzag_march`]).
///
/// # Returns
/// Match length in BASES, in `0..=pivot + 1`. Zero means even the single base at `pivot` fails the
/// floor (or is an `N`).
fn backward_lem(lsa: &LearnedSa, codes: &[u8], pivot: usize, min_intv: i64) -> usize {
    // The reverse-complemented left context, built forwards: pat[t] is the complement of the base t
    // positions LEFT of the pivot. Searching this forward over the `[fwd][rc]` SA is equivalent to
    // searching leftward from the pivot in the forward reference.
    let mut pat = Vec::with_capacity(pivot + 1);
    // Distance in bases walked left of the pivot so far; also the write index into `pat`.
    let mut t = 0usize;
    loop {
        // Signed READ offset pivot - t; goes negative exactly when the read start is passed.
        let idx = pivot as isize - t as isize;
        if idx < 0 {
            break;
        }
        // Base at that READ offset; an `N` truncates the pattern (no match may span it).
        let c = codes[idx as usize];
        if c >= 4 {
            break;
        }
        pat.push(3 - c);
        t += 1;
    }
    lsa.lem_min_intv(&pat, min_intv).0
}

/// Forward LEM from `pivot`, occurring at least `min_intv` times, capped at the first `N`.
///
/// Returns `(length, lo, hi)`: the match length in bases and the half-open SA range `[lo, hi)` of
/// its occurrences, so the occurrence count is `hi - lo` (the FM path's `s`).
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `pivot`: inclusive READ offset the match must START at; `< codes.len()`.
/// - `min_intv`: occurrence-count floor (unitless, `>= 1`); the returned length is the longest one
///   still occurring at least this often.
///
/// # Returns
/// `(length_in_bases, lo, hi)` where `lo`/`hi` are half-open suffix-array ROW indices in
/// `0..=lsa.len()`, not reference positions. Length 0 means no qualifying match.
fn forward_lem(
    lsa: &LearnedSa,
    codes: &[u8],
    pivot: usize,
    min_intv: i64,
) -> (usize, usize, usize) {
    // Exclusive READ offset where the searchable stretch ends: the first `N` at or after the pivot,
    // else the read end. No match may span an ambiguous base.
    let end = codes[pivot..]
        .iter()
        .position(|&c| c >= 4)
        .map(|p| pivot + p)
        .unwrap_or(codes.len());
    lsa.lem_min_intv(&codes[pivot..end], min_intv)
}

/// Append one SMEM to `out`, converting the learned index's half-open row range into the FM path's
/// `(k, s)` start-plus-count form.
///
/// # Parameters
/// - `out`: SMEM sink, appended to, never cleared.
/// - `lo`, `hi`: half-open suffix-array ROW range of the occurrences (`lo <= hi <= lsa.len()`), as
///   returned by [`forward_lem`] or [`LearnedSa::exact_interval`]. Stored as `k = lo`, `s = hi - lo`.
/// - `m`, `n`: inclusive READ offsets of the matched span, `m <= n < read length`.
///
/// `rid` is 0 and `l` is 0: neither is meaningful during seeding (see the module header).
#[inline]
fn push_smem(out: &mut Vec<Smem>, lo: usize, hi: usize, m: usize, n: usize) {
    out.push(Smem {
        rid: 0,
        m: m as u32,
        n: n as u32,
        k: lo as i64,
        l: 0,
        s: (hi - lo) as i64,
    });
}

/// The shared zigzag body: starting at `start`, alternate backward (find left boundary, no emit) and
/// forward (emit) LEMs, marching the right boundary until it reaches `next_pivot`. `min_intv` gates
/// the LEM occurrence count (1 for round 1, parent-hitcount+1 for round-2 reseeding).
///
/// # Why "zigzag"
///
/// The FM path finds each SMEM by extending right then walking left one base at a time. Here each
/// half is a single jump instead, so the cursor traces a zigzag: extend LEFT from the cursor to find
/// where the maximal match beginning around here actually starts (`cur = cur + 1 - left_len`), then
/// extend RIGHT from that boundary to find how far it reaches, and emit that as the SMEM. The right
/// end then becomes the next cursor, so the read is covered in a small number of jumps rather than
/// one FM step per base. Both extensions must use the same `min_intv`, otherwise the left boundary
/// and the right reach would be maximal for different occurrence thresholds and the emitted span
/// would not be a real SMEM.
///
/// Parameters:
/// - `lsa`: the learned suffix array over the `[fwd][rc]` reference.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `min_intv`: occurrence-count floor (unitless, `>= 1`) applied identically to BOTH the left and
///   the right LEM, for the reason given above. 1 for round 1; parent hitcount + 1 for round 2.
/// - `start`: read offset to begin at, `< codes.len()`.
/// - `next_pivot`: exclusive right limit for the cursor. `codes.len()` for round 1 (march to the
///   read end); this pivot's forward reach for round-2 reseeding, which is what confines a reseed to
///   a single position's worth of work.
/// - `min_seed_len`: emission floor in bases, and also the early-exit floor: once fewer than this
///   many bases remain, no further SMEM can qualify, so the march stops.
/// - `out`: appended to, never cleared.
fn zigzag_march(
    lsa: &LearnedSa,
    codes: &[u8],
    start: usize,
    next_pivot: usize,
    min_intv: i64,
    min_seed_len: usize,
    out: &mut Vec<Smem>,
) {
    // Read length in bases.
    let l_seq = codes.len();
    // READ offset of the position whose SMEM is being sought this iteration; the loop's progress
    // counter against `next_pivot`. Advances to the right end of each emitted SMEM.
    let mut search_pivot = start;
    // The working cursor, also a READ offset. Two roles per iteration: on entry it is the position
    // the left extension must END at; after the left jump it is the SMEM's left boundary `m`.
    let mut cur = start;
    // Invariant at the top of each iteration: every SMEM starting at a read offset `< search_pivot`
    // that qualifies has already been emitted, and `cur == search_pivot`.
    while search_pivot < next_pivot {
        if codes[search_pivot] >= 4 {
            if l_seq - search_pivot < min_seed_len {
                break;
            }
            search_pivot += 1;
            continue;
        }
        // Left extension (no emit): longest match ending at `cur`.
        // `left_len` is a LENGTH IN BASES (>= 1 here, since codes[cur] is a real base), so the
        // implied left boundary is the READ offset `cur + 1 - left_len`.
        let left_len = backward_lem(lsa, codes, cur, min_intv);
        cur = cur + 1 - left_len;
        if next_pivot - cur < min_seed_len {
            break;
        }
        // Right extension (emit): longest match starting at the new left boundary.
        // `rlen` = match length in BASES from `cur`; `[lo, hi)` = its suffix-array ROW range, so the
        // occurrence count stored as `s` is `hi - lo`. The SMEM spans READ offsets [cur, cur+rlen-1].
        let (rlen, lo, hi) = forward_lem(lsa, codes, cur, min_intv);
        if rlen >= min_seed_len {
            push_smem(out, lo, hi, cur, cur + rlen - 1);
        }
        search_pivot = cur + rlen;
        cur = search_pivot;
    }
}

/// Round-1 SMEM collection via BWA-MEME's fast zigzag (`Learned_getSMEMsAllPosOneThread`/`_step1`,
/// mode-1 non-tradeoff): never computes shallow intervals — each step jumps to a LEM. Emits SMEMs
/// `>= min_seed_len`. bwa-mem2-concordant (validated against the FM path below).
///
/// # Parameters
/// - `lsa`: the learned suffix array over the `[fwd][rc]` reference.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `min_seed_len`: emission floor in BASES (`MemOpt::min_seed_len`, bwa default 19). Signed only
///   to match the FM path's signature; must be `>= 1`.
///
/// # Returns
/// A freshly allocated SMEM list in emission order (left to right along the read). `k` fields are
/// suffix-array ROWS, not reference positions; call [`seeds_from_smem_lsa`] to resolve them.
pub fn collect_smems_lsa_zigzag(lsa: &LearnedSa, codes: &[u8], min_seed_len: i32) -> Vec<Smem> {
    // Read length in bases.
    let l_seq = codes.len();
    let min_seed_len = min_seed_len as usize;
    let mut out = Vec::new();
    // READ offset of the position currently being seeded from.
    let mut pivot = 0usize;
    while pivot < l_seq {
        if codes[pivot] >= 4 {
            pivot = if l_seq - pivot < min_seed_len {
                l_seq
            } else {
                pivot + 1
            };
            continue;
        }
        // Does anything extensible sit to the LEFT of this pivot? Only if the pivot is not at offset
        // 0 and the preceding base is a real base. If so the zigzag applies: the match covering the
        // pivot may begin further left, so the left boundary must be found before emitting.
        if pivot != 0 && codes[pivot - 1] < 4 {
            // Middle pivot: march to the read end (all-position step1). One march covers every
            // remaining position, which is why `pivot` jumps straight to `l_seq` afterwards.
            zigzag_march(lsa, codes, pivot, l_seq, 1, min_seed_len, &mut out);
            pivot = l_seq;
        } else {
            // Read start (or preceded by N): nothing to the left, so the maximal match covering the
            // pivot must begin exactly at it, and a single forward LEM is the whole answer. No
            // backward step is possible, hence no zigzag.
            // `rlen` = match length in BASES from `pivot`; `[lo, hi)` = its suffix-array ROW range.
            let (rlen, lo, hi) = forward_lem(lsa, codes, pivot, 1);
            if rlen >= min_seed_len {
                push_smem(&mut out, lo, hi, pivot, pivot + rlen - 1);
            }
            // `.max(1)` guarantees forward progress when the LEM is empty.
            pivot += rlen.max(1);
        }
    }
    out
}

/// Single-position SMEM search (`Learned_getSMEMsOnePosOneThread`), for round-2 reseeding from a
/// pivot with a given `min_intv`. Like step1 but `next_pivot` is bounded by this pivot's forward
/// reach (an initial forward LEM) rather than the read end.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `pivot`: READ offset to reseed from, `< codes.len()`. Round 2 passes the midpoint of the
///   parent SMEM.
/// - `min_intv`: occurrence-count floor (unitless). Round 2 passes `parent.s + 1`, forcing a
///   strictly more frequent, hence strictly shorter, match than the SMEM being reseeded.
/// - `min_seed_len`: emission floor in BASES.
/// - `out`: SMEM sink, appended to. Round 2 passes the very vector it is iterating a prefix of, so
///   this function must only ever push.
fn smems_one_pos(
    lsa: &LearnedSa,
    codes: &[u8],
    pivot: usize,
    min_intv: i64,
    min_seed_len: usize,
    out: &mut Vec<Smem>,
) {
    if codes[pivot] >= 4 {
        return;
    }
    if pivot != 0 && codes[pivot - 1] < 4 {
        // `freach` = how far right, in BASES, a match anchored at `pivot` reaches. Its row range is
        // discarded: this call only measures the reach.
        let (freach, _, _) = forward_lem(lsa, codes, pivot, min_intv);
        // Exclusive READ offset limiting the march, so the reseed stays within this pivot's own
        // reach instead of running to the read end as round 1 does.
        let next_pivot = pivot + freach;
        zigzag_march(lsa, codes, pivot, next_pivot, min_intv, min_seed_len, out);
    } else {
        // Nothing extensible to the left, so the maximal match starts exactly at the pivot: one
        // forward LEM is the whole answer. `rlen` in BASES, `[lo, hi)` in suffix-array ROWS.
        let (rlen, lo, hi) = forward_lem(lsa, codes, pivot, min_intv);
        if rlen >= min_seed_len {
            push_smem(out, lo, hi, pivot, pivot + rlen - 1);
        }
    }
}

/// Round-2 reseeding (fast): re-seed each long, non-repetitive round-1 SMEM from its midpoint with
/// `min_intv = hitcount + 1`, appending in place. Mirrors `smem_round_2`.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes. Indexed by READ offset.
/// - `opt`: alignment options; only `min_seed_len`, `split_factor` and `split_width` are read.
/// - `smems`: round-1 SMEMs in, round-1 plus round-2 SMEMs out. Grown in place; the round-2 SMEMs
///   appended during the loop are deliberately NOT themselves reseeded (see `num1`).
fn smem_round_2_lsa_fast(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    // Minimum SMEM length in BASES that qualifies for reseeding: min_seed_len * split_factor
    // (bwa default 1.5), the +0.499 reproducing bwa-mem2's rounding of the f32 product exactly.
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    // Number of round-1 SMEMs, frozen before the loop so newly appended round-2 SMEMs are not
    // reseeded in turn (that would recurse and diverge from bwa-mem2).
    let num1 = smems.len();
    // min_seed_len as usize, for the usize-typed callee.
    let msl = opt.min_seed_len as usize;
    for idx in 0..num1 {
        // Copy, not a borrow: `smems` is handed to the callee mutably below.
        let p = smems[idx];
        // Inclusive/exclusive READ offsets of the parent SMEM, so `end - start` is its length in
        // bases. Signed to match bwa-mem2's int arithmetic.
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        // Skip SMEMs that are too short to be worth splitting, or too repetitive (occurrence count
        // above `split_width`, bwa default 10) for a reseed to add anything.
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        // Midpoint READ offset of the parent SMEM: the reseed pivot.
        let x = ((end + start) >> 1) as usize;
        smems_one_pos(lsa, codes, x, p.s + 1, msl, smems);
    }
}

/// Round-3 forward-only seeding (fast): for each pivot emit the shortest forward match that first
/// drops below `max_intv` occurrences (and is `>= min_seed_len`). Mirrors `bwt_seed_strategy`, but
/// jumps to the LEM and binary-searches the length where the occurrence count crosses `max_intv`
/// instead of narrowing base-by-base.
///
/// The binary search is valid because occurrence count is **monotonically non-increasing** in match
/// length: a longer pattern cannot occur more often. So "smallest L with `occ(L) < max_intv`" is a
/// well-defined predicate flip, searched over `[min_seed_len, lem_len]`. The `min_seed_len` case is
/// tested separately first because the loop below starts at `min_seed_len + 1`.
///
/// See the UNVERIFIED note on the no-emit branch inside: this function is concordant with the FM
/// path, not proven identical to it.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `max_intv`: occurrence-count CEILING (unitless). A seed is emitted at the shortest length whose
///   occurrence count is strictly below this. From `MemOpt::max_mem_intv` (bwa default 20); callers
///   only invoke round 3 when it is `> 0`.
/// - `min_seed_len`: length floor in BASES. Callers pass `opt.min_seed_len + 1`, matching bwa's
///   `bwt_seed_strategy` call.
/// - `out`: SMEM sink, appended to (round-3 seeds are added after rounds 1 and 2).
fn bwt_seed_strategy_lsa_fast(
    lsa: &LearnedSa,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    // Read length in bases.
    let l_seq = codes.len();
    let min_seed_len = min_seed_len as usize;
    // Occurrence oracle: given a start READ offset `x` and a length `l` in BASES, returns
    // (count, lo, hi) where `[lo, hi)` are suffix-array ROWS and count = hi - lo. This is the
    // monotone predicate source the binary search below probes; each call is a full from-scratch
    // learned-index lookup, so the search costs O(log lem_len) of them.
    let occ = |x: usize, l: usize| -> (i64, usize, usize) {
        let (a, b) = lsa.exact_interval(&codes[x..x + l]);
        ((b - a) as i64, a, b)
    };
    // READ offset of the pivot currently being tried.
    let mut x = 0usize;
    while x < l_seq {
        // Where the scan resumes; overwritten by whichever branch below runs.
        let mut next_x = x + 1;
        if codes[x] < 4 {
            // Longest match anchored at `x` with no occurrence floor: the upper end of the length
            // range the binary search works over. In BASES; its row range is not needed.
            let (lem_len, _, _) = forward_lem(lsa, codes, x, 1);
            if lem_len == 0 {
                x = next_x;
                continue;
            }
            // Occurrence count at the full LEM length: the minimum achievable count here. If even
            // this does not fall below `max_intv`, no length qualifies and the no-emit branch runs.
            let (occ_lem, _, _) = occ(x, lem_len);
            if lem_len >= min_seed_len && occ_lem < max_intv {
                // Smallest L in [min_seed_len, lem_len] with occ(L) < max_intv (occ decreasing).
                // `l_star` is that length, in BASES: the shortest qualifying seed at this pivot.
                let l_star = if occ(x, min_seed_len).0 < max_intv {
                    min_seed_len
                } else {
                    // Binary search over LENGTHS (not rows, not offsets). Invariant at the top of
                    // each iteration: the answer lies in [lo, hi], every length `< lo` has
                    // occ >= max_intv, and `hi` is known to satisfy occ < max_intv.
                    let (mut lo, mut hi) = (min_seed_len + 1, lem_len);
                    while lo < hi {
                        // Candidate length under test.
                        let mid = (lo + hi) / 2;
                        if occ(x, mid).0 < max_intv {
                            hi = mid;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    lo
                };
                // Final lookup at the chosen length: `s` is the occurrence count that becomes the
                // SMEM's `s`, `[lo, hi)` the suffix-array ROW range that becomes `k` and `s`.
                let (s, lo, hi) = occ(x, l_star);
                if s > 0 {
                    push_smem(out, lo, hi, x, x + l_star - 1);
                }
                // Resume just past the emitted seed.
                next_x = x + l_star;
            } else {
                // No emit: advance past the explored match (matches FM's next_x after the forward
                // loop stops — approximate for the no-emit branch; validated against the FM path).
                //
                // UNVERIFIED: this branch is an approximation of the FM path's `next_x`, not a
                // derivation from it, and the comment above says so. The FM `bwt_seed_strategy` sets
                // `next_x = j + 1` at whatever position `j` its forward loop stopped at, which is
                // not in general `x + lem_len + 1` nor `x + min_seed_len`. It survives because
                // round 3 seeds are a superset-ish safety net and the seed-set test below only
                // demands high concordance (>= 98% exact, Jaccard >= 0.9), not equality. Do not
                // treat this function as byte-identical to the FM path.
                next_x = (x + lem_len + 1).min(l_seq);
                if lem_len < min_seed_len {
                    next_x = (x + min_seed_len).min(l_seq);
                }
            }
        }
        x = next_x;
    }
}

/// Rounds 1+2 only (no round-3 `bwt_seed_strategy`), for isolating round costs in benchmarks.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`.
/// - `opt`: alignment options; `min_seed_len`, `split_factor` and `split_width` are read.
///
/// # Returns
/// The round-1 SMEMs followed by the round-2 reseeds, `k` fields being suffix-array ROWS.
pub fn mem_collect_smem_lsa_12(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = collect_smems_lsa_zigzag(lsa, codes, opt.min_seed_len);
    smem_round_2_lsa_fast(lsa, codes, opt, &mut smems);
    smems
}

/// Full fast SMEM collection: round-1 zigzag + round-2 reseed + round-3 strategy, the LISA analog of
/// [`crate::mem_collect_smem`]. Concordant seed set (validated against the FM path on real reads).
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`.
/// - `opt`: alignment options; adds `max_mem_intv` (round 3 runs only when it is `> 0`) to those
///   read by [`mem_collect_smem_lsa_12`].
///
/// # Returns
/// All three rounds' SMEMs concatenated in round order, `k` fields being suffix-array ROWS.
pub fn mem_collect_smem_lsa_fast(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = mem_collect_smem_lsa_12(lsa, codes, opt);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_lsa_fast(
            lsa,
            codes,
            opt.max_mem_intv,
            opt.min_seed_len + 1,
            &mut smems,
        );
    }
    smems
}

/// Collect all round-1 SMEMs of `codes` via the learned index (LISA analog of [`crate::collect_smems`]).
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `min_seed_len`: emission floor in BASES (`MemOpt::min_seed_len`).
/// - `min_intv`: occurrence-count floor (unitless); callers pass 1 for round 1.
///
/// # Returns
/// The SMEM list in emission order. This is the byte-identical (slow) path, unlike
/// [`collect_smems_lsa_zigzag`].
pub fn collect_smems_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    min_seed_len: i32,
    min_intv: i64,
) -> Vec<Smem> {
    let mut out = Vec::new();
    // Frontier scratch reused across every pivot (see `prev` in `smems_from_pos_lsa`). Sized
    // `codes.len() + 2`: at most one entry per read offset can have a distinct interval size, plus
    // slack for the final push.
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    // READ offset of the pivot; advanced by whatever `smems_from_pos_lsa` reports as `next_x`.
    let mut x = 0usize;
    while x < codes.len() {
        x = smems_from_pos_lsa(
            lsa,
            codes,
            x,
            min_seed_len,
            min_intv,
            &mut scratch,
            &mut out,
        );
    }
    out
}

/// Round-3 forward-only seeding (LISA analog of `bwt_seed_strategy`): emit a seed when the interval
/// first drops below `max_intv` and the seed is at least `min_seed_len` long.
///
/// Unlike [`bwt_seed_strategy_lsa_fast`] this narrows one base at a time, so its `next_x` is derived
/// rather than approximated and the result is byte-identical to the FM path.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`. Indexed by READ offset.
/// - `max_intv`: occurrence-count CEILING (unitless), from `MemOpt::max_mem_intv`.
/// - `min_seed_len`: length floor in BASES; callers pass `opt.min_seed_len + 1`.
/// - `out`: SMEM sink, appended to.
fn bwt_seed_strategy_lsa(
    lsa: &LearnedSa,
    codes: &[u8],
    max_intv: i64,
    min_seed_len: i32,
    out: &mut Vec<Smem>,
) {
    // Read length in bases.
    let readlength = codes.len();
    // Total suffix-array ROW count, the initial full bracket.
    let n_sa = lsa.len();
    // READ offset of the pivot being tried.
    let mut x = 0usize;
    while x < readlength {
        // Where the scan resumes; kept in step with bwa-mem2's `next_x` as the walk advances.
        let mut next_x = x + 1;
        if codes[x] < 4 {
            // Forward-only: fully incremental narrowing (append codes[j] at column j-x).
            // `[lo, hi)` is the SA ROW bracket of the span matched so far, initially just codes[x].
            let (mut lo, mut hi) = lsa.narrow(0, n_sa, 0, codes[x]);
            // Invariant at the top of each iteration: `[lo, hi)` brackets the rows matching the READ
            // span `codes[x..j]`, and that span's occurrence count is still `>= max_intv` or shorter
            // than `min_seed_len` (otherwise the loop would have emitted and broken).
            let mut j = x + 1;
            while j < readlength {
                next_x = j + 1;
                // Base being appended, at READ offset j (pattern column j - x).
                let aj = codes[j];
                if aj >= 4 {
                    break;
                }
                // Row bracket after appending: a sub-range of [lo, hi).
                let (nlo, nhi) = lsa.narrow(lo, hi, j - x, aj);
                // Occurrence count of the one-base-longer span, non-increasing in span length.
                let s = (nhi - nlo) as i64;
                let smem = Smem {
                    rid: 0,
                    m: x as u32,
                    n: j as u32,
                    k: nlo as i64,
                    l: 0,
                    s,
                };
                // First length at which the count drops below the ceiling AND the span is long
                // enough: emit and stop extending this pivot.
                if smem.s < max_intv
                    && (i64::from(smem.n) - i64::from(smem.m) + 1) >= i64::from(min_seed_len)
                {
                    if smem.s > 0 {
                        out.push(smem);
                    }
                    break;
                }
                lo = nlo;
                hi = nhi;
                j += 1;
            }
        }
        x = next_x;
    }
}

/// Round 2: re-seed each long, non-repetitive round-1 SMEM from its midpoint (LISA analog of
/// `smem_round_2`). Byte-identical counterpart of [`smem_round_2_lsa_fast`].
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`.
/// - `opt`: alignment options; `min_seed_len`, `split_factor` and `split_width` are read.
/// - `smems`: round-1 SMEMs in, round-1 plus round-2 SMEMs out, grown in place.
fn smem_round_2_lsa(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt, smems: &mut Vec<Smem>) {
    // Minimum parent SMEM length in BASES to qualify for reseeding (min_seed_len * split_factor,
    // +0.499 reproducing bwa-mem2's f32 rounding).
    let split_len = (opt.min_seed_len as f32 * opt.split_factor + 0.499) as i32;
    // Round-1 count, frozen so round-2 output is not itself reseeded.
    let num1 = smems.len();
    // Frontier scratch shared by every reseed call, allocated once.
    let mut scratch: Vec<Smem> = vec![Smem::default(); codes.len() + 2];
    for idx in 0..num1 {
        // Copy of the parent SMEM (`smems` is passed mutably below).
        let p = smems[idx];
        // Inclusive/exclusive READ offsets of the parent SMEM; `end - start` is its length in bases.
        let start = p.m as i32;
        let end = p.n as i32 + 1;
        // Too short to split, or too repetitive for a reseed to help.
        if end - start < split_len || p.s > i64::from(opt.split_width) {
            continue;
        }
        // Midpoint READ offset of the parent SMEM: the reseed pivot.
        let x = ((end + start) >> 1) as usize;
        smems_from_pos_lsa(
            lsa,
            codes,
            x,
            opt.min_seed_len,
            p.s + 1,
            &mut scratch,
            smems,
        );
    }
}

/// Collect SMEMs across bwa-mem2's three rounds via the learned index (LISA analog of
/// [`crate::mem_collect_smem`]). Byte-identical SMEM set to the FM path.
///
/// # Parameters
/// - `lsa`: the learned suffix array.
/// - `codes`: the whole read, 2-bit base codes with `4` = `N`.
/// - `opt`: alignment options; `min_seed_len`, `split_factor`, `split_width` and `max_mem_intv`
///   are read. Round 3 runs only when `max_mem_intv > 0`.
///
/// # Returns
/// All three rounds' SMEMs in round order, `k` fields being suffix-array ROWS.
pub fn mem_collect_smem_lsa(lsa: &LearnedSa, codes: &[u8], opt: &MemOpt) -> Vec<Smem> {
    let mut smems = collect_smems_lsa(lsa, codes, opt.min_seed_len, 1);
    smem_round_2_lsa(lsa, codes, opt, &mut smems);
    if opt.max_mem_intv > 0 {
        bwt_seed_strategy_lsa(
            lsa,
            codes,
            opt.max_mem_intv,
            opt.min_seed_len + 1,
            &mut smems,
        );
    }
    smems
}

/// Turn one SMEM into reference-coordinate seeds using the learned suffix array (LISA analog of
/// [`crate::seeds_from_smem`]). `lsa.sa()[j]` equals `fm.get_sa(j)`, so the seeds are byte-identical.
///
/// This is the one place where suffix-array ROWS become reference POSITIONS: `sa_at(row)` returns a
/// position in the 2L space `0..2*l_pac`, where a value `>= l_pac` denotes the reverse strand.
/// Downstream code (not here) folds those back to forward-strand coordinates.
///
/// # Parameters
/// - `lsa`: the learned suffix array; must be the same index the SMEM's `k` was produced against.
/// - `smem`: one SMEM. Only `m`, `n` (inclusive READ offsets), `k` (first SA row) and `s`
///   (occurrence count) are read.
/// - `max_occ`: cap on how many seeds one SMEM may yield (`MemOpt::max_occ`, bwa default 500).
///   Must be `> 0`; a repetitive SMEM is sub-sampled rather than truncated.
///
/// # Returns
/// Up to `max_occ` seeds, each with `rbeg` a 2L-space reference POSITION, `qbeg` the READ offset
/// `smem.m`, and `len`/`score` both the span length in bases.
pub fn seeds_from_smem_lsa(lsa: &LearnedSa, smem: &Smem, max_occ: i32) -> Vec<MemSeed> {
    // Span length in BASES (inclusive offsets, hence the +1); used as both length and initial score.
    let len = (i64::from(smem.n) - i64::from(smem.m) + 1) as i32;
    let max_occ = i64::from(max_occ);
    // Stride in SA ROWS between sampled occurrences. For a repetitive SMEM this spreads the `max_occ`
    // samples across the whole interval instead of taking the first `max_occ` rows; 1 otherwise.
    let step = if smem.s > max_occ {
        smem.s / max_occ
    } else {
        1
    };
    let mut seeds = Vec::new();
    // Number of seeds emitted so far, capped at max_occ.
    let mut c = 0i64;
    // Current suffix-array ROW being resolved to a reference position. Starts at the interval start.
    let mut j = smem.k;
    while j < smem.k + smem.s && c < max_occ {
        seeds.push(MemSeed {
            rbeg: lsa.sa_at(j as usize),
            qbeg: smem.m as i32,
            len,
            score: len,
        });
        j += step;
        c += 1;
    }
    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use bwa_index::FmIndex;
    use std::path::Path;

    /// Deterministic pseudo-random generator for the tests (Knuth's MMIX linear congruential
    /// constants). Deterministic on purpose: a failing read must be reproducible from the hard-coded
    /// starting seed alone.
    ///
    /// # Parameters
    /// - `seed`: the generator state, advanced in place. Any starting value works.
    ///
    /// # Returns
    /// The top 31 bits of the new state (`>> 33`), which are the well-mixed ones; the low bits of an
    /// LCG have short periods and must not be used.
    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// The fast zigzag round-1 SMEM set must match the FM round-1 SMEM set (`collect_smems`,
    /// min_intv=1) as a set of `(m, n, k, s)` on real reads. BWA-MEME reproduces bwa-mem2 seeds, so
    /// this should be equal (or at worst concordant).
    #[test]
    fn zigzag_smems_match_fmindex_round1() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        // 4096 = number of P-RMI second-stage leaves. A model-capacity knob only: a different value
        // changes prediction accuracy (hence speed), never the returned intervals.
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let msl = MemOpt::default().min_seed_len;
        // Length of the forward half of the 2L reference, in bases. Reads are drawn only from
        // [0, l_pac) so they are forward-strand substrings.
        let l_pac = fm.l_pac() as usize;

        // Fixed starting state: the 500 synthetic reads below are identical on every run.
        let mut seed = 0xa11ce_5eed_1234u64;
        // Count of reads whose FM and LISA SMEM sets disagreed; must end at 0.
        let mut mism = 0usize;
        for _ in 0..500 {
            // Synthetic read length in bases, uniform over 40..=159.
            let rlen = 40 + (lcg(&mut seed) as usize % 120);
            // Forward-strand reference POSITION the read is copied from, in [0, l_pac - rlen).
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            // The read itself: a real reference substring, so it has genuine SMEMs.
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            // 0..=3 random substitutions, which split the read into several SMEMs.
            for _ in 0..(lcg(&mut seed) as usize % 4) {
                // READ offset of the base to mutate.
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }

            // Both sides reduced to comparable tuples (m, n, k, s): inclusive READ offsets plus the
            // SA row start and occurrence count. `l` and `rid` are excluded (see the module header).
            let mut fm_set: Vec<(u32, u32, i64, i64)> = crate::collect_smems(&fm, &codes, msl, 1)
                .iter()
                .map(|s| (s.m, s.n, s.k, s.s))
                .collect();
            let mut lsa_set: Vec<(u32, u32, i64, i64)> =
                collect_smems_lsa_zigzag(&lsa, &codes, msl)
                    .iter()
                    .map(|s| (s.m, s.n, s.k, s.s))
                    .collect();
            fm_set.sort_unstable();
            fm_set.dedup();
            lsa_set.sort_unstable();
            lsa_set.dedup();
            if fm_set != lsa_set {
                mism += 1;
                if mism <= 3 {
                    eprintln!("read@{start} len {rlen}\n  FM  : {fm_set:?}\n  LISA: {lsa_set:?}");
                }
            }
        }
        assert_eq!(mism, 0, "{mism}/500 reads had differing round-1 SMEM sets");
    }

    /// The full fast path (rounds 1+2+3) must produce a concordant SEED set to the FM
    /// `mem_collect_smem`. Compares the derived seeds (rbeg, qbeg, len) as a set, since the fast path
    /// emits in a different order and may differ in benign ways (duplicate/contained seeds that
    /// chaining discards). Reports the Jaccard overlap; requires exact-or-near-exact match.
    #[test]
    fn fast_full_seedset_concordant_with_fmindex() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let opt = MemOpt::default();
        let l_pac = fm.l_pac() as usize;

        // Reduce an SMEM list to a sorted, deduplicated set of (rbeg, qbeg, len) triples: a 2L-space
        // reference POSITION, a READ offset, and a length in bases. Sorting removes the emission
        // order difference between the two paths; dedup removes benign duplicate seeds.
        let seeds_set = |smems: &[Smem], seeds_of: &dyn Fn(&Smem) -> Vec<MemSeed>| {
            let mut v: Vec<(i64, i32, i32)> = smems
                .iter()
                .flat_map(|s| seeds_of(s).into_iter().map(|d| (d.rbeg, d.qbeg, d.len)))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };

        let mut seed = 0xf00d_1234_5678u64;
        // `total` = reads compared, `exact` = reads whose seed sets were identical, `worst_jac` =
        // smallest Jaccard overlap seen among the non-identical ones (starts at the perfect 1.0 and
        // only ever decreases). The assertion at the end thresholds exact/total and worst_jac.
        let (mut total, mut exact, mut worst_jac) = (0usize, 0usize, 1.0f64);
        for _ in 0..500 {
            // Synthetic read: length in bases (60..=159), forward-strand reference POSITION it is
            // copied from, then the copied bases indexed by READ offset. Longer than the round-1
            // test's reads so rounds 2 and 3 have something to reseed.
            let rlen = 60 + (lcg(&mut seed) as usize % 100);
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            for _ in 0..(lcg(&mut seed) as usize % 4) {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }

            let fm_smems = crate::mem_collect_smem(&fm, &codes, &opt);
            let lsa_smems = mem_collect_smem_lsa_fast(&lsa, &codes, &opt);
            let fm_seeds = seeds_set(&fm_smems, &|s| crate::seeds_from_smem(&fm, s, opt.max_occ));
            let lsa_seeds = seeds_set(&lsa_smems, &|s| seeds_from_smem_lsa(&lsa, s, opt.max_occ));

            total += 1;
            if fm_seeds == lsa_seeds {
                exact += 1;
            } else {
                // Set intersection size (binary search is valid: `seeds_set` returned sorted vecs).
                let inter = fm_seeds
                    .iter()
                    .filter(|x| lsa_seeds.binary_search(x).is_ok())
                    .count();
                // Union size by inclusion-exclusion, and the Jaccard index in [0, 1]. Two empty
                // seed sets count as perfect agreement rather than 0/0.
                let uni = fm_seeds.len() + lsa_seeds.len() - inter;
                let jac = if uni == 0 {
                    1.0
                } else {
                    inter as f64 / uni as f64
                };
                if jac < worst_jac {
                    worst_jac = jac;
                }
            }
        }
        eprintln!("fast full: {exact}/{total} exact, worst Jaccard {worst_jac:.3}");
        assert!(
            exact as f64 / total as f64 >= 0.98 && worst_jac >= 0.9,
            "seed concordance too low: {exact}/{total} exact, worst Jaccard {worst_jac:.3}"
        );
    }

    /// The LISA SMEM set must byte-match the FM path on real reads over the tiny reference, at both
    /// the SMEM level (m, n, k, s) and the derived seed level (rbeg, qbeg, len).
    #[test]
    fn lisa_seeding_matches_fmindex() {
        let prefix = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/tiny/tiny.fa");
        let fm = FmIndex::load(Path::new(prefix)).unwrap();
        let reference = fm.reference().to_vec();
        let lsa = LearnedSa::build(reference.clone(), 4096);
        let opt = MemOpt::default();
        let l_pac = fm.l_pac() as usize;

        let mut seed = 0x51_5a_51_5a_1234_5678u64;
        for _ in 0..400 {
            // A read = a real substring of the forward reference (so it has genuine SMEMs), with a
            // few random mismatches sprinkled in to create multiple SMEMs.
            let rlen = 40 + (lcg(&mut seed) as usize % 120);
            let start = lcg(&mut seed) as usize % (l_pac - rlen);
            let mut codes: Vec<u8> = reference[start..start + rlen].to_vec();
            // Number of substitutions to apply, 0..=3.
            let n_mm = lcg(&mut seed) as usize % 4;
            for _ in 0..n_mm {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = (lcg(&mut seed) % 4) as u8;
            }
            // Occasional N.
            if lcg(&mut seed) % 5 == 0 {
                let p = lcg(&mut seed) as usize % rlen;
                codes[p] = 4;
            }

            let fm_smems = crate::mem_collect_smem(&fm, &codes, &opt);
            let lsa_smems = mem_collect_smem_lsa(&lsa, &codes, &opt);

            // Compare SMEM sets on (m, n, k, s) — l is internal, rid is 0 during seeding.
            // Order-preserving (no sort): this path claims byte-identity, not mere concordance.
            let key = |v: &[Smem]| -> Vec<(u32, u32, i64, i64)> {
                v.iter().map(|s| (s.m, s.n, s.k, s.s)).collect()
            };
            assert_eq!(
                key(&fm_smems),
                key(&lsa_smems),
                "SMEM mismatch: read start {start} len {rlen}"
            );

            // Compare the derived seeds too.
            for (a, b) in fm_smems.iter().zip(lsa_smems.iter()) {
                let sa_seeds = crate::seeds_from_smem(&fm, a, opt.max_occ);
                let lsa_seeds = seeds_from_smem_lsa(&lsa, b, opt.max_occ);
                assert_eq!(sa_seeds, lsa_seeds, "seed mismatch: read start {start}");
            }
        }
    }
}
