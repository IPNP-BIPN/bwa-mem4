//! Scalar banded Smith-Waterman seed extension, a faithful port of bwa's `ksw_extend2`
//! (`reference/bwa-mem2/src/ksw.cpp`): local extension from an initial score `h0` with affine
//! gaps, a band `w`, and z-drop early termination. This is the bit-identity source of truth for
//! seed extension; SIMD/GPU backends must reproduce its integer results.
//!
//! # Where this sits, and the order to read it in
//!
//! The aligner finds short exact matches ("seeds") in the reference, then *extends* each seed
//! outwards with the banded Smith-Waterman below to turn it into a full alignment. This file is the
//! scalar, portable, authoritative version of that step. `bwa-neon` reimplements the
//! same arithmetic on SIMD lanes and on the GPU, and are accepted only if they return the identical
//! integers.
//!
//! Read the functions in this order:
//!
//! 1. [`ksw_extend2`], the seed extension itself (ports `ksw_extend2`, `ksw.cpp:432-533`). Everything
//!    else in the crate exists to serve or to mirror it.
//! 2. [`ksw_global2`], end-to-end alignment with traceback, which turns a settled region into an
//!    actual CIGAR string (ports `ksw_global2`, `ksw.cpp:558-668`). [`push_cigar`] is its helper.
//! 3. [`ksw_local_fwd`], one forward local-SW scoring pass used by *mate rescue*, a different
//!    problem with a deliberately different convention (ports `ksw_u8`, `ksw.cpp:111`).
//! 4. [`ksw_align2`], which is [`ksw_local_fwd`] run twice (forwards, then on the reversed prefixes)
//!    to recover where the alignment started (ports `ksw_align2`, `ksw.cpp:370`).
//!
//! # Rust mechanics used in this file
//!
//! This file is deliberately written in the least idiomatic Rust in the tree, and that is the
//! point: it is a line-by-line port of C, and diffing it against `ksw.cpp` is the workflow that
//! finds parity bugs. So it uses indexed loops rather than iterators, single-letter names rather
//! than descriptive ones, and explicit `mut` counters rather than functional pipelines. Idiomatic
//! Rust here would be less readable, not more, because it would no longer line up with the C.
//!
//! The one thing that genuinely matters is INTEGER BEHAVIOUR. Byte-identity means reproducing the
//! C's arithmetic exactly, including where it truncates and where it can overflow.
//!
//! | Construct | What it means |
//! |-----------|---------------|
//! | `i32`, `i8` | signed 32- and 8-bit integers. The widths are copied from the C, not chosen. A wider type would silently avoid an overflow the C has, and produce a different score. |
//! | `as i32` on a float | truncation TOWARD ZERO, discarding the fraction. This matches C's `(int)` cast exactly, which is why the band-width computation can be transcribed directly. |
//! | `as usize` | conversion to the index type. Every one of these marks a place where a signed C index becomes a Rust index; they are frequent here because the C uses `int` for both arithmetic and indexing, and Rust does not. |
//! | `vec![0i32; n]` | allocates a vector of `n` zeroed values. The suffix on the literal fixes the element type. |
//! | `let mut x = ...` | a variable that will be reassigned. Rust bindings are read-only by default, so every `mut` in the DP loops marks genuine mutable state, which makes the loop-carried variables visible at a glance. |
//! | `x.max(0)` / `x.min(y)` | the larger or smaller of two values, as a method rather than a macro. `x.max(0)` is the local-alignment floor and appears wherever the C writes a conditional zeroing. |
//! | `.iter().copied().max()` | walks a slice, yields values rather than references, and returns the largest as an `Option`. Used once, to find the best entry of the scoring matrix. |
//! | `.unwrap_or(0)` | supplies a fallback for that `Option`, covering an empty matrix. |
//! | `mat[..n]` | a slice of the first `n` entries, borrowed with no copy. |
//! | `-1i32` | a typed literal. Several counters use -1 as "not set yet", exactly as the C does, so the type must be signed. |
//! | `for i in 0..n` | counts 0 to n-1. The upper bound is excluded. |
//! | `if a > b { a } else { b }` written out | kept in preference to `.max()` in the innermost DP cells, because it mirrors the C's expression shape line for line. |
//!
//! # Glossary: the short names kept from the C, in plain language
//!
//! These names are **deliberately not renamed**. Checking this file line by line against
//! `reference/bwa-mem2/src/ksw.cpp` is the routine that has found every parity bug so far, and that
//! only works while the identifiers line up. What each one means:
//!
//! | name | plain language |
//! |---|---|
//! | `i` | index of the current **target** (reference) base: the DP row |
//! | `j` | index of the current **query** (read) base: the DP column |
//! | `h` / `H` | best score of any alignment ending exactly at this cell |
//! | `h1` | the `H` of the cell immediately to the left, carried into the next column |
//! | `h0` | the score the seed already earned, i.e. what the extension starts from |
//! | `m` / `M` | best score ending at this cell *on the diagonal* (an aligned pair, CIGAR `M`) |
//! | `e` / `E` | best score ending here with a **deletion** open (gap in the query, CIGAR `D`) |
//! | `f` / `F` | best score ending here with an **insertion** open (gap in the target, CIGAR `I`) |
//! | `mj` | the column where the current row's best `H` was found |
//! | `m11` | bwa-mem2's vector-kernel name for `M`, quoted in comments below |
//! | `oe_del` / `oe_ins` | cost of the *first* base of a gap: open + one extend (`o_* + e_*`) |
//! | `beg` / `end` | half-open range of columns still worth evaluating in this row |
//! | `qle` / `tle` | query / target bases consumed by the best **local** alignment |
//! | `gtle` / `gscore` | target bases consumed by, and score of, the best **query-exhausting** one |
//! | `max_off` | furthest the best cell of any row strayed from the main diagonal |
//! | `zdrop` | how far the score may fall below its best before the DP gives up |
//! | `qp` | "query profile", `mat` re-laid-out so the inner loop does one contiguous load |
//!
//! Note that `m` is overloaded, as it is in the C: as a *parameter* it is the alphabet size (always
//! 5), while inside the DP loops `M` is the diagonal score. This file spells the latter `big_m` to
//! keep them apart.
//!
//! # The recurrence, once, for the whole file
//!
//! Every kernel here is affine-gap dynamic programming over a matrix indexed by target row `i` and
//! query column `j`. Three surfaces are carried:
//!
//! * `M(i,j) = H(i-1,j-1) + s(t[i], q[j])`, the "arrive on the diagonal" score, i.e. this cell is
//!   an aligned pair (a CIGAR `M`, match or mismatch).
//! * `E(i,j)`, the best score arriving with a **deletion** open in the query (consuming target,
//!   CIGAR `D`). Extending it costs `e_del` per base; opening a fresh one costs `o_del + e_del`.
//! * `F(i,j)`, the best score arriving with an **insertion** open (consuming query, CIGAR `I`),
//!   costing `e_ins` to extend and `o_ins + e_ins` to open.
//! * `H(i,j) = max(M, E, F)` (plus `max(.., 0)` in the local kernels), the cell's score.
//!
//! Two conventions in this file are worth memorising because they are where ports go wrong:
//!
//! 1. **Gaps open from M, never from H.** In [`ksw_extend2`] and in bwa-mem2's vectorized
//!    `bandedSWA` (`MAIN_CODE16`, `bandedSWA.cpp:327-342`, which subtracts `oe_ins256`/`oe_del256`
//!    from `m11`, not `h11`) the next row's `E`/`F` floor is `M - oe_*`. Opening from
//!    `H = max(M,E,F)` would permit an insertion immediately followed by a deletion, that is a
//!    CIGAR like `100M3I3D20M`; the C says so at `ksw.cpp:487` ("separating H and M to disallow a
//!    cigar like 100M3I3D20M") and again at `ksw.cpp:604`.
//! 2. **The mate-rescue kernel is the deliberate exception.** [`ksw_local_fwd`] ports `ksw_u8`
//!    (`ksw.cpp:111`), whose lazy-F striped SIMD formulation subtracts from `h`, not `m`:
//!    `ksw.cpp:168` is `t = _mm_subs_epu8(h, oe_del)` and `ksw.cpp:173` is
//!    `t = _mm_subs_epu8(h, oe_ins)`, both taken after `h` has already absorbed `max(e, f)` at
//!    lines 162-163. So the rescue SW really does open gaps from H. That asymmetry between
//!    `ksw_extend2` and `ksw_u8` is upstream's, it is not a bug here, and it must be preserved:
//!    "fixing" it would change rescue scores and break byte-identity.
//!
//! # Sequence and matrix encoding (shared by every function below)
//!
//! `query` and `target` are 2-bit base codes, one byte per base: `0=A 1=C 2=G 3=T`, `4=N`
//! (anything ambiguous), as produced by bwa's `nst_nt4_table`. They are **not** ASCII. `m` is the
//! alphabet size, always 5 in bwa. `mat` is the `m*m` substitution matrix in row-major order, so
//! `mat[t*m + q]` scores target base `t` against query base `q`; `bwa_fill_scmat` fills it with
//! `+a` on the diagonal, `-b` off it, and `-1` on the whole `N` row and column. Scores are plain
//! integers in "score units" (bwa's default `a = 1`), so every penalty below is expressed in the
//! same units and only differences between them matter.
//!
//! # Gap parameters (units: score points)
//!
//! `o_del`/`o_ins` are the one-off gap **open** costs and `e_del`/`e_ins` the per-base **extend**
//! costs, all supplied as positive magnitudes and subtracted. bwa's CLI defaults are `-O 6` and
//! `-E 1` for both strands of the asymmetry, and `-A 1 -B 4`; `-O`/`-E` accept `INT1,INT2` to set
//! deletion and insertion independently, which is exactly the regime where an H-vs-M gap-open bug
//! becomes visible. All of them come from `mem_opt_t` (`opt.o_del`, `opt.e_del`, ...).

/// Result of a seed extension.
///
/// Two *competing* alignments are reported, and the caller (`bwamem.cpp:2495-2504` and its three
/// copy-pasted twins) picks between them: the best **local** one (`score`, `qle`, `tle`), which may
/// stop anywhere, and the best **to-the-end-of-query** one (`gscore`, `gtle`), which is forced to
/// consume all of `query`. The C takes the local one (and soft-clips) when
/// `gscore <= 0 || gscore <= score - pen_clip5`, otherwise it extends to the query end and sets
/// `a->qb = 0`. So getting `gscore` wrong does not merely move a score, it flips a soft-clip
/// decision and therefore `qb`/`rb` (or `qe`/`re` on the right extension) and any resulting
/// supplementary record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtendResult {
    /// Best local score (the return value of `ksw_extend2`). Includes `h0`: the DP is seeded with
    /// the seed's own score, so `score >= h0` always (`max` starts at `h0`, `ksw.cpp:463`).
    pub score: i32,
    /// Query length of the best local alignment (`max_j + 1`): number of query bases consumed, so
    /// it is a length, not an index. `0` means the extension added nothing.
    pub qle: i32,
    /// Target length of the best local alignment (`max_i + 1`), same length convention as `qle`.
    pub tle: i32,
    /// Target length when the alignment reaches the query end (`max_ie + 1`), i.e. how much target
    /// the *to-end* alignment consumes. Pairs with `gscore`, not with `score`.
    pub gtle: i32,
    /// Score when the alignment reaches the query end. `-1` (not `0`) when no row ever reached
    /// column `qlen`, which is how callers detect "there is no to-end alignment" (`ksw.cpp:463`
    /// initialises `gscore = -1`).
    pub gscore: i32,
    /// Largest `|mj - i|` seen at a row maximum, i.e. how far the best cell of a row wandered off
    /// the main diagonal. bwa uses it as a "was the band wide enough" test: the caller accepts the
    /// result only when `max_off < (w>>1) + (w>>2)` (three quarters of the band), and otherwise
    /// retries with a wider band, up to `MAX_BAND_TRY = 2` attempts total (`bwamem.cpp:51`,
    /// `2472`, `2495-2496`). So `max_off` feeds back into control flow: an off-by-one here can
    /// silently change which band a read was aligned under, and hence its SAM record.
    pub max_off: i32,
}

/// Banded local extension of `query` against `target` starting from seed score `h0`: returns the
/// best local score and the best to-query-end score, with the lengths each consumes. Faithful port
/// of `ksw_extend2` (`ksw.cpp:432-533`).
///
/// This is *extension*, not full alignment: the alignment is pinned to start at cell `(-1,-1)` with
/// score `h0` (the seed's score) and grows forward only. Callers reverse both sequences to extend
/// leftward from a seed.
///
/// # Parameters
///
/// * `query`, `target`: 2-bit base codes (`0..=4`, see the module docs), already oriented so that
///   extension proceeds left to right from the seed end. Supplied by `mem_chain2aln`. Lengths are
///   independent; neither is padded.
/// * `m`: alphabet size, always `5` in bwa. `mat`: the `m*m` row-major substitution matrix.
/// * `o_del`, `e_del`: deletion (target-consuming, CIGAR `D`) open and extend penalties, positive
///   magnitudes, subtracted. From `opt.o_del` / `opt.e_del` (`-O`/`-E`, defaults 6 and 1).
/// * `o_ins`, `e_ins`: the same for insertions (query-consuming, CIGAR `I`). `e_ins` and `e_del`
///   must be `>= 1`: they divide in the band clamp below, and the C would divide by zero too.
/// * `w`: half band width in cells. Only query columns within `|j - i| <= w` of the diagonal are
///   evaluated, which is what makes this O(w * tlen) rather than O(qlen * tlen). Supplied as
///   `opt.w` (default 100), possibly already doubled by the caller's retry loop.
/// * `end_bonus`: bonus (score points) the caller awards for reaching the query end, `opt.pen_clip`
///   in practice. Note that it is **not** added to any score in here: `ksw_extend2` uses it solely
///   to widen the feasible-gap estimate in the band clamp. The caller applies the actual bonus.
/// * `zdrop`: early-exit threshold in score points (`opt.zdrop`, default 100). Stop the row loop
///   once the running best has dropped by more than `zdrop` after accounting for the gap length
///   implied by the drift off the diagonal. `<= 0` disables it.
/// * `h0`: the seed's score, the DP's initial value at the origin. Must be `> 0`
///   (`assert(h0 > 0)`, `ksw.cpp:437`); the whole recurrence's "a zero H means no alignment here"
///   sentinel depends on it.
///
/// # Returns
///
/// An [`ExtendResult`] carrying two competing alignments: the best local one (`score`, and the
/// `qle`/`tle` *lengths* it consumes) and the best query-exhausting one (`gscore`, `gtle`), plus
/// `max_off` for the caller's band-retry test. See [`ExtendResult`] for each field's sentinel
/// values; note `score >= h0` always, and `gscore == -1` (not 0) means no row ever reached the
/// query end.
#[allow(clippy::too_many_arguments)]
pub fn ksw_extend2(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
    end_bonus: i32,
    zdrop: i32,
    h0: i32,
) -> ExtendResult {
    // Sequence lengths in bases: `qlen` is the number of DP columns, `tlen` the number of rows.
    let qlen = query.len();
    let tlen = target.len();
    debug_assert!(h0 > 0);
    // Cost of the *first* base of a gap, open plus one extend. Precomputed because the inner loop
    // subtracts it once per cell; `oe_del` for deletions (consuming target), `oe_ins` for
    // insertions (consuming query).
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    // ---------------------------------------------------------------------------------------
    // Setup 1/3: query profile
    // ---------------------------------------------------------------------------------------
    // Query profile: qp[c*qlen + j] = score of target base c against query base j. Built once so
    // the inner loop does one contiguous load instead of a two-level matrix index per cell; this is
    // a pure layout transform of `mat` and cannot change results (`ksw.cpp:444-447`).
    let mut qp = vec![0i8; qlen * m];
    // Flat cursor into `qp`: index of the next entry to write. Walks all `m * qlen` entries in
    // order and ends at `qlen * m`.
    let mut write_pos = 0;
    for target_code in 0..m {
        // The `mat` row for this target base: `mat_row[q]` scores it against query code `q`.
        let mat_row = &mat[target_code * m..target_code * m + m];
        for &query_base in query {
            qp[write_pos] = mat_row[query_base as usize];
            write_pos += 1;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Setup 2/3: DP score arrays and the "row minus one" boundary condition
    // ---------------------------------------------------------------------------------------
    // Score arrays, one entry per query column plus a sentinel. `eh_h[j]` holds H of the *previous*
    // row at column j-1 and `eh_e[j]` holds E of the current row at column j; the inner loop rotates
    // them in place, which is why it reads both before overwriting. Both are `int32_t h, e` fields of
    // the C's single `eh_t` array of structs (`ksw.cpp:428-430`); split here into two vectors, which
    // is layout-only. `calloc` in the C gives the same all-zero start (`ksw.cpp:441`).
    //
    // Zero is not a score here, it is the sentinel "no alignment reaches this cell": the local
    // recurrence floors at 0 and the inner loop tests `big_m != 0` before adding a substitution
    // score. That is also why `h0 > 0` is required.
    //
    // Rust: `vec![0i32; n]` allocates and zeroes in one step, which is what makes it the exact
    // equivalent of the C's `calloc` here, including the all-zero starting state the recurrence
    // depends on. The C keeps one array of two-field structs; splitting it into two vectors is a
    // memory-layout change only and cannot alter any computed value.
    let mut eh_h = vec![0i32; qlen + 1];
    let mut eh_e = vec![0i32; qlen + 1];
    // Row -1 (the "before the seed" row): the alignment starts at score h0 in column -1, and every
    // column j > 0 of this row is reached only by an insertion of length j, costing o_ins + j*e_ins.
    // The C spells the first step separately then runs the arithmetic progression until it would go
    // non-positive, at which point the rest stays 0 = unreachable (`ksw.cpp:449-451`).
    eh_h[0] = h0;
    eh_h[1] = if h0 > oe_ins { h0 - oe_ins } else { 0 };
    {
        let mut j = 2;
        // Loop condition is `> e_ins`, not `>= e_ins`: it stops one step before the value would
        // reach 0, since 0 means unreachable rather than "score zero". Matches the C exactly.
        while j <= qlen && eh_h[j - 1] > e_ins {
            eh_h[j] = eh_h[j - 1] - e_ins;
            j += 1;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Setup 3/3: band clamp
    // ---------------------------------------------------------------------------------------
    // Shrink the requested band to the widest gap that could ever pay for itself. A gap of length L
    // costs o + L*e, and the most an alignment can possibly earn is qlen * max_sc (+ end_bonus), so
    // any L beyond (qlen*max_sc + end_bonus - o)/e + 1 is unreachable and evaluating those cells is
    // wasted work. `max_sc` is the largest entry of the whole matrix, i.e. the match score `a`.
    let max_sc = mat[..m * m].iter().copied().max().unwrap_or(0) as i32;
    let mut w = w;
    // The f64 is not incidental and must not be "simplified" to integer division. The C is
    // `(int)((double)(qlen * max + end_bonus - o_ins) / e_ins + 1.)` (`ksw.cpp:456`): a true
    // division, then `+ 1.0`, then a C cast to int which truncates **toward zero**, not floors.
    // Integer division would truncate before the `+1` and round negatives the other way, giving a
    // different `w` on some inputs and thus a different (still self-consistent, but non-identical)
    // alignment. Rust's `as i32` on f64 also truncates toward zero, so this matches.
    let max_ins = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_ins))
        / f64::from(e_ins))
        + 1.0) as i32;
    // Floor at 1: a zero or negative band would evaluate no cells at all.
    let max_ins = max_ins.max(1);
    w = w.min(max_ins);
    // Same clamp for deletions. The C tags this one "TODO: is this necessary?" (`ksw.cpp:461`); it
    // is kept because removing it would change `w` and therefore the output.
    let max_del = (((qlen as f64 * f64::from(max_sc) + f64::from(end_bonus) - f64::from(o_del))
        / f64::from(e_del))
        + 1.0) as i32;
    let max_del = max_del.max(1);
    w = w.min(max_del);

    // Running best local cell. `max` starts at h0, so the seed itself is the baseline and a
    // negative-scoring extension simply never beats it; `max_i`/`max_j` stay -1 in that case, which
    // is what makes `qle`/`tle` come out 0 (`ksw.cpp:463-465`).
    let mut max = h0;
    let mut max_i = -1i32;
    let mut max_j = -1i32;
    // Best row that reached the query end, and its score. -1 = never happened.
    let mut max_ie = -1i32;
    let mut gscore = -1i32;
    let mut max_off = 0i32;
    // Live column window [beg, end) for the current row. It only ever narrows within a row and is
    // re-derived at the bottom of the loop, so this is an *adaptive* band, tighter than `w`.
    let mut beg = 0i32;
    let mut end = qlen as i32;

    // ===========================================================================================
    // Main DP: one iteration per target base (`i` = row), inner loop over query columns (`j`)
    // Reminder: H = best score ending here, M = ending on the diagonal, E = with a deletion open,
    // F = with an insertion open. See the glossary in the module header.
    // ===========================================================================================
    for i in 0..tlen as i32 {
        let mut f = 0i32; // F(i, beg): no insertion is open at the row's left edge
        let mut row_max = 0i32; // best H in this row (the C's `m`)
        let mut mj = -1i32; // column achieving `row_max`
        let tc = target[i as usize] as usize;
        // Row of the query profile for this target base: q[j] = s(target[i], query[j]).
        let q = &qp[tc * qlen..tc * qlen + qlen];

        // --- row prologue: intersect the band, seed the left-edge cell -------------------------

        // Intersect the carried-over live window with the band around diagonal i. `beg` only grows
        // and `end` only shrinks, hence `<`/`>` rather than plain assignment (`ksw.cpp:470-472`).
        if beg < i - w {
            beg = i - w;
        }
        if end > i + w + 1 {
            end = i + w + 1;
        }
        if end > qlen as i32 {
            end = qlen as i32;
        }
        // H(i, beg-1). When the window still touches column 0, the cell to the left of the row is
        // reached from the seed origin by a deletion of length i+1, costing o_del + (i+1)*e_del;
        // floored at 0 = unreachable. When the band has already left column 0 there is no such
        // path, so it is 0 outright (`ksw.cpp:474-477`).
        let mut h1 = if beg == 0 {
            let reached_by_deletion = h0 - (o_del + e_del * (i + 1));
            reached_by_deletion.max(0)
        } else {
            0
        };

        // --- inner loop: one cell per query column ---------------------------------------------
        let mut j = beg;
        while j < end {
            let ju = j as usize;
            let big_m = eh_h[ju]; // H(i-1, j-1)
                                  // E(i, j): the best score reaching cell (i, j) with a gap open in the query. It was
                                  // computed while row i-1 was being walked, which is why it is already in the array, but
                                  // the cell it belongs to is (i, j), not (i-1, j). The C states the same convention:
                                  // "At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }" (`ksw.cpp:479`).
            let mut e = eh_e[ju]; // E(i, j)
            eh_h[ju] = h1; // H(i, j-1) for next row
                           // M(i,j) = H(i-1,j-1) + s(t[i], q[j]), but only if the diagonal predecessor was actually
                           // reachable. H == 0 is the "no alignment here" sentinel, so a zero diagonal must NOT be
                           // extended by a (possibly positive) substitution score; it stays 0. This single test is
                           // what makes the surface local (`ksw.cpp:487`).
            let big_m = if big_m != 0 {
                big_m + i32::from(q[ju])
            } else {
                0
            };
            // H(i,j) = max(M, E, F). No explicit max(.., 0) is needed: E and F are floored at 0
            // below and M is 0 when unreachable, so h >= 0 falls out (`ksw.cpp:488`).
            let mut h = if big_m > e { big_m } else { e };
            h = if h > f { h } else { f };
            h1 = h; // becomes H(i, j) for the next column
                    // Row max with a **strict** `>` on the incumbent, i.e. ties go to the *later* column.
                    // The C is `mj = m > h? mj : j` (`ksw.cpp:491`), so on `row_max == h` it takes `j`.
                    // Flipping this tie-break silently shifts qle/tle on repeat-rich reads.
            mj = if row_max > h { mj } else { j };
            row_max = if row_max > h { row_max } else { h };
            // Gaps open from M (the diagonal score), not from H = max(M, E, F). Both `ksw_extend2`
            // and the vectorized `bandedSWA` do this: MAIN_CODE16 subtracts oe_ins/oe_del from
            // `m11`, not `h11`. Using H instead only agrees while H == M, which holds on ordinary
            // extensions but not inside satellite repeats, where it silently turns a local extension
            // into a to-end one (wrong gscore => wrong qe/re, and a lost supplementary record).
            // E(i+1,j) = max(0, E(i,j) - e_del, M(i,j) - o_del - e_del): either keep the deletion
            // that was already open (pay one more extend) or open a new one from the diagonal. The
            // `t.max(0)` is the local floor, and it is applied to the *open* candidate only, exactly
            // as the C does (`ksw.cpp:493-497`); `e` itself cannot go below 0 because `t >= 0` and
            // the max picks whichever is larger.
            let mut t = big_m - oe_del;
            t = t.max(0);
            e -= e_del;
            e = if e > t { e } else { t };
            eh_e[ju] = e;
            // F(i,j+1) = max(0, F(i,j) - e_ins, M(i,j) - o_ins - e_ins). F never leaves a register:
            // it flows rightward along the row only, which is why there is no `eh_f` array.
            let mut t = big_m - oe_ins;
            t = t.max(0);
            f -= e_ins;
            f = if f > t { f } else { t };
            j += 1;
        }
        // --- row epilogue: sentinel column, to-end score, termination tests, band tightening ----
        // Flush the sentinel column: H(i, end-1) has to be visible to row i+1 as its diagonal
        // predecessor, and E is reset because column `end` was never evaluated this row.
        eh_h[end as usize] = h1;
        eh_e[end as usize] = 0;
        // `j == qlen` means the row ran all the way to the query end, so h1 is the to-end score for
        // this row. `<=` not `<`: on a tie the C keeps the **later** target row (`ksw.cpp:504-507`
        // writes `max_ie` unconditionally when `gscore <= h1`). That picks the longest to-end
        // alignment among equals, and it is load-bearing for `gtle`.
        if j == qlen as i32 && gscore <= h1 {
            max_ie = i;
            gscore = h1;
        }
        // Every cell in the row is unreachable, so no later row can be reachable either: stop.
        if row_max == 0 {
            break;
        }
        if row_max > max {
            max = row_max;
            max_i = i;
            max_j = mj;
            let off = (mj - i).abs();
            if off > max_off {
                max_off = off;
            }
        } else if zdrop > 0 {
            // Z-drop: give up once the score has fallen too far below the best cell seen. The naive
            // test `max - row_max > zdrop` would fire on any long gap, so the drop is first
            // discounted by the gap the drift off the diagonal implies. Moving from (max_i, max_j)
            // to (i, mj) advances `i - max_i` target bases and `mj - max_j` query bases; the excess
            // on either side is a forced gap of that length, worth `len * e_*`. Whichever side is
            // longer decides whether it is a deletion (e_del) or an insertion (e_ins). Only the
            // *extend* cost is subtracted, not the open cost, which makes the test conservative.
            // Verbatim structure of `ksw.cpp:512-517`.
            if i - max_i > mj - max_j {
                if max - row_max - ((i - max_i) - (mj - max_j)) * e_del > zdrop {
                    break;
                }
            } else if max - row_max - ((mj - max_j) - (i - max_i)) * e_ins > zdrop {
                break;
            }
        }

        // Shrink the live window to the cells that can still seed row i+1: drop leading and trailing
        // columns whose H and E are both 0 (unreachable). This is the adaptive part of the band and
        // is a pure speed optimisation *only* because unreachable cells provably stay unreachable.
        // `ksw.cpp:520-523`.
        // Scans right from `beg`; on exit it is the leftmost column of this row that is still
        // reachable (or `end` if none is), which becomes row i+1's `beg`.
        let mut first_live = beg;
        while first_live < end && eh_h[first_live as usize] == 0 && eh_e[first_live as usize] == 0 {
            first_live += 1;
        }
        beg = first_live;
        // Scans left from `end`; on exit it is the rightmost reachable column of this row.
        let mut last_live = end;
        while last_live >= beg && eh_h[last_live as usize] == 0 && eh_e[last_live as usize] == 0 {
            last_live -= 1;
        }
        // `last_live + 2`, not `+ 1`: `last_live` is the last live column, the next one is what it
        // can reach by an insertion next row, and `end` is exclusive. Clamped to qlen.
        end = if last_live + 2 < qlen as i32 {
            last_live + 2
        } else {
            qlen as i32
        };
    }

    // `+ 1` turns the 0-based index of the best cell into a consumed length; with max_i/max_j still
    // -1 (no extension beat h0) this yields 0, and with max_ie = -1 gtle is 0 while gscore stays -1.
    ExtendResult {
        score: max,
        qle: max_j + 1,
        tle: max_i + 1,
        gtle: max_ie + 1,
        gscore,
        max_off,
    }
}

/// The C's `#define MINUS_INF -0x40000000` (`ksw.cpp:544`). Not `i32::MIN`: the DP subtracts gap
/// penalties from it repeatedly, and starting a quarter of the way up from the bottom leaves ample
/// headroom so those subtractions cannot wrap. Reproduce the constant exactly, since an unreachable
/// cell's exact value can leak into a tie-break.
pub(crate) const MINUS_INF: i32 = -0x4000_0000;

/// Append `(op, len)` to a CIGAR (op-merged), mirroring bwa's `push_cigar` (`ksw.cpp:546-556`).
/// Ops follow the BAM encoding used throughout bwa: 0=M, 1=I, 2=D, packed as `len << 4 | op`.
/// Merging into the previous entry when the op repeats is required, not cosmetic: the traceback
/// emits one base at a time, and an unmerged CIGAR would be a different SAM field.
///
/// # Parameters
///
/// * `cigar`: the CIGAR being built, one packed `len << 4 | op` word per run. Appended to in place.
///   Because [`ksw_global2`]'s traceback walks backwards, this vector is in *reverse* order until
///   the caller reverses it.
/// * `op`: the operation code, `0 = M` (aligned pair, match or mismatch), `1 = I` (insertion,
///   consumes query only), `2 = D` (deletion, consumes target only). Only those three occur here.
///   Must fit in 4 bits.
/// * `len`: run length in bases, `>= 1`. Normally `1` (the traceback emits a single base per step);
///   the two leading-gap flushes at the end of the traceback pass a longer run.
pub(crate) fn push_cigar(cigar: &mut Vec<u32>, op: u32, len: u32) {
    if let Some(last) = cigar.last_mut() {
        if (*last & 0xf) == op {
            *last += len << 4;
            return;
        }
    }
    cigar.push((len << 4) | op);
}

/// A traceback's CIGAR, built back to front one op per step with the current `(op, len)` run kept
/// apart and handed to [`push_cigar`] only when the op changes: the same words one push per step
/// builds, without the per-step check of the last word (fg-labs/bwa-mem3's `ksw_g2_backtrack`).
#[derive(Default)]
pub(crate) struct RunCigar {
    cigar: Vec<u32>,
    op: u32,
    len: u32,
}

impl RunCigar {
    /// One step of operation `op` (`0 = M`, `1 = I`, `2 = D`).
    #[inline(always)]
    pub(crate) fn step(&mut self, op: u32) {
        if op == self.op {
            self.len += 1;
        } else {
            if self.len > 0 {
                push_cigar(&mut self.cigar, self.op, self.len);
            }
            (self.op, self.len) = (op, 1);
        }
    }

    /// The CIGAR so far, still back to front, with the open run flushed.
    pub(crate) fn finish(mut self) -> Vec<u32> {
        if self.len > 0 {
            push_cigar(&mut self.cigar, self.op, self.len);
        }
        self.cigar
    }
}

/// Gap costs of [`ksw_global2`]'s cell: first base (open + extend) and each further base.
struct GlobalGaps {
    oe_del: i32,
    e_del: i32,
    oe_ins: i32,
    e_ins: i32,
}

/// One cell of [`ksw_global2`]'s forward pass. `mm` is H(i-1, j-1) and `e` E(i, j) on entry, `s`
/// the score of (target[i], query[j]); updates `f` (to F(i, j+1)) and `h1` (to H(i, j)) and
/// returns the traceback byte and E(i+1, j).
///
/// Direction bits 0-1 say how H was reached (0 = M, 1 = E, 2 = F); `>=` throughout means ties
/// prefer M over E over F (`ksw.cpp:613-616`), which is part of the output CIGAR. E and F open from
/// M, never from H (`ksw.cpp:604-607`); bit 2 records a continued deletion, bit 4 a continued
/// insertion, two bits per field so the traceback shifts by `which << 1` uniformly. Every select
/// goes through `select_unpredictable`: the conditions flip on mismatch positions, and LLVM's x86
/// backend otherwise compiled them into branches (~21 cycles per cell on Zen 3 at ~50%
/// misprediction). The hint forces conditional moves; the values are untouched.
#[inline(always)]
fn global_cell(mm: i32, e: i32, s: i8, f: &mut i32, h1: &mut i32, g: &GlobalGaps) -> (u8, i32) {
    use std::hint::select_unpredictable;
    let mm = mm + i32::from(s);
    let mut d: u8 = u8::from(mm < e);
    let mut h = select_unpredictable(mm >= e, mm, e);
    d = select_unpredictable(h >= *f, d, 2);
    h = select_unpredictable(h >= *f, h, *f);
    *h1 = h;
    let t = mm - g.oe_del;
    let e = e - g.e_del;
    d |= select_unpredictable(e > t, 1 << 2, 0);
    let e = select_unpredictable(e > t, e, t);
    let t = mm - g.oe_ins;
    *f -= g.e_ins;
    d |= select_unpredictable(*f > t, 2 << 4, 0);
    *f = select_unpredictable(*f > t, *f, t);
    (d, e)
}

/// `row[j] = mat_row[query[j]]`, 16 columns per table lookup where the CPU has one (fg-labs/
/// bwa-mem3's `ksw_g2_query_profile`). A block holding a code past the row falls to the scalar loop,
/// which indexes exactly as before.
fn query_profile_row(row: &mut [i8], query: &[u8], mat_row: &[i8]) {
    let mut j = 0;
    if mat_row.len() <= 16 {
        let mut tab = [0i8; 16];
        tab[..mat_row.len()].copy_from_slice(mat_row);
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is mandatory on AArch64; every load and store is 16 bytes inside
        // `query[j..]` / `row[j..]` with `j + 16 <= len`.
        unsafe {
            use std::arch::aarch64::*;
            let t = vld1q_u8(tab.as_ptr().cast());
            while j + 16 <= query.len() {
                let idx = vld1q_u8(query.as_ptr().add(j));
                if usize::from(vmaxvq_u8(idx)) >= mat_row.len() {
                    break;
                }
                vst1q_u8(row.as_mut_ptr().add(j).cast(), vqtbl1q_u8(t, idx));
                j += 16;
            }
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("ssse3") {
            // SAFETY: SSSE3 was just detected.
            j = unsafe { query_profile_ssse3(row, query, &tab, mat_row.len()) };
        }
    }
    for (r, &q) in row[j..].iter_mut().zip(&query[j..]) {
        *r = mat_row[q as usize];
    }
}

/// [`query_profile_row`]'s x86 blocks: `pshufb` over the 16-entry table. Returns where they stopped.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn query_profile_ssse3(row: &mut [i8], query: &[u8], tab: &[i8; 16], n: usize) -> usize {
    use std::arch::x86_64::*;
    let t = _mm_loadu_si128(tab.as_ptr().cast());
    // Codes `0..n` are kept; `min(idx, n - 1) == idx` fails for any larger one, and for bytes with
    // the top bit set, which `pshufb` would turn into 0.
    let lim = _mm_set1_epi8((n - 1) as i8);
    let mut j = 0;
    while j + 16 <= query.len() {
        let idx = _mm_loadu_si128(query.as_ptr().add(j).cast());
        if _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_min_epu8(idx, lim), idx)) != 0xffff {
            break;
        }
        _mm_storeu_si128(row.as_mut_ptr().add(j).cast(), _mm_shuffle_epi8(t, idx));
        j += 16;
    }
    j
}

/// The interior rows `[i0, i1)` of [`ksw_global2`] at band `w = (N - 1) / 2`: each row is the full
/// `N` cells `[i - w, i + w]` with `h1` and `f` starting at MINUS_INF, reads `eh[i-w ..= i+w]`, and
/// writes back the same span plus `eh[i+w+1] = {h1, MINUS_INF}`. Row `i + 1` reads only what row
/// `i` wrote and nothing reads `eh[i-w]` again, so the window stays in registers, shifted by one
/// cell per row, and is stored back after the last row. Same cells, same order as the row loop.
/// `target` and `z` are cut to the rows up front and walked as iterators, so the loop keeps no
/// bounds of its own for them (N = 3 then has no stack traffic left, and every N runs 8-22% fewer
/// instructions per row); the gaps are copied local so they stay in registers.
#[allow(clippy::type_complexity)]
#[inline(never)]
fn global_mid_rows<const N: usize>(
    (eh_h, eh_e, qp, target, z, qlen, g): (
        &mut [i32],
        &mut [i32],
        &[i8],
        &[u8],
        &mut [u8],
        usize,
        &GlobalGaps,
    ),
    i0: usize,
    i1: usize,
) {
    let w = (N - 1) / 2;
    let g = GlobalGaps { ..*g };
    let mut wh = [0i32; N];
    let mut we = [0i32; N];
    wh.copy_from_slice(&eh_h[i0 - w..i0 - w + N]);
    we.copy_from_slice(&eh_e[i0 - w..i0 - w + N]);
    let rows = target[i0..i1]
        .iter()
        .zip(z[i0 * N..i1 * N].as_chunks_mut::<N>().0);
    for (iw, (&tc, zi)) in (i0 - w..).zip(rows) {
        let q: &[i8; N] = qp[tc as usize * qlen + iw..][..N].try_into().unwrap();
        let (mut f, mut h1) = (MINUS_INF, MINUS_INF);
        let (mut nh, mut ne) = ([0i32; N], [0i32; N]);
        for k in 0..N {
            let s = h1;
            let (d, e) = global_cell(wh[k], we[k], q[k], &mut f, &mut h1, &g);
            zi[k] = d;
            // eh[beg + k] becomes {s, e}; the next row's window starts one cell later.
            if k > 0 {
                nh[k - 1] = s;
                ne[k - 1] = e;
            }
        }
        nh[N - 1] = h1;
        ne[N - 1] = MINUS_INF;
        wh = nh;
        we = ne;
    }
    eh_h[i1 - w..i1 - w + N].copy_from_slice(&wh);
    eh_e[i1 - w..i1 - w + N].copy_from_slice(&we);
}

/// Banded **global** (end-to-end, Needleman-Wunsch with affine gaps) alignment with traceback: a
/// faithful port of `ksw_global2` (`ksw.cpp:558-668`). Returns the score of aligning all of `query`
/// against all of `target` and the CIGAR realising it (`len<<4 | op`, op 0=M/1=I/2=D).
///
/// This is the kernel that turns a seed-extension *region* into an actual CIGAR: `mem_chain2aln`
/// has already fixed the query and target intervals, so nothing may be clipped and there is no
/// local floor at 0. Cells outside the band hold [`MINUS_INF`] rather than 0.
///
/// Parameters are as [`ksw_extend2`] (same encodings, same units, same suppliers) minus `h0`,
/// `end_bonus` and `zdrop`, which are meaningless for a global alignment. `w` is again the half
/// band width; the caller sizes it from the observed indel content of the region, so a too-small
/// `w` here does not merely slow things down, it can exclude the true path.
///
/// # Parameters
///
/// * `query`, `target`: the two sequences to align end to end, 2-bit base codes (`0=A 1=C 2=G 3=T`,
///   `4=N`), one byte per base. Both are consumed in full: nothing may be clipped. Supplied by
///   `mem_chain2aln`, which has already fixed the query and target intervals of the region.
/// * `m`: alphabet size, always `5`. `mat`: the `m*m` row-major substitution matrix, `mat[t*m + q]`,
///   in score points.
/// * `o_del`, `e_del`: deletion (target-consuming, CIGAR `D`) open and extend penalties, positive
///   magnitudes that are subtracted. `o_ins`, `e_ins`: the same for insertions (CIGAR `I`).
/// * `w`: half band width in cells; only columns with `|j - i| <= w` are evaluated. Negative values
///   are clamped to `0` below. Unlike the local kernel there is no adaptive shrink and no band
///   clamp, so `w` is exactly what gets used, and too small a `w` can make the true path
///   unrepresentable rather than merely slow.
///
/// # Returns
///
/// `(score, cigar)`. `score` is the end-to-end alignment score in score points and may be negative
/// (a global alignment has no local floor at 0); it is `eh_h[qlen]` of the last row. `cigar` is the
/// packed run-length encoding, one `len << 4 | op` word per run with `op` in `{0=M, 1=I, 2=D}`, in
/// left-to-right order. Its `M`+`I` lengths sum to `query.len()` and its `M`+`D` lengths sum to
/// `target.len()`.
#[allow(clippy::too_many_arguments)]
pub fn ksw_global2(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
) -> (i32, Vec<u32>) {
    // The wavefront kernel (NEON, or SSE4.1 on x86_64) where it is proven byte-identical and
    // wide enough to pay.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if let Some(r) =
        crate::global_wave::try_wave16(query, target, m, mat, o_del, e_del, o_ins, e_ins, w)
    {
        return r;
    }
    ksw_global2_scalar(query, target, m, mat, o_del, e_del, o_ins, e_ins, w)
}

/// [`ksw_global2`] on the scalar row-major kernel, every shape.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn ksw_global2_scalar(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w: i32,
) -> (i32, Vec<u32>) {
    // Sequence lengths in bases: `qlen` DP columns by `tlen` DP rows.
    let qlen = query.len();
    let tlen = target.len();
    // Cost of a gap's first base (open plus one extend), as in `ksw_extend2`.
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;
    // Half band width as an unsigned cell count; a negative request means "no band slack at all",
    // i.e. the diagonal only.
    let w = w.max(0) as usize;
    // Widest a banded row can be: 2w+1 cells, or the whole query if that is narrower
    // (`ksw.cpp:566`). Every row of the traceback matrix is stored in exactly `n_col` bytes, indexed
    // relative to that row's own `beg`, which is why the traceback has to recompute `beg` to read it.
    let n_col = qlen.min(2 * w + 1);

    // Traceback matrix, one byte per evaluated cell: `f<<4 | e<<2 | h` (`ksw.cpp:563`). Bits 0-1
    // say how H(i,j) was reached (0 = diagonal/M, 1 = from E, 2 = from F); bits 2-3 say whether the
    // outgoing E continued an existing deletion; bits 4-5 the same for F. Storing the continuation
    // bits is what lets the traceback stay inside a gap instead of re-deciding at every cell.
    // Per-call scratch, REUSED across calls through a thread-local. The four buffers here total
    // ~5.5 KB and this function runs millions of times per batch from rayon workers; fresh vectors
    // meant four allocations and ~5.5 KB of first-touch page traffic per call, which is the
    // remaining suspect for this function costing 5.7x more per call on Zen 3 than on an M4 after
    // the branch hypothesis died against the bwa-mem2 canary. Byte-safety of the reuse: `qp` is
    // fully rewritten below; `eh_h`/`eh_e` are refilled to MINUS_INF wholesale; `z` is zero-filled,
    // and in any case the traceback only reads cells the forward pass wrote this call, because the
    // recorded directions keep the walk inside the band by construction.
    /// The reused buffers: traceback matrix, query profile, and the two DP rows.
    struct GlobalScratch {
        z: Vec<u8>,
        qp: Vec<i8>,
        eh_h: Vec<i32>,
        eh_e: Vec<i32>,
    }
    thread_local! {
        static SCRATCH: std::cell::RefCell<GlobalScratch> = const {
            std::cell::RefCell::new(GlobalScratch {
                z: Vec::new(),
                qp: Vec::new(),
                eh_h: Vec::new(),
                eh_e: Vec::new(),
            })
        };
    }
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let GlobalScratch {
            z: z_buf,
            qp: qp_buf,
            eh_h: eh_h_buf,
            eh_e: eh_e_buf,
        } = &mut *scratch;
        // Grown, never cleared: the forward pass writes every `z` cell the traceback reads and
        // every `qp` entry (ksw.c mallocs both uninitialised), so zeroing them was pure cost.
        if z_buf.len() < n_col * tlen {
            z_buf.resize(n_col * tlen, 0u8);
        }
        if qp_buf.len() < qlen * m {
            qp_buf.resize(qlen * m, 0i8);
        }
        // Fixed-length reslices, and they are not cosmetic: `resize` leaves the vector length opaque to
        // LLVM, which reinstated a bounds check on every `eh_h[j]`/`eh_e[j]`/`z[..]` access in the
        // inner loop and cost 2.1 us per call on ARM (7.66 -> 10.75 s CPU over 1.46M calls) before
        // these lines existed. A slice of statically-related length restores the elision the local
        // `vec![...]` used to provide.
        let z: &mut [u8] = &mut z_buf[..n_col * tlen];
        let qp: &mut [i8] = &mut qp_buf[..qlen * m];
        for (target_code, row) in qp.chunks_exact_mut(qlen).enumerate() {
            // The `mat` row for this target base: `mat_row[q]` scores it against query code `q`.
            query_profile_row(row, query, &mat[target_code * m..target_code * m + m]);
        }

        // Row -1. Unlike the local kernel there is no 0 sentinel: unreachable is MINUS_INF, and the
        // origin starts at score 0 because a global alignment is pinned there. Columns 1..=w are reached
        // by a leading insertion of length j; columns beyond the band stay MINUS_INF
        // (`ksw.cpp:584-587`).
        eh_h_buf.clear();
        eh_h_buf.resize(qlen + 1, MINUS_INF);
        eh_e_buf.clear();
        eh_e_buf.resize(qlen + 1, MINUS_INF);
        let eh_h: &mut [i32] = &mut eh_h_buf[..qlen + 1];
        let eh_e: &mut [i32] = &mut eh_e_buf[..qlen + 1];
        eh_h[0] = 0;
        for j in 1..=qlen.min(w) {
            eh_h[j] = -(o_ins + e_ins * j as i32);
        }

        // ===========================================================================================
        // Forward pass: fill the score arrays and record, per cell, how each of H/E/F was reached
        // ===========================================================================================
        // Rows strictly inside the band (`w < i`, `i + w + 1 < qlen`) cover exactly `2w + 1` cells
        // each, and each one reads only what the row before wrote; for a narrow band those rows run
        // in a register window instead (`global_mid_rows`, fg-labs/bwa-mem3's `mid_rows`).
        let mid_lo = (w + 1).min(tlen);
        let mid_hi = qlen.saturating_sub(w + 1).min(tlen).max(mid_lo);
        let g = GlobalGaps {
            oe_del,
            e_del,
            oe_ins,
            e_ins,
        };
        let mut i = 0;
        while i < tlen {
            if i == mid_lo && mid_hi > mid_lo && (1..=7).contains(&w) {
                let rows = (&mut *eh_h, &mut *eh_e, &*qp, target, &mut *z, qlen, &g);
                match w {
                    1 => global_mid_rows::<3>(rows, mid_lo, mid_hi),
                    2 => global_mid_rows::<5>(rows, mid_lo, mid_hi),
                    3 => global_mid_rows::<7>(rows, mid_lo, mid_hi),
                    4 => global_mid_rows::<9>(rows, mid_lo, mid_hi),
                    5 => global_mid_rows::<11>(rows, mid_lo, mid_hi),
                    6 => global_mid_rows::<13>(rows, mid_lo, mid_hi),
                    _ => global_mid_rows::<15>(rows, mid_lo, mid_hi),
                }
                i = mid_hi;
                continue;
            }
            // F(i, beg): no insertion can be open at the row's left edge, and "impossible" is MINUS_INF
            // here rather than 0, since a global alignment has no local floor.
            let mut f = MINUS_INF;
            // Half-open column window `[beg, end)` for this row: the band around diagonal `i`, clipped
            // to the query. Unlike the local kernel this is fixed by `w` alone, with no adaptive shrink,
            // which is what lets the traceback recompute it from `i` and `w`.
            let beg = i.saturating_sub(w);
            let end = (i + w + 1).min(qlen);
            // H(i, beg-1), the cell to the left of the row. When the window still touches column 0 that
            // cell is reached from the pinned origin by a leading deletion of length i+1; once the band
            // has left column 0 no path reaches it at all.
            let mut h1 = if beg == 0 {
                -(o_del + e_del * (i as i32 + 1))
            } else {
                MINUS_INF
            };
            // This row's target base code, and the query-profile row it selects: `q[j] = s(target[i],
            // query[j])`, one contiguous load per cell.
            let tc = target[i] as usize;
            let q = &qp[tc * qlen..tc * qlen + qlen];
            // Where row `i`'s slice of the traceback matrix starts. Each row occupies exactly `n_col`
            // bytes and is indexed band-relative, so column `j` lives at `z_row_offset + (j - beg)`.
            let z_row_offset = i * n_col;
            for j in beg..end {
                // `mm` is M(i,j). Note it is added to unconditionally: there is no `!= 0` guard as in
                // the local kernel, because MINUS_INF + a small score is still hugely negative and
                // "unreachable" needs no special case here.
                // On entry `mm` is H(i-1, j-1) (the diagonal predecessor, still in the array from the
                // previous row) and `e` is E(i, j); after the `+= q[j]` below, `mm` is M(i,j). The
                // store of `h1` into `eh_h[j]` publishes H(i, j-1), which is what row i+1 will read as
                // *its* diagonal predecessor, so both reads must happen before it.
                let (mm, e) = (eh_h[j], eh_e[j]);
                eh_h[j] = h1;
                let (d, e) = global_cell(mm, e, q[j], &mut f, &mut h1, &g);
                eh_e[j] = e;
                // Stored band-relative: column j of row i lives at offset j - beg.
                z[z_row_offset + (j - beg)] = d;
            }
            eh_h[end] = h1;
            eh_e[end] = MINUS_INF;
            i += 1;
        }
        // H(tlen-1, qlen-1) as flushed by the last row's epilogue: the score of aligning all of `query`
        // against all of `target`. May be negative.
        let score = eh_h[qlen];

        // ===========================================================================================
        // Traceback: walk the recorded direction bits back to the origin, emitting one CIGAR op per step
        // ===========================================================================================
        // Traceback from the bottom-right cell of the band (`ksw.cpp:650-664`). `k` is the last query
        // column row `tlen-1` actually evaluated, i.e. `min(i + w + 1, qlen) - 1`.
        // The CIGAR under construction, built back to front (reversed at the very end).
        let mut run = RunCigar::default();
        // The traceback cursor: `i` is the current target row and `k` the current query column, both
        // 0-based and both walking down to -1. At the top of each iteration they name the cell whose
        // direction bits are about to be read; the step then decrements whichever sequence the emitted
        // operation consumes.
        let mut i = tlen as i64 - 1;
        let mut k = (tlen as i64 - 1 + w as i64 + 1).min(qlen as i64) - 1;
        // `which` is the state machine: 0 = currently on the M/diagonal surface, 1 = inside a deletion,
        // 2 = inside an insertion. It is carried between steps and used to *select which pair of bits*
        // to read, hence the `>> (which << 1)`: state 0 reads bits 0-1 (how H was reached), state 1
        // reads bits 2-3 (does the deletion continue), state 2 reads bits 4-5. That is the whole reason
        // the forward pass stored the continuation bits.
        let mut which = 0u8;
        if i >= 0 && k >= 0 {
            // Where cell (i, k) is stored, band-relative as the forward pass wrote it, moved with
            // each step rather than recomputed: `beg` drops by one with `i` exactly when `i > w`.
            let mut idx = i as usize * n_col + (k as usize - (i as usize).saturating_sub(w));
            loop {
                // The traceback byte the forward pass stored for cell (i, k): `f<<4 | e<<2 | h`.
                which = (z[idx] >> (which << 1)) & 3;
                run.step(match which {
                    0 => 0,
                    1 => 2,
                    _ => 1,
                });
                let shift = usize::from(i > w as i64);
                match which {
                    0 => {
                        idx = idx.wrapping_sub(n_col + 1 - shift);
                        i -= 1;
                        k -= 1;
                    }
                    1 => {
                        idx = idx.wrapping_sub(n_col - shift);
                        i -= 1;
                    }
                    _ => {
                        idx = idx.wrapping_sub(1);
                        k -= 1;
                    }
                }
                if i < 0 || k < 0 {
                    break;
                }
            }
        }
        let mut cigar = run.finish();
        // The loop stops as soon as *either* sequence is exhausted; whatever remains of the other must
        // be a single leading gap, emitted here (target left over = D, query left over = I).
        if i >= 0 {
            push_cigar(&mut cigar, 2, (i + 1) as u32);
        }
        if k >= 0 {
            push_cigar(&mut cigar, 1, (k + 1) as u32);
        }
        // Built back to front. Note the C reverses element-wise *without* re-merging, so an op cannot
        // straddle the seam; matching that is why we reverse rather than build forwards.
        cigar.reverse();
        (score, cigar)
    })
}

/// The narrowest half band that [`ksw_global2`] can be given without changing a byte of its result,
/// and the score of a real alignment that proves it. Not in the C: this only decides how few cells
/// the DP needs to visit.
///
/// `S` is the score of the best alignment with a single gap of exactly the length difference (no gap
/// at all for equal lengths), found by one linear scan. The optimum is at least `S`. A path that
/// strays `v >= 1` columns past the offsets the length difference requires carries at least `v` of
/// the short-side gap and `|D| + v` of the long-side one, and two gap opens, so with `a` the largest
/// matrix cell and `L` the shorter length its score is at most
///
/// ```text
/// (L - v) * a - (o_del + o_ins + |D| * e_long + v * (e_del + e_ins))
/// ```
///
/// That falls strictly below `S` once `v` exceeds the `v_max` computed below, so a band of
/// `|D| + v_max` holds every path scoring `S` or better. The bound counts gaps and pairs and nothing else, so it
/// also covers a D directly followed by an I, which the recurrence forbids but the traceback can
/// walk: when a gap was opened from M and H(i, j) came from F, the traceback reads H's bits and
/// leaves the path the score was computed on.
///
/// Why byte-identical, including every tie: narrowing a band can only lower a cell. Take any cell
/// and state the traceback stands on, with the suffix walked so far. Each candidate it compares has
/// a value which, joined to that suffix, either reaches at least the optimum or does not. Those that
/// do are paths scoring at least `S`, so all of them lie inside the narrow band and keep their exact
/// value; the rest were already strictly lower and can only drop further. Every comparison therefore
/// resolves as it did, ties included, and the walk reads the same bits. `tests::
/// tightened_band_is_byte_identical` holds this against scoring chosen to force ties.
///
/// # Returns
///
/// `(w, S)` with `w <= w_` the band to use. `w == 0` happens only for equal lengths and means the
/// diagonal is the sole path scoring `S` or better, so the result is `S` and a single `M` run and
/// the DP can be skipped. Returns `(w_, S)` unchanged for a matrix with no positive cell or negative
/// gap penalties, where the bound's monotonicity does not hold.
///
/// # Parameters
///
/// As [`ksw_global2`], whose `w` is `w_` here and must span the length difference, which is what
/// keeps the single-gap path inside the original band.
#[allow(clippy::too_many_arguments)]
pub fn tightened_global_band(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    w_: i32,
) -> (i32, i32) {
    let (qlen, tlen) = (query.len(), target.len());
    // `short_len`: the shorter length, which caps the aligned pairs. `gap`: the length difference.
    let short_len = qlen.min(tlen);
    let gap = qlen.abs_diff(tlen);
    let (t_shift, q_shift) = if tlen >= qlen { (gap, 0) } else { (0, gap) };
    // bwa's matrices score a pair by equality alone, so compares stand in for the table: one
    // lookup per pair was the whole cost of this scan on x86.
    let best = match uniform_cells(mat, m) {
        Some(u) => single_gap_best(query, target, short_len, q_shift, t_shift, uniform_cell(u)),
        None => single_gap_best(query, target, short_len, q_shift, t_shift, |x, y| {
            i32::from(mat[y as usize * m + x as usize])
        }),
    };
    let gap = gap as i32;
    // `gap_cost`: what the one mandatory gap costs, and the extend penalty of its side.
    let (gap_cost, e_long) = if gap == 0 {
        (0, 0)
    } else if tlen > qlen {
        (o_del + gap * e_del, e_del)
    } else {
        (o_ins + gap * e_ins, e_ins)
    };
    let score = best - gap_cost;

    let a = i32::from(*mat[..m * m].iter().max().expect("non-empty matrix"));
    if a <= 0 || o_del < 0 || e_del < 0 || o_ins < 0 || e_ins < 0 {
        return (w_, score);
    }
    let denom = a + e_del + e_ins;
    let budget = short_len as i32 * a - score - (o_del + o_ins) - gap * e_long;
    // `v_max`: the largest overshoot whose ceiling still reaches `score`.
    let v_max = (budget / denom).max(0);
    ((gap + v_max).min(w_), score)
}

/// `(match, mismatch, n)` when the `m x m` matrix scores every pair of codes `0..4` by equality
/// alone and every pair involving code 4 (`N`) alike, as `bwa_fill_scmat` builds it.
fn uniform_cells(mat: &[i8], m: usize) -> Option<(i32, i32, i32)> {
    if m != 5 {
        return None;
    }
    let (ma, mi, n) = (mat[0], mat[1], mat[4]);
    let expect = |i: usize, j: usize| match (i == 4 || j == 4, i == j) {
        (true, _) => n,
        (false, true) => ma,
        (false, false) => mi,
    };
    (0..25)
        .all(|c| mat[c] == expect(c / 5, c % 5))
        .then_some((ma.into(), mi.into(), n.into()))
}

/// The score of a query code against a target code under [`uniform_cells`]'s `(match, mismatch, n)`.
#[inline(always)]
fn uniform_cell((ma, mi, n): (i32, i32, i32)) -> impl Fn(u8, u8) -> i32 {
    move |x, y| {
        if x >= 4 || y >= 4 {
            n
        } else if x == y {
            ma
        } else {
            mi
        }
    }
}

/// [`tightened_global_band`]'s scan: the best total over the alignments with one gap of the length
/// difference. `run` starts with every pair on the far diagonal (`q_shift`/`t_shift` in) and moves
/// one pair per step onto the near one (`k, k`); the highest total seen is returned. `cell(q, t)`
/// scores a query code against a target code.
#[inline(always)]
fn single_gap_best(
    query: &[u8],
    target: &[u8],
    short_len: usize,
    q_shift: usize,
    t_shift: usize,
    cell: impl Fn(u8, u8) -> i32,
) -> i32 {
    let (qf, tf) = (
        &query[q_shift..q_shift + short_len],
        &target[t_shift..t_shift + short_len],
    );
    let mut run: i32 = qf.iter().zip(tf).map(|(&x, &y)| cell(x, y)).sum();
    if q_shift == t_shift {
        return run;
    }
    let mut best = run;
    // In blocks, so the per-pair differences vectorize and only the running maximum is serial.
    let mut d = [0i32; 64];
    let (qn, tn) = (&query[..short_len], &target[..short_len]);
    for k0 in (0..short_len).step_by(d.len()) {
        let r = k0..(k0 + d.len()).min(short_len);
        let n = r.len();
        for (o, ((&a, &b), (&c, &e))) in d[..n].iter_mut().zip(
            qn[r.clone()]
                .iter()
                .zip(&tn[r.clone()])
                .zip(qf[r.clone()].iter().zip(&tf[r])),
        ) {
            *o = cell(a, b) - cell(c, e);
        }
        for &v in &d[..n] {
            run += v;
            best = best.max(run);
        }
    }
    best
}

/// Result of `ksw_align2`: a local (Smith-Waterman) alignment with start/end coordinates and the
/// 2nd-best score. Mirrors bwa's `kswr_t` (the fields `mem_matesw` uses), whose all-miss default is
/// `g_defr = { 0, -1, -1, -1, -1, -1, -1 }` (`ksw.h:52`), i.e. score 0 and everything else -1.
///
/// Unlike [`ExtendResult`] these are **inclusive 0-based positions**, not lengths, so a caller that
/// wants a half-open interval writes `qe + 1`. `mem_matesw` does exactly that
/// (`bwamem_pair.cpp:218-222`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KswAlignResult {
    /// Best local alignment score in score points, `>= 0` (the local recurrence floors at 0, so an
    /// all-negative comparison yields `0`, matching ksw's `g_defr`). Unlike [`ExtendResult::score`]
    /// there is no `h0` to include: mate rescue starts from nothing. `mem_matesw` compares it
    /// against `opt->min_seed_len * opt->a` before accepting the rescue.
    pub score: i32,
    /// 0-based query end / start of the best alignment (`qb <= qe`). `qb` is `-1` when the
    /// start-recovery pass failed; `mem_matesw` treats that as "something goes wrong" and discards
    /// the rescue entirely (`bwamem_pair.cpp:216`), so the sentinel is load-bearing.
    pub qb: i32,
    pub qe: i32,
    /// 0-based target end / start of the best alignment (`tb <= te`).
    pub tb: i32,
    pub te: i32,
    /// 2nd-best score, taken over target-end columns far enough from `te` to be a genuinely
    /// different alignment rather than a neighbour of the best one, and its column. `-1` when
    /// nothing qualifies. `mem_matesw` copies `score2` straight into `mem_alnreg_t::csub`
    /// (`bwamem_pair.cpp:224`), which caps MAPQ via `raw_mapq(score - csub, a)`
    /// (`bwamem_pair.cpp:466`), so the difference between `-1` and `0` here is a difference of one
    /// MAPQ input point.
    pub score2: i32,
    /// The 0-based target end position at which `score2` was attained, or `-1` when `score2` is
    /// `-1`. Reported for diagnostics only: bwa's `mem_matesw` reads `score2` and ignores this.
    pub te2: i32,
}

/// ksw's suboptimal-alignment tracker: the `b` array of `KSW_XSUBO` plus the exclusion-window scan
/// that turns it into `(score2, te2)`.
///
/// This exists as a shared type because the same logic is needed twice, from two crates: once by the
/// scalar [`ksw_local_fwd`] below, and once per lane by the batched NEON kernels in `bwa-neon`, which
/// derive `score2` from an SoA row-max buffer instead of a running scalar. Both must agree to the
/// byte, and the two rules that make that non-obvious (the merge advances the stored row only when
/// the score improves; `score2` defaults to `-1` rather than `0`) had already been duplicated once.
/// Keeping one copy is the point of the type.
///
/// Callers push row maxima in increasing row order and then call [`Self::finish`]. Rows a lane never
/// processed (an `endsc` early stop) are simply not pushed.
#[derive(Debug, Default, Clone)]
pub struct SuboptimalTracker {
    /// Surviving `(row max score, target row)` candidates in increasing row order, runs of
    /// consecutive rows already merged. Never longer than the number of rows.
    b: Vec<(i32, i32)>,
}

impl SuboptimalTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record target row `row`'s maximum H. Below `minsc` the row is not a candidate at all
    /// (`KSW_XSUBO`); pass `minsc = i32::MAX` to suppress tracking, as the reverse pass does.
    ///
    /// Adjacent rows are *merged* (keep the higher) rather than appended, so one alignment's plateau
    /// across several target end positions counts once instead of dominating the array
    /// (`ksw.cpp:194-202`).
    pub fn push_row(&mut self, row: i32, row_max: i32, minsc: i32) {
        if row_max < minsc {
            return;
        }
        match self.b.last() {
            // Row `row` directly follows the last entry's row: same alignment drifting, so merge
            // rather than append. bwa keeps the higher score AND moves the stored row in the same
            // step (`ksw.cpp:201`), which is why the row is only advanced inside the `<` branch.
            // Advancing it unconditionally would look harmless and would change which entries the
            // exclusion window below rejects.
            Some(&(_, last_row)) if last_row + 1 == row => {
                if self.b.last().unwrap().0 < row_max {
                    *self.b.last_mut().unwrap() = (row_max, row);
                }
            }
            // A gap in rows: a genuinely separate candidate (`ksw.cpp:195-200`).
            _ => self.b.push((row_max, row)),
        }
    }

    /// `BWA4_SUBOPT_SHAPE` probe: how many candidates a job actually accumulates, and how many rows
    /// were pushed to produce them.
    ///
    /// The question it answers is whether the `score2` pass can move into a GPU thread. The tracker
    /// is per job and strictly sequential, so a thread that owns a job could run it in registers,
    /// which would delete both the row-maxima buffer and the host pass that reads it. What stops
    /// that is `b` being unbounded: a GPU thread needs a fixed-size array. The distribution of
    /// `b.len()` is therefore the whole feasibility question, and it is measured rather than
    /// guessed. Rows pushed is recorded beside it because the ratio is the compression a GPU-side
    /// tracker would buy.
    pub fn record_shape(&self, rows_pushed: u64) {
        if !subopt_shape::enabled() {
            return;
        }
        use std::sync::atomic::Ordering::Relaxed;
        subopt_shape::JOBS.fetch_add(1, Relaxed);
        subopt_shape::ROWS.fetch_add(rows_pushed, Relaxed);
        subopt_shape::CANDS.fetch_add(self.b.len() as u64, Relaxed);
        let bucket = match self.b.len() {
            0 => 0,
            1 => 1,
            2..=3 => 2,
            4..=7 => 3,
            8..=15 => 4,
            16..=31 => 5,
            32..=63 => 6,
            _ => 7,
        };
        subopt_shape::HIST[bucket].fetch_add(1, Relaxed);
        let mut m = subopt_shape::MAXLEN.load(Relaxed);
        while (self.b.len() as u64) > m {
            match subopt_shape::MAXLEN.compare_exchange_weak(
                m,
                self.b.len() as u64,
                Relaxed,
                Relaxed,
            ) {
                Ok(_) => break,
                Err(cur) => m = cur,
            }
        }
    }

    /// Best candidate lying outside the exclusion window around `te`, as `(score2, te2)`.
    ///
    /// Both are `-1` when nothing qualifies, **not** `0`: ksw returns
    /// `g_defr = {0, -1, -1, -1, -1, -1, -1}` in that case and `mem_matesw` copies it straight into
    /// `csub`. The sign is observable downstream, since `mem_sam_pe` caps MAPQ with
    /// `raw_mapq(score - csub, a)`, so `csub = -1` yields `score + 1` where `0` yields `score`.
    pub fn finish(&self, gmax: i32, te: i32, max_sc: i32) -> (i32, i32) {
        let mut score2 = -1i32;
        let mut te2 = -1i32;
        if gmax > 0 && !self.b.is_empty() {
            // ceil(gmax / max_sc): the fewest columns an alignment scoring `gmax` can span, since no
            // single cell contributes more than `max_sc`. Any candidate whose row lies within that
            // many rows of `te` could be the same alignment ending slightly early or late, so it is
            // excluded rather than counted as a rival (`ksw.cpp:221-227`). The band may run negative
            // or past the last row, which is harmless.
            let exclusion_half_width = (gmax + max_sc - 1) / max_sc;
            let (low, high) = (te - exclusion_half_width, te + exclusion_half_width);
            for &(cand_score, cand_te) in &self.b {
                if (cand_te < low || cand_te > high) && cand_score > score2 {
                    score2 = cand_score;
                    te2 = cand_te;
                }
            }
        }
        (score2, te2)
    }
}

/// Forward local-SW pass returning `(score, te, qe, score2, te2)` (no start coords). This is the
/// per-lane semantics a batched/vectorized mate-rescue kernel must reproduce: [`ksw_align2`] is just
/// this forward pass plus a second, reversed forward pass to recover `qb`/`tb` (`KSW_XSTART`).
///
/// `lanes` is ksw's SIMD width (16 for the u8 kernel, 8 for i16), and it is **not** a performance
/// knob: `ksw_qinit` rounds the query profile up to `slen * lanes` columns and fills the tail with
/// score 0 (`(k >= qlen? 0 : ma[query[k]]) + shift`). A zero-score column leaves `h = h_diag`, so
/// the padding carries a diagonal forward and its cells land in ksw's per-row `max` -- which feeds
/// the `b` array and hence `score2`. Dropping the padding makes row maxima decay where bwa's plateau,
/// so `score2` (and the `csub` mate rescue derives from it) comes out too low.
///
/// # This kernel opens gaps from H, and that is correct
///
/// Everywhere else in this crate the E/F floor is `M - oe_*`. Here it is `h - oe_*`, where
/// `h = max(M, E, F)`. That is not an oversight and must not be "fixed": `ksw_u8`'s striped inner
/// loop computes `h` first (`ksw.cpp:159-163`) and then derives both gaps from it,
/// `t = _mm_subs_epu8(h, oe_del)` at `ksw.cpp:168` and `t = _mm_subs_epu8(h, oe_ins)` at
/// `ksw.cpp:173`. `ksw_i16` does the same at `ksw.cpp:278-279`. Upstream accepts here the
/// `100M3I3D20M` CIGARs that `ksw_extend2` goes out of its way to forbid, because mate rescue only
/// consumes coordinates and a score, never a CIGAR. Aligning the two conventions would change
/// rescue scores and break byte-identity.
///
/// # Parameters
///
/// Sequences, `m` and `mat` are encoded as in [`ksw_extend2`]; gap penalties likewise. The rest:
///
/// * `minsc`: score threshold below which a row max is not even recorded in the suboptimal `b`
///   array (bwa's `KSW_XSUBO` payload, the low 16 bits of `xtra`). `mem_matesw` passes
///   `opt->min_seed_len * opt->a` (`bwamem_pair.cpp:208`). Pass `i32::MAX` to suppress `b` entirely,
///   which is what the reverse pass does.
/// * `endsc`: stop as soon as a row max reaches this (`KSW_XSTOP`). `i32::MAX` disables it. The
///   reverse start-recovery pass passes the forward pass's `score` here, since it only needs to
///   find where that score was first attained.
/// * `max_sc`: the largest single-cell match score, `opt->a`. Used only to size the exclusion
///   window around `te` when picking `score2` (a `score`-point alignment must span at least
///   `score / max_sc` columns, so anything closer than that overlaps the best alignment).
/// * `lanes`: ksw's SIMD width, 16 or 8, chosen by the caller from `KSW_XBYTE`. See above: this
///   changes the answer.
///
/// # Known divergence from the C, deliberately not modelled
///
/// `ksw_u8` works in **saturating uint8** with a bias (`q->shift`), so it caps at 255 and bails out
/// of the row loop early via `if (gmax + q->shift >= 255 || gmax >= endsc) break;` (`ksw.cpp:207`),
/// after which `ksw.cpp:211-213` clamps `r.score` to 255 and skips computing `qe`/`score2`
/// altogether. This port is plain `i32` and has no such ceiling; it implements only the
/// `gmax >= endsc` half of that break. The paths agree while scores stay under the cap, and
/// `mem_matesw` picks `KSW_XBYTE` only when `l_ms * opt->a < 250` (`bwamem_pair.cpp:208`) precisely
/// to keep them there. UNVERIFIED: whether the 250 guard makes overflow strictly impossible for
/// every input bwa can construct, or merely overwhelmingly unlikely. If a saturating case is ever
/// observed, this is where it will diverge.
///
/// # Returns
///
/// `(score, te, qe, score2, te2)`:
/// * `score`: the best local alignment score in score points, `>= 0`.
/// * `te`: 0-based **target** position where that score was attained, i.e. the last target base of
///   the best alignment. `-1` when no row scored above 0. First row to attain the max wins ties.
/// * `qe`: 0-based **query** position of the same cell, the smallest column attaining the max in
///   row `te`. `-1` when `te` is `-1`. It can land in the zero-scoring padding columns, which is
///   deliberate (see the `lanes` note above).
/// * `score2`, `te2`: the 2nd-best score and its target position, restricted to rows far enough
///   from `te` to be a genuinely different alignment. Both `-1` when nothing qualifies, and the
///   difference between `-1` and `0` is worth one MAPQ input point downstream.
///
/// These are **inclusive positions**, not consumed lengths: the opposite convention from
/// [`ksw_extend2`]'s `qle`/`tle`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn ksw_local_fwd(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    endsc: i32,
    max_sc: i32,
    lanes: usize,
) -> (i32, i32, i32, i32, i32) {
    // The query's true length in bases. `qlen` below is the *padded* column count the DP actually
    // iterates, so the two must not be confused: columns in `qlen_real..qlen` are the score-0
    // padding, and only `qlen_real` bases really exist.
    let qlen_real = query.len();
    // `slen * lanes` columns, the tail scoring 0 (see above).
    // `slen` is ksw's "segment length": the number of query columns each SIMD lane owns in the
    // striped layout, hence the round-up. `qlen` is the padded column count.
    let slen = qlen_real.div_ceil(lanes);
    let qlen = slen * lanes;
    // Number of target bases, i.e. of DP rows.
    let tlen = target.len();
    let oe_del = o_del + e_del;
    let oe_ins = o_ins + e_ins;

    let mut h_prev = vec![0i32; qlen]; // H(i-1, .)
    let mut h_cur = vec![0i32; qlen]; // H(i, .)
    let mut e = vec![0i32; qlen]; // E(i, j), persists across target rows
    let mut hmax_col = vec![0i32; qlen]; // H column at the best target end `te`
                                         // Running best cell over the whole matrix so far: `gmax` is its score (starts at 0, the local
                                         // floor, so an all-negative comparison ends with 0) and `te` the target row that attained it
                                         // (-1 = never). `hmax_col` above is the snapshot of that row's H values, kept so `qe` can be
                                         // recovered after the loop.
    let mut gmax = 0i32;
    let mut te = -1i32;
    // Suboptimal tracker: row maxima >= `minsc`, consecutive rows merged. Shared with the batched
    // NEON kernels so the merge rule and the exclusion window cannot drift between the two.
    let mut b = SuboptimalTracker::new();

    // ===========================================================================================
    // Main DP. Reminder: H = best score ending at this cell, E = with a deletion (gap in query)
    // open, F = with an insertion (gap in target) open. Unlike ksw_extend2, gaps here open from H.
    // ===========================================================================================
    for i in 0..tlen {
        // Offset of this target base's row inside the flat `mat`: `mat[s_row + q]` scores it
        // against query code `q`. (No precomputed query profile here, unlike ksw_extend2.)
        let s_row = target[i] as usize * m;
        // F(i, -1): no insertion is open at the row's left edge. Flows rightward in this register
        // only, so there is no F array.
        let mut f = 0i32;
        let mut h_diag = 0i32; // H(i-1, -1) = 0
                               // Best H seen in this row so far (the C's per-row `imax`), floored at 0. At the end of the
                               // row it is the row maximum, which is what feeds `gmax` and the suboptimal `b` array.
        let mut imax = 0i32;
        for j in 0..qlen {
            // s(target[i], query[j]), the substitution score for this cell.
            let sc = if j < qlen_real {
                i32::from(mat[s_row + query[j] as usize])
            } else {
                0 // padding column: h = h_diag, carrying the diagonal forward
            };
            // H(i,j) = max{0, H(i-1,j-1)+s, E(i,j), F(i,j)}  (E,F are already >= 0).
            // The explicit floor at 0 stands in for the C's saturating `_mm_adds_epu8` /
            // `_mm_subs_epu8` on the biased byte representation, which cannot go below zero at all.
            // Note there is no `h_diag != 0` guard as in ksw_extend2: this kernel really does allow
            // a zero cell to be extended, matching the unguarded vector add at `ksw.cpp:159`.
            // Loop invariant: at the top of iteration `j`, `h_diag` holds H(i-1, j-1), `e[j]` holds
            // E(i, j) as left by the previous row, and `f` holds F(i, j) from the cell to the left.
            // `h` starts as the diagonal candidate M(i,j) and ends as H(i,j) = max(0, M, E, F).
            let mut h = h_diag + sc;
            if h < 0 {
                h = 0;
            }
            if e[j] > h {
                h = e[j];
            }
            if f > h {
                h = f;
            }
            if h > imax {
                imax = h;
            }
            h_cur[j] = h;
            // E(i+1,j) = max{0, E(i,j)-e_del, H(i,j)-o_del-e_del}. `h`, NOT the diagonal M: see the
            // "opens gaps from H" section in this function's docs. `ksw.cpp:167-170`.
            // `e_new` is the "keep the deletion already open" candidate, `open_del` the "start a
            // fresh deletion from this cell" one; the larger, floored at 0, becomes E(i+1, j).
            let mut e_new = e[j] - e_del;
            let open_del = h - oe_del;
            if open_del > e_new {
                e_new = open_del;
            }
            e[j] = e_new.max(0);
            // F(i,j+1) = max{0, F(i,j)-e_ins, H(i,j)-o_ins-e_ins}, again from `h` (`ksw.cpp:172-174`).
            // The C needs a whole "lazy-F" fixup loop after the row (`ksw.cpp:179-190`) because its
            // striped layout cannot propagate F across segment boundaries in one pass. Iterating the
            // row in plain left-to-right order here makes F exact immediately, so the lazy-F loop has
            // no scalar counterpart: it is a vectorization artifact, not part of the recurrence.
            // Same pair for the insertion side: extend the open one vs open a new one from `h`. The
            // larger, floored at 0, becomes F(i, j+1).
            let mut f_new = f - e_ins;
            let open_ins = h - oe_ins;
            if open_ins > f_new {
                f_new = open_ins;
            }
            f = f_new.max(0);
            h_diag = h_prev[j];
        }
        // Record this row's max in the suboptimal tracker (see `SuboptimalTracker::push_row` for the
        // merge rule this encodes).
        b.push_row(i as i32, imax, minsc);
        // Strict `>`: the first target row attaining the maximum wins, later ties do not displace it.
        // Snapshot the whole H row, because `qe` is recovered later by re-scanning the winning row
        // (the C keeps the same snapshot in its `Hmax` vector, `ksw.cpp:203-208`).
        if imax > gmax {
            gmax = imax;
            te = i as i32;
            hmax_col.copy_from_slice(&h_cur);
            if gmax >= endsc {
                break;
            }
        }
        std::mem::swap(&mut h_prev, &mut h_cur);
    }

    // Query end: smallest query column reaching the max at `te` (ksw scans Hmax in *striped byte*
    // order, mapping byte i to column `i/lanes + i%lanes*slen`, but only the min-on-tie survives, so
    // the order does not matter -- the padded columns being in range does).
    let mut qe = -1i32;
    if te >= 0 {
        // Best H found in the winning row so far, and (in `qe`) the column achieving it. `best_h`
        // starts at -1 rather than 0 so that even an all-zero row still sets some `qe`.
        let mut best_h = -1i32;
        for i in 0..qlen {
            // `i` walks the snapshot in ksw's striped *byte* order; `col` is the real query column
            // that byte corresponds to. The mapping only matters for which column wins a tie, and
            // the explicit min-on-tie below makes even that order-independent.
            let col = i / lanes + (i % lanes) * slen;
            let h_here = hmax_col[col];
            if h_here > best_h {
                best_h = h_here;
                qe = col as i32;
            } else if h_here == best_h && (col as i32) < qe {
                qe = col as i32;
            }
        }
    }

    // 2nd-best score: best `b` entry whose row lies outside the exclusion window around `te`.
    let (score2, te2) = b.finish(gmax, te, max_sc);
    (gmax, te, qe, score2, te2)
}

/// Local Smith-Waterman with affine gaps, returning best-alignment coords and the 2nd-best score.
/// Faithful scalar port of `ksw_align2` (with `KSW_XSTART | KSW_XSUBO`). `max_sc` is the maximum
/// single match score (`opt.a`), used for the suboptimal-window width. `lanes` selects ksw's kernel
/// width (16 = u8 / `KSW_XBYTE`, 8 = i16); both passes use the same one, and it changes the result
/// (see [`ksw_local_fwd`]).
///
/// # Parameters
///
/// * `query`, `target`: the read and the reference window to search it in, 2-bit base codes
///   (`0=A 1=C 2=G 3=T`, `4=N`), one byte per base. In mate rescue `query` is the unaligned mate's
///   sequence and `target` the reference window predicted from its partner's position, so `target`
///   is typically much the longer of the two.
/// * `m`: alphabet size, always `5`. `mat`: the `m*m` row-major substitution matrix, `mat[t*m + q]`.
/// * `o_del`, `e_del`, `o_ins`, `e_ins`: affine gap open and extend penalties, positive magnitudes
///   in score points that are subtracted. Extends must be `>= 1`.
/// * `minsc`: minimum score worth reporting. Serves two purposes: below it a row max is not entered
///   into the suboptimal array, and below it the start-recovery pass is skipped entirely and `qb`
///   /`tb` stay `-1`. `mem_matesw` passes `opt->min_seed_len * opt->a`.
/// * `max_sc`: the largest single-cell match score, `opt->a`. Used only to size the exclusion window
///   around `te` when choosing `score2`. Must be `>= 1` (it divides).
/// * `lanes`: ksw's SIMD width, `16` for the u8 kernel or `8` for i16. Not a performance knob here:
///   it sets the query-profile padding and therefore changes `score2`. Both passes must use the
///   same value.
///
/// # Returns
///
/// A [`KswAlignResult`]. Coordinates are **inclusive 0-based positions**, not lengths. `qb`/`tb` are
/// `-1` whenever the start-recovery pass did not run or did not reproduce the forward score, and
/// `mem_matesw` discards the whole rescue on `qb < 0`, so that sentinel is load-bearing rather than
/// cosmetic.
#[allow(clippy::too_many_arguments)]
pub fn ksw_align2(
    query: &[u8],
    target: &[u8],
    m: usize,
    mat: &[i8],
    o_del: i32,
    e_del: i32,
    o_ins: i32,
    e_ins: i32,
    minsc: i32,
    max_sc: i32,
    lanes: usize,
) -> KswAlignResult {
    let (score, te, qe, score2, te2) = ksw_local_fwd(
        query,
        target,
        m,
        mat,
        o_del,
        e_del,
        o_ins,
        e_ins,
        minsc,
        i32::MAX,
        max_sc,
        lanes,
    );
    let mut r = KswAlignResult {
        score,
        qb: -1,
        qe,
        tb: -1,
        te,
        score2,
        te2,
    };
    // KSW_XSTART: recover the start by aligning the reversed prefixes and stopping at `score`.
    if score < minsc || qe < 0 {
        return r;
    }
    // bwa does `revseq(r.qe + 1, query); revseq(r.te + 1, target);` -- both *in place* -- then
    // re-inits the query profile at length `qe + 1` but calls the kernel with the **full** `tlen`.
    // So the query really is truncated, while the target is not: only its first `te + 1` bases are
    // reversed and the untouched tail still gets scanned. That matters, because the pass stops via
    // KSW_XSTOP once it reaches `score`, and if the reversed prefix alone never gets there the tail
    // can still reach it -- which sets qb/tb, and mem_matesw drops the rescue when qb < 0.
    // `qrev` is the query prefix up to and including `qe`, reversed: aligning it recovers how far
    // back from `qe` the alignment started. `trev` is the *whole* target with only its first
    // `te + 1` bases reversed, exactly as bwa's in-place `revseq(r.te + 1, target)` leaves it; the
    // untouched tail is still scanned by the pass below, which is deliberate (see the comment
    // above).
    let qrev: Vec<u8> = query[..=qe as usize].iter().rev().copied().collect();
    let mut trev: Vec<u8> = target.to_vec();
    trev[..=te as usize].reverse();
    // Results of the reversed pass: `rscore` should reproduce the forward `score` (it is passed as
    // `endsc`, so the pass stops the moment it does), and `rte`/`rqe` are how many target/query
    // bases back from `te`/`qe` the alignment ran, hence the subtractions below. `minsc` is
    // `i32::MAX` here to suppress the suboptimal array, which this pass has no use for.
    let (rscore, rte, rqe, _, _) = ksw_local_fwd(
        &qrev,
        &trev,
        m,
        mat,
        o_del,
        e_del,
        o_ins,
        e_ins,
        i32::MAX,
        score,
        max_sc,
        lanes,
    );
    if score == rscore {
        r.tb = te - rte;
        r.qb = qe - rqe;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    // 5x5 matrix like bwa: match a, mismatch -b, N row/col -1.
    // `a` is the match score and `b` the mismatch penalty, both positive magnitudes in score points
    // (bwa's `-A`/`-B`, defaults 1 and 4). Returns the 25-entry row-major matrix indexed
    // `mat[t*5 + q]`.
    fn scmat(a: i8, b: i8) -> Vec<i8> {
        let mut mat = vec![0i8; 25];
        // Flat write cursor into `mat`, walking the 25 cells in row-major order.
        let mut k = 0;
        for i in 0..4 {
            for j in 0..4 {
                mat[k] = if i == j { a } else { -b };
                k += 1;
            }
            mat[k] = -1;
            k += 1;
        }
        for _ in 0..5 {
            mat[k] = -1;
            k += 1;
        }
        mat
    }

    /// `tightened_global_band` promises the same score and CIGAR, ties included, as the band it
    /// replaces. A two-letter alphabet, cheap gaps and low-complexity targets are what make ties and
    /// gap-adjacent paths common; matching bwa's defaults alone would hardly exercise them.
    /// The compare-based scan of a uniform matrix gives the table lookup's best single-gap total,
    /// N codes, both gap sides and block boundaries included; a non-uniform matrix keeps the table.
    #[test]
    fn single_gap_scan_matches_table() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: u64| -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        for _ in 0..20_000 {
            let mat = scmat(1 + next(4) as i8, 1 + next(6) as i8);
            let u = uniform_cells(&mat, 5).expect("scmat is uniform");
            let (ql, tl) = (1 + next(200) as usize, 1 + next(200) as usize);
            let q: Vec<u8> = (0..ql).map(|_| next(5) as u8).collect();
            let t: Vec<u8> = (0..tl).map(|_| next(5) as u8).collect();
            let (short, gap) = (ql.min(tl), ql.abs_diff(tl));
            let (ts, qs) = if tl >= ql { (gap, 0) } else { (0, gap) };
            // The scan as first written: one table lookup per pair.
            let cell = |pq: usize, pt: usize| i32::from(mat[t[pt] as usize * 5 + q[pq] as usize]);
            let mut run: i32 = (0..short).map(|k| cell(k + qs, k + ts)).sum();
            let mut best = run;
            if gap != 0 {
                for k in 0..short {
                    run += cell(k, k) - cell(k + qs, k + ts);
                    best = best.max(run);
                }
            }
            assert_eq!(
                single_gap_best(&q, &t, short, qs, ts, uniform_cell(u)),
                best
            );
        }
        let mut odd = scmat(1, 4);
        odd[2] = -3;
        assert_eq!(uniform_cells(&odd, 5), None);
    }

    #[test]
    fn tightened_band_is_byte_identical() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move |n: u64| -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % n
        };
        let (mut narrowed, mut skipped) = (0u64, 0u64);
        for _ in 0..300_000 {
            let a = 1 + next(3) as i8;
            let b = 1 + next(6) as i8;
            let mat = scmat(a, b);
            let (o_del, e_del) = (next(9) as i32, 1 + next(3) as i32);
            let (o_ins, e_ins) = if next(2) == 0 {
                (o_del, e_del)
            } else {
                (next(9) as i32, 1 + next(3) as i32)
            };
            let alphabet = 2 + next(3);
            let len = 1 + next(60) as usize;
            let target: Vec<u8> = (0..len).map(|_| next(alphabet) as u8).collect();
            // The query is the target through a few edits, so it stays near enough to tighten.
            let mut query = Vec::with_capacity(len + 8);
            let rate = [0, 20, 60, 200][next(4) as usize];
            let mut i = 0;
            while i < target.len() {
                match next(1000) {
                    x if x < rate => query.push(next(alphabet) as u8),
                    x if x < rate + rate / 4 => query.push(4),
                    x if x < rate + rate / 2 => i += next(5) as usize,
                    x if x < rate + 3 * rate / 4 => {
                        for _ in 0..next(5) {
                            query.push(next(alphabet) as u8);
                        }
                        query.push(target[i]);
                    }
                    _ => query.push(target[i]),
                }
                i += 1;
            }
            if query.is_empty() {
                continue;
            }
            let span = query.len().abs_diff(target.len()) as i32;
            let w_ = span + next(40) as i32;
            let (w, lower_bound) =
                tightened_global_band(&query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w_);
            let wide = ksw_global2(&query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w_);
            let tight = ksw_global2(&query, &target, 5, &mat, o_del, e_del, o_ins, e_ins, w);
            assert_eq!(
                wide, tight,
                "w_={w_} w={w} q={query:?} t={target:?} a={a} b={b} O={o_del},{o_ins} E={e_del},{e_ins}"
            );
            assert!(lower_bound <= wide.0 || w == w_, "bound above the optimum");
            narrowed += u64::from(w < w_);
            if w == 0 {
                skipped += 1;
                assert_eq!(wide, (lower_bound, vec![(query.len() as u32) << 4]));
            }
        }
        // Guards the test itself: a generator that never narrows would pass by doing nothing.
        assert!(
            narrowed > 100_000 && skipped > 1_000,
            "{narrowed} {skipped}"
        );
    }

    #[test]
    fn ksw_align2_perfect_match() {
        // query == target: full-length local alignment, score = len*a, spanning both from 0.
        let mat = scmat(1, 4);
        let q = [0u8, 1, 2, 3, 0, 1, 2, 3];
        let r = ksw_align2(&q, &q, 5, &mat, 6, 1, 6, 1, 4, 1, 16);
        assert_eq!(r.score, 8);
        assert_eq!((r.qb, r.qe), (0, 7));
        assert_eq!((r.tb, r.te), (0, 7));
    }

    #[test]
    fn ksw_align2_local_trims_flanks() {
        // A perfect core flanked by mismatches: local alignment picks the core only.
        // query core AACC at query[2..6]; target has the core at target[1..5].
        let mat = scmat(1, 4);
        //            0  1  2  3  4  5  6  7
        let q = [3u8, 3, 0, 0, 1, 1, 3, 3];
        let t = [2u8, 0, 0, 1, 1, 2];
        let r = ksw_align2(&q, &t, 5, &mat, 6, 1, 6, 1, 1, 1, 16);
        assert_eq!(r.score, 4); // 4 matching bases
        assert_eq!((r.qb, r.qe), (2, 5));
        assert_eq!((r.tb, r.te), (1, 4));
    }

    /// Unbanded reference of the same recurrence (no band, no zdrop, no zero-row break), for
    /// validating the core DP where those heuristics don't fire.
    ///
    /// # Parameters
    ///
    /// Identical in meaning, encoding and units to [`ksw_extend2`]'s parameters of the same names:
    /// `query`/`target` are 2-bit base codes, `m` is the alphabet size (5), `mat` the `m*m`
    /// row-major matrix, the four gap parameters are positive magnitudes, and `h0 > 0` is the
    /// seed's score. There is no `w`, `end_bonus` or `zdrop` because this reference deliberately
    /// omits the band and the early exits.
    ///
    /// # Returns
    ///
    /// The best local score only (the equivalent of [`ExtendResult::score`]); the position and
    /// to-end fields are not tracked, since the test compares scores.
    #[allow(clippy::too_many_arguments)]
    fn ref_extend(
        query: &[u8],
        target: &[u8],
        m: usize,
        mat: &[i8],
        o_del: i32,
        e_del: i32,
        o_ins: i32,
        e_ins: i32,
        h0: i32,
    ) -> i32 {
        let qlen = query.len();
        let tlen = target.len();
        let oe_del = o_del + e_del;
        let oe_ins = o_ins + e_ins;
        let mut eh_h = vec![0i32; qlen + 1];
        let mut eh_e = vec![0i32; qlen + 1];
        eh_h[0] = h0;
        eh_h[1] = if h0 > oe_ins { h0 - oe_ins } else { 0 };
        let mut j = 2;
        while j <= qlen && eh_h[j - 1] > e_ins {
            eh_h[j] = eh_h[j - 1] - e_ins;
            j += 1;
        }
        let mut max = h0;
        for i in 0..tlen {
            let mut f = 0i32;
            let mut h1 = (h0 - (o_del + e_del * (i as i32 + 1))).max(0);
            for j in 0..qlen {
                let sc = i32::from(mat[target[i] as usize * m + query[j] as usize]);
                let big_m = eh_h[j];
                let mut e = eh_e[j];
                eh_h[j] = h1;
                let big_m = if big_m != 0 { big_m + sc } else { 0 };
                let mut h = big_m.max(e).max(f);
                h1 = h;
                if h > max {
                    max = h;
                }
                let t = (big_m - oe_del).max(0);
                e = (e - e_del).max(t);
                eh_e[j] = e;
                let t = (big_m - oe_ins).max(0);
                f = (f - e_ins).max(t);
                let _ = &mut h;
            }
            eh_h[qlen] = h1;
            eh_e[qlen] = 0;
        }
        max
    }

    // Shorthand for the tests: run `ksw_extend2` under bwa's default scoring (`-O 6 -E 1`) with a
    // band of 100 and zdrop 100, varying only the two sequences, the matrix and the seed score.
    fn call(query: &[u8], target: &[u8], mat: &[i8], h0: i32) -> ExtendResult {
        ksw_extend2(query, target, 5, mat, 6, 1, 6, 1, 100, 0, 100, h0)
    }

    #[test]
    fn exact_match_scores_full_length() {
        let mat = scmat(1, 4);
        let s: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let r = call(&s, &s, &mat, 1);
        // h0 + qlen matches of score 1.
        assert_eq!(r.score, 1 + s.len() as i32);
        assert_eq!(r.qle, s.len() as i32);
        assert_eq!(r.tle, s.len() as i32);
        assert_eq!(r.gscore, 1 + s.len() as i32);
    }

    #[test]
    fn matches_unbanded_reference() {
        let mat = scmat(1, 4);
        // Fixed-seed xorshift64 state, so the 500 cases replay identically on every run/machine.
        let mut state: u64 = 0xa5a5_1234_9999_0001;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..500 {
            // Build a target that shares a positive-scoring prefix with the query so the local
            // extension never hits a zero row (keeping band/zdrop inactive for the comparison).
            // Sequence length in bases for this case, 20..=59; `base` is the random query, and
            // `target` starts as an exact copy of it so the alignment is guaranteed positive
            // throughout (which is what keeps the band and z-drop inactive).
            let len = 20 + (next() % 40) as usize;
            let base: Vec<u8> = (0..len).map(|_| (next() % 4) as u8).collect();
            let mut target = base.clone();
            // introduce a couple of mismatches late, staying positive
            if len > 25 {
                // Position of the injected mismatch: near the end, so the running score has already
                // accumulated enough that the mismatch cannot drive the row to zero.
                let p = len - 3 - (next() % 3) as usize;
                target[p] = (target[p] + 1) % 4;
            }
            let got = ksw_extend2(&base, &target, 5, &mat, 6, 1, 6, 1, 1000, 0, 1_000_000, 1);
            let want = ref_extend(&base, &target, 5, &mat, 6, 1, 6, 1, 1);
            assert_eq!(got.score, want, "len={len}");
        }
    }

    #[test]
    fn global_exact_match_is_all_m() {
        let mat = scmat(1, 4);
        let s: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let (score, cigar) = ksw_global2(&s, &s, 5, &mat, 6, 1, 6, 1, 100);
        assert_eq!(score, s.len() as i32);
        assert_eq!(cigar, vec![(s.len() as u32) << 4]); // "<len>M"
    }

    #[test]
    fn global_single_deletion() {
        let mat = scmat(1, 4);
        // target has one extra base vs query -> a 1bp deletion (D) in the CIGAR.
        let query: Vec<u8> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let mut target = query.clone();
        target.insert(4, 2); // extra base in the middle of target
        let (_score, cigar) = ksw_global2(&query, &target, 5, &mat, 6, 1, 6, 1, 100);
        // total reference length consumed == target length; exactly one D of length 1.
        let dsum: u32 = cigar.iter().filter(|c| *c & 0xf == 2).map(|c| c >> 4).sum();
        assert_eq!(dsum, 1);
        let msum: u32 = cigar.iter().filter(|c| *c & 0xf == 0).map(|c| c >> 4).sum();
        assert_eq!(msum, query.len() as u32);
    }

    #[test]
    fn zdrop_stops_runaway_extension() {
        let mat = scmat(1, 4);
        // A short match then a long mismatched tail: zdrop caps the target length used.
        let mut query = vec![0u8; 10];
        query.extend(vec![1u8; 40]);
        let mut target = vec![0u8; 10];
        target.extend(vec![2u8; 40]); // tail all mismatched vs query tail
        let r = ksw_extend2(&query, &target, 5, &mat, 6, 1, 6, 1, 100, 0, 100, 1);
        assert_eq!(r.score, 1 + 10); // only the 10 matching bases contribute
        assert_eq!(r.tle, 10);
    }
}

/// Counters for [`SuboptimalTracker::record_shape`]. See it for what the question is.
pub mod subopt_shape {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    pub static JOBS: AtomicU64 = AtomicU64::new(0);
    pub static ROWS: AtomicU64 = AtomicU64::new(0);
    pub static CANDS: AtomicU64 = AtomicU64::new(0);
    pub static MAXLEN: AtomicU64 = AtomicU64::new(0);
    pub static HIST: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

    /// Whether the probe records. Read once and cached, like every other probe in this workspace.
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("BWA4_SUBOPT_SHAPE").is_some())
    }

    /// Print the distribution once at end of run. No-op unless [`enabled`].
    pub fn dump() {
        if !enabled() {
            return;
        }
        let jobs = JOBS.load(Relaxed).max(1);
        let labels = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64+"];
        let h: Vec<String> = HIST
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}:{}", labels[i], c.load(Relaxed)))
            .collect();
        eprintln!(
            "[subopt-shape] {} jobs, {} rows pushed ({:.1} per job), {} candidates kept ({:.2} per job), longest {}",
            JOBS.load(Relaxed),
            ROWS.load(Relaxed),
            ROWS.load(Relaxed) as f64 / jobs as f64,
            CANDS.load(Relaxed),
            CANDS.load(Relaxed) as f64 / jobs as f64,
            MAXLEN.load(Relaxed),
        );
        eprintln!("[subopt-shape] candidates per job: {}", h.join(" "));
    }
}
