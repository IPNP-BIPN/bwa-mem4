//! Cross-check our local Smith-Waterman score against an INDEPENDENT implementation.
//!
//! Every other parity test in this workspace compares our SIMD kernels against our own scalar
//! `ksw_local_fwd`. That catches a vectorisation bug and cannot, by construction, catch a bug the
//! scalar reference shares with them: a misread of `ksw.cpp` would be reproduced faithfully by all
//! six kernels and every test would stay green. This test closes that hole with a second opinion
//! from code that has never seen ours.
//!
//! The oracle is [hyalite](https://github.com/Psy-Fer/hyalite), a Rust reimplementation of Opal
//! (Rognes-style inter-sequence SIMD Smith-Waterman), MIT-licensed, dev-dependency only. It is a
//! deliberately narrow check, and the narrowness is the point: it validates exactly the one quantity
//! both implementations agree on the meaning of.
//!
//! # What this test does NOT check, and why
//!
//! `ksw_align2` returns seven values; only `score` is comparable.
//!
//! - `qe`/`te`/`qb`/`tb` depend on bwa's tie-breaking among equally scoring alignments and on the
//!   `KSW_XSTART` reverse pass over reversed prefixes. hyalite reports its own end positions under
//!   its own rules, so a mismatch there would mean nothing.
//! - `qb`/`tb` come from the `KSW_XSTART` reverse pass over reversed prefixes, and that pass only
//!   runs when the score clears `minsc`; below it ksw reports `-1` and `mem_matesw` discards the
//!   rescue. hyalite 0.2 reports its own start coordinates from a traceback, so the two ARE
//!   comparable, but only on the pairs where our pass ran at all.
//!
//! hyalite 0.2 changed what is checkable here twice over. It reports **start coordinates** from a
//! traceback (`align`), so the four span endpoints can be compared against ksw's `KSW_XSTART`
//! recovery: over 600 random pairs, every pair whose score cleared `minsc`, and therefore ran the
//! recovery, agreed on all four (499 of 499; the rest report `-1` by construction and are skipped).
//! Note the conventions differ, ksw's ends are inclusive and hyalite's exclusive, so `qe` is
//! `query_end - 1`.
//!
//! It also reports the **per-target-position maxima**
//! (`align_pair_position_max`: `out[t]` is the best local score ending at target position `t`) and
//! a `score2` built from them, so this test also cross-checks the second-best score and its column,
//! which drive `csub` and therefore MAPQ on every rescued mate. Those were previously validated only
//! against our own scalar reference, which is exactly the hole this file exists to close: a misread
//! of `ksw.cpp`'s `score2` recipe would have been reproduced identically by all six of our kernels.
//!
//! One subtlety worth stating because it is the difference between a real check and a tautology:
//! `align_pair_position_max` is a pure per-column maximum of an independently written DP. hyalite's
//! own `score2` then applies bwa's peak-collapsing recipe to it. Comparing against that checks BOTH
//! the DP and the recipe, and a disagreement has to be read carefully, since the recipe is
//! documented as a reimplementation of ours rather than an independent invention.
//!
//! # Why the second-best check runs only on lane-aligned queries
//!
//! bwa's per-row maximum is not the DP's per-column maximum, and the difference is observable here.
//! `ksw_qinit` rounds the query profile up to `slen * lanes` columns and fills the tail with
//! score-0 entries; a zero-score column leaves `h = h_diag`, so those padding cells carry a diagonal
//! forward and land in the row max that feeds the `b` array and therefore `score2`. `sw.rs` says so
//! from reading `ksw.cpp`; this test now says so from measurement.
//!
//! Measured over 600 random pairs at two scoring schemes: on queries whose length is a multiple of
//! the 16-lane width, where the profile needs no padding at all, our `(score, te, score2, te2)` and
//! hyalite's agree on every pair. On ragged queries 8 pairs disagree, and inspecting one of them
//! (`score2 = 20` at target column 162, where an independent naive DP and hyalite both put the true
//! column maximum at 10) shows the padding artifact directly: the score exists in bwa's padded row
//! max and nowhere in the real matrix.
//!
//! So the quadruple is asserted on lane-aligned queries, which is a real check of the second-best
//! machinery, and the score alone is asserted on arbitrary lengths. Widening the first would not
//! find a bug in us; it would only re-discover that we reproduce bwa's padding, which is required
//! for byte-identity and is not negotiable.
//!
//! # Two conventions that must be translated, not assumed
//!
//! 1. **Gap cost.** hyalite follows Opal: a gap of length `n` costs `open + (n - 1) * ext`. ksw and
//!    we charge `o + n * e`. They are the same affine function under `open = o + e`, `ext = e`, and
//!    that substitution is what this test passes. Getting it wrong would produce a plausible-looking
//!    failure that has nothing to do with either implementation being wrong.
//! 2. **Matrix indexing.** hyalite reads `matrix[q * alphabet_len + t]`, bwa reads `mat[t * m + q]`.
//!    bwa's DNA matrix is symmetric, so the transposition is a no-op here; it is written out anyway
//!    so a future asymmetric matrix does not silently transpose.
//!
//! hyalite carries a single gap-penalty pair, so it cannot express `o_del != o_ins`. The test
//! therefore runs symmetric penalties, which is bwa's default (`-O 6 -E 1`).

/// ksw's SIMD width for the u8 kernel, and the modulus that decides whether a query needs padding.
/// Not a tuning knob here: it is the number `ksw_qinit` rounds the profile up to.
const LANES: usize = 16;

use bwa_mem4_extend::ksw_align2;
use hyalite::{align, align_pair, align_pair_position_max, Mode, Scoring, SearchType};

/// bwa's default DNA matrix, built as `bwa_fill_scmat` does.
///
/// # Returns
/// A 25-entry (`m = 5`) row-major matrix: `+a` on the diagonal, `-b` off-diagonal among the four
/// concrete bases, `-1` on the whole N row and N column.
fn bwa_matrix(a: i8, b: i8) -> Vec<i8> {
    let mut mat = vec![0i8; 25];
    // Write cursor walking the matrix in row-major order: four rows of (4 base scores + the N
    // column), then the final all-N row.
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

/// The same matrix in hyalite's layout and element type.
///
/// # Parameters
/// - `mat`: bwa's `m * m` matrix, indexed `mat[t * m + q]`.
/// - `m`: alphabet size (5 here).
///
/// # Returns
/// An `m * m` `i32` matrix indexed `out[q * m + t]`, i.e. the transpose, widened.
fn to_hyalite_matrix(mat: &[i8], m: usize) -> Vec<i32> {
    let mut out = vec![0i32; m * m];
    for t in 0..m {
        for q in 0..m {
            out[q * m + t] = i32::from(mat[t * m + q]);
        }
    }
    out
}

/// Our best local-alignment score for one pair, with the `score2` machinery suppressed.
///
/// # Parameters
/// - `query`, `target`: nt4 codes, `0..=4`.
/// - `mat`, `o`, `e`: bwa's matrix and its symmetric gap open / extend magnitudes.
///
/// # Returns
/// `KswAlignResult::score`, the only field this test compares.
fn our_score(query: &[u8], target: &[u8], mat: &[i8], o: i32, e: i32) -> i32 {
    // `minsc = i32::MAX` suppresses the 2nd-best tracker (nothing can clear it), and `lanes` only
    // sets the query-profile padding, which is observable in `score2` and never in `score`.
    ksw_align2(
        query,
        target,
        5,
        mat,
        o,
        e,
        o,
        e,
        i32::MAX,
        i32::from(mat[0]),
        16,
    )
    .score
}

/// hyalite's best local-alignment score for the same pair.
///
/// # Returns
/// `BestHit::score` from `Mode::Sw` (the only local mode it offers), under the translated gap
/// convention described in the module docs.
fn oracle_score(query: &[u8], target: &[u8], mat: &[i8], o: i32, e: i32) -> i32 {
    // `open = o + e` and `ext = e`: hyalite charges `open + (n-1)*ext` where we charge `o + n*e`.
    let scoring = Scoring::new(5, to_hyalite_matrix(mat, 5), o + e, e)
        .expect("bwa's matrix and penalties are a valid hyalite scoring scheme");
    align_pair(query, target, &scoring, Mode::Sw, SearchType::Score)
        .expect("codes are all inside the 5-symbol alphabet")
        .score
}

/// Our full mate-rescue result for one pair, second-best tracking ENABLED.
///
/// # Parameters
/// - `query`, `target`: nt4 codes, `0..=4`.
/// - `mat`, `o`, `e`: bwa's matrix and its symmetric gap open / extend magnitudes.
/// - `minsc`: bwa's `minsc`, the score a column must reach before it can be a second-best peak.
///
/// # Returns
/// `(score, te, score2, te2)`, the four fields hyalite 0.2 can now be asked about.
fn ours_with_score2(
    query: &[u8],
    target: &[u8],
    mat: &[i8],
    o: i32,
    e: i32,
    minsc: i32,
) -> (i32, i32, i32, i32) {
    let r = ksw_align2(
        query,
        target,
        5,
        mat,
        o,
        e,
        o,
        e,
        minsc,
        i32::from(mat[0]),
        16,
    );
    (r.score, r.te, r.score2, r.te2)
}

/// hyalite's answer to the same question, via the per-target-position maxima it gained in 0.2.
///
/// # Returns
/// `(score, te, score2, te2)` in our conventions: `te`/`te2` are 0-based target columns, and a
/// missing second-best is `-1` in both fields, which is what `ksw_align2` reports.
fn oracle_with_score2(
    query: &[u8],
    target: &[u8],
    mat: &[i8],
    o: i32,
    e: i32,
    minsc: i32,
) -> (i32, i32, i32, i32) {
    let scoring = Scoring::new(5, to_hyalite_matrix(mat, 5), o + e, e)
        .expect("bwa's matrix and penalties are a valid hyalite scoring scheme");
    // `colmax[t]` = best local score ending at target column `t`, from hyalite's own DP.
    let mut colmax: Vec<i32> = Vec::new();
    let hit = align_pair_position_max(query, target, &scoring, &mut colmax)
        .expect("codes are all inside the 5-symbol alphabet");
    // `target_end` is `None` exactly when the best score is 0, i.e. no positive alignment; our
    // `te` is `-1` there, which is the same statement in ksw's vocabulary.
    let te = hit.target_end.map_or(-1, |t| t as i32);
    // The window half-width uses the matrix maximum, not the best score, which for bwa's DNA
    // matrix is the match bonus.
    let (score2, te2) = match hyalite::score2(
        &colmax,
        hit.score,
        te.max(0) as usize,
        i32::from(mat[0]),
        minsc,
    ) {
        Some((s, p)) => (s, p as i32),
        None => (-1, -1),
    };
    (hit.score, te, score2, te2)
}

/// Our local SW score must equal an independent implementation's, over randomised pairs.
///
/// The pairs are built the way the kernel tests build them: mostly a copy of the query so the
/// alignment is a noisy relative rather than unrelated noise, since two unrelated sequences score
/// near zero and would exercise nothing. `N` is injected on both sides so the `-1` row and column of
/// the matrix are covered.
#[test]
fn local_sw_score_matches_independent_implementation() {
    // bwa's `-A 1 -B 4 -O 6 -E 1`, plus a second scheme so the check is not tied to one point in
    // parameter space.
    for &(a, b, o, e) in &[(1i8, 4i8, 6i32, 1i32), (2, 3, 5, 2)] {
        let mat = bwa_matrix(a, b);
        // Deterministic LCG (Numerical Recipes 64-bit constants), top 31 bits taken so the
        // low-order-bit weakness never reaches the small moduli below. Fixed seed: a failure is
        // reproducible and the assert prints the round.
        let mut state = 0x51D3_9CF1_0BAD_C0DEu64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        // Counted and asserted non-zero at the end: a generator change that stopped producing
        // lane-aligned queries would silently turn the second-best check into dead code.
        let mut checked_quadruples = 0u32;
        let mut checked_spans = 0u32;
        for round in 0..300u32 {
            let qlen = 1 + (next() % 160) as usize;
            // Query over the full alphabet, N included (code 4) at ~3%.
            let q: Vec<u8> = (0..qlen)
                .map(|_| {
                    if next() % 100 < 3 {
                        4
                    } else {
                        (next() % 4) as u8
                    }
                })
                .collect();
            // Target: a noisy relative of the query, with room to run off either end.
            let tlen = qlen + (next() % 200) as usize;
            let mut t: Vec<u8> = Vec::with_capacity(tlen);
            // Read cursor into `q`: mostly copy the next query base, otherwise emit a random base
            // and coin-flip whether to also advance, giving mismatches and indels respectively.
            let mut qi = 0usize;
            while t.len() < tlen {
                if qi < q.len() && next() % 100 >= 8 {
                    t.push(q[qi]);
                    qi += 1;
                } else if next() % 100 < 3 {
                    t.push(4);
                    if next() % 2 == 0 {
                        qi += 1;
                    }
                } else {
                    t.push((next() % 4) as u8);
                    if next() % 2 == 0 {
                        qi += 1;
                    }
                }
            }

            let ours = our_score(&q, &t, &mat, o, e);
            let theirs = oracle_score(&q, &t, &mat, o, e);
            assert_eq!(
                ours, theirs,
                "local SW score disagrees with hyalite at A={a} B={b} O={o} E={e} round {round}, \
                 qlen={qlen} tlen={tlen}"
            );

            // The mate-rescue quadruple, on the lane-aligned queries only (see the module docs for
            // why ragged ones are not comparable). `minsc` is bwa's own threshold for a rescue to
            // count, `min_seed_len * a` at the defaults, so the second-best tracker is exercised in
            // the regime `mem_matesw` actually runs it in rather than at some test-only setting.
            let minsc = 19 * i32::from(a);
            // Span endpoints, on every length. `qb == -1` means the start-recovery pass did not run
            // (score below `minsc`), which is bwa's own behaviour and leaves nothing to compare.
            {
                let r = ksw_align2(&q, &t, 5, &mat, o, e, o, e, minsc, i32::from(mat[0]), 16);
                if r.qb >= 0 {
                    let scoring = Scoring::new(5, to_hyalite_matrix(&mat, 5), o + e, e)
                        .expect("bwa's matrix and penalties are a valid hyalite scoring scheme");
                    // `1 << 30` is the traceback's memory ceiling: generous, since these are test
                    // sized pairs, and an error here would mean the pair was too big rather than
                    // that anything disagreed.
                    let al = align(&q, &t, &scoring, Mode::Sw, 1 << 30)
                        .expect("codes are all inside the 5-symbol alphabet");
                    assert_eq!(
                        (r.score, r.qb, r.qe, r.tb, r.te),
                        (
                            al.score,
                            al.query_start as i32,
                            al.query_end as i32 - 1,
                            al.target_start as i32,
                            al.target_end as i32 - 1
                        ),
                        "alignment span disagrees with hyalite at A={a} B={b} O={o} E={e} \
                         round {round}, qlen={qlen} tlen={tlen}"
                    );
                    checked_spans += 1;
                }
            }
            if qlen.is_multiple_of(LANES) {
                let ours4 = ours_with_score2(&q, &t, &mat, o, e, minsc);
                let theirs4 = oracle_with_score2(&q, &t, &mat, o, e, minsc);
                assert_eq!(
                    ours4, theirs4,
                    "(score, te, score2, te2) disagrees with hyalite at A={a} B={b} O={o} E={e} \
                     round {round}, qlen={qlen} tlen={tlen}"
                );
                checked_quadruples += 1;
            }
        }
        assert!(
            checked_quadruples > 0,
            "no lane-aligned query was generated at A={a} B={b}, so score2 went unchecked"
        );
        assert!(
            checked_spans > 0,
            "no pair cleared minsc at A={a} B={b}, so the span endpoints went unchecked"
        );
    }
}
